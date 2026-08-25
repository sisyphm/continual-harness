"""BFS_SWEEP expedition block (W33 corpus-v2 §2): spatial coverage, direction-balanced.

Replaces the v1 coverage family INSIDE an expedition: for each assigned map, a
navigator-driven tile tour — ONE `goto` call whose goal mask is "walkable ∧ unvisited"
and whose visit hook marks every tile the walk lands on, so the walk continuously
re-plans to the nearest unvisited tile until the map is covered (mask empty → 'stuck'
on a clean grid) or the per-map budget expires. Planning is from LIVE RAM only
(collision/elevation/behaviors of the current gBackupMapLayout) — no restore-based
graph building, so every frame is a real recorded frame (§3.2).

Tour containment: the avoid mask forbids planning outside the map interior (border
cells belong to the neighbour — entering one crosses the connection) and onto the
manifest's warp tiles (mats/stairs fire on step-on). Tiles whose entry is refused
repeatedly (parked NPC — the Brendan-house mom burned ~9.8k frames before this
existed) are DENIED as targets with an expiry, then permanently after 3 rounds;
they are reported, not silently dropped.

Direction-balance legs (§2, owner: reverse-direction rendering is weak): each
assigned (mapA, mapB) pair is crossed A→B then B→A through the direct manifest
connection (or warp), recording each directional crossing that really changed maps.

Wild battles anywhere in the sweep: flee (base.flee_battle) and resume.

Buffer indexing is BOUNDS-GUARDED. The map IDENTITY (gSaveBlock1 bank/number) and the
map LAYOUT (gBackupMapLayout, which is `t.grid` AND t.map_width/height) update on
DIFFERENT frames: stepping on a door flips the id to the destination ~230 frames before
the destination's layout arrives (measured 2026-08-26 by replaying rg_020 into
RustboroCity_Mart: id 11,7 from frame 124, layout still 40x60 Rustboro until 356). Every
coordinate this block indexes — the manifest's warp tiles, the assigned slice, visited
and denied tiles — is a MAP coordinate of the id's map, so inside that window it lands in
another map's buffer. Both directions crashed the whole block:
  rg_000/rg_001  avoid()'s warp mark, gym warp (4,111) into Petalburg City's 30x30
                 buffer -> IndexError: index 118 out of bounds for axis 0 with size 44
  rg_020         visit() accepted the Rustboro door tile (16,45) as a Mart tile (the
                 stale header said 40x60), and the next replan on the Mart's real 11x8
                 buffer -> IndexError: index 52 out of bounds for axis 0 with size 22
Both maps ended 0 tiles visited. A coordinate that does not fit the LIVE layout is never
a tile of the live map, so it is skipped and REPORTED in summary["oob_tiles_per_map"] —
never indexed, never fatal. (The plan is NOT the cause: no assigned tile in any of the
60 regen runs is outside its map, and every slice is a subset of that map's reachability
matrix entry.)
"""
from __future__ import annotations

import numpy as np

from collection import navigator as nav
from collection.playthrough.blocks.base import flee_battle

_DENY_FRAMES = 1200          # a wandering NPC moves on; retry the tile after ~20 s
_DENY_ROUNDS = 3             # then treat it as parked-on (mom) and stop targeting it


def _in(m, x: int, y: int) -> bool:
    """Does map tile (x, y) land inside buffer-space mask `m` (border offset +7)?"""
    return 0 <= y + 7 < m.shape[0] and 0 <= x + 7 < m.shape[1]


class BfsSweep:
    name = "bfs_sweep"
    phase = "bfs_sweep"

    def __init__(self, maps: list[str], legs: list | tuple = (),
                 per_map_frames: int = 45000, leg_frames: int = 15000,
                 frames: int | None = None, tiles: dict | None = None):
        self.maps = list(dict.fromkeys(maps))        # map keys "group,num", deduped in order
        # W33 regen: optional per-map ASSIGNED TILE SETS from the reachability matrix.
        # {map_key: [[x,y], ...]}. When present for a map, the tour's goal mask is
        # restricted to assigned ∧ unvisited -- the rotation slices a stage-world
        # across the fleet instead of every run re-walking whole maps. Cone-avoidance
        # is upstream: the planner simply does not assign tiles inside trainer sight
        # cones to pure-coverage slices.
        self.tiles = {k: {tuple(t) for t in v} for k, v in (tiles or {}).items()}
        self.legs = [tuple(l) for l in legs]         # (mapA, mapB) connection pairs
        self.per_map_frames = per_map_frames
        self.leg_frames = leg_frames
        # Whole-block frame budget (W33 loop-bounding invariant): LINEAR in the work
        # list — each map tour gets per_map_frames ONCE (its ensure-map included) and
        # each directional leg gets leg_frames ONCE (ensure + crossing retries
        # included). Every other block already had a whole-block deadline; without
        # one here, retry multiplication (3x ensure x 3x cross x 2 directions x
        # N legs) legally burned ~1M frames on plans with unreachable leg endpoints
        # (attempt-4 pilots: all three runs stuck in one RUSTBORO sweep at 3.5x the
        # plan's WHOLE-RUN frame target).
        self.frames = frames if frames is not None else (
            per_map_frames * len(self.maps) + 2 * leg_frames * len(self.legs))

    # ---------------------------------------------------------------- helpers

    def _ensure_map(self, runner, mk, key: str, summary: dict, *, budget: int) -> bool:
        # `budget` is the TOTAL frame allotment for reaching `key` — the retry loop
        # shares one deadline instead of re-granting the full budget per try (the
        # W33 loop-bounding invariant: retries always consume the same budget).
        deadline = runner.frame_idx + max(0, budget)
        for _ in range(3):
            t, _, _ = nav._state(runner)
            if t is not None and f"{t.map_group},{t.map_num}" == key:
                return True
            left = deadline - runner.frame_idx
            if left <= 0:
                break
            r = nav.goto_map(runner, mk, key, hop_budget=left)
            if r == "battle":
                flee_battle(runner)
                summary["battles_fled"] += 1
                continue
            if r == "arrived":
                return True
        summary["unreached_maps"].append(key)
        return False

    # ---------------------------------------------------------------- tile tour

    def _tour(self, runner, mk, key: str, summary: dict, *, hard_deadline: int | None = None) -> None:
        kg = tuple(int(v) for v in key.split(","))
        warp_tiles = {(w["x"], w["y"]) for w in mk.warps.get(key, [])}
        visited: set[tuple[int, int]] = set()
        denied: dict[tuple[int, int], int] = {}      # tile -> refusal rounds
        deny_until: dict[tuple[int, int], int] = {}  # tile -> frame the denial expires
        remaining = [None]                           # unvisited ∧ not perma-denied, per replan
        oob: dict[str, set] = {}                     # skipped coords, by what asked for them
        # Live layout dims seen during the tour -> [replans on them, max walkable seen].
        # Per-layout because a replan inside the id/layout transition window measures the
        # PREVIOUS map's buffer: a single max() reported RustboroCity_Mart (11x8, ~50
        # walkable) as 1303 walkable — Rustboro City's number, from three stale replans.
        # The dominant layout (most replans) is the map's own; a one-layout tour is
        # unchanged by this.
        layouts: dict[tuple[int, int], list[int]] = {}
        deadline = runner.frame_idx + self.per_map_frames
        if hard_deadline is not None:
            deadline = min(deadline, hard_deadline)  # whole-block budget wins

        def mark(m, x, y, val, why: str) -> bool:
            """Bounds-guarded buffer write. A coordinate outside the LIVE layout is not
            a tile of the live map (stale layout in the id/layout transition window, or
            a bad plan/manifest entry): skip it, record it, never index."""
            if _in(m, x, y):
                m[y + 7, x + 7] = val
                return True
            oob.setdefault(why, set()).add((x, y))
            return False

        def visit(t, x, y):
            if (t.map_group, t.map_num) == kg and 0 <= x < t.map_width and 0 <= y < t.map_height:
                visited.add((x, y))

        def avoid(t, beh):
            a = np.ones(t.grid.shape, bool)
            a[7:7 + t.map_height, 7:7 + t.map_width] = False
            for wx, wy in warp_tiles:
                mark(a, wx, wy, True, "warp")
            return a

        def miss(x, y):
            denied[(x, y)] = denied.get((x, y), 0) + 1
            deny_until[(x, y)] = runner.frame_idx + _DENY_FRAMES

        assigned = self.tiles.get(key)

        def goal(t, beh):
            g = (((t.grid >> 10) & 3) == 0) & ~avoid(t, beh)
            seen = layouts.setdefault((t.map_width, t.map_height), [0, 0])
            seen[0] += 1
            seen[1] = max(seen[1], int(g.sum()))
            if assigned:                             # regen: tour only the assigned slice
                m = np.zeros_like(g)
                for ax, ay in assigned:
                    mark(m, ax, ay, True, "assigned")
                g &= m
            for vx, vy in visited:
                mark(g, vx, vy, False, "visited")
            live = int(g.sum())                      # unvisited before denials = real deficit
            for (dx_, dy_), until in deny_until.items():
                if denied[(dx_, dy_)] >= _DENY_ROUNDS or runner.frame_idx < until:
                    if not _in(g, dx_, dy_):
                        oob.setdefault("denied", set()).add((dx_, dy_))
                        continue
                    if g[dy_ + 7, dx_ + 7]:
                        g[dy_ + 7, dx_ + 7] = False
                        if denied[(dx_, dy_)] >= _DENY_ROUNDS:
                            live -= 1
            remaining[0] = live
            return g

        stucks = 0
        stuck_resets = 0             # W34: 104 tours quit at stucks>=4 with 93% of
        while runner.frame_idx < deadline:   # budget unused and 15-21 tiles left
            r = nav.goto(runner, mk, goal, budget=deadline - runner.frame_idx, phase=self.phase,
                         stop_fn=lambda t: "left_map" if (t.map_group, t.map_num) != kg else None,
                         visit_fn=visit, avoid_fn=avoid, miss_fn=miss)
            if r == "battle":
                flee_battle(runner)
                summary["battles_fled"] += 1
            elif r == "left_map":                    # scripted warp/trigger pulled us out
                if not self._ensure_map(runner, mk, key, summary,
                                        budget=max(0, deadline - runner.frame_idx)):
                    break
            elif r == "stuck":
                if remaining[0] is not None and remaining[0] <= 0:
                    break                            # covered: everything left is perma-denied
                stucks += 1
                if stucks >= 4:
                    # Persist while real budget remains: transient blockers (NPC
                    # walk cycles, elevation-model refusals) shift on their own —
                    # a 600f wander-wait then a fresh round, three times max.
                    if stuck_resets < 3 and (deadline - runner.frame_idx) > self.per_map_frames // 3:
                        stuck_resets += 1
                        stucks = 0
                        nav._hold(runner, [], 600, self.phase)
                        continue
                    break
                # Wait out the earliest transient denial (else the retry sees the same
                # empty goal mask and burns a stuck round for nothing — measured on the
                # Brendan-house mom: 2 denials < perma, mask empty, instant re-stucks),
                # then retry; a still-parked NPC accumulates to perma within a round.
                active = [u for c, u in deny_until.items() if denied[c] < _DENY_ROUNDS]
                wait = 150
                if active:
                    wait = min(max(active) - runner.frame_idx + 30, _DENY_FRAMES + 30)
                nav._hold(runner, [], max(150, min(wait, deadline - runner.frame_idx)),
                          self.phase)
            else:                                    # 'budget' ('arrived' can't happen: the
                break                                #  goal mask never contains our own tile)
        summary["tiles_visited_per_map"][key] = (
            len(visited & assigned) if assigned else len(visited))
        summary["walkable_tiles_per_map"][key] = (
            max(layouts.values(), key=lambda s: s[0])[1] if layouts else 0)
        if assigned:
            summary.setdefault("tiles_assigned_per_map", {})[key] = len(assigned)
        unreached = sorted({(x, y) for x, y in denied if denied[(x, y)] >= _DENY_ROUNDS}
                           - visited)
        if unreached:
            summary["denied_tiles_per_map"][key] = [list(c) for c in unreached]  # json-stable
        if oob:
            # `layouts` disambiguates the two causes for an audit: a SINGLE layout that
            # is the map's own dims ⇒ the coordinates are wrong (plan/manifest bug);
            # MORE than one ⇒ the tour merely planned across the id/layout transition
            # window and the skipped writes were fine tiles hitting a stale buffer.
            summary.setdefault("oob_tiles_per_map", {})[key] = dict(
                layouts=[[w, h, n] for (w, h), (n, _) in sorted(layouts.items())],
                **{why: dict(n=len(c), sample=[list(t) for t in sorted(c)[:5]])
                   for why, c in sorted(oob.items())})
        if runner.frame_idx >= deadline:
            summary["budget_expired_maps"].append(key)

    # ---------------------------------------------------------------- direction legs

    def _cross(self, runner, mk, src: str, dst: str, summary: dict,
               *, hard_deadline: int | None = None) -> bool:
        """One directional crossing src→dst through the DIRECT manifest hop.
        Failures are REPORTED in summary["connections_failed"], never silently dropped.
        The whole crossing — ensure-src + up to 3 attempts — shares ONE leg_frames
        deadline (capped by the block's hard_deadline): the W33 loop-bounding
        invariant; previously each retry re-granted the full budget."""
        def fail(reason: str) -> bool:
            summary["connections_failed"].append([src, dst, reason])
            return False

        def now() -> int:
            return getattr(runner, "frame_idx", 0)   # unit tests drive _cross runner-less

        leg_deadline = now() + self.leg_frames
        if hard_deadline is not None:
            leg_deadline = min(leg_deadline, hard_deadline)

        def left() -> int:
            return max(0, leg_deadline - now())

        if left() == 0:
            return fail("leg_budget_expired")
        if not self._ensure_map(runner, mk, src, summary, budget=left()):
            return fail("src_unreached")
        conn = next((c for c in mk.connections.get(src, []) if c["dst_map"] == dst), None)
        warp = next((w for w in mk.warps.get(src, []) if w["dst_map"] == dst), None)
        for _ in range(3):
            if left() == 0:
                return fail("leg_budget_expired")
            if conn is not None:
                r = nav.cross_connection(runner, mk, conn["direction"], budget=left())
            elif warp is not None:
                r = nav.goto_warp(runner, mk, warp["x"], warp["y"], budget=left())
            else:
                return fail("no_direct_hop")
            if r == "battle":
                flee_battle(runner)
                summary["battles_fled"] += 1
                continue
            t, _, _ = nav._state(runner)
            if r == "crossed" and t is not None and f"{t.map_group},{t.map_num}" == dst:
                summary["connections_crossed"].append([src, dst])
                return True
        return fail("not_crossed")

    # ---------------------------------------------------------------- entry

    def run(self, runner, mk, ctx) -> dict:
        summary = dict(tiles_visited_per_map={}, walkable_tiles_per_map={},
                       denied_tiles_per_map={}, connections_crossed=[],
                       connections_failed=[], battles_fled=0,
                       unreached_maps=[], budget_expired_maps=[], frames=0)
        f0 = runner.frame_idx
        # Enter grass work healthy (W34): sweeps on encounter maps eat chip damage
        # before each flee, and a low lead entering a long sweep is how attrition
        # wedges start. One check per block, BEFORE the budget clock starts — the
        # trip must never starve the sweep it protects.
        from collection.playthrough.heal import ensure_healthy
        if ensure_healthy(runner, mk, src="bfs_sweep_heal") == "heal_failed":
            summary["heal_failed"] = True            # loud, but sweep what we can
        deadline = runner.frame_idx + self.frames    # whole-block budget (see __init__)
        for key in self.maps:
            budget_left = deadline - runner.frame_idx
            if budget_left <= 0:
                summary["block_budget_expired"] = True
                break
            if self._ensure_map(runner, mk, key, summary,
                                budget=min(self.per_map_frames, budget_left)):
                self._tour(runner, mk, key, summary, hard_deadline=deadline)
        for a, b in self.legs:
            if deadline - runner.frame_idx <= 0:
                summary["block_budget_expired"] = True
                break
            if self._cross(runner, mk, a, b, summary, hard_deadline=deadline):
                self._cross(runner, mk, b, a, summary, hard_deadline=deadline)
        summary["frames"] = runner.frame_idx - f0
        return summary
