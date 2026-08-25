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
"""
from __future__ import annotations

import numpy as np

from collection import navigator as nav
from collection.playthrough.blocks.base import flee_battle

_DENY_FRAMES = 1200          # a wandering NPC moves on; retry the tile after ~20 s
_DENY_ROUNDS = 3             # then treat it as parked-on (mom) and stop targeting it


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
        walkable_total = 0
        remaining = [None]                           # unvisited ∧ not perma-denied, per replan
        deadline = runner.frame_idx + self.per_map_frames
        if hard_deadline is not None:
            deadline = min(deadline, hard_deadline)  # whole-block budget wins

        def visit(t, x, y):
            if (t.map_group, t.map_num) == kg and 0 <= x < t.map_width and 0 <= y < t.map_height:
                visited.add((x, y))

        def avoid(t, beh):
            a = np.ones(t.grid.shape, bool)
            a[7:7 + t.map_height, 7:7 + t.map_width] = False
            for wx, wy in warp_tiles:
                a[wy + 7, wx + 7] = True
            return a

        def miss(x, y):
            denied[(x, y)] = denied.get((x, y), 0) + 1
            deny_until[(x, y)] = runner.frame_idx + _DENY_FRAMES

        assigned = self.tiles.get(key)

        def goal(t, beh):
            nonlocal walkable_total
            g = (((t.grid >> 10) & 3) == 0) & ~avoid(t, beh)
            walkable_total = max(walkable_total, int(g.sum()))
            if assigned:                             # regen: tour only the assigned slice
                m = np.zeros_like(g)
                for ax, ay in assigned:
                    if 0 <= ay + 7 < m.shape[0] and 0 <= ax + 7 < m.shape[1]:
                        m[ay + 7, ax + 7] = True
                g &= m
            for vx, vy in visited:
                g[vy + 7, vx + 7] = False
            live = int(g.sum())                      # unvisited before denials = real deficit
            for (dx_, dy_), until in deny_until.items():
                if denied[(dx_, dy_)] >= _DENY_ROUNDS or runner.frame_idx < until:
                    if g[dy_ + 7, dx_ + 7]:
                        g[dy_ + 7, dx_ + 7] = False
                        if denied[(dx_, dy_)] >= _DENY_ROUNDS:
                            live -= 1
            remaining[0] = live
            return g

        stucks = 0
        while runner.frame_idx < deadline:
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
        summary["walkable_tiles_per_map"][key] = walkable_total
        if assigned:
            summary.setdefault("tiles_assigned_per_map", {})[key] = len(assigned)
        unreached = sorted({(x, y) for x, y in denied if denied[(x, y)] >= _DENY_ROUNDS}
                           - visited)
        if unreached:
            summary["denied_tiles_per_map"][key] = [list(c) for c in unreached]  # json-stable
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
        deadline = f0 + self.frames                  # whole-block budget (see __init__)
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
