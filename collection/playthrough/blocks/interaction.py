"""INTERACTION expedition block (W33 corpus-v2 §2): FIRST-CLASS interaction coverage
(owner: v1 interaction was thin and the demo's interaction behaviour is weak).

Per assigned map, the interactables are enumerated from the coverage manifest
(MapKnowledge's objects/signs/warps) PLUS the live gObjectEvents table (stage-spawned
NPCs the ROM template list can't know), and a fixed verb set is performed and COUNTED:

  npc_talk        goto adjacent, face (RAM-feedback turn), A, dwell on the box, then
                  close with the B-ONLY closer (see _close_dialog — the A,A,B clearer
                  re-talks a faced NPC on the stale-mask beat and wedges the block)
  npc_retalk      talk AGAIN from the same tile (second-dialog state differs)
  sign_read       same for signs; bg-event kinds 1-4 are facing-locked, so the stand
                  tile + facing come from the kind (pokeemerald BG_EVENT_PLAYER_FACING_*)
  dialog_cancel   open a dialog, advance ONE page with A, then B-out mid-way
  object_interact non-sign bg_events (hidden items kind 7, secret-base spots kind 8):
                  best effort — counted when a dialog really opened, else skip-with-note
  ledge_hop       one deliberate jump-ledge hop (navigator._JUMP behaviors), verified
                  by the 2-tile displacement
  door_bounce     enter a warp and come straight back — records the transition pair in
                  both directions (link-room/counter maps excluded by policy)
  save_dialog     START -> SAVE, A-confirm through to the composed "... saved the game"
                  text (gStringVar4 — extractors/text.last_message), close; menu slots
                  via the ledger-verified START machinery in blocks/menus.py

Talk success is a dialog OPENING, verified with the VISION-VALIDATED check
(heatz_adapter._visual_dialog_open — the documented fix for the raw BG0 window mask,
which goes PERMANENTLY stale-True after any box closes on some maps: measured
2026-08-20 on Oldale, where mask-based baselines skipped every target after the first
talk). The raw mask is only the fallback when no screenshot is available.

Mobile/parked NPCs reuse the bfs_sweep denial pattern: bounded goto attempts with a
wander-out wait between rounds, live position re-targeting on arrival (they drift while
we walk — the trainer_hunt lesson), and a skipped-with-reason row instead of a silent
drop. Wild battles anywhere: flee and resume.

Summary: per-verb counts + skipped-with-reason list + per-map enumeration, so the
accounting CLOSES: every enumerated target is either counted or explained.
"""
from __future__ import annotations

import random

import numpy as np

from collection import navigator as nav
from collection.collect_behaviors import _seek_start_slot
from collection.extractors.entities import npcs, player_pace_read
from collection.extractors.ram import GBAState
from collection.extractors.text import last_message
from collection.playthrough.blocks.base import flee_battle
from collection.playthrough.blocks.menus import (
    close_start_menu, open_start_menu, start_slots,
)

VERBS = ("npc_talk", "npc_retalk", "sign_read", "dialog_cancel", "object_interact",
         "ledge_hop", "door_bounce", "save_dialog")

_SIGN_KINDS = {0, 1, 2, 3, 4}       # BG_EVENT_PLAYER_FACING_ANY/NORTH/SOUTH/EAST/WEST
# facing-locked sign kinds -> (stand offset from the sign, facing to press A with)
_SIGN_SIDE = {1: ((0, 1), "UP"), 2: ((0, -1), "DOWN"), 3: ((-1, 0), "RIGHT"),
              4: ((1, 0), "LEFT")}
_RETRY_WAIT = 240                   # frames between goto rounds: let a wanderer move on
_ATTEMPTS = 3


def _face(runner, d: str, phase: str) -> bool:
    """Turn in place until the OBJECT-RECORD facing (the rendered truth) reads `d`."""
    for _ in range(4):
        rec = player_pace_read(runner.env)
        if rec is not None and rec[1] == d:
            return True
        nav._hold(runner, [d], 4, phase)
        nav._hold(runner, [], 8, phase)
    rec = player_pace_read(runner.env)
    return rec is not None and rec[1] == d


def _dialog_open_live(runner) -> bool:
    """Is a dialog box REALLY on screen? The vision-validated pixel check (the
    explorer's false-dialogue fix); raw window mask only when no screenshot exists."""
    from collection.heatz_adapter import _visual_dialog_open
    v = _visual_dialog_open(runner.env)
    return nav._dialog_open(runner) if v is None else bool(v)


def _close_dialog(runner, phase: str, max_cycles: int = 40) -> bool:
    """Close an open dialog with B ONLY. navigator._clear_dialog mixes in A presses,
    and A while FACING a talkable NPC re-opens the talk the moment the box closes —
    measured 2026-08-20: the Oldale save-boy dialog re-triggered until every later
    verb was wedged inside a box. B advances and closes text exactly like A, declines
    YES/NO prompts, and can never START a talk. True = no box on screen."""
    for _ in range(max_cycles):
        if not _dialog_open_live(runner):
            break
        nav._hold(runner, ["B"], 4, phase)
        nav._hold(runner, [], 14, phase)
    nav._hold(runner, [], 10, phase)
    return not _dialog_open_live(runner)


def _dialog_baseline(runner, phase: str) -> bool:
    """False = no box is really open (a later True is a REAL open). Closes a leftover
    box first (B-only); True = a box that will not close — counting here would lie."""
    if not _dialog_open_live(runner):
        return False
    _close_dialog(runner, phase)
    nav._hold(runner, [], 20, phase)
    return _dialog_open_live(runner)


def _press_a_dialog(runner, phase: str, wait: int = 90) -> bool:
    """One A press; True when a dialog visibly opens within `wait` frames."""
    runner.perform_action("A", record_end_state=False, metadata={"block": "interaction"})
    for _ in range(max(1, wait // 10)):
        if _dialog_open_live(runner):
            return True
        nav._hold(runner, [], 10, phase)
    return False


class Interaction:
    name = "interaction"
    phase = "interaction"

    def __init__(self, maps: list[str], frames: int = 90000, seed: int = 0,
                 save: bool = True):
        self.maps = list(maps)
        self.frames = frames                     # whole-block frame budget
        self.seed = seed
        self.save = save                         # perform the save_dialog verb once

    # ---------------------------------------------------------------- plumbing

    def _skip(self, summary, verb, key, ident, reason):
        summary["skipped"].append(dict(verb=verb, map=key, target=ident, reason=reason))

    def _ensure_map(self, runner, mk, key: str, summary: dict, *, budget: int) -> bool:
        for _ in range(3):
            t, _, _ = nav._state(runner)
            if t is not None and f"{t.map_group},{t.map_num}" == key:
                return True
            r = nav.goto_map(runner, mk, key, hop_budget=budget)
            if r == "battle":
                flee_battle(runner)
                summary["battles_fled"] += 1
                continue
            if r == "arrived":
                return True
        return False

    def _goto_adjacent(self, runner, mk, cells_fn, summary: dict, *, budget: int) -> str:
        """Walk to any cell of `cells_fn()` [(x, y), ...] — nav.goto's BFS walk minus
        its `_unstick` A-mash. Talk approach walks INTO NPC-adjacent cells by design,
        and goto's script-lock unstick (A presses on the first refused step) then TALKS
        to whatever NPC blocked us, wedging the walk inside a real dialog box for its
        whole budget (measured 2026-08-20: every post-wedge target burned 3x its goto
        slice on Oldale). Here: refusals transient-block the cell and route around
        (bfs_sweep's denial idea); an en-route box (scripted greeting, unstick-free
        bump-talk) is B-closed; battles are fled. The goal mask is rebuilt from
        cells_fn on EVERY replan, so a mobile NPC is chased at its live position
        (a stale mask measured 21k frames of ping-pong against the mart promoter).
        Returns 'arrived' | 'stuck' | 'budget'."""
        blocked: dict[tuple[int, int], int] = {}
        misses = 0
        resets = 0
        it = 0
        start = runner.frame_idx
        while runner.frame_idx - start < budget:
            it += 1
            if runner.nav_state().in_battle:
                flee_battle(runner)
                summary["battles_fled"] += 1
                continue
            # dialog watch runs SPARSELY (the fishing lesson: tight screenshot/snapshot
            # polling can core-dump mgba; en-route boxes are rare without goto's A-mash)
            if it % 8 == 1 and _dialog_open_live(runner):
                _close_dialog(runner, self.phase)
                continue
            t, x, y = nav._state(runner)
            if t is None:
                nav._hold(runner, [], 30, self.phase)
                continue
            walk = ((t.grid >> 10) & 3) == 0
            goals = np.zeros(t.grid.shape, bool)
            for cx, cy in cells_fn():
                if 0 <= cy + 7 < goals.shape[0] and 0 <= cx + 7 < goals.shape[1]:
                    goals[cy + 7, cx + 7] = True
            goals &= walk
            bx, by = x + 7, y + 7
            if not (0 <= by < goals.shape[0] and 0 <= bx < goals.shape[1]):
                nav._hold(runner, [], 30, self.phase)
                continue
            if goals[by, bx]:
                return "arrived"
            blocked = {c: f for c, f in blocked.items() if runner.frame_idx - f < 600}
            for cx, cy in blocked:
                if 0 <= cy < walk.shape[0] and 0 <= cx < walk.shape[1]:
                    walk[cy, cx] = False
            step = nav._bfs_step(walk, (bx, by), goals,
                                 elev=((t.grid >> 12) & 0xF).astype(np.uint8),
                                 beh=mk.behaviors(t))
            if step is None:
                if blocked and resets < 4:               # dead-ended by our own blocks
                    blocked.clear()
                    resets += 1
                    nav._hold(runner, [], 60, self.phase)
                    continue
                return "stuck"
            if nav._step(runner, nav.DIRS[step]):
                misses = 0
            else:
                misses += 1
                blocked[(bx + step[0], by + step[1])] = runner.frame_idx
                if misses >= 8:
                    return "stuck"
                nav._hold(runner, [], 10, self.phase)
        return "budget"

    def _talk_at(self, runner, tx: int, ty: int, *, facing: str | None = None) -> bool:
        """From a tile adjacent to (tx, ty): face it (or the forced `facing`) and A."""
        t, x, y = nav._state(runner)
        if t is None or abs(tx - x) + abs(ty - y) != 1:
            return False
        d = facing or nav.DIRS[(tx - x, ty - y)]
        if not _face(runner, d, self.phase):
            return False
        return _press_a_dialog(runner, self.phase)

    # ------------------------------------------------------------------- verbs

    def _talk_target(self, runner, mk, key, verb, ident, tx, ty, summary, *,
                     live_id=None, cancel_pool=None, facing=None, sides=None) -> bool:
        """The shared talk driver: goto adjacent (denial-style bounded rounds), live
        chase for NPCs, face, A, count `verb`; NPC success chains npc_retalk."""
        if runner.frame_idx >= self._deadline:
            self._skip(summary, verb, key, ident, "block frame budget expired")
            return False
        tx0, ty0 = tx, ty                                # template anchor: the live gate
                                                         # (connected maps' local_ids collide)

        def pos():
            if live_id is None:
                return tx0, ty0
            e = next((e for e in npcs(GBAState.from_env(runner.env))
                      if e.local_id == live_id
                      and abs(e.x - tx0) + abs(e.y - ty0) <= 8), None)
            return (e.x, e.y) if e is not None else (tx0, ty0)

        def cells():
            px, py = pos()
            offs = [o for o, _f in sides] if sides is not None else list(nav.DIRS)
            return [(px + dx, py + dy) for dx, dy in offs]

        for attempt in range(_ATTEMPTS):
            r = self._goto_adjacent(runner, mk, cells, summary,
                                    budget=min(3000, max(1000, self._deadline - runner.frame_idx)))
            if r != "arrived":
                if attempt < _ATTEMPTS - 1:
                    if r == "stuck":                     # parked blocker? let it wander off
                        nav._hold(runner, [], _RETRY_WAIT, self.phase)
                    continue
                self._skip(summary, verb, key, ident,
                           f"unreachable after {_ATTEMPTS} goto rounds ({r})")
                return False
            if _dialog_baseline(runner, self.phase):
                self._skip(summary, verb, key, ident,
                           "stale window mask would false-count (no_dialog limitation)")
                return False
            if self._talk_at(runner, *pos(), facing=facing):
                summary["verbs"][verb] += 1
                nav._hold(runner, [], 30, self.phase)    # let the box type a while
                _close_dialog(runner, self.phase)
                if verb == "npc_talk":
                    self._retalk(runner, mk, pos, key, ident, summary, facing=facing)
                if cancel_pool is not None and summary["verbs"]["dialog_cancel"] == 0:
                    self._dialog_cancel(runner, pos, key, ident, summary, facing=facing)
                return True
            nav._hold(runner, [], 30, self.phase)        # a wanderer stepped away mid-turn:
        self._skip(summary, verb, key, ident, "no dialog opened on A")
        return False

    def _retalk(self, runner, mk, pos, key, ident, summary, *, facing=None) -> None:
        """Second talk — the second-dialog state differs. One chase round when the
        NPC stepped away while the first box was up (the mart promoter did)."""
        nav._hold(runner, [], 20, self.phase)
        for round_ in range(2):
            if self._talk_at(runner, *pos(), facing=facing):
                summary["verbs"]["npc_retalk"] += 1
                nav._hold(runner, [], 30, self.phase)
                _close_dialog(runner, self.phase)
                return
            if round_ == 0:
                self._goto_adjacent(
                    runner, mk,
                    lambda: [(pos()[0] + dx, pos()[1] + dy) for dx, dy in nav.DIRS],
                    summary, budget=1500)
        self._skip(summary, "npc_retalk", key, ident, "second talk opened no dialog")

    def _dialog_cancel(self, runner, pos, key, ident, summary, *, facing=None) -> None:
        """Open, advance ONE page with A, then B-out mid-way (the open-and-cancel input
        pattern; on last-page boxes the B lands as the close itself)."""
        nav._hold(runner, [], 20, self.phase)
        if not self._talk_at(runner, *pos(), facing=facing):
            self._skip(summary, "dialog_cancel", key, ident, "dialog would not reopen")
            return
        nav._hold(runner, ["A"], 4, self.phase)          # one page
        nav._hold(runner, [], 20, self.phase)
        for _ in range(12):                              # B-out
            if not _dialog_open_live(runner):
                break
            nav._hold(runner, ["B"], 4, self.phase)
            nav._hold(runner, [], 14, self.phase)
        _close_dialog(runner, self.phase)                # safety: never leave a box open
        summary["verbs"]["dialog_cancel"] += 1

    def _budget_left(self, runner, summary, verb, key) -> int:
        left = self._deadline - runner.frame_idx
        if left <= 0:
            self._skip(summary, verb, key, None, "block frame budget expired")
        return left

    def _ledge_hop(self, runner, mk, key, summary) -> None:
        """One deliberate hop over a jump-ledge edge, verified by the 2-tile move."""
        if self._budget_left(runner, summary, "ledge_hop", key) <= 0:
            return
        t, _, _ = nav._state(runner)
        beh = mk.behaviors(t) if t is not None else None
        if beh is None:
            self._skip(summary, "ledge_hop", key, None, "no behavior table for map")
            return
        walk = ((t.grid >> 10) & 3) == 0
        h, w = walk.shape
        cands = []                                       # (stand, dir, landing) in map coords
        for bv, (dx, dy) in nav._JUMP.items():
            for ly, lx in zip(*np.nonzero(beh == bv)):
                sy, sx, ty_, tx_ = ly - dy, lx - dx, ly + dy, lx + dx
                if (0 <= sy < h and 0 <= sx < w and 0 <= ty_ < h and 0 <= tx_ < w
                        and walk[sy, sx] and walk[ty_, tx_]):
                    cands.append(((sx - 7, sy - 7), (dx, dy), (tx_ - 7, ty_ - 7)))
        if not cands:
            self._skip(summary, "ledge_hop", key, None, "map has no jump-ledge edges")
            return
        for (sx, sy), (dx, dy), (lx2, ly2) in cands[:4]:
            left = self._deadline - runner.frame_idx
            if left <= 0:
                break
            def goal(t_, beh_, sx=sx, sy=sy):
                m = np.zeros(t_.grid.shape, bool)
                if 0 <= sy + 7 < m.shape[0] and 0 <= sx + 7 < m.shape[1]:
                    m[sy + 7, sx + 7] = True
                return m
            r = nav.goto(runner, mk, goal, budget=min(6000, left), phase=self.phase)
            if r == "battle":
                flee_battle(runner)
                summary["battles_fled"] += 1
                continue
            if r != "arrived":
                continue
            nav._step(runner, nav.DIRS[(dx, dy)], max_frames=80)   # the hop (2 cells)
            nav._hold(runner, [], 30, self.phase)                  # land + settle
            t2, x2, y2 = nav._state(runner)
            if t2 is not None and (x2, y2) == (lx2, ly2):
                summary["verbs"]["ledge_hop"] += 1
                return
        self._skip(summary, "ledge_hop", key, None, "no candidate hop committed")

    def _door_bounce(self, runner, mk, key, summary) -> None:
        """Enter a warp, come straight back: the transition pair in both directions."""
        if self._budget_left(runner, summary, "door_bounce", key) <= 0:
            return

        def hazardous(k):                                # link-room / counter-map policy
            return k.startswith("25,") or any(w["dst_map"].startswith("25,")
                                              for w in mk.warps.get(k, []))
        cands = [w for w in mk.warps.get(key, [])
                 if not hazardous(w["dst_map"])
                 and any(b["dst_map"] == key for b in mk.warps.get(w["dst_map"], []))]
        if not cands:
            self._skip(summary, "door_bounce", key, None, "no bounceable warp pair")
            return
        for w in cands[:3]:
            left = self._deadline - runner.frame_idx
            if left <= 0:
                break
            r = nav.goto_warp(runner, mk, w["x"], w["y"], budget=min(8000, left))
            if r == "battle":
                flee_battle(runner)
                summary["battles_fled"] += 1
                continue
            if r != "crossed":
                continue
            dst = w["dst_map"]
            back = next(b for b in mk.warps.get(dst, []) if b["dst_map"] == key)
            r2 = nav.goto_warp(runner, mk, back["x"], back["y"], budget=8000)
            if r2 == "crossed":
                summary["verbs"]["door_bounce"] += 1
                summary["door_pairs"].append([key, dst])
                return
            self._ensure_map(runner, mk, key, summary, budget=8000)   # walk back regardless
        self._skip(summary, "door_bounce", key, None, "warp round trip never completed")

    def _save_dialog(self, runner, key, summary) -> None:
        """START -> SAVE -> A-confirm through to the composed '... saved the game' text
        (gStringVar4; the question/saving boxes are static ROM text and never land
        there), close. Menu slot machinery = the ledger-verified helpers."""
        p = self.phase
        _close_dialog(runner, p)                         # a leftover box eats the START
        if not open_start_menu(runner, p):
            self._skip(summary, "save_dialog", key, None, "start menu never opened")
            return
        slots = start_slots(runner, p)
        if not _seek_start_slot(runner, slots["save"], p):
            close_start_menu(runner, p)
            self._skip(summary, "save_dialog", key, None, "SAVE cursor seek failed")
            return
        nav._hold(runner, ["A"], 4, p)
        nav._hold(runner, [], 40, p)
        before = last_message(GBAState.snapshot(runner.env)).lower()
        saved = False
        for _ in range(60):                              # YES/NO -> (overwrite) -> SAVING...
            msg = last_message(GBAState.snapshot(runner.env)).lower()
            if "saved the" in msg and msg != before:     # freshly composed, not a stale var4
                saved = True
                break
            nav._hold(runner, ["A"], 4, p)
            nav._hold(runner, [], 30, p)
        _close_dialog(runner, p)                         # dismiss the "saved!" box
        close_start_menu(runner, p)                      # (already closed by the save flow)
        if saved:
            summary["verbs"]["save_dialog"] += 1
        else:
            self._skip(summary, "save_dialog", key, None,
                       "'saved the game' text never composed")

    # ---------------------------------------------------------------- per map

    def _map_verbs(self, runner, mk, key, summary) -> None:
        t, _, _ = nav._state(runner)
        objs = list(mk.objects.get(key, []))
        live_extra = []
        if t is not None:
            known = {o["local_id"] for o in objs}
            live_extra = [e for e in npcs(GBAState.snapshot(runner.env))
                          if e.local_id not in known
                          and 0 <= e.x < t.map_width and 0 <= e.y < t.map_height]
        signs = [s for s in mk.signs.get(key, []) if s["kind"] in _SIGN_KINDS]
        bgobj = [s for s in mk.signs.get(key, []) if s["kind"] not in _SIGN_KINDS]
        summary["enumerated"][key] = dict(npcs=len(objs) + len(live_extra),
                                          signs=len(signs), bg_objects=len(bgobj))
        # --- NPCs (manifest templates + live extras); trainers are the spine's job
        for o in objs:
            if o.get("trainer_id"):
                self._skip(summary, "npc_talk", key, o["local_id"],
                           "trainer NPC — battles belong to the spine")
                continue
            self._talk_target(runner, mk, key, "npc_talk", o["local_id"], o["x"], o["y"],
                              summary, live_id=o["local_id"], cancel_pool=True)
        for e in live_extra:
            self._talk_target(runner, mk, key, "npc_talk", f"live:{e.local_id}", e.x, e.y,
                              summary, live_id=e.local_id, cancel_pool=True)
        # --- signs (facing-locked kinds pin the stand tile + facing)
        for s in signs:
            if s["kind"] in _SIGN_SIDE:
                (dx, dy), f = _SIGN_SIDE[s["kind"]]
                sides, facing = [((dx, dy), f)], f
            else:
                sides, facing = None, None
            self._talk_target(runner, mk, key, "sign_read", (s["x"], s["y"]),
                              s["x"], s["y"], summary, cancel_pool=True,
                              facing=facing, sides=sides)
        # --- non-sign bg events: best effort
        for s in bgobj:
            self._talk_target(runner, mk, key, "object_interact", (s["x"], s["y"]),
                              s["x"], s["y"], summary)
        self._ledge_hop(runner, mk, key, summary)
        self._door_bounce(runner, mk, key, summary)

    # ------------------------------------------------------------------ entry

    def run(self, runner, mk, ctx) -> dict:
        random.Random(self.seed)                     # seed reserved (verb-order jitter)
        summary = dict(verbs={v: 0 for v in VERBS}, skipped=[], enumerated={},
                       door_pairs=[], battles_fled=0, frames=0)
        f0 = runner.frame_idx
        deadline = self._deadline = f0 + self.frames
        for key in self.maps:
            if runner.frame_idx >= deadline:
                self._skip(summary, "all", key, None, "block frame budget expired")
                continue
            if not self._ensure_map(runner, mk, key, summary,
                                    budget=max(2000, deadline - runner.frame_idx)):
                self._skip(summary, "all", key, None, "map unreached")
                continue
            self._map_verbs(runner, mk, key, summary)
        if self.save and runner.frame_idx < deadline:
            t, _, _ = nav._state(runner)
            here = f"{t.map_group},{t.map_num}" if t is not None else "?"
            self._save_dialog(runner, here, summary)
        elif self.save:
            self._skip(summary, "save_dialog", None, None, "block frame budget expired")
        summary["frames"] = runner.frame_idx - f0
        return summary
