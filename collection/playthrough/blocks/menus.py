"""MENUS expedition block (W33 corpus-v2 §2): START-menu browsing with RAM-verified
screens — party list -> slot-0 summary (the move/PP revelation view for the ledger's
knowability audit) -> back out, BAG -> pocket browse -> back out, close.

Every screen is confirmed through the ledger's OWN verified ui_state sources
(extractors/ledger_panel: start-menu window byte 0x0203CD8C, party/summary/bag cb2s —
all live-verified in the §10.2 work; constants imported, never redefined), so the
summary counts are ground truth, not press-count hope. Screen polls use the cheap
env-backed reads (cb2/menu byte are RAM-bus); the full-snapshot ui_state read runs
only at confirmation moments (tight snapshot polling core-dumps mgba — the fishing
lesson) and feeds `ui_trace`, the block's recorded transition evidence.

Slot layout: the START cursor is sought by RAM feedback (collect_behaviors'
wrap-proof `_seek_start_slot`); BAG's slot is detected once per menu open via the
seek-slot-6 probe (slot 6 exists only in the 7-entry post-Pokédex menu), and
PARTY/SAVE derive from it (party = bag-1, save = bag+2 — the fixed Emerald order).
"""
from __future__ import annotations

import random

from collection import navigator as nav
from collection.collect_behaviors import SLOT_BAG, _seek_start_slot
from collection.extractors.ledger_panel import (
    CB2_ADDR, CB2_BAG, CB2_OVERWORLD, CB2_PARTY_MENU, CB2_SUMMARY, PARTY_COUNT_ADDR,
    START_MENU_WINDOW_ID, UI_BAG, UI_PARTY_MENU, UI_START_MENU, UI_SUMMARY, ui_state,
)
from collection.extractors.ram import GBAState
from collection.playthrough.blocks.base import flee_battle


def start_menu_open(runner) -> bool:
    """The verified openness byte (0xFF = no start-menu window; ledger_panel §10.2)."""
    return GBAState.from_env(runner.env).u8(START_MENU_WINDOW_ID) != 0xFF


def _poll(runner, pred, frames: int, phase: str, every: int = 4) -> bool:
    """Poll a cheap env-backed predicate while idling recorded frames."""
    for _ in range(max(1, frames // every)):
        if pred(GBAState.from_env(runner.env)):
            return True
        nav._hold(runner, [], every, phase)
    return pred(GBAState.from_env(runner.env))


def open_start_menu(runner, phase: str) -> bool:
    for _ in range(4):
        if start_menu_open(runner):
            nav._hold(runner, [], 20, phase)             # let the window finish drawing
            return True
        nav._hold(runner, ["START"], 4, phase)
        if _poll(runner, lambda st: st.u8(START_MENU_WINDOW_ID) != 0xFF, 60, phase):
            nav._hold(runner, [], 20, phase)
            return True
    return False


def close_start_menu(runner, phase: str) -> bool:
    for _ in range(8):
        if not start_menu_open(runner):
            nav._hold(runner, [], 20, phase)
        if not start_menu_open(runner):
            return True
        nav._hold(runner, ["B"], 4, phase)
        nav._hold(runner, [], 18, phase)
    return not start_menu_open(runner)


def start_slots(runner, phase: str) -> dict[str, int]:
    """{'party','bag','save'} cursor slots for the CURRENT menu, detected once per
    runner via the seek-slot-6 probe (the _cast_rod pattern: slot 6 exists only in the
    7-entry post-Pokédex menu; the cursor byte wraps mod entry count otherwise).
    Call with the start menu OPEN."""
    bag = getattr(runner, "_bag_slot", None)
    if bag is None:
        bag = SLOT_BAG if _seek_start_slot(runner, 6, phase) else 1
        runner._bag_slot = bag
    return {"party": bag - 1, "bag": bag, "save": bag + 2}


class Menus:
    name = "menus"
    phase = "menus"

    def __init__(self, script: list | tuple = ("party", "summary", "bag"),
                 frames: int = 8000, seed: int = 0):
        self.script = tuple(script)
        self.frames = frames
        self.seed = seed                                 # reserved: dwell jitter (unused today)

    # ------------------------------------------------------------------ views

    def _back_to_start_menu(self, runner, summary: dict) -> bool:
        """B out of an own-cb2 screen until the START menu is redrawn over the field.
        Measured (Oldale probe, 2026-08-20): the start-menu window id stays allocated
        (0x1) THROUGH party/summary/bag, so 'back at the start menu' is cb2==overworld
        AND byte!=0xFF; and B from the summary returns to the party list with the
        action popup re-shown — the exit needs popup-close + list-exit B presses."""
        p = self.phase
        for _ in range(6):
            if _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_OVERWORLD
                     and st.u8(START_MENU_WINDOW_ID) != 0xFF, 40, p):
                nav._hold(runner, [], 20, p)
                return self._confirm(runner, UI_START_MENU, summary)
            nav._hold(runner, ["B"], 4, p)
            nav._hold(runner, [], 30, p)
        return False

    def _confirm(self, runner, want: int, summary: dict) -> bool:
        """Full-snapshot ui_state confirmation (the RAM-byte evidence) + trace entry."""
        u = ui_state(GBAState.snapshot(runner.env))
        if u == want and (not summary["ui_trace"] or summary["ui_trace"][-1] != u):
            summary["ui_trace"].append(u)
        return u == want

    def _party_and_summary(self, runner, slots, summary: dict, want_summary: bool) -> None:
        p = self.phase
        if not _seek_start_slot(runner, slots["party"], p):
            summary["skipped"].append(dict(view="party", reason="cursor seek failed"))
            return
        nav._hold(runner, ["A"], 4, p)
        if not _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_PARTY_MENU, 300, p):
            summary["skipped"].append(dict(view="party", reason="party cb2 never came up"))
            return
        nav._hold(runner, [], 40, p)                     # party screen init
        if self._confirm(runner, UI_PARTY_MENU, summary):
            summary["party_menu_views"] += 1
        if want_summary:
            nav._hold(runner, ["A"], 4, p)               # select slot 0 -> action popup
            nav._hold(runner, [], 30, p)                 # (same cb2 — ledger-verified)
            nav._hold(runner, ["A"], 4, p)               # SUMMARY (first option)
            if _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_SUMMARY, 300, p):
                nav._hold(runner, [], 40, p)
                if self._confirm(runner, UI_SUMMARY, summary):
                    summary["summary_views"] += 1
                for _ in range(2):                       # INFO -> SKILLS -> BATTLE MOVES:
                    nav._hold(runner, ["RIGHT"], 4, p)   # the move/PP revelation pages
                    nav._hold(runner, [], 45, p)
                nav._hold(runner, ["B"], 4, p)           # back to the party list (the
                _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_PARTY_MENU, 300, p)
                nav._hold(runner, [], 30, p)             # action popup re-shows here)
                self._confirm(runner, UI_PARTY_MENU, summary)
            else:
                summary["skipped"].append(dict(view="summary", reason="summary cb2 never came up"))
        self._back_to_start_menu(runner, summary)

    def _bag(self, runner, slots, summary: dict) -> None:
        p = self.phase
        if not _seek_start_slot(runner, slots["bag"], p):
            summary["skipped"].append(dict(view="bag", reason="cursor seek failed"))
            return
        nav._hold(runner, ["A"], 4, p)
        if not _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_BAG, 300, p):
            summary["skipped"].append(dict(view="bag", reason="bag cb2 never came up"))
            return
        nav._hold(runner, [], 40, p)
        if self._confirm(runner, UI_BAG, summary):
            summary["bag_views"] += 1
        for b in ("RIGHT", "DOWN", "DOWN", "UP"):        # browse a pocket
            nav._hold(runner, [b], 4, p)
            nav._hold(runner, [], 25, p)
        self._back_to_start_menu(runner, summary)

    # ------------------------------------------------------------------ entry

    def run(self, runner, mk, ctx) -> dict:
        random.Random(self.seed)                         # seed accepted for future jitter
        summary = dict(party_menu_views=0, summary_views=0, bag_views=0,
                       ui_trace=[], skipped=[], frames=0)
        f0 = runner.frame_idx
        if runner.nav_state().in_battle:                 # can't menu from a battle
            flee_battle(runner)
        if not open_start_menu(runner, self.phase):
            summary["skipped"].append(dict(view="start", reason="start menu never opened"))
            summary["frames"] = runner.frame_idx - f0
            return summary
        self._confirm(runner, UI_START_MENU, summary)
        slots = start_slots(runner, self.phase)
        have_party = GBAState.from_env(runner.env).u8(PARTY_COUNT_ADDR) > 0
        if ("party" in self.script or "summary" in self.script) and not have_party:
            summary["skipped"].append(dict(view="party", reason="empty party — no POKéMON entry"))
        elif "party" in self.script or "summary" in self.script:
            self._party_and_summary(runner, slots, summary,
                                    want_summary="summary" in self.script)
        if "bag" in self.script:
            self._bag(runner, slots, summary)
        close_start_menu(runner, self.phase)
        summary["ui_trace"].append(self._observe_overworld(runner, mk))
        summary["frames"] = runner.frame_idx - f0
        return summary

    def _observe_overworld(self, runner, mk) -> int:
        """Observe the post-close overworld ui_state. The BG0 band keeps stale menu
        tiles until a scroll redraws it (the documented dialog-residue limitation), so
        when the standing read stays dirty, take one safe step out and back (walkable,
        not grass, not a warp mat — position-neutral) to force the redraw."""
        u = 0
        for _ in range(6):
            u = ui_state(GBAState.snapshot(runner.env))
            if u == 0:
                return u
            nav._hold(runner, [], 10, self.phase)
        t, x, y = nav._state(runner)
        if t is None:
            return u
        beh = mk.behaviors(t)
        walk = ((t.grid >> 10) & 3) == 0
        warp_tiles = {(w["x"], w["y"]) for w in mk.warps.get(f"{t.map_group},{t.map_num}", [])}
        back = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        order = {"DOWN": 0, "RIGHT": 1, "LEFT": 2, "UP": 3}    # DOWN scrolls the stale
        for (dx, dy), d in sorted(nav.DIRS.items(),           # rows OUT of the band
                                  key=lambda kv: order[kv[1]]):
            steps = 0
            for k in (1, 2):
                nx, ny = x + k * dx, y + k * dy
                if not (0 <= ny + 7 < walk.shape[0] and 0 <= nx + 7 < walk.shape[1]):
                    break
                if (not walk[ny + 7, nx + 7] or (nx, ny) in warp_tiles
                        or (beh is not None and beh[ny + 7, nx + 7] == nav.GRASS)):
                    break
                if not nav._step(runner, d):
                    break
                steps += 1
                for _ in range(3):
                    u = ui_state(GBAState.snapshot(runner.env))
                    if u == 0:
                        break
                    nav._hold(runner, [], 8, self.phase)
                if u == 0:
                    break
            for _ in range(steps):
                nav._step(runner, back[d])
            if steps:
                break
        return u
