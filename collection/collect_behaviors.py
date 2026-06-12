"""Deficit-driven behavior collection — Phase 2 of the Rung-A′ plan (shopping list S1–S10).

Each JOB is a player-like behavior policy that the goal-directed producers never exhibited
(measured in `WORLD_MODEL_PLAN_RUNG_A_PRIME_PHASE0.md` §3). Runs record through the SAME
substrate as all existing data (ChunkRecorder + WorldModelSink: RGB + action + full PPU/WRAM +
semantic per frame), so every audit and extractor applies to the new data unchanged.

  idle      S1  stand still at varied spots × facings × durations (NPCs keep wandering)
  fidget    S2-S4  tap-turns, wall bumps, walk bursts, B-held walking (tests run-shoes), pauses
  battle    S5-S7  hunt wild encounters; per battle: fight / catch / flee (faints & catches are
            wanted data; a caught mon joining the party diversifies player-side species)
  menus     S8  seeded random UI walk (START + arrows/A/B with dwells) — whatever screens come up
            get recorded with full state, so even "wrong" navigation is valid data
  dialogue  S9  face NPCs/objects and talk, advancing text with varied styles (mash-A, hold-A,
            slow, B-spam)

Seeding: any savestate (the storyline segments' initial/final.state files form the bank).

Usage:
  CUDA_VISIBLE_DEVICES= .venv/bin/python -m collection.collect_behaviors \
      --job idle --load_state <file.state> --output data/behaviors/idle__SEG --frames 6000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from collection.direct_runner import DirectEmulatorRunner
from collection.heatz_adapter import build_heatz_state, handle_battle
from collection.recorder import ChunkRecorder
from collection.world_model_sink import WorldModelSink

DIRS = ("UP", "DOWN", "LEFT", "RIGHT")
IDLE_DURATIONS = (30, 60, 120, 240, 420)         # frames @60fps: 0.5s … 7s


def _hold(runner, buttons: list[str], frames: int, phase: str):
    d = next((b for b in buttons if b in DIRS), None)
    if d is not None:
        from collection.actions import update_facing
        runner.facing = update_facing(runner.facing, d)   # keep the semantic facing tracker honest
    for _ in range(frames):
        runner.step_frame(buttons, phase=phase, record_state=False)


def _in_battle(runner) -> bool:
    return bool(runner.nav_state().in_battle)


def _tap_turn(runner, d: str):
    """A short directional press turns in place without stepping."""
    _hold(runner, [d], 4, "fidget")
    _hold(runner, [], 6, "fidget")


# ---------------------------------------------------------------- jobs

def job_idle(runner, rng: random.Random, budget: int):
    """Walk a short burst to a new spot, then stand still through varied facings/durations."""
    while runner.frame_idx < budget:
        for _ in range(rng.randint(2, 6)):                    # relocate
            _hold(runner, [rng.choice(DIRS)], rng.randint(16, 24), "idle_move")
            if _in_battle(runner):
                _battle_one(runner, rng, "run")
        _hold(runner, [], 12, "idle")
        for d in rng.sample(DIRS, k=rng.randint(1, 4)):       # stand, facing around
            _tap_turn(runner, d)
            _hold(runner, [], rng.choice(IDLE_DURATIONS), "idle")


def job_fidget(runner, rng: random.Random, budget: int):
    """Player-like input noise: turns, bumps, bursts, B-held walking, pauses."""
    while runner.frame_idx < budget:
        kind = rng.random()
        d = rng.choice(DIRS)
        if kind < 0.25:
            _tap_turn(runner, d)
        elif kind < 0.50:
            _hold(runner, [d], rng.randint(24, 60), "fidget_bump_or_walk")
        elif kind < 0.70:
            _hold(runner, ["B", d], rng.randint(24, 60), "fidget_b_walk")
        elif kind < 0.85:
            _hold(runner, [], rng.randint(10, 60), "fidget_pause")
        else:
            for _ in range(rng.randint(2, 5)):                # quick zigzag
                _hold(runner, [rng.choice(DIRS)], rng.randint(8, 18), "fidget_zigzag")
        if _in_battle(runner):
            _battle_one(runner, rng, "run")


def _balls_qty(runner) -> int:
    """Poké Ball count from the bag (security-key decode); -1 if the slot isn't Poké Balls."""
    from collection.extractors.ram import GBAState
    st = GBAState.snapshot(runner.env)
    key16 = st.u32(st.u32(0x03005D90) + 0xAC) & 0xFFFF
    a = st.u32(0x03005D8C) + 0x650
    return st.u16(a + 2) ^ key16 if st.u16(a) == 4 else -1


def _await_battle_menu(runner, max_advances: int = 60) -> bool:
    """Advance battle text with A until the action menu ('What will X do?') is up."""
    from collection.extractors.ram import GBAState
    from collection.extractors.text import last_message
    for _ in range(max_advances):
        if not _in_battle(runner):
            return False
        if last_message(GBAState.snapshot(runner.env), in_battle=True).startswith("What will"):
            return True
        _hold(runner, ["A"], 4, "battle_text"); _hold(runner, [], 24, "battle_text")
    return False


def _throw_ball(runner) -> bool:
    """One Poké Ball throw from the action menu. The battle bag REMEMBERS its pocket across opens
    (visually established 2026-06-12), so blind navigation desyncs; we self-align with ball-count
    feedback: if a throw didn't consume a ball, rotate one more pocket next time (5 pockets)."""
    before = _balls_qty(runner)
    if before <= 0:
        return False
    rights = getattr(runner, "_bag_rights", 1)        # pocket steps after opening (1 on first open:
    for b in ["RIGHT", "A"] + ["RIGHT"] * rights + ["A", "A"]:        # the bag starts on ITEMS)
        _hold(runner, [b], 4, "battle_catch"); _hold(runner, [], 50, "battle_catch")
    _hold(runner, [], 700, "battle_catch")            # throw + shakes (or whatever happened)
    after = _balls_qty(runner)
    if after == before:                               # didn't throw: mis-aligned pocket — rotate
        runner._bag_rights = (rights + 1) % 5
        for _ in range(4):                            # close any half-open UI
            _hold(runner, ["B"], 4, "battle_catch"); _hold(runner, [], 20, "battle_catch")
        return False
    runner._bag_rights = 0                            # bag auto-closed on the BALLS pocket
    return True


def _battle_one(runner, rng: random.Random, strategy: str, max_steps: int = 400):
    """Play out one battle with the heatz battle machine ('fight'/'run'); 'catch' = weaken with one
    fight turn, then up to 4 self-aligning ball throws (nickname/dex prompts are declined by the
    fight fallback's navigation). Faints/whiteouts/breakouts are fine — wanted data."""
    steps = 0
    if strategy == "catch":
        if _await_battle_menu(runner):                # one FIGHT turn to weaken (better odds)
            _hold(runner, ["A"], 4, "battle_catch"); _hold(runner, [], 40, "battle_catch")
            _hold(runner, ["A"], 4, "battle_catch"); _hold(runner, [], 700, "battle_catch")
        throws = 0
        while throws < 4 and _in_battle(runner) and _await_battle_menu(runner):
            if _throw_ball(runner):
                throws += 1
        # caught -> nickname/dex prompts; broke out -> battle continues: both handled below
        strategy = "fight"
    ui_mem: dict = {}                  # handle_battle keeps cursor-sequence counters in the state
    while _in_battle(runner) and steps < max_steps:                    # dict — persist them across
        st = build_heatz_state(runner.env, frame_idx=runner.frame_idx,  # rebuilds or the cursor
                               story_bucket="behaviors", facing=runner.facing)   # never advances
        st.update(ui_mem)
        act = handle_battle(st, strategy=strategy)
        ui_mem = {k: v for k, v in st.items() if k.startswith("_") and k != "_env"}
        runner.perform_action(act, speed="fast", record_end_state=False)
        steps += 1
    _hold(runner, [], 30, "battle_settle")


def job_battle(runner, rng: random.Random, budget: int, catchy: bool = False):
    """Hunt encounters by wandering; resolve each with a mixed policy ('catchy' = catch-heavy,
    for the closure catch axis: full ball pockets from the constructed bank)."""
    while runner.frame_idx < budget:
        _hold(runner, [rng.choice(DIRS)], rng.randint(16, 48), "battle_hunt")
        if _in_battle(runner):
            r = rng.random()
            if catchy:
                strategy = "catch" if r < 0.7 else "fight"
            else:
                strategy = "fight" if r < 0.55 else ("catch" if r < 0.8 else "run")
            _battle_one(runner, rng, strategy)


def job_menus(runner, rng: random.Random, budget: int):
    """Seeded UI walk: open START, wander the menus with dwells, eventually back out."""
    while runner.frame_idx < budget:
        runner.perform_action("START", speed="normal", record_end_state=False)
        _hold(runner, [], 30, "menu")
        for _ in range(rng.randint(4, 14)):
            act = rng.choice(["UP", "DOWN", "LEFT", "RIGHT", "A", "A", "B"])
            runner.perform_action(act, speed="normal", record_end_state=False)
            _hold(runner, [], rng.randint(10, 50), "menu_dwell")
            if _in_battle(runner):
                _battle_one(runner, rng, "run")
        for _ in range(6):                                    # make sure we're back outside
            runner.perform_action("B", speed="fast", record_end_state=False)
        _hold(runner, [], 20, "menu")
        for _ in range(rng.randint(1, 4)):                    # roam a little between menu dives
            _hold(runner, [rng.choice(DIRS)], rng.randint(16, 32), "menu_roam")


ADVANCE_STYLES = ("mash_a", "hold_a", "slow_a", "b_spam")


def _advance_dialog(runner, rng: random.Random, style: str, max_frames: int = 1200):
    from collection.heatz_adapter import _visual_dialog_open
    waited = 0
    while waited < max_frames and (_visual_dialog_open(runner.env) or False):
        if style == "mash_a":
            _hold(runner, ["A"], 2, "dialog"); _hold(runner, [], 3, "dialog")
        elif style == "hold_a":
            _hold(runner, ["A"], 24, "dialog")
        elif style == "slow_a":
            _hold(runner, [], rng.randint(30, 90), "dialog"); _hold(runner, ["A"], 2, "dialog")
        else:                                                  # b_spam
            _hold(runner, ["B"], 2, "dialog"); _hold(runner, [], 3, "dialog")
        waited += 30
    _hold(runner, [], 10, "dialog")


def job_dialogue(runner, rng: random.Random, budget: int):
    """Face every direction and press A (signs/NPCs), advancing with varied styles; roam between."""
    while runner.frame_idx < budget:
        for d in rng.sample(DIRS, k=4):
            _tap_turn(runner, d)
            runner.perform_action("A", speed="normal", record_end_state=False)
            _hold(runner, [], 20, "dialog_probe")
            _advance_dialog(runner, rng, rng.choice(ADVANCE_STYLES))
            if _in_battle(runner):
                _battle_one(runner, rng, "run")
        for _ in range(rng.randint(2, 6)):
            _hold(runner, [rng.choice(DIRS)], rng.randint(16, 32), "dialog_roam")


def job_run(runner, rng: random.Random, budget: int):
    """S2: sustained B-held running (the gait the corpus never had). Long runs, wall turns,
    occasional walk/run alternation so both gaits appear in the same places."""
    d = rng.choice(DIRS)
    while runner.frame_idx < budget:
        before = runner.nav_state()
        _hold(runner, ["B", d], rng.randint(40, 120), "run")
        after = runner.nav_state()
        if (before.x, before.y, before.map) == (after.x, after.y, after.map):
            d = rng.choice([x for x in DIRS if x != d])      # wall — turn
        if rng.random() < 0.25:
            _hold(runner, [d], rng.randint(24, 48), "run_walk_mix")
        if rng.random() < 0.15:
            _hold(runner, [], rng.randint(12, 40), "run_pause")
        if _in_battle(runner):
            _battle_one(runner, rng, "run")


def job_battle_nav(runner, rng: random.Random, budget: int, catchy: bool = False):
    """Navigator-driven battle farming: WALK TO grass (BFS over live collision + ROM behaviors),
    pace inside it until an encounter, resolve, repeat — replaces blind wandering, which failed
    to find grass at all from some spawns (waves 1-3 lesson)."""
    from collection.navigator import MapKnowledge, goto_grass, pace_grass
    mk = MapKnowledge()
    while runner.frame_idx < budget:
        r = goto_grass(runner, mk, budget=6000)
        if r == "arrived":
            r = pace_grass(runner, mk, rng, budget=4000)
        if not _in_battle(runner):
            if r in ("stuck", "budget", "left"):              # fall back to a wander burst, retry
                _hold(runner, [rng.choice(DIRS)], rng.randint(16, 48), "battle_hunt")
            continue
        rr = rng.random()
        if catchy:
            strategy = "catch" if rr < 0.7 else "fight"
        else:
            strategy = "fight" if rr < 0.6 else ("catch" if rr < 0.85 else "run")
        _battle_one(runner, rng, strategy)


def job_warp_cycle(runner, rng: random.Random, budget: int):
    """Warp coverage: repeatedly cross doors/connections ON PURPOSE (manifest warp events +
    navigator). Bouncing between maps cycles pairs in both directions; battles en route are fled.

    Hazard policy: wireless-club counters (warp dst map group 25 — Union/Trade link rooms) are
    MULTIPLAYER features, excluded by policy; maps containing them (Center 2Fs) are not entered —
    their attendant's auto-greeting script can trap a run at the counter (found the hard way).
    Unforeseen traps: a map that yields repeated 'stuck' gets blacklisted + an escape burst."""
    from collection.navigator import MapKnowledge, _state, _unstick, goto_warp
    mk = MapKnowledge()

    def hazardous(map_key: str) -> bool:
        return any(w["dst_map"].startswith("25,") for w in mk.warps.get(map_key, []))

    blacklist: set[str] = set()
    last = None
    stuck_here = 0
    while runner.frame_idx < budget:
        if _in_battle(runner):
            _battle_one(runner, rng, "run")
        t, x, y = _state(runner)
        if t is None:
            _hold(runner, [], 30, "warp_cycle"); continue
        key = f"{t.map_group},{t.map_num}"
        warps = [w for w in mk.warps.get(key, [])
                 if not w["dst_map"].startswith("25,")        # link rooms: policy-excluded
                 and not hazardous(w["dst_map"])              # don't enter counter maps
                 and w["dst_map"] not in blacklist]
        if not warps:
            _hold(runner, [rng.choice(DIRS)], rng.randint(16, 48), "warp_roam"); continue
        cand = [w for w in warps if (w["x"], w["y"]) != last] or warps
        w = rng.choice(cand)
        r = goto_warp(runner, mk, w["x"], w["y"], budget=6000)
        if r == "crossed":
            last, stuck_here = None, 0                        # arrived on the far-side warp tile
            _hold(runner, [], rng.randint(20, 60), "warp_cycle")
        elif r == "battle":
            continue
        else:
            last = (w["x"], w["y"])                           # unreachable warp: try another next
            stuck_here += 1
            if stuck_here >= 3:                               # trapped? escape + blacklist the map
                blacklist.add(key)
                _unstick(runner, "warp_cycle")
                for _ in range(6):
                    _hold(runner, [rng.choice(DIRS)], rng.randint(24, 48), "warp_escape")
                stuck_here = 0
            _hold(runner, [rng.choice(DIRS)], rng.randint(16, 32), "warp_roam")


def _cast_rod(runner) -> bool:
    """One Old Rod cast via the overworld bag; True when the cast visibly started (the fishing
    dots box opens). Anchors the START-menu cursor (UP×7 — no wrap), then BAG = DOWN×2; the bag
    REMEMBERS its pocket across opens (the ball-thrower lesson), so the KEY-ITEMS pocket offset
    self-aligns on dots-box feedback. The fishing UI bypasses the text printers entirely, so the
    window mask is the only cast signal."""
    from collection.navigator import _dialog_open
    def press(b, wait=35):
        _hold(runner, [b], 4, "fish"); _hold(runner, [], wait, "fish")
    lefts = getattr(runner, "_bag_lefts", 1)                  # 1 on first open (bag starts ITEMS)
    press("START", 50)
    for _ in range(7):
        press("UP", 10)
    press("DOWN"); press("DOWN"); press("A", 70)              # BAG
    for _ in range(lefts):
        press("LEFT", 35)                                     # to KEY ITEMS
    press("A", 40); press("A", 50)                            # OLD ROD -> USE
    for _ in range(10):
        _hold(runner, [], 10, "fish")
        if _dialog_open(runner):                              # the dots box: cast confirmed
            runner._bag_lefts = 0
            return True
    runner._bag_lefts = (lefts + 1) % 5                       # mis-aligned pocket: rotate
    for _ in range(5):
        press("B", 20)
    return False


def job_fish(runner, rng: random.Random, budget: int):
    """Fishing: the ONLY pre-badge access to water/fish encounter tables (7 species, 0 frames
    without this). Navigate to a castable shore cell, face the water, cast, hook on a timer
    (the bite is invisible to the text extractor), resolve the battle, repeat."""
    from collection.navigator import (DIRS as NDIRS, MapKnowledge, WATER, _state, _step,
                                      _unstick, goto, water_adjacent_goal)
    import numpy as np
    mk = MapKnowledge()
    while runner.frame_idx < budget:
        r = goto(runner, mk, water_adjacent_goal, budget=8000, phase="fish_nav")
        if r == "battle":
            _battle_one(runner, rng, "fight" if rng.random() < 0.7 else "run")
            continue
        if r != "arrived":
            _hold(runner, [rng.choice(DIRS)], rng.randint(16, 48), "fish_nav")
            continue
        t, x, y = _state(runner)
        beh = mk.behaviors(t)
        if beh is None:
            continue
        water = np.isin(beh, list(WATER))
        for (dx, dy), d in NDIRS.items():
            if water[y + 7 + dy, x + 7 + dx]:
                _step(runner, d, max_frames=12)               # tap-face the water
                break
        for cast in range(8):
            if runner.frame_idx >= budget:
                return
            if not _cast_rod(runner):
                continue
            _hold(runner, [], rng.randint(50, 160), "fish")   # the dots
            for _ in range(20):                               # hook attempts
                _hold(runner, ["A"], 4, "fish"); _hold(runner, [], 12, "fish")
                if _in_battle(runner):
                    break
            _hold(runner, [], 240, "fish")                    # battle intro or back to field
            if _in_battle(runner):
                _battle_one(runner, rng, "fight" if rng.random() < 0.6 else "catch")
            _unstick(runner, "fish")


def job_dialogue_nav(runner, rng: random.Random, budget: int):
    """Dialogue EXHAUSTION: walk to every NPC and sign on the map ON PURPOSE and talk/read,
    advancing with varied styles (vs the old job's blind facing-and-pressing). NPCs are live
    entity records (they wander — re-target on arrival); signs come from the ROM manifest."""
    from collection.extractors.entities import npcs
    from collection.extractors.ram import GBAState
    from collection.navigator import DIRS as ND
    from collection.navigator import MapKnowledge, _clear_dialog, _state, goto
    import numpy as np
    mk = MapKnowledge()
    visited: set = set()
    while runner.frame_idx < budget:
        if _in_battle(runner):
            _battle_one(runner, rng, "run")
        t, x, y = _state(runner)
        if t is None:
            _hold(runner, [], 30, "dialog_nav"); continue
        key = f"{t.map_group},{t.map_num}"
        targets = [("npc", e.local_id, e.x, e.y) for e in npcs(GBAState.snapshot(runner.env))
                   if ("npc", key, e.local_id) not in visited and abs(e.x) < 200]
        targets += [("sign", (sg["x"], sg["y"]), sg["x"], sg["y"]) for sg in mk.signs.get(key, [])
                    if ("sign", key, (sg["x"], sg["y"])) not in visited]
        if not targets:
            _hold(runner, [rng.choice(DIRS)], rng.randint(16, 48), "dialog_roam"); continue
        kind, ident, tx, ty = min(targets, key=lambda c: abs(c[2] - x) + abs(c[3] - y))

        def goal(t_, beh, tx=tx, ty=ty):
            m = np.zeros(t_.grid.shape, bool)
            for dx, dy in ND:
                gx, gy = tx + dx + 7, ty + dy + 7
                if 0 <= gy < m.shape[0] and 0 <= gx < m.shape[1]:
                    m[gy, gx] = True
            return m & (((t_.grid >> 10) & 3) == 0)

        r = goto(runner, mk, goal, budget=5000, phase="dialog_nav")
        visited.add((kind, key, ident))                      # tried — don't loop on the unreachable
        if r != "arrived":
            continue
        t, x, y = _state(runner)
        d = {(1, 0): "RIGHT", (-1, 0): "LEFT", (0, 1): "DOWN", (0, -1): "UP"}.get(
            (max(-1, min(1, tx - x)), max(-1, min(1, ty - y))), "UP")
        _tap_turn(runner, d)
        runner.perform_action("A", speed="normal", record_end_state=False)
        _hold(runner, [], 25, "dialog_nav")
        _advance_dialog(runner, rng, rng.choice(ADVANCE_STYLES))
        _clear_dialog(runner, "dialog_nav")


# the labeled-menu script: each section anchors the START cursor (UP x7, no wrap), opens one
# entry, walks a couple of in-screen stages, then B-retreats to the field
MENU_SCRIPT = (
    ("start_menu",   0, (), 50),
    ("pokedex",      0, ("A", "DOWN", "A", "RIGHT"), 70),
    ("party",        1, ("A", "DOWN"), 70),
    ("summary_info", 1, ("A", "A"), 80),
    ("summary_skills", 1, ("A", "A", "RIGHT"), 80),
    ("bag_items",    2, ("A",), 70),
    ("bag_balls",    2, ("A", "RIGHT"), 70),
    ("bag_tms",      2, ("A", "RIGHT", "RIGHT"), 70),
    ("bag_keyitems", 2, ("A", "LEFT"), 70),
    ("trainer_card", 3, ("A",), 90),
    ("save_dialog",  4, ("A",), 80),
    ("options",      5, ("A", "DOWN", "RIGHT"), 80),
)


def job_menus_labeled(runner, rng: random.Random, budget: int):
    """The LABELED menu crawler: deterministic screen visits with a labels sidecar
    (labels.jsonl: {label, start, end} frame ranges) — turns the menus axis from a geometry
    proxy into measurable per-screen coverage."""
    labels = []
    while runner.frame_idx < budget:
        for label, slot, presses, dwell in MENU_SCRIPT:
            if runner.frame_idx >= budget:
                break
            runner.perform_action("START", speed="normal", record_end_state=False)
            _hold(runner, [], 30, "menus")
            for _ in range(7):
                _hold(runner, ["UP"], 4, "menus"); _hold(runner, [], 10, "menus")
            for _ in range(slot):
                _hold(runner, ["DOWN"], 4, "menus"); _hold(runner, [], 12, "menus")
            start = runner.frame_idx
            for b in presses:
                _hold(runner, [b], 4, "menus"); _hold(runner, [], 35, "menus")
            _hold(runner, [], dwell, "menus")
            labels.append({"label": label, "start": start, "end": runner.frame_idx})
            for _ in range(8):                               # retreat to the field
                _hold(runner, ["B"], 4, "menus"); _hold(runner, [], 18, "menus")
        for _ in range(rng.randint(2, 5)):                   # roam between sweeps
            _hold(runner, [rng.choice(DIRS)], rng.randint(16, 32), "menu_roam")
            if _in_battle(runner):
                _battle_one(runner, rng, "run")
    runner._menu_labels = labels                             # collect_behavior persists this


def job_story(runner, rng: random.Random, budget: int):
    """One-off scene replayer: a generic story-advancer for scripted sequences (intro + naming
    screen + truck + Birch rescue). Dialog boxes advance with varied styles; choice menus get A
    (first option — keeps gender=BOY policy) with occasional DOWN first (varies the preset NAME
    pick); otherwise watch or wander gently. Seeded at start/truck states ×N = the one-off axis."""
    from collection.navigator import _dialog_open
    while runner.frame_idx < budget:
        if _in_battle(runner):
            _battle_one(runner, rng, "fight")
            continue
        if _dialog_open(runner):
            style = rng.choice(ADVANCE_STYLES)
            if style == "b_spam":
                _hold(runner, ["B"], 3, "story"); _hold(runner, [], 10, "story")
            elif style == "slow_a":
                _hold(runner, [], rng.randint(20, 60), "story")
                _hold(runner, ["A"], 3, "story"); _hold(runner, [], 10, "story")
            else:
                _hold(runner, ["A"], 3, "story"); _hold(runner, [], 12, "story")
            continue
        r = rng.random()
        if r < 0.45:
            _hold(runner, [], rng.randint(10, 40), "story")            # watch the cutscene
        elif r < 0.75:
            if rng.random() < 0.3:
                _hold(runner, ["DOWN"], 4, "story"); _hold(runner, [], 12, "story")
            _hold(runner, ["A"], 4, "story"); _hold(runner, [], 20, "story")
        else:
            _hold(runner, [rng.choice(DIRS)], rng.randint(8, 24), "story")


import functools

JOBS = {"idle": job_idle, "fidget": job_fidget, "battle": job_battle,
        "battle_catch": functools.partial(job_battle, catchy=True),
        "battle_nav": job_battle_nav,
        "battle_nav_catch": functools.partial(job_battle_nav, catchy=True),
        "warp_cycle": job_warp_cycle,
        "fish": job_fish,
        "dialogue_nav": job_dialogue_nav,
        "menus_labeled": job_menus_labeled,
        "story": job_story,
        "menus": job_menus, "dialogue": job_dialogue, "run": job_run}


def collect_behavior(*, job: str, load_state: str, output_dir: str, rom_path: str,
                     frames: int, seed: int = 0, backend: str = "npz") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with ChunkRecorder(out, run_id=f"behavior_{job}", emulator_fps=60, visual_fps=60,
                       backend=backend, lean=True, metadata={"story_bucket": f"behavior_{job}"}) as rec:
        sink = WorldModelSink(out)
        runner = DirectEmulatorRunner(rom_path=rom_path, load_state=load_state,
                                      story_bucket=f"behavior_{job}", recorder=rec,
                                      emulator_fps=60, frame_hook=sink.capture)
        runner.initialize(); sink.capture(runner)
        runner.settle_to_free_overworld()
        JOBS[job](runner, rng, frames)
        n = runner.frame_idx
        sink.close(); runner.close()
    labels = getattr(runner, "_menu_labels", None)
    if labels:
        (out / "labels.jsonl").write_text("\n".join(json.dumps(l) for l in labels))
    summary = {"job": job, "frames": n, "seed": seed, "load_state": str(load_state)}
    (out / "behavior_summary.json").write_text(json.dumps(summary))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=sorted(JOBS))
    ap.add_argument("--load_state", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rom_path", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--frames", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="npz")
    args = ap.parse_args()
    print(json.dumps(collect_behavior(job=args.job, load_state=args.load_state,
                                      output_dir=args.output, rom_path=args.rom_path,
                                      frames=args.frames, seed=args.seed, backend=args.backend)))


if __name__ == "__main__":
    main()
