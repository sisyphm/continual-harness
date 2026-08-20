"""W33 corpus-v2 §2 expedition blocks, item 3b: interaction + grind_evolve + idle +
menus. Real emulator, real recorder, storyline final.states (skipped cleanly when the
storyline corpus isn't mounted). Battles are stochastic — encounter assertions are
>= 1 within a generous budget, never exact counts.

Measured ground truth these tests pin (2026-08-20 build session):
- Oldale Town (0,10) @ OLDALE_TOWN: 4 manifest NPC templates + 5 signs (kinds 0/1),
  0 non-sign bg_events, a bounceable Center door pair; the pre-Pokédex 6-entry START
  menu (BAG detected at slot 1 via the seek-slot-6 probe, SAVE at 3).
- START-menu machinery: the start-menu window id byte stays ALLOCATED (0x1) through
  the party/summary/bag screens — "back at the start menu" is cb2==overworld AND
  byte!=0xFF; B from the summary re-shows the party action popup (exit needs extra Bs).
- Post-close ui reads 1 for a while (stale BG0 menu tiles — the documented residue);
  a 1-2 tile walk scrolls them out and ui 0 is genuinely observed (the menus block's
  final trace entry).
- Route 101 @ BACK_TO_ROUTE101_FROM_OLDALE: lv-8 Mudkip lead (hp 16/28), wild battles
  every ~0.5-1.5k paced grass frames, lv 2-3 foes — the battle-win path closes fast.
- Evolution completion: no storyline fixture carries a lv-15 lead, so the
  evolution-path test runs from tests/states/grind_lv15_route116.state — a REAL-PLAY
  state produced by the 2026-08-20 pilot (TRAINER_JOSH_BATTLE state ground on Route
  116 through the block's own battle/watch machinery to lv 15, exp just under the
  lv-16 threshold) — and is skipped when that state is absent. The evolution cutscene
  (Mudkip -> Marshtomp, A-only advance, no B) was pilot-verified live before the
  fixture was frozen; nothing here is synthesised.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from collection.direct_runner import DirectEmulatorRunner
from collection.playthrough.blocks.base import run_nav_block
from collection.playthrough.blocks.grind_evolve import GrindEvolve
from collection.playthrough.blocks.idle import Idle
from collection.playthrough.blocks.menus import Menus

_STORYLINE_ROOTS = [Path("data/storyline_wm"),
                    Path("../pokemon-worldmodel/data/storyline_wm"),
                    Path("/root/data0/proj-minhyuk-2026/pokemon-worldmodel/data/storyline_wm")]
STORYLINE = next((p for p in _STORYLINE_ROOTS if p.is_dir()), None)

needs_storyline = pytest.mark.skipif(STORYLINE is None, reason="storyline_wm corpus not mounted")

EVOLVE_STATE = Path("tests/states/grind_lv15_route116.state")


def _state_path(milestone: str) -> str:
    p = STORYLINE / milestone / "attempt_000001" / "final.state"
    if not p.exists():
        pytest.skip(f"missing storyline state {p}")
    return str(p)


def _runner(state: str, recorder=None) -> DirectEmulatorRunner:
    r = DirectEmulatorRunner(load_state=state, recorder=recorder, savestate_every=0)
    r.initialize()
    return r


def _subseq(trace: list, want: list) -> bool:
    it = iter(trace)
    return all(any(x == w for x in it) for w in want)


# ---------------------------------------------------------------------------
# registry / schedule (no emulator)


def test_expedition_registry_and_schedule_v2b():
    from collection.playthrough.schedule import EXPEDITION_BLOCKS, build_expedition_schedule

    assert set(EXPEDITION_BLOCKS) >= {"interaction", "grind_evolve", "idle", "menus"}
    sched = build_expedition_schedule([
        {"after": "OLDALE_TOWN", "block": "interaction", "maps": ["0,10"], "frames": 50000},
        {"after": "OLDALE_TOWN", "block": "grind_evolve", "target_level": 9, "frames": 4000},
        {"after": "OLDALE_TOWN", "block": "idle", "frames": 2000, "seed": 2},
        {"after": "OLDALE_TOWN", "block": "menus", "script": ["party", "bag"]},
    ])
    blocks = sched["OLDALE_TOWN"]
    assert [b.name for b in blocks] == ["interaction", "grind_evolve", "idle", "menus"]
    # nav blocks carry the director dispatch marker (.run) and their §2 phase tags
    assert all(hasattr(b, "run") for b in blocks)
    assert [b.phase for b in blocks] == ["interaction", "grind", "idle", "menus"]
    assert blocks[0].maps == ["0,10"] and blocks[1].target_level == 9
    assert blocks[2].seed == 2 and blocks[3].script == ("party", "bag")


# ---------------------------------------------------------------------------
# acceptance 1: interaction on a town map — verbs counted, save completes,
# accounting closes, phase rows tagged (through the recorded standalone entry)


@needs_storyline
def test_interaction_town_verbs_save_and_phase(tmp_path):
    from collection.playthrough.expedition import run_expedition_block

    out = run_expedition_block(
        load_state=_state_path("OLDALE_TOWN"), out_dir=str(tmp_path),
        block="interaction", kwargs={"maps": ["0,10"], "frames": 90000, "seed": 5},
        visual_fps=1)

    assert out["ran"] is True
    v = out["verbs"]
    assert v["npc_talk"] >= 1 and v["sign_read"] >= 1 and v["dialog_cancel"] >= 1
    assert v["save_dialog"] == 1                     # the owner-added §10 verb, completed
    assert sum(1 for c in v.values() if c > 0) >= 3
    # summary accounting closes EXACTLY: every enumerated target is either counted
    # or explained in the skipped-with-reason list
    skips = Counter(s["verb"] for s in out["skipped"])
    en = out["enumerated"]["0,10"]
    assert v["npc_talk"] + skips["npc_talk"] == en["npcs"]
    assert v["sign_read"] + skips["sign_read"] == en["signs"]
    assert v["object_interact"] + skips["object_interact"] == en["bg_objects"]
    assert v["npc_retalk"] + skips["npc_retalk"] == v["npc_talk"]
    assert all(s["reason"] for s in out["skipped"])
    assert v["door_bounce"] == len(out["door_pairs"])
    assert out["returned"] is True

    # phase tag in the RECORDED action rows, and the boundary transitions logged
    rows = [json.loads(l) for l in open(tmp_path / "actions.jsonl")]
    assert all("buttons_held" in r for r in rows)    # M1: stream stays homogeneous
    tagged = [r for r in rows if r.get("block_phase") == "interaction"]
    assert len(tagged) > 1000                        # the block really ran under its tag
    trans = [json.loads(l)["phase"] for l in open(tmp_path / "phases.jsonl")]
    assert trans == ["interaction", "spine"]
    assert json.loads((tmp_path / "block_summary.json").read_text()) == out


# ---------------------------------------------------------------------------
# acceptance 2: menus — all three views confirmed via the ledger's RAM sources


@needs_storyline
def test_menus_views_ram_verified():
    r = _runner(_state_path("OLDALE_TOWN"))
    try:
        out = run_nav_block(r, Menus(frames=9000))
        assert out["ran"]
        assert out["party_menu_views"] >= 1
        assert out["summary_views"] >= 1
        assert out["bag_views"] >= 1
        # ui_state transitions observed through the verified RAM sources:
        # start_menu(2) -> party(4) -> summary(5) -> ... -> bag(3), back out to 0
        assert _subseq(out["ui_trace"], [2, 4, 5, 3])
        assert out["ui_trace"][-1] == 0              # overworld ui really observed
        assert out["skipped"] == []
        assert out["returned"] is True
        n = r.nav_state()
        assert n.in_battle is False and n.control_mode == "free_overworld"
    finally:
        r.close()


# ---------------------------------------------------------------------------
# acceptance 3: idle — dwell schedule runs, frames counted, control returned


@needs_storyline
def test_idle_dwell_schedule_and_return():
    r = _runner(_state_path("OLDALE_TOWN"))
    try:
        out = run_nav_block(r, Idle(frames=2500, seed=1))
        assert out["ran"]
        assert out["relocations"] >= 2
        assert out["dwell_frames"] > 0
        assert out["dwell_frames"] < out["frames"]   # dwells + relocations + turns
        assert out["frames"] >= 2500                 # the budget really elapsed
        assert out["frames"] <= out["frames_total"]
        assert out["returned"] is True
        n = r.nav_state()
        assert n.in_battle is False and n.control_mode == "free_overworld"
    finally:
        r.close()


# ---------------------------------------------------------------------------
# acceptance 4: grind_evolve — battle-win path from a grassy storyline state


@needs_storyline
def test_grind_evolve_battle_win_path():
    r = _runner(_state_path("BACK_TO_ROUTE101_FROM_OLDALE"))
    try:
        out = run_nav_block(r, GrindEvolve(target_level=9, frames=70000, seed=3))
        assert out["ran"]
        assert out["battles_won"] >= 1               # stochastic: >= 1, never exact
        assert out["battles"] >= out["battles_won"]
        assert out["levels_gained"] >= 0
        assert out["final_level"] >= 8               # the lv-8 lead never loses a level
        assert out["ended"] in ("target_level", "budget", "lead_hp_low")
        assert out["skipped"] == []
        assert r.nav_state().in_battle is False
    finally:
        r.close()


# evolution completion: REAL-PLAY fixture from the pilot (see module docstring);
# never synthesised — skipped when the pilot state isn't present.


@pytest.mark.skipif(not EVOLVE_STATE.exists(),
                    reason="pilot-played lv-15 state not present")
def test_grind_evolve_completes_evolution():
    r = _runner(str(EVOLVE_STATE))
    try:
        out = run_nav_block(r, GrindEvolve(target_level=16, frames=120000, seed=7))
        assert out["ran"]
        assert out["battles_won"] >= 1
        assert out["evolved"] is True                # Mudkip -> Marshtomp, cutscene kept
        assert out["final_level"] >= 16
        assert out["ended"] == "target_level"
        assert r.nav_state().in_battle is False      # scene completed, control returned
    finally:
        r.close()
