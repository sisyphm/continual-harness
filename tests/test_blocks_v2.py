"""W33 corpus-v2 §2 expedition blocks: bfs_sweep + encounter_farm. Real emulator,
real recorder, storyline final.states (skipped cleanly when the storyline corpus
isn't mounted). Battles are stochastic — encounter assertions are >= 1 within a
generous budget, never exact counts.

Measured ground truth these tests pin (2026-08-20 build session):
- Brendan-house 1F (map 1,0): 59 targetable walkable tiles (62 minus 3 warp tiles);
  the sweep covered 58 with the 59th (mom's parked tile) reported in
  denied_tiles_per_map — accounting closes exactly.
- Route 101 (0,16): grass reached in <100 frames from the north entry; wild battles
  every ~0.5-1.5k paced frames; gBattleMons[1] already holds the CURRENT encounter by
  the frame the battle flag reads set (species always in the map's manifest land table).
- Route 101 <-> Oldale (0,10): both directional connection crossings in ~240 frames.
"""

import json
from pathlib import Path

import pytest

from collection.direct_runner import DirectEmulatorRunner
from collection.playthrough.blocks.base import run_nav_block
from collection.playthrough.blocks.bfs_sweep import BfsSweep
from collection.playthrough.blocks.encounter_farm import EncounterFarm

_STORYLINE_ROOTS = [Path("data/storyline_wm"),
                    Path("../pokemon-worldmodel/data/storyline_wm"),
                    Path("/root/data0/proj-minhyuk-2026/pokemon-worldmodel/data/storyline_wm")]
STORYLINE = next((p for p in _STORYLINE_ROOTS if p.is_dir()), None)

needs_storyline = pytest.mark.skipif(STORYLINE is None, reason="storyline_wm corpus not mounted")


def _state_path(milestone: str) -> str:
    p = STORYLINE / milestone / "attempt_000001" / "final.state"
    if not p.exists():
        pytest.skip(f"missing storyline state {p}")
    return str(p)


def _runner(state: str, recorder=None) -> DirectEmulatorRunner:
    r = DirectEmulatorRunner(load_state=state, recorder=recorder, savestate_every=0)
    r.initialize()
    return r


# ---------------------------------------------------------------------------
# registry / schedule (no emulator)


def test_expedition_registry_and_schedule():
    from collection.playthrough.schedule import EXPEDITION_BLOCKS, build_expedition_schedule

    assert set(EXPEDITION_BLOCKS) >= {"bfs_sweep", "encounter_farm"}
    sched = build_expedition_schedule([
        {"after": "OLDALE_TOWN", "block": "bfs_sweep", "maps": ["0,10"],
         "legs": [["0,10", "0,16"]], "per_map_frames": 1000},
        {"after": "OLDALE_TOWN", "block": "encounter_farm", "frames": 500,
         "targets": [{"map": "0,16", "area": "land", "species_targets": {"288": 1}}]},
    ])
    blocks = sched["OLDALE_TOWN"]
    assert [b.name for b in blocks] == ["bfs_sweep", "encounter_farm"]
    # nav blocks carry the director dispatch marker (.run) and a recording phase
    assert all(hasattr(b, "run") for b in blocks)
    assert [b.phase for b in blocks] == ["bfs_sweep", "encounter"]
    assert blocks[0].per_map_frames == 1000 and blocks[0].legs == [("0,10", "0,16")]


# ---------------------------------------------------------------------------
# acceptance 1: sweep coverage + phase tag + anchor return (small indoor map)


@needs_storyline
def test_bfs_sweep_indoor_coverage_phase_and_anchor(tmp_path):
    """Runs through the standalone job entry (the deficit-filling path), so this also
    pins recorder integration: phase rows stamped, summary persisted."""
    from collection.playthrough.expedition import run_expedition_block

    out = run_expedition_block(
        load_state=_state_path("GO_DOWNSTAIRS_TO_1F"), out_dir=str(tmp_path),
        block="bfs_sweep", kwargs={"maps": ["1,0"], "per_map_frames": 40000},
        visual_fps=1)

    assert out["ran"] is True
    assert out["anchor"] == ["1,0", 7, 3]            # the block's start tile
    assert out["returned"] is True                   # anchor-return left us there
    visited = out["tiles_visited_per_map"]["1,0"]
    walkable = out["walkable_tiles_per_map"]["1,0"]
    denied = len(out["denied_tiles_per_map"].get("1,0", []))
    assert out["budget_expired_maps"] == []
    assert visited / walkable >= 0.8, f"coverage {visited}/{walkable}"
    # summary accounting closes against the visited set: every targetable walkable
    # tile is either visited or explicitly reported as NPC-denied
    assert visited <= walkable and visited + denied >= walkable
    assert 0 < out["frames"] <= out["frames_total"]

    # phase tag in the RECORDED action rows, and the boundary transitions logged
    rows = [json.loads(l) for l in open(tmp_path / "actions.jsonl")]
    assert all("buttons_held" in r for r in rows)    # M1: stream stays homogeneous
    swept = [r for r in rows if r.get("block_phase") == "bfs_sweep"]
    assert len(swept) > 1000                         # the sweep really ran under its tag
    trans = [json.loads(l)["phase"] for l in open(tmp_path / "phases.jsonl")]
    assert trans == ["bfs_sweep", "spine"]           # stamped at start, restored at end
    # persisted job summary matches the returned outcome
    assert json.loads((tmp_path / "block_summary.json").read_text()) == out


# ---------------------------------------------------------------------------
# acceptance 2: direction-balance leg (both directions recorded)


@needs_storyline
def test_bfs_sweep_direction_balance_leg():
    r = _runner(_state_path("BACK_TO_ROUTE101_FROM_OLDALE"))
    try:
        out = run_nav_block(r, BfsSweep(maps=[], legs=[("0,16", "0,10")], leg_frames=15000))
        assert out["ran"]
        assert out["connections_crossed"] == [["0,16", "0,10"], ["0,10", "0,16"]]
        assert out["returned"] is True
        n = r.nav_state()
        assert (n.x, n.y) == tuple(out["anchor"][1:]) and not n.in_battle
    finally:
        r.close()


# ---------------------------------------------------------------------------
# acceptance 3: encounter_farm — grass, >= 1 wild battle, species read, flee


@needs_storyline
def test_encounter_farm_grass_battle_species_flee():
    from collection.playthrough.blocks.encounter_farm import mk_wild
    from collection.navigator import MapKnowledge

    r = _runner(_state_path("BACK_TO_ROUTE101_FROM_OLDALE"))
    try:
        out = run_nav_block(r, EncounterFarm(frames=80000, seed=3, targets=[
            {"map": "0,16", "area": "land", "species_targets": {286: 1}},
            {"map": "0,16", "area": "water", "species_targets": {}},
        ]))
        assert out["ran"]
        assert out["battles"] >= 1                   # stochastic: >= 1, never exact
        assert out["sightings"], "no species recorded"
        table = {m[0] for m in mk_wild(MapKnowledge(), "0,16")["land"]["mons"]}
        for sid, n in out["sightings"].items():
            assert int(sid) > 0 and n >= 1
            assert int(sid) in table, f"foe {sid} not in Route 101's land table {table}"
        # fled back to the overworld and anchor-returned
        n = r.nav_state()
        assert n.in_battle is False
        assert out["returned"] is True
        # the water target was skipped WITH a documented reason, never silently
        assert [s["area"] for s in out["skipped"]] == ["water"]
        assert "Surf" in out["skipped"][0]["reason"]
    finally:
        r.close()


# ---------------------------------------------------------------------------
# adversarial-review fixes: unit proofs (no emulator)


def test_bfs_sweep_dedupes_maps_and_reports_failed_connections(monkeypatch):
    """Sharp minor 2: duplicate map keys deduped (order kept); a _cross failure is
    REPORTED in connections_failed, never silently dropped."""
    b = BfsSweep(maps=["1,0", "0,10", "1,0"], legs=[("0,10", "0,16")])
    assert b.maps == ["1,0", "0,10"]

    class MK:                                        # manifest with NO hop for the pair
        connections: dict = {}
        warps: dict = {}

    monkeypatch.setattr(BfsSweep, "_ensure_map",
                        lambda self, runner, mk, key, summary, budget: True)
    summary = dict(connections_crossed=[], connections_failed=[], battles_fled=0,
                   unreached_maps=[])
    assert b._cross(None, MK(), "0,10", "0,16", summary) is False
    assert summary["connections_failed"] == [["0,10", "0,16", "no_direct_hop"]]
    assert summary["connections_crossed"] == []


def test_read_foe_species_unstable_returns_none(monkeypatch):
    """Sharp minor 3: a species that never passes the 3-poll stability gate must NOT
    be counted — pre-fix the last flapping read was returned as a sighting."""
    import collection.playthrough.blocks.encounter_farm as ef

    class St:
        def u8(self, addr):
            return 0b10                              # gMain inBattle bit set

    class FakeGBA:
        @staticmethod
        def from_env(env):
            return St()

    seq = iter([5, 6, 5, 6, 5, 6, 5, 6])             # flapping species, never 3-stable
    monkeypatch.setattr(ef, "GBAState", FakeGBA)
    monkeypatch.setattr(ef, "_battle_mon", lambda st, i: {
        "species": next(seq, 6), "hp": 10, "max_hp": 10, "level": 5})
    monkeypatch.setattr(ef.nav, "_hold", lambda runner, buttons, n, phase: None)

    class R:
        env = None

    assert ef.read_foe_species(R(), min_wait=0, cap=40) is None


@needs_storyline
def test_run_nav_block_contains_block_errors(monkeypatch):
    """Sharp minor 6: a crashing block must never break the spine — the error is
    recorded in the summary, the anchor return still runs, nothing propagates; and a
    second failure in the anchor return does not mask the first."""
    import collection.playthrough.blocks.base as base

    class Boom:
        name = "boom"
        phase = "boom"

        def run(self, runner, mk, ctx):
            raise RuntimeError("block exploded")

    r = _runner(_state_path("BACK_TO_ROUTE101_FROM_OLDALE"))
    try:
        out = run_nav_block(r, Boom())
        assert out["ran"] is True
        assert out["error"] == "RuntimeError('block exploded')"
        assert out["returned"] is True               # never moved; anchor return trivially ok
        n = r.nav_state()
        assert (n.x, n.y) == tuple(out["anchor"][1:])

        def raise_return(*a, **k):
            raise RuntimeError("return failed")

        monkeypatch.setattr(base, "_return_to_anchor_nav", raise_return)
        out2 = run_nav_block(r, Boom())
        assert out2["error"] == "RuntimeError('block exploded')"
        assert out2["return_error"] == "RuntimeError('return failed')"
        assert out2["returned"] is False
    finally:
        r.close()
