"""W33 corpus-v2 item 4 — the last build item before the pilot: change-matrix probe
(plan_change_matrix), rotation scheduler (plan_expeditions), v2 audit extension
(audits/coverage v2 axes + density), derivation kit (derive). Real emulator, real
storyline stage states, one real recorded multi-block expedition-ish run.

Measured ground truth these tests pin (2026-08-21 build session):
- Change matrix over all 51 storyline stage states (~3 s): Littleroot ("0,9")
  changes at LITTLEROOT_TOWN/PLAYER_HOUSE_ENTERED/RECEIVED_POKEDEX/
  ROUTE101_AFTER_POKEDEX; late routes "0,30"/"0,31" (115/116) have NO post-init
  changes. Object-count parity ROM templates vs manifest on all 46 scope maps.
- Fixture run (BACK_TO_ROUTE101_FROM_OLDALE + 5 blocks, sink attached,
  savestate_every=200, ~29.5k frames, ~145 s): legs crossed both directions in
  ~250 f; Route-101 tour 67 tiles/3.8k f (16.4 tiles/1k); menus 3 views/1.3k f;
  encounter_farm 7 battles/8k f with in-table sightings; idle 823 dwell-f/1k.
  The tour's anchor return may legitimately FAIL (wild battles consume the
  bounded retry rounds — pilot punch-list item); the job runner's between-block
  recovery (flee + vision-gated B-close) keeps later blocks running.
- Derive probe on that run: 155/155 gaps bit-verified (six-block RAM digest,
  leading gap from the original load_state + settle frame included), 29,557
  ticks field-equal to the run's own recorded ledger across all 28 fields.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from collection.catalog import MILESTONE_ORDER

_STORYLINE_ROOTS = [Path("data/storyline_wm"),
                    Path("../pokemon-worldmodel/data/storyline_wm"),
                    Path("/root/data0/proj-minhyuk-2026/pokemon-worldmodel/data/storyline_wm")]
STORYLINE = next((p for p in _STORYLINE_ROOTS if p.is_dir()), None)
DATA_ROOT = STORYLINE.parent if STORYLINE else None

needs_storyline = pytest.mark.skipif(
    STORYLINE is None or DATA_ROOT is None
    or not (DATA_ROOT / "processed/coverage_manifest.json").exists(),
    reason="storyline_wm corpus / coverage manifest not mounted")

_midx = MILESTONE_ORDER.index


# ---------------------------------------------------------------------------
# shared fixtures: the change matrix (built once from the REAL stage states) and
# one REAL recorded multi-block run (built once, consumed by audits + derive)


@pytest.fixture(scope="module")
def change_matrix(tmp_path_factory):
    from collection.plan_change_matrix import build_change_matrix

    m = build_change_matrix(data_root=DATA_ROOT)
    out = tmp_path_factory.mktemp("matrix") / "w33_change_matrix.json"
    out.write_text(json.dumps(m, indent=1, sort_keys=True))
    return m, out


@pytest.fixture(scope="module")
def v2_run(tmp_path_factory):
    from collection.playthrough.expedition import run_expedition_block

    out = tmp_path_factory.mktemp("v2run")
    summaries = run_expedition_block(
        load_state=str(STORYLINE / "BACK_TO_ROUTE101_FROM_OLDALE"
                       / "attempt_000001" / "final.state"),
        out_dir=str(out),
        blocks=[
            {"block": "bfs_sweep", "kwargs": {"maps": [], "legs": [["0,16", "0,10"]]}},
            {"block": "bfs_sweep", "kwargs": {"maps": ["0,16"], "per_map_frames": 3000}},
            {"block": "menus", "kwargs": {"frames": 7000}},
            {"block": "encounter_farm", "kwargs": {
                "frames": 8000, "seed": 3,
                "targets": [{"map": "0,16", "area": "land", "species_targets": {}}]}},
            {"block": "idle", "kwargs": {"frames": 2000, "seed": 1}},
        ],
        visual_fps=1, record_wm=True, savestate_every=200, return_budget=6000)
    return out, summaries


# ---------------------------------------------------------------------------
# acceptance 1: change matrix over real stage states -> plausible rows, file written


@needs_storyline
def test_change_matrix_real_stages(change_matrix):
    m, out_path = change_matrix
    assert len(m["stages"]) >= 5                     # ran over >= 5 real stage states
    assert out_path.exists()
    reloaded = json.loads(out_path.read_text())
    assert reloaded["matrix"] == m["matrix"]

    stages = m["stages"]
    init = set(stages[:2])                           # pre-game + world-init baseline
    post = {k: [s for s in v if s not in init] for k, v in m["matrix"].items()}

    # LITTLEROOT ("0,9") changes at EARLY stages (mom/rival/Birch actors)
    lr = post.get("0,9", [])
    assert len(lr) >= 2, f"Littleroot should change early, got {lr}"
    assert all(_midx(s) < _midx("PETALBURG_CITY") for s in lr)
    # a LATE route is STABLE post-init (Route 116 / Route 115)
    assert post.get("0,31") == []
    assert post.get("0,30") == []

    # every in-scope map fingerprinted at every stage, both hash parts present
    for k, per_stage in m["fingerprints"].items():
        assert set(per_stage) == set(stages)
        assert all(len(v["static"]) == 64 for v in per_stage.values())
    # the loaded-map live hash exists exactly where the player stood
    for s in stages:
        pm = m["player_map"].get(s)
        if pm in m["fingerprints"]:
            assert m["fingerprints"][pm][s]["live"] is not None


# ---------------------------------------------------------------------------
# acceptance 2: scheduler — N=12 plan, reduced floors, green / balanced /
# holdouts / deterministic / constructible blocks


TEST_FLOORS = dict(anchor_stages_per_map=1, connection_rounds=1, species_sightings=1,
                   grind_runs_per_starter=1, mart_runs_per_town=1, pc_runs_per_center=1,
                   item_use_runs=1, idle_run_share=6, menus_run_share=6)


@needs_storyline
def test_scheduler_n12_green_balanced_deterministic(change_matrix):
    from collection.plan_expeditions import build_plan, check_plan, plan_run_to_expedition_entries
    from collection.playthrough.schedule import EXPEDITION_BLOCKS, build_expedition_schedule

    m, _ = change_matrix
    manifest = json.loads((DATA_ROOT / "processed/coverage_manifest.json").read_text())

    plan = build_plan(n_runs=12, seed=7, manifest=manifest, matrix=m,
                      floors=TEST_FLOORS)
    # deterministic: same seed -> byte-identical plan; different seed -> different
    again = build_plan(n_runs=12, seed=7, manifest=manifest, matrix=m,
                       floors=TEST_FLOORS)
    assert json.dumps(plan, sort_keys=True) == json.dumps(again, sort_keys=True)
    other = build_plan(n_runs=12, seed=8, manifest=manifest, matrix=m,
                       floors=TEST_FLOORS)
    assert json.dumps(plan, sort_keys=True) != json.dumps(other, sort_keys=True)

    # --check green: coverage recomputed FROM the plan closes every requirement row
    rep = check_plan(plan, manifest=manifest, matrix=m)
    assert rep["green"], f"plan self-audit red: {rep['deficits'][:5]}"
    assert all(rs for rs in rep["rows"].values())    # every axis produced rows

    # balanced starters (12 -> 4/4/4) and 3 holdouts, one per starter (§10.1)
    counts = Counter(r["starter"] for r in plan["runs"])
    assert set(counts.values()) == {4}
    hold = [r for r in plan["runs"] if r["holdout"]]
    assert len(hold) == 3
    assert sorted(r["starter"] for r in hold) == ["mudkip", "torchic", "treecko"]

    # every scheduled block really constructs (names registered, kwargs valid) and
    # the plan converts into director expedition entries
    for r in plan["runs"]:
        assert r["block_schedule"], f"{r['run_id']} got no blocks"
        for idx, name, args in r["block_schedule"]:
            assert 0 <= idx < len(MILESTONE_ORDER)
            EXPEDITION_BLOCKS[name](**args)          # raises on a bad kwarg
        sched = build_expedition_schedule(plan_run_to_expedition_entries(r))
        assert sum(len(v) for v in sched.values()) == len(r["block_schedule"])


# ---------------------------------------------------------------------------
# acceptance 3: extended audit against ONE real recorded v2 run — the new axes
# produce rows with CORRECT support counting (ok may be False at tiny scale)


@needs_storyline
def test_v2_audit_axes_on_real_run(change_matrix, v2_run):
    from collection.audits.coverage import audit_v2, v2_phase_frames
    from collection.plan_change_matrix import window_of

    m, _ = change_matrix
    run_dir, summaries = v2_run
    by_block: dict[str, list] = {}
    for s in summaries:
        by_block.setdefault(s["block"], []).append(s)

    rep = audit_v2([str(run_dir)], change_matrix=m)
    rows = rep["rows"]
    for ax in ("interaction_verbs", "mart_pc_item", "sweep_cells",
               "sweep_windows_per_map", "connection_pairs", "outcomes_v2",
               "level_hist", "battle_situation"):
        assert rows[ax], f"axis {ax} produced no rows"

    # interaction verbs: a row per verb even at zero support
    assert {r["key"] for r in rows["interaction_verbs"]} >= {
        "npc_talk", "sign_read", "save_dialog", "door_bounce"}

    # sweep cell: the Route-101 tour lands in the window its stage belongs to,
    # with support == the block's own tiles_visited (correct support counting)
    tours = [s for s in by_block["bfs_sweep"] if s.get("tiles_visited_per_map")]
    assert tours, "no tour summary recorded"
    tiles = sum(s["tiles_visited_per_map"].get("0,16", 0) for s in tours)
    win = window_of(m, "0,16", "BACK_TO_ROUTE101_FROM_OLDALE")
    cell = next(r for r in rows["sweep_cells"] if r["key"] == f"0,16@{win}")
    assert cell["support"] == tiles > 0

    # direction balance: one leg round trip -> pair support 1 (min of directions)
    legs = [s for s in by_block["bfs_sweep"] if s.get("connections_crossed")]
    if legs:                                          # leg success is nav-dependent
        pair = next(r for r in rows["connection_pairs"] if r["key"] == "0,10<->0,16")
        assert pair["support"] == 1

    # ledger axes: lead level histogram counted from the per-tick ledger
    lv_rows = {r["key"]: r["support"] for r in rows["level_hist"]}
    assert lv_rows.get("lv8", 0) > 1000               # lv-8 Mudkip lead, most frames
    assert lv_rows["level_spread"] >= 1
    # whiteout/evolution rows EXIST (support 0 at this scale is correct counting)
    oc = {r["key"]: r for r in rows["outcomes_v2"]}
    assert oc["whiteout"]["support"] == 0 and oc["whiteout"]["ok"] is False
    assert oc["evolution"]["support"] == 0

    # battle-situation: measurable rows + the documented §11-deferred rows
    bs = {r["key"]: r for r in rows["battle_situation"]}
    assert "low_hp_bar" in bs and "battle_level_up" in bs
    for k in ("status_psn", "status_slp", "status_par", "crit"):
        assert "§11" in bs[k]["deferred"]

    # density: every executed phase produced a row; events tie back to summaries
    dens = {r["phase"]: r for r in rep["density"]}
    assert set(dens) >= {"bfs_sweep", "menus", "encounter", "idle"}
    enc = by_block["encounter_farm"][0]
    assert enc["battles"] >= 1                        # stochastic: >= 1, never exact
    assert dens["encounter"]["events"] == enc["battles"]
    assert dens["bfs_sweep"]["events"] == tiles
    for r in dens.values():
        assert r["per_1k"] == pytest.approx(
            r["events"] * 1000 / r["block_frames"], abs=0.01)
    # measured-density floors hold on the real run (provenance of DENSITY_FLOORS)
    assert dens["bfs_sweep"]["ok"] and dens["idle"]["ok"]

    # phases.jsonl spans cover every executed phase
    spans = v2_phase_frames(run_dir)
    assert set(spans) >= {"bfs_sweep", "menus", "encounter", "idle"}
    assert all(v > 0 for v in spans.values())


# ---------------------------------------------------------------------------
# acceptance 4: derivation kit — per-gap replay fidelity PASSES and the derived
# ledger EQUALS the run's own recorded chunks (the §11 contract, end-to-end)


@needs_storyline
def test_derive_probe_fidelity_and_ledger_equality(v2_run):
    from collection.derive import ReplayGapError, probe

    run_dir, _ = v2_run
    s = probe(str(run_dir))                          # raises on ANY gap/field failure
    assert s["gaps_verified"] >= 10                  # savestate_every=200 -> many gaps
    assert s["gaps_unverified"] == []                # leading gap replayed too
    assert s["ticks_compared"] > 1000
    assert s["field_mismatches"] == 0
    for f in ("species", "level", "money"):          # the acceptance-named fields
        assert f in s["fields"]

    # a non-v2 dir (no savestates) is a loud error, never a silent no-op
    with pytest.raises(ReplayGapError):
        probe(str(Path(run_dir).parent))


@needs_storyline
def test_derive_field_custom_reader(v2_run):
    """derive_field with a NEW field (the §11 point): a one-liner RAM reader is
    enough — here the ledger's money read as a standalone derived array."""
    import numpy as np

    from collection.derive import derive_field, load_run_ledger

    run_dir, _ = v2_run

    def money_reader(env):
        from collection.extractors.ledger_panel import SB1_MONEY, SB1_PTR, SB2_KEY, SB2_PTR
        from collection.extractors.ram import GBAState
        st = GBAState.from_env(env)
        sb1, sb2 = st.u32(SB1_PTR), st.u32(SB2_PTR)
        return (st.u32(sb1 + SB1_MONEY) ^ st.u32(sb2 + SB2_KEY)) & 0xFFFFFFFF

    r = derive_field(str(run_dir), money_reader, max_gaps=3)
    assert sum(1 for g in r["gaps"] if g.get("verified")) == 3
    assert len(r["value"]) == len(r["frame_idx"]) > 0
    led = load_run_ledger(run_dir)
    by_frame = {int(f): i for i, f in enumerate(led["frame_idx"])}
    checked = 0
    for j, f in enumerate(r["frame_idx"]):
        i = by_frame.get(int(f))
        if i is not None:
            assert int(r["value"][j]) == int(led["money"][i])
            checked += 1
    assert checked > 100
    assert np.asarray(r["value"]).dtype.kind in "iu"
