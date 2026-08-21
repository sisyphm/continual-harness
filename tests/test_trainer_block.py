"""Task #46 acceptance: trainer-engagement coverage for the W33 fleet — the
trainer_engagement block (real emulator), the plan-level trainer scheduling, and
the per-trainer audit axis. Owner-measured evidence behind the item: scripted
policies bypass optional trainers (treecko fought 11/18, torchic 6/18 tracked),
so per-trainer coverage must be SCHEDULED, not lucky.

Measured ground truth these tests pin (2026-08-21 build session, live emulator):
- Flag mapping flag = 0x500 + script trainer id (the manifest's `trainerbattle`
  0x5C parse) verified by full 300-byte flags-slice diffs across one engagement
  each: CALVIN 318 -> 0x63E (ROUTE_102 state), JAMES 621 -> 0x76D
  (TEAM_AQUA_GRUNT_DEFEATED state), WINSTON 136 -> 0x588 (ROUTE_104_NORTH state;
  the ONLY flip in the whole slice), GRUNT 10 -> 0x50A (scripted woods battle).
  Lost battles (Karen 280, Tommy 321 vs the lv-12 Mudkip lead) flip NOTHING —
  the flag records victory, not talk. Details in blocks/trainer_engagement.py.
- Calvin standalone: engaged+won in ~3.6k block frames; Winston in ~2.8k.
- Rustboro Gym lesson: the beaten JOSH parks on the entrance pocket's only exit
  tile; nav.goto's unstick A-mash re-talks him and wedges — the block's denial
  walker (_walk_to) routes around instead.
"""

import json
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

CALVIN_FLAG = 0x500 + 318                            # Route 102, map "0,17"


def _state_path(milestone: str) -> str:
    p = STORYLINE / milestone / "attempt_000001" / "final.state"
    if not p.exists():
        pytest.skip(f"missing storyline state {p}")
    return str(p)


# ---------------------------------------------------------------------------
# registry / constants (no emulator)


def test_registry_flags_and_schedule():
    from collection.playthrough.blocks.trainer_engagement import (
        SPINE_TRAINER_FLAGS, TRACKED_TRAINER_FLAGS, TRAINER_FLAG_BASE)
    from collection.playthrough.schedule import EXPEDITION_BLOCKS, build_expedition_schedule

    assert "trainer_engagement" in EXPEDITION_BLOCKS
    assert len(TRACKED_TRAINER_FLAGS) == 18
    assert len(set(TRACKED_TRAINER_FLAGS)) == 18
    assert all(f > TRAINER_FLAG_BASE for f in TRACKED_TRAINER_FLAGS)
    # the three spine-guaranteed trainers are tracked AND marked
    assert set(SPINE_TRAINER_FLAGS) <= set(TRACKED_TRAINER_FLAGS)
    assert sorted(SPINE_TRAINER_FLAGS.values()) == [
        "ROXANNE_BATTLE", "TEAM_AQUA_GRUNT_DEFEATED", "TRAINER_JOSH_BATTLE"]

    sched = build_expedition_schedule([
        {"after": "ROUTE_102", "block": "trainer_engagement",
         "targets": [{"map": "0,17", "trainer_flag": CALVIN_FLAG}], "frames": 1000}])
    (blk,) = sched["ROUTE_102"]
    assert blk.name == "trainer_engagement" and blk.phase == "battle"
    assert hasattr(blk, "run")                       # director nav-block dispatch marker
    assert blk.targets == [{"map": "0,17", "trainer_flag": CALVIN_FLAG}]


# ---------------------------------------------------------------------------
# acceptance 2: standalone block from a storyline state — engages, wins, flag
# verified flipped, summary exact (real emulator, recorded run + ledger)


@pytest.fixture(scope="module")
def calvin_run(tmp_path_factory):
    from collection.playthrough.expedition import run_expedition_block

    out = tmp_path_factory.mktemp("trainer_run")
    summary = run_expedition_block(
        load_state=_state_path("ROUTE_102"), out_dir=str(out),
        block="trainer_engagement",
        kwargs={"targets": [{"map": "0,17", "trainer_flag": CALVIN_FLAG}],
                "frames": 60000, "seed": 1},
        visual_fps=1, record_wm=True, savestate_every=2000)
    return out, summary


@needs_storyline
def test_trainer_engagement_calvin_engages_wins_flag_flips(calvin_run):
    run_dir, out = calvin_run
    assert out["ran"] is True
    # summary EXACT: one engagement, won, nothing skipped
    assert out["engaged"] == [{"flag": CALVIN_FLAG, "won": True}]
    assert out["skipped_with_reason"] == []
    assert out["battles"] >= 1
    assert out["ended"] == "done"
    assert 0 < out["frames"] <= out["frames_total"]

    # the flag flip is IN THE RECORDING: the run's final valid ledger frame carries
    # flag N at byte N//8 bit N%8 (the audit's exact decode path)
    import numpy as np
    from collection.derive import load_run_ledger
    led = load_run_ledger(run_dir)
    assert led is not None
    vi = np.flatnonzero(led["valid"] > 0)
    first, last = led["flags"][vi[0]], led["flags"][vi[-1]]
    assert not (int(first[CALVIN_FLAG // 8]) >> (CALVIN_FLAG % 8)) & 1  # unfought at load
    assert (int(last[CALVIN_FLAG // 8]) >> (CALVIN_FLAG % 8)) & 1      # defeated at end

    # phase "battle" stamped into the recorded streams and restored to spine
    rows = [json.loads(l) for l in open(run_dir / "actions.jsonl")]
    assert sum(1 for r in rows if r.get("block_phase") == "battle") > 100
    trans = [json.loads(l)["phase"] for l in open(run_dir / "phases.jsonl")]
    assert trans == ["battle", "spine"]
    # persisted job summary matches the returned outcome
    assert json.loads((run_dir / "block_summary.json").read_text()) == out


@needs_storyline
def test_audit_trainer_axis_on_real_run(calvin_run):
    from collection.audits.coverage import audit_v2
    from collection.playthrough.blocks.trainer_engagement import (
        SPINE_TRAINER_FLAGS, TRACKED_TRAINER_FLAGS)

    run_dir, _ = calvin_run
    rep = audit_v2([str(run_dir)])
    rows = rep["rows"]["trainer_flags"]
    assert len(rows) == 18                           # a row per tracked trainer,
    by_key = {r["key"]: r for r in rows}             # present even at zero support
    assert set(by_key) == {f"tr_flag_{f:#06x}" for f in TRACKED_TRAINER_FLAGS}
    calvin = by_key[f"tr_flag_{CALVIN_FLAG:#06x}"]
    assert calvin["support"] == 1                    # this ONE run set the flag
    assert calvin["ok"] is False                     # floor is 2 RUNS — correct at scale
    # every other OPTIONAL trainer reads zero support from this run
    for f in TRACKED_TRAINER_FLAGS:
        if f != CALVIN_FLAG and f not in SPINE_TRAINER_FLAGS:
            assert by_key[f"tr_flag_{f:#06x}"]["support"] == 0
    assert rep["summary"]["trainer_flags"]["rows"] == 18


# ---------------------------------------------------------------------------
# acceptance 3: scheduler — trainer rows green, deterministic, boundary-eligible


TEST_FLOORS = dict(anchor_stages_per_map=1, connection_rounds=1, species_sightings=1,
                   grind_runs_per_starter=1, mart_runs_per_town=1, pc_runs_per_center=1,
                   item_use_runs=1, idle_run_share=6, menus_run_share=6,
                   trainer_runs_per_trainer=2)


@needs_storyline
def test_scheduler_trainer_rows_and_boundaries():
    from collection.plan_expeditions import (
        INELIGIBLE_BLOCK_BOUNDARIES, SPINE_TRAINER_FLAGS, TRACKED_TRAINER_FLAGS,
        build_plan, check_plan)
    from collection.playthrough.schedule import EXPEDITION_BLOCKS

    manifest = json.loads((DATA_ROOT / "processed/coverage_manifest.json").read_text())
    matrix = json.loads((DATA_ROOT / "processed/w33_change_matrix.json").read_text())

    plan = build_plan(n_runs=12, seed=7, manifest=manifest, matrix=matrix,
                      floors=TEST_FLOORS)
    again = build_plan(n_runs=12, seed=7, manifest=manifest, matrix=matrix,
                       floors=TEST_FLOORS)
    assert json.dumps(plan, sort_keys=True) == json.dumps(again, sort_keys=True)

    rep = check_plan(plan, manifest=manifest, matrix=matrix)
    assert rep["green"], f"plan self-audit red: {rep['deficits'][:5]}"

    # 18 trainer rows, all green; the spine three annotated with their milestone
    trows = rep["rows"]["trainers"]
    assert len(trows) == 18 and all(r["ok"] for r in trows)
    assert sum(1 for r in trows if "spine" in r) == 3

    # every OPTIONAL tracked trainer is assigned to >= 2 DISTINCT non-holdout runs
    per_flag: dict[int, set] = {}
    for r in plan["runs"]:
        for idx, name, args in r["block_schedule"]:
            if name != "trainer_engagement":
                continue
            # Coverage trainer blocks stay off holdout runs. The exception is
            # torchic's Route 116 sweep at RUSTBORO_CENTER_EXITED, which is not
            # coverage at all: 116's grass is sparse enough that the walker steps
            # straight back out of it, so its trainers are how a torchic run reaches
            # L16 and evolves. A holdout run still has to CLEAR the game.
            _progression = (r["starter"] == "torchic"
                            and all(t["map"] == "0,31" for t in args["targets"]))
            assert not r["holdout"] or _progression, \
                "coverage trainer blocks must ride non-holdout runs"
            if r["holdout"]:
                continue          # holdout runs never count toward coverage tallies
            for t in args["targets"]:
                per_flag.setdefault(t["trainer_flag"], set()).add(r["run_id"])
    optional = [f for f in TRACKED_TRAINER_FLAGS if f not in SPINE_TRAINER_FLAGS]
    assert len(optional) == 15
    for f in optional:
        assert len(per_flag.get(f, set())) >= 2, f"flag {f:#x} under-assigned"
    # dodge-variance preserved: no single run was forced to fight all 18
    per_run: dict[str, set] = {}
    for f, rids in per_flag.items():
        for rid in rids:
            per_run.setdefault(rid, set()).add(f)
    assert all(len(v) < 15 for v in per_run.values())

    # pilot-evidence eligibility: NO block (of any kind) sits on a scripted-tail
    # boundary — ROUTE_101 (index 14) is the measured case; the plan-level row agrees
    assert "ROUTE_101" in INELIGIBLE_BLOCK_BOUNDARIES
    for r in plan["runs"]:
        for idx, name, args in r["block_schedule"]:
            assert MILESTONE_ORDER[idx] not in INELIGIBLE_BLOCK_BOUNDARIES, \
                f"{r['run_id']} schedules {name} at ineligible {MILESTONE_ORDER[idx]}"
    (brow,) = rep["rows"]["block_boundaries"]
    assert brow["ok"] and brow["support"] == 0

    # every scheduled trainer block constructs (registered name, valid kwargs)
    for r in plan["runs"]:
        for idx, name, args in r["block_schedule"]:
            if name == "trainer_engagement":
                blk = EXPEDITION_BLOCKS[name](**args)
                assert blk.phase == "battle"
