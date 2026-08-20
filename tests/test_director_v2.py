"""W33 §13 solve-then-record + §14 persona layer: director/spine surgery acceptance.
Real emulator, real recorder, tmp dirs (pattern from test_pacing_v2 / test_blocks_v2).

Measured ground truth these tests pin (2026-08-21 build session, real emulator):
- EXIT_RIVAL_HOUSE from GO_DOWNSTAIRS_RIVAL_HOUSE_completed.state: dry solve ~121
  actions / ~1.26k frames; recorded replay is frame-exact and outcome-identical
  (ledger vector + (map,x,y)) — the continuous determinism check passes.
- Blanking the first ~20 held frames of the captured schedule desyncs every later
  press: the replay ends INSIDE the house ((1,4) on MAYS_HOUSE_1F vs (14,9) in
  LITTLEROOT) -> DeterminismError.
- no_dialog1.state (Littleroot, player (12,12)): goal (17,14) has multiple equal-cost
  shortest paths; tie-break seeds 1 vs 3 walk different 8-tile sequences, both
  'arrived'. Seed 2 first-steps RIGHT into the parked NPC at (13,12) — the documented
  pre-existing goto/_unstick NPC-retalk pathology (goto docstring's Brendan-house-mom
  case), NOT a tie-break defect; test seeds are pinned to NPC-free first steps.
"""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from collection.direct_runner import DirectEmulatorRunner
from collection.recorder import ChunkRecorder
from collection.playthrough.director import (
    DeterminismError,
    build_persona,
    make_tic_fn,
    solve_then_record_milestone,
)

POLICY_DIR = "../pokeagent-solution/expert_policies_by_llm"
EVENT = "EXIT_RIVAL_HOUSE"
PRE_STATE = f"{POLICY_DIR}/GO_DOWNSTAIRS_RIVAL_HOUSE/GO_DOWNSTAIRS_RIVAL_HOUSE_completed.state"
COMPLETED = f"{POLICY_DIR}/{EVENT}/{EVENT}_completed.state"
LITTLEROOT_STATE = "tests/states/no_dialog1.state"

needs_policies = pytest.mark.skipif(not Path(PRE_STATE).exists(),
                                    reason="expert policy corpus not mounted")


def _milestone_rig(tmp_path):
    import collection.collect_events as ce

    rec = ChunkRecorder(tmp_path, run_id="director_v2", backend="npz",
                        emulator_fps=60, visual_fps=60)
    runner = DirectEmulatorRunner(load_state=PRE_STATE, recorder=rec, savestate_every=0)
    runner.initialize()
    exp = ce._load_expected_state(rom_path="Emerald-GBAdvance/rom.gba",
                                  completed_state=COMPLETED, event_id=EVENT)
    return rec, runner, exp


def _solve(runner, exp, persona, **hooks):
    return solve_then_record_milestone(
        runner, event_id=EVENT, policy_dir=POLICY_DIR, expected_state=exp,
        postcondition=EVENT, start_money=0, max_actions=2000, starter="mudkip",
        persona=persona, **hooks)


def _action_rows(out):
    return [json.loads(l) for l in open(out / "actions.jsonl")]


# ---------------------------------------------------------------------------
# acceptance 1: solve-then-record on one quick real milestone


@needs_policies
def test_solve_then_record_clean_milestone(tmp_path):
    """Dry-solve succeeds, recorded replay matches (no DeterminismError), the recording
    contains ONLY the successful trajectory (rows == replay frames, jitter included),
    and the retry log lands in the result."""
    rec, runner, exp = _milestone_rig(tmp_path)
    try:
        res = _solve(runner, exp, build_persona(11))
        assert res["validation"] == "passed"
        t = res["retry"]
        assert t["attempts"] == 1 and t["dry_frames_wasted"] == 0
        assert t["replay_frames"] == t["dry_frames_solve"] > 0
        assert res["end_frame"] == runner.frame_idx

        # recording == the replayed schedule exactly: contiguous, no dry residue
        rec.close()
        rows = _action_rows(tmp_path)
        assert len(rows) == t["replay_frames"] == res["end_frame"]
        assert [r["frame_idx"] for r in rows] == list(range(len(rows)))
        # §14.2 always-on jitter: K0 idle frames ARE the recording's opening content
        jit = [r for r in rows if r["phase"] == "milestone_jitter"]
        assert len(jit) == t["jitters"][0]
        assert all(r["buttons_held"] == [] for r in jit)
        assert all(r["metadata"]["event_id"] == EVENT for r in jit)
        assert rows[0]["phase"] == "milestone_jitter"

        # states stream: real rows only, never a recorded restore
        srows = [json.loads(l) for l in open(tmp_path / "states.jsonl")]
        assert all(r.get("record_phase") != "restore" for r in srows)
        assert sum(r.get("record_phase") == "action_end" for r in srows) == res["actions_taken"]

        # every solve restore is frame-neutral (recorded False) and tagged
        mani = json.loads((tmp_path / "manifest.json").read_text())
        assert mani["restores"], "solve restore must be tallied"
        assert all(e["recorded"] is False for e in mani["restores"])
        assert any(e.get("solve") == EVENT for e in mani["restores"])
    finally:
        rec.close()
        runner.close()


# ---------------------------------------------------------------------------
# acceptance 2: forced-retry path — attempt 1 fails, attempt 2 runs with jitter


@needs_policies
def test_forced_retry_recovers_with_shifted_jitter(tmp_path):
    rec, runner, exp = _milestone_rig(tmp_path)
    try:
        res = _solve(runner, exp, build_persona(11),
                     fail_injector=lambda attempt: {"max_actions": 5} if attempt == 0 else {})
        assert res["validation"] == "passed"
        t = res["retry"]
        assert t["attempts"] == 2
        assert t["jitters"][1] - t["jitters"][0] == 17     # K = K0 + attempt*17
        assert t["dry_frames_wasted"] > 0                  # attempt 1's frames, discarded

        rec.close()
        rows = _action_rows(tmp_path)
        # final recording is CLEAN: only attempt 2's replay, contiguous from 0
        assert len(rows) == t["replay_frames"] == res["end_frame"]
        assert [r["frame_idx"] for r in rows] == list(range(len(rows)))
        jit = [r for r in rows if r["phase"] == "milestone_jitter"]
        assert len(jit) == t["jitters"][1]                 # attempt 2's K, not attempt 1's
        assert {r["metadata"]["attempt"] for r in jit} == {1}
    finally:
        rec.close()
        runner.close()


# ---------------------------------------------------------------------------
# acceptance 3: determinism verification — corrupted replay raises, milestone named


@needs_policies
def test_corrupt_replay_raises_determinism_error(tmp_path):
    rec, runner, exp = _milestone_rig(tmp_path)

    def corrupt(log):
        out, blanked = [], 0
        for e in log:
            if e[0] == "frame" and e[1] and blanked < 20:  # blank the first ~20 held frames
                out.append(("frame", [], e[2], e[3]))
                blanked += 1
            else:
                out.append(e)
        return out

    try:
        with pytest.raises(DeterminismError, match=EVENT):
            _solve(runner, exp, build_persona(11), corrupt_replay=corrupt)
    finally:
        rec.close()
        runner.close()


# ---------------------------------------------------------------------------
# acceptance 4: §14.1 seeded BFS tie-breaking — route diversity at zero cost


def test_tiebreak_unit_bfs_first_step_diversity():
    """Pure BFS: rng=None keeps the historical fixed order; seeded rngs spread the
    first step across the equal-cost options — and NEVER change the path cost class
    (every returned step is toward the goal on an open grid)."""
    from collection.navigator import _bfs_step

    walk = np.ones((15, 15), bool)
    goals = np.zeros((15, 15), bool)
    goals[12, 12] = True
    fixed = {_bfs_step(walk, (5, 5), goals) for _ in range(5)}
    assert len(fixed) == 1                                 # rng=None: deterministic
    seeded = {_bfs_step(walk, (5, 5), goals, rng=random.Random(s)) for s in range(12)}
    assert len(seeded) >= 2, f"tie-break never varied the first step: {seeded}"
    assert seeded <= {(1, 0), (0, 1)}                      # only steps TOWARD the goal


def test_tiebreak_real_map_same_seed_same_path_diff_seed_diff_path():
    """no_dialog1.state, goal (17,14) (multiple equal-cost shortest paths, first steps
    NPC-free for the pinned seeds): same seed -> identical tile sequence; seeds 1 vs 3
    -> different sequences of the SAME length, both arriving."""
    from collection import navigator as nav

    r = DirectEmulatorRunner(load_state=LITTLEROOT_STATE, savestate_every=0)
    r.initialize()
    try:
        mk = nav.MapKnowledge()
        t, x, y = nav._state(r)
        assert (x, y) == (12, 12)
        snap = r.save_state_bytes()
        f0, fc0 = r.frame_idx, r.facing

        def goal(t, beh):
            m = np.zeros(t.grid.shape, bool)
            m[14 + 7, 17 + 7] = True
            return m

        def run(seed):
            r.frame_idx, r.facing = f0, fc0
            r.load_state_bytes(snap, record=False)
            visited = []

            def visit(t, xx, yy):
                if not visited or visited[-1] != (xx, yy):
                    visited.append((xx, yy))

            res = nav.goto(r, mk, goal, budget=6000, visit_fn=visit,
                           rng=random.Random(seed))
            return res, visited

        res_a, path_a = run(1)
        res_a2, path_a2 = run(1)
        res_b, path_b = run(3)
        assert res_a == res_a2 == res_b == "arrived"
        assert path_a == path_a2                           # same seed -> same path
        assert path_a != path_b                            # different seed -> different route
        assert len(path_a) == len(path_b)                  # ... at the SAME (shortest) cost
        # the pilot's diversity number, on this micro-route
        inter, union = set(path_a) & set(path_b), set(path_a) | set(path_b)
        jacc = len(inter) / len(union)
        print(f"tie-break path overlap seeds 1 vs 3: jaccard={jacc:.3f} "
              f"({len(inter)}/{len(union)} tiles)")
        assert jacc < 1.0
    finally:
        r.close()


def test_mapknowledge_rng_is_gotos_default():
    """goto picks up mk.rng when no explicit rng is passed (persona plumbing seam)."""
    from collection import navigator as nav

    mk = nav.MapKnowledge()
    assert mk.rng is None                                  # default: historical order
    assert nav.MapKnowledge(rng=random.Random(7)).rng is not None


# ---------------------------------------------------------------------------
# acceptance 5: §14.3 micro-tics — present at high rate, bounded away from battle/dialog


class _FakeTicRunner:
    def __init__(self, in_battle=False, control_mode="free_overworld"):
        self.facing = "DOWN"
        self.stepped = []
        self._nav = SimpleNamespace(in_battle=in_battle, control_mode=control_mode)

    def nav_state(self):
        return self._nav

    def step_frame(self, buttons, *, phase, metadata=None, record_state=False):
        self.stepped.append((list(buttons), phase))


def test_tic_bound_battle_dialog_never_tics(monkeypatch):
    from collection import navigator as nav

    for rig in (_FakeTicRunner(in_battle=True, control_mode="battle"),
                _FakeTicRunner(control_mode="dialog")):
        make_tic_fn(rig, random.Random(0), rate=1.0)()
        assert rig.stepped == []
    # free overworld but the window-mask reads a textbox: skip (false positive = safe skip)
    rig = _FakeTicRunner()
    monkeypatch.setattr(nav, "_dialog_open", lambda runner: True)
    make_tic_fn(rig, random.Random(0), rate=1.0)()
    assert rig.stepped == []
    # clean overworld: the tic really emits frames, all phase-tagged "tic"
    monkeypatch.setattr(nav, "_dialog_open", lambda runner: False)
    make_tic_fn(rig, random.Random(0), rate=1.0)()
    assert rig.stepped and all(p == "tic" for _, p in rig.stepped)


@needs_policies
def test_tics_appear_in_captured_schedule_and_replay(tmp_path):
    """tic_rate=1.0: tics land in the captured schedule, are replayed into the
    recording, and the milestone still passes + verifies."""
    rec, runner, exp = _milestone_rig(tmp_path)
    try:
        res = _solve(runner, exp, build_persona(11, {"tic_rate": 1.0}))
        assert res["validation"] == "passed"               # tics never break the solve
        rec.close()
        rows = _action_rows(tmp_path)
        tics = [r for r in rows if r["phase"] == "tic"]
        assert len(tics) > 20, "high tic rate must produce tics in the recording"
        assert {r["metadata"]["kind"] for r in tics} <= {"pause", "turn"}
    finally:
        rec.close()
        runner.close()


# ---------------------------------------------------------------------------
# §14.3 wander/browse restoration + §14.4 persona plumbing + §14.5 overlap metric


def test_wander_browse_schedulable_with_phase_tags(tmp_path):
    """v1 Wander/Browse are registered expedition blocks; run through run_nav_block
    their frames carry the 'wander'/'browse' phase tag and the phase restores to
    spine (measured live: wander 13 actions/~0.4k frames, browse 70/~0.7k)."""
    from collection.playthrough.blocks.base import run_nav_block
    from collection.playthrough.schedule import EXPEDITION_BLOCKS, build_expedition_schedule

    assert {"wander", "browse"} <= set(EXPEDITION_BLOCKS)
    sched = build_expedition_schedule([
        {"after": "OLDALE_TOWN", "block": "wander", "seed": 5, "steps": 3},
        {"after": "OLDALE_TOWN", "block": "browse", "seed": 5},
    ])
    assert [b.phase for b in sched["OLDALE_TOWN"]] == ["wander", "browse"]
    assert all(hasattr(b, "run") for b in sched["OLDALE_TOWN"])

    rec = ChunkRecorder(tmp_path, run_id="wander_v2", backend="npz",
                        emulator_fps=60, visual_fps=60)
    r = DirectEmulatorRunner(load_state=LITTLEROOT_STATE, recorder=rec, savestate_every=0)
    r.initialize()
    try:
        out = run_nav_block(r, EXPEDITION_BLOCKS["wander"](seed=5, steps=3, max_actions=60))
        assert out["ran"] is True and out["legacy_ran"] is True
        assert out["returned"] is True                     # anchor contract preserved
        rec.close()
        trans = [json.loads(l)["phase"] for l in open(tmp_path / "phases.jsonl")]
        assert trans == ["wander", "spine"]                # tagged, then restored
        tagged = {row.get("block_phase")
                  for row in _action_rows(tmp_path) if row.get("block_phase")}
        assert tagged == {"wander"}
    finally:
        rec.close()
        r.close()


def test_persona_defaults_and_overrides():
    p = build_persona(9)
    assert p == {"seed": 9, "tie_break": True, "tic_rate": 0.02, "jitter": 30}
    p = build_persona(9, {"tic_rate": 0.5, "jitter": 4})
    assert p["seed"] == 9 and p["tic_rate"] == 0.5 and p["jitter"] == 4
    assert build_persona(9, {"seed": 123})["seed"] == 123  # explicit seed wins


def test_path_overlap_metric(tmp_path):
    """§14.5 diversity audit: (map,x,y) visit overlap from states.jsonl, spine frames
    only by default (phases.jsonl ranges; None/'spine' count as spine)."""
    from collection.audits.path_overlap import path_overlap

    def write_run(name, states, phases=None):
        d = tmp_path / name
        d.mkdir()
        with (d / "states.jsonl").open("w") as f:
            for fr, m, x, y in states:
                f.write(json.dumps({"frame_idx": fr, "map": m, "x": x, "y": y}) + "\n")
        if phases is not None:
            with (d / "phases.jsonl").open("w") as f:
                for fr, ph in phases:
                    f.write(json.dumps({"frame_idx": fr, "phase": ph}) + "\n")
        return d

    a = write_run("a", [(0, "M", 0, 0), (1, "M", 1, 0), (2, "M", 2, 0), (10, "M", 9, 9)],
                  phases=[(5, "bfs_sweep")])               # frame 10 is block work
    b = write_run("b", [(0, "M", 0, 0), (1, "M", 1, 1), (2, "M", 2, 0)])

    r = path_overlap(a, b)                                 # spine-only default
    assert (r["tiles_a"], r["tiles_b"], r["tiles_shared"]) == (3, 3, 2)
    assert r["overlap_jaccard"] == pytest.approx(0.5)
    assert r["shared_frac_a"] == pytest.approx(2 / 3)

    r_all = path_overlap(a, b, spine_only=False)           # block frames included
    assert r_all["tiles_a"] == 4
    assert r_all["overlap_jaccard"] == pytest.approx(2 / 5)


# ---------------------------------------------------------------------------
# persona -> manifest/summary plumbing through the real director (no milestones)


def test_run_playthrough_persona_in_manifest_and_summary(tmp_path, monkeypatch):
    """run_playthrough with an EMPTY policy dir: boot only, zero milestones — proves
    the persona lands in manifest['persona'] + the summary, and the §13 telemetry
    aggregates exist, without paying for a full spine."""
    import collection.playthrough.director as director

    class _DummySink:
        def __init__(self, *a, **k):
            self.capture = lambda runner: None

        def close(self):
            pass

    monkeypatch.setattr(director, "WorldModelSink", _DummySink)
    policies = tmp_path / "policies"
    policies.mkdir()
    out = tmp_path / "run"
    s = director.run_playthrough(policy_dir=str(policies), out_dir=str(out),
                                 starter="mudkip", seed=4, record=True,
                                 persona={"tic_rate": 0.1})
    persona = {"seed": 4, "tie_break": True, "tic_rate": 0.1, "jitter": 30}
    assert s["persona"] == persona
    assert s["milestones_total"] == 0 and s["aborted_milestone"] is None
    assert s["retry"] == {} and s["retry_total"]["attempts"] == 0
    mani = json.loads((out / "manifest.json").read_text())
    assert mani["persona"] == persona                      # §14.4: first-class manifest field
    assert mani["metadata"]["persona"] == persona
    summary = json.loads((out / "playthrough_summary.json").read_text())
    assert summary["persona"] == persona
