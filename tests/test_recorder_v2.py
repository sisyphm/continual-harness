"""W33 corpus-v2 recorder upgrades: periodic savestates, phase tags, no wall-clock rows,
provenance + restore tally in the manifest. Real emulator, real recorder, tmp dir."""

import glob
import json
import zlib

import pytest

from collection.direct_runner import DirectEmulatorRunner
from collection.recorder import ChunkRecorder

STATE = sorted(glob.glob("tests/states/*.state"))[0]


@pytest.fixture()
def rig(tmp_path):
    rec = ChunkRecorder(output_dir=tmp_path, run_id="v2test", backend="npz",
                        emulator_fps=60, visual_fps=60)
    runner = DirectEmulatorRunner(load_state=STATE, recorder=rec, savestate_every=10)
    runner.initialize()
    yield runner, rec, tmp_path
    rec.close()
    runner.close()


def test_v2_recording_contract(rig):
    runner, rec, out = rig
    runner.set_phase("bfs_sweep")
    for i in range(25):
        runner.step_frame(["a"] if i % 5 == 0 else [], phase="action")
    runner.set_phase("spine")
    for _ in range(5):
        runner.step_frame([], phase="action")
    rec.close(details={})

    # periodic savestates at the cadence + phase boundaries, decompressible
    states = sorted(glob.glob(str(out / "savestates" / "*.state.z")))
    assert len(states) >= 3
    assert len(zlib.decompress(open(states[0], "rb").read())) > 100_000

    # action rows: no wall-clock, block_phase stamped inline, stream HOMOGENEOUS
    # (M1: every actions.jsonl row is an action row — consumers index known keys)
    rows = [json.loads(l) for l in open(out / "actions.jsonl")]
    assert all("timestamp" not in r for r in rows)
    assert all("buttons_held" in r and "phase_transition" not in r for r in rows)
    tagged = [r for r in rows if r.get("block_phase")]
    assert {r["block_phase"] for r in tagged} == {"bfs_sweep", "spine"}
    # transitions live in their OWN stream: phases.jsonl
    phases = [json.loads(l) for l in open(out / "phases.jsonl")]
    assert [p["phase"] for p in phases] == ["bfs_sweep", "spine"]
    assert all(set(p) == {"frame_idx", "phase"} for p in phases)

    # manifest: provenance pinned, restores tallied (initial load is not a restore)
    mani = json.loads((out / "manifest.json").read_text())
    assert len(mani["provenance"]["rom_sha256"]) == 64
    assert mani["provenance"]["fixed_rtc_value"] is not None
    assert mani["restores"] == []

    # states.jsonl rows carry no wall-clock either
    srows = [json.loads(l) for l in open(out / "states.jsonl")]
    assert srows and all("timestamp" not in r for r in srows)


def test_phase_rows_never_pollute_action_stream(rig):
    """M1 proof: every consumer loads actions.jsonl assuming homogeneous action rows.
    Run a recording WITH phase changes, then execute the real audit loader
    (collection.audits.replay.load_actions) — it KeyError'd on transition rows pre-fix."""
    from collection.audits.replay import load_actions

    runner, rec, out = rig
    runner.set_phase("bfs_sweep")
    for _ in range(3):
        runner.step_frame([], phase="action")
    runner.set_phase("spine")
    runner.step_frame([], phase="action")
    rec.close(details={})

    actions = load_actions(out)                   # the audits/replay.py loading path
    assert sorted(actions) == list(range(4))      # one action row per frame, no gaps
    assert all(isinstance(v, list) for v in actions.values())
    phases = [json.loads(l) for l in open(out / "phases.jsonl")]
    assert [(p["frame_idx"], p["phase"]) for p in phases] == [(0, "bfs_sweep"), (3, "spine")]


def test_set_phase_same_phase_skips_boundary_savestate(rig):
    """Sharp minor 7: a redundant set_phase (recorder no-ops the transition) must not
    drop a duplicate boundary savestate either."""
    runner, rec, out = rig

    def n_states():
        return len(glob.glob(str(out / "savestates" / "*.state.z")))

    runner.set_phase("bfs_sweep")                 # boundary savestate at frame 0
    runner.step_frame([], phase="action")         # move off the boundary frame
    n0 = n_states()
    runner.set_phase("bfs_sweep")                 # no transition -> NO new savestate
    assert n_states() == n0
    runner.set_phase("spine")                     # real transition -> exactly one more
    assert n_states() == n0 + 1
