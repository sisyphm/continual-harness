"""W33 §3.5 condition-based action pacing: measured speedup, the turn-then-step
correctness that killed the old 6-frame fixed hold, tap latency, planner/execution
binding, and the unchanged recording contract. Real emulator, real recorder, tmp dir
(pattern from test_recorder_v2).

Measured ground truth these tests pin (2026-08-20 feasibility study): walk = 16
frames/tile; the tile coord commits to the step's destination on its first frame; a
turn from standstill consumes 8 frames before that commit; fixed normal (12+48=60)
wasted ~73% of its frames."""

import json
import statistics

import pytest

from collection.actions import (
    ActionTiming,
    DIRECTIONAL_SETTLE,
    PaceProbe,
    TAP_SETTLE,
    paced_action_frames,
)
from collection.direct_runner import DirectEmulatorRunner
from collection.recorder import ChunkRecorder

# Littleroot Town (12,12): free overworld, no wild grass, long N-S walkable corridor.
STATE = "tests/states/no_dialog1.state"


@pytest.fixture()
def runner():
    r = DirectEmulatorRunner(load_state=STATE, savestate_every=0)
    r.initialize()
    yield r
    r.close()


def _tile(r):
    n = r.nav_state()
    return (n.map, n.x, n.y)


# ---------------------------------------------------------------------------
# generator semantics (no emulator): the caps and the poll-release contract


def _fake_probe(positions, mids):
    """probe() returning scripted (pos, mid_step) per call; repeats the last entries.
    Call 0 is the generator's pre-frame baseline; call N is the poll after frame N."""
    calls = {"n": 0}

    def probe():
        i = calls["n"]
        calls["n"] += 1
        return PaceProbe(
            pos=positions[min(i, len(positions) - 1)],
            facing="DOWN",
            mid_step=mids[min(i, len(mids) - 1)],
        )

    return probe


def test_generator_blocked_directional_pays_cap_then_settles():
    # pos never changes: full hold cap, then settle exits after stable_frames
    probe = _fake_probe([("M", 1, 1)], [False])
    frames = list(paced_action_frames("DOWN", probe))
    expected = DIRECTIONAL_SETTLE.hold_frames + DIRECTIONAL_SETTLE.stable_frames
    assert len(frames) == expected
    assert frames[: DIRECTIONAL_SETTLE.hold_frames] == [["DOWN"]] * DIRECTIONAL_SETTLE.hold_frames
    assert frames[DIRECTIONAL_SETTLE.hold_frames:] == [[]] * DIRECTIONAL_SETTLE.stable_frames


def test_generator_releases_when_tile_commits():
    # commit on the 3rd held frame -> exactly 3 held frames, never the full cap
    positions = [("M", 1, 1)] * 3 + [("M", 1, 2)]
    probe = _fake_probe(positions, [False])
    frames = list(paced_action_frames("DOWN", probe))
    held = [f for f in frames if f]
    assert len(held) == 3
    assert all(not f for f in frames[3:])


def test_generator_settle_waits_for_mid_step_and_cap():
    # mid-step for 10 settle probes: settle must not end before it clears (+2 stable),
    # and a mid-step that outlives the cap ends at the cap
    positions = [("M", 1, 1), ("M", 1, 2)]
    probe = _fake_probe(positions, [False] + [True] * 10 + [False])
    frames = list(paced_action_frames("DOWN", probe))
    assert len([f for f in frames if not f]) > 10
    probe = _fake_probe(positions, [False] + [True] * 100)
    frames = list(paced_action_frames("DOWN", probe))
    assert len([f for f in frames if not f]) == DIRECTIONAL_SETTLE.settle_cap


def test_generator_tap_is_short_fixed_hold():
    probe = _fake_probe([("M", 1, 1)], [False])
    frames = list(paced_action_frames("A", probe))
    assert frames[: TAP_SETTLE.hold_frames] == [["A"]] * TAP_SETTLE.hold_frames
    assert len(frames) == TAP_SETTLE.hold_frames + TAP_SETTLE.stable_frames


# ---------------------------------------------------------------------------
# real-emulator acceptance


def test_measured_speedup_frames_per_tile(runner):
    """~20 directional presses through the new path: median frames per COMPLETED tile
    move must be <= 24 (vs 60 fixed) and >= 16 (can't beat the walk animation)."""
    direction = "DOWN"
    per_move = []
    for _ in range(20):
        before = _tile(runner)
        f0 = runner.frame_idx
        runner.perform_action(direction, record_end_state=False)
        used = runner.frame_idx - f0
        if _tile(runner) != before:
            per_move.append(used)
        else:
            # bounced off the corridor end (or a wandering NPC): walk back
            direction = "UP" if direction == "DOWN" else "DOWN"
    assert len(per_move) >= 12
    median = statistics.median(per_move)
    assert 16 <= median <= 24, f"median frames/tile {median} (moves: {per_move})"


def test_turn_then_step_completes_the_move(runner):
    """The documented 6-frame-hold trap: a press opposite to the current facing must
    still COMPLETE the tile move (turn ~8 frames, then the step)."""
    runner.perform_action("UP", record_end_state=False)   # end settled, facing UP
    n0 = runner.nav_state()
    runner.perform_action("DOWN", record_end_state=False)
    n1 = runner.nav_state()
    assert (n1.x, n1.y) == (n0.x, n0.y + 1), f"turn ate the step: {(n0.x, n0.y)} -> {(n1.x, n1.y)}"


def test_tap_action_is_short_when_nothing_reacts(runner):
    runner.perform_action("UP", record_end_state=False)   # (12,11) facing UP: empty tile ahead
    f0 = runner.frame_idx
    runner.perform_action("A", record_end_state=False)
    assert runner.frame_idx - f0 <= 20


def test_explicit_timing_override_keeps_fixed_schedule(runner):
    """Compatibility contract: `timing=` callers get the exact pre-W33 fixed schedule."""
    f0 = runner.frame_idx
    runner.perform_action("DOWN", timing=ActionTiming(hold_frames=12, release_frames=48), record_end_state=False)
    assert runner.frame_idx - f0 == 60


def test_planner_simulates_execution_exactly(runner):
    """collect_events._simulate_action shares the frame schedule with perform_action:
    the simulated landing state must equal the really-executed one."""
    import collection.collect_events as ce

    sb = runner.save_state_bytes()
    n0 = len(runner.restore_log)
    sim = ce._simulate_action(runner, sb, runner.facing, "DOWN")
    assert sim is not None
    post, _post_bytes, _post_facing = sim
    # sharp minor 8: the tally covers ALL restores — the sim's two direct env.load_state
    # calls (enter-sim + finally-restore) are logged as non-recorded planning restores
    sims = runner.restore_log[n0:]
    assert [(e["recorded"], e["sim"]) for e in sims] == [(False, True), (False, True)]
    assert all(e["frame_idx"] == runner.frame_idx for e in sims)
    runner.perform_action("DOWN", record_end_state=False)
    n = runner.nav_state()
    assert (post["map"], post["x"], post["y"]) == (n.map, n.x, n.y)


def test_settle_budget_rescaled_for_paced_actions():
    """Sharp minor 1: settle_to_free_overworld's action budget carries the standard
    x3 condition-based-pacing rescale (24 -> 72)."""
    import inspect

    sig = inspect.signature(DirectEmulatorRunner.settle_to_free_overworld)
    assert sig.parameters["max_actions"].default == 72


def test_recording_contract_unchanged(tmp_path):
    """Every polled frame is a recorded frame: one action row per frame, contiguous
    frame_idx, buttons in rows == what was actually held that frame."""
    rec = ChunkRecorder(output_dir=tmp_path, run_id="pacing", backend="npz",
                        emulator_fps=60, visual_fps=60)
    r = DirectEmulatorRunner(load_state=STATE, recorder=rec, savestate_every=0)
    r.initialize()
    r.set_phase("pacing")                 # a mid-recording transition must not pollute actions.jsonl
    for act in ("DOWN", "A", "UP", "WAIT"):
        r.perform_action(act)
    total = r.frame_idx
    rec.close()
    r.close()

    # M1: actions.jsonl is homogeneous — EVERY row is an action row, contiguous, no filtering
    rows = [json.loads(l) for l in open(tmp_path / "actions.jsonl")]
    assert all("buttons_held" in row for row in rows)
    assert [row["frame_idx"] for row in rows] == list(range(total))
    assert all(row["next_frame_idx"] == row["frame_idx"] + 1 for row in rows)
    for row in rows:
        if row["phase"] == "hold":
            assert row["buttons_held"] == [row["metadata"]["action"]]
        else:
            assert row["buttons_held"] == []
    held = [row for row in rows if row["phase"] == "hold"]
    assert {row["metadata"]["action"] for row in held} == {"DOWN", "A", "UP"}  # WAIT holds nothing
