import json
from pathlib import Path

import numpy as np
import pytest

from collection.actions import action_to_buttons, button_vec, normalize_action, run_action_frames, timing_for, update_facing
from collection.audit_events import _recommendation
from collection.catalog import discover_heatz_events, previous_milestone
from collection.recorder import ChunkRecorder
from collection.state import AbstractState, result_label, stable_hash


def test_action_normalization_and_timing():
    assert normalize_action("up") == "UP"
    assert normalize_action("no_op") == "WAIT"
    assert action_to_buttons("WAIT") == []
    assert action_to_buttons("a") == ["A"]
    assert update_facing("DOWN", "left") == "LEFT"
    timing = timing_for("fast", hold_frames=2, release_frames=1)
    assert run_action_frames("A", timing) == [["A"], ["A"], []]
    assert button_vec(["A", "LEFT"])["A"] == 1
    assert button_vec(["A", "LEFT"])["RIGHT"] == 0


def _state(**overrides):
    data = dict(
        frame_idx=0,
        story_bucket="TEST",
        map="ROUTE_102",
        x=1,
        y=2,
        facing="DOWN",
        control_mode="free_overworld",
        game_state="overworld",
        dialogue=False,
        in_battle=False,
        badges=0,
        badge_names=[],
        party_summary=[],
        money=0,
        milestone=None,
        flags_hash="f",
        raw_state_hash="r",
        timestamp=0.0,
    )
    data.update(overrides)
    return AbstractState(**data)


def test_state_keys_hashes_and_result_labels():
    st = _state()
    assert st.explore_key("UP") == "TEST|ROUTE_102|1|2|DOWN|UP"
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})
    assert result_label(st, _state(x=1, y=1)) == "move"
    assert result_label(st, _state(facing="LEFT")) == "turn_only"
    assert result_label(st, _state(map="OLDALE TOWN")) == "warp"
    assert result_label(st, _state(control_mode="battle")) == "battle_start"
    assert result_label(st, _state(control_mode="dialogue")) == "dialogue_open"
    assert result_label(st, _state()) == "blocked"


def test_npz_chunk_recorder_writes_metadata(tmp_path):
    recorder = ChunkRecorder(tmp_path, run_id="test", backend="npz", emulator_fps=80, visual_fps=40, max_chunk_visual_frames=2)
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    for idx in range(5):
        recorder.record_visual_frame(emulator_frame_idx=idx, screenshot=frame, state_hash=f"s{idx}")
        recorder.record_action(frame_idx=idx, next_frame_idx=idx + 1, buttons=["A"], phase="hold")
        recorder.record_state({"frame_idx": idx})
    recorder.record_segment({"segment_type": "test", "validation": "passed"})
    recorder.close()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["backend"] == "npz"
    assert manifest["status"] == "complete"
    assert manifest["visual_frame_count"] == 3
    assert len(list((tmp_path / "chunks").glob("*.npz"))) == 2
    assert sum(1 for _ in open(tmp_path / "frames.jsonl")) == 3
    assert sum(1 for _ in open(tmp_path / "actions.jsonl")) == 5


def test_catalog_discovers_heatz_events(tmp_path):
    base = tmp_path / "policies"
    for name in ["GAME_RUNNING", "PLAYER_NAME_SET"]:
        d = base / name
        d.mkdir(parents=True)
        (d / f"{name}.py").write_text("def run(state): return 'a'\n")
        (d / f"{name}_completed.state").write_bytes(b"state")
    rows = discover_heatz_events(base)
    by_id = {row["event_id"]: row for row in rows}
    assert by_id["PLAYER_NAME_SET"]["pre_milestone"] == "GAME_RUNNING"
    assert by_id["PLAYER_NAME_SET"]["pre_state"].endswith("GAME_RUNNING_completed.state")
    assert previous_milestone("PLAYER_NAME_SET") == "GAME_RUNNING"



def test_audit_recommendation_classifies_event_starts():
    start = _state(map="LAB", x=1, y=1)
    expected = _state(map="LAB", x=5, y=5)
    probes = [
        {"action": "UP", "result": "blocked"},
        {"action": "DOWN", "result": "move"},
        {"action": "A", "result": "blocked"},
    ]
    assert _recommendation(start_state=start, expected_state=expected, probes=probes, policy_validation="passed") == (
        "collect",
        "policy_validated",
    )
    assert _recommendation(start_state=expected, expected_state=expected, probes=probes, policy_validation=None) == (
        "skip",
        "already_complete_start",
    )
    stuck = [{"action": action, "result": "turn_only"} for action in ["UP", "DOWN", "LEFT", "RIGHT"]]
    assert _recommendation(start_state=start, expected_state=expected, probes=stuck, policy_validation=None) == (
        "skip",
        "script_or_dialog_control_lock",
    )
