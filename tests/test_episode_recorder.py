import json
from pathlib import Path

from PIL import Image

from utils.data_collection.episode_recorder import EpisodeRecorder


class FakeMilestoneTracker:
    def __init__(self):
        self.milestones = {
            "GAME_RUNNING": {"completed": True, "name": "Game Running", "category": "system", "timestamp": 1.0},
            "STONE_BADGE": {"completed": False, "name": "Stone Badge", "category": "gym"},
        }

    def get_latest_milestone_info(self):
        completed = [key for key, value in self.milestones.items() if value.get("completed")]
        return (completed[-1] if completed else "NONE", None, None)


class FakeEnv:
    def __init__(self):
        self.milestone_tracker = FakeMilestoneTracker()

    def get_comprehensive_state(self, screenshot=None):
        return {
            "player": {
                "location": "LITTLEROOT_TOWN",
                "position": {"x": 5, "y": 8},
                "party": [
                    {
                        "species_name": "MUDKIP",
                        "level": 5,
                        "current_hp": 20,
                        "max_hp": 20,
                        "status": "OK",
                    }
                ],
            },
            "game": {
                "money": 3000,
                "game_state": "overworld",
                "is_in_battle": False,
                "badges": [],
                "dialogue_detected": {"has_dialogue": False},
            },
            "map": {},
        }


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_episode_recorder_writes_frame_action_state_alignment(tmp_path):
    env = FakeEnv()
    recorder = EpisodeRecorder(
        tmp_path / "episode_000001",
        run_id="run_test",
        game="emerald",
        rom_path="Emerald-GBAdvance/rom.gba",
        rom_sha1="abc123",
        state_interval=1,
        fps_target=80,
    )

    recorder.record_initial_frame(Image.new("RGB", (240, 160), "black"), env=env)
    recorder.record_transition(
        screenshot=Image.new("RGB", (240, 160), "white"),
        actions_pressed=["A"],
        action_context={
            "phase": "hold",
            "current_action": "A",
            "request_id": 7,
            "sequence_index": 0,
            "sequence_length": 1,
            "queue_length": 0,
            "speed": "normal",
            "hold_frames": 10,
            "release_frames": 8,
            "source": "test",
            "metadata": {"reason": "unit"},
        },
        env=env,
    )
    recorder.finalize(end_reason="unit_test", success=True)

    episode_dir = tmp_path / "episode_000001"
    assert (episode_dir / "frames" / "000000.png").exists()
    assert (episode_dir / "frames" / "000001.png").exists()

    actions = _jsonl(episode_dir / "actions.jsonl")
    assert len(actions) == 1
    assert actions[0]["transition_idx"] == 0
    assert actions[0]["frame_idx"] == 0
    assert actions[0]["next_frame_idx"] == 1
    assert actions[0]["buttons_held"] == ["A"]
    assert actions[0]["button_vec"]["A"] == 1
    assert actions[0]["button_vec"]["B"] == 0
    assert actions[0]["request_id"] == 7

    states = _jsonl(episode_dir / "states.jsonl")
    assert [row["frame_idx"] for row in states] == [0, 1]
    assert states[0]["location"] == "LITTLEROOT_TOWN"
    assert states[0]["x"] == 5
    assert states[0]["party"][0]["species"] == "MUDKIP"

    manifest = json.loads((episode_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["success"] is True
    assert manifest["frame_count"] == 2
    assert manifest["transition_count"] == 1


def test_episode_recorder_records_completed_milestones_once(tmp_path):
    env = FakeEnv()
    recorder = EpisodeRecorder(tmp_path / "episode_000001", run_id="run_test", game="emerald")
    recorder.record_initial_frame(Image.new("RGB", (240, 160), "black"), env=env)

    env.milestone_tracker.milestones["STONE_BADGE"].update({"completed": True, "timestamp": 2.0})
    recorder.record_transition(
        screenshot=Image.new("RGB", (240, 160), "white"),
        actions_pressed=[],
        action_context={"phase": "idle"},
        env=env,
    )
    recorder.record_milestone_changes(1, env)
    recorder.finalize(end_reason="first_badge", success=True)

    milestones = _jsonl(tmp_path / "episode_000001" / "milestones.jsonl")
    completed = [row.get("milestone") for row in milestones if row.get("completed")]
    assert completed.count("GAME_RUNNING") == 1
    assert completed.count("STONE_BADGE") == 1


def test_episode_recorder_parallel_frame_writer_keeps_schema(tmp_path):
    env = FakeEnv()
    recorder = EpisodeRecorder(
        tmp_path / "episode_000001",
        run_id="run_parallel",
        game="emerald",
        frame_writer_workers=4,
        png_compress_level=0,
        state_interval=2,
    )

    recorder.record_initial_frame(Image.new("RGB", (240, 160), "black"), env=env)
    for idx in range(1, 6):
        recorder.record_transition(
            screenshot=Image.new("RGB", (240, 160), (idx, idx, idx)),
            actions_pressed=["RIGHT"] if idx % 2 else [],
            action_context={"phase": "hold" if idx % 2 else "idle", "request_id": idx},
            env=env,
        )
    recorder.finalize(end_reason="unit_test", success=False)

    episode_dir = tmp_path / "episode_000001"
    assert sorted(path.name for path in (episode_dir / "frames").glob("*.png")) == [
        "000000.png",
        "000001.png",
        "000002.png",
        "000003.png",
        "000004.png",
        "000005.png",
    ]
    actions = _jsonl(episode_dir / "actions.jsonl")
    assert len(actions) == 5
    assert actions[0]["request_id"] == 1
    assert actions[-1]["next_frame_idx"] == 5

    states = _jsonl(episode_dir / "states.jsonl")
    assert [row["frame_idx"] for row in states] == [0, 2, 4]

    manifest = json.loads((episode_dir / "manifest.json").read_text())
    assert manifest["frame_writer_workers"] == 4
    assert manifest["png_compress_level"] == 0
    assert manifest["frame_count"] == 6
    assert manifest["transition_count"] == 5


def test_episode_recorder_default_frame_writer_workers_is_parallel_safe(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "episode_000001", run_id="run_default_workers", game="emerald")
    try:
        assert recorder.frame_writer_workers == 4
    finally:
        recorder.finalize(end_reason="unit_test", success=False)
