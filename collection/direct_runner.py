"""Direct emulator stepping runtime for fast collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from collection.actions import ActionTiming, run_action_frames, timing_for, update_facing
from collection.recorder import ChunkRecorder
from collection.state import AbstractState, control_mode, location_name, read_compact_state, safe_call


@dataclass(frozen=True)
class NavSnapshot:
    map: str | None
    x: int | None
    y: int | None
    facing: str
    control_mode: str
    game_state: str | None = None
    in_battle: bool | None = None


class DirectEmulatorRunner:
    """Owns one Emerald emulator and steps it without wall-clock throttling."""

    def __init__(
        self,
        *,
        rom_path: str = "Emerald-GBAdvance/rom.gba",
        load_state: str | None = None,
        story_bucket: str = "UNKNOWN",
        facing: str = "DOWN",
        recorder: ChunkRecorder | None = None,
        emulator_fps: int = 80,
        frame_hook=None,
    ):
        self.rom_path = str(rom_path)
        self.load_state = str(load_state) if load_state is not None else None
        self.story_bucket = story_bucket
        self.facing = facing
        self.recorder = recorder
        self.emulator_fps = emulator_fps
        # Called as frame_hook(self) after each *real* recorded frame (not planning sims,
        # which bypass step_frame). Used to capture the per-frame world-model condition.
        self.frame_hook = frame_hook
        self.env = None
        self.frame_idx = 0
        self.last_recorded_state: AbstractState | None = None

    def initialize(self) -> None:
        from pokemon_env.emulator import EmeraldEmulator

        self.env = EmeraldEmulator(rom_path=self.rom_path)
        self.env.initialize()
        if self.load_state:
            self.env.load_state(self.load_state)
        self.frame_idx = 0
        self.record_current_frame(phase="initial")

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.stop()
            except Exception:
                pass

    def screenshot(self):
        assert self.env is not None
        return self.env.get_screenshot()

    def state(self) -> AbstractState:
        assert self.env is not None
        return read_compact_state(self.env, frame_idx=self.frame_idx, story_bucket=self.story_bucket, facing=self.facing)

    def nav_state(self) -> NavSnapshot:
        assert self.env is not None
        reader = getattr(self.env, "memory_reader", None)
        coords = safe_call(reader.read_coordinates) if reader else None
        x, y = coords if isinstance(coords, tuple) and len(coords) >= 2 else (None, None)
        location = location_name(safe_call(reader.read_location) if reader else None)
        game_state = safe_call(reader.get_game_state) if reader else None
        in_battle = safe_call(reader.is_in_battle) if reader else None
        # Avoid the expensive visual dialogue validation in the fast path. Full
        # AbstractState is still read at segment boundaries for final labels.
        dialogue = str(game_state).lower() == "dialog" if game_state is not None else False
        return NavSnapshot(
            map=location,
            x=x,
            y=y,
            facing=self.facing,
            control_mode=control_mode(game_state=game_state, in_battle=in_battle, dialogue=dialogue),
            game_state=str(game_state) if game_state is not None else None,
            in_battle=bool(in_battle) if in_battle is not None else None,
        )

    def record_state_snapshot(self, *, phase: str, metadata: dict[str, Any] | None = None, state: AbstractState | None = None) -> AbstractState:
        state = state or self.state()
        self.last_recorded_state = state
        if self.recorder:
            row = state.to_dict()
            row["record_phase"] = phase
            if metadata:
                row["record_metadata"] = metadata
            self.recorder.record_state(row)
        return state

    def record_visual_snapshot(self, *, state_hash: str | None = None) -> None:
        if not self.recorder:
            return
        # Decide whether this frame is kept (visual_fps subsamples ~5/6 away) BEFORE
        # grabbing/converting the screenshot, so we don't pay for frames we'd discard.
        if not self.recorder.should_record_visual(self.frame_idx):
            return
        screenshot = self.screenshot()
        if screenshot is not None:
            self.recorder.record_visual_frame(
                emulator_frame_idx=self.frame_idx,
                screenshot=np.asarray(screenshot, dtype=np.uint8),
                state_hash=state_hash,
                already_gated=True,
            )

    def record_current_frame(
        self,
        *,
        phase: str,
        metadata: dict[str, Any] | None = None,
        record_state: bool = True,
        record_visual: bool = True,
    ) -> AbstractState | None:
        state = self.record_state_snapshot(phase=phase, metadata=metadata) if record_state else None
        if record_visual:
            self.record_visual_snapshot(state_hash=state.raw_state_hash if state else None)
        return state

    def step_frame(
        self,
        buttons: list[str],
        *,
        phase: str,
        metadata: dict[str, Any] | None = None,
        record_state: bool = False,
    ) -> None:
        assert self.env is not None
        source_frame = self.frame_idx
        self.env.run_frame_with_buttons([button.lower() for button in buttons])
        self.frame_idx += 1
        if self.recorder:
            self.recorder.record_action(
                frame_idx=source_frame,
                next_frame_idx=self.frame_idx,
                buttons=buttons,
                phase=phase,
                metadata=metadata or {},
            )
        self.record_current_frame(phase=phase, metadata=metadata, record_state=record_state, record_visual=True)
        if self.frame_hook is not None:
            self.frame_hook(self)

    def perform_action(
        self,
        action: str,
        *,
        speed: str = "normal",
        timing: ActionTiming | None = None,
        metadata: dict[str, Any] | None = None,
        record_end_state: bool = True,
    ) -> AbstractState | None:
        timing = timing or timing_for(speed)
        self.facing = update_facing(self.facing, action)
        schedule = run_action_frames(action, timing)
        for idx, buttons in enumerate(schedule):
            phase = "hold" if buttons else "release"
            self.step_frame(
                buttons,
                phase=phase,
                metadata={
                    "action": action,
                    "speed": speed,
                    "schedule_index": idx,
                    "schedule_length": len(schedule),
                    **(metadata or {}),
                },
                record_state=False,
            )
        # Callers that immediately re-read state (e.g. the explorer's wait_until_stable)
        # can skip this ~12 ms read_compact_state by passing record_end_state=False.
        if not record_end_state:
            return None
        return self.record_state_snapshot(phase="action_end", metadata={"action": action, **(metadata or {})})

    def save_state_bytes(self) -> bytes | None:
        assert self.env is not None
        return self.env.save_state()

    def load_state_bytes(self, state_bytes: bytes, *, record: bool = True) -> None:
        assert self.env is not None
        self.env.load_state(state_bytes=state_bytes)
        # load_state runs one emulator frame internally. Keep it silent for BFS restores
        # so metadata frame ranges describe only collected transitions.
        if record:
            self.frame_idx += 1
            self.record_current_frame(phase="restore", record_state=True, record_visual=True)

    def wait_until_stable(
        self,
        *,
        max_frames: int = 160,
        stable_frames: int = 6,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractState:
        last = self.nav_state()
        stable = 0
        for _ in range(max_frames):
            self.step_frame([], phase="stabilize", metadata=metadata, record_state=False)
            current = self.nav_state()
            if (
                current.map == last.map
                and current.x == last.x
                and current.y == last.y
                and current.facing == last.facing
                and current.control_mode == last.control_mode
            ):
                stable += 1
            else:
                stable = 0
            last = current
            if stable >= stable_frames:
                return self.record_state_snapshot(phase="stable", metadata=metadata)
        return self.record_state_snapshot(phase="stable_timeout", metadata=metadata)

    def settle_to_free_overworld(
        self,
        *,
        max_actions: int = 24,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractState:
        """Clear checkpoint-tail dialogue/menu frames before directional BFS."""
        state = self.wait_until_stable(max_frames=40, stable_frames=4, metadata={"settle": "initial", **(metadata or {})})
        if state.control_mode == "free_overworld":
            return state

        # Dialogues and simple confirmation menus are deterministic tails of events.
        # Alternate confirm/cancel with waits so the collector can reach the first
        # controllable overworld frame without treating these as explore actions.
        schedule = ("A", "A", "B", "WAIT")
        for idx in range(max_actions):
            action = schedule[idx % len(schedule)]
            self.perform_action(
                action,
                speed="fast",
                metadata={"segment_type": "settle", "settle_action_index": idx, **(metadata or {})},
            )
            state = self.wait_until_stable(
                max_frames=40,
                stable_frames=4,
                metadata={"segment_type": "settle", "settle_action_index": idx, **(metadata or {})},
            )
            if state.control_mode == "free_overworld":
                return state
        return state


class TimedRun:
    def __init__(self):
        self.started = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started


def state_file_for_heatz_policy(policy_dir: str | Path, event_id: str) -> Path:
    return Path(policy_dir) / event_id / f"{event_id}_completed.state"

