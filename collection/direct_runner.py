"""Direct emulator stepping runtime for fast collection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from collection.catch_guard import ball_throw_blocked
from collection.nickname_guard import nickname_prompt_open
from collection.move_keeper import resolve as move_keeper_resolve, swap_prompt_open
from collection.battle_menu_guard import submenu_trap_open
from collection.actions import ActionTiming, PaceProbe, paced_action_frames, run_action_frames, update_facing
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
        savestate_every: int = 4000,
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
        self._in_move_keeper = False   # re-entrancy: move_keeper drives its own actions
        self.last_recorded_state: AbstractState | None = None
        # W33: every restore is tallied (frame position + whether it was recorded) so
        # the manifest can prove "no hidden frames" — the replay audit's precondition.
        self.restore_log: list[dict] = []
        # W33 §3.4: periodic savestates -> ANY future field derivable by load+short
        # replay, and runs resumable at block granularity. 0 disables (planning runners).
        self.savestate_every = savestate_every
        # W33 §13 solve-then-record capture mode: when not None, every step_frame
        # appends ("frame", buttons, phase, metadata) and every record_state_snapshot
        # appends ("state", phase, metadata) — the DRY (recorder-detached) attempt's
        # exact per-frame schedule plus its sparse state-row schedule, re-executable
        # under recording by replay_capture(). The director arms/disarms this around
        # each dry milestone attempt; nothing else touches it.
        self.capture_log: list[tuple] | None = None

    def _save_periodic_state(self) -> None:
        import zlib

        sb = self.save_state_bytes()
        if sb is None or self.recorder is None:
            return
        d = self.recorder.output_dir / "savestates"
        d.mkdir(exist_ok=True)
        (d / f"{self.frame_idx:08d}.state.z").write_bytes(zlib.compress(sb, 6))

    def set_phase(self, phase: str | None) -> None:
        """W33 §2: director stamps the activity phase; a savestate marks the boundary
        (block-granular resume + the phase's exact start state on disk)."""
        if self.recorder is not None:
            if phase == self.recorder.block_phase:
                return  # no transition (recorder would no-op) -> no boundary savestate
            self.recorder.set_block_phase(phase, frame_idx=self.frame_idx)
            if self.savestate_every:
                self._save_periodic_state()

    def initialize(self) -> None:
        from pokemon_env.emulator import EmeraldEmulator

        self.env = EmeraldEmulator(rom_path=self.rom_path)
        self.env.initialize()
        if self.recorder is not None:
            # W33 §3.1/§10.3: recording REQUIRES the fixed RTC (wall-clock RTC broke
            # replay for every pre-2026-06-28 family) and pins the environment.
            if getattr(self.env, "_pokemon_wm_fixed_rtc_value", None) is None:
                raise RuntimeError(
                    "recording without a fixed RTC is forbidden (POKEMON_WM_FIXED_RTC "
                    "was disabled?) — replay determinism would be silently lost")
            from collection.provenance import collect_provenance

            self.recorder.manifest["provenance"] = collect_provenance(self.env, self.rom_path)
            # live reference: close() serializes the final contents of this list
            self.recorder.manifest["restores"] = self.restore_log
            self.recorder._write_manifest()
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
        if self.capture_log is not None:
            self.capture_log.append(("state", phase, metadata))
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
        if self.capture_log is not None:
            self.capture_log.append(("frame", list(buttons), phase, metadata))
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
        if (self.savestate_every and self.recorder is not None
                and self.frame_idx % self.savestate_every == 0):
            self._save_periodic_state()

    def pace_probe(self) -> PaceProbe:
        """Light per-frame poll for condition-based pacing (W33 §3.5): location/coords
        from the memory reader + ONE gObjectEvents bus read for mid-step + true facing.
        Deliberately skips game_state/battle/dialogue (the expensive parts of nav_state)
        — the pacing predicate only needs tile identity and rest."""
        assert self.env is not None
        from collection.extractors.entities import player_pace_read

        reader = getattr(self.env, "memory_reader", None)
        coords = safe_call(reader.read_coordinates) if reader else None
        x, y = coords if isinstance(coords, tuple) and len(coords) >= 2 else (None, None)
        location = location_name(safe_call(reader.read_location) if reader else None)
        rec = safe_call(lambda: player_pace_read(self.env))
        mid_step, facing = rec if rec is not None else (False, None)
        return PaceProbe(pos=(location, x, y), facing=facing, mid_step=bool(mid_step))

    def perform_action(
        self,
        action: str,
        *,
        speed: str = "normal",
        timing: ActionTiming | None = None,
        metadata: dict[str, Any] | None = None,
        record_end_state: bool = True,
    ) -> AbstractState | None:
        # Refuse nickname offers wherever the press comes from. This has to sit at the
        # one choke point every action passes through: the A presses that accepted them
        # came from several different mashing loops (policy confirm, walk-and-talk NPC
        # recovery, dialog-release, unstick), and gating them one at a time leaves the
        # next one to rediscover the bug. See collection/nickname_guard.py.
        if str(action).upper() == "A" and (
                nickname_prompt_open(self.env) or ball_throw_blocked(self)):
            action = "B"

        # A battle party/summary screen answers only to B. Every button, not just A:
        # the steer's UP/LEFT do nothing there, so translating A alone would take ~15
        # presses to walk out instead of 5. See collection/battle_menu_guard.py for the
        # 50,000-frame full-HP deadlock this ends.
        if submenu_trap_open(self):
            action = "B"

        # Same reasoning, one level up: the forget-a-move prompt cannot be answered by
        # vetoing a single button — declining it takes two correctly timed answers and
        # every attempt at that still lost the move. So this guard RESOLVES the prompt
        # itself, which means re-entering perform_action; the flag keeps that one level
        # deep. See collection/move_keeper.py for the frame-by-frame loss it prevents.
        if not self._in_move_keeper and swap_prompt_open(self):
            self._in_move_keeper = True
            try:
                kept = move_keeper_resolve(self)
                print(f"move_keeper: forget-prompt answered, Double Kick "
                      f"{'kept' if kept else 'LOST'}", flush=True)
            finally:
                self._in_move_keeper = False

        self.facing = update_facing(self.facing, action)
        if timing is not None:
            # Explicit fixed schedule: bit-identical to the pre-W33 behavior for any
            # caller that passes `timing=` (the compatibility contract, W33 §3.5).
            schedule = run_action_frames(action, timing)
        else:
            # W33 §3.5 condition-based pacing: hold while polling until the effect
            # commits, then settle until stable — see actions.paced_action_frames.
            # Every polled frame goes through step_frame, so the recording semantics
            # are unchanged: one action row per frame, buttons = what was really held.
            # `speed` no longer picks a schedule; it is kept as row provenance.
            schedule = paced_action_frames(action, self.pace_probe)
        for idx, buttons in enumerate(schedule):
            phase = "hold" if buttons else "release"
            self.step_frame(
                buttons,
                phase=phase,
                metadata={
                    "action": action,
                    "speed": speed,
                    "schedule_index": idx,
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
        if record:
            # recorded restore: settle frame refreshes the video buffer and is COUNTED
            self.env.load_state(state_bytes=state_bytes)
            self.frame_idx += 1
            self.record_current_frame(phase="restore", record_state=True, record_visual=True)
        else:
            # FRAME-NEUTRAL restore (W33 §3.2): no settle frame -> the emulator is
            # byte-exactly the saved state; nothing hidden ever elapses. This closes
            # the v1 replay leak (every silent restore used to burn one unrecorded
            # frame). Restores are tallied for the manifest regardless.
            self.env.load_state(state_bytes=state_bytes, run_settle_frame=False)
        self.restore_log.append({"frame_idx": self.frame_idx, "recorded": bool(record)})

    def replay_capture(self, log: list[tuple]) -> None:
        """W33 §13 solve-then-record: re-execute a captured DRY schedule frame-by-frame
        UNDER the attached recorder/sinks. `log` entries are ("frame", buttons, phase,
        metadata) and ("state", phase, metadata) in execution order (see capture_log),
        so the recording is row-for-row what a live recorded run of the same buttons
        would have produced (action rows keep their semantic phase/metadata; the sparse
        states.jsonl rows re-read the — deterministically identical — emulator state)."""
        assert self.capture_log is None, "cannot replay while capture mode is armed"
        for entry in log:
            if entry[0] == "frame":
                _, buttons, phase, metadata = entry
                self.step_frame(list(buttons), phase=phase, metadata=metadata, record_state=False)
            else:
                _, phase, metadata = entry
                self.record_state_snapshot(phase=phase, metadata=metadata)

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
        max_actions: int = 72,  # rescaled for condition-based pacing (W33 §3.5): ~3x more actions per frame
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

