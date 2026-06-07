"""Frame/action aligned episode recorder for emulator rollouts."""

from __future__ import annotations

import datetime
import json
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from PIL import Image


BUTTON_ORDER = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON runtime values into plain JSON values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _normalize_button(button: Any) -> str:
    return str(button).strip().upper()


def _button_vec(buttons: Iterable[Any]) -> Dict[str, int]:
    held = {_normalize_button(button) for button in buttons}
    return {button: int(button in held) for button in BUTTON_ORDER}


def _location_name(location: Any) -> Optional[str]:
    if isinstance(location, dict):
        return (
            location.get("map_name")
            or location.get("name")
            or location.get("location")
            or location.get("current_location")
        )
    if location is None:
        return None
    return str(location)


def _position_xy(position: Any) -> tuple[Optional[int], Optional[int]]:
    if isinstance(position, dict):
        return position.get("x"), position.get("y")
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        return position[0], position[1]
    return None, None


def _party_summary(party: Any) -> list[dict[str, Any]]:
    if not isinstance(party, list):
        return []

    summary = []
    for pokemon in party:
        if isinstance(pokemon, dict):
            summary.append(
                {
                    "species": pokemon.get("species_name") or pokemon.get("species") or pokemon.get("name"),
                    "level": pokemon.get("level"),
                    "hp": pokemon.get("current_hp") or pokemon.get("hp"),
                    "max_hp": pokemon.get("max_hp"),
                    "status": pokemon.get("status"),
                }
            )
        else:
            summary.append(
                {
                    "species": getattr(pokemon, "species_name", None) or getattr(pokemon, "name", None),
                    "level": getattr(pokemon, "level", None),
                    "hp": getattr(pokemon, "current_hp", None),
                    "max_hp": getattr(pokemon, "max_hp", None),
                    "status": getattr(pokemon, "status", None),
                }
            )
    return _json_safe(summary)


def _copy_screenshot_for_worker(screenshot: Any) -> Any:
    if isinstance(screenshot, np.ndarray):
        return np.array(screenshot, copy=True)
    if hasattr(screenshot, "copy") and hasattr(screenshot, "save"):
        return screenshot.copy()
    return screenshot


def _write_png_frame(frame_path: Path, screenshot: Any, compress_level: int) -> None:
    if hasattr(screenshot, "save"):
        screenshot.save(frame_path, format="PNG", compress_level=compress_level)
    elif isinstance(screenshot, np.ndarray):
        Image.fromarray(screenshot).save(frame_path, format="PNG", compress_level=compress_level)
    else:
        raise ValueError(f"Unsupported screenshot type: {type(screenshot)}")


class EpisodeRecorder:
    """Write frames plus per-frame transition labels for one gameplay episode."""

    def __init__(
        self,
        episode_dir: Path | str,
        *,
        run_id: str,
        game: str,
        rom_path: Optional[str] = None,
        rom_sha1: Optional[str] = None,
        state_interval: int = 1,
        fps_target: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        frame_writer_workers: Optional[int] = None,
        png_compress_level: int = 6,
        max_pending_frame_writes: Optional[int] = None,
        compact_state_mode: str = "fast",
    ):
        self.episode_dir = Path(episode_dir)
        self.frames_dir = self.episode_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id
        self.game = game
        self.rom_path = rom_path
        self.rom_sha1 = rom_sha1
        self.state_interval = max(1, int(state_interval or 1))
        self.fps_target = fps_target
        self.metadata = metadata or {}
        if frame_writer_workers is None:
            frame_writer_workers = 4
        self.frame_writer_workers = max(0, int(frame_writer_workers or 0))
        self.png_compress_level = max(0, min(9, int(png_compress_level)))
        self.compact_state_mode = compact_state_mode if compact_state_mode in {"fast", "comprehensive"} else "fast"
        self._frame_executor = (
            ThreadPoolExecutor(max_workers=self.frame_writer_workers, thread_name_prefix="episode-frame-writer")
            if self.frame_writer_workers > 1
            else None
        )
        self._pending_frame_writes: set[Future] = set()
        default_max_pending = max(8, self.frame_writer_workers * 4) if self.frame_writer_workers > 1 else 0
        self._max_pending_frame_writes = max(1, int(max_pending_frame_writes or default_max_pending))

        self.frame_count = 0
        self.transition_count = 0
        self.state_count = 0
        self.milestone_count = 0
        self._started_at_monotonic = time.monotonic()
        self._finalized = False
        self._seen_completed_milestones: set[str] = set()

        self.actions_file = (self.episode_dir / "actions.jsonl").open("w", encoding="utf-8", buffering=1)
        self.states_file = (self.episode_dir / "states.jsonl").open("w", encoding="utf-8", buffering=1)
        self.milestones_file = (self.episode_dir / "milestones.jsonl").open("w", encoding="utf-8", buffering=1)
        self.manifest_path = self.episode_dir / "manifest.json"

        self.manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "game": game,
            "scope": "pokemon_emerald_start_to_first_gym_clear",
            "frame_format": "png",
            "frame_size": [240, 160],
            "button_order": BUTTON_ORDER,
            "state_interval": self.state_interval,
            "fps_target": fps_target,
            "frame_writer_workers": self.frame_writer_workers,
            "png_compress_level": self.png_compress_level,
            "compact_state_mode": self.compact_state_mode,
            "rom_path": rom_path,
            "rom_sha1": rom_sha1,
            "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "running",
            "metadata": _json_safe(self.metadata),
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(_json_safe(self.manifest), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.manifest_path)

    def _write_jsonl(self, handle, row: dict[str, Any]) -> None:
        handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")

    def _drain_completed_frame_writes(self, *, block: bool = False) -> None:
        if not self._pending_frame_writes:
            return
        if block:
            done, pending = wait(self._pending_frame_writes, return_when=FIRST_COMPLETED)
            self._pending_frame_writes = set(pending)
        else:
            done = {future for future in self._pending_frame_writes if future.done()}
            self._pending_frame_writes.difference_update(done)
        for future in done:
            future.result()

    def _wait_for_frame_writes(self) -> None:
        while self._pending_frame_writes:
            self._drain_completed_frame_writes(block=True)
        if self._frame_executor is not None:
            self._frame_executor.shutdown(wait=True)
            self._frame_executor = None

    def _save_frame(self, frame_idx: int, screenshot: Any) -> str:
        frame_path = self.frames_dir / f"{frame_idx:06d}.png"
        if self._frame_executor is None:
            _write_png_frame(frame_path, screenshot, self.png_compress_level)
        else:
            self._drain_completed_frame_writes(block=False)
            while len(self._pending_frame_writes) >= self._max_pending_frame_writes:
                self._drain_completed_frame_writes(block=True)
            frame_copy = _copy_screenshot_for_worker(screenshot)
            future = self._frame_executor.submit(_write_png_frame, frame_path, frame_copy, self.png_compress_level)
            self._pending_frame_writes.add(future)
        return str(frame_path.relative_to(self.episode_dir))

    def record_initial_frame(self, screenshot: Any, env: Any = None) -> None:
        """Save frame 0 before the first transition."""
        if self.frame_count > 0:
            return
        self._save_frame(0, screenshot)
        self.frame_count = 1
        self.record_state(0, env=env, screenshot=screenshot)
        self.record_milestone_changes(0, env)
        self._write_manifest()

    def record_transition(
        self,
        *,
        screenshot: Any,
        actions_pressed: Iterable[Any],
        action_context: Optional[dict[str, Any]] = None,
        env: Any = None,
    ) -> None:
        """Save the next frame and the action that produced it."""
        if self.frame_count == 0:
            self.record_initial_frame(screenshot, env=env)
            return

        transition_idx = self.transition_count
        source_frame_idx = self.frame_count - 1
        next_frame_idx = self.frame_count
        context = action_context or {}
        normalized_buttons = [_normalize_button(button) for button in actions_pressed]

        self._save_frame(next_frame_idx, screenshot)
        action_row = {
            "transition_idx": transition_idx,
            "frame_idx": source_frame_idx,
            "next_frame_idx": next_frame_idx,
            "buttons_held": normalized_buttons,
            "button_vec": _button_vec(normalized_buttons),
            "phase": context.get("phase", "idle"),
            "current_action": context.get("current_action"),
            "request_id": context.get("request_id"),
            "sequence_index": context.get("sequence_index"),
            "sequence_length": context.get("sequence_length"),
            "queue_length": context.get("queue_length"),
            "speed": context.get("speed"),
            "hold_frames": context.get("hold_frames"),
            "release_frames": context.get("release_frames"),
            "source": context.get("source"),
            "metadata": context.get("metadata") or {},
            "timestamp": time.time(),
        }
        self._write_jsonl(self.actions_file, action_row)

        self.frame_count += 1
        self.transition_count += 1

        if next_frame_idx % self.state_interval == 0:
            self.record_state(next_frame_idx, env=env, screenshot=screenshot)
        self.record_milestone_changes(next_frame_idx, env)

    def record_state(self, frame_idx: int, *, env: Any = None, screenshot: Any = None) -> None:
        state = self._compact_state(frame_idx, env=env, screenshot=screenshot)
        self._write_jsonl(self.states_file, state)
        self.state_count += 1

    def _safe_call(self, fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    def _latest_milestone(self, env: Any) -> Optional[str]:
        if env is not None and hasattr(env, "milestone_tracker"):
            try:
                milestone, _, _ = env.milestone_tracker.get_latest_milestone_info()
                return milestone
            except Exception:
                return None
        return None

    def _compact_state_fast(self, frame_idx: int, *, env: Any = None) -> Optional[dict[str, Any]]:
        reader = getattr(env, "memory_reader", None) if env is not None else None
        if reader is None:
            return None

        coords = self._safe_call(reader.read_coordinates)
        x, y = _position_xy(coords)
        location = self._safe_call(reader.read_location)
        money = self._safe_call(reader.read_money)
        game_state = self._safe_call(reader.get_game_state)
        in_battle = self._safe_call(reader.is_in_battle)
        has_dialogue = self._safe_call(reader.is_in_dialog)
        badges = self._safe_call(reader.read_badges, []) or []

        party = []
        if hasattr(env, "get_party_pokemon"):
            party = self._safe_call(env.get_party_pokemon, []) or []
        if not party and hasattr(reader, "read_party_pokemon"):
            party = self._safe_call(reader.read_party_pokemon, []) or []

        return {
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "location": _location_name(location),
            "x": x,
            "y": y,
            "game_state": game_state,
            "in_battle": in_battle,
            "dialogue": has_dialogue,
            "badges": len(badges) if isinstance(badges, list) else badges,
            "badge_names": badges,
            "party": _party_summary(party),
            "money": money,
            "milestone": self._latest_milestone(env),
        }

    def _compact_state(self, frame_idx: int, *, env: Any = None, screenshot: Any = None) -> dict[str, Any]:
        if self.compact_state_mode == "fast":
            fast_state = self._compact_state_fast(frame_idx, env=env)
            if fast_state is not None:
                return fast_state

        comprehensive = {}
        if env is not None and hasattr(env, "get_comprehensive_state"):
            try:
                comprehensive = env.get_comprehensive_state(screenshot=screenshot)
            except TypeError:
                comprehensive = env.get_comprehensive_state()
            except Exception as exc:
                comprehensive = {"error": str(exc)}

        player = comprehensive.get("player", {}) if isinstance(comprehensive, dict) else {}
        game = comprehensive.get("game", {}) if isinstance(comprehensive, dict) else {}
        map_data = comprehensive.get("map", {}) if isinstance(comprehensive, dict) else {}

        location = _location_name(player.get("location") or map_data.get("location"))
        x, y = _position_xy(player.get("position"))

        dialogue_info = game.get("dialogue_detected") if isinstance(game, dict) else {}
        has_dialogue = None
        if isinstance(dialogue_info, dict):
            has_dialogue = dialogue_info.get("has_dialogue")

        badges = game.get("badges") if isinstance(game, dict) else []
        if badges is None:
            badges = []
        party = player.get("party") if isinstance(player, dict) else []

        milestone = self._latest_milestone(env)

        return {
            "frame_idx": frame_idx,
            "timestamp": time.time(),
            "location": location,
            "x": x,
            "y": y,
            "game_state": game.get("game_state") if isinstance(game, dict) else None,
            "in_battle": game.get("is_in_battle") if isinstance(game, dict) else None,
            "dialogue": has_dialogue,
            "badges": len(badges) if isinstance(badges, list) else badges,
            "badge_names": badges,
            "party": _party_summary(party),
            "money": game.get("money") if isinstance(game, dict) else None,
            "milestone": milestone,
        }

    def record_milestone_changes(self, frame_idx: int, env: Any) -> None:
        if env is None or not hasattr(env, "milestone_tracker"):
            return
        milestones = getattr(env.milestone_tracker, "milestones", {})
        if not isinstance(milestones, dict):
            return

        for milestone_id, data in milestones.items():
            if not isinstance(data, dict) or not data.get("completed"):
                continue
            if milestone_id in self._seen_completed_milestones:
                continue
            self._seen_completed_milestones.add(milestone_id)
            row = {
                "frame_idx": frame_idx,
                "milestone": milestone_id,
                "name": data.get("name", milestone_id),
                "category": data.get("category"),
                "completed": True,
                "timestamp": data.get("timestamp") or time.time(),
            }
            self._write_jsonl(self.milestones_file, row)
            self.milestone_count += 1

    def record_terminal_event(self, frame_idx: int, *, reason: str, success: bool, details: Optional[dict[str, Any]] = None) -> None:
        self._write_jsonl(
            self.milestones_file,
            {
                "frame_idx": frame_idx,
                "event": "terminal",
                "reason": reason,
                "success": success,
                "details": details or {},
                "timestamp": time.time(),
            },
        )
        self.milestone_count += 1

    def copy_auxiliary_files(
        self,
        *,
        trajectory_path: Optional[str | Path] = None,
        submission_log_path: Optional[str | Path] = None,
        llm_log_path: Optional[str | Path] = None,
    ) -> None:
        copies = {
            "trajectory_history.jsonl": trajectory_path,
            "submission.log": submission_log_path,
            "llm_log.jsonl": llm_log_path,
        }
        for filename, source in copies.items():
            if not source:
                continue
            source_path = Path(source)
            if source_path.exists():
                shutil.copy2(source_path, self.episode_dir / filename)

    def finalize(
        self,
        *,
        end_reason: str = "shutdown",
        success: bool = False,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._finalized:
            return
        self._wait_for_frame_writes()
        self._finalized = True

        self.manifest.update(
            {
                "status": "complete" if success else "stopped",
                "success": success,
                "end_reason": end_reason,
                "ended_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - self._started_at_monotonic,
                "frame_count": self.frame_count,
                "transition_count": self.transition_count,
                "state_count": self.state_count,
                "milestone_count": self.milestone_count,
                "details": _json_safe(details or {}),
            }
        )
        self._write_manifest()

        self.actions_file.close()
        self.states_file.close()
        self.milestones_file.close()

