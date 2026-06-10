"""Lossless chunk recorder for systematic collection."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from collection.actions import button_vec, normalize_button_list
from collection.state import json_safe


class ChunkRecorder:
    """Record sampled visual frames plus full-rate actions/states/segments.

    Preferred backend is ``ffv1`` via an ffmpeg pipe. When ffmpeg is unavailable,
    ``npz`` is a lossless fallback intended for tests and small pilots.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        run_id: str,
        emulator_fps: int = 80,
        visual_fps: int = 30,
        frame_size: tuple[int, int] = (240, 160),
        backend: str = "auto",
        max_chunk_visual_frames: int = 10_000,
        metadata: dict[str, Any] | None = None,
        lean: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.emulator_fps = int(emulator_fps)
        self.visual_fps = int(visual_fps)
        self.frame_size = tuple(frame_size)
        self.max_chunk_visual_frames = int(max_chunk_visual_frames)
        self.metadata = metadata or {}

        ffmpeg_path = shutil.which("ffmpeg")
        if backend == "auto":
            backend = "ffv1" if ffmpeg_path else "npz"
        if backend == "ffv1" and not ffmpeg_path:
            raise RuntimeError("FFV1 backend requires ffmpeg, but ffmpeg was not found in PATH")
        if backend not in {"ffv1", "npz"}:
            raise ValueError(f"Unsupported chunk backend: {backend}")
        self.backend = backend
        self.ffmpeg_path = ffmpeg_path

        # lean: drop the legacy per-frame state + segment streams (unused by collect_coverage;
        # semantic/abstract state is derivable from the stored condition). Keeps the dataset clean.
        self.lean = bool(lean)
        self.frames_file = (self.output_dir / "frames.jsonl").open("w", encoding="utf-8", buffering=1)
        self.actions_file = (self.output_dir / "actions.jsonl").open("w", encoding="utf-8", buffering=1)
        self.states_file = None if self.lean else (self.output_dir / "states.jsonl").open("w", encoding="utf-8", buffering=1)
        self.segments_file = None if self.lean else (self.output_dir / "segments.jsonl").open("w", encoding="utf-8", buffering=1)
        self.manifest_path = self.output_dir / "manifest.json"

        self.chunk_index = 0
        self.chunk_visual_frame_idx = 0
        self.visual_frame_idx = 0
        self.last_visual_emulator_frame: int | None = None
        self._next_visual_at = 0.0
        self._chunk_process: subprocess.Popen | None = None
        self._npz_frames: list[np.ndarray] = []
        self._current_chunk_name: str | None = None
        self._closed = False
        self._started = time.monotonic()

        self.manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "backend": self.backend,
            "emulator_fps": self.emulator_fps,
            "visual_fps": self.visual_fps,
            "frame_size": list(self.frame_size),
            "max_chunk_visual_frames": self.max_chunk_visual_frames,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "running",
            "metadata": json_safe(self.metadata),
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(json_safe(self.manifest), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def _write_jsonl(self, handle, row: dict[str, Any]) -> None:
        handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")

    def _start_chunk(self) -> None:
        self.chunk_index += 1
        self.chunk_visual_frame_idx = 0
        suffix = "mkv" if self.backend == "ffv1" else "npz"
        self._current_chunk_name = f"chunks/chunk_{self.chunk_index:06d}.{suffix}"
        chunk_path = self.output_dir / self._current_chunk_name
        if self.backend == "ffv1":
            width, height = self.frame_size
            cmd = [
                self.ffmpeg_path or "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(self.visual_fps),
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-g",
                "1",
                str(chunk_path),
            ]
            self._chunk_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self._npz_frames = []

    def _finish_chunk(self) -> None:
        if not self._current_chunk_name:
            return
        if self.backend == "ffv1":
            assert self._chunk_process is not None
            if self._chunk_process.stdin:
                self._chunk_process.stdin.close()
            rc = self._chunk_process.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited with status {rc}")
            self._chunk_process = None
        else:
            chunk_path = self.output_dir / self._current_chunk_name
            frames = np.stack(self._npz_frames, axis=0) if self._npz_frames else np.zeros((0, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
            np.savez_compressed(chunk_path, frames=frames)
            self._npz_frames = []
        self._current_chunk_name = None

    def should_record_visual(self, emulator_frame_idx: int) -> bool:
        if self.visual_fps >= self.emulator_fps:
            return True
        if emulator_frame_idx + 1e-6 >= self._next_visual_at:
            self._next_visual_at += self.emulator_fps / self.visual_fps
            return True
        return False

    def record_visual_frame(self, *, emulator_frame_idx: int, screenshot: Any, state_hash: str | None = None, already_gated: bool = False) -> None:
        if not already_gated and not self.should_record_visual(emulator_frame_idx):
            return
        if self._current_chunk_name is None or self.chunk_visual_frame_idx >= self.max_chunk_visual_frames:
            self._finish_chunk()
            self._start_chunk()
        frame = np.asarray(screenshot, dtype=np.uint8)
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            raise ValueError(f"Unexpected frame shape {frame.shape}; expected {self.frame_size}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame, got shape {frame.shape}")
        if self.backend == "ffv1":
            assert self._chunk_process is not None and self._chunk_process.stdin is not None
            self._chunk_process.stdin.write(frame.tobytes())
        else:
            self._npz_frames.append(np.array(frame, copy=True))
        self._write_jsonl(
            self.frames_file,
            {
                "visual_frame_idx": self.visual_frame_idx,
                "emulator_frame_idx": emulator_frame_idx,
                "chunk": self._current_chunk_name,
                "chunk_frame_idx": self.chunk_visual_frame_idx,
                "state_hash": state_hash,
                "timestamp": time.time(),
            },
        )
        self.last_visual_emulator_frame = emulator_frame_idx
        self.visual_frame_idx += 1
        self.chunk_visual_frame_idx += 1

    def record_action(self, *, frame_idx: int, next_frame_idx: int, buttons: list[str], phase: str, metadata: dict[str, Any] | None = None) -> None:
        normalized = normalize_button_list(buttons)
        self._write_jsonl(
            self.actions_file,
            {
                "frame_idx": frame_idx,
                "next_frame_idx": next_frame_idx,
                "buttons_held": normalized,
                "button_vec": button_vec(normalized),
                "phase": phase,
                "metadata": metadata or {},
                "timestamp": time.time(),
            },
        )

    def record_state(self, state: dict[str, Any]) -> None:
        if self.states_file is not None:
            self._write_jsonl(self.states_file, state)

    def record_segment(self, segment: dict[str, Any]) -> None:
        if self.segments_file is not None:
            self._write_jsonl(self.segments_file, segment)

    def close(self, *, status: str = "complete", details: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._finish_chunk()
        self._closed = True
        for handle in (self.frames_file, self.actions_file, self.states_file, self.segments_file):
            if handle is not None:
                handle.close()
        self.manifest.update(
            {
                "status": status,
                "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - self._started,
                "visual_frame_count": self.visual_frame_idx,
                "chunk_count": self.chunk_index,
                "details": json_safe(details or {}),
            }
        )
        self._write_manifest()

    def __enter__(self) -> "ChunkRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(status="failed" if exc else "complete", details={"error": str(exc)} if exc else None)

