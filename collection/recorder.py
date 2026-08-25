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
        resume: dict | str | Path | None = None,
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

        # RESUME/STITCH (W33 §3.4): with a resume handshake from
        # collection.resume_stitch.prepare_resume the recorder APPENDS to a rewound
        # recording instead of starting one — every channel opens "a", the counters
        # continue where the truncated prefix stopped, and the existing manifest
        # (restores / persona / provenance / metadata) is kept rather than rewritten.
        self.resume = None
        if resume is not None:
            from collection.resume_stitch import load_resume

            self.resume = load_resume(resume)
        mode = "a" if self.resume else "w"

        self.frames_file = (self.output_dir / "frames.jsonl").open(mode, encoding="utf-8", buffering=1)
        self.actions_file = (self.output_dir / "actions.jsonl").open(mode, encoding="utf-8", buffering=1)
        # phase transitions live in their OWN stream (spec §2): actions.jsonl stays
        # homogeneous — every row is an action row (consumers index buttons_held etc.)
        self.phases_file = (self.output_dir / "phases.jsonl").open(mode, encoding="utf-8", buffering=1)
        # states/segments are unconditional (spec §3.7)
        self.states_file = (self.output_dir / "states.jsonl").open(mode, encoding="utf-8", buffering=1)
        self.segments_file = (self.output_dir / "segments.jsonl").open(mode, encoding="utf-8", buffering=1)
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
        self.block_phase: str | None = None       # W33 §2 activity tag (see set_block_phase)

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
        if self.resume:
            self._restore_from(self.resume)
        self._write_manifest()

    def _restore_from(self, cfg: dict[str, Any]) -> None:
        """Continue a rewound recording: counters, chunk numbering and manifest."""
        cut = int(cfg["cut_frame"])
        self.chunk_index = int(cfg["chunk_index"])          # next chunk is this + 1
        self.visual_frame_idx = int(cfg["visual_count"])    # next visual_frame_idx
        self.chunk_visual_frame_idx = 0                     # boundary chunk is sealed
        self.last_visual_emulator_frame = cut
        # subsampling clock: the next frame at/after the seam is the next kept one
        self._next_visual_at = float(cut)
        old = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}
        if old:
            fresh, self.manifest = self.manifest, old
            if old.get("backend") not in (None, fresh["backend"]):
                raise ValueError(
                    f"resume backend mismatch: recording is {old.get('backend')!r}, "
                    f"this recorder is {fresh['backend']!r}")
            # keep restores / persona / provenance / metadata / started_at from the
            # prefix; only the live-run fields are refreshed
            self.manifest["status"] = "running"
            self.manifest["run_id"] = fresh["run_id"]
            self.manifest["visual_fps"] = fresh["visual_fps"]
            self.manifest["max_chunk_visual_frames"] = fresh["max_chunk_visual_frames"]
        self.manifest.setdefault("restores", [])
        # The seam frame: the resumed director restores the savestate as a RECORDED
        # restore, so one visual frame at the seam carries no action row. That entry is
        # logged HERE (the director suppresses the runner's duplicate) — the gate's
        # provenance rule counts recorded restores, so exactly one may exist per seam.
        self.manifest["restores"].append({"frame_idx": cut, "recorded": True, "resume": True,
                                          "savestate": cfg.get("savestate")})
        # a resumed run is frames A..B from one build and B..C from another: say so
        self.manifest.setdefault("resume_segments", []).append({
            "cut_frame": cut,
            "savestate": cfg.get("savestate"),
            "visual_count": cfg.get("visual_count"),
            "actions_count": cfg.get("actions_count"),
            "chunk_index": cfg.get("chunk_index"),
            "resumed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            # initialize() overwrites the top-level provenance with the resuming
            # build's; the prefix's is kept here so the seam stays auditable
            "prefix_provenance": self.manifest.get("provenance"),
        })

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
                # W33 §3.6: no wall-clock in row payloads — byte-identical trajectories
                # must produce byte-identical rows (replay audit). Run-level timing
                # lives in the manifest (started_at/ended_at) only.
            },
        )
        self.last_visual_emulator_frame = emulator_frame_idx
        self.visual_frame_idx += 1
        self.chunk_visual_frame_idx += 1

    def record_action(self, *, frame_idx: int, next_frame_idx: int, buttons: list[str], phase: str, metadata: dict[str, Any] | None = None) -> None:
        normalized = normalize_button_list(buttons)
        row = {
            "frame_idx": frame_idx,
            "next_frame_idx": next_frame_idx,
            "buttons_held": normalized,
            "button_vec": button_vec(normalized),
            "phase": phase,
            "metadata": metadata or {},
            # no wall-clock (W33 §3.6) — see record_visual_frame
        }
        if self.block_phase is not None:
            row["block_phase"] = self.block_phase   # W33 §2: director-stamped activity tag
        self._write_jsonl(self.actions_file, row)

    def set_block_phase(self, phase: str | None, *, frame_idx: int) -> None:
        """W33 §2: director-stamped activity tag. Transitions are logged to phases.jsonl
        so training can slice by phase (actions.jsonl must stay homogeneous — replay/l1/
        views and the model repo's loader index action keys on every row); subsequent
        action rows carry the tag inline."""
        if phase == self.block_phase:
            return
        self.block_phase = phase
        self._write_jsonl(self.phases_file, {"frame_idx": frame_idx, "phase": phase})

    def record_state(self, state: dict[str, Any]) -> None:
        if self.states_file is not None:
            state = {k: v for k, v in state.items() if k != "timestamp"}  # W33 §3.6
            self._write_jsonl(self.states_file, state)

    def record_segment(self, segment: dict[str, Any]) -> None:
        if self.segments_file is not None:
            self._write_jsonl(self.segments_file, segment)

    def close(self, *, status: str = "complete", details: dict[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._finish_chunk()
        self._closed = True
        for handle in (self.frames_file, self.actions_file, self.phases_file, self.states_file, self.segments_file):
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

