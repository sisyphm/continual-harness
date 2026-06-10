"""Shared per-frame world-model recorder: the full PPU condition (delta-compressed) +
semantic state, captured for every emulated frame. Used as a DirectEmulatorRunner
frame_hook so both the exploration tours (collect_rollout) and the storyline playthrough
(collect_events) emit the same RGB + action + full-PPU + semantic dataset.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from collection.render_state import BLOCK_SIZES, extract_full_ppu_state, state_from_blocks


def _serialize_ppu(state: dict) -> bytes:
    return b"".join(state[name] for name, _ in BLOCK_SIZES)


def deserialize_ppu(blob: bytes) -> dict:
    """Inverse of _serialize_ppu: raw blob -> render-ready state (regs/hblank derived)."""
    blocks, off = {}, 0
    for name, sz in BLOCK_SIZES:
        blocks[name] = blob[off:off + sz]
        off += sz
    return state_from_blocks(blocks)


class PPUDeltaWriter:
    """Per-frame PPU state as keyframe + XOR-delta (zlib). >99% static => tiny."""

    def __init__(self, path: Path, keyframe_interval: int = 300):
        self.f = open(path, "wb")
        self.index: list = []
        self.kfi = keyframe_interval
        self.prev = None
        self.n = 0
        self.bytes_written = 0

    def add(self, frame_idx: int, state: dict) -> None:
        blob = np.frombuffer(_serialize_ppu(state), dtype=np.uint8)
        if self.prev is None or self.n % self.kfi == 0:
            kind, payload = b"K", blob
        else:
            kind, payload = b"D", np.bitwise_xor(blob, self.prev)
        comp = zlib.compress(payload.tobytes(), 6)
        off = self.f.tell()
        self.f.write(kind + len(comp).to_bytes(4, "little") + comp)
        self.index.append([frame_idx, off, kind.decode()])
        self.bytes_written += len(comp) + 5
        self.prev = blob
        self.n += 1

    def close(self) -> None:
        name = self.f.name
        self.f.close()
        Path(name + ".idx.json").write_text(json.dumps({"block_sizes": BLOCK_SIZES, "frames": self.index}))


def extract_objects(env) -> list:
    """Active gObjectEvents: graphics_id (identity) + tile coords. Read directly (the
    convenience reader over-filters)."""
    BASE, SZ = 0x02037230, 68
    out = []
    for i in range(16):
        if not (env.read_u8(BASE + i * SZ) & 1):
            continue
        x = int.from_bytes(env.read_memory(BASE + i * SZ + 0x10, 2), "little", signed=True)
        y = int.from_bytes(env.read_memory(BASE + i * SZ + 0x12, 2), "little", signed=True)
        if not (-50 <= x <= 2000 and -50 <= y <= 2000) or (x == 1023 and y == 1023):
            continue  # inactive/garbage slot
        out.append({"slot": i, "graphics_id": env.read_u8(BASE + i * SZ + 0x03), "x": x, "y": y})
    return out


class WorldModelSink:
    """Attach as `runner.frame_hook`: captures full PPU + semantic per real frame."""

    def __init__(self, output_dir: str | Path, keyframe_interval: int = 300):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.ppu = PPUDeltaWriter(out / "ppu_state.bin", keyframe_interval)
        self.sem = (out / "semantic.jsonl").open("w", buffering=1)
        self.frames = 0

    def capture(self, runner) -> None:
        env = runner.env
        self.ppu.add(runner.frame_idx, extract_full_ppu_state(env))
        nav = runner.nav_state()
        self.sem.write(json.dumps({
            "frame": runner.frame_idx, "x": nav.x, "y": nav.y, "facing": runner.facing,
            "map": nav.map, "in_battle": nav.in_battle, "objects": extract_objects(env),
        }) + "\n")
        self.frames += 1

    def close(self) -> None:
        self.ppu.close()
        self.sem.close()
