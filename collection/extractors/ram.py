"""`GBAState` — one read API over GBA memory, backed by either a RECORDED condition blob or the
LIVE emulator. This is the offline=online seam: every extractor parses through this class only, so
the precompute (recorded ewram) and the interactive demo (live mgba) can never skew.

Recorded blobs are the 396,288-byte concatenation of `render_state.BLOCK_SIZES`
(io | palette | oam | vram | ewram | iwram), addressed here by their GBA bus addresses:

    io      0x04000000 (register file)     palette 0x05000000      vram 0x06000000
    oam     0x07000000                     ewram   0x02000000      iwram 0x03000000

`iter_states(run_dir)` is the canonical streaming walk of a run's `ppu_state.bin` delta chain
(keyframe + XOR deltas; holds ONE frame at a time — never materialize a whole run).
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np

from collection.render_state import BLOCK_SIZES

# GBA bus base address per block, in BLOCK_SIZES order
_BLOCK_BASES = {"io": 0x04000000, "palette": 0x05000000, "oam": 0x07000000,
                "vram": 0x06000000, "ewram": 0x02000000, "iwram": 0x03000000}

# (bus_base, blob_offset, size) per block — blob offsets follow BLOCK_SIZES order
_RANGES: list[tuple[int, int, int]] = []
_off = 0
for _name, _sz in BLOCK_SIZES:
    _RANGES.append((_BLOCK_BASES[_name], _off, _sz))
    _off += _sz
BLOB_SIZE = _off                                          # 396,288


class GBAState:
    """Read u8/u16/u32/s16/bytes at GBA addresses, from a recorded blob or a live emulator."""

    def __init__(self, blob: bytes | np.ndarray | None = None, env=None):
        assert (blob is None) != (env is None), "exactly one backend: blob or env"
        self._blob = np.frombuffer(blob, np.uint8) if isinstance(blob, (bytes, bytearray)) else blob
        self._env = env

    @classmethod
    def from_blob(cls, blob: bytes | np.ndarray) -> "GBAState":
        return cls(blob=blob)

    @classmethod
    def from_env(cls, env) -> "GBAState":
        """Live backend over the harness `EmeraldEmulator` (uses its read_memory)."""
        return cls(env=env)

    # -- core reads -------------------------------------------------------------
    def bytes(self, addr: int, n: int) -> bytes:
        if self._env is not None:
            return bytes(self._env.read_memory(addr, n))
        for base, off, sz in _RANGES:
            if base <= addr and addr + n <= base + sz:
                o = off + (addr - base)
                return self._blob[o:o + n].tobytes()
        raise ValueError(f"address {addr:#x}+{n} not in any captured block")

    def u8(self, addr: int) -> int:
        return self.bytes(addr, 1)[0]

    def u16(self, addr: int) -> int:
        b = self.bytes(addr, 2)
        return b[0] | (b[1] << 8)

    def u32(self, addr: int) -> int:
        b = self.bytes(addr, 4)
        return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

    def s16(self, addr: int) -> int:
        v = self.u16(addr)
        return v - 0x10000 if v >= 0x8000 else v

    def array_u16(self, addr: int, n: int) -> np.ndarray:
        return np.frombuffer(self.bytes(addr, 2 * n), "<u2")


def iter_states(run_dir: str | Path) -> Iterator[tuple[int, GBAState]]:
    """Stream a recorded run: yields (emulator_frame_idx, GBAState) one frame at a time."""
    d = Path(run_dir)
    idx = json.loads((d / "ppu_state.bin.idx.json").read_text())
    raw = (d / "ppu_state.bin").read_bytes()
    cur: np.ndarray | None = None
    for f, off, _kind in idx["frames"]:
        ln = int.from_bytes(raw[off + 1:off + 5], "little")
        payload = np.frombuffer(zlib.decompress(raw[off + 5:off + 5 + ln]), np.uint8)
        cur = payload.copy() if raw[off:off + 1] == b"K" else (cur ^ payload)
        assert cur is not None and len(cur) == BLOB_SIZE
        yield int(f), GBAState.from_blob(cur)


def state_at(run_dir: str | Path, frame_idx: int) -> GBAState:
    """Random access to one frame's state (walks the chain up to it)."""
    for f, st in iter_states(run_dir):
        if f >= frame_idx:
            return st
    raise IndexError(f"frame {frame_idx} beyond run {run_dir}")
