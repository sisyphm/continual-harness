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

from collection.extractors.entities import entities
from collection.extractors.ledger_panel import FIELDS as LEDGER_FIELDS, read_ledger
from collection.extractors.ram import GBAState
from collection.render_state import BLOCK_SIZES, extract_full_ppu_state, state_from_blocks


def serialize_ppu(state: dict) -> bytes:
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

    def __init__(self, path: Path, keyframe_interval: int = 300, resume: bool = False):
        # RESUME (W33 §3.4): append to a stream truncated by resume_stitch. The XOR
        # chain cannot cross the seam (`prev` belongs to the frames that were cut), so
        # the first appended record is FORCED to be a keyframe — which `prev is None`
        # already guarantees, and `n = 0` keeps the interval counting from the seam.
        self.f = open(path, "ab" if resume else "wb")
        self.index: list = []
        if resume:
            idx = Path(str(path) + ".idx.json")
            if idx.exists():
                self.index = json.loads(idx.read_text()).get("frames", [])
        self.kfi = keyframe_interval
        self.prev = None                       # -> forced keyframe on the first add()
        self.n = 0
        self.bytes_written = 0

    def add(self, frame_idx: int, state: dict, blob: bytes | None = None) -> None:
        # `blob` lets the caller reuse an already-serialized state (no double serialize)
        blob = np.frombuffer(blob if blob is not None else serialize_ppu(state), dtype=np.uint8)
        if self.prev is None or self.n % self.kfi == 0:
            kind, payload = b"K", blob
        else:
            kind, payload = b"D", np.bitwise_xor(blob, self.prev)
        # W34 speed: level 6 on every frame was 72% of the whole per-frame budget
        # (3.45 ms vs 0.64 ms of emulator) — the recorder compressed 5x longer than
        # the game played. Deltas go level 1 (0.82 ms, sparse XOR compresses fine);
        # keyframes keep 6 (1-in-300, they carry the size). Measured on 1,500 real
        # records: 4.2x faster, ~2.6 vs 1.2 KB/frame before keyframe amortization.
        comp = zlib.compress(payload.tobytes(), 6 if kind == b"K" else 1)
        off = self.f.tell()
        self.f.write(kind + len(comp).to_bytes(4, "little") + comp)
        self.index.append([frame_idx, off, kind.decode()])
        self.bytes_written += len(comp) + 5
        self.prev = blob
        self.n += 1
        # W34: a killed run used to lose the block-buffered bin tail AND the whole
        # index (close-only write) — resume_stitch then had to rescan the stream.
        # Flush the bin every record (one syscall against an emulator frame; noise)
        # and snapshot the index every 10k frames, so a SIGKILL costs at most 10k
        # frames of INDEX (the data itself is on disk) and zero bytes of stream.
        self.f.flush()
        if self.n % 10_000 == 0:
            self._write_index()

    def _write_index(self) -> None:
        Path(self.f.name + ".idx.json").write_text(
            json.dumps({"block_sizes": BLOCK_SIZES, "frames": self.index}))

    def close(self) -> None:
        self._write_index()
        self.f.close()


def extract_objects_v0_legacy(env) -> list:
    """The v1-era object reader — WRONG layout (stride 68, gfx@+0x03; truth is 0x24/+0x05, see
    `extractors.entities`), so its output is garbage. Kept ONLY because the v0 model was TRAINED
    on this stream: the v0 live demo must keep feeding the same distribution at inference.
    Every new consumer uses `extractors.entities` (the sink below already does)."""
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


class LedgerWriter:
    """W33 §3.3 per-tick ledger: one `ledger_panel.read_ledger` row per real frame,
    flushed as `ledger/chunk_%06d.npz` every `chunk_frames` (+ partial on close) with an
    `index.json` schema so consumers never re-derive dtypes/shapes from the arrays."""

    def __init__(self, out_dir: str | Path, chunk_frames: int = 4096, resume: bool = False):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.chunk_frames = chunk_frames
        self.chunks: list = []                  # [filename, start_row, n_rows]
        self.total = 0
        if resume and (self.dir / "index.json").exists():
            # continue the truncated ledger: new chunk names follow len(self.chunks),
            # and close() must not clobber the prefix's index with an empty one
            old = json.loads((self.dir / "index.json").read_text())
            self.chunks = old.get("chunks", [])
            self.total = int(old.get("frames", 0))
        self._frame_idx: list[int] = []
        self._rows: dict[str, list] = {name: [] for name in LEDGER_FIELDS}

    def add(self, frame_idx: int, row: dict) -> None:
        self._frame_idx.append(frame_idx)
        for name in LEDGER_FIELDS:
            self._rows[name].append(row[name])
        if len(self._frame_idx) >= self.chunk_frames:
            self._flush()

    def _flush(self) -> None:
        n = len(self._frame_idx)
        if n == 0:
            return
        arrs = {"frame_idx": np.asarray(self._frame_idx, np.uint32)}
        for name, (dt, shape) in LEDGER_FIELDS.items():
            arrs[name] = np.asarray(self._rows[name], dtype=dt).reshape((n, *shape))
        fn = f"chunk_{len(self.chunks):06d}.npz"
        np.savez_compressed(self.dir / fn, **arrs)
        self.chunks.append([fn, self.total, n])
        self.total += n
        self._frame_idx = []
        self._rows = {name: [] for name in LEDGER_FIELDS}
        # refresh the index on EVERY flush (not only close): a crashed run keeps a
        # valid schema covering everything flushed so far
        self._write_index()

    def _write_index(self) -> None:
        fields = {"frame_idx": {"dtype": "uint32", "shape": []}}
        for name, (dt, shape) in LEDGER_FIELDS.items():
            fields[name] = {"dtype": np.dtype(dt).name, "shape": list(shape)}
        (self.dir / "index.json").write_text(json.dumps(
            {"chunk_frames": self.chunk_frames, "fields": fields,
             "chunks": self.chunks, "frames": self.total}))

    def close(self) -> None:
        self._flush()
        self._write_index()   # 0-row runs still get a valid (empty) schema


class WorldModelSink:
    """Attach as `runner.frame_hook`: captures full PPU + semantic + ledger per real frame."""

    def __init__(self, output_dir: str | Path, keyframe_interval: int = 300,
                 resume: dict | str | Path | None = None):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # RESUME/STITCH: append to the channels rewound by resume_stitch.prepare_resume
        # (ppu stream + index, semantic.jsonl, ledger chunks) instead of truncating them.
        self.resume = None
        if resume is not None:
            from collection.resume_stitch import load_resume

            self.resume = load_resume(resume)
        self.ppu = PPUDeltaWriter(out / "ppu_state.bin", keyframe_interval,
                                  resume=bool(self.resume))
        self.sem = (out / "semantic.jsonl").open("a" if self.resume else "w", buffering=1)
        self.ledger = LedgerWriter(out / "ledger", resume=bool(self.resume))
        self.frames = 0

    def capture(self, runner) -> None:
        # W33 root-cause knobs: same-seed v1 (no sink) passed the rival house, v2
        # (slim sink) wedged -- capture_mode lets a differential pin which component:
        #   full (default) | ppu-only (no entities/semantic) | sem-only (no blob)
        env = runner.env
        mode = getattr(self, "capture_mode", "full")
        if mode == "sem-only":
            nav = runner.nav_state()
            self.sem.write(json.dumps({"frame": runner.frame_idx, "x": nav.x, "y": nav.y,
                                       "map": nav.map, "in_battle": nav.in_battle}) + "\n")
            self.frames += 1
            return
        ppu = extract_full_ppu_state(env)
        blob = serialize_ppu(ppu)                # serialized ONCE, shared by both writers
        self.ppu.add(runner.frame_idx, ppu, blob=blob)
        # W33 SLIM MODE (the pilot's catch): fast-record originally detached this WHOLE
        # sink, which silently dropped ppu_state.bin -- the sufficient statistic that
        # offline extraction and the gate's badge decode read. Six pilot runs recorded
        # unusable-as-corpus before gate_run caught it. The expensive part was only
        # ever the ~100-field read_ledger python extraction; the blob write + semantic
        # line must ALWAYS be recorded. skip_ledger keeps those and drops the rest --
        # labels come from collection/extract_ledger.py after the verify stage.
        if not getattr(self, "skip_ledger", False):
            # ledger reads the SAME captured blob (not the live env): row == stored
            # frame by construction, with the io/vram blocks the window-mask needs.
            self.ledger.add(runner.frame_idx, read_ledger(GBAState.from_blob(blob)))
        if mode == "ppu-only":
            self.frames += 1
            return
        nav = runner.nav_state()
        # objects + facing via the VALIDATED extractor (extractors.entities over the live seam).
        # Runs recorded before 2026-06 carry the legacy garbage objects and input-tracker facing
        # instead — schema_version marks the cut (pokemon-worldmodel docs/STRUCTURE.md §3); the A′ precompute never read
        # either field (it re-extracts from the PPU blobs), so old and new runs train identically.
        ents = entities(GBAState.from_env(env))
        player = next((e for e in ents if e.is_player), None)
        self.sem.write(json.dumps({
            "frame": runner.frame_idx, "x": nav.x, "y": nav.y,
            "facing": (player.facing if player and player.facing else runner.facing),
            "map": nav.map, "in_battle": nav.in_battle,
            "objects": [{"slot": e.slot, "graphics_id": e.graphics_id, "local_id": e.local_id,
                         "x": e.x, "y": e.y, "facing": e.facing} for e in ents if not e.is_player],
        }) + "\n")
        self.frames += 1

    def close(self) -> None:
        self.ppu.close()
        self.sem.close()
        self.ledger.close()
