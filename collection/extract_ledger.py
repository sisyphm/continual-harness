"""OFFLINE ledger extractor — rebuild a run's per-frame ledger from `ppu_state.bin`.

The live recorder (`collection.world_model_sink.WorldModelSink.capture`) writes the ledger
by calling `ledger_panel.read_ledger(GBAState.from_blob(blob))` on the SAME serialized PPU
blob it hands to `PPUDeltaWriter`. So the ledger is a pure function of the recorded condition
stream: replaying the delta chain and calling the identical row extractor reproduces it
byte-for-byte, with no frame-alignment ambiguity (row == stored frame by construction — there
is no mid-frame/post-frame skew to correct for).

This module replays that chain offline into `<run>/ledger_offline/`, using the SAME row
function, field set, dtypes, chunk size and file naming as the live sink (the chunk writer
subclasses `LedgerWriter`, so `index.json` and the npz layout come from the live code path).

    PYTHONPATH=. .venv/bin/python -m collection.extract_ledger --run <dir> \
        [--start F --end F] [--workers N] [--validate]

`--start/--end` are emulator frame indices, half-open [start, end); default = the whole run.
`--workers N` splits the record stream at KEYFRAME boundaries (index tag 'K'), so every worker
decodes from a self-contained keyframe and the XOR chain is never crossed between processes.
`--validate` compares `ledger_offline/` against the live `ledger/` field by field, frame by
frame, over the extracted range; exit 0 only when every field matches on every compared frame.
"""

from __future__ import annotations

import argparse
import json
import mmap
import multiprocessing as mp
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from collection.extractors.ledger_panel import FIELDS, read_ledger
from collection.extractors.ram import BLOB_SIZE, GBAState
from collection.world_model_sink import LedgerWriter

CHUNK_FRAMES = 4096                 # == LedgerWriter's default; keeps chunk boundaries aligned
OUT_DIRNAME = "ledger_offline"
KEYFRAME_TAG = 0x4B                 # b"K"

# ---------------------------------------------------------------------------
# index / decode
# ---------------------------------------------------------------------------


def parse_index(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`ppu_state.bin.idx.json` -> (frame_idx, byte_offset, is_keyframe) arrays, record order."""
    frames = json.loads((run / "ppu_state.bin.idx.json").read_text())["frames"]
    n = len(frames)
    return (np.fromiter((r[0] for r in frames), np.int64, n),
            np.fromiter((r[1] for r in frames), np.int64, n),
            np.fromiter((r[2] == "K" for r in frames), np.bool_, n))


# worker-side state: set in the parent BEFORE the fork pool starts, inherited copy-on-write
_BIN_PATH: str | None = None
_OFFS: np.ndarray | None = None
_FRAMES: np.ndarray | None = None
_MM: mmap.mmap | None = None
_FH = None                          # keep the handle alive: mmap needs the fd to stay open


def _mm() -> mmap.mmap:
    global _MM, _FH
    if _MM is None:
        _FH = open(_BIN_PATH, "rb")
        _MM = mmap.mmap(_FH.fileno(), 0, access=mmap.ACCESS_READ)
    return _MM


def _decode_span(task: tuple[int, int, int]) -> tuple[int, np.ndarray, dict]:
    """Emit ledger rows for records [lo, hi), decoding the XOR chain from record `dec`
    (the nearest preceding keyframe). Returns (lo, frame_idx, {field: array})."""
    lo, hi, dec = task
    mm, n = _mm(), hi - lo
    cols = {name: np.zeros((n, *shape), dtype=dt) for name, (dt, shape) in FIELDS.items()}
    cur: np.ndarray | None = None
    for i in range(dec, hi):
        off = int(_OFFS[i])
        ln = int.from_bytes(mm[off + 1:off + 5], "little")
        payload = np.frombuffer(zlib.decompress(mm[off + 5:off + 5 + ln]), np.uint8)
        cur = payload.copy() if mm[off] == KEYFRAME_TAG else np.bitwise_xor(cur, payload)
        if len(cur) != BLOB_SIZE:
            raise ValueError(f"record {i}: blob {len(cur)} != {BLOB_SIZE}")
        if i >= lo:
            row = read_ledger(GBAState.from_blob(cur))
            j = i - lo
            for name in FIELDS:
                cols[name][j] = row[name]
    return lo, _FRAMES[lo:hi].astype(np.uint32), cols


def plan_tasks(is_key: np.ndarray, r0: int, r1: int, ntasks: int) -> list[tuple[int, int, int]]:
    """Split records [r0, r1) into <= ntasks spans that START on keyframes (except the first,
    which decodes from its nearest preceding keyframe)."""
    keys = np.nonzero(is_key)[0]
    first_dec = int(keys[np.searchsorted(keys, r0, "right") - 1]) if len(keys) and keys[0] <= r0 else 0
    if ntasks <= 1:
        return [(r0, r1, first_dec)]
    inner = keys[(keys > r0) & (keys < r1)]                    # candidate split points
    if len(inner) == 0:
        return [(r0, r1, first_dec)]
    want = min(ntasks - 1, len(inner))
    picks = sorted({int(inner[i]) for i in
                    np.linspace(0, len(inner) - 1, want).round().astype(int)})
    bounds = [r0, *picks, r1]
    return [(bounds[i], bounds[i + 1], bounds[i] if i else first_dec)
            for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]


# ---------------------------------------------------------------------------
# chunk writer (live format, batch input)
# ---------------------------------------------------------------------------


class BatchLedgerWriter(LedgerWriter):
    """`LedgerWriter` with an array-batch input path. Inherits `_write_index` verbatim and
    mirrors `_flush`'s array construction, so chunk naming/contents/index.json are the live
    sink's format; only the buffering differs (whole spans instead of one row at a time)."""

    def __init__(self, out_dir: str | Path, chunk_frames: int = CHUNK_FRAMES):
        super().__init__(out_dir, chunk_frames)
        self._pend_f: list[np.ndarray] = []
        self._pend: dict[str, list[np.ndarray]] = {name: [] for name in FIELDS}
        self._n = 0

    def add_batch(self, frame_idx: np.ndarray, cols: dict) -> None:
        self._pend_f.append(frame_idx)
        for name in FIELDS:
            self._pend[name].append(cols[name])
        self._n += len(frame_idx)
        if self._n < self.chunk_frames:
            return
        f = np.concatenate(self._pend_f)
        c = {name: np.concatenate(self._pend[name]) for name in FIELDS}
        i = 0
        while self._n - i >= self.chunk_frames:
            j = i + self.chunk_frames
            self._emit(f[i:j], {name: c[name][i:j] for name in FIELDS})
            i = j
        self._pend_f = [f[i:]]
        self._pend = {name: [c[name][i:]] for name in FIELDS}
        self._n -= i

    def _emit(self, frame_idx: np.ndarray, cols: dict) -> None:
        n = len(frame_idx)
        arrs = {"frame_idx": np.asarray(frame_idx, np.uint32)}
        for name, (dt, shape) in FIELDS.items():
            arrs[name] = np.asarray(cols[name], dtype=dt).reshape((n, *shape))
        fn = f"chunk_{len(self.chunks):06d}.npz"
        np.savez_compressed(self.dir / fn, **arrs)
        self.chunks.append([fn, self.total, n])
        self.total += n
        self._write_index()

    def close(self) -> None:
        if self._n:
            self._emit(np.concatenate(self._pend_f),
                       {name: np.concatenate(self._pend[name]) for name in FIELDS})
            self._pend_f, self._pend, self._n = [], {name: [] for name in FIELDS}, 0
        self._write_index()


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def extract(run: Path, start: int | None, end: int | None, workers: int,
            out_base: Path | None = None) -> tuple[int, float]:
    """Rebuild [start, end) into <run>/ledger_offline. Returns (n_frames, seconds)."""
    global _BIN_PATH, _OFFS, _FRAMES
    frames, offs, is_key = parse_index(run)
    r0 = 0 if start is None else int(np.searchsorted(frames, start, "left"))
    r1 = len(frames) if end is None else int(np.searchsorted(frames, end, "left"))
    if r1 <= r0:
        raise SystemExit(f"empty range: frames [{start}, {end}) matches no records")
    _BIN_PATH, _OFFS, _FRAMES = str(run / "ppu_state.bin"), offs, frames

    workers = max(1, workers)
    tasks = plan_tasks(is_key, r0, r1, workers * 4 if workers > 1 else 1)
    print(f"[extract] records [{r0}, {r1}) = frames [{frames[r0]}, {frames[r1 - 1]}] "
          f"({r1 - r0} frames), {len(tasks)} spans, {workers} workers")

    out_root = (out_base / run.name) if out_base is not None else run
    out = BatchLedgerWriter(out_root / OUT_DIRNAME)
    t0 = time.time()
    if workers == 1 or len(tasks) == 1:
        for t in tasks:
            _, fidx, cols = _decode_span(t)
            out.add_batch(fidx, cols)
    else:
        # bounded in-flight window: results must be consumed IN ORDER (chunk assembly is
        # sequential), and unbounded submission would buffer the whole run in RAM
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            pending, nxt = [], 0
            while nxt < len(tasks) and len(pending) < workers * 2:
                pending.append(ex.submit(_decode_span, tasks[nxt])); nxt += 1
            while pending:
                _, fidx, cols = pending.pop(0).result()
                out.add_batch(fidx, cols)
                if nxt < len(tasks):
                    pending.append(ex.submit(_decode_span, tasks[nxt])); nxt += 1
    out.close()
    dt = time.time() - t0
    n = r1 - r0
    print(f"[extract] {n} frames -> {out.dir} ({len(out.chunks)} chunks) "
          f"in {dt:.1f}s = {n / dt:,.0f} frames/sec")
    return n, dt


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _load_ledger_index(d: Path) -> tuple[list, np.ndarray]:
    """(chunks list, concatenated frame_idx) for a ledger dir."""
    chunks = json.loads((d / "index.json").read_text())["chunks"]
    fidx = [np.load(d / fn)["frame_idx"] for fn, _, _ in chunks]
    return chunks, (np.concatenate(fidx) if fidx else np.zeros(0, np.uint32))


def _gather_live(live: Path, chunks: list, lo: int, hi: int) -> dict:
    """All fields for live global rows [lo, hi), loading only the chunks that span them."""
    parts: dict[str, list] = {name: [] for name in FIELDS}
    for fn, start, n in chunks:
        if start >= hi or start + n <= lo:
            continue
        z = np.load(live / fn)
        a, b = max(lo, start) - start, min(hi, start + n) - start
        for name in FIELDS:
            parts[name].append(z[name][a:b])
    return {name: np.concatenate(parts[name]) for name in FIELDS}


def validate(run: Path) -> bool:
    """Field-by-field, frame-by-frame comparison of ledger_offline/ against the live ledger/."""
    live_d, off_d = run / "ledger", run / OUT_DIRNAME
    if not (live_d / "index.json").exists():
        raise SystemExit(f"no live ledger to validate against: {live_d}")
    live_chunks, live_f = _load_ledger_index(live_d)
    off_chunks, off_f = _load_ledger_index(off_d)

    pos = np.searchsorted(live_f, off_f)
    missing = (pos >= len(live_f)) | (live_f[np.minimum(pos, len(live_f) - 1)] != off_f)
    n_missing = int(missing.sum())

    mism = {name: 0 for name in FIELDS}
    compared = 0
    row0 = 0
    for fn, _start, n in off_chunks:
        z = np.load(off_d / fn)
        sl = slice(row0, row0 + n)
        row0 += n
        keep = ~missing[sl]
        if not keep.any():
            continue
        p = pos[sl][keep]
        live = _gather_live(live_d, live_chunks, int(p.min()), int(p.max()) + 1)
        take = p - int(p.min())
        for name in FIELDS:
            a, b = z[name][keep], live[name][take]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise SystemExit(f"schema drift on {name}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
            diff = a != b
            mism[name] += int(diff.any(axis=tuple(range(1, diff.ndim))).sum() if diff.ndim > 1
                              else diff.sum())
        compared += int(keep.sum())

    w = max(len(n) for n in FIELDS)
    print(f"\n[validate] {compared} frames compared, {len(FIELDS)} fields "
          f"(frames {off_f[0]}..{off_f[-1]})")
    print(f"  {'field'.ljust(w)}  mismatched_frames")
    for name in FIELDS:
        print(f"  {name.ljust(w)}  {mism[name]}")
    total = sum(mism.values())
    print(f"  {'TOTAL'.ljust(w)}  {total}")
    if n_missing:
        print(f"  ({n_missing} extracted frames absent from the live ledger)")
    ok = total == 0 and n_missing == 0 and compared > 0
    print("\nALL FIELDS MATCH" if ok else "\nMISMATCH")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--start", type=int, default=None, help="first emulator frame index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last emulator frame index (exclusive)")
    ap.add_argument("--out", default=None, help="write <out>/<run_name>/ledger_offline instead of into the run dir (frozen corpora)")
    ap.add_argument("--workers", type=int, default=1, help="processes; split at keyframe boundaries")
    ap.add_argument("--validate", action="store_true", help="diff ledger_offline/ vs live ledger/")
    a = ap.parse_args(argv)
    if not (a.run / "ppu_state.bin.idx.json").exists():
        raise SystemExit(f"not a recording: {a.run}")
    extract(a.run, a.start, a.end, a.workers,
            out_base=Path(a.out) if a.out else None)
    return 0 if (not a.validate or validate(a.run)) else 1


if __name__ == "__main__":
    sys.exit(main())
