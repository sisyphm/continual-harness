"""Phase-0 audit — per-frame RGB CHANGE magnitude (mean |Δ| vs previous frame), cached per run as
`framediff.npy`. Feeds the unexplained-dynamics detector: frames where pixels change but the
schema-v0 condition doesn't = the learned tail (or missing state), made measurable.

Usage:
  .venv/bin/python -m collection.audits.framediff --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from collection.corpus import discover_runs


def framediff_of(run_dir: Path) -> np.ndarray:
    cache = run_dir / "framediff.npy"
    if cache.exists():
        return np.load(cache)
    metas = [json.loads(l) for l in (run_dir / "frames.jsonl").open()]
    out = np.zeros(len(metas), np.float32)                       # [0] stays 0 (no previous)
    prev_last = None                                             # last frame of the previous chunk
    # chunks are consecutive; iterate in order so the cross-chunk diff is correct
    chunks: dict[str, list[tuple[int, int]]] = {}
    for i, m in enumerate(metas):
        chunks.setdefault(m["chunk"], []).append((i, m["chunk_frame_idx"]))
    for chunk in sorted(chunks, key=lambda c: chunks[c][0][0]):
        pairs = chunks[chunk]
        fr = np.load(run_dir / chunk)["frames"].astype(np.int16)
        d = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2, 3))     # (N-1,)
        idx0 = pairs[0][0]
        for i, ci in pairs:
            if ci > 0:
                out[i] = d[ci - 1]
            elif prev_last is not None and i > 0:
                out[i] = np.abs(fr[0].astype(np.int16) - prev_last).mean()
        prev_last = fr[-1]
    np.save(cache, out)
    return out


def _one(args: tuple[str, str, str]) -> str:
    name, d, kind = args
    f = framediff_of(Path(d))
    return f"{kind}/{name}: mean|Δ|={f.mean():.2f}  p95={np.percentile(f, 95):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    runs = [(n, str(d), k) for n, d, k in discover_runs(Path(args.data_root))]
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(_one, runs):
            print(" ", msg)


if __name__ == "__main__":
    main()
