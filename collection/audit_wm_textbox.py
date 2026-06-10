"""Phase-0 audit — per-frame TEXTBOX-VISIBLE flag, detected visually from RGB.

The known RAM dialogue addresses are stale for this build (DIALOG_STATE reads constant garbage), so
dialogue is detected from pixels: Emerald's message box is a bright, low-variance band across the
bottom of the screen (rows ~112-156). Validated against known segments (POKEDEX_DIALOG_CONFIRMED
≈ all dialogue; LITTLEROOT_TO_ROUTE101 ≈ none). Cached per run as `textbox.npy` (bool per frame).

This flag is what separates "standing in dialogue/cutscene" from FREE IDLE in the sufficiency
numbers, and gives the dialogue-mode mass for the coverage report.

Usage:
  .venv/bin/python -m collection.audit_wm_textbox --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from collection.audit_wm import discover_runs

# message-box interior band (excludes the border): rows 116..152, cols 16..224 at 240x160.
# Probed on real frames: dialogue interiors mean 223-247 (text glyphs RAISE std — no std test);
# overworld bands mean ~155. Threshold sits between.
Y0, Y1, X0, X1 = 116, 152, 16, 224
BRIGHT_MIN = 200.0


def textbox_flags(run_dir: Path) -> np.ndarray:
    cache = run_dir / "textbox.npy"
    if cache.exists():
        return np.load(cache)
    metas = [json.loads(l) for l in (run_dir / "frames.jsonl").open()]
    out = np.zeros(len(metas), bool)
    by_chunk: dict[str, list[tuple[int, int]]] = {}
    for i, m in enumerate(metas):
        by_chunk.setdefault(m["chunk"], []).append((i, m["chunk_frame_idx"]))
    for chunk, pairs in by_chunk.items():
        frames = np.load(run_dir / chunk)["frames"]
        idx = np.array([p[1] for p in pairs])
        band = frames[idx, Y0:Y1, X0:X1].astype(np.float32).mean(-1)     # (n, H', W') gray
        out[[p[0] for p in pairs]] = band.mean((1, 2)) > BRIGHT_MIN
    np.save(cache, out)
    return out


def _one(args: tuple[str, str, str]) -> str:
    name, d, kind = args
    f = textbox_flags(Path(d))
    return f"{kind}/{name}: {f.mean():6.1%} textbox ({f.sum()}/{len(f)})"


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
