"""Phase-0 audit — cb2 contact sheets: one image grid per distinct gMain.callback2 value, so each
screen type can be labeled by INSPECTION (the empirical mode taxonomy — no trusting symbol lists).

Reads the per-run RAM field files (audits.ram_fields) + the RGB chunks, samples up to `--per` frames per
cb2 value (spread across runs), and writes `cb2_<value>__<count>f.png` sheets plus `cb2_index.json`
(value -> frames, runs, battle/dark fractions, sampled frame refs).

Usage:
  .venv/bin/python -m collection.audits.modes --data_root ../pokemon-worldmodel/data \
      --audit_dir ../pokemon-worldmodel/data/processed/audit
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from collection.corpus import discover_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--audit_dir", default="../pokemon-worldmodel/data/processed/audit")
    ap.add_argument("--per", type=int, default=8, help="sample frames per cb2 value")
    args = ap.parse_args()

    root, audit = Path(args.data_root), Path(args.audit_dir)
    sheets = audit / "cb2_sheets"; sheets.mkdir(parents=True, exist_ok=True)
    run_dirs = {f"{kind}__{name}": d for name, d, kind in discover_runs(root)}

    # pass 1: where does each cb2 value occur (run, position-in-run)?
    occ: dict[int, list[tuple[str, int, int]]] = defaultdict(list)   # cb2 -> [(runkey, i, fidx)]
    count: dict[int, int] = defaultdict(int)
    for f in sorted((audit / "ram").glob("*.v2.npz")):
        runkey = f.name[:-len(".v2.npz")]
        z = np.load(f)
        cb2, fidx = z["cb2"], z["fidx"]
        for v in np.unique(cb2):
            idxs = np.flatnonzero(cb2 == v)
            count[int(v)] += len(idxs)
            for j in idxs[np.linspace(0, len(idxs) - 1, min(4, len(idxs)), dtype=int)]:
                occ[int(v)].append((runkey, int(j), int(fidx[j])))

    # pass 2: pull sampled RGB frames (group loads per chunk to avoid reloading npz files)
    index = {}
    for v, samples in sorted(occ.items(), key=lambda kv: -count[kv[0]]):
        picks = samples[:: max(1, len(samples) // args.per)][: args.per]
        tiles = []
        for runkey, _, fidx in picks:
            d = run_dirs[runkey]
            metas = {json.loads(l)["emulator_frame_idx"]: json.loads(l) for l in (d / "frames.jsonl").open()}
            m = metas[fidx]
            frames = np.load(d / m["chunk"])["frames"]
            tiles.append(frames[m["chunk_frame_idx"]])
        rows = [np.concatenate(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])), axis=1)
                for i in range(0, len(tiles), 4)]
        sheet = np.concatenate(rows, axis=0)
        name = f"cb2_{v:#x}__{count[v]}f.png"
        Image.fromarray(sheet).save(sheets / name)
        index[f"{v:#x}"] = {"frames": count[v], "sheet": name,
                            "samples": [(rk, fi) for rk, _, fi in picks]}
        print(f"  {v:#10x} {count[v]:7d} frames -> {name}")

    (audit / "cb2_index.json").write_text(json.dumps(index, indent=1))
    print(f"\n{len(index)} cb2 values; sheets in {sheets}")


if __name__ == "__main__":
    main()
