"""Run the A′ condition precompute over every recorded run (parallel, one npz per run).

Usage:
  .venv/bin/python -m collection.extractors.precompute_conditions \
      --data_root ../pokemon-worldmodel/data --out_dir ../pokemon-worldmodel/data/processed/conditions
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

from collection.corpus import discover_runs, write_manifest
from collection.extractors.conditions import precompute_run
from collection.extractors.text import export_charset

ROM = Path(__file__).resolve().parents[2] / "Emerald-GBAdvance/rom.gba"


def _one(args: tuple[str, str, str]) -> str:
    run_dir, out, _name = args
    try:
        return "  " + precompute_run(run_dir, out, ROM.read_bytes())
    except Exception as e:                                  # noqa: BLE001 — surface, don't kill the pool
        return f"  FAILED {_name}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out_dir", default="../pokemon-worldmodel/data/processed/conditions")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    export_charset(out_dir / "charset.json")               # the charset's cross-repo home
    write_manifest(Path(args.data_root))                   # the corpus as the model repo consumes it
    todo = []
    for name, d, kind in discover_runs(Path(args.data_root)):
        out = out_dir / f"{kind}__{name}.npz"
        if not out.exists():
            todo.append((str(d), str(out), f"{kind}__{name}"))
    print(f"{len(todo)} runs to precompute…")
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(_one, todo):
            print(msg, flush=True)
    print("done")


if __name__ == "__main__":
    main()
