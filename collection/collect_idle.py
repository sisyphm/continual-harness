"""Idle-gap closer: run job_idle over the constructed-state bank so the corpus finally contains
"player paused, world holds static" frames (only 0.9% of the corpus before this — the biggest
behavioral gap; a human pauses constantly). job_idle relocates a short burst then stands still
through varied facings/durations, so we get idle frames at DIVERSE positions/maps, not one spot.

Usage:
  CUDA_VISIBLE_DEVICES= .venv/bin/python -m collection.collect_idle \
      --data_root ../pokemon-worldmodel/data [--workers 8] [--frames 9000]
"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from collection.collect_behaviors import collect_behavior


def _one(args: tuple) -> str:
    name, state, frames, out, rom, seed = args
    try:
        s = collect_behavior(job="idle", load_state=state, output_dir=out, rom_path=rom,
                             frames=frames, seed=seed)
        return f"OK   {name}: {s['frames']} frames"
    except Exception as e:                                   # noqa: BLE001 — surface, don't kill pool
        traceback.print_exc()
        return f"FAIL {name}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frames", type=int, default=9000)
    ap.add_argument("--wave", type=int, default=1)
    args = ap.parse_args()
    root = Path(args.data_root)
    bank = json.loads((root / "processed/state_bank/bank.json").read_text())

    todo = []
    for i, b in enumerate(bank["bases"]):
        out = root / "behaviors" / f"idle__{b['name']}__w{args.wave}_{i:02d}"
        if (out / "behavior_summary.json").exists():
            continue
        todo.append((b["name"], b["state"], args.frames, str(out), args.rom,
                     40000 + i + 1000 * args.wave))
    print(f"{len(todo)} idle runs to collect…")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, t) for t in todo]
        for f in as_completed(futs):
            print(" ", f.result(), flush=True)
    print("done — idle gap collection complete")


if __name__ == "__main__":
    main()
