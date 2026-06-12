"""Data-closure Phase C: the closure collection wave — bank-seeded jobs driven by the red rows.

Schedules `collect_behaviors` jobs over the constructed-state bank (`build_state_bank`):

  species_* entries  -> battle job: hunts wild encounters with the constructed lead — every
                        battle farms player-side frames for a species the corpus never had
  evolve_*  entries  -> battle job: the lead is ONE FIGHT from leveling; a won battle plays the
                        level-up + evolution cutscene (outcome axis: evolution, currently 0)
  catch_*   entries  -> battle job: full ball pocket + the existing catch strategy mix — catches
                        finally succeed (outcome axis: catch, currently 0); caught mons join the
                        party and further diversify player-side species

Runs record through the SAME substrate as all corpus data (recorder + WorldModelSink), land in
data/behaviors/closure__*, and are picked up by the standard corpus pipeline (manifest →
conditions → index). Re-run `audits.coverage` afterwards; iterate until green.

Usage:
  CUDA_VISIBLE_DEVICES= .venv/bin/python -m collection.collect_closure \
      --data_root ../pokemon-worldmodel/data [--limit N] [--workers 10]
"""

from __future__ import annotations

import argparse
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from collection.collect_behaviors import collect_behavior

FRAMES = {"battle": 9000, "evolve": 11000, "catch": 12000}


def _one(args: tuple) -> str:
    name, state, frames, out, rom, seed, job = args
    try:
        s = collect_behavior(job=job, load_state=state, output_dir=out, rom_path=rom,
                             frames=frames, seed=seed)
        return f"OK   {name}: {s['frames']} frames"
    except Exception as e:                                   # noqa: BLE001 — surface, don't kill the pool
        traceback.print_exc()
        return f"FAIL {name}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="schedule at most N runs (pilot)")
    ap.add_argument("--wave", type=int, default=1, help="run-dir suffix wave (re-runs entries)")
    ap.add_argument("--frames", type=int, default=0, help="override frames per run (pilot)")
    args = ap.parse_args()
    root = Path(args.data_root)
    bank = json.loads((root / "processed/state_bank/bank.json").read_text())

    todo = []
    for i, e in enumerate(bank["entries"]):
        out = root / "behaviors" / f"closure__{e['name']}__w{args.wave}_{i:02d}"
        if (out / "behavior_summary.json").exists():
            continue
        frames = args.frames or FRAMES[e["intent"]]
        job = "battle_catch" if e["intent"] == "catch" else "battle"
        todo.append((e["name"], e["state"], frames, str(out), args.rom, 1000 + i, job))
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} closure runs to collect…")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, t) for t in todo]
        for f in as_completed(futs):
            print(" ", f.result(), flush=True)
    print("done — re-run audits.coverage and iterate to green")


if __name__ == "__main__":
    main()
