"""Phase-2 collection wave — schedule the behavior jobs over the savestate bank, sized to the
deficit shopping list (PHASE0 doc §3), and run them in parallel emulator processes.

Savestate bank = the storyline segments' final.state files (varied story flags/locations/party).
Job→state matching: battle jobs seed at grass-adjacent segments; dialogue/menus at towns and
interiors; idle/fidget everywhere. Targets (frames @60fps):

  idle     ~50k   S1     fidget  ~70k   S2-S4     battle  ~220k  S5-S7
  menus    ~40k   S8     dialogue ~50k  S9

Usage:
  CUDA_VISIBLE_DEVICES= .venv/bin/python -m collection.collect_behaviors_all \
      --data_root ../pokemon-worldmodel/data --out_root ../pokemon-worldmodel/data/behaviors
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from collection.collect_behaviors import collect_behavior

# segment-name patterns with grass routes nearby (battle hunting grounds)
GRASSY = ("ROUTE101", "ROUTE_101", "ROUTE_102", "ROUTE_103", "ROUTE101_AFTER_POKEDEX",
          "ROUTE101_TO_OLDALE_FIRST_TIME", "ROUTE_104_NORTH", "ROUTE_104_SOUTH",
          "PETALBURG_WOODS", "BACK_TO_OLDALE_FROM_ROUTE103", "BACK_TO_ROUTE101_FROM_OLDALE")
# towny/indoor segments (NPCs + signs + menus make sense anywhere)
TOWNY = ("LITTLEROOT_TOWN", "OLDALE_TOWN", "PETALBURG_CITY", "RUSTBORO_CITY", "RIVAL_HOUSE",
         "PLAYER_HOUSE_ENTERED", "BIRCH_LAB_VISITED", "ENTER_PETALBURG_CENTER",
         "RUSTBORO_CENTER_ENTERED", "GYM_CUTSCENE_OUTSIDE", "EXIT_RIVAL_HOUSE",
         "HEAL_AT_PETALBURG_CENTER", "HEAL_AT_RUSTBORO_CENTER")

PLAN = (        # (job, state-pool, runs, frames-per-run)
    ("idle",     TOWNY + GRASSY, 10, 5000),
    ("fidget",   TOWNY + GRASSY, 12, 6000),
    ("battle",   GRASSY,         12, 18000),
    ("menus",    TOWNY,          8,  5000),
    ("dialogue", TOWNY,          8,  6000),
    # top-up wave (post-re-audit): S2 running volume + extra turn/bump texture
    ("run",      GRASSY + TOWNY, 10, 6000),
    ("fidget",   GRASSY,         6,  6000),
)


def _bank(data_root: Path) -> dict[str, Path]:
    out = {}
    for seg in sorted((data_root / "storyline_wm").iterdir()):
        st = seg / "attempt_000001" / "final.state"
        if st.exists():
            out[seg.name] = st
    return out


def _one(args: tuple) -> str:
    job, state, out, rom, frames, seed = args
    try:
        s = collect_behavior(job=job, load_state=state, output_dir=out, rom_path=rom,
                             frames=frames, seed=seed)
        return f"  {Path(out).name}: {s['frames']} frames"
    except Exception as e:                                  # noqa: BLE001
        return f"  FAILED {Path(out).name}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out_root", default="../pokemon-worldmodel/data/behaviors")
    ap.add_argument("--rom_path", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    bank = _bank(Path(args.data_root))
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    seed = 0
    job_counter: dict[str, int] = {}
    for job, pool, runs, frames in PLAN:
        states = [s for s in pool if s in bank]
        for _ in range(runs):
            i = job_counter.get(job, 0)
            job_counter[job] = i + 1
            seg = states[i % len(states)]
            out = out_root / f"{job}__{seg}__{i:02d}"
            if (out / "behavior_summary.json").exists():
                continue
            jobs.append((job, str(bank[seg]), str(out), args.rom_path, frames, seed))
            seed += 1
    total = sum(j[4] for j in jobs)
    print(f"{len(jobs)} runs scheduled (~{total:,} frames)…")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for f in as_completed(futs):
            print(f.result(), flush=True)
    print("done")


if __name__ == "__main__":
    main()
