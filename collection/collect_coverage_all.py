"""Run the unified coverage collector from a set of seed checkpoints, in parallel.

Each seed's run is cross-map and self-mapping, so it already covers that checkpoint's whole
connected world in one pass. Pick seeds with `--only` (e.g. one per story era) — running
from every checkpoint would re-cover overlapping worlds. The storyline half is still
`collect_events --world-model --chain`.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from collection.collect_coverage import collect_coverage


def _one(*, seed_id, load_state, output_dir, rom_path, backend, max_clips, max_tiles) -> dict:
    try:
        return collect_coverage(load_state=load_state, output_dir=str(Path(output_dir) / seed_id),
                                rom_path=rom_path, backend=backend, max_clips=max_clips,
                                max_tiles=max_tiles, story_bucket=seed_id)
    except Exception as exc:  # one bad seed must not sink the batch
        return {"story_bucket": seed_id, "error": str(exc), "graph_tiles": 0, "visited": 0,
                "coverage_pct": 0, "map_count": 0, "frames": 0, "clips": 0, "battles": 0, "ppu_bytes": 0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events-dir", required=True, help="collect_events output (checkpoints to seed from)")
    p.add_argument("--output", required=True)
    p.add_argument("--rom-path", default="Emerald-GBAdvance/rom.gba")
    p.add_argument("--backend", default="npz", choices=["auto", "ffv1", "npz"])
    p.add_argument("--max-clips", type=int, default=4096, help="safety cap; stall-break ends earlier")
    p.add_argument("--max-tiles", type=int, default=0, help="0 = whole connected world per seed")
    p.add_argument("--workers", type=int, default=0, help="0 = auto (cpu_count-2, capped at 16)")
    p.add_argument("--only", default=None, help="comma-separated subset of checkpoint ids (recommended)")
    args = p.parse_args()

    only = {e.strip() for e in args.only.split(",")} if args.only else None
    seeds = []
    for ed in sorted(Path(args.events_dir).iterdir()):
        if not ed.is_dir() or (only and ed.name not in only):
            continue
        ck = ed / "attempt_000001" / "final.state"
        if ck.exists():
            seeds.append((ed.name, str(ck)))
    if not seeds:
        raise SystemExit(f"No seed checkpoints found under {args.events_dir}")

    workers = args.workers or max(1, min(16, (os.cpu_count() or 4) - 2))
    print(f"covering from {len(seeds)} seeds with {workers} workers", flush=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, seed_id=sid, load_state=ck, output_dir=args.output, rom_path=args.rom_path,
                          backend=args.backend, max_clips=args.max_clips, max_tiles=args.max_tiles): sid
                for sid, ck in seeds}
        for fut in as_completed(futs):
            r = fut.result(); results.append(r)
            tag = f"ERROR {r['error']}" if r.get("error") else \
                f"{r['visited']:5d}/{r['graph_tiles']} tiles ({r['coverage_pct']}%), {r['map_count']:2d} maps, " \
                f"{r['clips']:3d} clips, {r['frames']:6d}f, {r['battles']} battles"
            print(f"  [{len(results):2d}/{len(seeds)}] {r['story_bucket']:32} {tag}", flush=True)

    totals = {"seeds": len(results), "total_frames": sum(r.get("frames", 0) for r in results),
              "total_visited": sum(r.get("visited", 0) for r in results),
              "errored": [r["story_bucket"] for r in results if r.get("error")]}
    (Path(args.output) / "coverage_all_summary.json").write_text(
        json.dumps({"totals": totals, "seeds": results}, indent=2, sort_keys=True))
    print(f"DONE: {totals['total_frames']} frames, {totals['total_visited']} tiles across {totals['seeds']} seeds", flush=True)


if __name__ == "__main__":
    main()
