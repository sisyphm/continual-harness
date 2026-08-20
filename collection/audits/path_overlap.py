"""W33 §14 diversity audit: path overlap between two runs — the pilot's diversity
NUMBER (owner: same-starter scripted runs must show measurable divergence, "a number,
not a vibe").

Metric: the set of distinct (map, x, y) tiles each run's recorded states visited,
restricted (by default) to SPINE frames — phases.jsonl gives the block-phase frame
ranges; frames before any transition, and ranges tagged None/"spine", count as spine
(blocks legitimately share sweeps, so including them would understate spine
divergence). Reported:

  overlap_jaccard   |A ∩ B| / |A ∪ B|      — the headline number
  shared_frac_a/b   |A ∩ B| / |A| (resp |B|) — per-run "fraction of shared visits"

Reads states.jsonl (every recorded run has one; rows carry frame_idx/map/x/y).

CLI:
  .venv/bin/python -m collection.audits.path_overlap RUN_A RUN_B [--all-phases]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _spine_ranges(run_dir: Path) -> list[tuple[int, float, str | None]] | None:
    """[(start_frame, end_frame, phase), ...] from phases.jsonl; None if absent."""
    p = run_dir / "phases.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.open()]
    out: list[tuple[int, float, str | None]] = []
    prev_frame, prev_phase = 0, None            # before the first transition: spine
    for r in rows:
        out.append((prev_frame, r["frame_idx"], prev_phase))
        prev_frame, prev_phase = r["frame_idx"], r["phase"]
    out.append((prev_frame, float("inf"), prev_phase))
    return out


def _is_spine(frame: int, ranges) -> bool:
    if ranges is None:
        return True
    for start, end, phase in ranges:
        if start <= frame < end:
            return phase in (None, "spine")
    return True


def visited_tiles(run_dir: str | Path, *, spine_only: bool = True) -> set[tuple]:
    """Distinct (map, x, y) visited by a run's recorded state rows."""
    run_dir = Path(run_dir)
    ranges = _spine_ranges(run_dir) if spine_only else None
    tiles: set[tuple] = set()
    with (run_dir / "states.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("map") is None or r.get("x") is None or r.get("y") is None:
                continue
            if spine_only and not _is_spine(int(r.get("frame_idx", 0)), ranges):
                continue
            tiles.add((r["map"], r["x"], r["y"]))
    return tiles


def path_overlap(run_a: str | Path, run_b: str | Path, *, spine_only: bool = True) -> dict:
    a = visited_tiles(run_a, spine_only=spine_only)
    b = visited_tiles(run_b, spine_only=spine_only)
    inter, union = a & b, a | b
    return {
        "run_a": str(run_a), "run_b": str(run_b), "spine_only": spine_only,
        "tiles_a": len(a), "tiles_b": len(b),
        "tiles_shared": len(inter),
        "overlap_jaccard": (len(inter) / len(union)) if union else 1.0,
        "shared_frac_a": (len(inter) / len(a)) if a else 1.0,
        "shared_frac_b": (len(inter) / len(b)) if b else 1.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--all-phases", action="store_true",
                    help="include block frames (default: spine frames only)")
    args = ap.parse_args()
    print(json.dumps(path_overlap(args.run_a, args.run_b,
                                  spine_only=not args.all_phases), indent=2))


if __name__ == "__main__":
    main()
