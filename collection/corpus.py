"""Corpus layout — the one place that knows where collected runs live and how to enumerate them.

The world-model corpus is three families of run directories under the model repo's `data/` root
(each run = RGB chunks + `frames.jsonl` + `actions.jsonl` + `semantic.jsonl` + `ppu_state.bin[.idx]`):

    storyline_wm/<segment>/attempt_*/    scripted start→badge-1 playthrough segments
    coverage_dataset/<seed>/             exploration walker runs
    behaviors/<run>/                     targeted behavior jobs (idle/fidget/battle/menus/dialogue/run)

Everything that walks the corpus — condition precompute, the audit chain, the model repo's clip
indexer — goes through `discover_runs` (directly, or via the manifest `write_manifest` emits into
`data/processed/`, which is the cross-repo interface so the model repo never re-implements
discovery). Schema details: docs/SCHEMA.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = 2          # version of the recorded-run/conditions schema (see docs/SCHEMA.md)


def discover_runs(data_root: Path) -> list[tuple[str, Path, str]]:
    """All audit-able runs: (name, dir, kind) — storyline attempts + coverage seeds + behavior runs."""
    runs = []
    for seg in sorted((data_root / "storyline_wm").iterdir()):
        for att in sorted(seg.glob("attempt_*")):
            if (att / "semantic.jsonl").exists():
                runs.append((seg.name, att, "storyline"))
    for seed in sorted((data_root / "coverage_dataset").iterdir()):
        if seed.is_dir() and (seed / "semantic.jsonl").exists():
            runs.append((seed.name, seed, "coverage"))
    beh = data_root / "behaviors"
    if beh.exists():
        for d in sorted(beh.iterdir()):
            if (d / "semantic.jsonl").exists():
                runs.append((d.name, d, "behavior"))
    return runs


def clip_starts(run_dir: Path) -> set[int]:
    """Reset/teleport frame indices (clips must not span them); absent summary -> none."""
    p = run_dir / "coverage_summary.json"
    return set(json.loads(p.read_text()).get("clip_start_frames", [])) if p.exists() else set()


def write_manifest(data_root: Path, out: Path | None = None) -> Path:
    """Emit `data/processed/corpus_manifest.json` — the corpus as the model repo consumes it.
    run_key == the conditions npz stem (`<kind>__<name>`, the precompute_conditions convention).
    A storyline segment with >1 attempt would collide under that convention — refuse loudly
    rather than silently shadowing (today every segment has exactly attempt_000001)."""
    runs, seen = [], set()
    for name, d, kind in discover_runs(data_root):
        key = f"{kind}__{name}"
        if key in seen:
            raise SystemExit(f"run_key collision: {key} ({d}) — multiple attempts per segment "
                             f"need a naming extension in precompute_conditions first")
        seen.add(key)
        n_frames = sum(1 for _ in (d / "semantic.jsonl").open())
        runs.append({"run_key": key, "kind": kind, "name": name,
                     "dir": str(d.resolve().relative_to(data_root.resolve())),   # portable: rel to data root
                     "frames": n_frames, "clip_starts": sorted(clip_starts(d))})
    out = out or (data_root / "processed" / "corpus_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": SCHEMA_VERSION,
                               "total_frames": sum(r["frames"] for r in runs),
                               "runs": runs}, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="write the corpus manifest (the cross-repo interface)")
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = write_manifest(Path(a.data_root), Path(a.out) if a.out else None)
    print(f"wrote {p}")
