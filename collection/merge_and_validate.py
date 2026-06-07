"""Merge systematic collection shards into a final manifest and coverage report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield {"error": "invalid_json", "raw": line[:500]}


def merge(input_dir: str | Path, output_dir: str | Path) -> dict:
    root = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    segment_files = sorted(root.glob("**/segments.jsonl"))
    validation_files = sorted(root.glob("**/validation.json"))
    coverage_files = sorted(root.glob("**/coverage.json"))
    segments = []
    for path in segment_files:
        for row in _iter_jsonl(path) or []:
            row["source_segments_file"] = str(path.relative_to(root))
            segments.append(row)
    validations = [_read_json(path) for path in validation_files]
    coverages = [_read_json(path) for path in coverage_files]
    validations = [v for v in validations if isinstance(v, dict)]
    coverages = [c for c in coverages if isinstance(c, dict)]
    coverage = {
        "segment_count": len(segments),
        "event_count": sum(1 for row in segments if row.get("segment_type") == "event"),
        "explore_transition_count": sum(1 for row in segments if row.get("segment_type") == "explore_transition"),
        "passed_segment_count": sum(1 for row in segments if row.get("validation") == "passed"),
        "failed_segment_count": sum(1 for row in segments if row.get("validation") == "failed"),
        "validation_files": len(validations),
        "coverage_files": len(coverages),
    }
    with (out / "segments.jsonl").open("w", encoding="utf-8") as f:
        for row in segments:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {"input": str(root), "segment_files": [str(p.relative_to(root)) for p in segment_files], "coverage": coverage}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.input, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
