"""Data-closure: OUTCOME accounting — catches and evolutions measured from the runs themselves.

The conditions corpus cannot see party composition, so `audits.coverage` carried hardcoded zeros
for catch/evolution rows. This audit walks each run's recorded state chain and measures the truth:
a CATCH is party growth across the run; an EVOLUTION is the lead species changing (constructed
evolve states change exactly at the level-up cutscene). Results are cached per run
(`audit/outcomes/<run>.json`) and folded into one report that `audits.coverage` consumes.

Usage:
  .venv/bin/python -m collection.audits.outcomes --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from collection.constructor import decrypt_box
from collection.corpus import discover_runs
from collection.extractors.ram import iter_states

G_PLAYER_PARTY = 0x020244EC


def _party(st) -> list[int]:
    out = []
    for i in range(6):
        try:
            d = decrypt_box(st.bytes(G_PLAYER_PARTY + i * 100, 80))
        except Exception:
            break
        if not (d["checksum_ok"] and 0 < d["species"] < 412):
            break
        out.append(d["species"])
    return out


def _one(args: tuple[str, str, str]) -> tuple[str, dict]:
    name, run_dir, cache = args
    c = Path(cache)
    if c.exists():
        return name, json.loads(c.read_text())
    first = last_st = None
    for _f, st in iter_states(run_dir):
        if first is None:
            first = _party(st)
        last_st = st
    last = _party(last_st) if last_st is not None else []
    r = {"party_first": first or [], "party_last": last,
         "catches": max(0, len(last) - len(first or [])),
         "evolution": bool(first and last and first[0] != last[0])}
    c.parent.mkdir(parents=True, exist_ok=True)
    c.write_text(json.dumps(r))
    return name, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.data_root)
    cache_dir = root / "processed/audit/outcomes"
    runs = discover_runs(root)
    jobs = [(f"{k}__{n}", str(d), str(cache_dir / f"{k}__{n}.json")) for n, d, k in runs]
    catches = evolutions = 0
    per_run = {}
    with Pool(args.workers) as pool:
        for name, r in pool.imap_unordered(_one, jobs):
            per_run[name] = r
            catches += r["catches"]
            evolutions += int(r["evolution"])
    out = root / "processed/audit/outcomes_report.json"
    out.write_text(json.dumps({"catches": catches, "evolutions": evolutions,
                               "per_run": per_run}, indent=1))
    print(f"outcomes: {catches} catches, {evolutions} evolutions -> {out}")


if __name__ == "__main__":
    main()
