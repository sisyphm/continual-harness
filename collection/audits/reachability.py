"""Data-closure: the REACHABILITY PROOF — scope as a theorem, not a tour's claim.

"In scope" was defined as the corpus-visited maps, which rested on the coverage tour's claim of
having reached everything. This audit computes the warp+connection graph closure from Littleroot
Town (map 0,9 — where the intro truck delivers the player) over the ROM-enumerated manifest and
compares it to the corpus:

  reachable \\ corpus   maps the player COULD walk to that the corpus never visited (red rows —
                        though some may be flag-gated in-game: graph reachability is an UPPER
                        bound; each such map is collected-or-explicitly-excluded, never unknown)
  corpus \\ reachable   visited maps with no graph path (cutscene-only teleport targets — should
                        be exactly the known scripted one-offs)

Policy exclusions are applied (link rooms, group 25). Usage:
  .venv/bin/python -m collection.audits.reachability --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

START = "0,9"                              # Littleroot Town (the truck's destination)
EXCLUDED_GROUPS = ("25,",)                 # wireless/link rooms: multiplayer, policy-excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    args = ap.parse_args()
    root = Path(args.data_root)
    M = json.loads((root / "processed/coverage_manifest.json").read_text())
    maps, scope = M["maps"], set(M["scope_maps"])

    seen = {START}
    q = deque([START])
    while q:
        k = q.popleft()
        m = maps.get(k)
        if m is None:
            continue
        for e in [w["dst_map"] for w in m["warps"]] + [c["dst_map"] for c in m["connections"]]:
            if e in seen or any(e.startswith(g) for g in EXCLUDED_GROUPS) or e not in maps:
                continue
            seen.add(e)
            q.append(e)

    missed = sorted(seen - scope)
    unreachable = sorted(scope - seen)
    # the flag-blind closure spans the whole game (the graph can't see story gates), so the
    # ACTIONABLE list is the 1-hop FRONTIER: maps directly adjacent to corpus maps that the
    # corpus never entered — each is either enterable pre-badge (a real coverage miss) or
    # flag-gated (verify once, then policy-record). Small and checkable, unlike the closure.
    frontier = set()
    for k in scope:
        m = maps.get(k)
        if m is None:
            continue
        for e in [w["dst_map"] for w in m["warps"]] + [c["dst_map"] for c in m["connections"]]:
            if e not in scope and e in maps and not any(e.startswith(g) for g in EXCLUDED_GROUPS):
                frontier.add((k, e))
    report = {"start": START, "reachable": len(seen), "scope": len(scope),
              "frontier_unvisited": sorted(f"{a}->{b}" for a, b in frontier),
              "reachable_not_in_corpus": missed, "corpus_not_graph_reachable": unreachable}
    out = root / "processed/audit/reachability_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"graph closure from {START}: {len(seen)} maps reachable (whole game, flag-blind)")
    print(f"  1-hop frontier (adjacent, never entered): {report['frontier_unvisited']}")
    print(f"  reachable but NOT in corpus (flag-blind whole-game bound): {len(missed)}")
    print(f"  in corpus but not graph-reachable (scripted one-offs expected): {unreachable}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
