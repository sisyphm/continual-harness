"""invariance_gate — CI check: recording must not change what the policy DOES.

The W33 pilot lost a week to wall-clock control flow: budgets and wedge windows
measured in seconds made the same seed play DIFFERENTLY at different fps, so turning
the recorder on (slower) changed trajectories and "recording bugs" appeared that were
really pacing bugs. The fix was frame-based control flow everywhere; THIS gate is the
standing proof it stays fixed.

Two arms, same seed, from fresh boot to --stop-after: arm `rec` records under the
production config (direct + fast), arm `dry` runs recorder-detached. PASS iff every
milestone ends on the SAME emulator frame in both arms. Any drift means someone
reintroduced time-dependent control flow — the gate names the first drifting
milestone, which is where to look.

    .venv/bin/python -m collection.tools.invariance_gate run
    .venv/bin/python -m collection.tools.invariance_gate run --stop-after GO_ROUTE103

Both arms run concurrently (two emulators, ~one wave's load). The baseline is not a
stored file but the twin arm, so the gate never goes stale when policy changes —
chained-walk (W34) deliberately changed trajectories and this gate is unaffected.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HARN = Path(__file__).resolve().parents[2]
POLICY = "../pokeagent-solution/expert_policies_by_llm"


def arm(args) -> int:
    from collection.playthrough.director import run_playthrough
    s = run_playthrough(
        policy_dir=POLICY, out_dir=args.out, starter=args.starter, seed=args.seed,
        record=bool(args.record), stop_after=args.stop_after, expedition=[],
        savestate_every=4000 if args.record else 0)
    print("ARM-DONE " + json.dumps({"badges": s.get("badges")}), flush=True)
    return 0


def run(args) -> int:
    stamp = time.strftime("%m%d_%H%M%S")
    out = HARN / "data" / "invariance" / stamp
    procs = {}
    for name, rec in (("rec", 1), ("dry", 0)):
        env = {**os.environ}
        if rec:
            env.update(W33_DIRECT_RECORD="1", W33_FAST_RECORD="1")
        d = out / name
        procs[name] = subprocess.Popen(
            [sys.executable, "-m", "collection.tools.invariance_gate", "arm",
             "--out", str(d), "--record", str(rec), "--seed", str(args.seed),
             "--starter", args.starter, "--stop-after", args.stop_after],
            cwd=HARN, env=env, stdout=open(out / f"{name}.log", "wb"),
            stderr=subprocess.STDOUT) if d.mkdir(parents=True, exist_ok=True) or True else None
    rc = {n: p.wait(timeout=args.timeout_s) for n, p in procs.items()}
    sums = {}
    for name in procs:
        p = out / name / "playthrough_summary.json"
        if not p.exists():
            print(f"VERDICT FAIL arm '{name}' produced no summary (rc={rc[name]}) — see {out}/{name}.log")
            return 1
        sums[name] = {m["event_id"]: m.get("end_frame")
                      for m in json.loads(p.read_text()).get("results", [])
                      if m.get("end_frame") is not None}
    shared = [m for m in sums["rec"] if m in sums["dry"]]
    drift = [(m, sums["rec"][m], sums["dry"][m]) for m in shared
             if sums["rec"][m] != sums["dry"][m]]
    for m in shared:
        tag = "==" if sums["rec"][m] == sums["dry"][m] else "DRIFT"
        print(f"{m:36s} rec={sums['rec'][m]:>8} dry={sums['dry'][m]:>8}  {tag}")
    if not shared:
        print(f"VERDICT FAIL no shared milestones (rec={len(sums['rec'])} dry={len(sums['dry'])})")
        return 1
    if drift:
        m0 = drift[0]
        print(f"VERDICT FAIL {len(drift)}/{len(shared)} milestones drift; first={m0[0]} "
              f"(rec={m0[1]} dry={m0[2]}) — time-dependent control flow reintroduced?")
        return 1
    print(f"VERDICT PASS {len(shared)} milestones frame-identical across recording modes ({out})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="invariance_gate")
    sub = ap.add_subparsers(dest="op", required=True)
    for name in ("run", "arm"):
        p = sub.add_parser(name)
        p.add_argument("--stop-after", default="EXIT_RIVAL_HOUSE")
        p.add_argument("--seed", type=int, default=730)
        p.add_argument("--starter", default="mudkip")
        if name == "run":
            p.add_argument("--timeout-s", type=int, default=5400)
        else:
            p.add_argument("--out", required=True)
            p.add_argument("--record", type=int, required=True)
    args = ap.parse_args()
    return run(args) if args.op == "run" else arm(args)


if __name__ == "__main__":
    sys.exit(main())
