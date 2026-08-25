"""milestone_lab — the parallel experiment harness for collector changes.

A full run reveals one bug per hour: only whichever blocker it reaches first. But
every milestone has an expert `<NAME>_completed.state`, so transition N -> N+1 can be
tested from N's frozen state in isolation — all 51 transitions in parallel cover the
whole game in ~8 minutes (W34, measured at 10 workers). This file makes that a repo
tool instead of scratchpad scripts, and adds the missing piece: N env-variants in one
command with an auto-verdict table, so "does my change regress anything?" is one line:

    .venv/bin/python -m collection.tools.milestone_lab sweep
    .venv/bin/python -m collection.tools.milestone_lab sweep --env W33_BATTLE_V2=1
    .venv/bin/python -m collection.tools.milestone_lab variants --spec lab.json

spec: {"workers": 10, "variants": {"base": {}, "bv2": {"W33_BATTLE_V2": "1"}}}
The table marks per-transition flips between variants, not just totals — a 49/51 that
passes DIFFERENT transitions than the baseline is a regression wearing a tie.

Results land under data/milestone_lab/<stamp>/ as one JSONL per variant. Workers are
subprocesses (one emulator each); the known-flaky pair (see KNOWN_FLAKY) is reported
but not counted against a variant.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HARN = Path(__file__).resolve().parents[2]
POLICY = "../pokeagent-solution/expert_policies_by_llm"
ROM = "Emerald-GBAdvance/rom.gba"

# Long transitions get bigger budgets, not exclusions. PETALBURG_WOODS (Aqua-grunt
# sequence + woods traverse) and ROXANNE_BATTLE (the full gym) failed EVERY recorded
# sweep including all baselines; the budgets below removed the wall-kill disguise and
# exposed the REAL defect — their predecessors' fixture files held title-screen
# states (see `refixture`, which earns a replacement). Budgets stay generous because
# the transitions are genuinely long. Values are (max_wall_s -> frame budget at
# x500, subprocess timeout s).
LONG_TRANSITIONS: dict[str, tuple[float, int]] = {
    "PETALBURG_WOODS": (480.0, 1200),
    "ROXANNE_BATTLE": (480.0, 1200),
}
DEFAULT_BUDGET = (240.0, 420)


def run_one(frm: str, tgt: str) -> dict:
    """Worker body: one transition from the expert state. Runs in THIS process."""
    import collection.collect_events as ce
    from collection.catalog import discover_heatz_events
    from collection.direct_runner import DirectEmulatorRunner
    from collection.playthrough.spine import run_milestone

    ce.set_expected_starter("Mudkip")
    events = {e["event_id"]: e for e in discover_heatz_events(POLICY)}
    runner = None
    try:
        if tgt not in events:
            raise RuntimeError(f"no heatz event for {tgt}")
        runner = DirectEmulatorRunner(
            rom_path=ROM, load_state=fixture_path(frm), story_bucket=tgt, savestate_every=0)
        runner.initialize()
        exp = ce._load_expected_state(
            rom_path=ROM, completed_state=events[tgt].get("completed_state"), event_id=tgt)
        f0, t0 = runner.frame_idx, time.time()
        r = run_milestone(
            runner, event_id=tgt, policy_dir=POLICY, expected_state=exp,
            postcondition=events[tgt].get("postcondition", tgt), start_money=0,
            starter="mudkip", max_actions=8000 if tgt in LONG_TRANSITIONS else 4000,
            max_wall_s=LONG_TRANSITIONS.get(tgt, DEFAULT_BUDGET)[0])
        return {"ok": r["validation"] in ("passed", "skipped"),
                "validation": r["validation"], "reason": r.get("failure_reason"),
                "actions": r.get("actions_taken"), "frames": runner.frame_idx - f0,
                "solve_s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"ok": False, "reason": f"exception: {repr(e)[:160]}"}
    finally:
        try:
            if runner is not None:
                runner.close()
        except Exception:
            pass


def _worker_main(frm: str, tgt: str) -> None:
    res = run_one(frm, tgt)
    print("RESULT " + json.dumps(res), flush=True)


FIXTURE_OVERRIDES = HARN / "data" / "fixtures"


def fixture_path(m: str) -> str:
    """The expert `_completed.state` for milestone m — preferring a repaired override
    from data/fixtures/ over the (read-only, sometimes broken) policy-dir asset."""
    o = FIXTURE_OVERRIDES / f"{m}_completed.state"
    return str(o) if o.exists() else f"{POLICY}/{m}/{m}_completed.state"


def refixture(tgt: str) -> int:
    """Repair a broken `<TGT>_completed.state` by EARNING it: run the predecessor's
    fixture through the TGT transition and save the runner's end state on PASS.

    Exists because ROUTE_104_SOUTH and TRAINER_JOSH_BATTLE's fixtures held TITLE-
    SCREEN states (probed W34) — every sweep in history stalled from action zero on
    the transitions that START there, disguised as timeouts by the old wall caps.
    The policy dir is upstream's and is never touched: repairs land in
    data/fixtures/, which fixture_path() prefers.
    """
    import collection.collect_events as ce
    from collection.catalog import MILESTONE_ORDER, discover_heatz_events
    from collection.direct_runner import DirectEmulatorRunner
    from collection.playthrough.spine import run_milestone

    frm = MILESTONE_ORDER[MILESTONE_ORDER.index(tgt) - 1]
    ce.set_expected_starter("Mudkip")
    events = {e["event_id"]: e for e in discover_heatz_events(POLICY)}
    runner = DirectEmulatorRunner(
        rom_path=ROM, load_state=fixture_path(frm), story_bucket=tgt, savestate_every=0)
    runner.initialize()
    exp = ce._load_expected_state(
        rom_path=ROM, completed_state=events[tgt].get("completed_state"), event_id=tgt)
    r = run_milestone(
        runner, event_id=tgt, policy_dir=POLICY, expected_state=exp,
        postcondition=events[tgt].get("postcondition", tgt), start_money=0,
        starter="mudkip", max_actions=8000,
        max_wall_s=LONG_TRANSITIONS.get(tgt, DEFAULT_BUDGET)[0])
    if r["validation"] not in ("passed", "skipped"):
        print(f"refixture {tgt}: transition FAILED ({r.get('failure_reason')}) — no override written")
        runner.close()
        return 1
    sb = runner.save_state_bytes()
    runner.close()
    FIXTURE_OVERRIDES.mkdir(parents=True, exist_ok=True)
    dst = FIXTURE_OVERRIDES / f"{tgt}_completed.state"
    dst.write_bytes(sb)
    print(f"refixture {tgt}: PASSED from {frm}, override written {dst} ({len(sb)} bytes)")
    return 0


def _spawn_one(args: tuple) -> dict:
    """Sweep-side wrapper: run one transition in a fresh subprocess (own emulator,
    own env), 420s hard timeout like a stall watchdog would give it."""
    frm, tgt, env = args
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, "-m", "collection.tools.milestone_lab", "one", frm, tgt],
            cwd=HARN, capture_output=True, text=True,
            timeout=LONG_TRANSITIONS.get(tgt, DEFAULT_BUDGET)[1],
            env={**os.environ, **env})
        lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        d = (json.loads(lines[-1][len("RESULT "):]) if lines else
             {"ok": False, "reason": "no result line", "stderr": (r.stderr or "")[-200:]})
    except subprocess.TimeoutExpired:
        d = {"ok": False, "reason": "timeout>420s"}
    except Exception as e:
        d = {"ok": False, "reason": repr(e)[:120]}
    d.update(from_state=frm, milestone=tgt, wall_s=round(time.time() - t0, 1))
    return d


def _transitions(limit: int = 0) -> list[tuple[str, str]]:
    from collection.catalog import MILESTONE_ORDER
    pairs = list(zip(MILESTONE_ORDER, MILESTONE_ORDER[1:]))
    return pairs[:limit] if limit else pairs


def sweep(env: dict, workers: int, limit: int, out: Path | None, tag: str) -> dict:
    pairs = _transitions(limit)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(_spawn_one, [(f, t, env) for f, t in pairs]):
            results.append(d)
            mark = "pass" if d["ok"] else f"FAIL({d.get('reason')})"
            print(f"[{tag}] {d['milestone']}: {mark}", flush=True)
    passed = sum(1 for d in results if d["ok"])
    summary = {"tag": tag, "env": env, "passed": passed, "total": len(results),
               "wall_min": round((time.time() - t0) / 60, 1),
               "failed": sorted(d["milestone"] for d in results if not d["ok"])}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for d in results:
                f.write(json.dumps(d) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")
    print(f"[{tag}] {passed}/{len(results)} in {summary['wall_min']} min "
          f"failed={summary['failed']}", flush=True)
    return summary


def variants(spec: dict, workers: int, out_dir: Path) -> None:
    names = list(spec["variants"])
    sums = {}
    for name in names:
        sums[name] = sweep(spec["variants"][name], workers, spec.get("limit", 0),
                           out_dir / f"{name}.jsonl", name)
    base = names[0]
    print("\n=== verdict table (baseline = first variant) ===")
    w = max(len(n) for n in names)
    for name in names:
        s = sums[name]
        flips_bad = sorted(set(s["failed"]) - set(sums[base]["failed"]))
        flips_good = sorted(set(sums[base]["failed"]) - set(s["failed"]))
        delta = "" if name == base else (
            f"  broke={flips_bad or '-'}  fixed={flips_good or '-'}")
        print(f"{name:<{w}}  {s['passed']}/{s['total']}  {s['wall_min']}min{delta}")
    (out_dir / "verdicts.json").write_text(json.dumps(sums, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(prog="milestone_lab")
    sub = ap.add_subparsers(dest="op", required=True)
    o = sub.add_parser("one")
    o.add_argument("frm")
    o.add_argument("tgt")
    rf = sub.add_parser("refixture")
    rf.add_argument("tgt")
    s = sub.add_parser("sweep")
    s.add_argument("--workers", type=int, default=10)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--env", action="append", default=[], help="KEY=VAL, repeatable")
    s.add_argument("--tag", default="sweep")
    v = sub.add_parser("variants")
    v.add_argument("--spec", required=True)
    v.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    stamp = time.strftime("%m%d_%H%M%S")
    if args.op == "one":
        _worker_main(args.frm, args.tgt)
        return 0
    if args.op == "refixture":
        return refixture(args.tgt)
    if args.op == "sweep":
        env = dict(kv.split("=", 1) for kv in args.env)
        out = HARN / "data" / "milestone_lab" / stamp / f"{args.tag}.jsonl"
        s = sweep(env, args.workers, args.limit, out, args.tag)
        return 0 if s["passed"] == s["total"] else 1
    spec = json.loads(Path(args.spec).read_text())
    variants(spec, args.workers, HARN / "data" / "milestone_lab" / stamp)
    return 0


if __name__ == "__main__":
    main()
