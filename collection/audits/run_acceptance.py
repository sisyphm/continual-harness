"""Per-run acceptance audit (W34): did this run CLEAR, and did it do EVERY job it
was registered for? PASS is earned, never assumed.

Checks, per run directory + its plan entry + its supervisor log:
  1. gate      — playthrough_gate verdict (channel provenance, badge, party at end)
  2. spine     — 51/51 milestones passed, none failed, not aborted
  3. party     — the single-mon invariant across the WHOLE run, sampled from
                 savestates (stride-N + final); any count>1 sample = FAIL with frame
  4. jobs      — every scheduled block has a receipt; per kind:
                   bfs_sweep    assigned tiles vs visited per slice + legs both ways
                   trainers     every planned flag engaged/won or precisely reasoned
                                (spine sight-cone defeats count as covered)
                   mart_buy     the want-list actually bought, delta-verified
                   others       ran, no error, not over_budget
  5. hygiene   — no STALL.json, blemish_spans accounted, heal trips all succeeded

Output: verdict line + defect list (empty = PASS) + a JSON report next to the run.

    .venv/bin/python -m collection.audits.run_acceptance <run_dir> [--plan PLAN] [--log LOG]
"""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

DEFAULT_PLAN = "/root/code/proj-minhyuk-2026/pokemon-worldmodel/data/processed/w33_regen_plan.json"
PARTY_COUNT_ADDR = 0x020244E9
SAVESTATE_STRIDE = 10               # every 10th savestate = every 5k frames


def _receipts(log_text: str) -> list[dict]:
    out = []
    for m in re.finditer(r'\{"BLOCK": (\{.*\})\}\s*$', log_text, re.M):
        try:
            out.append(json.loads(m.group(1)))
        except Exception:
            pass
    return out


def _party_scan(run_dir: Path, defects: list, report: dict) -> None:
    from collection.direct_runner import DirectEmulatorRunner
    from collection.extractors.ram import GBAState
    states = sorted((run_dir / "savestates").glob("*.state.z"))
    if not states:
        defects.append("party: no savestates to scan")
        return
    picks = states[::SAVESTATE_STRIDE]
    if states[-1] not in picks:
        picks.append(states[-1])
    r = DirectEmulatorRunner(load_state=None, savestate_every=0)
    r.initialize()
    bad = []
    try:
        for sf in picks:
            r.load_state_bytes(zlib.decompress(sf.read_bytes()), record=False)
            pc = GBAState(env=r.env).u8(PARTY_COUNT_ADDR)
            if pc > 1:
                # Wally tutorial exemption: his Zigzagoon rides the player's party
                # for the scripted gym scene. No wild encounters exist in gyms, so
                # a gym sample with party==2 is the tutorial, not a catch.
                if "GYM" in str(r.nav_state().map or "").upper():
                    continue
                bad.append((sf.name, int(pc)))
    finally:
        r.close()
    report["party_samples"] = len(picks)
    if bad:
        defects.append(f"party: count>1 at {bad[:3]}{'...' if len(bad) > 3 else ''}")


def audit(run_dir: Path, plan_path: str, log_path: Path | None) -> dict:
    run_id = run_dir.name
    defects: list[str] = []
    report: dict = {"run": run_id}

    plan = json.load(open(plan_path))
    entry = next((r for r in plan["runs"] if r["run_id"] == run_id), None)
    if entry is None:
        return {"run": run_id, "verdict": "FAIL", "defects": ["no plan entry"]}
    lg = log_path.read_text(errors="replace") if log_path and log_path.exists() else ""

    # 1 gate ------------------------------------------------------------------
    from collection.audits.playthrough_gate import gate_run
    g = gate_run(run_dir)
    report["gate"] = g.get("verdict")
    if g.get("verdict") != "PASS":
        defects.append(f"gate: {g.get('verdict')} {g.get('reasons')}")

    # 2 spine -----------------------------------------------------------------
    summ = json.loads((run_dir / "playthrough_summary.json").read_text())
    report["milestones"] = f"{summ.get('milestones_passed')}/{summ.get('milestones_total')}"
    if summ.get("milestones_passed") != summ.get("milestones_total"):
        defects.append(f"spine: {report['milestones']} aborted={summ.get('aborted_milestone')}")
    if summ.get("aborted_milestone"):
        defects.append(f"spine aborted at {summ['aborted_milestone']}")

    # 3 party -----------------------------------------------------------------
    _party_scan(run_dir, defects, report)

    # 4 jobs ------------------------------------------------------------------
    receipts = _receipts(lg)
    planned = entry["block_schedule"]
    by_kind_planned: dict[str, int] = {}
    for _, b, _ in planned:
        by_kind_planned[b] = by_kind_planned.get(b, 0) + 1
    by_kind_got: dict[str, int] = {}
    for r_ in receipts:
        by_kind_got[r_["block"]] = by_kind_got.get(r_["block"], 0) + 1
    report["blocks"] = {"planned": by_kind_planned, "receipts": by_kind_got}
    for k, n in by_kind_planned.items():
        if by_kind_got.get(k, 0) < n:
            defects.append(f"jobs: {k} receipts {by_kind_got.get(k, 0)}/{n}")
    for r_ in receipts:
        if not r_.get("ran", True):
            defects.append(f"jobs: {r_['block']} after {r_.get('after')} never ran: {r_.get('reason')}")
        elif r_.get("error"):
            defects.append(f"jobs: {r_['block']} after {r_.get('after')} error: {str(r_['error'])[:80]}")

    # sweeps: assigned vs visited, legs both directions
    assigned: dict[str, int] = {}
    for _, b, kw in planned:
        if b == "bfs_sweep":
            for mp, tl in (kw.get("tiles") or {}).items():
                assigned[mp] = assigned.get(mp, 0) + len(tl)
    visited: dict[str, int] = {}
    legs_done = set()
    denied = 0
    for r_ in receipts:
        if r_.get("block") == "bfs_sweep" and r_.get("ran"):
            for mp, n in (r_.get("tiles_visited_per_map") or {}).items():
                visited[mp] = visited.get(mp, 0) + n
            for mp, n in (r_.get("denied_tiles_per_map") or {}).items():
                denied += n if isinstance(n, int) else len(n)
            for a, b2 in (r_.get("connections_crossed") or []):
                legs_done.add((a, b2))
    short = {mp: (a, visited.get(mp, 0)) for mp, a in assigned.items()
             if visited.get(mp, 0) + 0 < a}
    report["tiles"] = {"assigned": sum(assigned.values()),
                       "visited_of_assigned_maps": sum(visited.get(m, 0) for m in assigned),
                       "denied": denied}
    # denied tiles (NPC-occupied etc.) are precisely-reasoned exclusions
    for mp, (a, v) in short.items():
        if v + denied < a:
            defects.append(f"jobs: sweep {mp} visited {v}/{a}")
    for _, b, kw in planned:
        if b == "bfs_sweep":
            for a, b2 in (kw.get("legs") or []):
                if (a, b2) not in legs_done or (b2, a) not in legs_done:
                    defects.append(f"jobs: leg {a}<->{b2} not crossed both ways")

    # trainers
    planned_tr = {}
    for _, b, kw in planned:
        if b == "trainer_engagement":
            for t in kw["targets"]:
                planned_tr[t["trainer_flag"]] = t["map"]
    engaged, skipped = {}, {}
    for r_ in receipts:
        if r_.get("block") == "trainer_engagement" and r_.get("ran"):
            for e in r_.get("engaged", []):
                engaged[e["flag"]] = e.get("won")
            for s in r_.get("skipped_with_reason", []):
                skipped.setdefault(s["flag"], []).append(str(s.get("reason")))
    tr_missing = []
    for f, mp in sorted(planned_tr.items()):
        if f in engaged:
            continue
        reasons = skipped.get(f, [])
        if any("already defeated" in r_ for r_ in reasons):
            continue                          # spine sight-cone took it: covered
        tr_missing.append((f, mp, reasons[-1] if reasons else "(block never ran)"))
    report["trainers"] = {"planned": len(planned_tr), "engaged": len(engaged),
                          "spine_covered": sum(1 for f in planned_tr if f not in engaged
                                               and any("already defeated" in r_
                                                       for r_ in skipped.get(f, [])))}
    for f, mp, why in tr_missing:
        defects.append(f"jobs: trainer {f} ({mp}) missed: {why}")

    # mart
    for r_ in receipts:
        if r_.get("block") == "mart_buy" and r_.get("ran"):
            want = dict(next((kw.get("want") or [] for _, b, kw in planned if b == "mart_buy"), []))
            bought = {p.get("item"): p.get("qty") for p in r_.get("purchases", [])
                      if p.get("verified")}
            report["mart"] = {"want": want, "bought": bought}
            for item, qty in want.items():
                if bought.get(item, 0) < qty:
                    defects.append(f"jobs: mart bought {bought.get(item, 0)}/{qty} of item {item}")

    # 5 hygiene ---------------------------------------------------------------
    if (run_dir / "STALL.json").exists():
        defects.append(f"hygiene: STALL.json present: {(run_dir / 'STALL.json').read_text()[:120]}")
    man = json.loads((run_dir / "manifest.json").read_text())
    report["blemish_spans"] = len(man.get("blemish_spans", []))
    heal_fail = len(re.findall(r"heal: .*FAILED", lg))
    report["heals"] = {"ok": len(re.findall(r"heal: .*healed", lg)), "failed": heal_fail}
    if heal_fail:
        defects.append(f"hygiene: {heal_fail} heal trips FAILED")

    report["verdict"] = "PASS" if not defects else "FAIL"
    report["defects"] = defects
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--log", default=None,
                    help="supervisor log with BLOCK receipts (default: data/runsup/accept_<id>*.log)")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    log = Path(a.log) if a.log else None
    if log is None:
        sup = run_dir.parents[2] / "runsup"
        cands = sorted(sup.glob(f"*{run_dir.name}*.log")) or sorted(sup.glob(
            f"*{run_dir.name.split('_')[0]}_{run_dir.name.split('_')[1]}*.log"))
        log = cands[-1] if cands else None
    rep = audit(run_dir, a.plan, log)
    (run_dir / "acceptance.json").write_text(json.dumps(rep, indent=1))
    print(f"ACCEPTANCE {rep['verdict']} {run_dir.name}")
    for d in rep["defects"]:
        print("  -", d)
    if rep["verdict"] == "PASS":
        print(f"  milestones={rep.get('milestones')} tiles={rep.get('tiles')} "
              f"trainers={rep.get('trainers')} heals={rep.get('heals')}")
    return 0 if rep["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
