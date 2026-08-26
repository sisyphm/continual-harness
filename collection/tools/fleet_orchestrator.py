"""Overnight fleet orchestrator: 60 plan runs -> gate-checked corpus by morning.

Lifecycle per run: launch under runsup -> in-process tripwires guard the run ->
on exit: PILOT-RESULT? -> run_acceptance audit -> classify:
  banked            acceptance PASS
  banked_shortfall  gate PASS + spine 51/51, job-layer defects only (logged for
                    morning top-off; NOT retried — systematic shortfalls loop)
  retry             gate FAIL / no PILOT-RESULT / STALL.json -> ONE fresh retry
                    (RETRY_SALT=1); second failure -> failed
Ledger: data/regen_v3/fleet/fleet_ledger.jsonl ; heartbeat every 5 min.
"""
import json, re, subprocess, time
from pathlib import Path

HARN = Path("/root/code/proj-minhyuk-2026/continual-harness")
PLAN = "/root/code/proj-minhyuk-2026/pokemon-worldmodel/data/processed/w33_regen_plan.json"
DRIVER = "/tmp/claude-0/-root-code-proj-minhyuk-2026/e184a1ee-ddfe-4df8-aa4c-32965acc1267/scratchpad/rg_driver.py"
OUT = Path("/root/data1/proj-minhyuk-2026/w33_fleet")
SUP = HARN / "data" / "runsup"
SLOTS = 9
LEDGER = OUT / "fleet_ledger.jsonl"
ENV = dict(W33_DIRECT_RECORD="1", W33_FAST_RECORD="1", W33_BATTLE_V2="1",
           PILOT_POLICY_DIR="/root/code/proj-minhyuk-2026/pokeagent-solution/expert_policies_by_llm",
           FLEET_OUT=str(OUT), SAVESTATE_EVERY="500")


def log(row):
    row["at"] = time.strftime("%H:%M:%S")
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def launch(rid, salt=0):
    import os
    name = f"fleet_{rid}" + (f"_r{salt}" if salt else "")
    env = {**ENV, **({"RETRY_SALT": str(salt)} if salt else {})}
    cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + \
          [str(HARN / ".venv/bin/python"), DRIVER, rid]
    subprocess.run([str(HARN / ".venv/bin/python"), "-m", "collection.tools.runsup",
                    "launch", "--name", name, "--cwd", str(HARN), "--"] + cmd,
                   cwd=HARN, capture_output=True)
    log({"ev": "launch", "run": rid, "salt": salt})
    return name


def alive(name):
    p = SUP / f"{name}.json"
    if not p.exists():
        return False
    rec = json.loads(p.read_text())
    try:
        stat = Path(f"/proc/{rec['pid']}/stat").read_text()
        return int(stat.rsplit(")", 1)[1].split()[19]) == rec["starttime"]
    except Exception:
        return False


def finished_ok(name):
    lg = (SUP / f"{name}.log").read_text(errors="replace")
    i = lg.rfind("No Pokemon found in party")
    return "PILOT-RESULT" in (lg[i:] if i > 0 else lg)


def audit(rid, name):
    try:
        r = subprocess.run([str(HARN / ".venv/bin/python"), "-m",
                            "collection.audits.run_acceptance", str(OUT / rid),
                            "--log", str(SUP / f"{name}.log")],
                           cwd=HARN, capture_output=True, text=True, timeout=900,
                           env={"PYTHONPATH": str(HARN), "PATH": "/usr/bin:/bin"})
        rep = json.loads((OUT / rid / "acceptance.json").read_text())
        return rep
    except Exception as e:
        return {"verdict": "AUDIT_ERROR", "defects": [repr(e)[:120]]}


plan = json.load(open(PLAN))
queue = [r["run_id"] for r in plan["runs"]]
running = {}                      # rid -> (sup_name, salt)
done, shortfall, failed = [], [], []
t0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)
log({"ev": "fleet_start", "queued": len(queue), "slots": SLOTS})

import shutil as _sh
def _free_gb():
    return _sh.disk_usage(OUT).free / 1e9

while queue or running:
    while queue and len(running) < SLOTS:
        if _free_gb() < 60:
            log({"ev": "disk_wait", "free_gb": round(_free_gb())})
            break
        rid = queue.pop(0)
        running[rid] = (launch(rid, 0), 0)
    time.sleep(60)
    for rid in list(running):
        name, salt = running[rid]
        if alive(name):
            continue
        del running[rid]
        ok = finished_ok(name)
        stall = (OUT / rid / "STALL.json").exists()
        if ok and not stall:
            rep = audit(rid, name)
            v = rep.get("verdict")
            gate = rep.get("gate")
            spine = rep.get("milestones")
            if v == "PASS":
                done.append(rid)
                log({"ev": "banked", "run": rid, "defects": 0})
                continue
            if gate == "PASS" and spine == "51/51":
                shortfall.append(rid)
                log({"ev": "banked_shortfall", "run": rid,
                     "defects": len(rep.get("defects", []))})
                continue
        if salt == 0:
            # fresh retry: preserve the failed dir for forensics, run under salt
            bad = OUT / rid
            if bad.exists():
                bad.rename(OUT / f"_failed0_{rid}")
            log({"ev": "retry", "run": rid, "stall": stall})
            running[rid] = (launch(rid, 1), 1)
        else:
            failed.append(rid)
            log({"ev": "failed", "run": rid})
    if int(time.time() - t0) % 300 < 60:
        log({"ev": "hb", "min": round((time.time() - t0) / 60),
             "running": len(running), "banked": len(done),
             "shortfall": len(shortfall), "failed": len(failed),
             "queued": len(queue)})

log({"ev": "fleet_done", "banked": len(done), "shortfall": len(shortfall),
     "failed": failed, "hours": round((time.time() - t0) / 3600, 1)})
