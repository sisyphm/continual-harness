"""W33 PILOT: one full-spine expedition per starter from the fleet plan (spec §5.4/§13).
Each worker runs one plan entry end-to-end: 52/52 required, solve-then-record + persona
+ 6 rotation blocks + full v2 recording. Usage: pilot_driver.py <run_id>"""
import json, sys, time
sys.path.insert(0, ".")
from collection.playthrough.director import run_playthrough

PLAN = "/root/code/proj-minhyuk-2026/pokemon-worldmodel/data/processed/w33_regen_plan.json"
POLICY = "policies_heatz"   # adjusted below if the real dir differs

run_id = sys.argv[1]
plan = json.load(open(PLAN))
runs = plan["runs"] if isinstance(plan, dict) else plan
entry = next(r for r in runs if r["run_id"] == run_id)

import glob, os
pol = os.environ.get("PILOT_POLICY_DIR")
assert pol and os.path.isdir(pol), f"PILOT_POLICY_DIR invalid: {pol}"

# RE-SEED ON RETRY. A wedge is DETERMINISTIC: same persona seed -> same BFS
# tie-breaks -> same route -> same blocked goal. exp_003_mudkip and exp_008_treecko
# each died three times at the identical goal because every retry replayed the same
# walk. RETRY_SALT (set by the fleet driver from how many wedge_/failed archives a
# run already has) shifts the persona seed so a retry takes a DIFFERENT shortest
# path. The plan seed still governs a first attempt, so a clean run is unchanged.
_salt = int(os.environ.get("RETRY_SALT", "0"))
_seed = entry["seed"] if not _salt else (entry["seed"] * 6364136223846793005 + _salt * 1442695040888963407) % (2**31)
if _salt:
    print(f"RETRY_SALT={_salt}: persona seed {entry['seed']} -> {_seed}", flush=True)

t0 = time.time()
summary = run_playthrough(
    policy_dir=pol,
    out_dir=f"{os.environ.get('FLEET_OUT', 'data/pilot_v2')}/{run_id}",
    starter=entry["starter"],
    seed=_seed,
    record=True,
    expedition=[{"after": m, "block": b, **kw} for m, b, kw in entry["block_schedule"]],
    persona={"seed": _seed},
    resume_state=os.environ.get("RESUME_STATE") or None,
    resume_after=os.environ.get("RESUME_AFTER") or None,
    resume=os.environ.get("RESUME_JSON") or None,
    savestate_every=int(os.environ.get("SAVESTATE_EVERY", "500")),
)
elapsed = time.time() - t0
res = {"run_id": run_id, "wall_s": elapsed,
       "milestones_passed": summary.get("milestones_passed"),
       "aborted": summary.get("aborted_milestone"),
       "retry_total": summary.get("retry_total")}
print("PILOT-RESULT " + json.dumps(res), flush=True)
