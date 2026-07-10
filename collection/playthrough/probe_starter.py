"""Empirical starter-select probe: boot -> spine through STARTER_CHOSEN, read the
resulting party species. No recording. Proves _select_starter_action picks the
run-configured starter. Usage:
  PYTHONPATH=. python -m collection.playthrough.probe_starter <mudkip|treecko|torchic>
"""
import sys, json
from pathlib import Path
from collection.catalog import MILESTONE_ORDER, discover_heatz_events
from collection.direct_runner import DirectEmulatorRunner
import collection.collect_events as ce
from collection.playthrough.spine import run_milestone

STARTER = sys.argv[1] if len(sys.argv) > 1 else "mudkip"
POLICY = "../pokeagent-solution/expert_policies_by_llm"

ce.set_expected_starter(STARTER.capitalize())
events = discover_heatz_events(POLICY)
by_id = {e["event_id"]: e for e in events}
order = [m for m in MILESTONE_ORDER if m in by_id]

runner = DirectEmulatorRunner(rom_path="Emerald-GBAdvance/rom.gba", load_state=None)
runner.initialize()
for _ in range(600):
    runner.step_frame(["a"], phase="boot")
    if runner.state().game_state not in ("title", "intro", None):
        break
try:
    for event_id in order:
        exp = ce._load_expected_state(rom_path="Emerald-GBAdvance/rom.gba",
                                      completed_state=by_id[event_id].get("completed_state"),
                                      event_id=event_id)
        post = by_id[event_id].get("postcondition", event_id)
        r = run_milestone(runner, event_id=event_id, policy_dir=POLICY, expected_state=exp,
                          postcondition=post, start_money=0, max_actions=6000, starter=STARTER)
        st = runner.state()
        if event_id == "STARTER_CHOSEN":
            party = [(m.get("species"), m.get("level")) for m in (st.party_summary or [])]
            print(json.dumps({"target": STARTER, "validation": r["validation"],
                              "party": party, "frames": runner.frame_idx}))
            break
        if r["validation"] not in ("passed", "skipped"):
            print(json.dumps({"FAILED_BEFORE_STARTER": event_id, "reason": r.get("failure_reason")}))
            break
finally:
    runner.close()
