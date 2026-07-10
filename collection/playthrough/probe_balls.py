"""Definitively map the starter ball layout: reach STARTER_CHOSEN's ball screen,
apply a raw button sequence, read the resulting party species. Ground truth, no
color heuristics. Usage:
  PYTHONPATH=. python -m collection.playthrough.probe_balls "right,right,a,a"
"""
import sys, json
from collection.catalog import MILESTONE_ORDER, discover_heatz_events
from collection.direct_runner import DirectEmulatorRunner
import collection.collect_events as ce
from collection.heatz_adapter import build_heatz_state, _starter_ui_active
from collection.playthrough.spine import run_milestone

SEQ = (sys.argv[1] if len(sys.argv) > 1 else "right,right,a,a").split(",")
POLICY = "../pokeagent-solution/expert_policies_by_llm"
events = discover_heatz_events(POLICY); by_id = {e["event_id"]: e for e in events}
order = [m for m in MILESTONE_ORDER if m in by_id]

runner = DirectEmulatorRunner(rom_path="Emerald-GBAdvance/rom.gba", load_state=None)
runner.initialize()
for _ in range(600):
    runner.step_frame(["a"], phase="boot")
    if runner.state().game_state not in ("title", "intro", None):
        break
try:
    # spine up to (not including) STARTER_CHOSEN
    for event_id in order:
        if event_id == "STARTER_CHOSEN":
            break
        exp = ce._load_expected_state(rom_path="Emerald-GBAdvance/rom.gba",
                                      completed_state=by_id[event_id].get("completed_state"), event_id=event_id)
        run_milestone(runner, event_id=event_id, policy_dir=POLICY, expected_state=exp,
                      postcondition=by_id[event_id].get("postcondition", event_id), start_money=0, max_actions=6000)
    # drive into STARTER_CHOSEN until the ball UI is active (walk to bag + open)
    from collection.heatz_adapter import HeatzPolicy
    from collection.actions import normalize_action
    pol = HeatzPolicy("STARTER_CHOSEN", f"{POLICY}/STARTER_CHOSEN/STARTER_CHOSEN.py")
    for _ in range(2000):
        if _starter_ui_active(runner.env):
            break
        h = build_heatz_state(runner.env, frame_idx=runner.frame_idx, story_bucket="STARTER_CHOSEN", facing=runner.facing, include_map=True)
        runner.perform_action(normalize_action(pol.act(h)), metadata={})
    ui = _starter_ui_active(runner.env)
    # apply the RAW sequence
    for b in SEQ:
        runner.perform_action(b.strip(), metadata={})
    # let it settle
    for _ in range(180):
        runner.step_frame([], phase="settle")
    st = runner.state()
    party = [(m.get("species"), m.get("level")) for m in (st.party_summary or [])]
    print(json.dumps({"seq": SEQ, "ball_ui_reached": bool(ui), "party": party}))
finally:
    runner.close()
