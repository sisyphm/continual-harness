"""Continuous playthrough director: one session, all milestones, no state loads."""
from __future__ import annotations
import json
import time
from pathlib import Path

from collection.catalog import MILESTONE_ORDER, discover_heatz_events
from collection.direct_runner import DirectEmulatorRunner
from collection.recorder import ChunkRecorder
from collection.world_model_sink import WorldModelSink
import collection.collect_events as ce
from collection.playthrough.spine import run_milestone


def run_playthrough(*, policy_dir: str, out_dir: str, rom_path: str = "Emerald-GBAdvance/rom.gba",
                    starter: str = "mudkip", seed: int = 0, record: bool = True,
                    stop_after: str | None = None, per_milestone_max: int = 18000,  # rescaled for condition-based pacing (W33 §3.5)
                    blocks: bool = False, expedition: list[dict] | None = None) -> dict:
    from collection.playthrough.schedule import build_expedition_schedule, build_schedule
    from collection.playthrough.blocks.base import run_block, run_nav_block
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule(seed) if blocks else {}
    # W33 §2: explicit expedition-block entries ([{after, block, **kwargs}]) merged in.
    for mid, blks in build_expedition_schedule(expedition or []).items():
        schedule.setdefault(mid, []).extend(blks)
    nav_mk = None
    block_log = []
    ce.set_expected_starter(starter.capitalize())       # STARTER_CHOSEN gate holds THIS species
    events = discover_heatz_events(policy_dir)          # ordered, chained
    by_id = {e["event_id"]: e for e in events}
    order = [m for m in MILESTONE_ORDER if m in by_id]  # skip GAME_RUNNING (no policy row)

    recorder_cm = ChunkRecorder(str(out), run_id=f"playthrough_{starter}_s{seed}",
                                visual_fps=1000 if record else 30, backend="npz",
                                metadata={"kind": "playthrough", "starter": starter, "seed": seed})
    results = []
    t_run = time.time()
    with recorder_cm as recorder:
        runner = DirectEmulatorRunner(rom_path=rom_path, load_state=None,
                                      recorder=recorder if record else None)
        runner.initialize()
        sink = WorldModelSink(str(out)) if record else None
        if sink is not None:
            runner.frame_hook = sink.capture
        # Boot past the title screen: mash A/START until GAME_RUNNING (new game begins).
        for _ in range(600):
            runner.step_frame(["a"], phase="boot")
            st = runner.state()
            if st.game_state not in ("title", "intro", None):
                break
        start_money = 0
        try:
            for event_id in order:
                exp = ce._load_expected_state(rom_path=rom_path,
                                              completed_state=by_id[event_id].get("completed_state"),
                                              event_id=event_id)
                post = by_id[event_id].get("postcondition", event_id)
                t0 = time.time()
                f0 = runner.frame_idx
                r = run_milestone(runner, event_id=event_id, policy_dir=policy_dir,
                                  expected_state=exp, postcondition=post, start_money=start_money,
                                  max_actions=per_milestone_max, starter=starter)
                r.update(event_id=event_id, wall_s=round(time.time() - t0, 1),
                         frames=runner.frame_idx - f0)
                results.append(r)
                st = runner.state()
                print(json.dumps({**r, "map": st.map, "gs": st.game_state}), flush=True)
                if r["validation"] not in ("passed", "skipped"):
                    r["FAILED_RUN_HERE"] = True
                    break
                # Life blocks scheduled after this milestone (diversity injection).
                for blk in schedule.get(event_id, []):
                    if hasattr(blk, "run"):          # navigator-driven expedition block (W33)
                        if nav_mk is None:
                            from collection.navigator import MapKnowledge
                            nav_mk = MapKnowledge(rom_path=rom_path)
                        outcome = run_nav_block(runner, blk, mk=nav_mk)
                    else:
                        outcome = run_block(runner, blk)
                    outcome["after"] = event_id
                    block_log.append(outcome)
                    print(json.dumps({"BLOCK": outcome}), flush=True)
                if stop_after and event_id == stop_after:
                    break
        finally:
            if sink is not None:
                sink.close()
            total_frames = runner.frame_idx
            runner.close()
    summary = dict(starter=starter, seed=seed, milestones_total=len(order),
                   milestones_passed=sum(1 for r in results if r["validation"] in ("passed", "skipped")),
                   total_frames=total_frames, wall_s=round(time.time() - t_run, 1), results=results,
                   blocks=block_log)
    (out / "playthrough_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
