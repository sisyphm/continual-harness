"""Life-block base wrapper: run a seeded 'living' behaviour between milestones on the
SHARED runner, with hard safety so a block can never break the spine.

Contract (agreed S2 spec):
  preconditions  — overworld, not in battle, no dialog (else skip the block cleanly)
  safety intercepts — battle → handle_battle('run'); dialog → navigate_ui(confirm)
  frame budget   — bounded; on exhaustion, abort
  abort/finish   — path back to the ENTRY tile (anchor) so the next milestone starts
                   from the same place the spine handed off
  logging        — returns an outcome dict for meta.json
The block itself only implements `act(state, ctx)` → button, and signals done.
"""
from __future__ import annotations
from typing import Any, Callable

from collection.actions import normalize_action
from collection.heatz_adapter import (
    build_heatz_state, handle_battle, navigate_ui, is_dialog_open, find_path_action,
)
import collection.collect_events as ce


def _hstate(runner, event_id="LIFE_BLOCK"):
    h = build_heatz_state(runner.env, frame_idx=runner.frame_idx, story_bucket=event_id,
                          facing=runner.facing, include_map=True)
    return h


def run_block(runner, block, *, max_actions: int = 300) -> dict:
    """Drive `runner` with `block.act` under the safety wrapper. `block` needs:
       .name, .setup(state)->anchor(x,y,map), .act(state, ctx)->(button|None; None=done)."""
    start_frame = runner.frame_idx
    # Settle: the spine may hand off on a script tail (a "skipped/already-complete"
    # milestone). Advance a few frames until free overworld control, else skip cleanly.
    st0 = runner.state()
    for _ in range(40):
        if not st0.in_battle and st0.control_mode == "free_overworld" and not is_dialog_open(_hstate(runner)):
            break
        runner.step_frame([], phase="block_settle")
        st0 = runner.state()
    if st0.in_battle or is_dialog_open(_hstate(runner)) or st0.control_mode != "free_overworld":
        return dict(block=block.name, ran=False, reason="precondition_not_overworld",
                    frames=runner.frame_idx - start_frame, actions=0)
    anchor = (st0.map, st0.x, st0.y)
    ctx: dict[str, Any] = {"anchor": anchor}
    block.setup(_hstate(runner), ctx)
    actions = 0
    aborted = None
    for _ in range(max_actions):
        st = runner.state()
        h = _hstate(runner)
        # Safety intercepts first — a wandered-into battle/dialog never derails the run.
        if st.in_battle:
            ctx["_in_battle"] = True
            strat = getattr(block, "battle_strategy", "run")
            runner.perform_action(normalize_action(handle_battle(h, strategy=strat)),
                                  metadata={"block": block.name, "src": "battle"})
            actions += 1
            continue
        if ctx.pop("_in_battle", False):
            ctx["encounters"] = ctx.get("encounters", 0) + 1  # just exited a battle
        if is_dialog_open(h):
            runner.perform_action(normalize_action(navigate_ui(h, intent="confirm")),
                                  metadata={"block": block.name, "src": "safety_dialog"})
            actions += 1
            continue
        btn = block.act(h, ctx)
        if btn is None:
            break
        runner.perform_action(normalize_action(btn), metadata={"block": block.name})
        actions += 1
    else:
        aborted = "max_actions"
    # Return to the anchor tile so the next milestone starts where the spine left off.
    returned = _return_to_anchor(runner, anchor, block.name)
    return dict(block=block.name, ran=True, reason=aborted or "done", returned=returned,
                anchor=list(anchor), actions=actions, encounters=ctx.get("encounters", 0),
                frames=runner.frame_idx - start_frame)


def _return_to_anchor(runner, anchor, name, *, max_actions: int = 120) -> bool:
    amap, ax, ay = anchor
    for _ in range(max_actions):
        st = runner.state()
        if st.in_battle:
            runner.perform_action(normalize_action(handle_battle(_hstate(runner), strategy="run")),
                                  metadata={"block": name, "src": "return_battle"})
            continue
        if st.map == amap and st.x == ax and st.y == ay:
            return True
        h = _hstate(runner)
        if is_dialog_open(h):
            runner.perform_action("a", metadata={"block": name, "src": "return_dialog"})
            continue
        btn = find_path_action(h, ax, ay)
        runner.perform_action(normalize_action(btn), metadata={"block": name, "src": "return"})
    st = runner.state()
    return st.map == amap and st.x == ax and st.y == ay
