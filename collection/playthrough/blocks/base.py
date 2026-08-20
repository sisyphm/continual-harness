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


def run_block(runner, block, *, max_actions: int = 900) -> dict:  # rescaled for condition-based pacing (W33 §3.5)
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


def flee_battle(runner, max_actions: int = 40) -> bool:
    """Escape a wild battle (RUN = bottom-right, then confirm) — collect_coverage's
    proven `_flee_battle` pattern, local so blocks don't import the coverage collector."""
    for _ in range(max_actions):
        for act in ("B", "DOWN", "RIGHT", "A"):
            runner.perform_action(act, speed="fast", record_end_state=False)
        if not runner.nav_state().in_battle:
            return True
    return not runner.nav_state().in_battle


def run_nav_block(runner, block, *, mk=None, return_budget: int = 30000) -> dict:
    """W33 §2 wrapper for NAVIGATOR-driven expedition blocks (bfs_sweep, encounter_farm):
    same safety contract as run_block — precondition (free overworld, no battle/dialog,
    else skip cleanly), bounded budgets (the block's own frame budgets), forced
    anchor-return — but the block steers the runner directly through the navigator
    instead of emitting one button per act, and battles/dialogs are handled INSIDE its
    machinery (goto interrupts on battle; blocks flee and resume). The block implements
    `.name`, `.phase` and `.run(runner, mk, ctx) -> summary dict`.

    Recording: `runner.set_phase(block.phase)` stamps every frame + drops the boundary
    savestate on entry; the phase is restored to "spine" on exit NO MATTER how the block
    ends, so a block can never mis-tag spine frames."""
    from collection import navigator as nav
    if mk is None:
        mk = nav.MapKnowledge()
    start_frame = runner.frame_idx
    # Same settle/precondition seam as run_block. Dialog check is the VISION-validated
    # heatz one — the raw BG0 window mask false-reads on stale post-close tiles (the
    # documented no_dialog* limitation) and would veto perfectly clean states.
    st0 = runner.state()
    for _ in range(40):                              # settle a spine hand-off script tail
        if not st0.in_battle and st0.control_mode == "free_overworld" and not is_dialog_open(_hstate(runner)):
            break
        runner.step_frame([], phase="block_settle")
        st0 = runner.state()
    t, x, y = nav._state(runner)
    if st0.in_battle or st0.control_mode != "free_overworld" or is_dialog_open(_hstate(runner)) or t is None:
        return dict(block=block.name, ran=False, reason="precondition_not_overworld",
                    frames=runner.frame_idx - start_frame)
    anchor = (f"{t.map_group},{t.map_num}", x, y)
    runner.set_phase(block.phase)
    summary: dict = {}
    try:
        summary = block.run(runner, mk, {"anchor": anchor})
    except Exception as e:
        # block-never-breaks-the-spine: the error is RECORDED, never propagated; the
        # anchor return + phase restore below still run.
        summary["error"] = repr(e)
    returned = False
    try:
        returned = _return_to_anchor_nav(runner, mk, anchor, budget=return_budget)
    except Exception as e:
        # a second failure must not mask the first — both end up in the summary
        summary["return_error"] = repr(e)
    finally:
        runner.set_phase("spine")
    # `frames` (in **summary) = the block's own work; frames_total adds settle + return.
    return dict(block=block.name, ran=True, anchor=list(anchor), returned=returned,
                frames_total=runner.frame_idx - start_frame, **summary)


def _return_to_anchor_nav(runner, mk, anchor, *, budget: int = 30000) -> bool:
    """Cross-map anchor return for nav blocks (a sweep/farm legitimately ends on another
    map — base's porymap `find_path_action` return is same-map only)."""
    from collection import navigator as nav
    amap, ax, ay = anchor
    deadline = runner.frame_idx + budget

    def goal(t, beh):
        import numpy as np
        m = np.zeros(t.grid.shape, bool)
        if 0 <= ay + 7 < m.shape[0] and 0 <= ax + 7 < m.shape[1]:
            m[ay + 7, ax + 7] = True
        return m

    for _ in range(6):
        if runner.nav_state().in_battle:
            flee_battle(runner)
            continue
        t, x, y = nav._state(runner)
        if t is None:
            nav._hold(runner, [], 30, "block_return")
            continue
        if f"{t.map_group},{t.map_num}" == amap and (x, y) == (ax, ay):
            return True
        if runner.frame_idx >= deadline:
            break
        if f"{t.map_group},{t.map_num}" != amap:
            nav.goto_map(runner, mk, amap, hop_budget=max(2000, deadline - runner.frame_idx))
        else:
            nav.goto(runner, mk, goal, budget=max(2000, deadline - runner.frame_idx),
                     phase="block_return")
    t, x, y = nav._state(runner)
    return t is not None and f"{t.map_group},{t.map_num}" == amap and (x, y) == (ax, ay)


def _return_to_anchor(runner, anchor, name, *, max_actions: int = 360) -> bool:  # rescaled for condition-based pacing (W33 §3.5)
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
