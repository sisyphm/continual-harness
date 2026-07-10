"""Run one milestone on an ALREADY-INITIALIZED runner (continuous, no state load).

This is the per-event action loop from collect_events.collect_one_event, lifted to
operate on a shared runner: no runner construction, no pre_state load, no recorder
management. All decision predicates are imported from collect_events (single source
of truth — collect_events itself is untouched and keeps working).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

from collection.actions import normalize_action
from collection.heatz_adapter import (
    HeatzPolicy, build_heatz_state, grind_action, heal_action, may_heal_action,
    _starter_ui_active,
)
import collection.collect_events as ce


def run_milestone(
    runner,
    *,
    event_id: str,
    policy_dir: str,
    expected_state,
    postcondition: str,
    start_money: int,
    max_actions: int = 4000,
    min_actions: int = 10,
    stall_actions: int = 400,
    blocked_nav_actions: int = 40,
    starter: str = "mudkip",
) -> dict:
    """Drive `runner` with policy `event_id` until its postcondition. Returns a dict
    {validation: passed|failed|skipped, failure_reason, actions_taken, start_frame,
    end_frame}. Mirrors collect_one_event's loop 1:1 (same predicates/branches)."""
    policy = HeatzPolicy(event_id, Path(policy_dir) / event_id / f"{event_id}.py")
    start_frame = runner.frame_idx
    start_state = runner.state()
    validation = "failed"
    failure_reason = None
    actions_taken = 0
    prev_action = None
    stuck_count = 0
    grind_state: dict = {}
    heal_state: dict = {}
    may_heal_state: dict = {}
    last_progress_key = ce._state_key(start_state)
    target_tail_actions = 0
    target_tail_stall_count = 0
    last_target_tail_key = None
    no_structural_progress_count = 0
    last_structural_progress_key = ce._structural_progress_key(start_state)
    blocked_nav_count = 0
    last_blocked_nav_key = None

    accept_unresponsive_target = event_id in ce._EVENTS_ACCEPTING_UNRESPONSIVE_TARGET

    # Already complete on entry (common in continuous mode: handoff lands us past the gate).
    if (
        ce._responsive_target_reached(runner, expected_state)
        or ce._semantic_postcondition_met(event_id, runner, runner.state(), start_money)
        or (accept_unresponsive_target and expected_state is not None
            and ce._target_reached(runner.state(), expected_state))
    ):
        return dict(validation="skipped", failure_reason="already_complete_start",
                    actions_taken=0, start_frame=start_frame, end_frame=runner.frame_idx)

    for _ in range(max_actions):
        policy_source = "heatz"
        current_before_action = runner.state()
        if not heal_state.get("active") and ce._postcondition_met(
            runner, postcondition, min_actions_met=actions_taken >= min_actions,
            expected_state=expected_state, accept_unresponsive_target=accept_unresponsive_target,
            event_id=event_id, start_money=start_money, current=current_before_action):
            validation = "passed"
            break
        healing = not current_before_action.in_battle and ce._should_heal(current_before_action, heal_state, event_id)
        may_healing = event_id == "MAY_ROUTE103_INTERACTION" and ce._may_heal_needed(current_before_action, may_heal_state)
        if may_healing:
            healing = False
        visible_dialog = ce._visible_dialog_open(runner)
        h_state = build_heatz_state(
            runner.env, frame_idx=runner.frame_idx, story_bucket=event_id,
            facing=runner.facing, include_map=event_id in ce._EVENTS_REQUIRING_MAP_STATE)
        h_state["_target_starter"] = starter   # steers _select_starter_action (STARTER_CHOSEN)
        if prev_action:
            h_state["prev_action"] = prev_action.lower()

        action = None
        if healing:
            if expected_state is not None and expected_state.x is not None and expected_state.y is not None:
                h_state["_heal_goal"] = (expected_state.map, expected_state.x, expected_state.y)
            heal_raw = heal_action(h_state)
            if heal_raw == "heal_skip":
                heal_state["active"] = False
                heal_state["skip"] = True
                heal_state.pop("settle_key", None)
                healing = False
            else:
                action = normalize_action(heal_raw)
                policy_source = "heatz_heal"
                stuck_count = 0

        if may_healing and not current_before_action.in_battle:
            action = normalize_action(may_heal_action(h_state))
            policy_source = "heatz_may_heal"
            stuck_count = 0
        elif healing and action is not None:
            pass
        elif not current_before_action.in_battle and ce._grind_target(event_id, current_before_action) is not None:
            action = normalize_action(grind_action(h_state, grind_state))
            policy_source = "heatz_grind"
            stuck_count = 0
        elif action is None:
            action = normalize_action(policy.act(h_state))
            at_physical_target = expected_state is not None and ce._physical_target_reached(current_before_action, expected_state)
            at_exact_target = (
                expected_state is not None and current_before_action.map == expected_state.map
                and current_before_action.x == expected_state.x and current_before_action.y == expected_state.y)
            policy_idle = action == "WAIT"
            can_plan = ce._can_plan_from_state(current_before_action, visible_dialog=visible_dialog)
            special_ui_active = ce._visual_clock_ui(runner.env) or (
                event_id == "STARTER_CHOSEN" and _starter_ui_active(runner.env))
            if not special_ui_active and policy_idle and at_physical_target and visible_dialog:
                action = ce._target_tail_action(target_tail_actions); target_tail_actions += 1
                policy_source = "heatz_script_tail_settle"; stuck_count = 0
            elif not special_ui_active and policy_idle and at_exact_target and not visible_dialog and not ce._movement_responsive(runner):
                action = ce._target_tail_action(target_tail_actions); target_tail_actions += 1
                policy_source = "heatz_script_tail_settle"; stuck_count = 0
            elif not special_ui_active and expected_state is not None and can_plan and (stuck_count >= 5 or action == "WAIT"):
                fb = ce._plan_first_action_to_expected(runner, expected_state)
                if fb:
                    action = normalize_action(fb); policy_source = "heatz_emulator_bfs_fallback"; stuck_count = 0



        state_before_perform = current_before_action
        current_state = runner.perform_action(action, metadata={"event_id": event_id, "policy_source": policy_source})
        prev_action = action
        actions_taken += 1
        progress_key = ce._state_key(current_state)
        if progress_key == last_progress_key and action in ce._NAV_ACTIONS:
            stuck_count += 1
        else:
            stuck_count = 0
            last_progress_key = progress_key

        visible_dialog_after = ce._visible_dialog_open(runner)
        structural_key = ce._structural_progress_key(current_state)
        if visible_dialog_after or current_state.in_battle:
            no_structural_progress_count = 0
            last_structural_progress_key = structural_key
        elif structural_key == last_structural_progress_key:
            no_structural_progress_count += 1
        else:
            no_structural_progress_count = 0
            last_structural_progress_key = structural_key
        if stall_actions > 0 and no_structural_progress_count >= stall_actions:
            failure_reason = "stalled_no_structural_progress"
            break

        if action in ce._NAV_ACTIONS and not current_state.in_battle and not state_before_perform.in_battle \
                and state_before_perform.map == current_state.map and state_before_perform.x == current_state.x \
                and state_before_perform.y == current_state.y:
            blocked_nav_key = (current_state.map, current_state.x, current_state.y, action)
            if blocked_nav_key == last_blocked_nav_key:
                blocked_nav_count += 1
            else:
                blocked_nav_count = 1
                last_blocked_nav_key = blocked_nav_key
            if blocked_nav_actions > 0 and blocked_nav_count >= blocked_nav_actions:
                failure_reason = "blocked_repeated_navigation"
                break
        else:
            blocked_nav_count = 0
            last_blocked_nav_key = None

        if expected_state is not None and ce._physical_target_reached(current_state, expected_state) and policy_source == "heatz_script_tail_settle":
            tail_key = ce._tail_progress_key(current_state, ce._visible_dialog_open(runner))
            if tail_key == last_target_tail_key:
                target_tail_stall_count += 1
            else:
                target_tail_stall_count = 0
                last_target_tail_key = tail_key
            if target_tail_stall_count >= ce._TARGET_TAIL_STALL_LIMIT:
                failure_reason = "target_tail_stalled"
                break
        else:
            target_tail_stall_count = 0
            last_target_tail_key = None

        if not heal_state.get("active") and ce._postcondition_met(
            runner, postcondition, min_actions_met=actions_taken >= min_actions,
            expected_state=expected_state, accept_unresponsive_target=accept_unresponsive_target,
            event_id=event_id, start_money=start_money, current=current_state):
            validation = "passed"
            break

    if validation != "passed" and failure_reason is None:
        failure_reason = "max_actions_exhausted"
    return dict(validation=validation, failure_reason=failure_reason, actions_taken=actions_taken,
                start_frame=start_frame, end_frame=runner.frame_idx)
