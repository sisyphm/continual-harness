"""Run one milestone on an ALREADY-INITIALIZED runner (continuous, no state load).

This is the per-event action loop from collect_events.collect_one_event, lifted to
operate on a shared runner: no runner construction, no pre_state load, no recorder
management. All decision predicates are imported from collect_events (single source
of truth — collect_events itself is untouched and keeps working).
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any

from collection.actions import ActionTiming, normalize_action
from collection.heatz_adapter import (
    HeatzPolicy, build_heatz_state, grind_action, heal_action, may_heal_action,
    _starter_ui_active,
)
import collection.collect_events as ce


# W33 special-UI fix (clock + starter bag): inside these screens the pacing settle
# predicate is blind (nothing the nav snapshot watches ever changes there), so
# condition pacing floors every tap at ~8 frames and outruns the UI's
# fade-ins/confirm transitions (proven wedges: CLOCK_INTERACT's confirm cycle,
# STARTER_CHOSEN's mudkip cursor trip). Actions taken while one of these UIs is on
# screen use the pre-W33 fixed schedule instead — scoped to the flagged UIs, never
# a global slowdown.
_SPECIAL_UI_TIMING = ActionTiming(hold_frames=12, release_frames=48)


def run_milestone(
    runner,
    *,
    event_id: str,
    policy_dir: str,
    expected_state,
    postcondition: str,
    start_money: int,
    # action-count defaults rescaled for condition-based pacing (W33 §3.5): ~3x more actions per frame
    max_actions: int = 12000,
    min_actions: int = 10,
    stall_actions: int = 1200,
    blocked_nav_actions: int = 120,
    starter: str = "mudkip",
    tic_fn=None,
    max_wall_s: float = 480.0,
) -> dict:
    """Drive `runner` with policy `event_id` until its postcondition. Returns a dict
    {validation: passed|failed|skipped, failure_reason, actions_taken, start_frame,
    end_frame}. Mirrors collect_one_event's loop 1:1 (same predicates/branches).

    `tic_fn` (W33 §14.3 micro-behavior noise): optional no-arg callable invoked once
    per action-loop iteration, before the policy reads state. The director passes a
    persona-seeded closure that occasionally emits a recorded human tic (short pause /
    facing flick) through runner.step_frame — during a solve-then-record DRY attempt
    those frames land in the capture log and are replayed like everything else, so
    the tic is simply part of the button schedule.

    `max_wall_s` (W33 loop-bounding invariant): every attempt is bounded by ACTIONS
    and by WALL TIME. The action-denominated stall detector is resettable — wild
    battles/dialogs zero it — so a stranded policy chasing an unreachable goal
    through encounter territory (the RUSTBORO_CENTER_ENTERED wedge: blocks left the
    runner mid-battle on ROUTE 104, the policy then pathed its Rustboro door coords
    against the wrong map for hours) could otherwise outlive any frame budget in
    wall-clock terms. Breaching the cap fails the attempt like a stall; the
    solve-then-record retry/abort machinery above stays in charge."""
    policy = HeatzPolicy(event_id, Path(policy_dir) / event_id / f"{event_id}.py")
    t_start = time.monotonic()
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
    _crossed_once = False                 # off-map pre-check fires at most once per attempt
    _adjacent_once = False                # blocked-goal pre-check, likewise
    battle_stall = 0                      # consecutive in-battle iterations with no frames
    nav_stall = 0                         # ditto, out of battle (map-edge / blocked goal)
    last_frame_seen = runner.frame_idx

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
        if max_wall_s and time.monotonic() - t_start > max_wall_s:
            failure_reason = "wall_time_exceeded"
            break
        # BATTLE WEDGE BREAKER (W33, measured on five runs at once). A TRAINER battle
        # the policy tries to RUN from deadlocks: RUN is refused, the party read fails
        # ("8650 is not a valid Move"), so the policy stops seeing a battle and paths
        # instead — printing "No progress possible toward (x, y)" forever while
        # stepping ZERO frames, which no action or wall budget can catch because
        # neither advances. Route 116, the leg-2 grind map, is thick with trainer
        # sight lines, so this stalled every run that reached it. Only fires on a
        # PROVEN stall (in battle, frame counter frozen across iterations), so a
        # healthy battle keeps whatever the policy was doing.
        if runner.nav_state().in_battle:
            # Count TIME IN BATTLE, not frozen frames. A deadlocked trainer battle
            # still burns frames in its input holds (measured: exp_001 sat in one
            # battle for 41k ticks at 3 HP, experience frozen at 1890, while stepping
            # 10k frames/70s — so a frame-progress check called it healthy). A real
            # battle resolves in a few hundred iterations; thousands means deadlock.
            battle_stall += 1
            if battle_stall >= 240:
                from collection.playthrough.blocks.base import force_fight
                from collection.playthrough.blocks.grind_evolve import await_overworld
                if force_fight(runner):
                    await_overworld(runner, phase="spine")
                    battle_stall = 0
                else:
                    # Could not play the battle out either. NEVER spin: fail the
                    # milestone so the attempt/retry machinery takes over — a run that
                    # dies in 30 seconds and is re-run beats one that burns a worker
                    # for 40 minutes and fails anyway.
                    failure_reason = "battle_wedge_unrecoverable"
                    break
            else:
                nav_stall = 0
        else:
            battle_stall = 0
            # NAVIGATION stall: same zero-frame spin without a battle. exp_004 froze on
            # goal (16, -1) — a map-edge crossing — printing "No progress possible"
            # while the frame counter never moved, so no action or wall budget applied.
            # Fail the milestone instead of spinning; retry/re-run is always cheaper.
            nav_stall = nav_stall + 1 if runner.frame_idx == last_frame_seen else 0
            if nav_stall >= 200:
                # WRONG-MAP GOAL. The policy paths the NEXT map's coordinates
                # against the CURRENT map's grid, so the goal lands OUTSIDE the map
                # and no route can reach it. Measured: exp_008_treecko stood in
                # OLDALE TOWN pathing to (4,-1) — Oldale's north exit is x=8..11 —
                # because the target belonged to ROUTE 103. It died there under two
                # different persona seeds while 30 other runs walked through.
                # An out-of-bounds goal IS the direction to travel, so cross that
                # connection with our navigator (which handles edges properly).
                _moved = False
                try:
                    from collection import navigator as _nav
                    _t, _, _ = _nav._state(runner)
                    _gx = getattr(expected_state, "x", None)
                    _gy = getattr(expected_state, "y", None)
                    if _t is not None and _gx is not None and _gy is not None:
                        _dir = (2 if _gy < 0 else 1 if _gy >= _t.map_height else
                                3 if _gx < 0 else 4 if _gx >= _t.map_width else None)
                        if _dir is not None:
                            _r = _nav.cross_connection(runner, _nav.MapKnowledge(), _dir,
                                                       budget=20_000)
                            _moved = _r == "crossed"
                            print(f"nav wedge: goal ({_gx},{_gy}) is off-map "
                                  f"{_t.map_width}x{_t.map_height}; crossed dir {_dir}: {_r}",
                                  flush=True)
                except Exception as _e:
                    print(f"nav wedge cross failed: {_e!r}", flush=True)
                # IN-MAP blocked goal — 54 of 91 failed attempts today, the single
                # biggest failure mode. The policy's goal is often an NPC's own tile
                # (TRAINER_JOSH_BATTLE targets (5,3), where Josh stands), which is by
                # definition unwalkable, so it reports "no progress" forever. Our
                # navigator reads collision and elevation from RAM and remembers
                # refused tiles, so hand it the goal AND its neighbours and let it
                # walk us adjacent — adjacency is all a talk/sight trigger needs.
                if not _moved:
                    try:
                        import numpy as _np
                        from collection import navigator as _nav
                        _t, _, _ = _nav._state(runner)
                        _gx = getattr(expected_state, "x", None)
                        _gy = getattr(expected_state, "y", None)
                        # ONLY when we are genuinely on the goal's map. Bounds alone
                        # cannot tell: TRAINER_JOSH_BATTLE and ROXANNE_BATTLE both
                        # target (5,3) inside RUSTBORO CITY GYM, and (5,3) also lies
                        # inside Route 116's bounds — walking there would be nonsense.
                        # Compare map NAMES, which both sides carry.
                        _same_map = (getattr(expected_state, "map", None)
                                     and runner.nav_state().map == expected_state.map)
                        if _same_map and _t is not None and _gx is not None and _gy is not None \
                                and 0 <= _gx < _t.map_width and 0 <= _gy < _t.map_height:
                            def _goal(t, beh, gx=_gx, gy=_gy):
                                m = _np.zeros(t.grid.shape, bool)
                                for dx, dy in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                                    yy, xx = gy + dy + 7, gx + dx + 7
                                    if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]:
                                        m[yy, xx] = True
                                return m & (((t.grid >> 10) & 3) == 0)
                            _r = _nav.goto(runner, _nav.MapKnowledge(), _goal,
                                           budget=15_000, phase="spine")
                            _moved = _r in ("arrived", "battle")
                            print(f"nav wedge: our navigator to ({_gx},{_gy})+adj -> {_r}",
                                  flush=True)
                    except Exception as _e:
                        print(f"nav wedge in-map fallback failed: {_e!r}", flush=True)
                nav_stall = 0
                if not _moved:
                    # Wrong-map goal we cannot cross to (e.g. a gym interior reached
                    # through a door): fail fast so retry/re-seed takes over instead
                    # of spinning. Recorded distinctly so the audit can count them.
                    _em = getattr(expected_state, "map", None)
                    failure_reason = ("nav_wedge_wrong_map"
                                      if _em and runner.nav_state().map != _em
                                      else "nav_wedge_no_frame_progress")
                    break
        last_frame_seen = runner.frame_idx
        if tic_fn is not None:
            tic_fn()
        # OFF-MAP GOAL PRE-CHECK. Do not wait for a stall: the policy retries for
        # minutes while the frame counter keeps moving, so a stall detector never
        # trips and the fallback below was dead code (measured: exp_008_treecko
        # failed 8 times on goal (4,-1) with zero fallback fires). If the target lies
        # outside this map, the ONLY way there is the connection in that direction,
        # so cross it now with our navigator. Once per milestone attempt.
        if not _crossed_once and expected_state is not None:
            _gx = getattr(expected_state, "x", None)
            _gy = getattr(expected_state, "y", None)
            if _gx is not None and _gy is not None:
                from collection import navigator as _nav
                _t, _, _ = _nav._state(runner)
                if _t is not None and not (0 <= _gx < _t.map_width
                                           and 0 <= _gy < _t.map_height):
                    _d = (2 if _gy < 0 else 1 if _gy >= _t.map_height else
                          3 if _gx < 0 else 4)
                    _crossed_once = True
                    _mk = _nav.MapKnowledge()
                    _key = f"{_t.map_group},{_t.map_num}"
                    _conns = {c.get("direction") for c in _mk.connections.get(_key, [])}
                    if _d in _conns:
                        _r = _nav.cross_connection(runner, _mk, _d, budget=20_000)
                        _how = f"crossed dir {_d}"
                    else:
                        # No connection that way: this is an interior, and a building is
                        # left through a scripted door the policy already knows. Do
                        # NOTHING rather than invent a route — an earlier version walked
                        # at the wall of Birch's lab (13x13, goal y=17) and then tried
                        # its warps, both 'stuck', purely wasting the budget.
                        _r = "skipped (interior)"
                        _how = f"no conn dir {_d}"
                    print(f"spine pre-check: goal ({_gx},{_gy}) off-map "
                          f"{_t.map_width}x{_t.map_height}; {_how} -> {_r}", flush=True)
        # BLOCKED-GOAL PRE-CHECK (same reasoning as the off-map one above: do not wait
        # for a stall, because during a dry solve the frame counter keeps moving and no
        # stall is ever detected). If the goal tile is on THIS map but not walkable, it
        # is an NPC's own square — TRAINER_JOSH_BATTLE and ROXANNE_BATTLE both target
        # (5,3) inside RUSTBORO CITY GYM, where Josh stands — so no route can ever end
        # there and the policy prints 'No progress possible' forever. Adjacency is all
        # a talk or sight trigger needs, so walk next to it with our navigator.
        if not _adjacent_once and expected_state is not None:
            _gx = getattr(expected_state, "x", None)
            _gy = getattr(expected_state, "y", None)
            _em = getattr(expected_state, "map", None)
            if _gx is not None and _gy is not None and _em \
                    and runner.nav_state().map == _em:
                import numpy as _np
                from collection import navigator as _nav
                _t, _, _ = _nav._state(runner)
                if _t is not None and 0 <= _gx < _t.map_width and 0 <= _gy < _t.map_height \
                        and ((_t.grid[_gy + 7, _gx + 7] >> 10) & 3) != 0:
                    _adjacent_once = True

                    def _goal(t, beh, gx=_gx, gy=_gy):
                        m = _np.zeros(t.grid.shape, bool)
                        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                            yy, xx = gy + dy + 7, gx + dx + 7
                            if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]:
                                m[yy, xx] = True
                        return m & (((t.grid >> 10) & 3) == 0)

                    _r = _nav.goto(runner, _nav.MapKnowledge(), _goal,
                                   budget=12_000, phase="spine")
                    print(f"spine pre-check: goal ({_gx},{_gy}) is a blocked tile on "
                          f"{_em}; walked adjacent -> {_r}", flush=True)
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
        clock_ui = ce._visual_clock_ui(runner.env)
        starter_ui = event_id == "STARTER_CHOSEN" and _starter_ui_active(runner.env)
        # Conservative pacing only inside the flagged special UIs; the starter gate
        # additionally requires pre-pick out-of-battle state so the rescue battle's
        # (equally white) text box never slows normal battle handling.
        special_ui_timing = clock_ui or (
            starter_ui and not current_before_action.in_battle
            and not (current_before_action.party_summary or []))
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
            special_ui_active = clock_ui or starter_ui
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
        current_state = runner.perform_action(
            action,
            timing=_SPECIAL_UI_TIMING if special_ui_timing else None,
            metadata={"event_id": event_id, "policy_source": policy_source})
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
