"""Collect Heatz scripted event segments with direct emulator stepping."""

from __future__ import annotations

import argparse
import json
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from collection.actions import normalize_action, run_action_frames, timing_for, update_facing
from collection.catalog import EVENT_POSTCONDITION_ALIAS, discover_heatz_events
from collection.direct_runner import DirectEmulatorRunner
from collection.heatz_adapter import HeatzPolicy, _starter_ui_active, _visual_clock_ui, _visual_dialog_open, build_heatz_state, grind_action, heal_action, may_heal_action
from collection.recorder import ChunkRecorder
from collection.world_model_sink import WorldModelSink
from collection.state import control_mode as classify_control_mode, location_name


def _full_state_for_milestones(env):
    screenshot = env.get_screenshot()
    return env.get_comprehensive_state(screenshot=screenshot)


def _load_expected_state(*, rom_path: str, completed_state: str | None, event_id: str):
    if not completed_state:
        return None
    probe = DirectEmulatorRunner(rom_path=rom_path, load_state=completed_state, story_bucket=event_id)
    try:
        probe.initialize()
        state = probe.state()
        # Some checkpoints are corrupt and load as the title screen. A bogus position
        # target would wedge the postcondition on _responsive_target_reached and never
        # fall through to the milestone check, so treat it as "no expected state".
        if str(state.map or "").upper().replace(" ", "_") in {"TITLE_SEQUENCE", "UNKNOWN", ""}:
            return None
        return state
    finally:
        probe.close()


def _write_event_result(
    *,
    event_output: Path,
    recorder: ChunkRecorder,
    event_id: str,
    pre_state: str,
    postcondition: str,
    expected_state,
    start_frame: int,
    end_frame: int,
    actions_taken: int,
    start_state,
    end_state,
    validation: str,
    failure_reason: str | None,
    policy_sources_used: set[str],
    initial_state_file: dict | None = None,
    final_state_file: dict | None = None,
    initial_frame_file: dict | None = None,
    final_frame_file: dict | None = None,
) -> dict:
    segment = {
        "segment_type": "event",
        "event_id": event_id,
        "policy_source": "heatz",
        "policy_sources_used": sorted(policy_sources_used) or ["heatz"],
        "pre_state": pre_state,
        "postcondition": postcondition,
        "expected_state": expected_state.to_dict() if expected_state else None,
        "frame_range": [start_frame, end_frame],
        "action_count": actions_taken,
        "pre_state_hash": start_state.raw_state_hash,
        "post_state_hash": end_state.raw_state_hash,
        "start_state": start_state.to_dict(),
        "end_state": end_state.to_dict(),
        "validation": validation,
        "failure_reason": failure_reason,
        "initial_state_file": initial_state_file,
        "final_state_file": final_state_file,
        "initial_frame_file": initial_frame_file,
        "final_frame_file": final_frame_file,
    }
    recorder.record_segment(segment)
    validation_payload = {
        "validation": validation,
        "failure_reason": failure_reason,
        "actions_taken": actions_taken,
        "end_state": end_state.to_dict(),
        "policy_sources_used": sorted(policy_sources_used) or ["heatz"],
        "initial_state_file": initial_state_file,
        "final_state_file": final_state_file,
        "initial_frame_file": initial_frame_file,
        "final_frame_file": final_frame_file,
    }
    (event_output / "validation.json").write_text(json.dumps(validation_payload, indent=2), encoding="utf-8")
    return {
        "event_id": event_id,
        "validation": validation,
        "failure_reason": failure_reason,
        "actions_taken": actions_taken,
        "output": str(event_output),
        "pre_state": pre_state,
        "initial_state_path": str(event_output / initial_state_file["path"]) if initial_state_file else None,
        "final_state_path": str(event_output / final_state_file["path"]) if final_state_file else None,
        "initial_frame_path": str(event_output / initial_frame_file["path"]) if initial_frame_file else None,
        "final_frame_path": str(event_output / final_frame_file["path"]) if final_frame_file else None,
        "initial_frame_4x_path": str(event_output / initial_frame_file["path_4x"]) if initial_frame_file else None,
        "final_frame_4x_path": str(event_output / final_frame_file["path_4x"]) if final_frame_file else None,
    }


def _persist_state_bytes(event_output: Path, filename: str, state_bytes: bytes | None) -> dict | None:
    if state_bytes is None:
        return None
    path = event_output / filename
    path.write_bytes(state_bytes)
    return {"path": filename, "bytes": len(state_bytes)}


def _persist_frame(event_output: Path, stem: str, runner: DirectEmulatorRunner) -> dict | None:
    try:
        image = runner.screenshot().convert("RGB")
    except Exception:
        return None
    path = event_output / f"{stem}.png"
    path_4x = event_output / f"{stem}_4x.png"
    image.save(path)
    image.resize((image.width * 4, image.height * 4)).save(path_4x)
    return {"path": path.name, "path_4x": path_4x.name, "size": list(image.size)}


def _party_has_species_level(state, species: str, min_level: int) -> bool:
    wanted = species.lower()
    for member in state.party_summary or []:
        if not isinstance(member, dict):
            continue
        name = str(member.get("species") or "").lower()
        level = member.get("level") or 0
        try:
            level = int(level)
        except (TypeError, ValueError):
            level = 0
        if name == wanted and level >= min_level:
            return True
    return False


# Towns with a Pokémon Center where we top the party up when it's run low — the
# same resource management a human does between routes (no heal is baked into the
# milestone chain until after the gauntlets that need it).
# Towns where we run an auto-heal detour. Only OLDALE qualifies: it's the lone Centre
# town before the Route 102 trainer gauntlet that has no scripted heal event, so a
# run-down Mudkip would otherwise black out there. Petalburg/Rustboro each have their
# own HEAL_AT_*_CENTER event, so auto-healing in them is redundant and only risks
# detouring the "reach the city" events away from their goal.
_HEAL_TOWNS = {"OLDALE TOWN"}
_HEAL_HP_FRACTION = 0.6


def _party_below_fraction(current, frac: float) -> bool:
    return any(
        (m.get("max_hp") or 0) > 0 and (m.get("hp") or 0) < frac * (m.get("max_hp") or 0)
        for m in (current.party_summary or [])
    )


def _party_all_full(current) -> bool:
    party = current.party_summary or []
    return all((m.get("hp") or 0) >= (m.get("max_hp") or 0) > 0 for m in party) if party else True


def _lead_at_level(state, min_level: int) -> bool:
    """Party LEAD (slot 0) is at least `min_level` — STARTER-AGNOSTIC. The grind/postcondition gates
    were written against "Mudkip" (the original playthrough's starter); for alt-starter collection
    (Treecko/Torchic) we gate on the lead's level regardless of species. Backward-compatible: when the
    lead IS Mudkip this is identical to the old species check."""
    party = state.party_summary or []
    if not party or not isinstance(party[0], dict):
        return False
    try:
        return int(party[0].get("level") or 0) >= min_level
    except (TypeError, ValueError):
        return False


# Events that own their own Pokémon-Center flow (entering, healing, exiting). The
# generic heal detour must stay out of their way, or it fights their navigation.
_HEAL_OWNING_EVENT = ("CENTER", "HEAL")


def _should_heal(current, heal_state: dict, event_id: str | None = None) -> bool:
    """Run a heal detour when the party is run-down in a Centre town, and keep it
    running (via heal_state) until we're healed and back outside the Centre.

    Skipped for events that manage their own Centre visit (their policy handles the
    nurse), so we only auto-heal during plain transit through a Centre town."""
    if event_id and any(tag in event_id.upper() for tag in _HEAL_OWNING_EVENT):
        return False
    if heal_state.get("skip"):  # heal_action vetoed the detour for this event
        return False
    map_name = str(current.map or "")
    in_center = "POKEMON CENTER" in map_name.upper()
    if heal_state.get("active"):
        if not in_center and _party_all_full(current):
            # Hand back to the policy only once we're settled on solid town ground.
            # The exit warp reports map-name and position on different frames, so a
            # single "in town" reading can be a transient mid-warp flicker; requiring
            # the same town tile twice in a row avoids deactivating while still on the
            # Centre door (where the policy would just walk back in).
            settle_key = (map_name, current.x, current.y)
            if map_name in _HEAL_TOWNS and settle_key == heal_state.get("settle_key"):
                heal_state["active"] = False
                heal_state.pop("settle_key", None)
                return False
            heal_state["settle_key"] = settle_key
            return True
        heal_state.pop("settle_key", None)
        return True
    if _heal_needed_here(current, event_id):
        heal_state["active"] = True
        return True
    return False


def _heal_needed_here(current, event_id: str | None = None) -> bool:
    """Pure (no side effects) test of whether a heal detour should *start* here:
    a run-down party in a Centre town, for an event that doesn't own its own Centre
    flow. Used to gate the post-action postcondition so the gate can't fire on the
    same step we walk into town, before the stateful _should_heal activates next
    iteration. Kept side-effect-free so it never disturbs the heal settle tracking."""
    if event_id and any(tag in event_id.upper() for tag in _HEAL_OWNING_EVENT):
        return False
    return str(current.map or "") in _HEAL_TOWNS and _party_below_fraction(current, _HEAL_HP_FRACTION)


# Grinding to the rival's level on Route 103 racks up cumulative chip damage with no
# Centre on the route, so Mudkip can black out mid-grind or while walking up to the
# rival. When it drops below this fraction we detour south to Oldale's Centre to heal.
_MAY_HEAL_FRACTION = 0.5


def _may_heal_needed(current, may_state: dict) -> bool:
    """Whether the MAY Route-103 grind/rival-approach should detour to Oldale to heal.
    Stays active (via may_state) until Mudkip is back on Route 103 at full HP."""
    on_route = str(current.map or "").upper().startswith("ROUTE 103")
    if may_state.get("active"):
        if on_route and _party_all_full(current):
            may_state["active"] = False
            return False
        return True
    if _party_below_fraction(current, _MAY_HEAL_FRACTION):
        may_state["active"] = True
        return True
    return False


# Story gates that require the starter to be a certain level. We grind the route's
# grass up to this level once the rest of the gate is satisfied (e.g. rival beaten).
_EVENT_GRIND_LEVEL = {"MAY_ROUTE103_INTERACTION": 7}


def _grind_target(event_id: str | None, current) -> int | None:
    """Return the level to grind Mudkip to, only when the gate's *other* conditions
    already hold and the level is the lone thing missing (so we don't grind before
    actually reaching/beating the gate). None means "don't grind"."""
    target = _EVENT_GRIND_LEVEL.get(event_id or "")
    if target is None or _lead_at_level(current, target):     # starter-agnostic (was hardcoded Mudkip)
        return None
    if event_id == "MAY_ROUTE103_INTERACTION":
        # Grind up FIRST (the rival's Treecko is super-effective vs Mudkip, a coin
        # flip at low level). Once at level the policy walks up and wins easily.
        if (
            current.map == "ROUTE 103"
            and current.control_mode == "free_overworld"
            and not current.in_battle
            and not current.dialogue
        ):
            return target
    return None


def _semantic_postcondition_met(event_id: str | None, runner: DirectEmulatorRunner, current, start_money: int = 0) -> bool:
    if event_id == "STARTER_CHOSEN":
        # Done the moment we hold Mudkip and are back in overworld control after the
        # rescue battle. The saved checkpoint sits on a script tail (Birch then walks
        # you to the lab), so a responsive position match would miss this window.
        return (
            _party_has_species_level(current, "Mudkip", 1)
            and current.control_mode == "free_overworld"
            and not current.in_battle
            and not current.dialogue
            and not _visible_dialog_open(runner)
        )
    if event_id == "MAY_ROUTE103_INTERACTION":
        return (
            current.map == "ROUTE 103"
            and current.control_mode == "free_overworld"
            and not current.in_battle
            and not current.dialogue
            and not _visible_dialog_open(runner)
            and (current.money or 0) >= 3300
            and _lead_at_level(current, 7)                     # starter-agnostic (was hardcoded Mudkip)
        )
    if event_id == "ROUTE_104_SOUTH":
        # The completed checkpoint is corrupt and the milestone is pre-marked, so the
        # gate would pass instantly back in Petalburg and never walk onto Route 104 —
        # leaving the next event (PETALBURG_WOODS) stranded a map away. Require actually
        # standing on Route 104 (its policy then heads for the woods entrance).
        return (
            "ROUTE 104" in str(current.map or "").upper()
            and not current.in_battle
            and not _visible_dialog_open(runner)  # memory cm falsely reads "dialogue" here
        )
    if event_id == "TRAINER_JOSH_BATTLE":
        # Corrupt checkpoint + no milestone is defined for this gym trainer, so detect
        # the win by the trainer payout: beating Josh bumps our money. Then we're back
        # in the overworld, free to head up to Roxanne (the next event).
        return (
            (current.money or 0) > start_money
            and not current.in_battle
            and not _visible_dialog_open(runner)
        )
    return False


# Events whose completion is decided ONLY by _semantic_postcondition_met (their stored
# milestone/checkpoint is unreliable, so we must not fall through to the sticky-flag
# milestone check, which would pass prematurely).
_SEMANTIC_ONLY_EVENTS = {"ROUTE_104_SOUTH", "TRAINER_JOSH_BATTLE"}


def _postcondition_met(
    runner: DirectEmulatorRunner,
    postcondition: str,
    *,
    min_actions_met: bool,
    expected_state=None,
    accept_unresponsive_target: bool = False,
    event_id: str | None = None,
    start_money: int = 0,
    current=None,
) -> bool:
    if not min_actions_met:
        return False
    # Reuse a state already read this iteration when given — r.state() is ~12 ms and
    # was otherwise re-read here every call (twice per action).
    if current is None:
        current = runner.state()
    if event_id == "STARTER_CHOSEN" and not _party_has_species_level(current, "Mudkip", 1):
        # The starter pick is only complete when we actually hold Mudkip; a wrong
        # starter (e.g. Torchic) must fail loudly instead of passing on position alone.
        return False
    if _semantic_postcondition_met(event_id, runner, current, start_money):
        return True
    if event_id in _SEMANTIC_ONLY_EVENTS:
        return False  # decided solely by the semantic check above; no milestone fallback
    if postcondition == "STONE_BADGE" and current.badges and current.badges >= 1:
        # The badge IS the goal — accept the instant it's earned, even while Roxanne's
        # long award dialog is still on screen (the generic visible-dialog / overworld
        # gates below would otherwise reject every frame of that cutscene).
        return True
    if (
        expected_state is not None
        and accept_unresponsive_target
        and not current.in_battle
        and _physical_target_reached(current, expected_state)
    ):
        # These checkpoints sit on a script/cutscene tail where memory's control_mode
        # is unreliable (e.g. it reads "dialogue" all over the Petalburg gym near
        # Norman with nothing on screen). Accept the target on position alone.
        return True
    if _visible_dialog_open(runner):
        return False
    if current.control_mode != "free_overworld":
        return False
    if postcondition == "STONE_BADGE":
        return bool(current.badges and current.badges >= 1)
    # Heatz milestone files can land on script/cutscene tails where memory reports
    # overworld even though movement inputs are ignored. Require a real movement
    # response before accepting save-state based postconditions.
    if expected_state is not None:
        return _responsive_target_reached(runner, expected_state)
    assert runner.env is not None
    try:
        state = _full_state_for_milestones(runner.env)
        runner.env.check_and_update_milestones(state)
        tracker = getattr(runner.env, "milestone_tracker", None)
        return bool(tracker and tracker.is_completed(postcondition))
    except Exception:
        return False


_NAV_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


_EVENTS_REQUIRING_MAP_STATE = {
    "PLAYER_HOUSE_ENTERED",
    "BACK_TO_LITTLEROOT_FOR_POKEDEX",
    "OLDALE_AFTER_POKEDEX",
    "ROUTE_103",
    "BACK_TO_ROUTE101_FROM_OLDALE",
}

# Some Heatz checkpoints are saved on the first frame of a map transition/cutscene
# trigger. Movement is not responsive there yet, but the event boundary is correct.
_EVENTS_ACCEPTING_UNRESPONSIVE_TARGET = {
    "LITTLEROOT_TO_ROUTE101",
    "ROUTE_101",
    # DAD_DIALOG_CONFIRMED's completed checkpoint sits on Norman's post-cutscene
    # greeting: memory reads free_overworld at (4,108) but a dialog box is still on
    # screen, so the visual-dialog veto would reject even the expert's own target.
    # Accept the position-based target here instead.
    "DAD_DIALOG_CONFIRMED",
    # BACK_IN_GYM_AFTER_CUTSCENE ends at the same Norman tile (4,108); memory's
    # dialogue/control_mode reads are unreliable there too.
    "BACK_IN_GYM_AFTER_CUTSCENE",
}


def _state_key(state) -> tuple:
    return (state.map, state.x, state.y, state.facing, state.control_mode)


def _structural_progress_key(state) -> tuple:
    return (state.map, state.x, state.y, state.facing, state.control_mode, state.raw_state_hash)


def _physical_target_reached(state, expected_state) -> bool:
    if expected_state is None:
        return False
    if state.map != expected_state.map:
        return False
    if state.x is None or state.y is None or expected_state.x is None or expected_state.y is None:
        return True
    return abs(state.x - expected_state.x) + abs(state.y - expected_state.y) <= 1


def _target_reached(state, expected_state) -> bool:
    return state.control_mode == "free_overworld" and _physical_target_reached(state, expected_state)


def _visible_dialog_open(runner: DirectEmulatorRunner) -> bool:
    if runner.env is None:
        return False
    return bool(_visual_dialog_open(runner.env))


_TARGET_TAIL_ACTIONS = ("A", "A", "WAIT", "A", "B", "WAIT")
_TARGET_TAIL_STALL_LIMIT = 96


def _target_tail_action(index: int) -> str:
    return _TARGET_TAIL_ACTIONS[index % len(_TARGET_TAIL_ACTIONS)]


def _tail_progress_key(state, visible_dialog: bool) -> tuple:
    return (state.map, state.x, state.y, state.facing, state.control_mode, state.raw_state_hash, visible_dialog)


def _can_plan_from_state(state, *, visible_dialog: bool) -> bool:
    # Trust the visual dialog check over the memory control_mode: this ROM sometimes
    # reports control_mode="dialogue" with no dialog box on screen (e.g. standing in
    # the Petalburg gym near Norman), which would otherwise wedge navigation. The
    # BFS fallback simulates real movement, so it harmlessly no-ops if we truly can't
    # move; what matters is not planning *through* an actual on-screen dialog/battle.
    return not visible_dialog and not state.in_battle


def _read_nav_snapshot(runner: DirectEmulatorRunner) -> dict:
    assert runner.env is not None
    reader = runner.env.memory_reader
    coords = reader.read_coordinates() if reader else None
    x, y = coords if isinstance(coords, tuple) and len(coords) >= 2 else (None, None)
    location = location_name(reader.read_location()) if reader else None
    game_state = reader.get_game_state() if reader else None
    in_battle = reader.is_in_battle() if reader else False
    # For planner search, raw residual dialogue should not block plain movement.
    dialogue = False if not in_battle else None
    mode = classify_control_mode(game_state=game_state, in_battle=in_battle, dialogue=dialogue)
    return {"map": location, "x": x, "y": y, "facing": runner.facing, "control_mode": mode}


def _nav_key(snapshot: dict) -> tuple:
    return (snapshot.get("map"), snapshot.get("x"), snapshot.get("y"), snapshot.get("facing"), snapshot.get("control_mode"))


def _nav_target_reached(snapshot: dict, expected_state) -> bool:
    if expected_state is None or snapshot.get("control_mode") != "free_overworld":
        return False
    if snapshot.get("map") != expected_state.map:
        return False
    x, y = snapshot.get("x"), snapshot.get("y")
    if x is None or y is None or expected_state.x is None or expected_state.y is None:
        return True
    return abs(x - expected_state.x) + abs(y - expected_state.y) <= 1


def _simulate_action(runner: DirectEmulatorRunner, state_bytes: bytes, facing: str, action: str):
    assert runner.env is not None
    current_bytes = runner.save_state_bytes()
    if current_bytes is None:
        return None
    saved_recorder = runner.recorder
    saved_frame_idx = runner.frame_idx
    saved_facing = runner.facing
    try:
        runner.recorder = None
        runner.frame_idx = saved_frame_idx
        runner.facing = facing
        runner.env.load_state(state_bytes=state_bytes)
        runner.facing = update_facing(facing, action)
        for buttons in run_action_frames(action, timing_for("normal")):
            runner.env.run_frame_with_buttons([button.lower() for button in buttons])
        for _ in range(24):
            runner.env.run_frame_with_buttons([])
        post = _read_nav_snapshot(runner)
        post_bytes = runner.save_state_bytes()
        if post_bytes is None:
            return None
        return post, post_bytes, runner.facing
    finally:
        runner.recorder = None
        runner.env.load_state(state_bytes=current_bytes)
        runner.frame_idx = saved_frame_idx
        runner.facing = saved_facing
        runner.recorder = saved_recorder


def _movement_position_key(snapshot: dict) -> tuple:
    return (snapshot.get("map"), snapshot.get("x"), snapshot.get("y"))


def _movement_responsive(runner: DirectEmulatorRunner) -> bool:
    start_bytes = runner.save_state_bytes()
    if start_bytes is None:
        return False
    start = _read_nav_snapshot(runner)
    start_key = _movement_position_key(start)
    for action in _NAV_ACTIONS:
        simulated = _simulate_action(runner, start_bytes, runner.facing, action)
        if simulated is None:
            continue
        post, _post_bytes, _post_facing = simulated
        if _movement_position_key(post) != start_key:
            return True
    return False


def _responsive_target_reached(runner: DirectEmulatorRunner, expected_state) -> bool:
    current = runner.state()
    return _target_reached(current, expected_state) and not _visible_dialog_open(runner) and _movement_responsive(runner)


def _plan_first_action_to_expected(
    runner: DirectEmulatorRunner,
    expected_state,
    *,
    max_depth: int = 14,
    max_nodes: int = 192,
) -> str | None:
    if expected_state is None:
        return None
    start_bytes = runner.save_state_bytes()
    if start_bytes is None:
        return None
    start = _read_nav_snapshot(runner)
    if _nav_target_reached(start, expected_state):
        return None
    queue = deque([(start_bytes, runner.facing, [])])
    seen = {_nav_key(start)}
    nodes = 0
    while queue and nodes < max_nodes:
        state_bytes, facing, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in _NAV_ACTIONS:
            simulated = _simulate_action(runner, state_bytes, facing, action)
            nodes += 1
            if simulated is None:
                continue
            post, post_bytes, post_facing = simulated
            key = _nav_key(post)
            if key in seen:
                continue
            next_path = [*path, action]
            if _nav_target_reached(post, expected_state):
                return next_path[0]
            seen.add(key)
            if post.get("control_mode") == "free_overworld":
                queue.append((post_bytes, post_facing, next_path))
            if nodes >= max_nodes:
                break
    return None


def collect_one_event(
    *,
    event_id: str,
    policy_dir: str,
    pre_state: str,
    completed_state: str | None,
    output_dir: str,
    rom_path: str,
    backend: str,
    visual_fps: int,
    max_actions: int,
    min_actions: int,
    stall_actions: int,
    blocked_nav_actions: int,
    record_condition: bool = False,
) -> dict:
    event_output = Path(output_dir) / event_id / "attempt_000001"
    event_output.mkdir(parents=True, exist_ok=True)
    policy = HeatzPolicy(event_id, Path(policy_dir) / event_id / f"{event_id}.py")
    postcondition = EVENT_POSTCONDITION_ALIAS.get(event_id, event_id)
    with ChunkRecorder(
        event_output,
        run_id=f"event_{event_id}",
        # World-model mode records every frame's RGB so it aligns 1:1 with the per-frame
        # PPU condition stream (the frame_hook fires on every step_frame).
        visual_fps=1000 if record_condition else visual_fps,
        backend=backend,
        metadata={"event_id": event_id, "pre_state": pre_state, "completed_state": completed_state, "postcondition": postcondition},
    ) as recorder:
        expected_state = _load_expected_state(rom_path=rom_path, completed_state=completed_state, event_id=event_id)
        runner = DirectEmulatorRunner(rom_path=rom_path, load_state=pre_state, story_bucket=event_id, recorder=recorder)
        runner.initialize()
        # World-model condition capture: fires only on real (step_frame) playthrough frames,
        # not on the restore-based planning sims (_simulate_action bypasses step_frame).
        sink = WorldModelSink(event_output) if record_condition else None
        if sink is not None:
            runner.frame_hook = sink.capture
            sink.capture(runner)  # align with the initial recorded frame
        start_state = runner.state()
        start_money = start_state.money or 0
        initial_state_file = _persist_state_bytes(event_output, "initial.state", runner.save_state_bytes())
        initial_frame_file = _persist_frame(event_output, "initial_frame", runner)
        start_frame = runner.frame_idx
        actions_taken = 0
        validation = "failed"
        failure_reason: str | None = None
        policy_sources_used: set[str] = set()
        prev_action: str | None = None
        stuck_count = 0
        grind_state: dict = {}
        heal_state: dict = {}
        may_heal_state: dict = {}
        last_progress_key = _state_key(start_state)
        target_tail_actions = 0
        target_tail_stall_count = 0
        last_target_tail_key: tuple | None = None
        no_structural_progress_count = 0
        last_structural_progress_key = _structural_progress_key(start_state)
        blocked_nav_count = 0
        last_blocked_nav_key: tuple | None = None
        try:
            accept_unresponsive_target = event_id in _EVENTS_ACCEPTING_UNRESPONSIVE_TARGET
            if (
                _responsive_target_reached(runner, expected_state)
                or _semantic_postcondition_met(event_id, runner, runner.state(), start_money)
                or (accept_unresponsive_target and expected_state is not None and _target_reached(runner.state(), expected_state))
            ):
                validation = "skipped"
                failure_reason = "already_complete_start"
                final_state_file = _persist_state_bytes(event_output, "final.state", runner.save_state_bytes())
                final_frame_file = _persist_frame(event_output, "final_frame", runner)
                return _write_event_result(
                    event_output=event_output,
                    recorder=recorder,
                    event_id=event_id,
                    pre_state=pre_state,
                    postcondition=postcondition,
                    expected_state=expected_state,
                    start_frame=start_frame,
                    end_frame=runner.frame_idx,
                    actions_taken=0,
                    start_state=start_state,
                    end_state=start_state,
                    validation=validation,
                    failure_reason=failure_reason,
                    policy_sources_used=policy_sources_used,
                    initial_state_file=initial_state_file,
                    final_state_file=final_state_file,
                    initial_frame_file=initial_frame_file,
                    final_frame_file=final_frame_file,
                )

            for _ in range(max_actions):
                policy_source = "heatz"
                current_before_action = runner.state()
                # The gate takes priority over *starting* a heal detour: if we've already
                # reached the event's goal (e.g. we entered Petalburg right beside it),
                # finish instead of wandering off to the Centre. An in-progress heal
                # (heal_state active) is never interrupted by this.
                if not heal_state.get("active") and _postcondition_met(
                    runner,
                    postcondition,
                    min_actions_met=actions_taken >= min_actions,
                    expected_state=expected_state,
                    accept_unresponsive_target=accept_unresponsive_target,
                    event_id=event_id,
                    start_money=start_money,
                    current=current_before_action,
                ):
                    validation = "passed"
                    break
                healing = not current_before_action.in_battle and _should_heal(current_before_action, heal_state, event_id)
                # MAY's grind has no Centre on Route 103; detour to Oldale to heal when
                # run-down. It only steers the overworld (descend → heal → ascend); the
                # policy fights any wild battle along the way (its battle-cursor state
                # persists, which a direct handle_battle call here would lack). It owns
                # healing on/around Route 103, so it pre-empts the generic heal.
                may_healing = event_id == "MAY_ROUTE103_INTERACTION" and _may_heal_needed(current_before_action, may_heal_state)
                if may_healing:
                    healing = False

                visible_dialog = _visible_dialog_open(runner)
                h_state = build_heatz_state(
                    runner.env,
                    frame_idx=runner.frame_idx,
                    story_bucket=event_id,
                    facing=runner.facing,
                    include_map=event_id in _EVENTS_REQUIRING_MAP_STATE,
                )
                if prev_action:
                    h_state["prev_action"] = prev_action.lower()
                # If the only thing left to satisfy the gate is Mudkip's level (the
                # rival is beaten and we're back in control but under-levelled), grind
                # in the route's grass instead of stalling — the policy itself fights
                # the wild battles this triggers.
                if healing:
                    # Tell the heal detour where the event wants to end up, so it can
                    # both decide whether the detour is worth taking and, after healing,
                    # walk back to the goal instead of stranding us by the Centre.
                    if expected_state is not None and expected_state.x is not None and expected_state.y is not None:
                        h_state["_heal_goal"] = (expected_state.map, expected_state.x, expected_state.y)
                    heal_raw = heal_action(h_state)
                    if heal_raw == "heal_skip":
                        # Centre is too far from this event's goal; skip the detour for
                        # the rest of the event and let the party heal somewhere closer.
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
                elif healing:
                    pass
                elif not current_before_action.in_battle and _grind_target(event_id, current_before_action) is not None:
                    action = normalize_action(grind_action(h_state, grind_state))
                    policy_source = "heatz_grind"
                    stuck_count = 0
                else:
                    action = normalize_action(policy.act(h_state))

                    at_physical_target = expected_state is not None and _physical_target_reached(current_before_action, expected_state)
                    # The tail-settle (mash A/B/WAIT) settles the script AT its exact
                    # destination tile — it has no movement, so it can't take the last step
                    # onto the target. Gate it on being exactly on the tile: when we're a
                    # tile short (e.g. DAD_DIALOG, where stepping up onto (4,108) triggers
                    # Norman) let the policy walk that step first; once we're on the tile the
                    # tail-settle mashes through any greeting cutscene (e.g. mom in
                    # PLAYER_HOUSE_ENTERED). Within-1-but-not-exact mashing would freeze us a
                    # tile away and its stray B would also cancel cutscene prompts.
                    at_exact_target = (
                        expected_state is not None
                        and current_before_action.map == expected_state.map
                        and current_before_action.x == expected_state.x
                        and current_before_action.y == expected_state.y
                    )
                    # Only override a policy that has nothing useful to do (returns WAIT).
                    # A policy actively walking the last tile to a trigger (DAD_DIALOG) or
                    # confirming a dialog must be trusted, not clobbered by the tail-settle.
                    policy_idle = action == "WAIT"
                    can_plan = _can_plan_from_state(current_before_action, visible_dialog=visible_dialog)
                    # Some screens (clock-setting, Birch's starter bag) read as overworld in
                    # memory but need real UI navigation that the policy handles itself. While
                    # such a UI is active, trust the policy's action and suppress the generic
                    # tail-settle / BFS fallbacks, or they clobber the menu interaction.
                    special_ui_active = _visual_clock_ui(runner.env) or (
                        event_id == "STARTER_CHOSEN" and _starter_ui_active(runner.env)
                    )
                    if not special_ui_active and policy_idle and at_physical_target and visible_dialog:
                        # A real on-screen dialog box is up at (or a tile from) the target
                        # and the policy is idle — mash A/B to advance it (e.g. mom's
                        # greeting in PLAYER_HOUSE_ENTERED). Keyed on the *visual* box, not
                        # the memory control_mode (which false-reports "dialogue" near
                        # Norman with nothing on screen — there we want to navigate, below).
                        action = _target_tail_action(target_tail_actions)
                        target_tail_actions += 1
                        policy_source = "heatz_script_tail_settle"
                        stuck_count = 0
                    elif not special_ui_active and policy_idle and at_exact_target and not visible_dialog and not _movement_responsive(runner):
                        action = _target_tail_action(target_tail_actions)
                        target_tail_actions += 1
                        policy_source = "heatz_script_tail_settle"
                        stuck_count = 0
                    elif not special_ui_active and expected_state is not None and can_plan and (stuck_count >= 5 or action == "WAIT"):
                        fallback_action = _plan_first_action_to_expected(runner, expected_state)
                        if fallback_action:
                            action = normalize_action(fallback_action)
                            policy_source = "heatz_emulator_bfs_fallback"
                            stuck_count = 0

                policy_sources_used.add(policy_source)
                state_before_perform = current_before_action
                # perform_action returns the post-action AbstractState (== runner.state());
                # reuse it instead of re-reading (~12 ms).
                current_state = runner.perform_action(action, metadata={"event_id": event_id, "policy_source": policy_source})
                prev_action = action
                actions_taken += 1
                progress_key = _state_key(current_state)
                if progress_key == last_progress_key and action in _NAV_ACTIONS:
                    stuck_count += 1
                else:
                    stuck_count = 0
                    last_progress_key = progress_key

                visible_dialog_after = _visible_dialog_open(runner)
                structural_key = _structural_progress_key(current_state)
                if visible_dialog_after or current_state.in_battle:
                    # Dialog/cutscene/battle pages often keep map/position/hash stable
                    # while A advances text or scripted actions. Do not classify that
                    # as stalled overworld navigation.
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

                if action in _NAV_ACTIONS and not current_state.in_battle and not state_before_perform.in_battle and state_before_perform.map == current_state.map and state_before_perform.x == current_state.x and state_before_perform.y == current_state.y:
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

                if expected_state is not None and _physical_target_reached(current_state, expected_state) and policy_source == "heatz_script_tail_settle":
                    tail_key = _tail_progress_key(current_state, _visible_dialog_open(runner))
                    if tail_key == last_target_tail_key:
                        target_tail_stall_count += 1
                    else:
                        target_tail_stall_count = 0
                        last_target_tail_key = tail_key
                    if target_tail_stall_count >= _TARGET_TAIL_STALL_LIMIT:
                        failure_reason = "target_tail_stalled"
                        break
                else:
                    target_tail_stall_count = 0
                    last_target_tail_key = None

                # The gate takes priority over starting a heal detour: if the event's
                # goal is already reached on arrival in a Centre town, just pass. A
                # detour there would only strand us far from a goal we've already met
                # (e.g. BACK_TO_OLDALE finishes at the north entry, opposite the
                # Centre). The heal still fires via the top-of-loop check for events
                # whose goal needs more navigation (their gate isn't met on arrival).
                if not heal_state.get("active") and _postcondition_met(
                    runner,
                    postcondition,
                    min_actions_met=actions_taken >= min_actions,
                    expected_state=expected_state,
                    accept_unresponsive_target=accept_unresponsive_target,
                    event_id=event_id,
                    start_money=start_money,
                    current=current_state,
                ):
                    validation = "passed"
                    break
            end_state = runner.state()
            final_state_file = _persist_state_bytes(event_output, "final.state", runner.save_state_bytes())
            final_frame_file = _persist_frame(event_output, "final_frame", runner)
            if validation != "passed" and failure_reason is None:
                failure_reason = "max_actions_exhausted"
            return _write_event_result(
                event_output=event_output,
                recorder=recorder,
                event_id=event_id,
                pre_state=pre_state,
                postcondition=postcondition,
                expected_state=expected_state,
                start_frame=start_frame,
                end_frame=runner.frame_idx,
                actions_taken=actions_taken,
                start_state=start_state,
                end_state=end_state,
                validation=validation,
                failure_reason=failure_reason,
                policy_sources_used=policy_sources_used,
                initial_state_file=initial_state_file,
                final_state_file=final_state_file,
                initial_frame_file=initial_frame_file,
                final_frame_file=final_frame_file,
            )
        finally:
            if sink is not None:
                sink.close()
            runner.close()


def _load_audit_recommendations(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    audit_path = Path(path)
    if not audit_path.exists():
        raise FileNotFoundError(f"Audit file not found: {audit_path}")
    rows = []
    if audit_path.suffix == ".jsonl":
        with audit_path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        rows = data.get("events", data if isinstance(data, list) else [])
    return {str(row.get("event_id")): row for row in rows if row.get("event_id")}


def _event_allowed_by_audit(event: dict, audit_rows: dict[str, dict]) -> bool:
    if not audit_rows:
        return True
    row = audit_rows.get(event["event_id"])
    if not row:
        return False
    return row.get("recommendation") == "collect"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatz-policy-dir", required=True)
    parser.add_argument("--event", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rom-path", default="Emerald-GBAdvance/rom.gba")
    parser.add_argument("--backend", default="auto", choices=["auto", "ffv1", "npz"])
    parser.add_argument("--visual-fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--audit-file", default=None, help="Optional event audit JSON/JSONL; only recommendation=collect events run")
    parser.add_argument("--max-actions", type=int, default=2500)
    parser.add_argument("--min-actions", type=int, default=10)
    parser.add_argument("--stall-actions", type=int, default=120, help="Fail early after this many actions without structural state progress; 0 disables")
    parser.add_argument("--blocked-nav-actions", type=int, default=8, help="Fail after repeating the same blocked movement this many times; 0 disables")
    parser.add_argument("--chain", action="store_true", help="Sequentially feed each event's collected final.state into dependent next events")
    parser.add_argument("--world-model", action="store_true", help="Record per-frame full-PPU condition + semantic (world-model dataset)")
    args = parser.parse_args()

    audit_rows = _load_audit_recommendations(args.audit_file)
    events = discover_heatz_events(args.heatz_policy_dir)
    if args.event:
        events = [event for event in events if event["event_id"] == args.event]
    events = [event for event in events if event.get("pre_state") and _event_allowed_by_audit(event, audit_rows)]
    if not events:
        raise SystemExit("No matching events with pre_state found")
    if args.chain and args.workers > 1:
        raise SystemExit("--chain requires --workers 1")

    results = []
    kwargs_for = lambda event: dict(
        event_id=event["event_id"],
        policy_dir=args.heatz_policy_dir,
        pre_state=event["pre_state"],
        completed_state=event.get("completed_state"),
        output_dir=args.output,
        rom_path=args.rom_path,
        backend=args.backend,
        visual_fps=args.visual_fps,
        max_actions=args.max_actions,
        min_actions=args.min_actions,
        stall_actions=args.stall_actions,
        blocked_nav_actions=args.blocked_nav_actions,
        record_condition=args.world_model,
    )
    if args.workers > 1 and len(events) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(collect_one_event, **kwargs_for(event)) for event in events]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps(result, sort_keys=True))
                results.append(result)
    else:
        final_state_by_event: dict[str, str] = {}
        for event in events:
            event_to_collect = dict(event)
            if args.chain:
                prev = event_to_collect.get("pre_milestone")
                if prev in final_state_by_event:
                    event_to_collect["pre_state"] = final_state_by_event[prev]
            result = collect_one_event(**kwargs_for(event_to_collect))
            print(json.dumps(result, sort_keys=True))
            results.append(result)
            if args.chain:
                if result.get("validation") not in {"passed", "skipped"}:
                    break
                final_state_path = result.get("final_state_path")
                if final_state_path:
                    final_state_by_event[event_to_collect["event_id"]] = final_state_path

    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
