"""Catalog definitions for first-gym systematic collection."""

from __future__ import annotations

import json
from pathlib import Path


MILESTONE_ORDER = [
    "GAME_RUNNING",
    "PLAYER_NAME_SET",
    "INTRO_CUTSCENE_COMPLETE",
    "LITTLEROOT_TOWN",
    "PLAYER_HOUSE_ENTERED",
    "PLAYER_BEDROOM",
    "CLOCK_INTERACT",
    "GO_DOWNSTAIRS_TO_1F",
    "LEAVE_HOUSE",
    "RIVAL_HOUSE",
    "RIVAL_BEDROOM",
    "GO_DOWNSTAIRS_RIVAL_HOUSE",
    "EXIT_RIVAL_HOUSE",
    "LITTLEROOT_TO_ROUTE101",
    "ROUTE_101",
    "STARTER_CHOSEN",
    "BIRCH_LAB_VISITED",
    "EXIT_BIRCH_LAB",
    "LITTLEROOT_TO_ROUTE101_AFTER_LAB",
    "ROUTE101_TO_OLDALE_FIRST_TIME",
    "OLDALE_TOWN",
    "ROUTE_103",
    "MAY_ROUTE103_INTERACTION",
    "BACK_TO_OLDALE_FROM_ROUTE103",
    "BACK_TO_ROUTE101_FROM_OLDALE",
    "BACK_TO_LITTLEROOT_FOR_POKEDEX",
    "ENTER_BIRCH_LAB_FOR_POKEDEX",
    "POKEDEX_DIALOG_CONFIRMED",
    "RECEIVED_POKEDEX",
    "ROUTE101_AFTER_POKEDEX",
    "OLDALE_AFTER_POKEDEX",
    "ROUTE_102",
    "PETALBURG_CITY",
    "DAD_FIRST_MEETING",
    "DAD_DIALOG_CONFIRMED",
    "GYM_CUTSCENE_OUTSIDE",
    "BACK_IN_GYM_AFTER_CUTSCENE",
    "EXIT_PETALBURG_GYM",
    "ENTER_PETALBURG_CENTER",
    "HEAL_AT_PETALBURG_CENTER",
    "EXIT_PETALBURG_CENTER",
    "ROUTE_104_SOUTH",
    "PETALBURG_WOODS",
    "TEAM_AQUA_GRUNT_DEFEATED",
    "ROUTE_104_NORTH",
    "RUSTBORO_CITY",
    "RUSTBORO_CENTER_ENTERED",
    "HEAL_AT_RUSTBORO_CENTER",
    "RUSTBORO_CENTER_EXITED",
    "RUSTBORO_GYM_ENTERED",
    "TRAINER_JOSH_BATTLE",
    "ROXANNE_BATTLE",
]

EVENT_POSTCONDITION_ALIAS = {"ROXANNE_BATTLE": "STONE_BADGE"}


def previous_milestone(event_id: str) -> str | None:
    try:
        idx = MILESTONE_ORDER.index(event_id)
    except ValueError:
        return None
    if idx <= 0:
        return None
    return MILESTONE_ORDER[idx - 1]


def discover_heatz_events(policy_dir: str | Path) -> list[dict]:
    base = Path(policy_dir)
    rows = []
    ordered_ids = list(MILESTONE_ORDER)
    extras = sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and (p / f"{p.name}.py").exists() and p.name not in set(ordered_ids)
    )
    for event_id in [*ordered_ids, *extras]:
        event_dir = base / event_id
        policy_path = event_dir / f"{event_id}.py"
        completed_state = event_dir / f"{event_id}_completed.state"
        if not policy_path.exists():
            continue
        prev = previous_milestone(event_id)
        prev_state = base / prev / f"{prev}_completed.state" if prev else None
        rows.append(
            {
                "event_id": event_id,
                "policy_path": str(policy_path),
                "pre_milestone": prev,
                "pre_state": str(prev_state) if prev_state and prev_state.exists() else None,
                "completed_state": str(completed_state) if completed_state.exists() else None,
                "postcondition": EVENT_POSTCONDITION_ALIAS.get(event_id, event_id),
            }
        )
    return rows


def write_catalog(policy_dir: str | Path, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = discover_heatz_events(policy_dir)
    (output / "event_catalog.json").write_text(json.dumps(events, indent=2, sort_keys=True), encoding="utf-8")
    story = [
        {"story_bucket": event["event_id"], "checkpoint": event["completed_state"]}
        for event in events
        if event.get("completed_state")
    ]
    (output / "story_buckets.json").write_text(json.dumps(story, indent=2, sort_keys=True), encoding="utf-8")
    explore_targets = [
        {
            "story_bucket": event["event_id"],
            "load_state": event["completed_state"],
            "source": "heatz_completed_state",
        }
        for event in events
        if event.get("completed_state")
    ]
    (output / "explore_targets.json").write_text(json.dumps({"targets": explore_targets}, indent=2, sort_keys=True), encoding="utf-8")
