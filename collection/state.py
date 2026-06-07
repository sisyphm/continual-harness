"""Canonical state abstraction for systematic Emerald collection."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def safe_call(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def location_name(location: Any) -> str | None:
    if isinstance(location, dict):
        return location.get("map_name") or location.get("name") or location.get("location")
    if location is None:
        return None
    return str(location)


def party_summary(party: Any) -> list[dict[str, Any]]:
    if not isinstance(party, list):
        return []
    out = []
    for pokemon in party:
        if isinstance(pokemon, dict):
            out.append(
                {
                    "species": pokemon.get("species_name") or pokemon.get("species") or pokemon.get("name"),
                    "level": pokemon.get("level"),
                    "hp": pokemon.get("current_hp") or pokemon.get("hp"),
                    "max_hp": pokemon.get("max_hp"),
                    "status": pokemon.get("status"),
                }
            )
        else:
            out.append(
                {
                    "species": getattr(pokemon, "species_name", None) or getattr(pokemon, "name", None),
                    "level": getattr(pokemon, "level", None),
                    "hp": getattr(pokemon, "current_hp", None),
                    "max_hp": getattr(pokemon, "max_hp", None),
                    "status": getattr(pokemon, "status", None),
                }
            )
    return json_safe(out)


def control_mode(*, game_state: Any, in_battle: Any, dialogue: Any) -> str:
    if bool(in_battle):
        return "battle"
    if bool(dialogue):
        return "dialogue"
    if game_state and str(game_state).lower() not in {"overworld", "field", "none"}:
        return str(game_state).lower()
    return "free_overworld"


def visually_detect_dialogue(env: Any) -> bool | None:
    """Validate raw dialogue flags against the actual frame when possible."""
    try:
        from utils.state_formatter import detect_dialogue_on_frame

        screenshot = env.get_screenshot() if hasattr(env, "get_screenshot") else None
        if screenshot is None:
            return None
        result = detect_dialogue_on_frame(frame_array=np.asarray(screenshot))
        if isinstance(result, dict):
            return bool(result.get("has_dialogue"))
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class AbstractState:
    frame_idx: int
    story_bucket: str
    map: str | None
    x: int | None
    y: int | None
    facing: str
    control_mode: str
    game_state: str | None
    dialogue: bool | None
    in_battle: bool | None
    badges: int | None
    badge_names: list[Any]
    party_summary: list[dict[str, Any]]
    money: int | None
    milestone: str | None
    flags_hash: str
    raw_state_hash: str
    timestamp: float

    @property
    def explore_key_prefix(self) -> str:
        return f"{self.story_bucket}|{self.map}|{self.x}|{self.y}|{self.facing}"

    def explore_key(self, action: str) -> str:
        return f"{self.explore_key_prefix}|{action}"

    def to_dict(self) -> dict[str, Any]:
        return json_safe(self.__dict__)


def latest_milestone(env: Any) -> str | None:
    tracker = getattr(env, "milestone_tracker", None)
    if tracker is None:
        return None
    try:
        milestone, _, _ = tracker.get_latest_milestone_info()
        return milestone
    except Exception:
        return None


def read_compact_state(env: Any, *, frame_idx: int, story_bucket: str, facing: str) -> AbstractState:
    reader = getattr(env, "memory_reader", None)
    coords = safe_call(reader.read_coordinates) if reader else None
    x, y = (coords if isinstance(coords, tuple) and len(coords) >= 2 else (None, None))
    location = location_name(safe_call(reader.read_location) if reader else None)
    game_state = safe_call(reader.get_game_state) if reader else None
    in_battle = safe_call(reader.is_in_battle) if reader else None
    raw_dialogue = safe_call(reader.is_in_dialog) if reader else None
    dialogue = raw_dialogue
    if raw_dialogue:
        visual_dialogue = visually_detect_dialogue(env)
        if visual_dialogue is not None:
            dialogue = visual_dialogue
    if not dialogue and game_state and str(game_state).lower() == "dialog":
        game_state = "overworld"
    badges = safe_call(reader.read_badges, []) if reader else []
    money = safe_call(reader.read_money) if reader else None
    party = safe_call(reader.read_party_pokemon, []) if reader and hasattr(reader, "read_party_pokemon") else []
    badge_count = len(badges) if isinstance(badges, list) else badges
    abstract_basis = {
        "story_bucket": story_bucket,
        "map": location,
        "x": x,
        "y": y,
        "facing": facing,
        "control_mode": control_mode(game_state=game_state, in_battle=in_battle, dialogue=dialogue),
        "badges": badge_count,
        "milestone": latest_milestone(env),
    }
    raw_basis = {
        **abstract_basis,
        "game_state": game_state,
        "dialogue": dialogue,
        "in_battle": in_battle,
        "badge_names": badges,
        "party": party_summary(party),
        "money": money,
    }
    return AbstractState(
        frame_idx=frame_idx,
        story_bucket=story_bucket,
        map=location,
        x=x,
        y=y,
        facing=facing,
        control_mode=abstract_basis["control_mode"],
        game_state=str(game_state) if game_state is not None else None,
        dialogue=bool(dialogue) if dialogue is not None else None,
        in_battle=bool(in_battle) if in_battle is not None else None,
        badges=int(badge_count) if isinstance(badge_count, (int, np.integer)) else None,
        badge_names=json_safe(badges) if isinstance(badges, list) else [],
        party_summary=party_summary(party),
        money=int(money) if isinstance(money, (int, np.integer)) else None,
        milestone=abstract_basis["milestone"],
        flags_hash=stable_hash(abstract_basis),
        raw_state_hash=stable_hash(raw_basis),
        timestamp=time.time(),
    )


def result_label(pre: AbstractState, post: AbstractState) -> str:
    if post.control_mode == "battle":
        return "battle_start"
    if post.control_mode == "dialogue":
        return "dialogue_open"
    if post.control_mode != "free_overworld":
        return "event_trigger"
    if pre.map != post.map:
        return "warp"
    if (pre.x, pre.y) != (post.x, post.y):
        return "move"
    if pre.facing != post.facing:
        return "turn_only"
    return "blocked"

