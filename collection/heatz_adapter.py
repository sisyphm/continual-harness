"""Adapter for running Heatz LLM-generated expert policies without LLM calls."""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np

from collection.actions import normalize_action
from collection.state import AbstractState, read_compact_state

logger = logging.getLogger(__name__)
_PORYMAP_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def _location_to_layout(location: str | None) -> str | None:
    if not location:
        return None
    return str(location).replace(" ", "").replace("_", "").upper()


def _read_dialog_text(env: Any) -> str | None:
    reader = getattr(env, "memory_reader", None)
    if reader is None:
        return None
    # Event collection is emulator-fast; OCR is too slow for per-action policy.
    try:
        text = reader.read_dialog()
        return text or None
    except Exception:
        return None


def _safe_comprehensive_state(env: Any) -> dict[str, Any]:
    try:
        screenshot = env.get_screenshot()
        return env.get_comprehensive_state(screenshot=screenshot)
    except Exception:
        compact = read_compact_state(env, frame_idx=0, story_bucket="UNKNOWN", facing="DOWN")
        return {
            "player": {"position": {"x": compact.x, "y": compact.y}, "location": compact.map},
            "game": {"game_state": compact.game_state, "is_in_battle": compact.in_battle, "badges": compact.badge_names},
            "map": {"location": compact.map},
        }


def ensure_porymap_state(state: dict[str, Any]) -> dict[str, Any]:
    """Populate ``state['map']['porymap']`` and Heatz-style ``ascii_map`` if possible."""
    location = state.get("player", {}).get("location") or state.get("map", {}).get("location")
    if not location or location in {"Unknown", "TITLE_SEQUENCE"}:
        return state
    if state.get("map", {}).get("porymap", {}).get("grid"):
        return state

    try:
        from pokemon_env.porymap_paths import get_porymap_root
        from utils.mapping.ascii_map_loader import get_effective_map_name, get_override
        from utils.mapping.porymap_json_builder import build_json_map_for_llm
        from utils.state_formatter import ROM_TO_PORYMAP_MAP

        root = get_porymap_root()
        porymap_name = ROM_TO_PORYMAP_MAP.get(location)
        if not root or not porymap_name:
            return state
        badges = state.get("game", {}).get("badges") or []
        badge_count = len(badges) if isinstance(badges, list) else int(badges or 0)
        cache_key = (str(location), badge_count)
        cached = _PORYMAP_CACHE.get(cache_key)
        if cached is None:
            effective_name = get_effective_map_name(porymap_name, badge_count=badge_count)
            override = get_override(effective_name)
            json_map = build_json_map_for_llm(porymap_name, root, badge_count=badge_count)
            if not json_map:
                return state
            cached = {
                "porymap": {
                    "grid": json_map.get("grid"),
                    "objects": json_map.get("objects", []),
                    "dimensions": json_map.get("dimensions", {}),
                    "warps": json_map.get("warps", []),
                    "raw_tiles": json_map.get("raw_tiles"),
                    "ascii": json_map.get("ascii"),
                },
                "ascii_map": (json_map.get("ascii") or _grid_to_ascii(json_map.get("grid"))).splitlines(),
            }
            if override and ("offset_x" in override or "offset_y" in override):
                cached["coord_offset"] = [override.get("offset_x", 0), override.get("offset_y", 0)]
            _PORYMAP_CACHE[cache_key] = cached
        state.setdefault("map", {})
        state["map"].update(copy.deepcopy(cached))
    except Exception as exc:
        logger.debug("Could not populate porymap state for %s: %s", location, exc)
    return state


def _grid_to_ascii(grid: Any) -> str:
    if not isinstance(grid, list):
        return ""
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def _heatz_facing(facing: str | None) -> str:
    facing_map = {
        "UP": "north",
        "DOWN": "south",
        "LEFT": "west",
        "RIGHT": "east",
        "NORTH": "north",
        "SOUTH": "south",
        "WEST": "west",
        "EAST": "east",
    }
    return facing_map.get(str(facing or "").upper(), "north")


def _visual_dialog_open(env: Any) -> bool | None:
    if env is None:
        return None
    try:
        frame = env.get_screenshot()
        if frame is None or not hasattr(frame, "getpixel"):
            return None
        if getattr(frame, "mode", "RGB") != "RGB":
            frame = frame.convert("RGB")
        if frame.size[0] < 12 or frame.size[1] < 121:
            return None
        return (
            frame.getpixel((10, 117)) == (0, 255, 156)
            and frame.getpixel((11, 117)) == (231, 239, 231)
            and frame.getpixel((10, 120)) == (255, 255, 255)
        )
    except Exception:
        return None


def _visual_clock_ui(env: Any) -> bool:
    if env is None:
        return False
    try:
        frame = env.get_screenshot()
        if frame is None or not hasattr(frame, "getpixel"):
            return False
        if getattr(frame, "mode", "RGB") != "RGB":
            frame = frame.convert("RGB")
        if frame.size[0] < 181 or frame.size[1] < 131:
            return False
        # Emerald clock-setting screen: stable teal background plus central clock face.
        teal_points = ((0, 0), (10, 10), (60, 124), (230, 10))
        teal_matches = sum(1 for point in teal_points if frame.getpixel(point) == (74, 181, 189))
        clock_dark = frame.getpixel((120, 80)) == (0, 0, 0)
        clock_blue = frame.getpixel((110, 80))[2] >= 240
        return teal_matches >= 3 and clock_dark and clock_blue
    except Exception:
        return False


def _visual_clock_yes_no_prompt(env: Any) -> bool:
    if not _visual_clock_ui(env):
        return False
    try:
        frame = env.get_screenshot()
        if getattr(frame, "mode", "RGB") != "RGB":
            frame = frame.convert("RGB")
        if frame.size[0] < 230 or frame.size[1] < 130:
            return False
        # Right-side Yes/No box plus bottom text box for "Is this the correct time?"
        box_white = all(all(channel >= 245 for channel in frame.getpixel(point)) for point in ((190, 42), (215, 70), (205, 93)))
        text_box = all(all(channel >= 245 for channel in frame.getpixel(point)) for point in ((25, 132), (120, 132), (205, 132)))
        return box_white and text_box
    except Exception:
        return False


def _clock_confirm_action(state: dict[str, Any]) -> str:
    # Emerald clock setup: A opens "Is this the correct time?", UP moves
    # the cursor from NO to YES, and A confirms. Repeating the sequence is
    # harmless if an input is dropped during the UI transition.
    return _next_sequence_action(state, "_ui_clock_confirm_step", ("a", "up", "a"), repeat_last=False)


def build_heatz_state(env: Any, *, frame_idx: int, story_bucket: str, facing: str, include_map: bool = False) -> dict[str, Any]:
    compact = read_compact_state(env, frame_idx=frame_idx, story_bucket=story_bucket, facing=facing)
    visual_dialog = _visual_dialog_open(env)
    dialog_text = _read_dialog_text(env) if visual_dialog is True or (visual_dialog is None and compact.dialogue) else None
    state = {
        "player": {
            "position": {"x": compact.x, "y": compact.y},
            "location": compact.map,
            "facing": _heatz_facing(facing),
            "party": compact.party_summary,
        },
        "facing": _heatz_facing(facing),
        "prev_action": "no_op",
        "game": {
            "game_state": compact.game_state,
            "is_in_battle": compact.in_battle,
            "dialog_text": dialog_text,
            "badges": compact.badge_names,
            "money": compact.money,
        },
        "map": {"location": compact.map},
        "battle_info": {"in_battle": compact.in_battle},
        "milestone": compact.milestone,
        "_abstract_state": compact.to_dict(),
        "_env": env,
    }
    if include_map:
        comprehensive = _safe_comprehensive_state(env)
        state["map"].update(comprehensive.get("map") or {})
        state["player"].update({k: v for k, v in (comprehensive.get("player") or {}).items() if k not in {"position", "location"}})
        state["game"].update(comprehensive.get("game") or {})
        ensure_porymap_state(state)
    return state


def is_dialog_open(state: dict[str, Any]) -> bool:
    game = state.get("game") or {}
    visual_dialog = _visual_dialog_open(state.get("_env"))
    if visual_dialog is True:
        return True
    if visual_dialog is False:
        return False
    if game.get("dialog_text"):
        return True
    abstract = state.get("_abstract_state") or {}
    if abstract.get("dialogue"):
        return True
    return str(game.get("game_state") or "").lower() == "dialog"


def _next_sequence_action(state: dict[str, Any], key: str, sequence: tuple[str, ...], *, repeat_last: bool = False) -> str:
    idx = _as_int(state.get(key)) or 0
    if idx < 0:
        idx = 0
    if idx >= len(sequence):
        idx = len(sequence) - 1 if repeat_last else 0
    action = sequence[idx]
    if repeat_last and idx == len(sequence) - 1:
        state[key] = idx
    else:
        state[key] = (idx + 1) % len(sequence)
    return action


def _visual_yes_no_prompt(env: Any) -> bool:
    if env is None or _visual_dialog_open(env) is not True:
        return False
    try:
        frame = env.get_screenshot()
        if frame is None or not hasattr(frame, "getpixel"):
            return False
        if getattr(frame, "mode", "RGB") != "RGB":
            frame = frame.convert("RGB")
        if frame.size[0] < 211 or frame.size[1] < 107:
            return False
        points = ((166, 70), (170, 74), (200, 90), (209, 105))
        return all(all(channel >= 245 for channel in frame.getpixel(point)) for point in points)
    except Exception:
        return False


def _dialog_text_lower(state: dict[str, Any]) -> str:
    text = str((state.get("game") or {}).get("dialog_text") or "")
    return text.replace("�", " ").lower()


def _is_mudkip_nickname_prompt(state: dict[str, Any]) -> bool:
    text = _dialog_text_lower(state)
    return "nickname" in text and "mudkip" in text


def _mudkip_nickname_action(state: dict[str, Any]) -> str:
    if _visual_yes_no_prompt(state.get("_env")):
        state["_ui_mudkip_nickname_decline_active"] = True
    if state.get("_ui_mudkip_nickname_decline_active"):
        return _next_sequence_action(state, "_ui_mudkip_nickname_decline_step", ("down", "down", "a"), repeat_last=True)
    return "a"


def _guidance_action(state: dict[str, Any], action_guidance: str) -> str:
    guidance = action_guidance.lower()
    if "starter" in guidance and "mudkip" in guidance:
        # Emerald bag screen starts on the left/middle ball in the relevant states.
        # Right/right is harmless if already on Mudkip; then A confirms.
        return _next_sequence_action(state, "_ui_starter_mudkip_step", ("right", "right", "a", "a"), repeat_last=True)
    if "your name" in guidance or "default name" in guidance or "on-screen keyboard" in guidance:
        # START accepts the default name on the Emerald naming keyboard; A clears
        # following confirmation/text pages if START landed on an adjacent screen.
        return _next_sequence_action(state, "_ui_default_name_step", ("start", "a", "a"), repeat_last=True)
    if "yes" in guidance and "no" not in guidance:
        return "a"
    if "no" in guidance and "yes" not in guidance:
        return _next_sequence_action(state, "_ui_select_no_step", ("down", "a"))
    return "a"


def navigate_ui(state: dict[str, Any], intent: str = "confirm", action_guidance: str | None = None) -> str:
    game = state.get("game") or {}
    game_state = str(game.get("game_state") or "").lower()
    in_battle = bool(game.get("is_in_battle"))
    dialog_open = is_dialog_open(state)
    if action_guidance:
        return _guidance_action(state, action_guidance)
    if _visual_clock_ui(state.get("_env")):
        if intent in {"confirm", "select_yes", "select_start"}:
            return _clock_confirm_action(state)
        return "b"
    if not in_battle and not dialog_open and game_state in {"", "overworld", "field", "none"}:
        return "no_op"
    if dialog_open:
        if _is_mudkip_nickname_prompt(state):
            return _mudkip_nickname_action(state)
        if intent == "select_no":
            return _next_sequence_action(state, "_ui_select_no_step", ("down", "a"))
        return "a"
    if intent in {"confirm", "select_yes"}:
        return "a"
    if intent == "select_no":
        return _next_sequence_action(state, "_ui_select_no_step", ("down", "a"))
    if intent in {"exit", "cancel"}:
        return "b"
    if intent == "select_start":
        return "start"
    return "a"


# Emerald gBattleTypeFlags; bit 0x8 = BATTLE_TYPE_TRAINER. (The memory reader's own
# is_trainer_battle uses a stale address + wrong bit and always reads 0, so read here.)
_BATTLE_TYPE_FLAGS_ADDR = 0x02022FEC
_BATTLE_TYPE_TRAINER = 0x08


def _is_trainer_battle(env: Any) -> bool:
    if env is None:
        return False
    try:
        reader = getattr(env, "memory_reader", None)
        if reader is None:
            return False
        return bool(reader._read_u32(_BATTLE_TYPE_FLAGS_ADDR) & _BATTLE_TYPE_TRAINER)
    except Exception:
        return False


def handle_battle(state: dict[str, Any], strategy: str = "fight") -> str:
    # Deterministic simple policies. For Mudkip/Roxanne, Water Gun is usually move slot 4 in Heatz assumptions.
    if strategy == "run" and _is_trainer_battle(state.get("_env")):
        # Trainer battles can't be fled, so fight through instead of looping on a dead run.
        strategy = "fight"
    if strategy == "water_gun":
        prev = state.get("_water_gun_prev_action")
        if prev is None:
            state["_water_gun_prev_action"] = "a"
            return "a"
        if prev == "a":
            state["_water_gun_prev_action"] = "right"
            return "right"
        if prev == "right":
            state["_water_gun_prev_action"] = "down"
            return "down"
        state["_water_gun_prev_action"] = "a"
        return "a"
    if strategy == "run":
        return _next_sequence_action(state, "_battle_run_step", ("a", "right", "down", "a"), repeat_last=False)
    # fight: just press A. Trainer battles are detected up front (we never navigate to
    # RUN in one), so the menu cursor stays on FIGHT and mashing A picks the first move
    # each turn — the fastest win, which also minimises chip damage taken.
    return "a"


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _goal_object_at(state: dict[str, Any], goal_x: int, goal_y: int) -> bool:
    map_state = state.get("map") or {}
    object_sources = [
        (map_state.get("porymap") or {}).get("objects") or [],
        map_state.get("object_events") or [],
        map_state.get("objects") or [],
    ]
    for objects in object_sources:
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_x = _as_int(obj.get("x") if "x" in obj else obj.get("current_x"))
            obj_y = _as_int(obj.get("y") if "y" in obj else obj.get("current_y"))
            if obj_x == goal_x and obj_y == goal_y:
                return True
    return False


def _direction_to_adjacent(x: int, y: int, goal_x: int, goal_y: int) -> str | None:
    dx = goal_x - x
    dy = goal_y - y
    if abs(dx) + abs(dy) != 1:
        return None
    if dx == -1:
        return "left"
    if dx == 1:
        return "right"
    if dy == -1:
        return "up"
    if dy == 1:
        return "down"
    return None


def _facing_matches_action(facing: str | None, action: str) -> bool:
    return _heatz_facing(facing) == {
        "up": "north",
        "down": "south",
        "left": "west",
        "right": "east",
    }[action]


_DIR_DELTA = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}


def _nav_memory(state: dict[str, Any], x: int, y: int, facing: Any, loc: Any) -> dict:
    """Per-map memory of tiles we bumped into.

    Trainers and other NPCs block tiles but are absent from the static collision
    map (object-event memory is often unreadable mid-route), so the pathfinder
    walks straight into them. We learn a tile is blocked when a directional move
    we were *already facing* failed to change our position (a first press toward a
    new facing only turns in place, so facing must already match to count as a
    real bump). Reset when the map changes.
    """
    nav = state.get("_nav_obstacles")
    if not isinstance(nav, dict) or nav.get("loc") != loc:
        nav = {"loc": loc, "blocked": [], "last": None}
    last = nav.get("last")
    if last is not None:
        lx, ly, lfacing, lact = last
        if (lx, ly) == (x, y) and lact in _DIR_DELTA and _facing_matches_action(lfacing, lact):
            dx, dy = _DIR_DELTA[lact]
            tile = [x + dx, y + dy]
            if tile not in nav["blocked"]:
                nav["blocked"].append(tile)
    return nav


def find_path_action(state: dict[str, Any], goal_x: int, goal_y: int, use_vlm_fallback: bool = False, max_distance: int = 150) -> str:
    if is_dialog_open(state):
        return "a"
    ensure_porymap_state(state)
    pos = state.get("player", {}).get("position") or {}
    x, y = pos.get("x"), pos.get("y")
    if x is None or y is None:
        return "no_op"
    x, y, goal_x, goal_y = int(x), int(y), int(goal_x), int(goal_y)
    facing = state.get("facing") or (state.get("player") or {}).get("facing")

    loc = (state.get("map") or {}).get("location")
    nav = _nav_memory(state, x, y, facing, loc)
    blocked = nav["blocked"]

    def _remember(action: str | None) -> str:
        nav["last"] = (x, y, facing, action) if action in _DIR_DELTA else None
        state["_nav_obstacles"] = nav
        return action if action is not None else "no_op"

    # Heatz's original pathfinding interacts with adjacent NPC/object goals.
    # Several generated policies target object coordinates, e.g. Birch's bag.
    adjacent_action = _direction_to_adjacent(x, y, goal_x, goal_y)
    if adjacent_action and _goal_object_at(state, goal_x, goal_y):
        if _facing_matches_action(facing, adjacent_action):
            _remember(None)  # interacting; no movement to track
            return "a"
        return _remember(adjacent_action)

    try:
        from utils.mapping.pathfinding import Pathfinder

        # Route around dynamically-discovered obstacles (e.g. trainers) that the
        # static collision map misses. blocked_coords=None preserves prior behaviour.
        path = Pathfinder().find_path(
            (x, y),
            (goal_x, goal_y),
            state,
            max_distance=max_distance,
            allow_partial=True,
            blocked_coords=[tuple(t) for t in blocked] or None,
        )
        if path:
            return _remember(str(path[0]).lower())
    except Exception as exc:
        logger.debug("Pathfinding failed: %s", exc)

    # Greedy fallback toward the goal, then any escape, always skipping known blocks.
    prefs: list[str] = []
    if goal_x < x:
        prefs.append("left")
    if goal_x > x:
        prefs.append("right")
    if goal_y < y:
        prefs.append("up")
    if goal_y > y:
        prefs.append("down")
    for cand in (*prefs, "down", "up", "left", "right"):
        dx, dy = _DIR_DELTA[cand]
        if [x + dx, y + dy] not in blocked:
            return _remember(cand)
    return _remember(None)


def log(message: object) -> None:
    logger.info("[heatz] %s", message)


def _starter_frame_rgb(env: Any):
    if env is None:
        return None
    try:
        frame = env.get_screenshot()
        if frame is None:
            return None
        if getattr(frame, "mode", "RGB") != "RGB":
            frame = frame.convert("RGB")
        if frame.size[0] < 240 or frame.size[1] < 160:
            return None
        return np.asarray(frame)  # shape (H, W, 3), indexed [y, x]
    except Exception:
        return None


def _region_mean(arr, x0: int, x1: int, y0: int, y1: int):
    return arr[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def _starter_ui_active(env: Any) -> bool:
    """True on Birch's bag starter screens (ball select or choose-confirm).

    The bottom message box is near-white on both starter screens but green grass
    in the overworld, which cleanly separates the UI from normal walking.
    """
    arr = _starter_frame_rgb(env)
    if arr is None:
        return False
    r, g, b = _region_mean(arr, 20, 220, 138, 156)
    return r > 200 and g > 200 and b > 200


def _starter_confirm_species(env: Any) -> str | None:
    """Species shown in the "Do you choose this POKéMON?" circle, or None.

    The big sprite circle has an unmistakable dominant colour per starter: Mudkip
    is blue, Torchic orange, Treecko green. Returns None on the ball-selection
    screen (no sprite circle) so the caller keeps navigating toward Mudkip.
    """
    arr = _starter_frame_rgb(env)
    if arr is None:
        return None
    r, g, b = _region_mean(arr, 108, 132, 58, 86)
    if b >= 175 and b > r + 15:
        return "mudkip"
    if r >= 195 and r > b + 40:
        return "torchic"
    if g >= 140 and b < 130 and r < 185:
        return "treecko"
    return None


def _select_starter_action(state: dict[str, Any]) -> str:
    """Deterministically choose Mudkip from Birch's bag, self-correcting.

    The original agent makes this a VLM-guided choice; a blind button sequence is
    unreliable here because inputs drop during UI transitions. Instead we read the
    confirm screen every step: confirm only when Mudkip's sprite is shown, cancel
    on any other starter, and otherwise step right toward the Mudkip ball.
    """
    env = state.get("_env")
    species = _starter_confirm_species(env)
    if species == "mudkip":
        return "a"  # cursor defaults to YES on the confirm prompt
    if species in ("torchic", "treecko"):
        return "b"  # wrong starter -> cancel back to ball selection
    # Ball-selection screen (or lead-in text): alternate right then A so we land
    # on the right-hand Mudkip ball and open its confirm. Alternating is robust to
    # a dropped directional input (the confirm check above catches a wrong pick).
    step = _as_int(state.get("_ui_starter_step")) or 0
    state["_ui_starter_step"] = (step + 1) % 2
    return "right" if step == 0 else "a"


class HeatzPolicy:
    def __init__(self, event_id: str, policy_path: str | Path):
        self.event_id = event_id
        self.policy_path = Path(policy_path)
        self.run_fn = self._load_policy()
        self._scratch: dict[str, Any] = {}

    def _load_policy(self) -> Callable[[dict[str, Any]], str]:
        code = self.policy_path.read_text(encoding="utf-8")
        tools_module = types.ModuleType("tools")
        for name, obj in {
            "find_path_action": find_path_action,
            "handle_battle": handle_battle,
            "navigate_ui": navigate_ui,
            "is_dialog_open": is_dialog_open,
            "log": log,
        }.items():
            setattr(tools_module, name, obj)
        sys.modules["tools"] = tools_module
        namespace = {
            "find_path_action": find_path_action,
            "handle_battle": handle_battle,
            "navigate_ui": navigate_ui,
            "is_dialog_open": is_dialog_open,
            "log": log,
        }
        exec(compile(code, str(self.policy_path), "exec"), namespace)
        run_fn = namespace.get("run")
        if not callable(run_fn):
            raise RuntimeError(f"Heatz policy {self.policy_path} does not define run(state)")
        return run_fn

    def act(self, state: dict[str, Any]) -> str:
        state.update(self._scratch)
        env = state.get("_env")
        if not (state.get("game") or {}).get("is_in_battle"):
            # Reset per-battle scratch between battles so each encounter's run/fight
            # button sequence starts clean (e.g. the fight sequence begins by pinning
            # the cursor to FIGHT rather than mid-cycle).
            for key in [k for k in self._scratch if k.startswith("_battle_")]:
                self._scratch.pop(key, None)
                state.pop(key, None)
        party = (state.get("player") or {}).get("party") or []
        if self.event_id == "STARTER_CHOSEN" and not party and _starter_ui_active(env):
            # Picking the starter is a VLM-guided decision in the original agent.
            # Replay it deterministically so we always end up with Mudkip (Water),
            # which every downstream policy and postcondition assumes.
            action = _select_starter_action(state)
        elif _visual_clock_ui(env):
            # The Emerald clock-setting screen reads as game_state="overworld" with no
            # dialog in memory, so per-event policies don't route to the clock handler.
            # Apply the global clock UI handling here, matching the original Heatz agent.
            action = navigate_ui(state, intent="confirm")
        else:
            action = self.run_fn(state)
        # Preserve tiny state-machine scratch keys used by helper policies.
        for key, value in state.items():
            if key.startswith(("_water_gun", "_ui_", "_battle_", "_nav")):
                self._scratch[key] = value
        try:
            return normalize_action(action)
        except ValueError:
            return "WAIT"


def policy_path(policy_dir: str | Path, event_id: str) -> Path:
    return Path(policy_dir) / event_id / f"{event_id}.py"


def completed_state_path(policy_dir: str | Path, event_id: str) -> Path:
    return Path(policy_dir) / event_id / f"{event_id}_completed.state"


def completed_milestones_path(policy_dir: str | Path, event_id: str) -> Path:
    return Path(policy_dir) / event_id / f"{event_id}_completed_milestones.json"

