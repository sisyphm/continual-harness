"""Action normalization and timing helpers for direct emulator collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


EXPLORE_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
BUTTON_ORDER = ("A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R")

_ACTION_ALIASES = {
    "NO_OP": "WAIT",
    "NOOP": "WAIT",
    "NONE": "WAIT",
    "": "WAIT",
}


@dataclass(frozen=True)
class ActionTiming:
    hold_frames: int
    release_frames: int


DEFAULT_TIMING = {
    "fast": ActionTiming(hold_frames=6, release_frames=12),
    "normal": ActionTiming(hold_frames=12, release_frames=48),
    "slow": ActionTiming(hold_frames=12, release_frames=60),
}


def normalize_action(action: object) -> str:
    """Return canonical uppercase action name."""
    normalized = str(action or "").strip().upper()
    normalized = _ACTION_ALIASES.get(normalized, normalized)
    if normalized not in BUTTON_ORDER and normalized != "WAIT":
        raise ValueError(f"Unsupported action: {action!r}")
    return normalized


def action_to_buttons(action: object) -> list[str]:
    """Convert one canonical action into per-frame emulator buttons."""
    normalized = normalize_action(action)
    if normalized == "WAIT":
        return []
    return [normalized]


def normalize_button_list(buttons: Iterable[object]) -> list[str]:
    return [normalize_action(button) for button in buttons if normalize_action(button) != "WAIT"]


def button_vec(buttons: Iterable[object]) -> dict[str, int]:
    held = set(normalize_button_list(buttons))
    return {button: int(button in held) for button in BUTTON_ORDER}


def timing_for(speed: str = "normal", *, hold_frames: int | None = None, release_frames: int | None = None) -> ActionTiming:
    base = DEFAULT_TIMING.get(speed, DEFAULT_TIMING["normal"])
    return ActionTiming(
        hold_frames=max(0, int(base.hold_frames if hold_frames is None else hold_frames)),
        release_frames=max(0, int(base.release_frames if release_frames is None else release_frames)),
    )


def update_facing(previous_facing: str | None, action: object) -> str:
    normalized = normalize_action(action)
    if normalized == "UP":
        return "UP"
    if normalized == "DOWN":
        return "DOWN"
    if normalized == "LEFT":
        return "LEFT"
    if normalized == "RIGHT":
        return "RIGHT"
    return previous_facing or "DOWN"


def run_action_frames(action: object, timing: ActionTiming) -> list[list[str]]:
    """Return full per-frame button schedule for one action plus release frames."""
    buttons = action_to_buttons(action)
    return [buttons for _ in range(timing.hold_frames)] + [[] for _ in range(timing.release_frames)]

