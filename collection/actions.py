"""Action normalization and timing helpers for direct emulator collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator


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


# Fixed schedules survive ONLY for explicit `timing=` overrides (identical to pre-W33
# behavior for callers that pass one); the default path is condition-based (below).
# "slow" retired (dead — no caller ever passed it).
DEFAULT_TIMING = {
    "fast": ActionTiming(hold_frames=6, release_frames=12),
    "normal": ActionTiming(hold_frames=12, release_frames=48),
}


# ---------------------------------------------------------------------------
# Condition-based pacing (W33 §3.5). Measured ground truth (2026-08-20 feasibility
# study, real emulator): the walk animation is exactly 16 frames/tile; the tile coord
# commits to the step's DESTINATION on the step's first frame; a turn from standstill
# consumes 8 frames BEFORE that commit (so a blanket 6-frame hold cannot start a
# post-turn step — the documented navigator failure, navigator.py _step). The fixed
# normal schedule (12+48=60) therefore wastes ~73% of its frames; coverage's
# press-until-change loop achieves median 16. This module encodes that policy once,
# shared by execution (perform_action) and planning (_simulate_action).


@dataclass(frozen=True)
class SettlePolicy:
    """Pacing for one action class: hold until the effect commits (or cap), then settle
    until the world is provably at rest (or cap)."""
    hold_frames: int        # taps: exact held frames; directional: CAP on the held poll
    poll_hold: bool         # directional: release as soon as the tile commits
    settle_cap: int         # max released frames waiting for the stability predicate
    stable_frames: int = 2  # consecutive stable probes that end the settle


# Hold-cap 20 = turn (8) + commit (1) with >2x margin; a blocked press pays the full
# cap (bump feedback still recorded), which the x3 action-budget rescale absorbs.
# Settle cap 24 > the 16-frame walk animation + stability confirmation.
DIRECTIONAL_SETTLE = SettlePolicy(hold_frames=20, poll_hold=True, settle_cap=24)
# Taps (A/B/START/SELECT/L/R and WAIT): 6 held frames register reliably; UI reactions
# don't move the light nav snapshot, so the settle exits after `stable_frames`
# (~8 frames/press when nothing reacts, vs 60 fixed).
TAP_SETTLE = SettlePolicy(hold_frames=6, poll_hold=False, settle_cap=12)


@dataclass(frozen=True)
class PaceProbe:
    """One polled frame for the pacing loop: player tile identity + rest state."""
    pos: tuple                # (map, x, y) — flips to a step's DESTINATION on its first frame
    facing: str | None        # facing nibble from the player's gObjectEvents record
    mid_step: bool            # Entity.mid_step: walking between tiles (animation running)

    @property
    def stable_key(self) -> tuple:
        return (*self.pos, self.facing)


def settle_policy_for(action: object) -> SettlePolicy:
    return DIRECTIONAL_SETTLE if normalize_action(action) in EXPLORE_ACTIONS else TAP_SETTLE


def paced_action_frames(
    action: object,
    probe: Callable[[], PaceProbe],
    *,
    policy: SettlePolicy | None = None,
) -> Iterator[list[str]]:
    """Yield per-frame button lists for ONE condition-paced press (W33 §3.5).

    `probe()` is sampled after the caller has run each yielded frame, so the generator
    sees the world's reaction to the frame it just scheduled. This generator is THE
    frame schedule for both real execution (DirectEmulatorRunner.perform_action, which
    records every yielded frame) and planning simulation (collect_events._simulate_action,
    scratch emulator) — one source, so the planner simulates EXACTLY what execution does.
    """
    normalized = normalize_action(action)
    buttons = action_to_buttons(normalized)
    policy = policy or settle_policy_for(normalized)
    start = probe()
    cur = start
    # Hold: directional presses poll for the commit (tile flips to the destination) and
    # release immediately, so one press can never chain into a second tile (the phantom
    # 2-tile-edge failure collect_coverage._step_once documents); taps hold their exact
    # registration window (their effects aren't visible in the probe).
    for _ in range(policy.hold_frames):
        yield buttons
        cur = probe()
        if policy.poll_hold and cur.pos != start.pos:
            break
    # Settle: input released, wait until the player is not mid-step AND the light nav
    # snapshot holds still for `stable_frames` consecutive frames, capped. A transition
    # that outlives the cap (e.g. a warp fade) simply continues into the next action's
    # frames — every frame is recorded either way.
    stable = 0
    last = cur
    for _ in range(policy.settle_cap):
        yield []
        cur = probe()
        if not cur.mid_step and cur.stable_key == last.stable_key:
            stable += 1
        else:
            stable = 0
        last = cur
        if stable >= policy.stable_frames:
            break


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

