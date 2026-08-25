"""Proactive heal-and-return (W34, owner ruling).

Faint-as-heal is demoted to an emergency backstop: a white-out teleports across
maps, halves the wallet and tears the run out of its block schedule mid-assignment
— wave 1's receipts show blocks abandoning trainer targets (`lead_hp_low`) rather
than risk it. This module makes health a solvable errand instead: walk to the
nearest stage-reachable Center, talk to the nurse ACROSS the counter, walk back
out, continue the assignment. Deterministic, ~2x travel + ~800 frames of dialog.

The nurse-trip mechanics are the battle-tested ones from grind_evolve (counter-talk
because her own tile is sealed behind "######" and pathing to it heals nothing;
explicit dialog release because returning mid-sign-off pins the walker for 90k+
frames). Extracted here so every block shares ONE implementation.

Triggers (`ensure_healthy`): lead HP below `hp_floor`, or every damaging move at
0 PP — the nurse restores PP with HP, and PP exhaustion is the quieter killer (a
dry lead Struggle-loops). Callers check between goals, never mid-battle.
"""

from __future__ import annotations

_NURSE = (7, 2)                 # nurse's own tile: talk ACROSS the counter, never path to it

# Nearest stage-reachable Center by overworld region (map keys "group,num").
# Verified interior keys from the frozen world data: Oldale 2,2 / Petalburg 8,4 /
# Rustboro 11,5. Unlisted maps (interiors, gyms) fall back to their region via the
# player's LAST overworld key, which every caller sits on at goal boundaries.
_CENTER_BY_REGION = {
    # Oldale basin: Littleroot, 101, Oldale, 102 east half, 103
    "0,9": "2,2", "0,16": "2,2", "0,10": "2,2", "0,18": "2,2",
    # Petalburg basin: 102, Petalburg, 104 south, Petalburg Woods
    "0,17": "8,4", "0,0": "8,4", "0,19": "8,4", "24,11": "8,4",
    # Rustboro basin: 104 north handled by 0,19's Petalburg pick pre-woods; 116,
    # Rustboro, 115 all route to Rustboro
    "0,3": "11,5", "0,31": "11,5", "0,30": "11,5",
}
_DEFAULT_CENTER = "11,5"


def nearest_center(map_key: str) -> str:
    return _CENTER_BY_REGION.get(map_key, _DEFAULT_CENTER)


def _release_dialog(runner, src: str, tries: int = 40) -> bool:
    """Step out of the nurse's closing dialog before anyone tries to walk (measured:
    a healed lead pinned at (7,4) for 93k frames without this; nav._clear_dialog
    cannot close this box, the confirm action that drove the heal can)."""
    from collection.playthrough.blocks.base import _hstate
    from collection.heatz_adapter import is_dialog_open, navigate_ui
    from collection.actions import normalize_action
    from collection import navigator as _nav
    for _ in range(tries):
        h = _hstate(runner)
        if is_dialog_open(h):
            runner.perform_action(normalize_action(navigate_ui(h, intent="confirm")),
                                  metadata={"src": src})
            continue
        if _nav._step(runner, "down"):
            return True
        runner.perform_action("B", metadata={"src": src})
    return False


def heal_at_center(runner, mk, center_key: str, *, deadline: int,
                   src: str = "heal_trip") -> bool:
    """Nurse-heal trip: enter the Center, walk toward the nurse, talk across the
    counter, confirm until the lead reads full HP (PP restores with it), release
    the sign-off dialog. True iff the lead left at full HP."""
    from collection.playthrough.blocks.base import _hstate, goto_map_safe
    from collection.playthrough.blocks.grind_evolve import read_lead
    from collection.heatz_adapter import is_dialog_open, navigate_ui
    from collection.actions import normalize_action
    from collection import navigator as _nav
    if not goto_map_safe(runner, mk, center_key, deadline):
        return False
    for _ in range(200):
        if runner.frame_idx >= deadline:
            break
        lead = read_lead(runner)
        if lead is not None and lead["max_hp"] and lead["hp"] == lead["max_hp"]:
            _release_dialog(runner, src)
            return True
        h = _hstate(runner)
        if is_dialog_open(h):
            runner.perform_action(normalize_action(navigate_ui(h, intent="confirm")),
                                  metadata={"src": src})
            continue
        _t, _cx, _cy = _nav._state(runner)
        if _t is None:
            runner.perform_action("A", metadata={"src": src})
            continue
        _dx, _dy = _NURSE[0] - _cx, _NURSE[1] - _cy
        _face = ("UP" if _dy < 0 else "DOWN" if _dy > 0
                 else "LEFT" if _dx < 0 else "RIGHT")
        runner.perform_action(_face, metadata={"src": src})
        runner.perform_action("A", metadata={"src": src})
    lead = read_lead(runner)
    _full = bool(lead is not None and lead["max_hp"] and lead["hp"] == lead["max_hp"])
    _release_dialog(runner, src)         # never hand back a frozen, dialog-locked run
    return _full


def needs_heal(runner, *, hp_floor: float = 0.35) -> bool:
    """True when the lead is below the HP floor or has no damaging PP left. Reads
    the PARTY (not the battle struct) — call only outside battle."""
    from collection.playthrough.blocks.grind_evolve import read_lead
    try:
        lead = read_lead(runner)
        if not lead or not lead.get("max_hp"):
            return False                  # unreadable: do not block the caller
        if lead["hp"] / lead["max_hp"] < hp_floor:
            return True
        from collection.move_data import damaging_slot
        moves, pp = lead.get("moves") or [], lead.get("pp") or []
        # damaging_slot returns None iff no damaging move has PP left — dry lead.
        return bool(moves) and damaging_slot(moves, pp) is None
    except Exception:
        return False


def ensure_healthy(runner, mk, *, hp_floor: float = 0.35, trip_frames: int = 12_000,
                   src: str = "ensure_healthy") -> str:
    """'healthy' (no trip needed) | 'healed' (trip succeeded) | 'heal_failed'.

    The caller resumes its OWN navigation afterwards — a goal-driven block re-walks
    to its current target anyway, so there is no separate 'return' step to break."""
    if not needs_heal(runner, hp_floor=hp_floor):
        return "healthy"
    from collection import navigator as _nav
    t, _, _ = _nav._state(runner)
    key = f"{t.map_group},{t.map_num}" if t is not None else _DEFAULT_CENTER
    center = nearest_center(key)
    ok = heal_at_center(runner, mk, center, deadline=runner.frame_idx + trip_frames, src=src)
    print(f"heal: {key} -> center {center}: {'healed' if ok else 'FAILED'}", flush=True)
    return "healed" if ok else "heal_failed"
