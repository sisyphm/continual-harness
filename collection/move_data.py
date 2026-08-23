"""Move power from the ROM, so "can this pokemon still attack?" is answerable.

The collector's battle driver always steers to move slot 0. That is fine until the slot
runs dry: Emerald then refuses the selection, reopens the menu, and the driver picks it
again — forever. Measured on the 2026-08-23 wave, this was 8 of 12 failures, always the
same shape:

    Scratch(pow40,pp0)  Growl(pow0,pp40)  FocusEnergy(pow0,pp30)  Ember(pow40,pp0)

Every DAMAGING move empty, PP left only in the zero-power ones. So "pick any slot with
PP" is the wrong repair — it would select GROWL and stall the battle instead of the
menu, which looks healthier (frames advance) while never winning. A slot is usable only
if it has PP *and* deals damage; when none does, the lead is out of ammunition and the
answer is a Centre trip, which restores PP.

gBattleMoves is located by structure, not by a hardcoded symbol address: the table is
the unique ROM offset where twelve known (move id -> base power) pairs all agree.
"""

from __future__ import annotations

_ENTRY = 12          # sizeof(struct BattleMove)
_POWER_OFF = 1       # u8 power at +1
_TABLE: int | None = None
_ROM: bytes | None = None

# (move id, base power) — enough distinct values to pin the table uniquely
_ANCHORS = ((1, 40), (10, 40), (33, 35), (45, 0), (52, 40), (116, 0),
            (24, 30), (71, 20), (98, 40), (43, 0), (55, 40), (64, 35))


def _load(rom_path: str = "Emerald-GBAdvance/rom.gba") -> None:
    global _TABLE, _ROM
    if _TABLE is not None:
        return
    _ROM = open(rom_path, "rb").read()
    for base in range(0, len(_ROM) - _ENTRY * 200, 4):
        if all(_ROM[base + _ENTRY * mid + _POWER_OFF] == want for mid, want in _ANCHORS):
            _TABLE = base
            return
    raise RuntimeError("gBattleMoves not found — move power unavailable")


def move_power(move_id: int, rom_path: str = "Emerald-GBAdvance/rom.gba") -> int:
    if not move_id:
        return 0
    _load(rom_path)
    off = _TABLE + _ENTRY * int(move_id) + _POWER_OFF
    return _ROM[off] if 0 <= off < len(_ROM) else 0


def damaging_slot(moves, pp) -> int | None:
    """First slot with PP that actually deals damage; None = cannot attack at all."""
    try:
        for i in range(4):
            if (i < len(moves) and i < len(pp) and moves[i]
                    and pp[i] > 0 and move_power(moves[i]) > 0):
                return i
    except Exception:
        return 0          # unreadable: assume slot 0 rather than force a heal loop
    return None


def out_of_ammo(lead) -> bool:
    """True when the lead holds no damaging move with PP (the deadlock precondition)."""
    if not lead:
        return False
    return damaging_slot(lead.get("moves") or [], lead.get("pp") or []) is None


def damaging_pp(lead) -> int:
    """Total PP across moves that can actually deal damage."""
    if not lead:
        return 99
    mv, pp = lead.get("moves") or [], lead.get("pp") or []
    return sum(pp[i] for i in range(min(len(mv), len(pp)))
               if mv[i] and pp[i] > 0 and move_power(mv[i]) > 0)


def low_ammo(lead, margin: int = 8) -> bool:
    """Nearly out of attacking PP — heal NOW, while a Centre trip is still possible.

    Waiting for zero is too late. A lead runs dry in the MIDDLE of a battle, and if that
    battle is a TRAINER fight it cannot be fled: measured on exp_008_treecko, the only
    PP left was Leer(29) and draining it to reach Struggle cost ~15k frames per turn,
    i.e. ~400k frames — far slower than the watchdog's patience. Healing at a margin
    means the run is never caught empty mid-battle in the first place.
    """
    return damaging_pp(lead) <= margin
