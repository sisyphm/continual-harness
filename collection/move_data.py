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


# --------------------------------------------------------------------------- types
# struct BaseStats is 28 bytes; type1 at +6, type2 at +7. Located the same structural
# way as gBattleMoves: the unique ROM offset where five known species types agree.
_BS_ENTRY = 28
_BS_T1, _BS_T2 = 6, 7
_BASE_STATS: int | None = None
_BS_ANCHORS = ((277, 12, 12), (280, 10, 10), (283, 11, 11), (74, 5, 4), (95, 5, 4))

NORMAL, FIGHT, FLY, POISON, GROUND, ROCK, BUG, GHOST, STEEL = 0, 1, 2, 3, 4, 5, 6, 7, 8
FIRE, WATER, GRASS, ELEC, PSY, ICE, DRAGON, DARK = 10, 11, 12, 13, 14, 15, 16, 17

# Gen-3 chart: only the non-1x entries.
_CHART = {
    NORMAL: {ROCK: .5, STEEL: .5, GHOST: 0},
    FIGHT:  {NORMAL: 2, ROCK: 2, STEEL: 2, ICE: 2, DARK: 2,
             FLY: .5, POISON: .5, BUG: .5, PSY: .5, GHOST: 0},
    FLY:    {FIGHT: 2, BUG: 2, GRASS: 2, ROCK: .5, STEEL: .5, ELEC: .5},
    POISON: {GRASS: 2, POISON: .5, GROUND: .5, ROCK: .5, GHOST: .5, STEEL: 0},
    GROUND: {FIRE: 2, ELEC: 2, POISON: 2, ROCK: 2, STEEL: 2, GRASS: .5, BUG: .5, FLY: 0},
    ROCK:   {FIRE: 2, ICE: 2, FLY: 2, BUG: 2, FIGHT: .5, GROUND: .5, STEEL: .5},
    BUG:    {GRASS: 2, PSY: 2, DARK: 2, FIRE: .5, FIGHT: .5, POISON: .5, FLY: .5,
             GHOST: .5, STEEL: .5},
    GHOST:  {PSY: 2, GHOST: 2, DARK: .5, STEEL: .5, NORMAL: 0},
    STEEL:  {ICE: 2, ROCK: 2, FIRE: .5, WATER: .5, ELEC: .5, STEEL: .5},
    FIRE:   {GRASS: 2, ICE: 2, BUG: 2, STEEL: 2, FIRE: .5, WATER: .5, ROCK: .5, DRAGON: .5},
    WATER:  {FIRE: 2, GROUND: 2, ROCK: 2, WATER: .5, GRASS: .5, DRAGON: .5},
    GRASS:  {WATER: 2, GROUND: 2, ROCK: 2, FIRE: .5, GRASS: .5, POISON: .5, FLY: .5,
             BUG: .5, DRAGON: .5, STEEL: .5},
    ELEC:   {WATER: 2, FLY: 2, ELEC: .5, GRASS: .5, DRAGON: .5, GROUND: 0},
    PSY:    {FIGHT: 2, POISON: 2, PSY: .5, STEEL: .5, DARK: 0},
    ICE:    {GRASS: 2, GROUND: 2, FLY: 2, DRAGON: 2, FIRE: .5, WATER: .5, ICE: .5, STEEL: .5},
    DRAGON: {DRAGON: 2, STEEL: .5},
    DARK:   {PSY: 2, GHOST: 2, FIGHT: .5, DARK: .5, STEEL: .5},
}


def _load_base_stats(rom_path: str = "Emerald-GBAdvance/rom.gba") -> None:
    global _BASE_STATS
    if _BASE_STATS is not None:
        return
    _load(rom_path)
    for base in range(0, len(_ROM) - _BS_ENTRY * 400, 4):
        if all(_ROM[base + _BS_ENTRY * s + _BS_T1] == t1
               and _ROM[base + _BS_ENTRY * s + _BS_T2] == t2 for s, t1, t2 in _BS_ANCHORS):
            _BASE_STATS = base
            return
    raise RuntimeError("gBaseStats not found")


def move_type(move_id: int) -> int:
    _load()
    return _ROM[_TABLE + _ENTRY * int(move_id) + 2] if move_id else NORMAL


def species_types(species_id: int) -> tuple[int, int]:
    _load_base_stats()
    o = _BASE_STATS + _BS_ENTRY * int(species_id)
    return _ROM[o + _BS_T1], _ROM[o + _BS_T2]


def effectiveness(mtype: int, foe_types) -> float:
    row = _CHART.get(mtype, {})
    mult = 1.0
    for t in dict.fromkeys(foe_types):        # dedupe single-typed mons
        mult *= row.get(t, 1.0)
    return mult


def best_slot(moves, pp, foe_species: int | None = None):
    """Slot that does the most damage to THIS foe. None = nothing damaging is available.

    The driver used to take slot 0 unconditionally, which is type-blind and lost the
    rock gym repeatedly: a treecko holding Absorb (GRASS, 2x on rock) attacked with
    Pound (NORMAL, 0.5x) and fought at a quarter of the damage it had. Every starter
    carries a super-effective answer there — Absorb, Double Kick, Water Gun/Mud-Slap —
    and the old selector could not see any of them.
    """
    try:
        foe = tuple(species_types(foe_species)) if foe_species else ()
    except Exception:
        foe = ()
    best, best_score = None, 0.0
    for i in range(4):
        if i >= len(moves) or i >= len(pp) or not moves[i] or pp[i] <= 0:
            continue
        power = move_power(moves[i])
        if power <= 0:
            continue
        score = power * (effectiveness(move_type(moves[i]), foe) if foe else 1.0)
        if score > best_score:
            best, best_score = i, score
    return best
