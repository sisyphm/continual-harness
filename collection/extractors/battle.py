"""Battle logic state — `gBattleMons` + battle flags, the A′ battle condition.

Addresses are VANILLA US Emerald, re-proven on recorded data: at a frame whose battle UI reads
"MUDKIP Lv10 31/33" the struct below yields species=283(Mudkip), hp=31@+0x28, level=10@+0x2A,
maxHP=33@+0x2C, moves=[Tackle,Growl,Mud-Slap,Water Gun] — every field matches the screen.

  gBattleMons          0x02024084 — struct BattlePokemon[4], 0x58 bytes each:
      +0x00 u16 species   +0x0C u16 moves[4]   +0x24 u8 pp[4]
      +0x28 u16 hp        +0x2A u8 level       +0x2C u16 maxHP    +0x4C u32 status1
  gBattleTypeFlags     0x02022AAE — u32 bitflags (wild/trainer/etc.), recorded raw
  gBattleCommunication 0x02024A60 — 8 bytes of battle phase machine, recorded raw (the model's
                       battle-phase token; exact semantics learned downstream, not asserted)
  gMain.inBattle       0x030026F9 bit1 (Phase-0-validated battle gate)
"""

from __future__ import annotations

from dataclasses import dataclass

from collection.extractors.ram import GBAState

G_BATTLE_MONS, MON_SIZE, N_BATTLERS = 0x02024084, 0x58, 4
G_BATTLE_TYPE = 0x02022AAE
G_BATTLE_COMM = 0x02024A60
IN_BATTLE_ADDR, IN_BATTLE_MASK = 0x030026F9, 0x02
MAX_SPECIES = 440                          # Gen-3 internal species id bound


@dataclass(frozen=True)
class Battler:
    slot: int                              # 0/2 = player side, 1/3 = foe side
    species: int
    level: int
    hp: int
    max_hp: int
    status1: int                           # sleep/poison/burn/... bitfield, raw
    moves: tuple[int, int, int, int]
    pp: tuple[int, int, int, int]


def in_battle(st: GBAState) -> bool:
    return bool(st.u8(IN_BATTLE_ADDR) & IN_BATTLE_MASK)


def battle_state(st: GBAState) -> dict | None:
    """Battlers + raw battle flags/phase, or None outside battle."""
    if not in_battle(st):
        return None
    battlers = []
    for slot in range(N_BATTLERS):
        b = G_BATTLE_MONS + slot * MON_SIZE
        species = st.u16(b)
        if not (0 < species <= MAX_SPECIES):
            continue                                       # empty slot (singles use 0 and 1)
        hp, max_hp = st.u16(b + 0x28), st.u16(b + 0x2C)
        if max_hp == 0 or hp > max_hp + 100:               # struct not yet initialized this battle
            continue
        battlers.append(Battler(
            slot=slot, species=species, level=st.u8(b + 0x2A), hp=hp, max_hp=max_hp,
            status1=st.u32(b + 0x4C),
            moves=tuple(st.u16(b + 0x0C + 2 * i) for i in range(4)),
            pp=tuple(st.u8(b + 0x24 + i) for i in range(4)),
        ))
    return {"battlers": battlers,
            "battle_type": st.u32(G_BATTLE_TYPE),
            "comm": st.bytes(G_BATTLE_COMM, 8)}
