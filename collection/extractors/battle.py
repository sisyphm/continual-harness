"""Battle logic state — `gBattleMons` + battle flags, the A′ battle condition.

Addresses are VANILLA US Emerald, re-proven on recorded data: at a frame whose battle UI reads
"MUDKIP Lv10 31/33" the struct below yields species=283(Mudkip), hp=31@+0x28, level=10@+0x2A,
maxHP=33@+0x2C, moves=[Tackle,Growl,Mud-Slap,Water Gun] — every field matches the screen.

  gBattleMons          0x02024084 — struct BattlePokemon[4], 0x58 bytes each:
      +0x00 u16 species   +0x0C u16 moves[4]   +0x24 u8 pp[4]
      +0x28 u16 hp        +0x2A u8 level       +0x2C u16 maxHP    +0x4C u32 status1
  gBattleTypeFlags     0x02022FEC — u32 bitflags (wild/trainer/double/etc.), recorded raw
                       (verified on the recorded Roxanne fight: bit 0x08=TRAINER set on every
                       battle frame at 0x02022FEC, while the old 0x02022AAE read 0 always — dead)
  gBattleCommunication 0x02024A60 — 8 bytes of battle phase machine, recorded raw (the model's
                       battle-phase token; exact semantics learned downstream, not asserted)
  gMain.inBattle       0x030026F9 bit1 (Phase-0-validated battle gate)
"""

from __future__ import annotations

from dataclasses import dataclass

from collection.extractors.ram import GBAState

G_BATTLE_MONS, MON_SIZE, N_BATTLERS = 0x02024084, 0x58, 4
G_BATTLE_TYPE = 0x02022FEC
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


# --- battle sprite splat (identity_grounded, PLAN §8.3) ----------------------------------------
# gBattlerSpriteIds @ 0x020241E4 (u8[4]) — battler -> gSprites slot. VALIDATED 2026-06-18 by overlay:
# gBattleMons matches the decomp exactly (0x02024084), so the pokeemerald symbol address applies;
# on a recorded Mudkip-vs-Poochyena battle, slot[0]->Mudkip back-sprite (sp283), slot[1]->Poochyena
# (sp286), positions match the on-screen mons AND their HP boxes; slot reallocation (enemy 3->4->6) is
# followed; the gSprites `invisible` flag (byte 0x3E bit 2) is the mon-is-out signal.
G_BATTLER_SPRITE_IDS = 0x020241E4
GSPRITES_BAT, SPRITE_STRIDE = 0x02020630, 0x44      # struct Sprite (pokeemerald): see offsets below
_OAM_DIMS = {(0, 0): (8, 8), (0, 1): (16, 16), (0, 2): (32, 32), (0, 3): (64, 64),
             (1, 0): (16, 8), (1, 1): (32, 8), (1, 2): (32, 16), (1, 3): (64, 32),
             (2, 0): (8, 16), (2, 1): (8, 32), (2, 2): (16, 32), (2, 3): (32, 64)}


def _s8(v: int) -> int:
    return v - 256 if v >= 128 else v


def battle_sprites(st) -> list[dict]:
    """On-screen battler MON sprites for the spatial species-splat. For each battler whose sprite is
    VISIBLE (mon actually out — not the pre-throw intro): species (gBattleMons) + role + screen
    top-left px + size px. Empty when no battle / mons not out. (Trainer-sprite identity is a
    separate follow-up.) Top-left = pos1(+0x20) + pos2(+0x24) + centerToCornerVec(s8 +0x28/+0x29)."""
    if not in_battle(st):
        return []
    out = []
    for battler in range(N_BATTLERS):
        sid = st.u8(G_BATTLER_SPRITE_IDS + battler)
        if sid >= 64:
            continue
        base = GSPRITES_BAT + sid * SPRITE_STRIDE
        if (st.u16(base + 0x3E) >> 2) & 1:                 # invisible -> mon not on screen yet
            continue
        species = st.u16(G_BATTLE_MONS + battler * MON_SIZE)
        if not (0 < species <= MAX_SPECIES):
            continue
        w, h = _OAM_DIMS.get(((st.u16(base) >> 14) & 3, (st.u16(base + 2) >> 14) & 3), (64, 64))
        x = st.s16(base + 0x20) + st.s16(base + 0x24) + _s8(st.u8(base + 0x28))
        y = st.s16(base + 0x22) + st.s16(base + 0x26) + _s8(st.u8(base + 0x29))
        if not (w >= 32 and h >= 32 and -w < x < 240 and -h < y < 160):
            continue                                       # reject uninitialized opening-wipe frames
        out.append({"battler": battler, "species": species,
                    "kind": 1 if battler == 0 else 2,       # 1=player-mon, 2=enemy-mon (singles)
                    "x": x, "y": y, "w": w, "h": h})
    return out
