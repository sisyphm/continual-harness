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
# gHealthboxSpriteIds @ 0x03005D70 (u8[4]) — battler -> its HP-box sprite. The HP box is created and
# made visible only once that battler's MON is on the field (sent out), so "healthbox visible" is the
# clean, universal MON-is-out gate. VALIDATED on recorded trainer/wild battles (2026-06-20): it flips
# visible exactly at send-out and stays visible through the rest of the battle.
G_HEALTHBOX = 0x03005D70
# gTrainerBattleOpponent_A @ 0x02038BCA (u16) — the opponent trainer id (index into gTrainers).
# VALIDATED: on every recorded trainer battle gTrainers[this].party[0].species == gBattleMons[1]
# (enemy lead), across 9 distinct opponents.
G_TRAINER_OPPONENT = 0x02038BCA
# OBJ palette slot Emerald assigns the OPPONENT trainer front-pic during the intro. VALIDATED on the
# corpus: the enemy MON is NEVER this palette (122k frames, all pal 1) and 9 distinct opponents all
# use it — so it cleanly separates the opening trainer pic from the enemy mon's slide-in (pal 1).
ENEMY_TRAINER_PAL = 7
_OAM_DIMS = {(0, 0): (8, 8), (0, 1): (16, 16), (0, 2): (32, 32), (0, 3): (64, 64),
             (1, 0): (16, 8), (1, 1): (32, 8), (1, 2): (32, 16), (1, 3): (64, 32),
             (2, 0): (8, 16), (2, 1): (8, 32), (2, 2): (16, 32), (2, 3): (32, 64)}


def _s8(v: int) -> int:
    return v - 256 if v >= 128 else v


def _healthbox_visible(st, battler: int) -> bool:
    """True iff `battler`'s HP box is on screen — i.e. its mon is out (see G_HEALTHBOX)."""
    sid = st.u8(G_HEALTHBOX + battler)
    if sid >= 64:
        return False
    return not ((st.u16(GSPRITES_BAT + sid * SPRITE_STRIDE + 0x3E) >> 2) & 1)


def trainer_pic_table(rom: bytes) -> dict[int, int]:
    """trainer id -> (trainerPic + 1), from gTrainers in ROM (0 reserved = none/non-trainer). Reuses
    rom_manifest.scan_trainers for the validated table base + the off-by-one base-index detection."""
    from collection.extractors.rom_manifest import Rom, scan_trainers
    r = Rom(rom)
    tr_base, trainers = scan_trainers(r)
    prev = tr_base - 0x28                                       # TRAINER_NONE sits one stride back
    base_idx = 1 if r.u32(prev + 0x20) == 0 and r.u32(prev + 0x24) == 0 else 0
    out: dict[int, int] = {}
    for t in trainers:
        a = tr_base + (t["id"] - base_idx) * 0x28
        out[t["id"]] = r.u8(a + 3) + 1                          # Trainer.trainerPic @ +0x03, +1 -> 1-based
    return out


def opponent_trainer_pic(st, table: dict[int, int]) -> int:
    """The (1-based) front-pic id of the current opponent trainer, or 0 if unknown / not a trainer."""
    return table.get(st.u16(G_TRAINER_OPPONENT), 0)


def battle_sprites(st) -> list[dict]:
    """On-screen battle sprites for the spatial splat. Two kinds, distinguished per frame:
      • MON (kind 1=player / 2=enemy): the gBattlerSpriteIds sprite of a battler whose HP box is up
        (mon is out) -> gBattleMons species. In a WILD battle (no intro) we keep the historical
        behaviour and emit the mon whenever its sprite is on screen; in a TRAINER battle we REQUIRE
        the healthbox, so the intro trainer pic (which occupies the SAME slot before send-out) is
        never mislabeled as the mon.
      • TRAINER (kind 3): during a trainer-battle opening the opponent's slot holds the trainer FRONT
        pic, not a mon. We emit it (species 0; identity carried by bat_sprite_trainer downstream),
        positively identified by the opponent OBJ palette so the enemy mon's slide-in is excluded.
        The player BACK pic is left unconditioned (brief, identical every battle, and its palette
        collides with the player mon's — the overworld conditioning already carries the avatar).
    Top-left = pos1(+0x20) + pos2(+0x24) + centerToCornerVec(s8 +0x28/+0x29)."""
    if not in_battle(st):
        return []
    trainer_batt = bool(st.u32(G_BATTLE_TYPE) & 0x08)
    out = []
    for battler in range(N_BATTLERS):
        sid = st.u8(G_BATTLER_SPRITE_IDS + battler)
        if sid >= 64:
            continue
        base = GSPRITES_BAT + sid * SPRITE_STRIDE
        if (st.u16(base + 0x3E) >> 2) & 1:                 # invisible -> sprite not on screen
            continue
        w, h = _OAM_DIMS.get(((st.u16(base) >> 14) & 3, (st.u16(base + 2) >> 14) & 3), (64, 64))
        x = st.s16(base + 0x20) + st.s16(base + 0x24) + _s8(st.u8(base + 0x28))
        y = st.s16(base + 0x22) + st.s16(base + 0x26) + _s8(st.u8(base + 0x29))
        if not (w >= 32 and h >= 32 and -w < x < 240 and -h < y < 160):
            continue                                       # reject uninitialized opening-wipe frames
        if trainer_batt and not _healthbox_visible(st, battler):
            pal = (st.u16(base + 0x04) >> 12) & 0xF
            if battler % 2 == 1 and pal == ENEMY_TRAINER_PAL:    # opponent trainer FRONT pic
                out.append({"battler": battler, "species": 0, "kind": 3,
                            "x": x, "y": y, "w": w, "h": h})
            continue                                        # player-back / mon slide-in -> skip
        species = st.u16(G_BATTLE_MONS + battler * MON_SIZE)
        if not (0 < species <= MAX_SPECIES):
            continue
        out.append({"battler": battler, "species": species,
                    "kind": 1 if battler % 2 == 0 else 2,       # 1=player-mon, 2=enemy-mon
                    "x": x, "y": y, "w": w, "h": h})
    return out
