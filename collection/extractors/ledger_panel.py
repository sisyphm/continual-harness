"""Per-frame game-state ledger — W33 corpus-v2 §3.3 (RamPanel in the recorder) + §10.2 (ui_state).

Port of the VERIFIED labeler extractor (pokemon-worldmodel `tools/state/label_corpus.py`
class `RamPanel`, the Σ-label authority) onto the harness's own memory seam (`GBAState`),
so labels become a collection output instead of a replay-archaeology project. Differences
from RamPanel are representational only, so every RamPanel field stays derivable from a row:

- RAW values, not panel buckets: hp/max_hp as u16 (RamPanel: 4-bucket), exp as u32
  (RamPanel: exp//256 capped 255). Buckets are derivable from raw; not vice versa (§11:
  every field derivable forever).
- The party-struct series and the battle-copy series are emitted SEPARATELY, plus per-slot
  personalities, so RamPanel's continuous fusion (the party struct FREEZES during battle;
  the battle copy is the live series for the active slot, matched by personality) is a
  downstream one-liner instead of a baked-in choice.
- Absent-slot / out-of-battle values are 0 (species 0 == SPECIES_NONE), not RamPanel's -1 —
  the ledger dtypes are unsigned. Consumers mask by species != 0 / in_battle.

Equality with RamPanel under these transforms is pinned by tests/test_ledger_sink.py
(field-for-field cross-check on live emulator states, battle included).

Address provenance — bases and struct offsets are RamPanel's (references/pokeemerald map
symbols + the T4 field table; gBattleMons empirically pattern-scanned in the model repo).
Money key = SaveBlock2 + 0xAC (address-validated there). pokemon_env/memory_reader.py's
SECURITY_KEY_OFFSET = 0x1F4 is NOT the struct offset — do not copy it.

ui_state byte (§10.2) — 0 none, 1 dialog_open, 2 start_menu, 3 bag, 4 party_menu,
5 summary, 6 mart, 7 pc, 8 battle_menu. Sources, each live-verified 2026-08-20 by driving
a real emulator and reading the screen (tests/test_ledger_sink.py re-drives them):

  1 dialog      BG0 bottom-band window mask — navigator `_dialog_open`'s signal. KNOWN
                LIMITATION: the BG0 tilemap keeps stale window tiles after a box closes,
                so post-close frames can false-read 1 (measured: the tests/states
                no_dialog* fixtures — the harness's documented false-dialogue cases that
                needed the explorer's vision fix — read 1 here).
  2 start_menu  start-menu window id byte 0x0203CD8C != 0xFF (probed 0xFF -> 0x01 -> 0xFF
                across open/close; also correctly 0xFF while the SAVE dialog is up). The
                cursor byte 0x0203760E is NOT an openness signal — it keeps its last value
                after the menu closes — so it is used by the collectors for slot seeking
                only, never here.
  3 bag         gMain.callback2 == 0x081AAD5D (CB2_BagMenuRun — confirmed by driving into
                the bag; also the reason ui.py saw this cb2 around battles: the in-battle
                bag runs the same callback while the screen fades).
  4 party_menu  callback2 == 0x081B01B1 (screen-verified: party list + action popup).
  5 summary     callback2 == 0x081BFAB5 (screen-verified: POKéMON INFO screen).
  6 mart        VERIFIED 2026-08-20 by driving the real Oldale mart (W33 item 3c probes,
                evidence per address in collection/menu_ram.py): Task_ShopMenu
                (0x080DFB89) live in gTasks while BUY/SELL/QUIT is up, and
                gMain.callback2 == 0x080DFD65 (CB2_BuyMenu) for the whole buy screen —
                both carriers emit 6. The SELL bag runs the regular bag cb2 and reads 3.
  7 pc          VERIFIED 2026-08-20 on the Oldale Center item PC: the player-PC session
                task funcs 0x0816AF99 (top menu) / 0x0816B249, 0x0816B369 (item-storage
                submenu) / 0x0816C30D (withdraw-toss list) / 0x0816CB05 (item picked)
                under the overworld cb2 emit 7. The DEPOSIT leg opens the regular bag
                (reads 3) and the which-PC script multichoice reads 1 — documented.
  8 battle_menu NO VERIFIED SOURCE — the battle action menu paints no BG0 window-mask
                footprint and no RAM flag for it was confirmed; never emitted. Battle
                frames carry in_battle=1 with ui_state 0.

`read_ledger` needs io+vram for the window mask, so it accepts only a FULL-blob `GBAState`
(`GBAState.snapshot(env)` live, or the sink's serialized PPU state) — the env-backed
`GBAState.from_env` maps RAM buses only and will fail.
"""

from __future__ import annotations

import numpy as np

from collection.extractors.ram import GBAState
from collection.extractors.ui import window_mask

IWRAM, EWRAM = 0x03000000, 0x02000000
GMAIN = IWRAM + 0x22C0
CB2_ADDR = GMAIN + 0x4                          # gMain.callback2 (== ui.CB2_ADDR)
IN_BATTLE_ADDR = GMAIN + 0x439                  # gMain bitfield; bit 1 = inBattle
SB1_PTR, SB2_PTR = IWRAM + 0x5D8C, IWRAM + 0x5D90
PARTY_COUNT_ADDR, PARTY_ADDR = EWRAM + 0x244E9, EWRAM + 0x244EC
BATTLE_MONS_ADDR, BATTLE_MON_STRIDE = EWRAM + 0x24084, 0x58
SB1_MONEY, SB1_FLAGS, SB1_VARS = 0x490, 0x1270, 0x139C
SB2_KEY = 0xAC
FLAGS_BYTES, VARS_BYTES = 300, 512
BADGE_FLAG0 = 0x867                             # FLAG_BADGE01_GET .. +7 = badge 8

CB2_OVERWORLD = 0x08085E5D
CB2_BAG = 0x081AAD5D
CB2_PARTY_MENU = 0x081B01B1
CB2_SUMMARY = 0x081BFAB5
START_MENU_WINDOW_ID = 0x0203CD8C               # 0xFF while no start-menu window exists

# W33 item 3c mart/pc carriers (live-verified; per-address evidence in menu_ram.py)
GTASKS_ADDR, TASK_SIZE = 0x03005E00, 0x28       # func at +0, isActive at +4
CB2_BUY_MENU = 0x080DFD65                       # the mart buy screen's callback2
TASK_SHOP_MENU = 0x080DFB89                     # BUY/SELL/QUIT handler task func
PC_TASK_FUNCS = frozenset({0x0816AF99, 0x0816B249, 0x0816B369, 0x0816C30D, 0x0816CB05})


def _active_task_funcs(st: GBAState) -> set[int]:
    return {st.u32(GTASKS_ADDR + i * TASK_SIZE) for i in range(16)
            if st.u8(GTASKS_ADDR + i * TASK_SIZE + 4)}

(UI_NONE, UI_DIALOG, UI_START_MENU, UI_BAG, UI_PARTY_MENU,
 UI_SUMMARY, UI_MART, UI_PC, UI_BATTLE_MENU) = range(9)

# field -> (numpy dtype, per-frame shape); the ledger writer's schema (index.json mirrors it)
FIELDS: dict[str, tuple[str, tuple[int, ...]]] = {
    "valid": ("u1", ()),                        # 0 = SaveBlock pointers out of EWRAM (reset/boot)
    "in_battle": ("u1", ()),
    "ui_state": ("u1", ()),
    "money": ("u4", ()),
    "badges": ("u1", ()),                       # bit i = FLAG_BADGE0{i+1}_GET
    "party_count": ("u1", ()),
    "species": ("u2", (3,)),
    "level": ("u1", (3,)),
    "hp": ("u2", (3,)),
    "max_hp": ("u2", (3,)),
    "exp": ("u4", (3,)),
    "moves": ("u2", (3, 4)),
    "pp": ("u1", (3, 4)),
    "personality": ("u4", (3,)),                # fusion key: match against active_personality
    "foe_species": ("u2", ()),
    "foe_level": ("u1", ()),
    "foe_hp": ("u2", ()),
    "foe_max_hp": ("u2", ()),
    "active_species": ("u2", ()),               # gBattleMons[0] = player's active battler
    "active_level": ("u1", ()),
    "active_hp": ("u2", ()),
    "active_max_hp": ("u2", ()),
    "active_exp": ("u4", ()),
    "active_moves": ("u2", (4,)),
    "active_pp": ("u1", (4,)),
    "active_personality": ("u4", ()),
    "flags": ("u1", (FLAGS_BYTES,)),
    "vars": ("u1", (VARS_BYTES,)),
}

# struct Pokemon substruct order by personality % 24 (port of the model repo's
# PokemonCodec — bijective, checksum-validated; G=growth A=attacks E=EVs M=misc)
_ORDERS = ["GAEM", "GAME", "GEAM", "GEMA", "GMAE", "GMEA",
           "AGEM", "AGME", "AEGM", "AEMG", "AMGE", "AMEG",
           "EGAM", "EGMA", "EAGM", "EAMG", "EMGA", "EMAG",
           "MGAE", "MGEA", "MAGE", "MAEG", "MEGA", "MEAG"]


def decrypt_party_mon(raw: bytes) -> dict | None:
    """One 100-byte party `struct Pokemon` -> fields, None when the checksum rejects it
    (empty slot garbage, mid-write). Exact port of PokemonCodec.decrypt, ledger fields only."""
    personality = int.from_bytes(raw[0:4], "little")
    ot_id = int.from_bytes(raw[4:8], "little")
    sec = np.frombuffer(raw[32:80], np.uint32) ^ (personality ^ ot_id)
    if (int(np.frombuffer(sec.tobytes(), np.uint16).sum()) & 0xFFFF) != int.from_bytes(raw[28:30], "little"):
        return None
    order = _ORDERS[personality % 24]
    sub = {order[i]: sec[i * 3:(i + 1) * 3].tobytes() for i in range(4)}
    g, a = sub["G"], sub["A"]
    return {
        "personality": personality,
        "species": int.from_bytes(g[0:2], "little"),
        "experience": int.from_bytes(g[4:8], "little"),
        "moves": [int.from_bytes(a[i:i + 2], "little") for i in range(0, 8, 2)],
        "pp": list(a[8:12]),
        "level": raw[84],
        "hp": int.from_bytes(raw[86:88], "little"),
        "max_hp": int.from_bytes(raw[88:90], "little"),
    }


def _battle_mon(st: GBAState, battler: int) -> dict:
    o = BATTLE_MONS_ADDR + BATTLE_MON_STRIDE * battler
    return {
        "species": st.u16(o),
        "moves": [st.u16(o + 0x0C + 2 * i) for i in range(4)],
        "pp": list(st.bytes(o + 0x24, 4)),
        "hp": st.u16(o + 0x28),
        "level": st.u8(o + 0x2A),
        "max_hp": st.u16(o + 0x2C),
        "experience": st.u32(o + 0x44),
        "personality": st.u32(o + 0x48),
    }


def ui_state(st: GBAState) -> int:
    """§10.2 ui_state byte. Own-screen cb2s first (bag/party/summary keep their cb2 when
    entered from battle too; the buy menu owns 0x080DFD65), then overworld-hosted UIs —
    mart/pc task carriers before the start-menu/dialog signals (their windows paint the
    bottom band and would false-read 1); everything else (battle main, transitions,
    intro) has no verified sub-state and reads 0."""
    cb2 = st.u32(CB2_ADDR)
    if cb2 == CB2_BUY_MENU:
        return UI_MART
    if cb2 == CB2_BAG:
        return UI_BAG
    if cb2 == CB2_PARTY_MENU:
        return UI_PARTY_MENU
    if cb2 == CB2_SUMMARY:
        return UI_SUMMARY
    if cb2 != CB2_OVERWORLD:
        return UI_NONE
    funcs = _active_task_funcs(st)
    if TASK_SHOP_MENU in funcs:
        return UI_MART
    if funcs & PC_TASK_FUNCS:
        return UI_PC
    if st.u8(START_MENU_WINDOW_ID) != 0xFF:
        return UI_START_MENU
    if window_mask(st)[14:20].any():
        return UI_DIALOG
    return UI_NONE


def _zero_row() -> dict:
    row = {name: (np.zeros(shape, dtype=dt) if shape else 0) for name, (dt, shape) in FIELDS.items()}
    return row


def read_ledger(st: GBAState) -> dict:
    """One ledger row (keys/shapes per FIELDS) from a full-blob GBAState."""
    row = _zero_row()
    sb1, sb2 = st.u32(SB1_PTR), st.u32(SB2_PTR)
    if not (EWRAM <= sb1 <= EWRAM + 0x40000 - (SB1_VARS + VARS_BYTES)
            and EWRAM <= sb2 <= EWRAM + 0x40000 - (SB2_KEY + 4)):
        return row                              # valid stays 0: no SaveBlocks (reset/boot)
    row["valid"] = 1
    key = st.u32(sb2 + SB2_KEY)
    row["money"] = (st.u32(sb1 + SB1_MONEY) ^ key) & 0xFFFFFFFF
    flags = np.frombuffer(st.bytes(sb1 + SB1_FLAGS, FLAGS_BYTES), np.uint8)
    row["flags"] = flags
    row["vars"] = np.frombuffer(st.bytes(sb1 + SB1_VARS, VARS_BYTES), np.uint8)
    badges = 0
    for i in range(8):
        f = BADGE_FLAG0 + i
        badges |= ((int(flags[f // 8]) >> (f % 8)) & 1) << i
    row["badges"] = badges
    row["in_battle"] = (st.u8(IN_BATTLE_ADDR) >> 1) & 1
    row["ui_state"] = ui_state(st)
    row["party_count"] = st.u8(PARTY_COUNT_ADDR)
    for slot in range(3):
        if row["party_count"] <= slot:
            break
        d = decrypt_party_mon(st.bytes(PARTY_ADDR + 100 * slot, 100))
        if d is None:
            continue
        row["personality"][slot] = d["personality"]
        row["species"][slot] = d["species"]
        row["level"][slot] = d["level"]
        row["hp"][slot] = d["hp"]
        row["max_hp"][slot] = d["max_hp"]
        row["exp"][slot] = d["experience"]
        row["moves"][slot] = d["moves"]
        row["pp"][slot] = d["pp"]
    if row["in_battle"]:
        mine, foe = _battle_mon(st, 0), _battle_mon(st, 1)
        row["foe_species"] = foe["species"]
        row["foe_level"] = foe["level"]
        row["foe_hp"] = foe["hp"]
        row["foe_max_hp"] = foe["max_hp"]
        row["active_species"] = mine["species"]
        row["active_level"] = mine["level"]
        row["active_hp"] = mine["hp"]
        row["active_max_hp"] = mine["max_hp"]
        row["active_exp"] = mine["experience"]
        row["active_moves"] = np.asarray(mine["moves"], np.uint16)
        row["active_pp"] = np.asarray(mine["pp"], np.uint8)
        row["active_personality"] = mine["personality"]
    return row
