"""Menu/transaction RAM seam for the mart_pc_item block family (W33 corpus-v2 §2, item 3c).

Every address here was DWARF-suggested and then LIVE-VERIFIED on the real emulator
(2026-08-20 probe session, Oldale mart 2,4 / Oldale Center 2,2 from the OLDALE_TOWN
storyline state) before being trusted; each constant's comment states the probe evidence.
All reads are RAM-bus only (ewram/iwram), so both `GBAState.from_env` (cheap, per-poll)
and full snapshots work.

The transaction contract these accessors serve (the _throw_ball pattern): every menu
transition is gated on an EXPECTED RAM change (gTasks func / cb2 / cursor), every
transaction is verified by delta (money + bag + purchase history), retried once on an
unchanged delta, then aborted with a reason.
"""
from __future__ import annotations

from typing import Callable

from collection.extractors.ledger_panel import (
    CB2_ADDR, CB2_BAG, CB2_BUY_MENU, CB2_OVERWORLD, CB2_PARTY_MENU, GTASKS_ADDR,
    PC_TASK_FUNCS, SB1_MONEY, SB1_PTR, SB2_KEY, SB2_PTR, START_MENU_WINDOW_ID,
    TASK_SHOP_MENU, TASK_SIZE,
)
from collection.extractors.ram import GBAState

# --- gTasks (constants shared with the ledger's ui_state live in ledger_panel and are
# IMPORTED, never redefined — the menus.py precedent) --------------------------------
# _verified: Task_ShopMenu appeared at +0 of a fresh entry after talking to the Oldale
# clerk, isActive byte +4 gated presence, and data diffs at +8 tracked the cursors below.
GTASKS = GTASKS_ADDR                            # func +0, isActive +4, s16 data[16] +8

# Shop task funcs (Thumb addresses as stored in gTasks; all seen live in the buy flow):
# TASK_SHOP_MENU (imported) _verified: present exactly while BUY/SELL/QUIT is up
TASK_BUY_MENU = 0x080E0AC9    # _verified: present while the buy item list is up
TASK_BUY_QTY = 0x080E0D89     # _verified: present in the how-many spinner; data[1] = qty
                              # (UP: 1->2), data[5] = item id (13/14 matched the pick)
LIST_MENU_TASK = 0x081AE459   # generic ListMenu task. _verified: data[12] = scroll,
                              # data[13] = row (DOWN diffs in buy list, sell bag and the
                              # PC withdraw list all moved exactly these)
# menu_helpers task funcs shared by shop buy/sell confirms (STATE-PROOF yes/no gates —
# sMenu.maxCursorPos goes stale at 1 after the FIRST confirm, so from the second
# purchase on only these funcs signal the prompt).
# _verified: 0x08121F3D present while 'that'll be X'/'I can pay X' printed, then
# 0x08121FDD exactly while the yes/no awaited input (probe traces, both flows).
TASK_MSG_PRINT = 0x08121F3D
TASK_YESNO = 0x08121FDD
# PC_TASK_FUNCS (imported): the player-PC / item-storage session task funcs (all under
# the overworld cb2). _verified: staged dumps through the Center PC flow — top menu
# 0x0816AF99 (sMenu max 2), storage submenu 0x0816B249 / 0x0816B369 (max 3),
# withdraw/toss list 0x0816C30D (+ LIST_MENU_TASK + populated page info), 0x0816CB05
# item-picked stage.
# CB2_BUY_MENU (imported) _verified: gMain.callback2 for the whole buy-menu screen
# (list, qty, confirm, purchase — read 0x080DFD65 throughout).

# --- shop statics -------------------------------------------------------------------
SMART_INFO = 0x02039F60       # _verified: +8 itemList -> 0x081FC260 (ROM), +12 count=4;
                              # the pointed list read [13,14,18,17] = the live Oldale stock
MART_HISTORY = 0x02039F80     # _verified: [(14,2)] after buying Antidote x2 (id,qty u16 pairs)
MART_HISTORY_SLOTS = 3

# --- menus --------------------------------------------------------------------------
SMENU = 0x0203CD90            # _verified: +2 cursorPos / +4 maxCursorPos — BUY/SELL/QUIT
                              # max 2, yes/no max 1 with cursor 0 = YES, which-PC max 2,
                              # storage submenu max 3, overworld item context max 3,
                              # battle item context max 1. STALE after close: never an
                              # openness signal, only a within-flow transition signal.
BAG_POSITION = 0x0203CE58     # _verified: +4 location (3 while selling, 1 in-battle),
                              # +5 pocket (0=ITEMS; RIGHT 0->1, LEFT back)
SPECIAL_VAR_ITEM_ID = 0x0203CE7C   # _verified: latched 14 on sell-select, 13 on use-select
PC_ITEM_PAGE = 0x0203BCB8     # _verified: +0 u16 cursorPos (DOWN 0->1), +2 u16 itemsAbove,
                              # +4 u8 count (rows incl. CANCEL: 2 with one stored item,
                              # 1 after it was withdrawn)

# --- SaveBlock1 item pockets --------------------------------------------------------
SB1_PC_ITEMS = 0x498          # _verified: read the starting PC Potion (13,1) plain, and the
                              # deposit/withdraw SB1 byte-diffs moved exactly these bytes
SB1_BAG_ITEMS = 0x560         # _verified: buy/sell/deposit SB1 diffs; qty is XORed with the
                              # low 16 bits of the SB2+0xAC key (the ledger's money route)
PC_ITEM_SLOTS, BAG_ITEM_SLOTS = 50, 30

# --- ROM item table -----------------------------------------------------------------
GITEMS_ROM = 0x085839A0       # _verified: found by scanning the ROM for 44-byte records
ITEM_STRIDE = 44              # with u16 itemId == index at +14 (indexes 0..40 checked);
ITEM_PRICE_OFF = 16           # prices matched play: Poké Ball 200, Potion 300 (charged
MAX_ITEM_ID = 376             # 600 for x2), Antidote 100 (200 for x2, sold back at 50)


def active_tasks(st: GBAState) -> list[tuple[int, int]]:
    """[(task_idx, func_ptr)] for active gTasks entries."""
    return [(i, st.u32(GTASKS + i * TASK_SIZE)) for i in range(16)
            if st.u8(GTASKS + i * TASK_SIZE + 4)]


def find_task(st: GBAState, func: int) -> int | None:
    return next((i for i, f in active_tasks(st) if f == func), None)


def task_data(st: GBAState, idx: int) -> list[int]:
    return [st.u16(GTASKS + idx * TASK_SIZE + 8 + 2 * k) for k in range(16)]


def menu_cursor(st: GBAState) -> tuple[int, int]:
    """(cursorPos, maxCursorPos) of sMenu — stale-persistent, gate on transitions only."""
    return st.u8(SMENU + 2), st.u8(SMENU + 4)


def list_cursor(st: GBAState) -> int | None:
    """Absolute row (scroll + row) of the active generic list menu, or None."""
    i = find_task(st, LIST_MENU_TASK)
    if i is None:
        return None
    d = task_data(st, i)
    return d[12] + d[13]


def bag_pocket(st: GBAState) -> int:
    return st.u8(BAG_POSITION + 5)


def money(st: GBAState) -> int:
    """Absolute money — the ledger's SB2+0xAC route (ledger_panel constants, not re-derived)."""
    return (st.u32(st.u32(SB1_PTR) + SB1_MONEY) ^ st.u32(st.u32(SB2_PTR) + SB2_KEY)) & 0xFFFFFFFF


def bag_items(st: GBAState) -> list[tuple[int, int]]:
    """ITEMS-pocket slots [(item_id, qty)], qty decrypted with the SB2 key's low half."""
    sb1, key = st.u32(SB1_PTR), st.u32(st.u32(SB2_PTR) + SB2_KEY) & 0xFFFF
    out = []
    for i in range(BAG_ITEM_SLOTS):
        iid = st.u16(sb1 + SB1_BAG_ITEMS + 4 * i)
        if iid:
            out.append((iid, st.u16(sb1 + SB1_BAG_ITEMS + 4 * i + 2) ^ key))
    return out


def pc_items(st: GBAState) -> list[tuple[int, int]]:
    """Item-PC storage slots [(item_id, qty)] — quantities are stored PLAIN."""
    sb1 = st.u32(SB1_PTR)
    return [(st.u16(sb1 + SB1_PC_ITEMS + 4 * i), st.u16(sb1 + SB1_PC_ITEMS + 4 * i + 2))
            for i in range(PC_ITEM_SLOTS) if st.u16(sb1 + SB1_PC_ITEMS + 4 * i)]


def mart_stock(st: GBAState, rom: bytes) -> list[int]:
    """Live town stock: sMartInfo.itemList (ROM ptr) x itemCount, [] when implausible."""
    ptr, count = st.u32(SMART_INFO + 8), st.u16(SMART_INFO + 12)
    if not (0x08000000 <= ptr < 0x08000000 + len(rom)) or not (0 < count <= 32):
        return []
    off = ptr - 0x08000000
    return [int.from_bytes(rom[off + 2 * i:off + 2 * i + 2], "little") for i in range(count)]


def mart_history(st: GBAState) -> list[tuple[int, int]]:
    return [(st.u16(MART_HISTORY + 4 * i), st.u16(MART_HISTORY + 4 * i + 2))
            for i in range(MART_HISTORY_SLOTS) if st.u16(MART_HISTORY + 4 * i)]


def item_price(rom: bytes, item_id: int) -> int:
    if not (0 < item_id <= MAX_ITEM_ID):
        return 0
    off = GITEMS_ROM - 0x08000000 + ITEM_STRIDE * item_id + ITEM_PRICE_OFF
    return int.from_bytes(rom[off:off + 2], "little")


def menu_kind(st: GBAState) -> str:
    """Coarse menu classification from gTasks funcs + cb2 (all live-verified carriers);
    the start menu additionally uses the ledger's verified window-id byte."""
    cb2 = st.u32(CB2_ADDR)
    if cb2 == CB2_BUY_MENU:
        return "buy_qty" if find_task(st, TASK_BUY_QTY) is not None else "buy_menu"
    if cb2 == CB2_BAG:
        return "bag"
    if cb2 == CB2_PARTY_MENU:
        return "party_menu"
    if cb2 != CB2_OVERWORLD:
        return "other"
    funcs = {f for _, f in active_tasks(st)}
    if TASK_SHOP_MENU in funcs:
        return "shop_menu"
    if funcs & PC_TASK_FUNCS:
        return "pc_storage"
    if st.u8(START_MENU_WINDOW_ID) != 0xFF:
        return "start_menu"
    return "overworld"


def seek_cursor(runner, read_fn: Callable[[GBAState], int | None], target: int,
                button: str | None = None, cap: int = 12, phase: str = "menu") -> bool:
    """Move a menu cursor to `target` by RAM feedback (the wrap-proof _seek_start_slot
    pattern). `read_fn(st)` reads the cursor from a fresh env-backed state; `button`
    fixes the press, or None auto-picks DOWN/UP from the comparison."""
    from collection import navigator as nav
    for _ in range(cap):
        cur = read_fn(GBAState.from_env(runner.env))
        if cur == target:
            return True
        if cur is None:
            nav._hold(runner, [], 12, phase)
            continue
        b = button or ("DOWN" if cur < target else "UP")
        nav._hold(runner, [b], 4, phase)
        nav._hold(runner, [], 18, phase)
    return read_fn(GBAState.from_env(runner.env)) == target
