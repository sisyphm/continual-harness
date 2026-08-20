"""MART_BUY expedition block (W33 corpus-v2 §2, item 3c — the mart_pc_item family).

Buys 1-3 affordable items at a real mart counter and optionally sells one back, with
every step RAM-gated and every transaction verified by delta — the _throw_ball
contract: menu transitions gate on the EXPECTED RAM change (gTasks func / cb2 /
cursor), transactions verify money + bag + purchase history, retry once on an
unchanged money delta, then abort with a reason. All addresses live in
collection/menu_ram.py with per-address probe evidence (2026-08-20 session).

Flow (all live-verified on Oldale mart 2,4):
  town -> mart      marts resolved from the manifest: the town warp whose destination
                    map has a gfx==83 (MART_EMPLOYEE) object; door entered by walking
                    to the town-side door tile and pushing UP (map-watched)
  counter talk      approach (1,5), face UP (object-record feedback), A across the
                    counter at (1,4); greeting pages A-advanced until Task_ShopMenu
                    (0x080DFB89) is live in gTasks
  stock             sMartInfo itemList (ROM ptr) x itemCount = the LIVE town stock;
                    prices from the gItems ROM table (0x085839A0, itemId==index scan)
  BUY               cursor 0 -> cb2 0x080DFD65 (buy menu); list row sought by the
                    generic ListMenu task's scroll+row; quantity sought in the
                    how-many task's data[1] with its data[5] item id cross-checked;
                    yes/no gated on sMenu.maxCursorPos == 1 (cursor 0 = YES)
  SELL (optional)   shop cursor 1 -> the bag in sell mode (cb2 bag, location byte 3);
                    the picked slot cross-checked via gSpecialVar_ItemId; driver
                    A-advances, answers the fresh yes/no, verifies money UP + bag DOWN
  QUIT              B until Task_ShopMenu is gone, B-only-close the goodbye box

Walking uses the UNSTICK-FREE walker below: nav.goto's script-lock unstick A-mashes
on the first refused step, and an A while facing a blocking NPC opens a talk — the
Oldale mart promoter (a parked NPC on the exit aisle, and the Center helper at (10,6))
wedged exactly this way, measured 2026-08-20. Refused steps here transient-block the
cell and route around (the bfs_sweep denial idea); building exits push DOWN on the
south-row mat (behavior 0x65 arrow warps fire on the pressed direction, and
goto_warp's UP-first door push steps OFF the mat — measured).

Summary: {towns, stock, purchases [{item_id, qty, price_paid}], sells,
aborts_with_reason, ui_seen, battles_fled, frames}.
"""
from __future__ import annotations

import random

import numpy as np

from collection import menu_ram as mr
from collection import navigator as nav
from collection.extractors.ledger_panel import CB2_ADDR, CB2_BAG, CB2_OVERWORLD, ui_state
from collection.extractors.ram import GBAState
from collection.playthrough.blocks.base import flee_battle
from collection.playthrough.blocks.interaction import _close_dialog, _dialog_open_live, _face

MART_GFX = 83                     # MART_EMPLOYEE object graphics id (manifest-verified)
APPROACH = (1, 5)                 # counter approach tile, identical on all three marts
ITEM_POTION = 13


def _env(runner) -> GBAState:
    return GBAState.from_env(runner.env)


def _poll(runner, pred, frames: int, phase: str, every: int = 10) -> bool:
    for _ in range(max(1, frames // every)):
        if pred(_env(runner)):
            return True
        nav._hold(runner, [], every, phase)
    return pred(_env(runner))


def _press(runner, b: str, wait: int, phase: str) -> None:
    nav._hold(runner, [b], 4, phase)
    nav._hold(runner, [], wait, phase)


# ---------------------------------------------------------------- unstick-free walking

def walk_to(runner, mk, cells, *, phase: str, budget: int = 8000) -> str:
    """BFS walk to any of `cells` [(x, y)] WITHOUT goto's unstick A-mash (which talks
    to whatever NPC refused the step — the measured mart-promoter/Center-helper wedge).
    Refusals transient-block and route around. A DOOR TILE goal auto-fires its warp on
    the step in (measured: the mart door), so a map change reports 'crossed'.
    'arrived' | 'crossed' | 'battle' | 'stuck' | 'budget'."""
    blocked: dict[tuple[int, int], int] = {}
    misses = 0
    start = runner.frame_idx
    src = None
    while runner.frame_idx - start < budget:
        if runner.nav_state().in_battle:
            return "battle"
        t, x, y = nav._state(runner)
        if t is None:
            nav._hold(runner, [], 30, phase)
            continue
        if src is None:
            src = (t.map_group, t.map_num)
        elif (t.map_group, t.map_num) != src:
            nav._hold(runner, [], 60, phase)                  # door/mat fired mid-walk
            return "crossed"
        if (x, y) in cells:
            return "arrived"
        walk = ((t.grid >> 10) & 3) == 0
        goals = np.zeros(t.grid.shape, bool)
        for cx, cy in cells:
            if 0 <= cy + 7 < goals.shape[0] and 0 <= cx + 7 < goals.shape[1]:
                goals[cy + 7, cx + 7] = True
        blocked = {c: f for c, f in blocked.items() if runner.frame_idx - f < 600}
        for cx, cy in blocked:
            walk[cy, cx] = False
        step = nav._bfs_step(walk, (x + 7, y + 7), goals,
                             elev=((t.grid >> 12) & 0xF).astype(np.uint8),
                             beh=mk.behaviors(t))
        if step is None:
            if blocked:
                blocked.clear()                       # dead-ended by our own blocks:
                nav._hold(runner, [], 60, phase)      # let the blocker wander off
                continue
            return "stuck"
        if nav._step(runner, nav.DIRS[step]):
            misses = 0
        else:
            misses += 1
            blocked[(x + 7 + step[0], y + 7 + step[1])] = runner.frame_idx
            if misses >= 10:
                return "stuck"
            nav._hold(runner, [], 10, phase)
    return "budget"


def _push_through(runner, d_first: str, phase: str) -> bool:
    """Push directions (d_first first) until the map changes — arrow mats/doors."""
    t0, _, _ = nav._state(runner)
    if t0 is None:
        return False
    src = (t0.map_group, t0.map_num)
    order = [d_first] + [d for d in ("DOWN", "UP", "LEFT", "RIGHT") if d != d_first]
    for d in order:
        nav._hold(runner, [d], 26, phase)
        nav._hold(runner, [], 40, phase)
        t, _, _ = nav._state(runner)
        if t is not None and (t.map_group, t.map_num) != src:
            nav._hold(runner, [], 60, phase)
            return True
    return False


def exit_building(runner, mk, *, phase: str) -> bool:
    """Leave an indoor map through any non-link-room warp; south-row mats need the
    DOWN push (behavior 0x65 arrow warp — goto_warp's UP-first push steps off it)."""
    t, _, _ = nav._state(runner)
    if t is None:
        return False
    key = f"{t.map_group},{t.map_num}"
    for w in mk.warps.get(key, []):
        if w["dst_map"].startswith("25,"):
            continue
        r = walk_to(runner, mk, [(w["x"], w["y"])], phase=phase, budget=6000)
        if r == "crossed":
            return True
        if r != "arrived":
            continue
        if _push_through(runner, "DOWN", phase):
            return True
    return False


def enter_warp(runner, mk, wx: int, wy: int, *, phase: str, push: str = "UP") -> bool:
    """Walk (unstick-free) onto/next to a door tile and push through it, map-watched.
    Animated doors auto-enter on the walk-in step (walk_to reports 'crossed')."""
    r = walk_to(runner, mk, [(wx, wy)], phase=phase, budget=8000)
    if r == "crossed":
        return True
    if r != "arrived":
        return False
    return _push_through(runner, push, phase)


def _cross_edge(runner, mk, direction: int, *, phase: str) -> bool:
    """Connection fallback: walk (unstick-free) to an edge column/row whose BEYOND-EDGE
    buffer cell is walkable (the connection strip — elsewhere the step is void, measured
    Oldale x=12), then step off, map-watched."""
    t, _, _ = nav._state(runner)
    if t is None:
        return False
    walk = ((t.grid >> 10) & 3) == 0
    w, h = t.map_width, t.map_height
    if direction == 1:                                        # south
        cells = [(x, h - 1) for x in range(w) if walk[h - 1 + 7, x + 7] and walk[h + 7, x + 7]]
        d = "DOWN"
    elif direction == 2:                                      # north
        cells = [(x, 0) for x in range(w) if walk[7, x + 7] and walk[6, x + 7]]
        d = "UP"
    elif direction == 3:                                      # west
        cells = [(0, y) for y in range(h) if walk[y + 7, 7] and walk[y + 7, 6]]
        d = "LEFT"
    else:                                                     # east
        cells = [(w - 1, y) for y in range(h) if walk[y + 7, w - 1 + 7] and walk[y + 7, w + 7]]
        d = "RIGHT"
    if not cells:
        return False
    r = walk_to(runner, mk, cells, phase=phase, budget=12000)
    if r == "crossed":
        return True
    if r != "arrived":
        return False
    src = (t.map_group, t.map_num)
    for _ in range(4):
        nav._step(runner, d)
        t2, _, _ = nav._state(runner)
        if t2 is not None and (t2.map_group, t2.map_num) != src:
            nav._hold(runner, [], 90, phase)
            return True
    return False


def return_to_anchor(runner, mk, ctx, summary: dict, *, phase: str) -> bool:
    """Walk back to the wrapper's anchor with the SAFE machinery, so base's own
    unstick-prone goto return (measured wedging on the mart promoter) is a no-op."""
    anchor = ctx.get("anchor")
    if not anchor:
        return False
    amap, ax, ay = anchor
    if not goto_map_safe(runner, mk, amap, summary, phase=phase):
        return False
    return walk_to(runner, mk, [(ax, ay)], phase=phase, budget=10000) == "arrived"


def goto_map_safe(runner, mk, dst: str, summary: dict, *, phase: str, tries: int = 4) -> bool:
    """goto_map with the wedge mitigations: exit indoor maps first (goto_map from inside
    a building wedged on parked NPCs — measured), retry with a jiggle, and fall back to
    the direct-connection edge walk when the destination is a map-edge neighbour."""
    rng = random.Random(runner.frame_idx)
    for _ in range(tries):
        if runner.nav_state().in_battle:
            flee_battle(runner)
            summary["battles_fled"] += 1
            continue
        t, _, _ = nav._state(runner)
        if t is None:
            nav._hold(runner, [], 30, phase)
            continue
        key = f"{t.map_group},{t.map_num}"
        if key == dst:
            return True
        if not mk.connections.get(key) and key != dst:        # indoors: walk out first
            exit_building(runner, mk, phase=phase)
            continue
        conn = next((c for c in mk.connections.get(key, []) if c["dst_map"] == dst), None)
        if conn is not None and _cross_edge(runner, mk, conn["direction"], phase=phase):
            return True
        if nav.goto_map(runner, mk, dst, hop_budget=9000) == "arrived":
            return True
        _close_dialog(runner, phase)                          # a wedge often leaves a box
        for d in rng.sample(list(nav.DIRS.values()), 4):      # dislodge a first-step block
            if nav._step(runner, d):
                break
        nav._hold(runner, [], 60, phase)
    t, _, _ = nav._state(runner)
    return t is not None and f"{t.map_group},{t.map_num}" == dst


# ------------------------------------------------------------------------- mart block

class MartBuy:
    name = "mart_buy"
    phase = "mart_pc_item"

    def __init__(self, towns: list[str], frames: int = 90000, want: list | None = None,
                 max_items: int = 3, sell: bool = True, seed: int = 0):
        self.towns = list(towns)
        self.frames = frames
        self.want = [tuple(w) for w in want] if want else None   # [(item_id, qty), ...]
        self.max_items = max_items
        self.sell = sell
        self.seed = seed

    # ------------------------------------------------------------------ helpers

    def _abort(self, summary, where, reason):
        summary["aborts_with_reason"].append(dict(where=where, reason=reason))

    def _ui_sample(self, runner, summary):
        u = ui_state(GBAState.snapshot(runner.env))
        if not summary["ui_seen"] or summary["ui_seen"][-1] != u:
            summary["ui_seen"].append(u)

    def _find_mart(self, mk, town: str) -> tuple[str, dict] | None:
        for w in mk.warps.get(town, []):
            dst = w["dst_map"]
            if any(o.get("gfx") == MART_GFX for o in mk.objects.get(dst, [])):
                return dst, w
        return None

    def _open_shop_menu(self, runner) -> bool:
        """Face the clerk across the counter and A until Task_ShopMenu is live."""
        p = self.phase
        for round_ in range(2):
            _close_dialog(runner, p)
            if not _face(runner, "UP", p):
                continue
            runner.perform_action("A", record_end_state=False,
                                  metadata={"block": self.name})
            for _ in range(10):
                if mr.find_task(_env(runner), mr.TASK_SHOP_MENU) is not None:
                    nav._hold(runner, [], 30, p)
                    return True
                _press(runner, "A", 30, p)
        return mr.find_task(_env(runner), mr.TASK_SHOP_MENU) is not None

    def _menu_cursor(self, st) -> int:
        return mr.menu_cursor(st)[0]

    def _enter_buy_menu(self, runner) -> bool:
        p = self.phase
        if not mr.seek_cursor(runner, self._menu_cursor, 0, phase=p):
            return False
        _press(runner, "A", 40, p)
        if not _poll(runner, lambda st: st.u32(CB2_ADDR) == mr.CB2_BUY_MENU, 400, p):
            return False
        nav._hold(runner, [], 90, p)                          # list fade-in settle
        return True

    def _back_to_item_list(self, runner) -> bool:
        """Post-purchase: A clears 'Here you go!'; a stray A can reopen the qty spinner
        (B-guarded); done when Task_BuyMenu is live again."""
        p = self.phase
        for _ in range(10):
            st = _env(runner)
            if mr.find_task(st, mr.TASK_BUY_MENU) is not None:
                return True
            if mr.find_task(st, mr.TASK_BUY_QTY) is not None:
                _press(runner, "B", 30, p)
                continue
            _press(runner, "A", 30, p)
        return mr.find_task(_env(runner), mr.TASK_BUY_MENU) is not None

    def _buy_one(self, runner, mk, row: int, item_id: int, qty: int, summary) -> bool:
        """One gated purchase from the OPEN buy menu; True = delta-verified."""
        p = self.phase
        price = mr.item_price(mk.rom, item_id)
        for attempt in range(2):                              # the retry contract
            if not mr.seek_cursor(runner, mr.list_cursor, row, phase=p):
                self._abort(summary, f"buy {item_id}", "list cursor seek failed")
                return False
            _press(runner, "A", 40, p)
            if not _poll(runner, lambda st: mr.find_task(st, mr.TASK_BUY_QTY) is not None,
                         300, p):
                self._abort(summary, f"buy {item_id}", "how-many task never came up")
                return False
            qt = mr.find_task(_env(runner), mr.TASK_BUY_QTY)
            got = mr.task_data(_env(runner), qt)[5]
            if got != item_id:                                # cursor drift: B out, retry
                _press(runner, "B", 30, p)
                self._back_to_item_list(runner)
                if attempt == 0:
                    continue
                self._abort(summary, f"buy {item_id}",
                            f"how-many spinner shows item {got}, wanted {item_id}")
                return False
            for _ in range(qty + 2):                          # qty by task-data readback
                if mr.task_data(_env(runner), qt)[1] >= qty:
                    break
                _press(runner, "UP", 20, p)
            m0 = mr.money(_env(runner))
            bag0 = dict(mr.bag_items(_env(runner)))
            hist0 = dict(mr.mart_history(_env(runner)))
            _press(runner, "A", 30, p)                        # -> 'that'll be X' + yes/no
            # the yes/no gate is the menu-helpers TASK func — sMenu.max stays stale at 1
            # after the first purchase's confirm, so only the task signals the prompt
            if not _poll(runner, lambda st: mr.find_task(st, mr.TASK_YESNO) is not None,
                         400, p):
                self._abort(summary, f"buy {item_id}", "yes/no task never came up")
                return False
            _press(runner, "A", 30, p)                        # YES (cursor 0 default)
            if _poll(runner, lambda st: mr.money(st) != m0, 400, p):
                st = _env(runner)
                paid = m0 - mr.money(st)
                bag_delta = dict(mr.bag_items(st)).get(item_id, 0) - bag0.get(item_id, 0)
                hist_delta = dict(mr.mart_history(st)).get(item_id, 0) - hist0.get(item_id, 0)
                ok = paid == price * qty and bag_delta == qty and hist_delta == qty
                summary["purchases"].append(dict(item_id=item_id, qty=qty, price_paid=paid,
                                                 rom_price=price, bag_delta=bag_delta,
                                                 history_delta=hist_delta, verified=ok))
                self._back_to_item_list(runner)
                return ok
            _press(runner, "B", 30, p)                        # unchanged money: reset and
            self._back_to_item_list(runner)                   # retry once, then abort
        self._abort(summary, f"buy {item_id}", "money unchanged after retry")
        return False

    def _sell_one(self, runner, mk, summary) -> bool:
        """Optional SELL from the shop menu; skip-with-note on any miss."""
        p = self.phase
        bag = mr.bag_items(_env(runner))
        target = next(((iid, q) for iid, q in bag if mr.item_price(mk.rom, iid) > 0), None)
        if target is None:
            self._abort(summary, "sell", "skip: no sellable item in bag")
            return False
        iid, _q = target
        price = mr.item_price(mk.rom, iid)
        if not mr.seek_cursor(runner, self._menu_cursor, 1, phase=p):   # SELL slot
            self._abort(summary, "sell", "skip: shop cursor seek failed")
            return False
        _press(runner, "A", 40, p)
        if not _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_BAG, 400, p):
            self._abort(summary, "sell", "skip: sell bag never opened")
            return False
        nav._hold(runner, [], 60, p)
        for _ in range(5):                                    # ITEMS pocket
            if mr.bag_pocket(_env(runner)) == 0:
                break
            _press(runner, "LEFT", 25, p)
        slot = next((i for i, (bid, _) in enumerate(mr.bag_items(_env(runner))) if bid == iid), None)
        if slot is None or not mr.seek_cursor(runner, mr.list_cursor, slot, phase=p):
            self._abort(summary, "sell", "skip: bag slot seek failed")
            self._leave_bag_to_shop(runner)
            return False
        m0 = mr.money(_env(runner))
        bag0 = dict(mr.bag_items(_env(runner)))
        _press(runner, "A", 40, p)
        st = _env(runner)
        if st.u16(mr.SPECIAL_VAR_ITEM_ID) != iid:             # picked the wrong slot
            self._abort(summary, "sell", f"skip: picked item {st.u16(mr.SPECIAL_VAR_ITEM_ID)}")
            self._leave_bag_to_shop(runner)
            return False
        for _ in range(24):                                   # the sell driver: qty spinner
            st = _env(runner)                                 # -> 'I can pay' -> yes/no;
            if mr.money(st) > m0:                             # A advances every stage and
                break                                         # answers YES (cursor 0)
            _press(runner, "A", 40, p)
        st = _env(runner)
        gained = mr.money(st) - m0
        sold = bag0.get(iid, 0) - dict(mr.bag_items(st)).get(iid, 0)
        if gained <= 0:
            self._abort(summary, "sell", "skip: money never increased")
            self._leave_bag_to_shop(runner)
            return False
        summary["sells"].append(dict(item_id=iid, qty=sold, money_gained=gained,
                                     rom_price=price, verified=gained == (price // 2) * sold))
        self._leave_bag_to_shop(runner)
        return True

    def _leave_bag_to_shop(self, runner) -> bool:
        p = self.phase
        for _ in range(10):
            st = _env(runner)
            if st.u32(CB2_ADDR) == CB2_OVERWORLD and mr.find_task(st, mr.TASK_SHOP_MENU) is not None:
                return True
            _press(runner, "B", 40, p)
        return False

    def _quit_shop(self, runner) -> bool:
        p = self.phase
        for _ in range(10):
            if mr.find_task(_env(runner), mr.TASK_SHOP_MENU) is None:
                break
            _press(runner, "B", 30, p)
        _close_dialog(runner, p)                              # 'Please come again!'
        return mr.find_task(_env(runner), mr.TASK_SHOP_MENU) is None \
            and not _dialog_open_live(runner)

    # -------------------------------------------------------------------- per town

    def _town(self, runner, mk, town: str, summary) -> None:
        p = self.phase
        if not goto_map_safe(runner, mk, town, summary, phase=p):
            self._abort(summary, town, "town unreached")
            return
        found = self._find_mart(mk, town)
        if found is None:
            self._abort(summary, town, "no mart (gfx==83 clerk) behind any town warp")
            return
        mart_key, door = found
        if not enter_warp(runner, mk, door["x"], door["y"], phase=p):
            self._abort(summary, town, "mart door never crossed")
            return
        if walk_to(runner, mk, [APPROACH], phase=p, budget=8000) != "arrived":
            self._abort(summary, town, "counter approach tile unreached")
            exit_building(runner, mk, phase=p)
            return
        if not self._open_shop_menu(runner):
            self._abort(summary, town, "shop menu (Task_ShopMenu) never opened")
            exit_building(runner, mk, phase=p)
            return
        stock = mr.mart_stock(_env(runner), mk.rom)
        summary["stock"][town] = stock
        self._ui_sample(runner, summary)                      # shop menu: expect UI_MART
        plan = self.want or [(iid, 1) for iid in stock[:self.max_items]]
        plan = [(iid, q) for iid, q in plan if iid in stock]
        if not plan:
            self._abort(summary, town, "no wanted item in stock")
        elif self._enter_buy_menu(runner):
            self._ui_sample(runner, summary)                  # buy menu: expect UI_MART
            for iid, qty in plan:
                if mr.item_price(mk.rom, iid) * qty > mr.money(_env(runner)):
                    self._abort(summary, f"buy {iid}", "not affordable")
                    continue
                self._buy_one(runner, mk, stock.index(iid), iid, qty, summary)
            for _ in range(10):                               # buy menu -> shop menu
                st = _env(runner)
                if st.u32(CB2_ADDR) == CB2_OVERWORLD and mr.find_task(st, mr.TASK_SHOP_MENU) is not None:
                    break
                _press(runner, "B", 40, p)
        else:
            self._abort(summary, town, "buy menu (cb2 0x080DFD65) never opened")
        if self.sell and mr.find_task(_env(runner), mr.TASK_SHOP_MENU) is not None:
            self._sell_one(runner, mk, summary)
        if not self._quit_shop(runner):
            self._abort(summary, town, "shop quit incomplete")
        if not exit_building(runner, mk, phase=p) and not exit_building(runner, mk, phase=p):
            self._abort(summary, town, "mart exit incomplete")

    # ----------------------------------------------------------------------- entry

    def run(self, runner, mk, ctx) -> dict:
        random.Random(self.seed)
        summary = dict(towns=[], stock={}, purchases=[], sells=[],
                       aborts_with_reason=[], ui_seen=[], battles_fled=0, frames=0,
                       money_before=mr.money(_env(runner)), money_after=0)
        f0 = runner.frame_idx
        deadline = f0 + self.frames
        for town in self.towns:
            if runner.frame_idx >= deadline:
                self._abort(summary, town, "block frame budget expired")
                continue
            summary["towns"].append(town)
            self._town(runner, mk, town, summary)
        return_to_anchor(runner, mk, ctx, summary, phase=self.phase)
        summary["money_after"] = mr.money(_env(runner))       # closes the exact accounting:
        summary["frames"] = runner.frame_idx - f0             # before - after == purchases - sells
        return summary
