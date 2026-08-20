"""PC_ACCESS expedition block (W33 corpus-v2 §2, item 3c — the mart_pc_item family).

Item-PC round trip at a Pokémon Center: boot the PC at (10,2) facing UP (the console
metatile behavior 0x83 at (10,1) — identical on all three Centers), enter the player's
PC item storage, and move one item BOTH ways with SaveBlock deltas verifying each leg
(bag pocket at SB1+0x560, key-encrypted qty; PC storage at SB1+0x498, plain qty —
both pinned by byte-diffs across live transactions).

Leg order adapts to reality instead of forcing it: storyline states carry an EMPTY bag
and the starting Potion in the PC, so the round trip runs withdraw-then-deposit there
(deposit-first when the bag already has an item); both legs always run and verify.

Gates (2026-08-20 probe evidence, constants in menu_ram):
  which-PC menu    A-driven until a dialog + sMenu max==2, then a MOVEMENT-PROOF
                   cursor seek (0 then 1) — sMenu is stale-persistent, so only a
                   cursor that provably moves counts as a live menu
  top menu         gTasks func 0x0816AF99 (ITEM STORAGE = cursor 0)
  storage submenu  gTasks func 0x0816B249/0x0816B369 (WITHDRAW 0 / DEPOSIT 1)
  withdraw list    generic ListMenu task + gPlayerPCItemPageInfo count > 0; row =
                   cursorPos+itemsAbove; qty-1 stores withdraw ON the select A, and
                   the PC-side slot clears on the next input (compaction — measured)
  deposit          the regular bag opens (cb2 CB2_BAG); potion picked via the list
                   task + gSpecialVar_ItemId latch; deltas: bag down AND pc up
  exit             B-ladder until overworld cb2, no PC-session tasks, no dialog
                   (measured 7 B presses from the withdraw list)

Summary: {deposits, withdrawals, order, ui_seen, aborts_with_reason, frames}.
"""
from __future__ import annotations

from collection import menu_ram as mr
from collection import navigator as nav
from collection.extractors.ledger_panel import CB2_ADDR, CB2_BAG, CB2_OVERWORLD, ui_state
from collection.extractors.ram import GBAState
from collection.playthrough.blocks.interaction import _close_dialog, _dialog_open_live, _face
from collection.playthrough.blocks.mart import (
    _env, _poll, _press, goto_map_safe, return_to_anchor, walk_to,
)

PC_TILE_APPROACH = (10, 2)                       # face UP at the 0x83 console at (10,1)
SUBMENU_TASKS = (0x0816B249, 0x0816B369)         # storage submenu handler funcs (probed)
TOP_MENU_TASK = 0x0816AF99


def _pc_task(st: GBAState, funcs) -> bool:
    return any(f in funcs for _, f in mr.active_tasks(st))


class PcAccess:
    name = "pc_access"
    phase = "mart_pc_item"

    def __init__(self, center: str = "2,2", frames: int = 60000, seed: int = 0):
        self.center = center                     # Center 1F map key (2,2 / 8,4 / 11,5)
        self.frames = frames
        self.seed = seed

    def _abort(self, summary, where, reason):
        summary["aborts_with_reason"].append(dict(where=where, reason=reason))

    def _ui_sample(self, runner, summary):
        u = ui_state(GBAState.snapshot(runner.env))
        if not summary["ui_seen"] or summary["ui_seen"][-1] != u:
            summary["ui_seen"].append(u)

    # ------------------------------------------------------------------ menu drive

    def _boot_to_storage_submenu(self, runner, summary) -> bool:
        p = self.phase
        _close_dialog(runner, p)
        if not _face(runner, "UP", p):
            self._abort(summary, "boot", "could not face the console")
            return False
        runner.perform_action("A", record_end_state=False, metadata={"block": self.name})
        # which-PC menu: dialog + max==2, then the movement-proof seek (stale sMenu
        # bytes can read 2 from an old menu — only a cursor that moves counts)
        live = False
        for _ in range(12):
            st = _env(runner)
            if _dialog_open_live(runner) and mr.menu_cursor(st)[1] == 2:
                if mr.seek_cursor(runner, lambda s: mr.menu_cursor(s)[0], 0, phase=p) and \
                        mr.seek_cursor(runner, lambda s: mr.menu_cursor(s)[0], 1, phase=p):
                    live = True
                    break
            _press(runner, "A", 40, p)
        if not live:
            self._abort(summary, "boot", "which-PC menu never became live")
            return False
        _press(runner, "A", 60, p)                            # PLAYER's PC (cursor 1)
        ok = False
        for _ in range(4):                                    # 'Accessed ...' prints, then
            if _poll(runner, lambda st: _pc_task(st, (TOP_MENU_TASK,)), 300, p):
                ok = True                                     # the top-menu task spawns
                break
            _press(runner, "A", 30, p)
        if not ok:
            self._abort(summary, "boot", "player-PC top menu task never spawned")
            return False
        if not mr.seek_cursor(runner, lambda s: mr.menu_cursor(s)[0], 0, phase=p):
            self._abort(summary, "boot", "ITEM STORAGE cursor seek failed")
            return False
        _press(runner, "A", 60, p)                            # ITEM STORAGE
        ok = False
        for _ in range(4):
            if _poll(runner, lambda st: _pc_task(st, SUBMENU_TASKS), 300, p):
                ok = True
                break
            _press(runner, "A", 30, p)
        if not ok:
            self._abort(summary, "boot", "item-storage submenu task never spawned")
        return ok

    def _back_to_submenu(self, runner) -> bool:
        p = self.phase
        for _ in range(8):
            st = _env(runner)
            if st.u32(CB2_ADDR) == CB2_OVERWORLD and _pc_task(st, SUBMENU_TASKS) \
                    and mr.find_task(st, mr.LIST_MENU_TASK) is None:
                return True
            _press(runner, "B", 45, p)
        return False

    # ----------------------------------------------------------------------- legs

    def _withdraw(self, runner, summary) -> bool:
        p = self.phase
        pc0 = mr.pc_items(_env(runner))
        if not pc0:
            self._abort(summary, "withdraw", "PC storage is empty")
            return False
        iid, _q = pc0[0]
        if not mr.seek_cursor(runner, lambda s: mr.menu_cursor(s)[0], 0, phase=p):
            self._abort(summary, "withdraw", "submenu cursor seek failed")
            return False
        _press(runner, "A", 60, p)
        if not _poll(runner, lambda st: mr.find_task(st, mr.LIST_MENU_TASK) is not None
                     and st.u8(mr.PC_ITEM_PAGE + 4) > 0, 400, p):
            self._abort(summary, "withdraw", "withdraw list never opened")
            return False

        def row(st):                                          # absolute storage row
            return st.u16(mr.PC_ITEM_PAGE) + st.u16(mr.PC_ITEM_PAGE + 2)
        if not mr.seek_cursor(runner, row, 0, phase=p):       # slot 0 = the target item
            self._abort(summary, "withdraw", "storage row seek failed")
            self._back_to_submenu(runner)
            return False
        bag_before = dict(mr.bag_items(_env(runner))).get(iid, 0)
        pc_before = dict(pc0).get(iid, 0)
        _press(runner, "A", 60, p)                            # qty-1 withdraws on select
        got = False
        for _ in range(10):                                   # (spinner + A when qty > 1)
            if dict(mr.bag_items(_env(runner))).get(iid, 0) > bag_before:
                got = True
                break
            _press(runner, "A", 40, p)
        if not got:
            self._abort(summary, "withdraw", "bag never gained the item")
            self._back_to_submenu(runner)
            return False
        for _ in range(6):                                    # PC slot clears on the NEXT
            if dict(mr.pc_items(_env(runner))).get(iid, 0) < pc_before:    # input (compaction)
                break
            _press(runner, "A", 40, p)
        bag_after = dict(mr.bag_items(_env(runner))).get(iid, 0)
        pc_after = dict(mr.pc_items(_env(runner))).get(iid, 0)
        summary["withdrawals"].append(dict(
            item_id=iid, qty=bag_after - bag_before,
            bag_delta=bag_after - bag_before, pc_delta=pc_after - pc_before,
            verified=bag_after - bag_before == 1 and pc_after - pc_before == -1))
        self._back_to_submenu(runner)
        return True

    def _deposit(self, runner, summary) -> bool:
        p = self.phase
        bag0 = mr.bag_items(_env(runner))
        if not bag0:
            self._abort(summary, "deposit", "bag is empty")
            return False
        iid, _q = bag0[0]
        if not mr.seek_cursor(runner, lambda s: mr.menu_cursor(s)[0], 1, phase=p):
            self._abort(summary, "deposit", "submenu cursor seek failed")
            return False
        _press(runner, "A", 60, p)
        if not _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_BAG, 500, p):
            self._abort(summary, "deposit", "deposit bag never opened")
            return False
        nav._hold(runner, [], 60, p)
        for _ in range(5):
            if mr.bag_pocket(_env(runner)) == 0:
                break
            _press(runner, "LEFT", 25, p)
        slot = next((i for i, (bid, _) in enumerate(mr.bag_items(_env(runner))) if bid == iid), None)
        if slot is None or not mr.seek_cursor(runner, mr.list_cursor, slot, phase=p):
            self._abort(summary, "deposit", "bag slot seek failed")
            self._back_to_submenu(runner)
            return False
        bag_before = dict(mr.bag_items(_env(runner))).get(iid, 0)
        pc_before = dict(mr.pc_items(_env(runner))).get(iid, 0)
        _press(runner, "A", 50, p)                            # select -> qty spinner
        _press(runner, "A", 50, p)                            # confirm qty 1
        got = False
        for _ in range(10):
            st = _env(runner)
            if dict(mr.bag_items(st)).get(iid, 0) < bag_before \
                    and dict(mr.pc_items(st)).get(iid, 0) > pc_before:
                got = True
                break
            _press(runner, "A", 40, p)
        bag_after = dict(mr.bag_items(_env(runner))).get(iid, 0)
        pc_after = dict(mr.pc_items(_env(runner))).get(iid, 0)
        summary["deposits"].append(dict(
            item_id=iid, qty=bag_before - bag_after,
            bag_delta=bag_after - bag_before, pc_delta=pc_after - pc_before,
            verified=bag_after - bag_before == -1 and pc_after - pc_before == 1))
        if not got:
            self._abort(summary, "deposit", "deltas never landed")
        self._back_to_submenu(runner)
        return got

    def _exit_pc(self, runner) -> bool:
        p = self.phase
        for _ in range(14):
            st = _env(runner)
            if st.u32(CB2_ADDR) == CB2_OVERWORLD \
                    and not _pc_task(st, mr.PC_TASK_FUNCS) \
                    and mr.find_task(st, mr.LIST_MENU_TASK) is None \
                    and _dialog_open_live(runner) is not True:
                return True
            _press(runner, "B", 50, p)
        return False

    # ---------------------------------------------------------------------- entry

    def run(self, runner, mk, ctx) -> dict:
        summary = dict(deposits=[], withdrawals=[], order=None, ui_seen=[],
                       aborts_with_reason=[], battles_fled=0, frames=0)
        f0 = runner.frame_idx
        p = self.phase
        if not goto_map_safe(runner, mk, self.center, summary, phase=p):
            self._abort(summary, "travel", f"center {self.center} unreached")
            summary["frames"] = runner.frame_idx - f0
            return summary
        if walk_to(runner, mk, [PC_TILE_APPROACH], phase=p, budget=8000) != "arrived":
            self._abort(summary, "travel", "PC approach tile unreached")
            summary["frames"] = runner.frame_idx - f0
            return summary
        if self._boot_to_storage_submenu(runner, summary):
            self._ui_sample(runner, summary)                  # submenu: expect UI_PC
            bag = mr.bag_items(_env(runner))
            pc = mr.pc_items(_env(runner))
            if bag:                                           # deposit-first when possible
                summary["order"] = ["deposit", "withdraw"]
                self._deposit(runner, summary)
                self._ui_sample(runner, summary)
                self._withdraw(runner, summary)
            elif pc:                                          # storyline states: empty bag,
                summary["order"] = ["withdraw", "deposit"]    # the starting Potion in the PC
                self._withdraw(runner, summary)
                self._ui_sample(runner, summary)
                self._deposit(runner, summary)
            else:
                self._abort(summary, "legs", "both bag and PC storage are empty")
        if not self._exit_pc(runner):
            self._abort(summary, "exit", "PC session did not close cleanly")
        return_to_anchor(runner, mk, ctx, summary, phase=p)
        summary["frames"] = runner.frame_idx - f0
        return summary
