"""ITEM_USE expedition block (W33 corpus-v2 §2, item 3c — the mart_pc_item family).

Uses a Potion on the lead both OVERWORLD (START -> BAG -> potion -> USE -> party) and
IN BATTLE (action menu -> BAG -> potion -> USE -> party), each verified by an HP delta
read from the RIGHT series: the party struct out of battle, gBattleMons[0] in battle
(the party struct FREEZES during battle — the ledger's documented fusion rule).

Composition: when the bag lacks Potions the block first runs the mart_buy machinery
(same phase) to buy them — blocks compose, nothing is planted in RAM. Low HP is
arranged LEGITIMATELY: wild-battle fight turns are played (watching gBattleMons[0].hp)
until the lead has taken damage, then the battle is fled.

Gates (all live-verified 2026-08-20, evidence in menu_ram):
  bag open          cb2 == CB2_BAG; ITEMS pocket via gBagPosition+5 (LEFT rewinds)
  potion slot       generic ListMenu task scroll+row sought to the SB1 bag slot; the
                    A-select cross-checked via gSpecialVar_ItemId == 13
  context menu      overworld USE/GIVE/TOSS (sMenu max 3), battle USE/CANCEL (max 1);
                    USE = cursor 0 in both
  party screen      cb2 == CB2_PARTY_MENU; A on slot 0 applies to the lead
  the delta         party-hp poll overworld ('MUDKIP's HP was restored by N points.'
                    measured 19 -> 21), gBattleMons[0].hp poll in battle
  battle return     A-advance until the action menu text 'What will ...' re-composes,
                    then the proven flee

Summary: {overworld_uses, battle_uses, hp_deltas, potions_bought, battles,
aborts_with_reason, frames}.
"""
from __future__ import annotations

import random

from collection import menu_ram as mr
from collection import navigator as nav
from collection.collect_behaviors import _await_battle_menu, _seek_start_slot
from collection.extractors.battle import G_BATTLE_MONS
from collection.extractors.ledger_panel import CB2_ADDR, CB2_BAG, CB2_OVERWORLD, CB2_PARTY_MENU
from collection.extractors.ram import GBAState
from collection.extractors.text import last_message
from collection.playthrough.blocks.base import flee_battle
from collection.playthrough.blocks.grind_evolve import await_overworld, read_lead
from collection.playthrough.blocks.mart import (
    ITEM_POTION, MartBuy, _env, _poll, _press, goto_map_safe, return_to_anchor,
)
from collection.playthrough.blocks.menus import close_start_menu, open_start_menu, start_slots


def _bmon_hp(st: GBAState) -> tuple[int, int]:
    return st.u16(G_BATTLE_MONS + 0x28), st.u16(G_BATTLE_MONS + 0x2C)


class ItemUse:
    name = "item_use"
    phase = "mart_pc_item"

    def __init__(self, grass_map: str, town: str = "0,10", frames: int = 120000,
                 overworld: bool = True, battle: bool = True, seed: int = 0):
        self.grass_map = grass_map               # map key with live grass (e.g. Route 101 "0,16")
        self.town = town                         # mart town for the potion restock composition
        self.frames = frames
        self.overworld = overworld
        self.battle = battle
        self.seed = seed

    def _abort(self, summary, where, reason):
        summary["aborts_with_reason"].append(dict(where=where, reason=reason))

    def _potions(self, runner) -> int:
        return dict(mr.bag_items(_env(runner))).get(ITEM_POTION, 0)

    # ------------------------------------------------------------- battle plumbing

    def _enter_battle(self, runner, mk, rng, summary, *, budget: int = 16000) -> bool:
        r = nav.goto_grass(runner, mk, budget=budget)
        if r == "arrived":
            r = nav.pace_grass(runner, mk, rng, budget=budget)
        if r == "battle":
            summary["battles"] += 1
            return True
        return False

    def _fight_until_damaged(self, runner, *, turns: int = 6) -> bool:
        """FIGHT turns until gBattleMons[0] shows damage; True = damaged AND in battle."""
        for _ in range(turns):
            if not runner.nav_state().in_battle or not _await_battle_menu(runner):
                return False
            hp, mx = _bmon_hp(_env(runner))
            if 0 < hp < mx:
                return True
            _press(runner, "A", 40, self.phase)              # FIGHT
            _press(runner, "A", 700, self.phase)             # first move; the turn plays
        hp, mx = _bmon_hp(_env(runner))
        return runner.nav_state().in_battle and 0 < hp < mx

    def _leave_battle(self, runner) -> None:
        if runner.nav_state().in_battle:
            flee_battle(runner)
        await_overworld(runner, phase=self.phase)
        nav._clear_dialog(runner, self.phase)

    def _damage_lead(self, runner, mk, rng, summary, *, attempts: int = 4) -> bool:
        """Arrange lead HP < max by real play (fight turns, then flee)."""
        for _ in range(attempts):
            lead = read_lead(runner)
            if lead is not None and 0 < lead["hp"] < lead["max_hp"]:
                return True
            if not self._enter_battle(runner, mk, rng, summary):
                continue
            self._fight_until_damaged(runner)
            self._leave_battle(runner)
        lead = read_lead(runner)
        return lead is not None and 0 < lead["hp"] < lead["max_hp"]

    # ------------------------------------------------------------- shared bag drive

    def _pick_potion_in_bag(self, runner, summary, where) -> bool:
        """From an OPEN bag (any location): ITEMS pocket, seek the potion slot, A,
        cross-check the latch. False leaves the selection unopened."""
        p = self.phase
        nav._hold(runner, [], 60, p)                         # bag fade-in settle
        for _ in range(5):
            if mr.bag_pocket(_env(runner)) == 0:
                break
            _press(runner, "LEFT", 25, p)
        for attempt in range(2):
            slot = next((i for i, (iid, _) in enumerate(mr.bag_items(_env(runner)))
                         if iid == ITEM_POTION), None)
            if slot is None or not mr.seek_cursor(runner, mr.list_cursor, slot, phase=p):
                self._abort(summary, where, "potion slot seek failed")
                return False
            _press(runner, "A", 40, p)
            if _env(runner).u16(mr.SPECIAL_VAR_ITEM_ID) == ITEM_POTION:
                return True
            _press(runner, "B", 30, p)                       # wrong latch: close the
        self._abort(summary, where, "selection latched a different item")   # context, retry
        return False

    def _use_on_lead(self, runner, summary, where, hp_read, hp0: int) -> int | None:
        """USE (context cursor 0) -> party screen -> A on the lead -> poll hp_read > hp0."""
        p = self.phase
        if not mr.seek_cursor(runner, lambda st: mr.menu_cursor(st)[0], 0, phase=p):
            self._abort(summary, where, "USE cursor seek failed")
            return None
        _press(runner, "A", 40, p)
        if not _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_PARTY_MENU, 500, p):
            self._abort(summary, where, "party screen never came up")
            return None
        nav._hold(runner, [], 60, p)
        _press(runner, "A", 60, p)                           # lead is slot 0
        for i in range(60):
            hp = hp_read(_env(runner))
            if hp > hp0:
                return hp - hp0
            nav._hold(runner, [], 10, p)
            if i % 8 == 7:                                   # heal text can wait on input
                _press(runner, "A", 20, p)
        self._abort(summary, where, "no HP delta after USE")
        return None

    # ------------------------------------------------------------------ overworld

    def _overworld_use(self, runner, mk, rng, summary) -> None:
        p = self.phase
        if not self._damage_lead(runner, mk, rng, summary):
            self._abort(summary, "overworld", "could not arrange lead HP < max")
            return
        bag0 = self._potions(runner)
        if not open_start_menu(runner, p):
            self._abort(summary, "overworld", "start menu never opened")
            return
        slots = start_slots(runner, p)
        if not _seek_start_slot(runner, slots["bag"], p):
            close_start_menu(runner, p)
            self._abort(summary, "overworld", "BAG cursor seek failed")
            return
        _press(runner, "A", 40, p)
        if not _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_BAG, 500, p):
            close_start_menu(runner, p)
            self._abort(summary, "overworld", "bag never opened")
            return
        if self._pick_potion_in_bag(runner, summary, "overworld"):
            hp0 = read_lead(runner)["hp"]
            delta = self._use_on_lead(runner, summary, "overworld",
                                      lambda st: (read_lead(runner) or {"hp": 0})["hp"], hp0)
            if delta is not None:
                summary["overworld_uses"] += 1
                summary["hp_deltas"]["overworld"].append(delta)
        for _ in range(12):                                  # back out to the field
            if _env(runner).u32(CB2_ADDR) == CB2_OVERWORLD:
                break
            _press(runner, "B", 40, p)
        close_start_menu(runner, p)
        if self._potions(runner) != bag0 - 1 and summary["overworld_uses"]:
            self._abort(summary, "overworld", "bag potion count did not drop by 1")

    # -------------------------------------------------------------------- battle

    def _battle_use(self, runner, mk, rng, summary) -> None:
        p = self.phase
        for attempt in range(3):
            if not self._enter_battle(runner, mk, rng, summary):
                continue
            if not self._fight_until_damaged(runner):        # foe fainted first, etc.
                self._leave_battle(runner)
                continue
            if not _await_battle_menu(runner):
                self._leave_battle(runner)
                continue
            bag0 = self._potions(runner)
            hp0, _mx = _bmon_hp(_env(runner))
            _press(runner, "RIGHT", 20, p)                   # action cursor -> BAG
            _press(runner, "A", 40, p)
            if not _poll(runner, lambda st: st.u32(CB2_ADDR) == CB2_BAG, 600, p):
                self._abort(summary, "battle", "battle bag never opened")
                self._leave_battle(runner)
                continue
            if self._pick_potion_in_bag(runner, summary, "battle"):
                delta = self._use_on_lead(runner, summary, "battle",
                                          lambda st: _bmon_hp(st)[0], hp0)
                if delta is not None:
                    summary["battle_uses"] += 1
                    summary["hp_deltas"]["battle"].append(delta)
                    if self._potions(runner) != bag0 - 1:
                        self._abort(summary, "battle", "bag potion count did not drop by 1")
            for _ in range(20):                              # back to the action menu
                if not runner.nav_state().in_battle:
                    break
                st = GBAState.snapshot(runner.env)
                if st.u32(CB2_ADDR) not in (CB2_BAG, CB2_PARTY_MENU) and \
                        last_message(st, in_battle=True).startswith("What will"):
                    break
                _press(runner, "A", 40, p)
            self._leave_battle(runner)
            if summary["battle_uses"]:
                return
        if not summary["battle_uses"]:
            self._abort(summary, "battle", "no verified in-battle use after 3 battles")

    # --------------------------------------------------------------------- entry

    def run(self, runner, mk, ctx) -> dict:
        rng = random.Random(self.seed)
        summary = dict(overworld_uses=0, battle_uses=0,
                       hp_deltas=dict(overworld=[], battle=[]), potions_bought=0,
                       battles=0, battles_fled=0, aborts_with_reason=[], frames=0)
        f0 = runner.frame_idx
        need = (1 if self.overworld else 0) + (1 if self.battle else 0)
        if self._potions(runner) < need:                     # compose: buy them for real
            mart = MartBuy(towns=[self.town], want=[(ITEM_POTION, max(2, need))],
                           sell=False, frames=min(50000, self.frames))
            sub = mart.run(runner, mk, {})                   # anchor-free: we return once, at
            #                                                # the end of THIS block's run
            summary["potions_bought"] = sum(pu["qty"] for pu in sub["purchases"]
                                            if pu["item_id"] == ITEM_POTION and pu["verified"])
            summary["mart_aborts"] = sub["aborts_with_reason"]
        if not goto_map_safe(runner, mk, self.grass_map, summary, phase=self.phase):
            self._abort(summary, "travel", f"grass map {self.grass_map} unreached")
            summary["frames"] = runner.frame_idx - f0
            return summary
        if self.overworld:
            if self._potions(runner) >= 1:
                self._overworld_use(runner, mk, rng, summary)
            else:
                self._abort(summary, "overworld", "no potion in bag (restock failed)")
        if self.battle:
            if self._potions(runner) >= 1:
                self._battle_use(runner, mk, rng, summary)
            else:
                self._abort(summary, "battle", "no potion in bag (restock failed)")
        return_to_anchor(runner, mk, ctx, summary, phase=self.phase)
        summary["frames"] = runner.frame_idx - f0
        return summary
