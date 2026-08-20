"""W33 corpus-v2 §2 expedition blocks, item 3c: the mart_pc_item family — mart_buy,
item_use, pc_access. Real emulator, real recorder, storyline final.states (skipped
cleanly when the storyline corpus isn't mounted).

Measured ground truth these tests pin (2026-08-20 build session; per-address probe
evidence lives in collection/menu_ram.py):
- Oldale mart (2,4): clerk gfx 83 at (1,3), approach (1,5) facing UP; the LIVE stock
  from sMartInfo read [13, 14, 18, 17] (Potion/Antidote/Parlyz Heal/Awakening) with
  gItems ROM prices 300/100/200/250; a measured purchase: Antidote x2 = money
  3000 -> 2800, bag +(14,2), gMartPurchaseHistory +(14,2); Potion sold back at 150
  (= 300 // 2). Every mart transition is task/cb2-gated (Task_ShopMenu 0x080DFB89,
  buy-menu cb2 0x080DFD65, the menu-helpers yes/no task 0x08121FDD).
- Oldale Center (2,2): item-PC console behavior 0x83 at (10,1), approach (10,2); the
  storyline PC holds the starting Potion (13,1) at SB1+0x498 (plain qty), the bag
  pocket lives at SB1+0x560 (qty XOR the SB2+0xAC key's low half) — both pinned by
  SaveBlock byte-diffs across live deposit/withdraw transactions. Storyline bags are
  EMPTY, so the round trip runs withdraw-then-deposit there (both legs verified).
- BACK_TO_ROUTE101_FROM_OLDALE: lv-8 Mudkip lead at 16/28 (already damaged — the
  overworld heal needs no arranging) and an empty bag, so item_use COMPOSES the mart
  block for its Potions (measured live: bought 2, overworld delta +12 = 28-16, then an
  in-battle delta on gBattleMons[0] after fight-turn damage).
- ui_state 6 (mart) and 7 (pc) are LIVE-PROBED during the flows: the blocks sample the
  ledger's ui_state from full snapshots at the gated screens and record them in
  summary["ui_seen"] (acceptance 5 — no static fixture reaches these UIs).
- Wild battles are stochastic: battle counts assert >= 1 in a generous budget, never
  exact counts (the v2b convention).
"""

import json
from pathlib import Path

import pytest

from collection.direct_runner import DirectEmulatorRunner
from collection.extractors.ledger_panel import UI_MART, UI_PC
from collection.playthrough.blocks.base import run_nav_block
from collection.playthrough.blocks.item_use import ItemUse
from collection.playthrough.blocks.pc_access import PcAccess

_STORYLINE_ROOTS = [Path("data/storyline_wm"),
                    Path("../pokemon-worldmodel/data/storyline_wm"),
                    Path("/root/data0/proj-minhyuk-2026/pokemon-worldmodel/data/storyline_wm")]
STORYLINE = next((p for p in _STORYLINE_ROOTS if p.is_dir()), None)

needs_storyline = pytest.mark.skipif(STORYLINE is None, reason="storyline_wm corpus not mounted")

MAX_ITEM_ID = 376                                # Emerald internal item id bound


def _state_path(milestone: str) -> str:
    p = STORYLINE / milestone / "attempt_000001" / "final.state"
    if not p.exists():
        pytest.skip(f"missing storyline state {p}")
    return str(p)


def _runner(state: str) -> DirectEmulatorRunner:
    r = DirectEmulatorRunner(load_state=state, savestate_every=0)
    r.initialize()
    return r


# ---------------------------------------------------------------------------
# registry / schedule (no emulator)


def test_expedition_registry_and_schedule_v2c():
    from collection.playthrough.schedule import EXPEDITION_BLOCKS, build_expedition_schedule

    assert set(EXPEDITION_BLOCKS) >= {"mart_buy", "item_use", "pc_access"}
    sched = build_expedition_schedule([
        {"after": "OLDALE_TOWN", "block": "mart_buy", "towns": ["0,10"], "frames": 60000},
        {"after": "OLDALE_TOWN", "block": "item_use", "grass_map": "0,16", "town": "0,10"},
        {"after": "OLDALE_TOWN", "block": "pc_access", "center": "2,2"},
    ])
    blocks = sched["OLDALE_TOWN"]
    assert [b.name for b in blocks] == ["mart_buy", "item_use", "pc_access"]
    assert all(hasattr(b, "run") for b in blocks)            # director dispatch marker
    assert [b.phase for b in blocks] == ["mart_pc_item"] * 3  # the family's phase tag
    assert blocks[0].towns == ["0,10"] and blocks[1].grass_map == "0,16"
    assert blocks[2].center == "2,2"


# ---------------------------------------------------------------------------
# acceptance 1 (+5 mart half): mart_buy at Oldale — verified purchases, plausible
# stock, exact money accounting, live UI_MART reads, anchor return, phase rows


@needs_storyline
def test_mart_buy_oldale_verified_purchases(tmp_path):
    from collection.playthrough.expedition import run_expedition_block

    out = run_expedition_block(
        load_state=_state_path("OLDALE_TOWN"), out_dir=str(tmp_path),
        block="mart_buy", kwargs={"towns": ["0,10"], "frames": 90000, "seed": 5},
        visual_fps=1)

    assert out["ran"] is True
    stock = out["stock"]["0,10"]
    assert len(stock) >= 1                                   # sMartInfo stock non-empty
    assert all(0 < iid <= MAX_ITEM_ID for iid in stock)      # plausible item ids
    assert len(out["purchases"]) >= 1
    for pu in out["purchases"]:
        assert pu["verified"] is True                        # the transaction contract:
        assert pu["price_paid"] == pu["rom_price"] * pu["qty"]   # money delta == price
        assert pu["bag_delta"] == pu["qty"]                  # bag delta == qty
        assert pu["history_delta"] == pu["qty"]              # purchase history updated
        assert pu["item_id"] in stock
    for s in out["sells"]:                                   # optional leg: verified when run
        assert s["verified"] is True
        assert s["money_gained"] == (s["rom_price"] // 2) * s["qty"]
    # exact accounting closes over the whole block
    spent = sum(pu["price_paid"] for pu in out["purchases"])
    gained = sum(s["money_gained"] for s in out["sells"])
    assert out["money_before"] - out["money_after"] == spent - gained
    assert UI_MART in out["ui_seen"]                         # live ui_state probe read 6
    assert out["returned"] is True

    # phase rows tagged through the recorded standalone entry
    rows = [json.loads(l) for l in open(tmp_path / "actions.jsonl")]
    assert all("buttons_held" in r for r in rows)            # M1: stream stays homogeneous
    tagged = [r for r in rows if r.get("block_phase") == "mart_pc_item"]
    assert len(tagged) > 500                                 # the block really ran tagged
    trans = [json.loads(l)["phase"] for l in open(tmp_path / "phases.jsonl")]
    assert trans == ["mart_pc_item", "spine"]
    assert json.loads((tmp_path / "block_summary.json").read_text()) == out


# ---------------------------------------------------------------------------
# acceptance 2: item_use overworld — potion use with a verified HP delta (low HP
# arranged legitimately; the storyline lead is already 16/28)


@needs_storyline
def test_item_use_overworld_hp_delta():
    r = _runner(_state_path("BACK_TO_ROUTE101_FROM_OLDALE"))
    try:
        out = run_nav_block(r, ItemUse(grass_map="0,16", town="0,10", frames=150000,
                                       overworld=True, battle=False, seed=2))
        assert out["ran"]
        assert out["overworld_uses"] == 1
        assert len(out["hp_deltas"]["overworld"]) == 1
        assert out["hp_deltas"]["overworld"][0] > 0          # party-struct delta verified
        assert out["battle_uses"] == 0                       # leg disabled
        assert not any(a["where"] == "overworld" for a in out["aborts_with_reason"])
        assert out["returned"] is True
        n = r.nav_state()
        assert n.in_battle is False and n.control_mode == "free_overworld"
    finally:
        r.close()


# ---------------------------------------------------------------------------
# acceptance 3: item_use in battle — potion used mid-battle (gBattleMons[0] delta;
# the party struct freezes in battle), battle exited


@needs_storyline
def test_item_use_battle_hp_delta():
    r = _runner(_state_path("BACK_TO_ROUTE101_FROM_OLDALE"))
    try:
        out = run_nav_block(r, ItemUse(grass_map="0,16", town="0,10", frames=150000,
                                       overworld=False, battle=True, seed=3))
        assert out["ran"]
        assert out["battle_uses"] == 1
        assert len(out["hp_deltas"]["battle"]) == 1
        assert out["hp_deltas"]["battle"][0] > 0             # battle-copy delta verified
        assert out["battles"] >= 1                           # stochastic: >= 1, never exact
        assert out["returned"] is True
        assert r.nav_state().in_battle is False              # battle really exited
    finally:
        r.close()


# ---------------------------------------------------------------------------
# acceptance 4 (+5 pc half): pc_access — deposit/withdraw round trip, both legs
# delta-verified (bag AND pc sides), live UI_PC reads, clean exit


@needs_storyline
def test_pc_access_round_trip():
    r = _runner(_state_path("OLDALE_TOWN"))
    try:
        out = run_nav_block(r, PcAccess(center="2,2", frames=60000))
        assert out["ran"]
        assert len(out["deposits"]) == 1 and len(out["withdrawals"]) == 1
        d, w = out["deposits"][0], out["withdrawals"][0]
        assert d["verified"] is True                         # bag -1 AND pc +1
        assert d["bag_delta"] == -1 and d["pc_delta"] == 1
        assert w["verified"] is True                         # bag +1 AND pc -1
        assert w["bag_delta"] == 1 and w["pc_delta"] == -1
        assert d["item_id"] == w["item_id"]                  # the same item went both ways
        assert sorted(out["order"]) == ["deposit", "withdraw"]
        assert UI_PC in out["ui_seen"]                       # live ui_state probe read 7
        assert out["aborts_with_reason"] == []
        assert out["returned"] is True
        n = r.nav_state()
        assert n.in_battle is False and n.control_mode == "free_overworld"
    finally:
        r.close()
