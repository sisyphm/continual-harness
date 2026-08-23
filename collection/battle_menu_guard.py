"""A battle the driver can never act in is a battle that never ends.

exp_019 sat in one wild battle for 50,000+ frames with BOTH sides at full HP -- Torchic
30/30 against a Marill 23/23, holding Scratch (pow 40, 35 PP) and Ember (pow 40, 25 PP).
Not a PP deadlock, not a losing fight: no turn ever executed.

Traced on savestate 00353500, the callback2 cycle is periodic with period 6 and never
once touches the battle menu:

    0x081BFAB5  SUMMARY      "DOCILE nature, met at 5, ROUTE 101"
      UP, LEFT  -> stays
      A         -> 0x081B01E1
      UP        -> 0x081B01B1  PARTY_MENU  "Do what with this PkMn?"
      LEFT      -> stays
      A         -> 0x081BFAE5
      UP        -> 0x081BFAB5  back to SUMMARY, forever

An A had landed on POKeMON in the action menu, opening the party screen. The battle
steer is (UP, LEFT, A) -- three keys that only move or confirm, with no B anywhere -- so
once inside the party/summary stack every A pushes deeper and the cycle closes on
itself. The action menu is unreachable from there by construction.

Worse, escaping is not enough on its own. Measured from the same state: four B presses
DO return to battle-main (0x08038421), but the steer immediately re-enters, because the
UP that should have moved the cursor off POKeMON gets consumed by the screen transition.
One press eaten during a transition is all it takes, and the trap is permanent.

So this guard answers every button with B while a party/summary screen is up in battle,
which walks back out in 4-5 presses instead of 15 (translating only A would leave the
UP/LEFT of the steer spinning). Verified on the fixture: escape, then the very next
steers land damage -- foe 23 -> 15, player 30 -> 28, i.e. turns executing again.

Exclusions, both deliberate:
  * move_keeper owns 0x081BFAB5 too (the forget-a-move prompt is raised under the
    summary callback). While it is driving its own exchange, its A presses must reach
    the game, so this guard stands aside on the re-entrancy flag.
  * a block that genuinely wants the battle party menu (a switch) can opt in, the same
    way item_use opts in to the battle bag -- see catch_guard.
"""

from __future__ import annotations

CB2_BATTLE_MAIN = 0x08038421

# Party/summary callbacks observed live inside the trap cycle. Kept as an explicit set
# rather than "anything that is not battle-main": a battle passes through plenty of
# legitimate callbacks (intro, level-up, the forget prompt), and answering B to all of
# them would fight the game instead of the bug.
_TRAP_CB2 = frozenset((
    0x081B01B1,   # CB2_PartyMenu      "Do what with this PkMn?"
    0x081B01E1,   # party menu variant reached from the summary
    0x081BFAB5,   # CB2_Summary        the stat screen
    0x081BFAE5,   # summary variant
))


def submenu_trap_open(runner) -> bool:
    """True when a battle party/summary screen is up and nothing legitimate wants it."""
    try:
        from collection.extractors.ledger_panel import CB2_ADDR
        from collection.extractors.ram import GBAState

        # Cheapest first (0.0025 ms). The overworld leaves immediately.
        if GBAState(env=runner.env).u32(CB2_ADDR) not in _TRAP_CB2:
            return False
        if getattr(runner, "_in_move_keeper", False):
            return False                  # the keeper is mid-exchange; do not eat its A
        if getattr(runner, "allow_battle_menu", False):
            return False                  # a block asked for the party menu on purpose
        return bool(runner.nav_state().in_battle)
    except Exception:
        return False
