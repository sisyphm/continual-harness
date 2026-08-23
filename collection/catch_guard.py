"""Corpus-wide invariant: the party is the starter and nothing else.

Two of 279 v2 runs finished holding a wild Pokemon (a Shroomish and a Poochyena, both
Petalburg Woods natives). Reconstructed frame by frame from wedge4_exp_001_torchic:

  f=124679  "Wild SHROOMISH appeared!" -- the encounter fires mid-walk
  f=125046  the battle action menu opens, cursor wandering FIGHT -> BAG -> RUN ->
            POKeMON -> BAG, driven by the directional presses the walk was still
            issuing (a battle freezes the player's tile, so every step is refused and
            the walker keeps trying new directions)
  f=126148  an A lands on BAG; the bag opens on the POKe BALLS pocket
  f=126270  "used POKe BALL!"   ->   f=126882  "Gotcha! SHROOMISH was caught!"

and the run then sat on the nickname keyboard for 27,000 frames (see nickname_guard).

The guard is deliberately narrow. The bag is opened LEGITIMATELY in battle by the
item_use block, which uses Potions out of the ITEMS pocket (pocket 0) -- blanket-
blocking the battle bag would delete that behavior from the corpus. Only the POKe
BALLS pocket (pocket 1, measured live during the throw above) is refused, and only
for A: directional presses stay untouched so item_use can still rewind pocket 1 -> 0
with LEFT when the battle bag happens to open on balls.
"""

from __future__ import annotations

_POCKET_BALLS = 1


def ball_throw_blocked(runner) -> bool:
    """True when an A press right now would throw a Poke Ball."""
    try:
        from collection.extractors.ledger_panel import CB2_ADDR, CB2_BAG
        from collection.extractors.ram import GBAState
        from collection.menu_ram import bag_pocket

        st = GBAState(env=runner.env)
        # Cheap RAM read first; nav_state() is comparatively expensive and this runs
        # before every action, so it is only consulted once the bag is actually up.
        if st.u32(CB2_ADDR) != CB2_BAG:
            return False
        if not runner.nav_state().in_battle:
            return False                      # overworld bag is legitimate behaviour
        # POCKET-BLIND ON PURPOSE. The first version refused A only on pocket 1, and a
        # run still caught a Whismur: "JAXSON used POKe BALL!" at frame 553,176 of
        # exp_001, straight out of a wild battle the walk had wandered into. Whatever
        # pocket the cursor is on, this collector has no business confirming anything
        # in a battle bag -- the party must stay exactly the starter. item_use is the
        # one legitimate user and opts in explicitly for the moment it needs.
        return not bool(getattr(runner, "allow_battle_bag", False))
    except Exception:
        return False
