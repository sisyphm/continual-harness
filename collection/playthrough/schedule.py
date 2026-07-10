"""seed -> deterministic life-block placement across milestone boundaries. Recorded
into meta.json so a run's diversity is fully reproducible. S2 scope: WANDER only
(overworld-safe). SAVE/BROWSE/MART/GRASS are menu/encounter UI blocks — deferred
(same visual-UI-automation class as the alt-starter blocker); their slots are
reserved here so adding them later is a one-line registry change."""
from __future__ import annotations
import random
from collection.playthrough.blocks.wander import Wander
from collection.playthrough.blocks.grass import Grass
from collection.playthrough.blocks.browse import Browse

# Overworld milestones where a WANDER block is safe to inject after completion.
_WANDER_OK = {
    "ROUTE_101", "OLDALE_TOWN", "ROUTE_103", "ROUTE_102", "PETALBURG_CITY",
    "ROUTE_104_SOUTH", "ROUTE_104_NORTH", "RUSTBORO_CITY", "OLDALE_AFTER_POKEDEX",
    "ROUTE101_AFTER_POKEDEX",
}


_GRASS_OK = {"ROUTE_102", "ROUTE_103", "ROUTE_104_SOUTH", "ROUTE_104_NORTH",
             "ROUTE101_AFTER_POKEDEX"}


def build_schedule(seed: int) -> dict[str, list]:
    """Return {milestone_id: [block, ...]} to run AFTER that milestone passes."""
    rng = random.Random(seed)
    sched: dict[str, list] = {}
    for mid in sorted(_WANDER_OK):
        if rng.random() < 0.5:                       # W p=0.5 per eligible boundary
            steps = rng.randint(12, 40)
            sched.setdefault(mid, []).append(Wander(seed=rng.randint(0, 1 << 30), steps=steps))
    # BROWSE (menu-screen diversity): safe menu-flash, base wrapper guarantees return.
    # Verified in isolation (returns to anchor, free_overworld). Adds bag/party/start-menu
    # pixels the corpus lacks. NOT transaction-verified (menu-mode RAM statics are stale-
    # persistent on this build — SAVE/MART transaction blocks remain an open wall).
    _BROWSE_OK = _WANDER_OK
    for mid in sorted(_BROWSE_OK):
        if rng.random() < 0.35:
            sched.setdefault(mid, []).append(Browse(seed=rng.randint(0, 1 << 30)))

    # GRASS blocks are DISABLED: the integration run (seed 7) showed them adding 0
    # encounters, returning to anchor only 1/4 times, ballooning frames (213k->656k)
    # and BREAKING completion (44/51). grind_action triggers encounters in isolation
    # (battle at step 28) but the block's local pacing + anchor-return is not robust
    # across route geometries (pathfinding to invalid tiles, e.g. y=-1). Re-enable once
    # grass.py hardens patch-targeting + return. Wander-only is proven safe (S1 51/51).
    #
    # for mid in sorted(_GRASS_OK):
    #     if rng.random() < 0.6:
    #         sched.setdefault(mid, []).append(Grass(...))
    return sched
