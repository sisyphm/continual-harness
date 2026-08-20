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
from collection.playthrough.blocks.bfs_sweep import BfsSweep
from collection.playthrough.blocks.encounter_farm import EncounterFarm
from collection.playthrough.blocks.grind_evolve import GrindEvolve
from collection.playthrough.blocks.idle import Idle
from collection.playthrough.blocks.interaction import Interaction
from collection.playthrough.blocks.item_use import ItemUse
from collection.playthrough.blocks.mart import MartBuy
from collection.playthrough.blocks.menus import Menus
from collection.playthrough.blocks.pc_access import PcAccess

class _LegacyLifeBlock:
    """W33 §14.3 wander/browse restoration: adapt a v1 act-style life block (run via
    base.run_block) to the v2 nav-block contract (.name/.phase/.run) so it is
    schedulable in expeditions AND its frames get a real phase tag — run_nav_block's
    set_phase(block.phase) stamps every frame + drops the boundary savestate, exactly
    like the native v2 blocks (v1 ran these untagged). run_block's own settle/anchor
    machinery still does the work; run_nav_block's outer anchor-return is then a no-op.
    Inner summary keys that collide with run_nav_block's outcome fields are prefixed
    `legacy_` (run_nav_block merges the summary via **kwargs)."""

    _inner_cls: type = None      # subclasses bind these
    phase: str = ""

    def __init__(self, max_actions: int = 900, **kwargs):
        self.inner = self._inner_cls(**kwargs)
        self.name = self.inner.name
        self.max_actions = max_actions

    def run(self, runner, mk, ctx) -> dict:
        from collection.playthrough.blocks.base import run_block
        out = run_block(runner, self.inner, max_actions=self.max_actions)
        return {(f"legacy_{k}" if k in ("block", "ran", "anchor", "returned") else k): v
                for k, v in out.items()}


class WanderBlock(_LegacyLifeBlock):
    _inner_cls, phase = Wander, "wander"


class BrowseBlock(_LegacyLifeBlock):
    _inner_cls, phase = Browse, "browse"


# W33 corpus-v2 §2 expedition blocks (navigator-driven; run via base.run_nav_block).
# The slot for the remaining library entry (fishing) stays reserved here — it lands
# as one registry line.
EXPEDITION_BLOCKS: dict[str, type] = {
    "bfs_sweep": BfsSweep,
    "encounter_farm": EncounterFarm,
    "interaction": Interaction,
    "grind_evolve": GrindEvolve,
    "idle": Idle,
    "menus": Menus,
    "mart_buy": MartBuy,                         # W33 item 3c: the mart_pc_item family
    "item_use": ItemUse,
    "pc_access": PcAccess,
    "wander": WanderBlock,                       # W33 §14.3: v1 life blocks restored,
    "browse": BrowseBlock,                       # phase-tagged via _LegacyLifeBlock
}


def build_expedition_schedule(entries: list[dict]) -> dict[str, list]:
    """Explicit, deterministic expedition schedule: [{"after": milestone_id,
    "block": name, **kwargs}] -> {milestone_id: [block, ...]}. The seeded ROTATION
    scheduler (W33 §2/§9 change-matrix targeting) is a later item; this makes the
    blocks carriable by a playthrough today, from config recorded in the manifest."""
    sched: dict[str, list] = {}
    for e in entries:
        kwargs = {k: v for k, v in e.items() if k not in ("after", "block")}
        sched.setdefault(e["after"], []).append(EXPEDITION_BLOCKS[e["block"]](**kwargs))
    return sched

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
