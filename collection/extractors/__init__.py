"""Rung-A′ condition extractors (Phase 1 of WORLD_MODEL_PLAN_RUNG_A_PRIME.md).

One module per condition group, all reading GBA memory through `ram.GBAState` — a single parser
code path that runs identically over RECORDED condition blobs (offline precompute/audit) and the
LIVE emulator (the interactive demo). Every RAM address used here is validated against recorded
data (`validate_phase1.py`) before anything trains on it — two of three legacy harness addresses
were stale for this build (Phase-0 finding), so nothing is trusted untested.
"""

from collection.extractors.battle import Battler, battle_state, in_battle
from collection.extractors.entities import Entity, entities, player_state
from collection.extractors.ram import GBAState, iter_states
from collection.extractors.terrain import Terrain
from collection.extractors.text import TextState, text_state

__all__ = ["GBAState", "iter_states", "Entity", "entities", "player_state", "Terrain",
           "TextState", "text_state", "Battler", "battle_state", "in_battle"]
