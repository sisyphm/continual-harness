"""GRASS life block: farm wild encounters by pacing the grass patch (reusing the
harness's proven grind_action pacing), fighting or fleeing per a seeded probability.
Encounters counted by base.py; heal-guard aborts when lead HP drops. Battles handled
by base.py via block.battle_strategy."""
from __future__ import annotations
import random
from collection.heatz_adapter import grind_action


class Grass:
    name = "grass"

    def __init__(self, seed: int, n_encounters: int = 2, fight_p: float = 0.5,
                 hp_floor: float = 0.35):
        self.rng = random.Random(seed)
        self.target = n_encounters
        self.hp_floor = hp_floor
        self.battle_strategy = "fight" if self.rng.random() < fight_p else "run"

    def setup(self, state, ctx):
        ctx["encounters"] = 0
        ctx["_grind"] = {}

    def _lead_hp_frac(self, state):
        party = (state.get("player") or {}).get("party") or []
        if not party:
            return 1.0
        p0 = party[0]
        cur, mx = p0.get("hp"), p0.get("max_hp")
        return (cur / mx) if (mx and cur is not None) else 1.0

    def act(self, state, ctx):
        if ctx.get("encounters", 0) >= self.target:
            return None
        if self._lead_hp_frac(state) < self.hp_floor:
            return None  # heal-guard
        return grind_action(state, ctx["_grind"])  # paces the grass patch → wild encounters
