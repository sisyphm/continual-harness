"""WANDER life block: seeded random walk on walkable porymap tiles within a radius
of the anchor, then base returns to the anchor. Overworld-only (no menu UI). Avoids
grass ('~') so it doesn't farm wilds (that's the GRASS block's job)."""
from __future__ import annotations
import random
from collection.heatz_adapter import find_path_action, ensure_porymap_state


class Wander:
    name = "wander"

    def __init__(self, seed: int, steps: int = 30, radius: int = 6):
        self.rng = random.Random(seed)
        self.steps = steps
        self.radius = radius

    def setup(self, state, ctx):
        ctx["remaining"] = self.steps
        ctx["goal"] = None

    def _walkables(self, state, ax, ay):
        ensure_porymap_state(state)
        grid = ((state.get("map") or {}).get("porymap") or {}).get("grid")
        if not grid:
            return []
        out = []
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c in (".", "P") and abs(x - ax) <= self.radius and abs(y - ay) <= self.radius:
                    out.append((x, y))
        return out

    def act(self, state, ctx):
        if ctx["remaining"] <= 0:
            return None
        pos = (state.get("player") or {}).get("position") or {}
        x, y = pos.get("x"), pos.get("y")
        _, ax, ay = ctx["anchor"]
        goal = ctx.get("goal")
        if goal is None or (x, y) == goal:
            cands = self._walkables(state, ax, ay)
            if not cands:
                return None
            ctx["goal"] = self.rng.choice(cands)
            ctx["remaining"] -= 1
            goal = ctx["goal"]
        return find_path_action(state, goal[0], goal[1])
