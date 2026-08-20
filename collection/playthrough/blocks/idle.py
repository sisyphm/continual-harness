"""IDLE expedition block (W33 corpus-v2 §2): interaction-robustness dwells — small,
since human data now covers the natural edge cases. Ports collect_behaviors.job_idle's
shape onto the nav-block contract: a few short relocations to a fresh spot, then stand
still cycling facings with seeded dwells from the classic (30, 60, 120, 240, 420)-frame
spectrum (0.5 s … 7 s at 60 fps) while the world keeps animating (NPCs wander, water
cycles). Wild battles wandered into during a relocation are fled and counted."""
from __future__ import annotations

import random

from collection import navigator as nav
from collection.playthrough.blocks.base import flee_battle

DIRS = ("UP", "DOWN", "LEFT", "RIGHT")
IDLE_DURATIONS = (30, 60, 120, 240, 420)         # job_idle's dwell spectrum (frames @60fps)


class Idle:
    name = "idle"
    phase = "idle"

    def __init__(self, dwell_spec: list | tuple = IDLE_DURATIONS,
                 frames: int = 6000, seed: int = 0):
        self.dwell_spec = tuple(dwell_spec)
        self.frames = frames                     # whole-block frame budget
        self.seed = seed

    def run(self, runner, mk, ctx) -> dict:
        rng = random.Random(self.seed)
        summary = dict(dwell_frames=0, relocations=0, battles_fled=0, frames=0)
        f0 = runner.frame_idx
        deadline = f0 + self.frames
        while runner.frame_idx < deadline:
            for _ in range(rng.randint(2, 6)):           # relocate: short walk bursts
                nav._hold(runner, [rng.choice(DIRS)], rng.randint(16, 24), self.phase)
                summary["relocations"] += 1
                if runner.nav_state().in_battle:
                    flee_battle(runner)
                    summary["battles_fled"] += 1
                if runner.frame_idx >= deadline:
                    break
            nav._hold(runner, [], 12, self.phase)
            for d in rng.sample(DIRS, k=rng.randint(1, 4)):   # stand, facing around
                if runner.frame_idx >= deadline:
                    break
                nav._hold(runner, [d], 4, self.phase)         # tap-turn in place
                nav._hold(runner, [], 6, self.phase)
                dwell = rng.choice(self.dwell_spec)
                nav._hold(runner, [], dwell, self.phase)
                summary["dwell_frames"] += dwell
        summary["frames"] = runner.frame_idx - f0
        return summary
