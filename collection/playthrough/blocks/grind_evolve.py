"""GRIND_EVOLVE expedition block (W33 corpus-v2 §2): level diversity + the natural
evolution cutscenes (all three starters must appear evolved in the corpus; RAM-synthesised
parties are banned, so evolution happens by PLAY here).

Loop: navigator into the current map's grass (encounter_farm's walk), pace until a wild
battle, FIGHT to win through the existing heatz battle machine
(collect_behaviors._battle_one at fast pacing), then hold an EVOLVE-AWARE watch until
gMain.callback2 is back on the overworld. The watch presses ONLY A: navigator's generic
dialog clearer mixes in B every third press, and B CANCELS an in-progress evolution —
the one input this block exists to protect. A level-up that crosses the evolution
threshold therefore plays its whole cutscene (evolve animation, congratulations text,
move-learning prompts — all A-advance to completion).

Preconditions (skip-with-note, never silent): a lead exists in the party, and the
current map has grass in its live layout. Heal policy: when the lead drops under
`hp_floor` (30%) of max HP the block ends EARLY with reason `lead_hp_low` instead of
risking a whiteout mid-block — the spine's own heal logic owns recovery.

Win detection is the lead's experience delta across the battle (the out-of-battle party
struct is live again by the time the overworld cb2 returns; fled/lost battles gain 0).

Evolution path PILOT-VERIFIED live 2026-08-20 (no synthesis): a lv-12 Mudkip from the
TRAINER_JOSH_BATTLE storyline state was ground on Route 116 to lv 15 / exp 2493 by
real play (nurse heals between sessions), frozen as tests/states/
grind_lv15_route116.state, and THIS block run from it: one won battle crossed the
lv-16 threshold and the whole chain — evolve animation, congratulations text, and the
move-learning dialogs (Marshtomp picked up move 341) — completed under the A-only
watch, ending {battles_won: 1, evolved: true, final_level: 16, ended: target_level}
with control back on the overworld and the anchor restored.

Summary: {battles_won, levels_gained, evolved, final_level} + battles/ended/skipped.
"""
from __future__ import annotations

import random

from collection import navigator as nav
from collection.extractors.ledger_panel import (
    CB2_ADDR, CB2_OVERWORLD, PARTY_ADDR, PARTY_COUNT_ADDR, decrypt_party_mon,
)
from collection.extractors.ram import GBAState


def read_lead(runner) -> dict | None:
    """Slot-0 party mon (decrypted, checksum-gated), or None (empty party/mid-write)."""
    st = GBAState.from_env(runner.env)
    for _ in range(3):
        if st.u8(PARTY_COUNT_ADDR) == 0:
            return None
        d = decrypt_party_mon(st.bytes(PARTY_ADDR, 100))
        if d is not None:
            return d
        nav._hold(runner, [], 8, "grind")        # mid-write: settle and re-read
    return None


def await_overworld(runner, *, budget: int = 9000, phase: str = "grind") -> bool:
    """A-ONLY advance until gMain.callback2 is the overworld again. Covers the battle
    teardown fade AND the evolution scene (its own cb2) with its dialog chain. Never
    presses B — B cancels an evolution in progress."""
    st = GBAState.from_env(runner.env)
    f0 = runner.frame_idx
    while runner.frame_idx - f0 < budget:
        if st.u32(CB2_ADDR) == CB2_OVERWORLD:
            return True
        nav._hold(runner, ["A"], 4, phase)
        nav._hold(runner, [], 20, phase)
    return st.u32(CB2_ADDR) == CB2_OVERWORLD


class GrindEvolve:
    name = "grind_evolve"
    phase = "grind"

    def __init__(self, target_level: int, frames: int = 60000, seed: int = 0,
                 hp_floor: float = 0.30):
        self.target_level = int(target_level)
        self.frames = frames                     # whole-block frame budget
        self.seed = seed
        self.hp_floor = hp_floor

    def _fight(self, runner, rng, summary: dict, exp0: int) -> None:
        from collection.collect_behaviors import _battle_one
        _battle_one(runner, rng, "fight")
        await_overworld(runner, phase=self.phase)
        if nav._dialog_open(runner):             # post-scene leftovers (evolution is over
            nav._clear_dialog(runner, self.phase)  # once the overworld cb2 is back)
        summary["battles"] += 1
        after = read_lead(runner)
        if after is not None and after["experience"] > exp0:
            summary["battles_won"] += 1

    def run(self, runner, mk, ctx) -> dict:
        rng = random.Random(self.seed)
        summary = dict(battles_won=0, levels_gained=0, evolved=False, final_level=0,
                       battles=0, ended="budget", skipped=[], frames=0)
        f0 = runner.frame_idx
        deadline = f0 + self.frames
        lead = read_lead(runner)
        if lead is None:
            summary["skipped"].append(dict(reason="no lead in party"))
            summary["ended"] = "no_lead"
            summary["frames"] = runner.frame_idx - f0
            return summary
        start_level, start_species = lead["level"], lead["species"]
        summary["final_level"] = start_level
        t, _, _ = nav._state(runner)
        g = nav.grass_goal(t, mk.behaviors(t)) if t is not None else None
        if g is None or not g.any():
            summary["skipped"].append(dict(reason="no grass in this map's live layout"))
            summary["ended"] = "no_grass"
            summary["frames"] = runner.frame_idx - f0
            return summary
        while runner.frame_idx < deadline:
            lead = read_lead(runner)
            if lead is None:
                summary["ended"] = "lead_unreadable"
                break
            summary["final_level"] = lead["level"]
            if lead["level"] >= self.target_level:
                summary["ended"] = "target_level"
                break
            if lead["max_hp"] and lead["hp"] / lead["max_hp"] < self.hp_floor:
                summary["ended"] = "lead_hp_low"     # spine heal logic owns recovery
                break
            r = nav.goto_grass(runner, mk, budget=deadline - runner.frame_idx)
            if r == "arrived":
                r = nav.pace_grass(runner, mk, rng,
                                   budget=min(6000, deadline - runner.frame_idx))
            if r == "battle":
                self._fight(runner, rng, summary, lead["experience"])
            elif r == "stuck":
                summary["skipped"].append(dict(reason="grass unreachable from here"))
                summary["ended"] = "grass_unreachable"
                break
            # 'left'/'budget': loop — the deadline governs
        after = read_lead(runner)
        if after is not None:
            summary["final_level"] = after["level"]
            summary["levels_gained"] = after["level"] - start_level
            summary["evolved"] = after["species"] != start_species
        summary["frames"] = runner.frame_idx - f0
        return summary
