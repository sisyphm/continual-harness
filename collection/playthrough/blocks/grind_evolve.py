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


_RETURN_RESERVE = 60_000     # frames held back for the walk home (116->Rustboro = 972)


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


def read_lead_stable(runner, tries: int = 6) -> dict | None:
    """A lead reading confirmed by TWO CONSECUTIVE agreeing samples.

    read_lead is checksum-gated but still returns occasional garbage — species 288
    and 0 both appear in live ledgers. That is not cosmetic: the grind block latches
    start_species at entry, and one bad baseline makes "species changed" true forever,
    so the block exits at the target level reporting evolved=true having never
    evolved (measured: exp_022_torchic, 45 wins, ended=target_level, evolved=true,
    ledger says species 280 start to finish). Agreement across reads costs a few
    frames and removes the whole class."""
    prev = None
    for _ in range(tries):
        d = read_lead(runner)
        if d is not None and prev is not None \
                and d["species"] == prev["species"] and d["level"] == prev["level"]:
            return d
        prev = d
        nav._hold(runner, [], 6, "grind")
    return prev


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
                 hp_floor: float = 0.30, grass_map: str | None = None,
                 heal_center: str | None = None, max_heals: int = 3):
        self.target_level = int(target_level)
        self.frames = frames                     # whole-block frame budget
        self.seed = seed
        self.hp_floor = hp_floor
        # Planner-designated grassy map (W33 grind fix): if set and the anchor is a
        # different map, hop there first — a grassless anchor (old RUSTBORO_CITY
        # placement) otherwise turns the whole block into a silent no-op.
        self.grass_map = grass_map
        # W33 sustain fix (measured): with no items, HP chip + PP exhaustion cap an
        # unhealed grind at ~9-15 won battles (~+2 levels) — L16 from L12 is
        # unreachable in one sitting (torchic stalled at L13/L14 across three runs).
        # When `heal_center` (a Center interior map key) is set, an hp_floor breach
        # becomes a bounded nurse trip (goto center -> nurse at (7,2) -> full heal
        # restores HP+PP -> hop back to grass) instead of ending the block, up to
        # `max_heals` cycles. Without it the old end-on-hp_floor behavior stands.
        self.heal_center = heal_center
        self.max_heals = int(max_heals)

    def _goto_map_safe(self, runner, mk, key: str, deadline: int) -> bool:
        from collection.playthrough.blocks.base import goto_map_safe
        return goto_map_safe(runner, mk, key, deadline)

    def _heal_at_center(self, runner, mk, deadline: int) -> bool:
        """Nurse-heal trip: enter the Center, walk to the nurse at (7, 2) — the
        pathfinder auto-presses A when adjacent and facing an NPC — and confirm
        dialogs until the lead reads full HP (the nurse restores PP with it, the
        actual sustain constraint). Bounded by actions and the caller's deadline."""
        from collection.playthrough.blocks.base import _hstate
        from collection.heatz_adapter import find_path_action, is_dialog_open, navigate_ui
        from collection.actions import normalize_action
        if not self._goto_map_safe(runner, mk, self.heal_center, deadline):
            return False
        for _ in range(160):
            if runner.frame_idx >= deadline:
                return False
            lead = read_lead(runner)
            if lead is not None and lead["max_hp"] and lead["hp"] == lead["max_hp"]:
                return True
            h = _hstate(runner)
            act = navigate_ui(h, intent="confirm") if is_dialog_open(h) else find_path_action(h, 7, 2)
            runner.perform_action(normalize_action(act),
                                  metadata={"block": self.name, "src": "heal_trip"})
        lead = read_lead(runner)
        return bool(lead is not None and lead["max_hp"] and lead["hp"] == lead["max_hp"])

    def _fight(self, runner, rng, summary: dict, exp0: int) -> None:
        from collection.collect_behaviors import _battle_one
        _battle_one(runner, rng, "fight")
        if runner.nav_state().in_battle:
            # the heatz battle machine can leave a trainer battle unresolved (its party
            # reader throws on some in-battle states); drive the menus from RAM instead
            from collection.playthrough.blocks.base import force_fight
            force_fight(runner)
        await_overworld(runner, phase=self.phase)
        # Only clear dialogs once the overworld cb2 is genuinely back: _clear_dialog
        # mixes a B press every third input, and B CANCELS an evolution in progress —
        # the one input this block exists to protect.
        if await_overworld(runner, budget=4000, phase=self.phase) and nav._dialog_open(runner):
            nav._clear_dialog(runner, self.phase)
        summary["battles"] += 1
        after = read_lead(runner)
        if after is not None and after["experience"] > exp0:
            summary["battles_won"] += 1
        elif after is not None and after["max_hp"] and after["hp"] == after["max_hp"]:
            summary["whiteouts"] = summary.get("whiteouts", 0) + 1   # the free heal

    def run(self, runner, mk, ctx) -> dict:
        rng = random.Random(self.seed)
        summary = dict(battles_won=0, levels_gained=0, evolved=False, final_level=0,
                       battles=0, ended="budget", skipped=[], frames=0)
        ctx["summary"] = summary   # live reference: survives a budget-guard cut (W33)
        f0 = runner.frame_idx
        deadline = f0 + self.frames
        # base restores the ENTRY savestate when its anchor return fails off-map,
        # which silently DISCARDS every level this block just earned (wave-3
        # measured: 21 won battles rolled back to L13, and the run then met
        # Roxanne under-levelled). Reserve budget to walk home ourselves.
        grind_until = deadline - _RETURN_RESERVE
        lead = read_lead_stable(runner)
        if lead is None:
            summary["skipped"].append(dict(reason="no lead in party"))
            summary["ended"] = "no_lead"
            summary["frames"] = runner.frame_idx - f0
            return summary
        start_level, start_species = lead["level"], lead["species"]
        summary["final_level"] = start_level
        # HEAL BEFORE LEAVING THE ANCHOR (measured): a heal trip that starts deep in
        # the grass fails (wild draws, trainer sight lines, one-way ledges), but the
        # leg-2 anchor stands beside the Rustboro Center — 972 frames, live-verified.
        # A block that arrives under the floor otherwise grinds one battle and quits.
        if self.heal_center and lead["max_hp"]:
            if lead["hp"] / lead["max_hp"] < self.hp_floor and self._heal_at_center(
                    runner, mk, min(deadline, runner.frame_idx + 40_000)):
                summary["heals"] = summary.get("heals", 0) + 1
        if self.grass_map and not self._goto_map_safe(runner, mk, self.grass_map, deadline):
            summary["skipped"].append(dict(reason=f"grass_map {self.grass_map} unreachable"))
            summary["ended"] = "grass_map_unreachable"
            summary["frames"] = runner.frame_idx - f0
            return summary
        t, _, _ = nav._state(runner)
        g = nav.grass_goal(t, mk.behaviors(t)) if t is not None else None
        if g is None or not g.any():
            summary["skipped"].append(dict(reason="no grass in this map's live layout"))
            summary["ended"] = "no_grass"
            summary["frames"] = runner.frame_idx - f0
            return summary
        _stuck = 0
        while runner.frame_idx < grind_until:
            # A whiteout drops us at the Center, so the grass map is re-established
            # every lap rather than assumed (no-op when we are already standing on it).
            if self.grass_map:
                _t, _, _ = nav._state(runner)
                _cur = None if _t is None else f"{_t.map_group},{_t.map_num}"
                if _cur != self.grass_map:
                    if _cur is not None and _cur != self.heal_center:
                        pass
                    if not self._goto_map_safe(runner, mk, self.grass_map, grind_until):
                        summary["ended"] = "grass_map_unreachable"
                        break
            lead = read_lead(runner)
            if lead is None:
                summary["ended"] = "lead_unreadable"
                break
            summary["final_level"] = lead["level"]
            # The goal is the EVOLUTION, not the number. Emerald fires the cutscene on
            # the level-up that crosses the threshold, and a stray B press cancels it —
            # measured: exp_016_torchic sat at L16 still species 280 while the block
            # reported evolved=True and exited. A cancelled evolution gets another
            # chance on the NEXT level-up, so keep grinding until the species actually
            # changes (or the budget ends).
            _chk = read_lead_stable(runner) or lead
            if _chk["level"] >= self.target_level and _chk["species"] != start_species:
                summary["ended"] = "target_level"
                break
            if lead["level"] >= self.target_level + 4:
                summary["ended"] = "level_cap_no_evolution"   # honest, not silent
                break
            frac = lead["hp"] / lead["max_hp"] if lead["max_hp"] else 1.0
            if frac < self.hp_floor:
                # NO heal trip from here. Every measured attempt that started inside
                # the grass failed and burned 25-117k frames: Route 104's crossing is
                # a dialog trap for a one-mon party, and the Route 116 walk draws wild
                # battles and trainer sight lines. The game already has a free, always
                # available heal — faint. A whiteout teleports us to the Center with
                # HP and PP fully restored, the loop head walks back to the grass, and
                # the grind continues. It costs money we never spend, and a whiteout is
                # authentic play that the corpus should contain anyway.
                summary["low_hp_laps"] = summary.get("low_hp_laps", 0) + 1
            r = nav.goto_grass(runner, mk, budget=deadline - runner.frame_idx)
            if r == "arrived":
                r = nav.pace_grass(runner, mk, rng,
                                   budget=min(6000, deadline - runner.frame_idx))
            if r == "battle":
                self._fight(runner, rng, summary, lead["experience"])
            elif r == "stuck":
                # Not necessarily terminal: the walk can be blocked by a passer-by, or
                # by our own transient refusal marks. Measured cost of quitting on the
                # first stuck: exp_019_torchic burned 36.7k frames and won ZERO battles
                # at anchor (10,30), then met the gym under-levelled. Give the grid a
                # few seconds to change and try again before writing the block off.
                _stuck += 1
                if _stuck >= 4:
                    summary["skipped"].append(dict(reason="grass unreachable from here"))
                    summary["ended"] = "grass_unreachable"
                    break
                nav._unstick(runner, self.phase)
                nav._hold(runner, [], 240, self.phase)
            # 'left'/'budget': loop — the deadline governs
        after = read_lead_stable(runner)
        if after is not None:
            summary["final_level"] = after["level"]
            summary["levels_gained"] = after["level"] - start_level
            # only a VALID baseline can prove an evolution; a mid-write read of 0
            # otherwise makes every species look like a change (exp_016 reported
            # evolved=True having never left species 280)
            summary["evolved"] = bool(start_species and after["species"]
                                      and after["species"] != start_species)
        # walk back to the anchor map ourselves so the XP survives (see above)
        amap = (ctx.get("anchor") or (None,))[0]
        if amap:
            t_end, _, _ = nav._state(runner)
            cur = None if t_end is None else f"{t_end.map_group},{t_end.map_num}"
            if cur != amap and not self._goto_map_safe(runner, mk, amap, deadline):
                summary["anchor_return_failed"] = True
        summary["frames"] = runner.frame_idx - f0
        return summary
