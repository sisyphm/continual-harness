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
from collection.move_data import out_of_ammo as _out_of_ammo
from collection.extractors.ledger_panel import (
    CB2_ADDR, CB2_OVERWORLD, PARTY_ADDR, PARTY_COUNT_ADDR, decrypt_party_mon,
)
from collection.extractors.ram import GBAState


_NURSE = (7, 2)              # nurse's own tile: talk ACROSS the counter, never path to it
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


def read_lead_stable(runner, tries: int = 10) -> dict | None:
    """A lead reading confirmed by TWO CONSECUTIVE agreeing samples.

    read_lead is checksum-gated but still returns occasional garbage — species 288
    and 0 both appear in live ledgers. That is not cosmetic: the grind block latches
    start_species at entry, and one bad baseline makes "species changed" true forever,
    so the block exits at the target level reporting evolved=true having never
    evolved (measured: exp_022_torchic, 45 wins, ended=target_level, evolved=true,
    ledger says species 280 start to finish). Agreement across reads costs a few
    frames and removes the whole class."""
    # NEVER sample during a battle. Measured over 1.84M ledger ticks: 2.13% carry an
    # implausible lead species, and EVERY implausible stretch runs 50+ consecutive
    # ticks — so two agreeing reads a few frames apart agree on the same garbage.
    # The values are foe species (288 = Vigoroth), i.e. the read lands on the enemy
    # party while a battle is up. Waiting for the overworld removes the whole class;
    # agreement then only has to catch the rare single-tick blip (702 of those, all
    # length 1).
    prev = None
    for _ in range(tries):
        if runner.nav_state().in_battle:
            nav._hold(runner, [], 30, "grind")
            continue
        d = read_lead(runner)
        if d is not None and prev is not None \
                and d["species"] == prev["species"] and d["level"] == prev["level"]:
            return d
        prev = d
        nav._hold(runner, [], 6, "grind")
    return prev


_EVOLVED_FORMS = {278, 279, 281, 282, 284, 285}
# Moves worth refusing a later prompt to keep. Double Kick (24) is the reason a
# torchic run can beat a ROCK gym at all.
_KEEPER_MOVES = {24}


def await_overworld(runner, *, budget: int = 9000, phase: str = "grind") -> bool:
    """Advance until gMain.callback2 is the overworld again.

    A-ONLY while the lead can still evolve: B cancels an evolution in progress, the
    one input this block exists to protect.

    Once the lead is ALREADY an evolved form, evolution cannot be pending, and the
    danger inverts. Combusken learns PECK at 17 and the move-learn prompt asks which
    move to forget; A takes the first slot, which is where Double Kick landed on
    evolving. Measured: [24,45,116,52] at L16 became [64,45,116,52] at L17, leaving a
    run to fight a ROCK gym with nothing that hurts rock. The level-up can happen
    mid-gym off Josh's or Roxanne's exp, so declining has to work everywhere, not
    just during the grind. B answers NO to that prompt."""
    st = GBAState.from_env(runner.env)
    lead = read_lead(runner)
    # The gate is whether the KEEPER move is already known, not whether we evolved.
    # Double Kick is offered immediately AFTER the evolution — the ledger shows
    # sp=281 [10,45,116,52] then [24,45,116,52] back to back — so declining on
    # "species is evolved" would refuse the very move we are protecting. Accept
    # prompts until the keeper is in hand; decline afterwards, when the only thing
    # on offer is Peck at 17 and A would forget the keeper's slot.
    has_keeper = bool(lead and _KEEPER_MOVES & set(lead.get("moves") or ()))
    # B-ONLY, not a mix: any A in the rotation answers "make room for PECK?" with yes
    # and the next one deletes slot 1. Safe here because a keeper move only exists
    # after evolving, so no evolution can be pending for B to cancel.
    keys = ["B"] if has_keeper else ["A"]
    f0 = runner.frame_idx
    i = 0
    while runner.frame_idx - f0 < budget:
        if st.u32(CB2_ADDR) == CB2_OVERWORLD:
            return True
        nav._hold(runner, [keys[i % len(keys)]], 4, phase)
        nav._hold(runner, [], 20, phase)
        i += 1
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

    def _release_dialog(self, runner, tries: int = 40) -> bool:
        """Step out of the nurse's closing dialog before anyone tries to walk.

        The heal loop returns the moment HP reads full, which is DURING the nurse's
        sign-off text, and an open dialog freezes movement. Measured: the lead healed
        to 50/50 and was then pinned at (7,4) for 93,157 frames while goto_warp and the
        anchor return both reported "stuck" -- and a block that ends off its anchor map
        has every earned level discarded by base's restore, so a working heal that
        cannot leave is WORSE than no heal at all. nav._clear_dialog does not close this
        one (still open after 1000 frames); the same confirm action that drove the heal
        does. With this, the same trip ends on RUSTBORO CITY at 50/50 in 5,551 frames.
        """
        from collection.playthrough.blocks.base import _hstate
        from collection.heatz_adapter import is_dialog_open, navigate_ui
        from collection.actions import normalize_action
        from collection import navigator as _nav
        for _ in range(tries):
            h = _hstate(runner)
            if is_dialog_open(h):
                runner.perform_action(normalize_action(navigate_ui(h, intent="confirm")),
                                      metadata={"block": self.name, "src": "heal_trip"})
                continue
            if _nav._step(runner, "down"):
                return True
            runner.perform_action("B", metadata={"block": self.name, "src": "heal_trip"})
        return False

    def _heal_at_center(self, runner, mk, deadline: int) -> bool:
        """Nurse-heal trip: enter the Center, walk to the nurse at (7, 2) — the
        pathfinder auto-presses A when adjacent and facing an NPC — and confirm
        dialogs until the lead reads full HP (the nurse restores PP with it, the
        actual sustain constraint). Bounded by actions and the caller's deadline."""
        from collection.playthrough.blocks.base import _hstate
        from collection.heatz_adapter import is_dialog_open, navigate_ui
        from collection.actions import normalize_action
        from collection import navigator as _nav
        if not self._goto_map_safe(runner, mk, self.heal_center, deadline):
            return False
        # DO NOT path TO the nurse. She stands at (7,2), which is walkable in the grid
        # but sealed off behind her counter -- the Center's row y=3 reads "....######...."
        # so BFS can never reach her, and the old find_path_action(h, 7, 2) spun its full
        # 160 actions printing "Goal (7, 2) is STILL BLOCKED" and healed NOTHING. That is
        # why every heal trip silently no-opped: measured, a lead went into this function
        # at 27/50 and came out at 27/50.
        # Walking toward her and talking ACROSS the counter is what actually works:
        # from the door at (7,8) the walk stops at (7,4) and A opens the nurse dialog.
        # Measured on that same lead: 27/50 -> 50/50 in 69 iterations.
        for _ in range(200):
            if runner.frame_idx >= deadline:
                break
            lead = read_lead(runner)
            if lead is not None and lead["max_hp"] and lead["hp"] == lead["max_hp"]:
                self._release_dialog(runner)
                return True
            h = _hstate(runner)
            if is_dialog_open(h):
                runner.perform_action(normalize_action(navigate_ui(h, intent="confirm")),
                                      metadata={"block": self.name, "src": "heal_trip"})
                continue
            _t, _cx, _cy = _nav._state(runner)
            if _t is None:
                runner.perform_action("A", metadata={"block": self.name, "src": "heal_trip"})
                continue
            _dx, _dy = _NURSE[0] - _cx, _NURSE[1] - _cy
            _face = ("UP" if _dy < 0 else "DOWN" if _dy > 0
                     else "LEFT" if _dx < 0 else "RIGHT")
            runner.perform_action(_face, metadata={"block": self.name, "src": "heal_trip"})
            runner.perform_action("A", metadata={"block": self.name, "src": "heal_trip"})
        lead = read_lead(runner)
        _full = bool(lead is not None and lead["max_hp"] and lead["hp"] == lead["max_hp"])
        self._release_dialog(runner)     # never hand back a frozen, dialog-locked run
        return _full

    def _fight(self, runner, rng, summary: dict, exp0: int) -> None:
        # Drive the battle from RAM directly rather than through the heatz machine.
        # Measured repeatedly tonight: that machine can sit inside a trainer battle
        # forever (its party reader throws on some in-battle states, and RUN is refused
        # outright), and because it never RETURNS, a fallback placed after it can never
        # run — exp_010 and exp_013 both deadlocked at L15, one level from evolving,
        # with force_fight sitting uselessly on the next line. force_fight steers
        # UP+LEFT to FIGHT / first move then A, which plays a wild battle out fine and
        # cannot get stuck on the RUN entry.
        # FAST PATH FIRST. Measured: the heatz machine resolves a wild battle in ~1,476
        # ticks, force_fight takes ~18,618 — 12x slower — because it drives menus
        # blindly instead of reading the battle. Making force_fight primary (to dodge
        # trainer deadlocks) cost the grind its throughput: 3 battles per chunk
        # instead of 35, so runs stalled at L14 having fought almost nothing.
        # Use the fast machine, and keep force_fight for exactly what it is good at:
        # a battle the machine could not finish.
        # OUT OF AMMUNITION MID-BATTLE -> LEAVE, DO NOT FIGHT. A lead goes dry DURING a
        # battle, never between them, so the lap-head heal check can never see it:
        # control stays inside this battle while force_fight below steers to slot 0 and
        # mashes A at an empty move ("There's no PP left for this move!"), which is the
        # deadlock the watchdog kills. Measured after the first PP fix shipped:
        # exp_004 still died on Route 104 with pp=[0,39,30,0], because that fix only
        # looked between laps.
        # Read PP from the BATTLE struct, not the party: the party copy is frozen for
        # the duration of a battle, so it still shows the PP this battle already spent.
        try:
            from collection.extractors.ledger_panel import _battle_mon as _bm
            from collection.extractors.ram import GBAState as _GS
            from collection.move_data import damaging_slot as _dslot
            _bmon = _bm(_GS(env=runner.env), 0)
            if _bmon and _dslot(_bmon.get("moves") or [], _bmon.get("pp") or []) is None:
                summary["dry_battles"] = summary.get("dry_battles", 0) + 1
                from collection.playthrough.spine import _flee_wild_if_critical
                if _flee_wild_if_critical(runner, floor=1.01):
                    summary["fled_dry"] = summary.get("fled_dry", 0) + 1
                    return                      # lap head now sees a dry lead -> Centre
        except Exception as _e:
            print(f"grind: dry-battle check skipped: {_e!r}", flush=True)

        from collection.collect_behaviors import _battle_one
        from collection.playthrough.blocks.base import force_fight
        # BOUND the fast machine so the fallback can actually run. _battle_one resolves
        # a wild battle in ~1,476 frames but can sit in a trainer battle forever, and
        # because it never RETURNS, a fallback on the next line is unreachable — that
        # is how runs kept burning 8 minutes until the watchdog killed them. Cap it at
        # ~4x a normal battle, then hand over to force_fight, which is slower but
        # always terminates.
        _deadline = runner.frame_idx + 6000
        _orig = runner.step_frame

        class _BattleTooLong(Exception):
            pass

        def _guarded(*a, **kw):
            if runner.frame_idx >= _deadline:
                raise _BattleTooLong()
            return _orig(*a, **kw)

        runner.step_frame = _guarded
        try:
            _battle_one(runner, rng, "fight")
        except _BattleTooLong:
            summary["battle_timeouts"] = summary.get("battle_timeouts", 0) + 1
        except Exception:
            raise
        finally:
            runner.step_frame = _orig
        if runner.nav_state().in_battle:
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
        # ALREADY EVOLVED -> do nothing. A second rep cannot change the species, so it
        # grinds to its whole budget and overshoots the level, and that is destructive:
        # Combusken learns Double Kick on evolving at 16 and PECK at 17, and the
        # move-learn prompt takes the first slot — which is Double Kick. Measured on
        # exp_013 and exp_019: [24,45,116,52] at L16 became [64,45,116,52] at L17, so
        # the run reaches a ROCK gym having deleted the only move that beats it.
        _EVOLVED = {278, 279, 281, 282, 284, 285}
        if start_species in _EVOLVED:
            summary["ended"] = "already_evolved"
            summary["evolved"] = True
            summary["frames"] = runner.frame_idx - f0
            return summary
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
            # OUT OF AMMUNITION IS ALSO A REASON TO HEAL, and the commoner one. The
            # trigger below was HP-only, but a grind burns PP faster than HP: Torchic
            # carries Scratch(35) + Ember(25) = 60 damaging turns and L11->L16 needs
            # far more, so the lead runs dry while still healthy. Measured on the
            # 2026-08-23 wave: 8 of 12 failures were PP, dying at 33-66% HP — one at
            # 33%, three points above this floor. With no damaging move the battle
            # driver reselects an empty slot forever ("There's no PP left for this
            # move!"), which the watchdog sees as a battle deadlock. The nurse restores
            # PP as well as HP, so the same Centre trip fixes both.
            _dry = _out_of_ammo(lead)
            if frac < self.hp_floor or _dry:
                # RIDING THE FAINT COSTS THE RUN THE MAP. The old comment below argued a
                # whiteout is a free heal that simply returns us to the grass. Measured
                # on exp_016_torchic, it does not: the whiteout respawns at a Centre,
                # the walk back lands in Route 104's SOUTH half, and the anchor (10,30)
                # is in the NORTH half — two regions of one map id ('0,19') joined only
                # through Petalburg Woods. The block recorded exactly that:
                #   low_hp_laps: 10, returned: False, anchor_pos_return: stuck
                # and ended at (27,54), from which RUSTBORO_CITY is unreachable. Every
                # run that got this far died there (8 of 8 in the frozen-build wave).
                # So take the Centre trip when one is configured. The earlier objection
                # (heal trips from inside the grass burn 25-117k frames) was measured
                # before the Centre walk worked; the Route 103 prep now does exactly
                # this trip reliably. Bounded by max_heals so a failing trip cannot
                # loop, and it falls through to the old faint behaviour when no Centre
                # is configured, which keeps every other leg byte-identical.
                summary["low_hp_laps"] = summary.get("low_hp_laps", 0) + 1
                # Cap ATTEMPTS, not successes. The first version counted only successful
                # heals toward max_heals, so a trip that kept failing was never capped:
                # exp_025 made six, and each half-finished walk left it further from the
                # anchor until it ended in Route 104's south half — the very
                # displacement this fix exists to prevent, arriving by another road.
                _tries = summary.get("heal_attempts", 0)
                if self.heal_center and _tries < self.max_heals:
                    summary["heal_attempts"] = _tries + 1
                    _hcap = min(deadline, runner.frame_idx + 60_000)
                    if self._heal_at_center(runner, mk, _hcap):
                        summary["heals"] = summary.get("heals", 0) + 1
                        summary["mid_grind_heals"] = summary.get("mid_grind_heals", 0) + 1
                    else:
                        summary["heal_trip_failed"] = summary.get("heal_trip_failed", 0) + 1
                        # Never leave the run wherever the failed walk stopped: get back
                        # onto the grind's own map before the next lap.
                        if self.grass_map:
                            self._goto_map_safe(runner, mk, self.grass_map, deadline)
                elif _dry and _tries >= self.max_heals:
                    # No ammunition and no heals left: further laps cannot win a battle,
                    # they can only deadlock one. End honestly so the retry machinery
                    # re-runs the block instead of burning the whole frame budget.
                    summary["ended"] = "out_of_pp"
                    break
            r = nav.goto_grass(runner, mk, budget=deadline - runner.frame_idx)
            _where = "goto_grass"
            if r == "arrived":
                _where = "pace_grass"
                r = nav.pace_grass(runner, mk, rng,
                                   budget=min(6000, deadline - runner.frame_idx))
            if r == "battle":
                # Reaching a battle PROVES the walk works, so the stuck budget starts
                # over. Without this reset the counter below is cumulative across the
                # whole block: four transient stucks scattered over 200k frames retire
                # a run that is otherwise grinding fine. Measured — exp_040 and
                # exp_049 both quit at f=38120 with 81% of their budget unspent.
                _stuck = 0
                self._fight(runner, rng, summary, lead["experience"])
            elif r == "stuck":
                # Not necessarily terminal: the walk can be blocked by a passer-by, or
                # by our own transient refusal marks. Measured cost of quitting on the
                # first stuck: exp_019_torchic burned 36.7k frames and won ZERO battles
                # at anchor (10,30), then met the gym under-levelled. Give the grid a
                # few seconds to change and try again before writing the block off.
                _stuck += 1
                summary["last_stuck_at"] = _where
                nav._unstick(runner, self.phase)
                nav._hold(runner, [], 240, self.phase)
                if _stuck in (4, 8) and self.grass_map:
                    # Heavier recovery before writing the block off: re-enter the grass
                    # map outright, which clears our own refusal marks and shakes off an
                    # NPC parked on the path. Quitting at four burned only 19% of the
                    # budget and banked nothing; the walk itself is known-good from both
                    # the expert anchor (412f) and the failing savestate (444f).
                    self._goto_map_safe(runner, mk, self.grass_map,
                                        min(grind_until, runner.frame_idx + 20_000))
                if _stuck >= 12:
                    summary["skipped"].append(dict(reason=f"stuck in {_where}"))
                    summary["ended"] = "grass_unreachable"
                    break
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
        # HEAL BEFORE HANDING BACK. A grind ends whenever its budget or target says so,
        # which is usually at low HP — and the next thing a torchic run does is walk
        # into a GYM. Measured: a run reached Roxanne, triggered the fight at 16/49 HP
        # and lost. The Center is one warp from this block's own anchor (972 frames,
        # live-verified), so topping up here costs almost nothing and is the difference
        # between arriving able to win and arriving unable to.
        _fin = read_lead_stable(runner)
        if (self.heal_center and _fin and _fin["max_hp"]
                and _fin["hp"] / _fin["max_hp"] < 0.8):
            if runner.nav_state().in_battle:
                from collection.playthrough.blocks.base import force_fight
                force_fight(runner)
                await_overworld(runner, budget=6000, phase=self.phase)
            # Bounded by the return reserve (60k): the walk home is measured at 972
            # frames, but a heal that eats the whole reserve would strand the block off
            # its anchor map, and base's restore would then discard every level earned.
            _heal_cap = min(runner.frame_idx + 20_000, deadline - 20_000)
            if _heal_cap > runner.frame_idx and self._heal_at_center(runner, mk, _heal_cap):
                summary["heals"] = summary.get("heals", 0) + 1
                summary["healed_before_exit"] = True
        # LEAVE CLEANLY. base restores the entry savestate when the block ends in a
        # battle or out of the overworld, even if we are standing on the right map —
        # and that restore discards every level earned. Measured across 40 grind
        # blocks tonight: 32 returned false and 3 were rolled back, while
        # anchor_return_failed stayed 0, i.e. the map was never the problem; the block
        # was simply still mid-battle when it stopped. exp_043 lost 34 wins that way.
        if runner.nav_state().in_battle:
            from collection.playthrough.blocks.base import force_fight
            force_fight(runner)
        await_overworld(runner, budget=6000, phase=self.phase)
        if nav._dialog_open(runner):
            nav._clear_dialog(runner, self.phase)
        # WALK BACK TO THE ANCHOR TILE, not merely the anchor MAP. This block anchors at
        # ("0,19", 10, 30) -- and (10,30) is the PETALBURG WOODS warp, the only way from
        # Route 104's south half to its north half. The old code took anchor[0] and threw
        # the coordinates away, so whenever the grind ended anywhere on Route 104 the map
        # already matched, the walk-back did NOTHING, and the run was left wherever the
        # grind wandered (measured: (22,57), deep in the south half).
        # From there the spine is asked to reach RUSTBORO CITY, and BFS proves it cannot:
        # 805 tiles reachable, not one on the north edge. The run oscillates at the wall
        # ~7s per iteration until the watchdog kills it. That is the stall that took a
        # dozen runs at 41-42 milestones.
        # Corroboration: of 30 runs that ever COMPLETED, 28 ran no grind block at all.
        # Runs that grind are the runs that die.
        _anchor = ctx.get("anchor") or (None, None, None)
        amap = _anchor[0]
        ax, ay = (_anchor[1], _anchor[2]) if len(_anchor) >= 3 else (None, None)
        if amap:
            t_end, _, _ = nav._state(runner)
            cur = None if t_end is None else f"{t_end.map_group},{t_end.map_num}"
            if cur != amap and not self._goto_map_safe(runner, mk, amap, deadline):
                summary["anchor_return_failed"] = True
            elif ax is not None and ay is not None:
                import numpy as _np

                def _anchor_goal(t, beh, gx=int(ax), gy=int(ay)):
                    m = _np.zeros(t.grid.shape, bool)
                    if 0 <= gy + 7 < m.shape[0] and 0 <= gx + 7 < m.shape[1]:
                        m[gy + 7, gx + 7] = True
                    return m & (((t.grid >> 10) & 3) == 0)

                _budget = max(0, min(deadline - runner.frame_idx, 25_000))
                if _budget > 0:
                    summary["anchor_pos_return"] = nav.goto(
                        runner, mk, _anchor_goal, budget=_budget, phase=self.phase)
                    _t2, _x2, _y2 = nav._state(runner)
                    summary["anchor_pos_final"] = (
                        None if _t2 is None else [int(_x2), int(_y2)])
        summary["frames"] = runner.frame_idx - f0
        return summary
