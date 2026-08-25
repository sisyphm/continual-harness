"""Run one milestone on an ALREADY-INITIALIZED runner (continuous, no state load).

This is the per-event action loop from collect_events.collect_one_event, lifted to
operate on a shared runner: no runner construction, no pre_state load, no recorder
management. All decision predicates are imported from collect_events (single source
of truth — collect_events itself is untouched and keeps working).
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Any

from collection.actions import ActionTiming, normalize_action
from collection.heatz_adapter import (
    HeatzPolicy, build_heatz_state, grind_action, heal_action, may_heal_action,
    _starter_ui_active,
)
import collection.collect_events as ce


# W33 special-UI fix (clock + starter bag): inside these screens the pacing settle
# predicate is blind (nothing the nav snapshot watches ever changes there), so
# condition pacing floors every tap at ~8 frames and outruns the UI's
# fade-ins/confirm transitions (proven wedges: CLOCK_INTERACT's confirm cycle,
# STARTER_CHOSEN's mudkip cursor trip). Actions taken while one of these UIs is on
# screen use the pre-W33 fixed schedule instead — scoped to the flagged UIs, never
# a global slowdown.
_SPECIAL_UI_TIMING = ActionTiming(hold_frames=12, release_frames=48)

# Oldale Town's Pokemon Center 1F. Identified by map dimensions (14x9), which match
# the two Centre keys already known to the collector — Petalburg 8,4 and Rustboro 11,5
# — rather than assumed from the group numbering.
_OLDALE_CENTER = "2,2"
_RUSTBORO_CENTER = "11,5"
_RIVAL_MIN_LEVEL = 6


def _rival_prep_phase(runner) -> str:
    """'grind' | 'heal' | 'engage' — how ready the lead is to fight the Route 103 rival.

    Owner's rule, from playing it: at level >= 6 on FULL HP the starter wins by pressing
    the first move, whichever starter it is. So the run has to arrive at the rival both
    levelled AND healed. It previously arrived at neither, because the stuck-recovery
    walks up and talks to May the moment the walk to her tile is refused — and her tile
    is refused by definition, she is standing on it. That started the fight at L6 on
    16/23 HP, lost it, and whited the run out (13 of the final wave's 17).
    """
    from collection.playthrough.blocks.grind_evolve import read_lead
    try:
        lead = read_lead(runner)
        if not lead or not lead.get("max_hp"):
            return "engage"                       # unreadable: don't block the milestone
        if int(lead.get("level") or 0) < _RIVAL_MIN_LEVEL:
            return "grind"
        return "engage" if lead["hp"] >= lead["max_hp"] else "heal"
    except Exception:
        return "engage"


def _heal_at_centre(runner, centre_key: str) -> bool:
    """Full-restore at a Centre nurse. The owner's requirement is explicit: use the
    Centre — never the faint-as-heal shortcut, which is what strands a run."""
    from collection import navigator as _nav
    from collection.playthrough.blocks.grind_evolve import GrindEvolve
    try:
        blk = GrindEvolve(target_level=_RIVAL_MIN_LEVEL, heal_center=centre_key)
        ok = blk._heal_at_center(runner, _nav.MapKnowledge(),
                                 runner.frame_idx + 60_000)
        print(f"spine: Centre heal at {centre_key} -> {ok}", flush=True)
        return bool(ok)
    except Exception as e:
        print(f"spine: Centre heal failed: {e!r}", flush=True)
        return False


def _walk_to_expected_tile(runner, expected_state) -> None:
    """Walk to the milestone's expected tile (or a neighbour) with our own navigator."""
    import numpy as _np
    from collection import navigator as _nav
    gx = getattr(expected_state, "x", None)
    gy = getattr(expected_state, "y", None)
    if gx is None or gy is None:
        return
    t, _, _ = _nav._state(runner)
    if t is None or not (0 <= gx < t.map_width and 0 <= gy < t.map_height):
        return

    def _goal(t, beh, gx=int(gx), gy=int(gy)):
        m = _np.zeros(t.grid.shape, bool)
        for dx, dy in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
            yy, xx = gy + dy + 7, gx + dx + 7
            if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]:
                m[yy, xx] = True
        return m & (((t.grid >> 10) & 3) == 0)

    r = _nav.goto(runner, _nav.MapKnowledge(), _goal, budget=25_000, phase="spine")
    print(f"spine: re-anchor tile ({gx},{gy}) -> {r}", flush=True)


def _reanchor_to_expected(runner, expected_state) -> bool:
    """Walk back to the milestone's own map after a whiteout dumped us elsewhere.

    Losing the Route 103 rival battle whites the run out to the player's bedroom in
    Littleroot, three maps from where the milestone expects to be — and the milestone
    still reports `passed`, because its postcondition is a story flag, not a win. The
    NEXT milestone then starts stranded: BACK_TO_OLDALE_FROM_ROUTE103's policy is
    `return 'down'`, so it walks into the bedroom wall while the emulator BFS spins on
    a blocked in-house tile. That is 13 of the final wave's 17 failures.

    Losing is legal Emerald and the story continues, so the run does not need to be
    abandoned — it only needs to get back on the map. Cross maps with our own
    navigator, which reads collision from RAM and handles warps and connections,
    instead of the policy's flat coordinate pathing.
    """
    from collection import navigator as _nav
    em = getattr(expected_state, "map", None) if expected_state is not None else None
    if not em or runner.nav_state().map == em:
        return False
    try:
        import json
        from pathlib import Path
        keys = json.loads((Path(__file__).resolve().parents[1]
                           / "map_name_keys.json").read_text())
        dst = keys.get(str(em).upper())
        if not dst:
            return False
        # If we are stranded in Route 104's south half, no amount of re-routing helps
        # until we are back in the north half — cross the seam first.
        _cross_route104_seam(runner)
        # The walk home crosses two routes of tall grass, so it WILL be interrupted;
        # goto_map returns "battle" partway and one attempt only gets us halfway
        # (measured: bedroom -> ROUTE 101 (13,13), still a map short of Oldale).
        # Resolve each interruption and carry on. Travelling, not grinding: flee every
        # wild battle regardless of HP, and only play out the ones we cannot flee.
        for _ in range(8):
            r = _nav.goto_map(runner, _nav.MapKnowledge(), dst)
            print(f"spine: re-anchor -> {em} ({dst}): {r}", flush=True)
            if r == "arrived" or runner.nav_state().map == em:
                break
            if r != "battle":
                break
            if not _flee_wild_if_critical(runner, floor=1.01):
                _battle_protected(runner, rounds=60)
        if runner.nav_state().map != em:
            return False
        # Landing on the map is not enough. Coming home the long way enters Oldale from
        # the SOUTH (measured: (11,19)), while BACK_TO_OLDALE_FROM_ROUTE103 expects the
        # NORTH entrance (10,1) — and that policy returns 'no_op' the moment it reads
        # OLDALE TOWN, so nothing walks the rest. Put the run on the tile the milestone
        # expects; adjacency is enough for anything that triggers on contact.
        _walk_to_expected_tile(runner, expected_state)
        return True
    except Exception as e:
        print(f"spine: re-anchor failed: {e!r}", flush=True)
        return False


def _flee_wild_if_critical(runner, floor: float = 0.30) -> bool:
    """Run from a WILD battle whose lead is under `floor` HP. True if the battle ended.

    grind_evolve deliberately treats fainting as a free heal ("a whiteout teleports us
    to the Center with HP and PP fully restored") and that is fine INSIDE the grind
    block, which walks itself back to the grass afterwards. Under a story milestone it
    is fatal: the whiteout respawn is the player's own bedroom in Littleroot, three maps
    from wherever the milestone expects to be, and the milestone policy has no idea it
    moved — BACK_TO_OLDALE_FROM_ROUTE103's policy is literally `return 'down'`. That is
    how 13 of the final wave's 17 runs died after the Route 103 rival battle.

    So keep the faint-as-heal trick, but not here: a story milestone flees instead.
    Wild only — RUN is refused in a trainer battle, and mashing it there would just
    walk the action cursor onto the wrong menu entry.
    """
    from collection import navigator as _nav
    from collection.extractors.ledger_panel import _battle_mon
    from collection.extractors.ram import GBAState
    from collection.heatz_adapter import _is_trainer_battle
    try:
        if _is_trainer_battle(runner.env):
            return False
        mon = _battle_mon(GBAState(env=runner.env), 0)
        if not mon or not mon.get("max_hp"):
            return False
        if mon["hp"] / mon["max_hp"] >= floor:
            return False
    except Exception:
        return False                                  # unreadable -> leave it alone
    for _ in range(6):                                # "Couldn't escape!" is possible
        if not runner.nav_state().in_battle:
            return True
        # B FIRST. The caller may be anywhere in the battle UI, not necessarily at the
        # action menu: a dry lead sits on the MOVE list under "There's no PP left for
        # this move!", where LEFT/UP/RIGHT/DOWN walk the move cursor and never reach
        # RUN (measured — the flee silently no-opped and the battle stayed wedged for
        # 37,502 frames). B backs out of a submenu or dismisses the message; from the
        # action menu it is harmless.
        # Then home the cursor on FIGHT (LEFT+UP) before walking it to RUN at
        # bottom-right: the cursor keeps wherever a previous press left it, so a bare
        # RIGHT+DOWN lands somewhere different every time.
        for _k in ("B", "B", "LEFT", "UP", "RIGHT", "DOWN", "A"):
            runner.perform_action(_k, metadata={"src": "flee_critical"})
        _nav._hold(runner, [], 30, "spine")
    return not runner.nav_state().in_battle


# Route 104 is ONE map id with TWO disjoint walkable halves, joined only through
# Petalburg Woods. The map graph cannot express that: a path that leaves 0,19 and
# re-enters 0,19 is unreachable in a BFS whose nodes are maps, so map_route always
# answers "cross the north edge" — unwalkable from the south, and 8 runs died on it.
# Measured live from a stranded run at (28,58): of Route 104's six woods warps only
# (10,38)/(11,38) are reachable from the south half — (32,42)/(33,42) and the north
# pair (10,30) all return "stuck". That warp lands in the woods at (17,38); the woods'
# NORTH exit (14,5)/(15,5) comes back out in Route 104's north half.
_R104 = "0,19"
_R104_SOUTH_MIN_Y = 40          # BFS from the south reaches no tile above y=38
_R104_SOUTH_WARP = (10, 38)
_WOODS = "24,11"
_WOODS_NORTH_WARP = (14, 5)


def _cross_route104_seam(runner) -> bool:
    """South half -> woods -> north half. True if we end up in the north half."""
    from collection import navigator as _nav
    t, x, y = _nav._state(runner)
    if t is None or f"{t.map_group},{t.map_num}" != _R104 or y < _R104_SOUTH_MIN_Y:
        return False
    print(f"spine: Route 104 seam — south half at ({x},{y}), routing via the woods",
          flush=True)
    def _warp_through_battles(wx, wy, tries=6):
        """goto_warp, resolving whatever interrupts it. Petalburg Woods is thick with
        wild grass and trainers, so a single call almost always returns "battle"
        partway: measured (17,38) -> (12,32) -> (10,32), real progress each time."""
        for _ in range(tries):
            r = _nav.goto_warp(runner, _nav.MapKnowledge(), wx, wy, budget=30_000)
            if r == "crossed":
                return True
            if r != "battle":
                return False
            if not _flee_wild_if_critical(runner, floor=1.01):
                _battle_protected(runner, rounds=80)     # trainer: play it out
        return False

    if not _warp_through_battles(*_R104_SOUTH_WARP):
        return False
    if not _warp_through_battles(*_WOODS_NORTH_WARP):
        return False
    t2, x2, y2 = _nav._state(runner)
    ok = t2 is not None and f"{t2.map_group},{t2.map_num}" == _R104 and y2 < _R104_SOUTH_MIN_Y
    print(f"spine: Route 104 seam -> ({x2},{y2}) north={ok}", flush=True)
    return ok


def _press_best_move(runner) -> None:
    """Select the move that does the most damage to the CURRENT foe, and use it.

    The old steer was UP+LEFT+A — always slot 0, type-blind. Against Roxanne's
    ROCK/GROUND mons that is actively wrong: a treecko holding Absorb (GRASS, 2x)
    attacked with Pound (NORMAL, 0.5x) and fought at a quarter of its damage; mudkip
    ignored Water Gun the same way. Every starter carries a super-effective answer
    there, so read the foe's species, score each slot by power x effectiveness, and
    press that one.
    """
    from collection.extractors.ledger_panel import _battle_mon
    from collection.extractors.ram import GBAState
    from collection.move_data import best_slot
    try:
        st = GBAState(env=runner.env)
        me, foe = _battle_mon(st, 0), _battle_mon(st, 1)
        i = best_slot(me.get("moves") or [], me.get("pp") or [], foe.get("species"))
    except Exception:
        i = 0
    if i is None:
        i = 0                                    # nothing damaging: let the caller cope
    # NO MENU GATE HERE. Tried gating the steer on "What will ... do?" being on screen,
    # on the theory that firing the sequence at a text box desyncs the menus. It made
    # things strictly worse: both gym fights that this selector had been winning
    # regressed to nav_wedge_frame_rate_collapse with badges=0, because the dialog
    # reader does not reliably show that prompt when the menu IS up, so the run just
    # pressed A forever and never chose a move. The unconditional sequence wins; the
    # stray presses land on text and are harmless.
    # DO NOT REWRITE THIS SEQUENCE. Two attempts to make it "safer" both regressed the
    # gym fights it was winning: gating it on the "What will ... do?" prompt (the text
    # reader does not reliably show that while the menu IS up -> pressed A forever,
    # badges=0 on both), and prefixing B presses (exp_031 went from a win to
    # wall_time_exceeded). Stray presses landing on text are harmless; the sequence as
    # written wins. Fix the ONE catastrophic outcome reactively instead, below.
    #
    # DO NOT ADD B PRESSES TO THIS PATH. Three separate attempts to make the steer
    # more defensive each regressed a gym fight it had been winning: a menu gate on
    # the "What will ... do?" text (badges=0 on both gyms), a blanket B prefix
    # (exp_031 win -> wall_time_exceeded), and an "unwind the submenu" guard keyed on
    # CB2 0x081A____/0x081B____ (exp_031 win -> nav_wedge_frame_rate_collapse again).
    # The last one explains the pattern: the battle's OWN action and move menus live
    # in that CB2 range, so the guard cancelled legitimate move selection every round.
    # The submenu desync is real -- exp_031 died cycling the POKeMON summary screens --
    # but it is intermittent and the retry machinery re-collects those runs, whereas
    # every "fix" so far has broken a fight that was working. Leave it alone until
    # there is a discriminator that separates the action menu from the party menu.
    # RUN-IN-A-TRAINER-BATTLE. If the opening A is eaten by a text box the following
    # UP/LEFT walk the ACTION cursor and the closing A can confirm RUN, which a
    # trainer battle refuses -- exp_019 spent a whole milestone reading "No! There's
    # no running from a TRAINER battle!". That refusal is visible, so react to it:
    # walk the cursor off RUN before steering again.
    try:
        from collection.heatz_adapter import _read_dialog_text
        if "no running" in (_read_dialog_text(runner.env) or "").lower():
            for _k in ("UP", "LEFT"):
                runner.perform_action(_k, metadata={"src": "unstick_run"})
    except Exception:
        pass
    # action menu -> FIGHT -> move list -> home -> step to the slot in the 2x2 grid
    seq = ["UP", "LEFT", "A", "UP", "LEFT"]
    if i % 2:
        seq.append("RIGHT")
    if i // 2:
        seq.append("DOWN")
    seq.append("A")
    for k in seq:
        runner.perform_action(k, metadata={"src": "best_move"})


def _battle_protected(runner, rounds: int = 160) -> None:
    """Play a battle out while REFUSING to trade Double Kick away.

    Steering (UP+LEFT+A) is the only cycle that actually selects a move, but the same
    UP+LEFT walks the "make room for PECK?" cursor onto Double Kick and the following A
    confirms the delete. Combusken learns Peck at L17, i.e. exactly in the middle of the
    Rustboro gym run, so a torchic reached Roxanne holding [64,45,116,52] -- Peck, not
    Double Kick -- and could not scratch her rock types. Peck/Ember are both resisted;
    Double Kick is the win condition.

    So watch the level between rounds: the moment it ticks up, answer with B (which
    declines the swap) before resuming the steer.
    """
    from collection.battle_driver import drive_battle, enabled as _bv2
    if _bv2():
        # v2 (W34): cursor-verified presses cannot walk the forget-cursor, and
        # move_keeper owns the prompt itself — the level-watch below is the blind
        # steer's compensation, unnecessary when no stray A can reach that cursor.
        drive_battle(runner, src="battle_protected_v2")
        return
    from collection import navigator as _nav
    from collection.extractors.ram import GBAState as _GS
    from collection.extractors.ledger_panel import _battle_mon as _bm

    def _rl(_r):
        # Read the BATTLE structure, not the party. read_lead decrypts the party mon
        # and throws mid-battle on some states ("... is not a valid Move"), returning
        # None at exactly the level-up we are watching for -- so every guard built on
        # it silently skipped and Peck ate Double Kick regardless.
        try:
            return _bm(_GS(env=_r.env), 0)
        except Exception:
            return None

    _KEEP = 24                                   # Double Kick
    _lv = (_rl(runner) or {}).get("level")
    _seq = ("UP", "LEFT", "A")
    _i = 0
    for _ in range(rounds * 3):
        if not runner.nav_state().in_battle:
            return
        # Poll before EVERY action, not once per three-action round: the level-up and
        # its prompt both land inside a single round, so a coarser check confirmed the
        # swap before it ever noticed. The mon goes in holding [24,45,116,52] and came
        # out holding [64,...] with a per-round check.
        _cur = _rl(runner)
        if _cur and _lv and _cur.get("level", 0) > _lv:
            # Declining takes TWO answers, not a B-mash: Emerald asks "Delete a move to
            # make room for PECK?" (B = No) and then "Give up on learning PECK?"
            # (A = Yes). Mashing B alone just bounces between the two boxes until the
            # steer resumes and its A confirms the delete -- which is why the lead kept
            # arriving at Roxanne holding Peck.
            # Don't fight the prompt -- REDIRECT it. Declining takes two correctly
            # timed answers ("delete a move?" No, then "give up learning?" Yes) and
            # every variant of that still lost the move. Accepting is deterministic:
            # say yes, then walk the forget-cursor DOWN off slot 0 (Double Kick) onto
            # slot 1 (Growl) and confirm. Peck replaces the junk move, Double Kick --
            # the only thing that beats her rock types -- survives.
            for _ in range(4):
                if _KEEP not in set((_rl(runner) or {}).get("moves") or ()):
                    break                        # already gone; stop burning frames
                for _k in ("A", "DOWN", "A", "A"):
                    runner.perform_action(_k)
                    _nav._hold(runner, [], 40, "spine")
                if not runner.nav_state().in_battle:
                    break
            _lv = _cur["level"]
            _i = 0
            continue
        _press_best_move(runner)
        _i += 1
        _nav._hold(runner, [], 40, "spine")


def run_milestone(
    runner,
    *,
    event_id: str,
    policy_dir: str,
    expected_state,
    postcondition: str,
    start_money: int,
    # action-count defaults rescaled for condition-based pacing (W33 §3.5): ~3x more actions per frame
    max_actions: int = 12000,
    min_actions: int = 10,
    stall_actions: int = 1200,
    blocked_nav_actions: int = 120,
    starter: str = "mudkip",
    tic_fn=None,
    max_wall_s: float = 480.0,
) -> dict:
    """Drive `runner` with policy `event_id` until its postcondition. Returns a dict
    {validation: passed|failed|skipped, failure_reason, actions_taken, start_frame,
    end_frame}. Mirrors collect_one_event's loop 1:1 (same predicates/branches).

    `tic_fn` (W33 §14.3 micro-behavior noise): optional no-arg callable invoked once
    per action-loop iteration, before the policy reads state. The director passes a
    persona-seeded closure that occasionally emits a recorded human tic (short pause /
    facing flick) through runner.step_frame — during a solve-then-record DRY attempt
    those frames land in the capture log and are replayed like everything else, so
    the tic is simply part of the button schedule.

    `max_wall_s` (W33 loop-bounding invariant): every attempt is bounded by ACTIONS
    and by WALL TIME. The action-denominated stall detector is resettable — wild
    battles/dialogs zero it — so a stranded policy chasing an unreachable goal
    through encounter territory (the RUSTBORO_CENTER_ENTERED wedge: blocks left the
    runner mid-battle on ROUTE 104, the policy then pathed its Rustboro door coords
    against the wrong map for hours) could otherwise outlive any frame budget in
    wall-clock terms. Breaching the cap fails the attempt like a stall; the
    solve-then-record retry/abort machinery above stays in charge."""
    policy = HeatzPolicy(event_id, Path(policy_dir) / event_id / f"{event_id}.py")
    # SPEED-INVARIANT BUDGETS: wall-clock decisions made the policy play a
    # DIFFERENT GAME at different fps (A/B-proven: same seed, sink-off sailed,
    # sink-on wedged). Budgets now count GAME FRAMES (125 f/s converts the old
    # seconds semantics); same seed -> same buttons at any speed/load/recording.
    t_start = time.monotonic()                      # telemetry only
    # Conversion preserves the MOST GENEROUS historical equivalent: the old 480 wall-
    # seconds at the no-sink ~300 fps ceiling = 144k frames. The first constant (125,
    # the slim-sink fleet rate) silently HALVED budgets for fast configs and killed
    # the scene-heavy house exit at 60k frames -- convicted by bisect, not by theory.
    budget_frames = int(max_wall_s * 500) if max_wall_s else 0
    # x500, not a "conversion": old wall budgets spanned 4-10x in frame terms across
    # configs (that variance WAS the bug). Frame budgets must exceed the worst
    # LEGITIMATE game need -- EXIT_RIVAL_HOUSE's entry-scene fixture needs ~85k frames
    # (measured, 40 actions); 240s-callers now get 120k. Generous bounds cost nothing
    # on passes; diagnostics use W33_DIAG (/8).
    # W33_DIAG=1: the DIAGNOSTIC profile (stack charter law 5). Verdicts are the
    # deliverable, not resilience: budgets /8 so a FAILING probe answers in ~1-2 min
    # instead of 10+. Never set on collection runs.
    import os as _osd
    if _osd.environ.get("W33_DIAG") == "1":
        budget_frames //= 8
    budget_f0 = runner.frame_idx
    start_frame = runner.frame_idx
    start_state = runner.state()
    validation = "failed"
    failure_reason = None
    actions_taken = 0
    prev_action = None
    stuck_count = 0
    grind_state: dict = {}
    heal_state: dict = {}
    may_heal_state: dict = {}
    last_progress_key = ce._state_key(start_state)
    target_tail_actions = 0
    target_tail_stall_count = 0
    last_target_tail_key = None
    no_structural_progress_count = 0
    last_structural_progress_key = ce._structural_progress_key(start_state)
    blocked_nav_count = 0
    last_blocked_nav_key = None

    # NEVER ENTER THE LEADER FIGHT WITHOUT AMMUNITION. Roxanne cost three runs at
    # 50/51 — the single most expensive place to fail, each having already burned
    # 80-140 minutes. exp_028 walked in holding Scratch(pp0)/Ember(pp0) and could not
    # select a damaging move at all, so its five attempts were lost before they began
    # ("battle_wedge_unrecoverable"). The Rustboro Centre is a few tiles from the gym
    # door and restores PP, so top up first rather than discover it mid-battle.
    if event_id == "ROXANNE_BATTLE":
        try:
            from collection.move_data import out_of_ammo as _oa
            from collection.playthrough.blocks.grind_evolve import read_lead as _rl0
            if _oa(_rl0(runner)):
                print("spine: gym gate — lead has no damaging PP, topping up first",
                      flush=True)
                _heal_at_centre(runner, _RUSTBORO_CENTER)
        except Exception as _e:
            print(f"spine: gym gate skipped: {_e!r}", flush=True)

    accept_unresponsive_target = event_id in ce._EVENTS_ACCEPTING_UNRESPONSIVE_TARGET
    _crossed_once = False                 # off-map recovery fires at most once per attempt
    _reanchored = False                   # whiteout re-anchor also fires at most once
    _wedge_t, _wedge_f = 0, start_frame   # rolling wedge window
    _rival_ready = False                  # levelled AND healed for the Route 103 rival
    _nurse_done = False                   # counter-talk heal fires at most once
    _last_pos = None                      # position-based stall signal (survives dry solves)
    _stuck_pos = 0
    _recent: list = []                    # sliding window of positions (catches oscillation)
    _adjacent_once = 0                    # blocked-goal walk: up to 3 tries per attempt,
                                          # because losing the fight it triggers must not
                                          # disable the only way to reach the trainer
    _adjacent_once = False                # blocked-goal pre-check, likewise
    _adjacent_fired = 0                   # total firings, incl. ones that met a trainer
    _warped_tries = 0                     # wrong-map (door) search: up to 3 per attempt
    _wrongmap = 0                         # consecutive iterations spent on the WRONG map
    _ongoal = 0                           # consecutive iterations STANDING on the goal tile
    t_start_frames = runner.frame_idx
    battle_stall = 0                      # consecutive in-battle iterations with no frames
    nav_stall = 0                         # ditto, out of battle (map-edge / blocked goal)
    last_frame_seen = runner.frame_idx

    # Already complete on entry (common in continuous mode: handoff lands us past the gate).
    if (
        ce._responsive_target_reached(runner, expected_state)
        or ce._semantic_postcondition_met(event_id, runner, runner.state(), start_money)
        or (accept_unresponsive_target and expected_state is not None
            and ce._target_reached(runner.state(), expected_state))
    ):
        return dict(validation="skipped", failure_reason="already_complete_start",
                    actions_taken=0, start_frame=start_frame, end_frame=runner.frame_idx)

    for _ in range(max_actions):
        if budget_frames and runner.frame_idx - budget_f0 > budget_frames:
            failure_reason = "wall_time_exceeded"   # name kept for monitors/audits
            break
        # FRAME-RATE COLLAPSE = wedged. A healthy milestone steps thousands of frames
        # per second; a wedged one crawls, because every iteration replans a route it
        # can never walk and the emulator barely advances. Measured on the final wave:
        # 846 frames in 240 s (~3.5 fps) after a whiteout teleported the run into its
        # own bedroom, three maps from the goal — the policy then pathed to a blocked
        # tile in the house forever. Neither existing recovery applies (frames DO tick,
        # so the zero-frame test resets; the goal IS in bounds, so the off-map test is
        # false), and at this speed the 200-sample position windows never even fill
        # before the wall budget expires. That cost 13 of the wave's 17 failures a full
        # 240 s each. Fail at a quarter of it: the retry machinery re-runs the milestone
        # from a clean state, which is always cheaper than spinning.
        # RECENT progress, not cumulative. The first version compared against the
        # MILESTONE's start frame, so a milestone that legitimately advanced and then
        # wedged could never trip it: exp_019 sat frozen on Roxanne's tile for two
        # hours with delta=38,419 -- far over the 2000 threshold -- while stepping zero
        # frames. Measure a rolling window instead, so a late stall is caught the same
        # as an early one.
        _wedge_t += 1                                # iterations, not seconds
        if _wedge_t >= 1800:                         # ~= old 90s at typical iter rate
            _wedge_stalled = (runner.frame_idx - _wedge_f) < 2000
            _wedge_t, _wedge_f = 0, runner.frame_idx
        else:
            _wedge_stalled = False
        if _wedge_stalled:
            # Displaced (whiteout) rather than merely stuck? Walk back onto the
            # milestone's map ONCE and give it a fresh budget, then fail if it is
            # still crawling — a run that recovers is worth far more than one that
            # dies correctly.
            if not _reanchored and _reanchor_to_expected(runner, expected_state):
                _reanchored = True
                budget_f0 = runner.frame_idx
                start_frame = runner.frame_idx
                _recent.clear()
                continue
            failure_reason = "nav_wedge_frame_rate_collapse"
            break
        # BATTLE WEDGE BREAKER (W33, measured on five runs at once). A TRAINER battle
        # the policy tries to RUN from deadlocks: RUN is refused, the party read fails
        # ("8650 is not a valid Move"), so the policy stops seeing a battle and paths
        # instead — printing "No progress possible toward (x, y)" forever while
        # stepping ZERO frames, which no action or wall budget can catch because
        # neither advances. Route 116, the leg-2 grind map, is thick with trainer
        # sight lines, so this stalled every run that reached it. Only fires on a
        # PROVEN stall (in battle, frame counter frozen across iterations), so a
        # healthy battle keeps whatever the policy was doing.
        if runner.nav_state().in_battle:
            # Count TIME IN BATTLE, not frozen frames. A deadlocked trainer battle
            # still burns frames in its input holds (measured: exp_001 sat in one
            # battle for 41k ticks at 3 HP, experience frozen at 1890, while stepping
            # 10k frames/70s — so a frame-progress check called it healthy). A real
            # battle resolves in a few hundred iterations; thousands means deadlock.
            battle_stall += 1
            if battle_stall >= 240:
                from collection.playthrough.blocks.base import force_fight
                from collection.playthrough.blocks.grind_evolve import (
                    await_overworld, read_lead as _rlk)
                # force_fight goes B-ONLY the moment the lead holds a keeper move, and
                # B never selects a move -- so on an evolved torchic this recovery could
                # not resolve a battle at all and the milestone died
                # "battle_wedge_unrecoverable". Measured directly: force_fight left that
                # lead at 34/53 with no badge. Steer instead, with the level watch that
                # keeps Double Kick, which is the cycle that beat Roxanne.
                _lk = _rlk(runner)
                if _lk and 24 in set(_lk.get("moves") or ()):
                    _battle_protected(runner, rounds=120)
                    _resolved = not runner.nav_state().in_battle
                else:
                    _resolved = force_fight(runner)
                if _resolved:
                    await_overworld(runner, phase="spine")
                    battle_stall = 0
                else:
                    # Could not play the battle out either. NEVER spin: fail the
                    # milestone so the attempt/retry machinery takes over — a run that
                    # dies in 30 seconds and is re-run beats one that burns a worker
                    # for 40 minutes and fails anyway.
                    failure_reason = "battle_wedge_unrecoverable"
                    break
            else:
                nav_stall = 0
        else:
            battle_stall = 0
            # NAVIGATION stall: same zero-frame spin without a battle. exp_004 froze on
            # goal (16, -1) — a map-edge crossing — printing "No progress possible"
            # while the frame counter never moved, so no action or wall budget applied.
            # Fail the milestone instead of spinning; retry/re-run is always cheaper.
            nav_stall = nav_stall + 1 if runner.frame_idx == last_frame_seen else 0
            # An OFF-MAP goal spins WITH frames. The zero-frame test above only catches
            # a walk that has stopped dead; pathing to a negative coordinate keeps
            # burning frames inside goto while going nowhere, so nav_stall reset every
            # iteration and this whole recovery was never reached. Measured today:
            # exp_001 and exp_004 both finished the grind (L16, evolved, 46/48 wins)
            # and then spun on goal (16,-1) printing "No progress possible" until the
            # watchdog killed them at 226k and 200k frames -- the two most advanced
            # runs of the wave. Position, not frames, is the honest progress signal
            # here, and it is restricted to the off-map case so the in-map gym
            # recoveries further down keep their turn.
            _goal_off = False
            try:
                _gx0 = getattr(expected_state, "x", None)
                _gy0 = getattr(expected_state, "y", None)
                _goal_off = (_gx0 is not None and _gy0 is not None
                             and (_gx0 < 0 or _gy0 < 0))
            except Exception:
                pass
            if nav_stall >= 200 or (_goal_off and _stuck_pos >= 200):
                # WRONG-MAP GOAL. The policy paths the NEXT map's coordinates
                # against the CURRENT map's grid, so the goal lands OUTSIDE the map
                # and no route can reach it. Measured: exp_008_treecko stood in
                # OLDALE TOWN pathing to (4,-1) — Oldale's north exit is x=8..11 —
                # because the target belonged to ROUTE 103. It died there under two
                # different persona seeds while 30 other runs walked through.
                # An out-of-bounds goal IS the direction to travel, so cross that
                # connection with our navigator (which handles edges properly).
                _moved = False
                try:
                    from collection import navigator as _nav
                    _t, _, _ = _nav._state(runner)
                    _gx = getattr(expected_state, "x", None)
                    _gy = getattr(expected_state, "y", None)
                    if _t is not None and _gx is not None and _gy is not None:
                        _dir = (2 if _gy < 0 else 1 if _gy >= _t.map_height else
                                3 if _gx < 0 else 4 if _gx >= _t.map_width else None)
                        if _dir is not None:
                            _r = _nav.cross_connection(runner, _nav.MapKnowledge(), _dir,
                                                       budget=20_000)
                            _moved = _r == "crossed"
                            print(f"nav wedge: goal ({_gx},{_gy}) is off-map "
                                  f"{_t.map_width}x{_t.map_height}; crossed dir {_dir}: {_r}",
                                  flush=True)
                except Exception as _e:
                    print(f"nav wedge cross failed: {_e!r}", flush=True)
                # IN-MAP blocked goal — 54 of 91 failed attempts today, the single
                # biggest failure mode. The policy's goal is often an NPC's own tile
                # (TRAINER_JOSH_BATTLE targets (5,3), where Josh stands), which is by
                # definition unwalkable, so it reports "no progress" forever. Our
                # navigator reads collision and elevation from RAM and remembers
                # refused tiles, so hand it the goal AND its neighbours and let it
                # walk us adjacent — adjacency is all a talk/sight trigger needs.
                if not _moved:
                    try:
                        import numpy as _np
                        from collection import navigator as _nav
                        _t, _, _ = _nav._state(runner)
                        _gx = getattr(expected_state, "x", None)
                        _gy = getattr(expected_state, "y", None)
                        # ONLY when we are genuinely on the goal's map. Bounds alone
                        # cannot tell: TRAINER_JOSH_BATTLE and ROXANNE_BATTLE both
                        # target (5,3) inside RUSTBORO CITY GYM, and (5,3) also lies
                        # inside Route 116's bounds — walking there would be nonsense.
                        # Compare map NAMES, which both sides carry.
                        _same_map = (getattr(expected_state, "map", None)
                                     and runner.nav_state().map == expected_state.map)
                        if _same_map and _t is not None and _gx is not None and _gy is not None \
                                and 0 <= _gx < _t.map_width and 0 <= _gy < _t.map_height:
                            def _goal(t, beh, gx=_gx, gy=_gy):
                                m = _np.zeros(t.grid.shape, bool)
                                for dx, dy in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                                    yy, xx = gy + dy + 7, gx + dx + 7
                                    if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]:
                                        m[yy, xx] = True
                                return m & (((t.grid >> 10) & 3) == 0)
                            _r = _nav.goto(runner, _nav.MapKnowledge(), _goal,
                                           budget=15_000, phase="spine")
                            _moved = _r in ("arrived", "battle")
                            print(f"nav wedge: our navigator to ({_gx},{_gy})+adj -> {_r}",
                                  flush=True)
                    except Exception as _e:
                        print(f"nav wedge in-map fallback failed: {_e!r}", flush=True)
                nav_stall = 0
                if not _moved:
                    # Wrong-map goal we cannot cross to (e.g. a gym interior reached
                    # through a door): fail fast so retry/re-seed takes over instead
                    # of spinning. Recorded distinctly so the audit can count them.
                    _em = getattr(expected_state, "map", None)
                    failure_reason = ("nav_wedge_wrong_map"
                                      if _em and runner.nav_state().map != _em
                                      else "nav_wedge_no_frame_progress")
                    break
        last_frame_seen = runner.frame_idx
        if tic_fn is not None:
            tic_fn()
        # (Two pre-checks lived here — off-map goal and blocked-tile goal — and both
        # are REVERTED. They ran BEFORE the policy on every iteration, and the
        # milestone sweep measured the cost: 46/51 transitions passed before them,
        # 37/51 after. They broke nine milestones that had always worked, including
        # EXIT_RIVAL_HOUSE -> LITTLEROOT_TO_ROUTE101, which stalled all ten runs of
        # wave 10. Acting ahead of the policy on a guess is far more dangerous than
        # reacting to a proven stall; the stall-triggered recovery below stays.)
        # OFF-MAP GOAL, but ONLY once the player has provably stopped moving. The
        # earlier version of this check ran speculatively on every iteration and broke
        # nine milestones (sweep 46->37 of 51). The difference now: position is the
        # stall signal — frame_idx keeps advancing during a dry solve, but a policy
        # that cannot route leaves the player on the same tile. Narrow on both sides:
        # the goal must lie outside this map AND this map must actually have a
        # connection that way, so interiors (leave by a door) are never touched.
        _pos = (runner.nav_state().map, runner.nav_state().x, runner.nav_state().y)
        # OSCILLATION COUNTS AS STUCK. Comparing against the single previous position
        # misses the common case: the policy shuffles the player between two or three
        # tiles, so an "unchanged" counter resets forever and the recovery never runs
        # (exp_037 sat on goal (16,-1) for 300s holding Double Kick while this check
        # did nothing). Track the recent window instead — a walk that only ever visits
        # a handful of tiles is not making progress, however much it moves.
        _recent.append(_pos)
        if len(_recent) > 200:
            _recent.pop(0)
        _stuck_pos = len(_recent) if (len(_recent) >= 200 and len(set(_recent)) <= 4) else 0
        # Dwell on the WRONG map. The position window above only catches a walk that has
        # gone still, but a policy pathing to a goal that lives on another map wanders a
        # whole city and never looks stuck: measured, ROXANNE_BATTLE burned all 6000
        # actions printing "No progress possible toward (5, 3)" from outside the gym,
        # because (5,3) exists in RUSTBORO CITY too. Time spent on the wrong map is the
        # honest signal, and it cannot fire on a milestone already standing where it
        # belongs.
        _em_now = getattr(expected_state, "map", None) if expected_state is not None else None
        _wrongmap = (_wrongmap + 1) if (_em_now and runner.nav_state().map != _em_now) else 0
        # CROSS, don't spin. When the milestone's target is on another map the policy
        # paths that map's coordinates against the grid under our feet and never
        # arrives: exp_034 spun 300+ times on (7,16), a tile inside Birch's Lab, while
        # standing in Littleroot. The frame-progress guards miss it because the spin
        # lives inside nested pathfinder calls, but THIS loop keeps iterating — so
        # count iterations on the wrong map and hand the crossing to goto_map, which
        # understands warps and doors.
        # FIRE EARLY. The crossing itself is slow (goto_map through grass and
        # trainers), so waiting 150 iterations left too little of the attempt's wall
        # budget to finish it: measured, the crossing SUCCEEDS from all four of
        # exp_034's archived wedge states when called directly, yet the cold run
        # failed four times -- it simply ran out of clock mid-crossing and the next
        # attempt started the count over. 40 iterations is still a real stall.
        # RETRY, don't fire once. The first version triggered on == 150 exactly, so a
        # crossing that failed (interrupted by a battle, say) was never attempted
        # again and the run spun out the milestone anyway -- exp_034 wedged on (7,16)
        # a fourth time that way, with the log showing the re-anchor had reached
        # Littleroot but never got through the door.
        if _wrongmap and _wrongmap % 40 == 0 and expected_state is not None:
            print(f"spine: {_wrongmap} iterations on the wrong map "
                  f"(want {_em_now}) — crossing", flush=True)
            _reanchor_to_expected(runner, expected_state)
        # ARRIVING IS ALSO THE TRIGGER, NOT ONLY STALLING. The talk recovery below is
        # gated on _stuck_pos, which needs 200 iterations spent inside four tiles -- a
        # total a run that WANDERS never reaches. Measured on exp_019: it walked to
        # (5,3), Roxanne's own doorstep, stood there, walked back out of the gym to
        # RUSTBORO CITY (20,22), came back, stood on (5,3) again -- so the position
        # window always held a dozen tiles, _stuck_pos stayed 0, and the recovery that
        # wins this fight never ran once in five 8-minute attempts. Standing ON the
        # goal tile with the postcondition still unmet is stronger evidence than any
        # stall: the walk has already done its job and something other than walking is
        # needed. Count that directly.
        _ongoal = (_ongoal + 1) if (
            expected_state is not None
            and _pos == (getattr(expected_state, "map", None),
                         getattr(expected_state, "x", None),
                         getattr(expected_state, "y", None))
            and not runner.nav_state().in_battle) else 0
        # ...BUT ONLY WHERE THERE IS SOMEONE TO TALK TO. Arrival alone is too weak a
        # trigger on its own: plenty of milestones park the player on the expected tile
        # while a SEMANTIC postcondition is still pending, and firing a talk recovery
        # there just burns the attempt's wall budget on an empty tile. Measured, that
        # cost TEAM_AQUA_GRUNT_DEFEATED -> ROUTE_104_NORTH the sweep, twice in a row at
        # exactly 40 actions. The recovery exists to reach an NPC standing next to the
        # goal, so require an NPC to actually be standing there.
        _npc_adj = False
        if _ongoal >= 3:
            try:
                from collection.heatz_adapter import _npc_blocked_tiles as _nbt
                _n2 = runner.nav_state()
                _npc_adj = any(abs(nx - _n2.x) + abs(ny - _n2.y) == 1
                               for nx, ny in _nbt(runner.env,
                                                  exclude_xy=(_n2.x, _n2.y)))
            except Exception:
                _npc_adj = False
        # ...AND ONLY WHERE TALKING IS WHAT THE MILESTONE WANTS. Requiring an adjacent
        # NPC is still not enough -- ROUTE 104 NORTH has villagers standing near its
        # goal tile, so the recovery fired there anyway and lost the sweep at exactly
        # 40 actions, twice, where the unpatched control passed at 79. The stall
        # trigger keeps its old, broad behaviour; ARRIVAL only opens the gate for the
        # milestones whose postcondition an NPC actually satisfies -- a trainer fight
        # or the rival. Everywhere else, standing on the goal tile means keep walking.
        _talk_wins = bool(event_id) and ("BATTLE" in event_id
                                         or event_id == "MAY_ROUTE103_INTERACTION")
        _arrived_at_npc = _ongoal >= 3 and _npc_adj and _talk_wins
        _last_pos = _pos
        if _stuck_pos and not _crossed_once and expected_state is not None:
            _gx = getattr(expected_state, "x", None)
            _gy = getattr(expected_state, "y", None)
            if _gx is not None and _gy is not None:
                from collection import navigator as _nav
                _t, _, _ = _nav._state(runner)
                if _t is not None and not (0 <= _gx < _t.map_width
                                           and 0 <= _gy < _t.map_height):
                    _d = (2 if _gy < 0 else 1 if _gy >= _t.map_height else
                          3 if _gx < 0 else 4)
                    _mk = _nav.MapKnowledge()
                    _key = f"{_t.map_group},{_t.map_num}"
                    if _d in {c.get("direction") for c in _mk.connections.get(_key, [])}:
                        _crossed_once = True
                        _r = _nav.cross_connection(runner, _mk, _d, budget=20_000)
                        print(f"spine: stuck {_stuck_pos} iters at {_pos}, goal "
                              f"({_gx},{_gy}) off-map; crossed dir {_d} -> {_r}", flush=True)
                        _stuck_pos = 0
        # WRONG-MAP GOAL, gated on the SAME proven stall. The goal is in bounds here
        # (so it is not the connection case) but belongs to another map: (5,3) exists
        # in RUSTBORO CITY as well as in RUSTBORO CITY GYM, so the policy walks
        # confidently to the wrong place. An earlier version of this ran speculatively
        # on every iteration and broke nine milestones; it now fires only after the
        # walk has visited four tiles or fewer for 200 iterations, and only when the
        # map NAME disagrees. Try this map's doors until the name matches, stepping
        # back out of any wrong building.
        # A SCRIPTED CUTSCENE IS NOT A STALL. Through the opening the player stands
        # still by design, so the position window latches _stuck_pos at 200 and the
        # recoveries fire into a scene they must not touch -- which stopped the FIRST
        # milestone from ever being solved. Cold runs then sat at 601 recorded frames,
        # which misled me for hours: solve-then-record only writes frames once a
        # milestone is solved, so 601 meant "nothing solved yet", not "frozen".
        _cutscene = False
        if _stuck_pos or _arrived_at_npc or _wrongmap >= 600:
            # _ongoal BELONGS IN THIS TEST TOO. Adding it to the recovery gate below
            # without adding it here left _cutscene stale at False whenever arrival was
            # the only trigger, so the recovery drove straight into scripted scenes:
            # measured, that regressed the sweep 49 -> 46, and the three it broke were
            # PLAYER_NAME_SET, STARTER_CHOSEN and ROUTE_104_NORTH -- the opening
            # cutscenes, where the player stands on the expected tile by design while
            # the scene plays. Every trigger that can open that gate has to ask whether
            # a cutscene is running first.
            from collection import navigator as _nav0
            _cutscene = _nav0._dialog_open(runner)
        if (_stuck_pos or _wrongmap >= 600) and not _cutscene and _warped_tries < 1 \
                and expected_state is not None:
            _em = getattr(expected_state, "map", None)
            if _em and runner.nav_state().map != _em:
                from collection import navigator as _nav
                _t, _, _ = _nav._state(runner)
                if _t is not None:
                    _warped_tries += 1
                    _wrongmap = 0
                    _mk = _nav.MapKnowledge()
                    _key = f"{_t.map_group},{_t.map_num}"
                    for _wp in (_mk.warps.get(_key) or [])[:8]:
                        _r = _nav.goto_warp(runner, _mk, _wp["x"], _wp["y"], budget=6_000)
                        # SETTLE THE SEAM before anyone reads coordinates. The map ID
                        # flips before the coordinates re-base, so a read taken straight
                        # after a warp returns the tile we came FROM: measured, standing
                        # inside RUSTBORO CITY GYM (11x20) the position still read
                        # (27,19), a city tile outside the interior entirely, and BFS
                        # planning from that foreign frame reported "stuck" forever.
                        # After settling it reads (5,19) and the walk to the leader
                        # succeeds on the first try.
                        _nav._settle_seam(runner, "spine")
                        if runner.nav_state().map == _em:
                            print(f"spine: stuck, entered {_em} via warp "
                                  f"({_wp['x']},{_wp['y']})", flush=True)
                            break
                        if _r == "crossed":
                            _t2, _, _ = _nav._state(runner)
                            _k2 = None if _t2 is None else f"{_t2.map_group},{_t2.map_num}"
                            _back = (_mk.warps.get(_k2) or [])[:1]
                            if _back:
                                _nav.goto_warp(runner, _mk, _back[0]["x"], _back[0]["y"],
                                               budget=6_000)
        # BLOCKED GOAL ON THIS MAP: the target tile is an NPC's own square, so no route
        # can ever end there. ROXANNE_BATTLE and TRAINER_JOSH_BATTLE both target (5,3)
        # inside RUSTBORO CITY GYM, where the trainer stands — exp_031 reached the gym
        # and then burned 2,039,665 frames over five attempts without ever starting the
        # fight. Adjacency is all a talk or sight trigger needs. Same stall gate as the
        # other recoveries, so it cannot fire speculatively.
        if (_stuck_pos or _arrived_at_npc) and not _cutscene and _adjacent_once < 8 \
                and _adjacent_fired < 20 \
                and not runner.nav_state().in_battle \
                and expected_state is not None:
            _em = getattr(expected_state, "map", None)
            _gx = getattr(expected_state, "x", None)
            _gy = getattr(expected_state, "y", None)
            if _em and _gx is not None and _gy is not None \
                    and runner.nav_state().map == _em:
                import numpy as _np
                from collection import navigator as _nav
                _nav._settle_seam(runner, "spine")   # never plan from a stale frame
                _t, _, _ = _nav._state(runner)
                # NOTE: do NOT require the goal tile to read as unwalkable. NPCs are
                # OBJECTS, not collision — Roxanne's square (5,3) reads perfectly
                # walkable in the grid while she stands on it, so that condition was
                # never true and this recovery never ran. Being stalled on the goal's
                # own map is enough; targeting the tile AND its neighbours lets BFS
                # settle for adjacency when the tile itself is occupied.
                # Never START the rival fight under-prepared. This recovery exists to
                # reach a goal tile an NPC is standing on, and for MAY_ROUTE103 that NPC
                # IS the fight — so walking up and talking here skips the grind the
                # milestone is built around. Hold it back until level >= 6 and full HP.
                if (event_id == "MAY_ROUTE103_INTERACTION"
                        and _rival_prep_phase(runner) != "engage"):
                    _recent.clear()
                    _stuck_pos = 0
                    _ongoal = 0
                    _prep = _rival_prep_phase(runner)
                    if _prep == "heal" and _heal_at_centre(runner, _OLDALE_CENTER):
                        # The nurse leaves us standing INSIDE the Centre, and the
                        # milestone policy only knows how to path on Route 103 — from
                        # in here it paths to (10,4) against the Centre's own grid and
                        # wedges. Walk back out to the rival's tile ourselves; the wild
                        # draws on the way are fled now that _rival_ready has latched.
                        _reanchor_to_expected(runner, expected_state)
                    continue
                if _t is not None and 0 <= _gx < _t.map_width and 0 <= _gy < _t.map_height:
                    _adjacent_fired += 1
                    # UN-LATCH THE STALL IMMEDIATELY. _stuck_pos stays true until the
                    # position window refills, so leaving the reset to the END of this
                    # block meant every following iteration re-entered it -- a 12k-frame
                    # goto plus a 45-press talk loop per iteration -- and actions_taken
                    # stopped advancing entirely. The run looked wedged at 601 recorded
                    # frames while its frame counter kept climbing.
                    _recent.clear()
                    _stuck_pos = 0
                    _ongoal = 0

                    def _goal(t, beh, gx=_gx, gy=_gy):
                        m = _np.zeros(t.grid.shape, bool)
                        for dx, dy in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                            yy, xx = gy + dy + 7, gx + dx + 7
                            if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1]:
                                m[yy, xx] = True
                        return m & (((t.grid >> 10) & 3) == 0)

                    _r = _nav.goto(runner, _nav.MapKnowledge(), _goal, budget=12_000,
                                   phase="spine")
                    print(f"spine: stuck, goal ({_gx},{_gy}) is an occupied tile on "
                          f"{_em}; walked adjacent -> {_r}", flush=True)
                    if runner.nav_state().in_battle:
                        # the gym's own trainers intercept on the way to the leader;
                        # drive those too, or their level-up prompt eats Double Kick
                        # before the leader fight ever begins
                        _battle_protected(runner)
                    # THEN TALK. Standing next to a gym leader accomplishes nothing --
                    # she fights only when spoken to, and her pre-battle speech swallows
                    # about 25 A presses before the battle starts. Measured: with 8
                    # presses this looked exactly like "talking does not work" and the
                    # walk wandered off to another adjacent tile forever; pressing
                    # through it triggered the fight (foe Geodude L12, her lead) and the
                    # run took the STONE BADGE -- the first badge any W33 run has won.
                    if not runner.nav_state().in_battle:
                        from collection.heatz_adapter import _npc_blocked_tiles
                        # STAND ON THE GOAL TILE ITSELF. The leader is adjacent to the
                        # expected tile, not to whichever of its neighbours BFS happened
                        # to settle on -- arriving at (4,3) leaves Roxanne at (5,2) two
                        # tiles away, so the talk below found nobody and the milestone
                        # burned all 8000 actions "arriving" over and over.
                        def _exact(t, beh, gx=_gx, gy=_gy):
                            m = _np.zeros(t.grid.shape, bool)
                            if 0 <= gy + 7 < m.shape[0] and 0 <= gx + 7 < m.shape[1]:
                                m[gy + 7, gx + 7] = True
                            return m & (((t.grid >> 10) & 3) == 0)

                        _nav.goto(runner, _nav.MapKnowledge(), _exact, budget=6_000,
                                  phase="spine")
                        if runner.nav_state().in_battle:
                            _recent.clear()
                            _stuck_pos = 0
                            continue
                        _t3, _px, _py = _nav._state(runner)
                        _near = [n for n in
                                 _npc_blocked_tiles(runner.env, exclude_xy=(_px, _py))
                                 if abs(n[0] - _px) + abs(n[1] - _py) == 1]
                        print(f"spine: on ({_px},{_py}) goal ({_gx},{_gy}); "
                              f"adjacent NPCs={_near}", flush=True)
                        if _near:
                            _nx, _ny = _near[0]
                            _face = {(1, 0): "RIGHT", (-1, 0): "LEFT",
                                     (0, 1): "DOWN", (0, -1): "UP"}.get(
                                         (_nx - _px, _ny - _py))
                            if _face:
                                runner.perform_action(_face)
                            for _ in range(45):
                                runner.perform_action("A")
                                # SPACE THE PRESSES. Her speech advances one box per
                                # press only if the button is released and the box has
                                # rendered: back-to-back presses are swallowed and look
                                # exactly like "she will not fight". Measured, 30 idle
                                # frames between presses is what turned this from
                                # nothing into the badge.
                                _nav._hold(runner, [], 30, "spine")
                                if runner.nav_state().in_battle:
                                    print(f"spine: talked to NPC at ({_nx},{_ny}) "
                                          f"-> battle", flush=True)
                                    # DRIVE IT OURSELVES. Handing the leader fight back
                                    # to the policy lost it every time (whiteout, full
                                    # HP, no badge, walk back, repeat), while the RAM
                                    # driver -- FIGHT then move slot 0, which is Double
                                    # Kick on an evolved torchic and super-effective on
                                    # her rock types -- beat her Geodude L12 lead and
                                    # took the STONE BADGE.
                                    # STEER, don't use force_fight here: it switches to
                                    # a B-ONLY cycle whenever the lead holds a keeper
                                    # move (Double Kick), and B advances text without
                                    # ever selecting a move -- fine for coasting through
                                    # a wild battle on menu memory, useless against a
                                    # gym leader. Measured: force_fight left the run at
                                    # 34/53 with no badge. This is the exact cycle that
                                    # beat her: FIGHT, first move, A, settle.
                                    # Steering is the ONLY way to attack, but the very
                                    # same UP+LEFT walks the "make room for PECK?" cursor
                                    # onto Double Kick and the next A deletes it -- the
                                    # run came out of this fight holding [64,45,116,52],
                                    # Peck instead of Double Kick, which is why it could
                                    # not beat rock types. Peck is learned at L17, so
                                    # watch the level: the instant it ticks up, answer
                                    # with B (declines the swap, keeps Double Kick) and
                                    # only then resume steering.
                                    _battle_protected(runner)
                                    break
                    # A walk that ended in a BATTLE is progress, not a failed attempt:
                    # the gym's own trainers intercept on the way to the leader, and
                    # each interception used to burn one of only three tries, so the
                    # recovery was exhausted before the leader was ever reached.
                    # Measured: the winning sequence needed three interceptions and
                    # THEN an arrival, i.e. four rounds minimum.
                    if not runner.nav_state().in_battle and _r != "battle":
                        _adjacent_once += 1
                    # Demand FRESH evidence before firing again. The stall window stays
                    # full right after a recovery, so without this the walk re-fired on
                    # every following iteration, started a battle it never let the policy
                    # play, and tripped the battle-wedge detector in 12 seconds flat.
                    _recent.clear()
                    _stuck_pos = 0
        # TAKE OVER ANY BATTLE FOUGHT WITH DOUBLE KICK IN HAND. The gym's trainers
        # intercept during ordinary policy iterations, and the policy answers the L17
        # "make room for PECK?" prompt with A -- deleting the only move that beats rock.
        # Measured: the lead entered the gym holding [24,45,116,52] and reached Roxanne
        # holding [64,45,116,52] every single time, losing a fight it should win.
        # This fires only while the lead actually holds move 24, i.e. an evolved
        # torchic; nothing else in the corpus can trigger it.
        if runner.nav_state().in_battle:
            # Approaching the Route 103 rival: flee EVERY wild draw, whatever our HP.
            # The whole point of the Centre trip is to reach May on full HP, and one
            # Wingull on the walk over undoes it (the grass runs right up to her).
            # Only once we are levelled and healed — before that the wild battles ARE
            # the grind, so they are fought normally under the low-HP guard below.
            if (event_id == "MAY_ROUTE103_INTERACTION" and _rival_ready
                    and _flee_wild_if_critical(runner, floor=1.01)):
                actions_taken += 1
                continue
            # Never faint under a story milestone (see _flee_wild_if_critical).
            if _flee_wild_if_critical(runner):
                actions_taken += 1
                continue
            from collection.playthrough.blocks.grind_evolve import read_lead as _rl2
            _lead2 = _rl2(runner)
            # ROXANNE gets the protected loop whatever the starter holds: it is the one
            # driver that picks by type, and treecko/mudkip never held the keeper move
            # so they used to fall through to the type-blind path and lose.
            if event_id == "ROXANNE_BATTLE" or (_lead2 and 24 in set(_lead2.get("moves") or ())):
                _battle_protected(runner, rounds=60)
                actions_taken += 1
                continue
        if event_id == "MAY_ROUTE103_INTERACTION" and not runner.nav_state().in_battle:
            _rival_ready = _rival_prep_phase(runner) == "engage"
        # THE NURSE IS UNREACHABLE BY DESIGN. HEAL_AT_RUSTBORO_CENTER's expert policy
        # does find_path_action(state, 7, 2) -- the nurse's OWN tile -- but she is
        # sealed behind her counter (row y=3 is "....######....") so BFS can never get
        # there. The policy therefore spins "No progress possible toward (7, 2)" until
        # the run dies; exp_031 was killed by the watchdog that way. Our own heal
        # already knows the trick -- stand at (7,4) and talk ACROSS the counter -- so
        # hand this milestone to it instead of letting the policy path into a wall.
        if event_id == "HEAL_AT_RUSTBORO_CENTER" and not _nurse_done:
            _nurse_done = True
            if _heal_at_centre(runner, _RUSTBORO_CENTER):
                print("spine: nurse heal via counter-talk (policy targets an "
                      "unreachable tile)", flush=True)
        policy_source = "heatz"
        current_before_action = runner.state()
        if not heal_state.get("active") and ce._postcondition_met(
            runner, postcondition, min_actions_met=actions_taken >= min_actions,
            expected_state=expected_state, accept_unresponsive_target=accept_unresponsive_target,
            event_id=event_id, start_money=start_money, current=current_before_action):
            validation = "passed"
            break
        healing = not current_before_action.in_battle and ce._should_heal(current_before_action, heal_state, event_id)
        may_healing = event_id == "MAY_ROUTE103_INTERACTION" and ce._may_heal_needed(current_before_action, may_heal_state)
        if may_healing:
            healing = False
        visible_dialog = ce._visible_dialog_open(runner)
        clock_ui = ce._visual_clock_ui(runner.env)
        starter_ui = event_id == "STARTER_CHOSEN" and _starter_ui_active(runner.env)
        # Conservative pacing only inside the flagged special UIs; the starter gate
        # additionally requires pre-pick out-of-battle state so the rescue battle's
        # (equally white) text box never slows normal battle handling.
        special_ui_timing = clock_ui or (
            starter_ui and not current_before_action.in_battle
            and not (current_before_action.party_summary or []))
        h_state = build_heatz_state(
            runner.env, frame_idx=runner.frame_idx, story_bucket=event_id,
            facing=runner.facing, include_map=event_id in ce._EVENTS_REQUIRING_MAP_STATE)
        h_state["_target_starter"] = starter   # steers _select_starter_action (STARTER_CHOSEN)
        if prev_action:
            h_state["prev_action"] = prev_action.lower()

        action = None
        if healing:
            if expected_state is not None and expected_state.x is not None and expected_state.y is not None:
                h_state["_heal_goal"] = (expected_state.map, expected_state.x, expected_state.y)
            heal_raw = heal_action(h_state)
            if heal_raw == "heal_skip":
                heal_state["active"] = False
                heal_state["skip"] = True
                heal_state.pop("settle_key", None)
                healing = False
            else:
                action = normalize_action(heal_raw)
                policy_source = "heatz_heal"
                stuck_count = 0

        if may_healing and not current_before_action.in_battle:
            action = normalize_action(may_heal_action(h_state))
            policy_source = "heatz_may_heal"
            stuck_count = 0
        elif healing and action is not None:
            pass
        elif not current_before_action.in_battle and ce._grind_target(event_id, current_before_action) is not None:
            action = normalize_action(grind_action(h_state, grind_state))
            policy_source = "heatz_grind"
            stuck_count = 0
        elif action is None:
            action = normalize_action(policy.act(h_state))
            at_physical_target = expected_state is not None and ce._physical_target_reached(current_before_action, expected_state)
            at_exact_target = (
                expected_state is not None and current_before_action.map == expected_state.map
                and current_before_action.x == expected_state.x and current_before_action.y == expected_state.y)
            policy_idle = action == "WAIT"
            can_plan = ce._can_plan_from_state(current_before_action, visible_dialog=visible_dialog)
            special_ui_active = clock_ui or starter_ui
            if not special_ui_active and policy_idle and at_physical_target and visible_dialog:
                action = ce._target_tail_action(target_tail_actions); target_tail_actions += 1
                policy_source = "heatz_script_tail_settle"; stuck_count = 0
            elif not special_ui_active and policy_idle and at_exact_target and not visible_dialog and not ce._movement_responsive(runner):
                action = ce._target_tail_action(target_tail_actions); target_tail_actions += 1
                policy_source = "heatz_script_tail_settle"; stuck_count = 0
            elif not special_ui_active and expected_state is not None and can_plan and (stuck_count >= 5 or action == "WAIT"):
                fb = ce._plan_first_action_to_expected(runner, expected_state)
                if fb:
                    action = normalize_action(fb); policy_source = "heatz_emulator_bfs_fallback"; stuck_count = 0



        state_before_perform = current_before_action
        current_state = runner.perform_action(
            action,
            timing=_SPECIAL_UI_TIMING if special_ui_timing else None,
            metadata={"event_id": event_id, "policy_source": policy_source})
        prev_action = action
        actions_taken += 1
        progress_key = ce._state_key(current_state)
        if progress_key == last_progress_key and action in ce._NAV_ACTIONS:
            stuck_count += 1
        else:
            stuck_count = 0
            last_progress_key = progress_key

        visible_dialog_after = ce._visible_dialog_open(runner)
        structural_key = ce._structural_progress_key(current_state)
        if visible_dialog_after or current_state.in_battle:
            no_structural_progress_count = 0
            last_structural_progress_key = structural_key
        elif structural_key == last_structural_progress_key:
            no_structural_progress_count += 1
        else:
            no_structural_progress_count = 0
            last_structural_progress_key = structural_key
        if stall_actions > 0 and no_structural_progress_count >= stall_actions:
            failure_reason = "stalled_no_structural_progress"
            break

        if action in ce._NAV_ACTIONS and not current_state.in_battle and not state_before_perform.in_battle \
                and state_before_perform.map == current_state.map and state_before_perform.x == current_state.x \
                and state_before_perform.y == current_state.y:
            blocked_nav_key = (current_state.map, current_state.x, current_state.y, action)
            if blocked_nav_key == last_blocked_nav_key:
                blocked_nav_count += 1
            else:
                blocked_nav_count = 1
                last_blocked_nav_key = blocked_nav_key
            if blocked_nav_actions > 0 and blocked_nav_count >= blocked_nav_actions:
                failure_reason = "blocked_repeated_navigation"
                break
        else:
            blocked_nav_count = 0
            last_blocked_nav_key = None

        if expected_state is not None and ce._physical_target_reached(current_state, expected_state) and policy_source == "heatz_script_tail_settle":
            tail_key = ce._tail_progress_key(current_state, ce._visible_dialog_open(runner))
            if tail_key == last_target_tail_key:
                target_tail_stall_count += 1
            else:
                target_tail_stall_count = 0
                last_target_tail_key = tail_key
            if target_tail_stall_count >= ce._TARGET_TAIL_STALL_LIMIT:
                failure_reason = "target_tail_stalled"
                break
        else:
            target_tail_stall_count = 0
            last_target_tail_key = None

        if not heal_state.get("active") and ce._postcondition_met(
            runner, postcondition, min_actions_met=actions_taken >= min_actions,
            expected_state=expected_state, accept_unresponsive_target=accept_unresponsive_target,
            event_id=event_id, start_money=start_money, current=current_state):
            validation = "passed"
            break

    if validation != "passed" and failure_reason is None:
        failure_reason = "max_actions_exhausted"
    return dict(validation=validation, failure_reason=failure_reason, actions_taken=actions_taken,
                start_frame=start_frame, end_frame=runner.frame_idx)
