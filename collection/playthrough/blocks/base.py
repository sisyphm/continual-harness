"""Life-block base wrapper: run a seeded 'living' behaviour between milestones on the
SHARED runner, with hard safety so a block can never break the spine.

Contract (agreed S2 spec):
  preconditions  — overworld, not in battle, no dialog (else skip the block cleanly)
  safety intercepts — battle → handle_battle('run'); dialog → navigate_ui(confirm)
  frame budget   — bounded; on exhaustion, abort
  abort/finish   — path back to the ENTRY tile (anchor) so the next milestone starts
                   from the same place the spine handed off
  logging        — returns an outcome dict for meta.json
The block itself only implements `act(state, ctx)` → button, and signals done.
"""
from __future__ import annotations
from typing import Any, Callable

from collection.actions import normalize_action
from collection.heatz_adapter import (
    build_heatz_state, handle_battle, navigate_ui, is_dialog_open, find_path_action,
)
import collection.collect_events as ce


def _hstate(runner, event_id="LIFE_BLOCK"):
    h = build_heatz_state(runner.env, frame_idx=runner.frame_idx, story_bucket=event_id,
                          facing=runner.facing, include_map=True)
    return h


def _settle_for_block(runner, *, max_actions: int = 60):
    """Bounded ACTIVE settle at a block boundary (W33 blocks fix). The spine can hand
    off on a script tail, and — worse — a previous same-boundary block that failed its
    anchor return can leave a LIVE battle or dialog behind. The old settle (40 idle
    frames) could clear neither, so every subsequent block at that boundary skipped on
    precondition_not_overworld (pilot evidence: torchic/treecko each lost 3 of their 4
    RUSTBORO_CITY blocks, including torchic's Roxanne-critical grind_evolve, after one
    bfs_sweep ended off-anchor). Reuse the blocks' own safety intercepts — flee
    battles, confirm dialogs, idle otherwise — bounded; if the world is still not
    free-overworld afterwards the caller's skip stays clean."""
    st = runner.state()
    for _ in range(max_actions):
        if not st.in_battle and st.control_mode == "free_overworld" and not is_dialog_open(_hstate(runner)):
            break
        h = _hstate(runner)
        if st.in_battle:
            # Cursor-verified flee (W34) — the heatz "run" strategy here was another
            # blind battle-menu cycle, the same family that caught rg_020's Poochyena.
            from collection.battle_driver import drive_battle
            drive_battle(runner, mode="flee", src="block_settle")
        elif is_dialog_open(h):
            runner.perform_action(normalize_action(navigate_ui(h, intent="confirm")),
                                  metadata={"src": "block_settle"})
        else:
            # Script tail: no battle, no dialog, control not free — a post-milestone
            # cutscene can hold control for thousands of frames, and the old 4-frame
            # idle × 60 gave up after 240 (wave 1: a whole trainer_engagement block —
            # three targets — skipped on precondition_not_overworld exactly this way).
            # Wait in 40-frame beats; max_actions bounds the total.
            for _ in range(40):
                runner.step_frame([], phase="block_settle")
        st = runner.state()
    return st


def run_block(runner, block, *, max_actions: int = 900) -> dict:  # rescaled for condition-based pacing (W33 §3.5)
    """Drive `runner` with `block.act` under the safety wrapper. `block` needs:
       .name, .setup(state)->anchor(x,y,map), .act(state, ctx)->(button|None; None=done)."""
    start_frame = runner.frame_idx
    # Settle: the spine may hand off on a script tail (a "skipped/already-complete"
    # milestone) and an earlier block may have left a battle/dialog. Active, bounded.
    st0 = _settle_for_block(runner)
    if st0.in_battle or is_dialog_open(_hstate(runner)):
        return dict(block=block.name, ran=False, reason="precondition_not_overworld",
                    frames=runner.frame_idx - start_frame, actions=0)
    if st0.control_mode != "free_overworld":
        # control_mode is POSITION-POISONED near script triggers (measured, W34:
        # Route 104 (39,63) reads 'dialogue' indefinitely while the player moves
        # freely in all four directions — this one lying flag skipped the same
        # trainer block in all six audited runs, five trainers each). The game's
        # own truth is a step: if any direction lands, the overworld is live.
        from collection import navigator as _nav
        back = None
        for d, b in (("left", "right"), ("right", "left"), ("up", "down"), ("down", "up")):
            if _nav._step(runner, d):
                back = b
                break
        if back is None:
            return dict(block=block.name, ran=False, reason="precondition_not_overworld",
                        frames=runner.frame_idx - start_frame, actions=0)
        _nav._step(runner, back)              # restore the anchor tile
        st0 = runner.state()
    anchor = (st0.map, st0.x, st0.y)
    entry_snap = runner.save_state_bytes()
    ctx: dict[str, Any] = {"anchor": anchor}
    block.setup(_hstate(runner), ctx)
    actions = 0
    aborted = None
    for _ in range(max_actions):
        st = runner.state()
        h = _hstate(runner)
        # Safety intercepts first — a wandered-into battle/dialog never derails the run.
        if st.in_battle:
            ctx["_in_battle"] = True
            strat = getattr(block, "battle_strategy", "run")
            runner.perform_action(normalize_action(handle_battle(h, strategy=strat)),
                                  metadata={"block": block.name, "src": "battle"})
            actions += 1
            continue
        if ctx.pop("_in_battle", False):
            ctx["encounters"] = ctx.get("encounters", 0) + 1  # just exited a battle
        if is_dialog_open(h):
            runner.perform_action(normalize_action(navigate_ui(h, intent="confirm")),
                                  metadata={"block": block.name, "src": "safety_dialog"})
            actions += 1
            continue
        btn = block.act(h, ctx)
        if btn is None:
            break
        runner.perform_action(normalize_action(btn), metadata={"block": block.name})
        actions += 1
    else:
        aborted = "max_actions"
    # Return to the anchor tile so the next milestone starts where the spine left off.
    returned = _return_to_anchor(runner, anchor, block.name)
    restored = False
    if not returned:
        # W33 anchor guarantee (same rule as run_nav_block): settle actively, and if
        # the runner is off the anchor map or not cleanly in the overworld, RECORDED
        # restore to the block-entry state so the spine can never start stranded.
        st_end = _settle_for_block(runner)
        if entry_snap is not None and (
                st_end.map != anchor[0] or st_end.in_battle or st_end.control_mode != "free_overworld"):
            runner.load_state_bytes(entry_snap, record=True)
            restored = True
    return dict(block=block.name, ran=True, reason=aborted or "done", returned=returned,
                restored_to_entry=restored,
                anchor=list(anchor), actions=actions, encounters=ctx.get("encounters", 0),
                frames=runner.frame_idx - start_frame)


class BlockBudgetExceeded(RuntimeError):
    """Raised by run_nav_block's frame guard when a block steps past its budget."""


# Guard fallback for blocks that declare no `frames` budget, and slack over a block's
# own deadline (a well-behaved block may overshoot by one navigator step's frames).
DEFAULT_BLOCK_FRAME_BUDGET = 120_000
_BLOCK_BUDGET_SLACK = 2_048


def flee_battle(runner, max_actions: int = 40) -> bool:
    """Escape a wild battle via the cursor-verified RUN (W34).

    The old blind cycle (B, DOWN, RIGHT, A fired at an unknown UI state) is the
    pattern that CAUGHT a Poochyena on rg_020 — an A eaten by a transition landed
    on BAG and bought its way to a second party mon, voiding the single-mon
    invariant. drive_battle(mode="flee") confirms RUN only with the cursor read
    back on it and answers everything else with B, so no press can reach the bag."""
    from collection.battle_driver import drive_battle
    out = drive_battle(runner, mode="flee", src="flee_battle")
    return not runner.nav_state().in_battle


def force_fight(runner, max_rounds: int = 500) -> bool:
    """Resolve a battle by driving the menus straight from RAM. True iff we got out.

    The heatz battle machine cannot be trusted here (measured on five wedged runs):
    its party reader throws on some in-battle states ("8650 is not a valid Move")
    while our own decrypt reads the same mon fine, so it never commits a move and the
    run sits at the move menu forever. Worse, the cursor often rests on RUN, and RUN
    is refused outright in a trainer battle — mashing A alone re-runs that refusal.

    Each round steers UP+LEFT (FIGHT in the main menu, the first move in the move
    menu) and then presses A, which also advances battle text. That is enough to play
    a battle to its end, win or faint — and a faint whites us out to a Center with a
    full heal, which the grind block already treats as its heal."""
    from collection.battle_driver import drive_battle, enabled as _bv2
    if _bv2():
        drive_battle(runner, src="force_fight_v2")
        return not runner.nav_state().in_battle
    # DO NOT STEER when the lead holds a keeper move. Measured directly: ten slow A
    # presses through a level-up leave [24,45,116,52] intact, but the UP+LEFT steer
    # walks the "which move should be forgotten?" cursor onto slot 1 — Double Kick —
    # and the following A confirms the delete. A alone answers the prompts without
    # ever moving that cursor, and the battle menu already remembers FIGHT and the
    # last move used, so A-only still fights.
    try:
        from collection.playthrough.blocks.grind_evolve import read_lead, _KEEPER_MOVES
        _l = read_lead(runner)
        _keep = bool(_l and _KEEPER_MOVES & set(_l.get("moves") or ()))
    except Exception:
        _keep = False
    # B-ONLY once a keeper move is held. Measured directly on a Combusken holding
    # [24,45,116,52]: 25 B presses through the L17 level-up leave the moveset intact,
    # while ANY A in the cycle eventually answers "make room for PECK?" with yes and
    # deletes slot 1 — Double Kick. B still advances battle text, and the battle menu
    # remembers FIGHT and the last move, so the fight continues without steering.
    # Non-keeper cycle stays UP+LEFT+A. A "text-first" variant (A,A,A,UP,LEFT,A) was
    # tried on the theory that most battle frames are dialogue, and measured WORSE by
    # a wide margin on Route 104: 1 battle in 93,246 frames against 4 in 9,778. The
    # extra A presses evidently disturb move selection rather than just advancing
    # text. Keeper cycle is B-only: any A there answers the move-learn prompt and
    # deletes Double Kick.
    _cycle = ("B",) if _keep else ("UP", "LEFT", "A")
    for _ in range(max_rounds):
        if not runner.nav_state().in_battle:
            return True
        for b in _cycle:
            _nav()._hold(runner, [b], 3, "battle_fix")
            _nav()._hold(runner, [], 10, "battle_fix")
    return not runner.nav_state().in_battle


def _nav():
    from collection import navigator as nav
    return nav


def leave_battle(runner) -> None:
    """Get out of whatever battle we are in, by ANY legal means.

    RUN is refused outright in a trainer battle ("no running from a trainer
    battle!"), so mashing it — 40 actions, then again on each of twelve hop
    attempts — is how every Route 116 heal trip burned its budget while the lead
    stayed at 4 HP. Fight instead: winning is XP, and losing whites out to the
    Center, which fully restores HP and PP. Both beat arguing with the RUN menu."""
    if flee_battle(runner, max_actions=12):
        return
    if runner.nav_state().in_battle:
        import random
        from collection.collect_behaviors import _battle_one
        from collection.playthrough.blocks.grind_evolve import await_overworld
        _battle_one(runner, random.Random(0), "fight")
        await_overworld(runner)


def goto_map_safe(runner, mk, key: str, deadline: int) -> bool:
    """Bounded cross-map hop with battle-flee retry; True iff we stand on `key`.
    Twelve attempts, not two (W33, measured): a hop that starts INSIDE grass draws
    wild encounters, and a 2-try hop died to the second battle every time."""
    from collection import navigator as nav
    for _ in range(12):
        t, _, _ = nav._state(runner)
        if t is None and runner.nav_state().in_battle:
            # terrain reads None for the whole battle (gBackupMapLayout torn down) —
            # goto_map would burn its holds inside the battle screen. Leave first.
            leave_battle(runner)
            continue
        if t is not None and f"{t.map_group},{t.map_num}" == key:
            return True
        if runner.frame_idx >= deadline:
            return False
        r = nav.goto_map(runner, mk, key,
                         hop_budget=max(0, min(20_000, deadline - runner.frame_idx)))
        if r == "battle":
            leave_battle(runner)
            continue
        break
    t, _, _ = nav._state(runner)
    return t is not None and f"{t.map_group},{t.map_num}" == key


def run_nav_block(runner, block, *, mk=None, return_budget: int = 30000) -> dict:
    """W33 §2 wrapper for NAVIGATOR-driven expedition blocks (bfs_sweep, encounter_farm):
    same safety contract as run_block — precondition (free overworld, no battle/dialog,
    else skip cleanly), bounded budgets (the block's own frame budgets), forced
    anchor-return — but the block steers the runner directly through the navigator
    instead of emitting one button per act, and battles/dialogs are handled INSIDE its
    machinery (goto interrupts on battle; blocks flee and resume). The block implements
    `.name`, `.phase` and `.run(runner, mk, ctx) -> summary dict`.

    Recording: `runner.set_phase(block.phase)` stamps every frame + drops the boundary
    savestate on entry; the phase is restored to "spine" on exit NO MATTER how the block
    ends, so a block can never mis-tag spine frames."""
    from collection import navigator as nav
    if mk is None:
        mk = nav.MapKnowledge()
    start_frame = runner.frame_idx
    # Same settle/precondition seam as run_block (active + bounded, W33 blocks fix).
    # Dialog check is the VISION-validated heatz one — the raw BG0 window mask
    # false-reads on stale post-close tiles (the documented no_dialog* limitation)
    # and would veto perfectly clean states.
    st0 = _settle_for_block(runner)
    t, x, y = nav._state(runner)
    if st0.in_battle or is_dialog_open(_hstate(runner)) or t is None:
        return dict(block=block.name, ran=False, reason="precondition_not_overworld",
                    frames=runner.frame_idx - start_frame)
    if st0.control_mode != "free_overworld":
        # Movement test beats the position-poisoned control_mode — the SAME fix as
        # run_block, which W34's verification wave proved insufficient alone: the
        # nav blocks come through THIS wrapper, and its untouched check kept
        # skipping the ROUTE_104_SOUTH trainer block in all three fresh runs while
        # the player stood freely movable in Petalburg reading mode='dialogue'.
        back = None
        for d, b in (("left", "right"), ("right", "left"), ("up", "down"), ("down", "up")):
            if nav._step(runner, d):
                back = b
                break
        if back is None:
            return dict(block=block.name, ran=False, reason="precondition_not_overworld",
                        frames=runner.frame_idx - start_frame)
        nav._step(runner, back)
        t, x, y = nav._state(runner)
        if t is None:
            return dict(block=block.name, ran=False, reason="precondition_not_overworld",
                        frames=runner.frame_idx - start_frame)
    anchor = (f"{t.map_group},{t.map_num}", x, y)
    entry_snap = runner.save_state_bytes()
    runner.set_phase(block.phase)
    # W33 loop-bounding invariant, INVOCATION-layer enforcement: whatever the block's
    # internal discipline, it cannot step past its declared frame budget (+slack) —
    # the guard cuts it at the next frame boundary, the outcome is marked over_budget,
    # and the wrapper's anchor-return/phase-restore/run-continues machinery proceeds.
    # (Attempt-4 pilots: one sweep with per-leg budgets but no whole-block bound
    # legally burned ~889k frames on every starter; internal budgets alone are trust,
    # this is enforcement.)
    budget = int(getattr(block, "frames", None) or DEFAULT_BLOCK_FRAME_BUDGET)
    hard_stop = runner.frame_idx + budget + _BLOCK_BUDGET_SLACK
    orig_step = runner.step_frame

    def _guarded_step(*a, **kw):
        if runner.frame_idx >= hard_stop:
            raise BlockBudgetExceeded(
                f"block {block.name!r} exceeded its frame budget ({budget} + slack)")
        return orig_step(*a, **kw)

    over_budget = False
    summary: dict = {}
    ctx: dict = {"anchor": anchor}
    runner.step_frame = _guarded_step
    try:
        summary = block.run(runner, mk, ctx)
    except BlockBudgetExceeded as e:
        over_budget = True
        # blocks that stash a live summary in ctx keep their partial counters even
        # when the guard cuts them (W33: a guard-cut grind was losing its
        # battles_won/final_level evidence exactly when it mattered most)
        summary = dict(ctx.get("summary") or {})
        summary["error"] = repr(e)
    except Exception as e:
        # block-never-breaks-the-spine: the error is RECORDED, never propagated; the
        # anchor return + phase restore below still run.
        summary = dict(ctx.get("summary") or {})
        summary["error"] = repr(e)
    finally:
        runner.step_frame = orig_step
    returned = False
    try:
        returned = _return_to_anchor_nav(runner, mk, anchor, budget=return_budget)
    except Exception as e:
        # a second failure must not mask the first — both end up in the summary
        summary["return_error"] = repr(e)
    finally:
        runner.set_phase("spine")
    if not returned:
        # W33 anchor guarantee: a failed return can leave the runner stranded —
        # mid-battle on a far map (the RUSTBORO wedge: the next spine milestone then
        # pathed its target coords against the WRONG map's grid and wandered through
        # encounter territory for hours). Settle actively; if the runner is still off
        # the anchor MAP or not cleanly in the overworld, do a RECORDED restore to the
        # block-entry state (honest in the recording: restore frame + manifest tally).
        # A same-map clean drift is kept — spine policies handle same-map distance,
        # and restoring would needlessly discard the block's world effects (XP etc.).
        st_end = _settle_for_block(runner)
        t2, x2, y2 = nav._state(runner)
        on_anchor_map = t2 is not None and f"{t2.map_group},{t2.map_num}" == anchor[0]
        if entry_snap is not None and (
                not on_anchor_map or st_end.in_battle or st_end.control_mode != "free_overworld"):
            runner.load_state_bytes(entry_snap, record=True)
            summary["restored_to_entry"] = True
            # the restore rewinds the WORLD: a block that earned levels/flags loses
            # them here. Never silent — audits must be able to find these.
            if summary.get("levels_gained") or summary.get("battles_won") \
                    or summary.get("engaged"):
                summary["restore_discarded_progress"] = True
    # `frames` (in **summary) = the block's own work; frames_total adds settle + return.
    return dict(block=block.name, ran=True, anchor=list(anchor), returned=returned,
                over_budget=over_budget, frame_budget=budget,
                frames_total=runner.frame_idx - start_frame, **summary)


def _return_to_anchor_nav(runner, mk, anchor, *, budget: int = 30000) -> bool:
    """Cross-map anchor return for nav blocks (a sweep/farm legitimately ends on another
    map — base's porymap `find_path_action` return is same-map only)."""
    from collection import navigator as nav
    amap, ax, ay = anchor
    deadline = runner.frame_idx + budget

    def goal(t, beh):
        import numpy as np
        m = np.zeros(t.grid.shape, bool)
        if 0 <= ay + 7 < m.shape[0] and 0 <= ax + 7 < m.shape[1]:
            m[ay + 7, ax + 7] = True
        return m

    for _ in range(6):
        if runner.nav_state().in_battle:
            flee_battle(runner)
            continue
        t, x, y = nav._state(runner)
        if t is None:
            nav._hold(runner, [], 30, "block_return")
            continue
        if f"{t.map_group},{t.map_num}" == amap and (x, y) == (ax, ay):
            return True
        if runner.frame_idx >= deadline:
            break
        if f"{t.map_group},{t.map_num}" != amap:
            nav.goto_map(runner, mk, amap, hop_budget=max(2000, deadline - runner.frame_idx))
        else:
            nav.goto(runner, mk, goal, budget=max(2000, deadline - runner.frame_idx),
                     phase="block_return")
    t, x, y = nav._state(runner)
    return t is not None and f"{t.map_group},{t.map_num}" == amap and (x, y) == (ax, ay)


def _return_to_anchor(runner, anchor, name, *, max_actions: int = 360) -> bool:  # rescaled for condition-based pacing (W33 §3.5)
    amap, ax, ay = anchor
    for _ in range(max_actions):
        st = runner.state()
        if st.in_battle:
            runner.perform_action(normalize_action(handle_battle(_hstate(runner), strategy="run")),
                                  metadata={"block": name, "src": "return_battle"})
            continue
        if st.map == amap and st.x == ax and st.y == ay:
            return True
        h = _hstate(runner)
        if is_dialog_open(h):
            runner.perform_action("a", metadata={"block": name, "src": "return_dialog"})
            continue
        btn = find_path_action(h, ax, ay)
        runner.perform_action(normalize_action(btn), metadata={"block": name, "src": "return"})
    st = runner.state()
    return st.map == amap and st.x == ax and st.y == ay
