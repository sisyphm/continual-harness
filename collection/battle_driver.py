"""RAM-cursor-aware battle driver (battle v2, W34).

The blind steer (UP, LEFT, A, ...) fired fixed sequences at an unknown UI state, and
every catastrophic battle outcome this project has logged traces to exactly that
blindness: an A landing on POKeMON opened the party stack (the 50,000-frame full-HP
deadlock, dead_2156), stray presses walked the forget-cursor onto Double Kick
(wedge9's DK loss), and an A confirmed RUN into "no running from a TRAINER battle!"
(exp_019 burned a milestone on it). Three attempts to make the steer safer with text
gates or B prefixes each regressed a gym fight — see the DO-NOT-REWRITE block in
spine._press_best_move — because dialog text is not a reliable menu signal.

This driver replaces guesswork with the game's own truth: the player battle
controller's function pointer says exactly when a menu awaits input, and the cursor
bytes say exactly where the cursor is. Press A only when the right menu is up AND the
cursor is verified on the target; answer everything else with B, which advances text
and declines prompts but never confirms anything. One press per loop iteration, state
re-read before every press, so a press eaten by a screen transition self-corrects
instead of desyncing the whole sequence (the trap that made blind escapes permanent).

Discriminators calibrated LIVE on the retail ROM (dead_2156 fixture 00353500, W34):

    gBattlerControllerFuncs[0] == 0x08057589   action menu awaits input
                                  (stable across 600 idle frames at "What will X do?")
    gBattlerControllerFuncs[0] == 0x08057BFD   move menu awaits input
    anything else                              busy: text, animation, or a submenu
    gActionSelectionCursor[0]                  action cursor, live readback verified
                                               (RIGHT: 2->3, LEFT: 3->2)
    gMoveSelectionCursor[0]                    move cursor, live readback verified
                                               (RIGHT: 0->1, LEFT: 1->0)

Symbol addresses from references/pokeemerald/pokeemerald.map (the matching build);
the two READY function-pointer values are empirical because the map strips statics.

Presses route through runner.perform_action, so the nickname guard, the
party/summary trap guard (all buttons -> B) and move_keeper's forget-prompt
resolution all apply unchanged. SINGLES ONLY: doubles add a target-select controller
state this driver treats as busy, and policy excludes doubles (1-mon party).
"""

from __future__ import annotations

import os

from collection.extractors.ledger_panel import CB2_ADDR

G_ACTION_CURSOR = 0x020244AC        # gActionSelectionCursor[4] (u8)
G_MOVE_CURSOR = 0x020244B0          # gMoveSelectionCursor[4] (u8)
G_CTRL_FUNCS = 0x03005D60           # gBattlerControllerFuncs[4] (fn ptr)
CTRL_ACTION_READY = 0x08057589      # HandleInputChooseAction (thumb)
CTRL_MOVE_READY = 0x08057BFD        # HandleInputChooseMove (thumb)

# The action menu is a 2x2 grid: FIGHT(0) BAG(1) / POKeMON(2) RUN(3). One corrective
# press per iteration, toward the target; the re-read next iteration verifies it
# landed. Neither route ever passes THROUGH BAG's confirm — a stray A on BAG is how
# rg_020 caught a Poochyena (blind flee cycle, W34 wave 1) and broke the single-mon
# party invariant the whole trainer policy rests on.
_ACTION_STEP = {1: "LEFT", 2: "UP", 3: "UP"}          # toward FIGHT(0)
_ACTION_STEP_RUN = {0: "DOWN", 1: "DOWN", 2: "RIGHT"}  # toward RUN(3)


def _battle_lead_foe(runner):
    """(me, foe) from the BATTLE structure, or (None, None). The party reader throws
    mid-battle on some states ('... is not a valid Move'); the battle read does not."""
    try:
        from collection.extractors.ledger_panel import _battle_mon
        from collection.extractors.ram import GBAState
        st = GBAState(env=runner.env)
        return _battle_mon(st, 0), _battle_mon(st, 1)
    except Exception:
        return None, None


def _best_move_slot(runner) -> int:
    me, foe = _battle_lead_foe(runner)
    try:
        from collection.move_data import best_slot
        i = best_slot((me or {}).get("moves") or [], (me or {}).get("pp") or [],
                      (foe or {}).get("species"))
    except Exception:
        i = 0
    return 0 if i is None else i


def drive_battle(runner, *, mode: str = "fight", budget_frames: int = 120_000,
                 src: str = "battle_v2") -> dict:
    """Play the current battle to its end. Returns an outcome dict:

        result   win | whiteout | fled | ended | timeout | not_in_battle
        frames   emulator frames consumed
        presses  {"A": n, "A_busy": n, "B": n, "cursor": n}
        moves    move-menu confirms per slot index

    mode="fight": cursor-verified FIGHT + best move every round.
    mode="flee" (wild only — callers gate on _is_trainer_battle): cursor-verified
    RUN; the move menu gets B (back out), busy states get B ONLY — a flee never
    needs a stray A, and stray A's are how the blind flee cycle bought a Poke Ball
    and caught a second party mon (rg_020, W34). "Couldn't escape!" just returns
    the action menu, so RUN retries naturally; after 8 refused confirms the driver
    falls back to fighting, which also ends the battle.

    'ended' = battle over but neither HP was seen at zero (both reads are
    best-effort); 'timeout' = frame budget exhausted while still in battle — the
    caller's stall machinery owns what happens next. Never opens the party menu.
    """
    from collection import navigator as _nav
    from collection.extractors.ram import GBAState

    f0 = runner.frame_idx
    presses = {"A": 0, "A_busy": 0, "B": 0, "cursor": 0}
    moves: dict[int, int] = {}
    foe_zero = me_zero = False
    fleeing = mode == "flee"
    run_confirms = 0

    if not runner.nav_state().in_battle:
        return {"result": "not_in_battle", "frames": 0, "presses": presses, "moves": moves}

    while runner.frame_idx - f0 < budget_frames:
        if not runner.nav_state().in_battle:
            break
        st = GBAState(env=runner.env)
        ctrl0 = st.u32(G_CTRL_FUNCS)

        if ctrl0 == CTRL_ACTION_READY:
            # A ready-state is the cheap moment to sample HP for outcome classing.
            me, foe = _battle_lead_foe(runner)
            if foe and foe.get("hp") == 0:
                foe_zero = True
            if me and me.get("hp") == 0:
                me_zero = True
            acur = st.u8(G_ACTION_CURSOR)
            if fleeing and run_confirms >= 8:
                fleeing = False           # escape keeps failing: win it instead
            target_pos = 3 if fleeing else 0
            if acur == target_pos:
                runner.perform_action("A", metadata={"src": src}, record_end_state=False)
                presses["A"] += 1
                if fleeing:
                    run_confirms += 1
            else:
                step = (_ACTION_STEP_RUN if fleeing else _ACTION_STEP).get(acur, "UP")
                runner.perform_action(step, metadata={"src": src}, record_end_state=False)
                presses["cursor"] += 1
        elif ctrl0 == CTRL_MOVE_READY:
            if fleeing:
                # Flee wants the ACTION menu; B backs out of the move list.
                runner.perform_action("B", metadata={"src": src}, record_end_state=False)
                presses["B"] += 1
                continue
            target = _best_move_slot(runner)
            mcur = st.u8(G_MOVE_CURSOR)
            if mcur == target:
                runner.perform_action("A", metadata={"src": src}, record_end_state=False)
                presses["A"] += 1
                moves[target] = moves.get(target, 0) + 1
            else:
                # 2x2 grid; fix the column first, then the row.
                if (mcur & 1) != (target & 1):
                    step = "RIGHT" if (target & 1) else "LEFT"
                else:
                    step = "DOWN" if (target >> 1) else "UP"
                runner.perform_action(step, metadata={"src": src}, record_end_state=False)
                presses["cursor"] += 1
        else:
            # Busy: text, animation, level-up, or a submenu. Mostly B — it advances
            # ordinary battle text and declines prompts without ever confirming — but
            # the END-of-battle text after a trainer win only answers to A (measured
            # on the wedge9 fixture: Roxanne's defeat speech ignored 8 straight B
            # presses, ctrl0 pinned at 080597b5, then advanced on the first A). So
            # every 4th busy press is an A. That A is safe by construction: the
            # nickname guard, the party/summary trap guard and move_keeper all sit
            # inside perform_action, and the ready-state branches above never leave
            # the action cursor anywhere but FIGHT — so an A racing a menu-open can
            # at worst enter the move menu, which the next iteration handles.
            busy = presses["B"] + presses["A_busy"]
            # Flee mode is B-ONLY: nothing in a wild flee ever needs A ("Got away
            # safely!" and "Couldn't escape!" both advance on B), and the periodic
            # A exists solely for trainer-END text, which a flee never reaches.
            key = "A" if (not fleeing and busy % 4 == 3) else "B"
            runner.perform_action(key, metadata={"src": src}, record_end_state=False)
            presses["A_busy" if key == "A" else "B"] += 1
            _nav._hold(runner, [], 10, src)

    in_b = runner.nav_state().in_battle
    if in_b:
        result = "timeout"
    else:
        # The battle can end without ever passing another ready-state (the killing
        # move's text rolls straight out), so the in-fight samples miss the final
        # HPs. gBattleMons persists stale after the exit — one last read closes it.
        me, foe = _battle_lead_foe(runner)
        if foe and foe.get("hp") == 0:
            foe_zero = True
        if me and me.get("hp") == 0:
            me_zero = True
        if me_zero:
            result = "whiteout"
        elif foe_zero:
            result = "win"
        elif mode == "flee":
            result = "fled"
        else:
            result = "ended"
    return {"result": result, "frames": runner.frame_idx - f0,
            "presses": presses, "moves": moves}


def enabled() -> bool:
    """Battle v2 is opt-in until it has beaten the blind steer across a full sweep."""
    return os.environ.get("W33_BATTLE_V2") == "1"
