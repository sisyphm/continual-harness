"""Never let a level-up prompt delete the move that wins the gym.

Combusken learns Double Kick at L16 and Peck at L17 -- both inside the Rustboro gym
run -- and its moveset is already full, so Emerald asks:

    "Delete a move to make room for PECK?"

The battle steer (UP, LEFT, A) that selects moves walks that same cursor onto slot 0
and its next A confirms the delete. Reconstructed from exp_019's ledger, frame by frame:

  frame 673,616  L16  (10,45,116,52) -> (24,45,116,52)   learned Double Kick
  frame 711,265  L17  (24,45,116,52) -> (64,45,116,52)   traded it away for Peck
  frame 717,903                                          arrives at Roxanne

It then stood on Roxanne's doorstep holding Peck and Ember, both resisted 0.5x by rock,
with no super-effective move left in the set. Double Kick is FIGHTING: 2x on her Geodude
and Nosepass. It is the win condition, and the run threw it away 6,600 frames early.

A keeper already existed inside `_battle_protected`, but that only guards battles the
spine drives itself. exp_019 hit L17 during a `trainer_engagement` battle, and nine
different blocks drive fights, so the guard has to live where every button goes --
the same choke point as nickname_guard and catch_guard.

DECLINE, DON'T REDIRECT. The obvious repair -- accept the swap, then walk the forget
cursor off slot 0 onto a junk move -- does not work, and I measured it losing the move
on this exact fixture: after the confirming A the move list has not rendered yet, so
the DOWN is swallowed and the next A deletes whatever slot 0 holds, which is Double
Kick. Declining is what actually survives, replayed against savestate 00711000:

    B            "Delete a move to make room for PECK?" dismissed
    A            "Stop learning PECK?"
    A A A        "COMBUSKEN did not learn PECK."   battle resumes, moves intact

The run gives up Peck, which costs nothing -- it is 0.5x into rock, the only fight that
matters before the first badge -- and keeps [24,45,116,52].

An earlier keeper reported declining as unreliable. That was true of a B-mash competing
with the battle steer, where a mistimed press bounces between the two boxes until the
steer's own A confirms the delete. It is not true here: this guard OWNS the exchange
(re-entrancy flag in direct_runner) and presses in order, so both boxes get exactly the
answer they are waiting for.
"""

from __future__ import annotations

# Double Kick. The only super-effective answer any starter line carries into Roxanne
# within this corpus's scope (start -> first badge); nothing else here is worth keeping
# a move slot for.
KEEP_MOVE = 24

_PROMPT = "delete a move to make"
_KEEP_NAME = "double kick"
# Callbacks the forget-prompt can be up under, sampled live on the L17 fixture: the
# prompt is NOT raised under battle-main. Gating on 0x08038421 alone looked right and
# silently disabled the whole guard -- the fixture lost Double Kick again with the
# keeper "installed". 0x081BFAB5 is the summary screen the learn flow opens.
# 0x0813E3A5 is the EVOLUTION scene (measured live, W34): Combusken's on-evolution
# Double Kick learn prompt runs under it, and the keeper never looked there — an
# L16 Combusken walked out of its own evolution still holding Scratch.
_PROMPT_CB2 = frozenset((0x08038421, 0x081BFAB5, 0x081BFAE5, 0x0813E3A5))


def _mon(runner):
    """The BATTLE structure, not the party: read_lead decrypts the party mon and throws
    mid-battle on some states, returning None at exactly the level-up we watch for."""
    try:
        from collection.extractors.ledger_panel import _battle_mon
        from collection.extractors.ram import GBAState
        return _battle_mon(GBAState(env=runner.env), 0)
    except Exception:
        return None


def swap_prompt_open(runner) -> bool:
    """True when the forget-a-move prompt is up and KEEP_MOVE is what's at risk.

    ORDER THESE BY COST. This runs before EVERY action, and the dialog read is not
    cheap: measured on this box, 3.4 ms against 0.0025 ms for a callback2 read -- about
    1,370x. Reading dialog first cost roughly 3.4 ms x every action of every milestone,
    which pushed wall-bounded milestones over budget and regressed the sweep on its own
    (TEAM_AQUA_GRUNT_DEFEATED -> ROUTE_104_NORTH: 79 actions and a pass became 40 and a
    timeout, with the spine changes bisected out).

    So: the battle-struct read first (0.0155 ms, and it rules out every mon with nothing
    left to lose), then one u32 to rule out the overworld, and only then the text. The
    expensive branch is live only while the keeper is still known and a battle-family
    screen is up -- exactly the window where the prompt can appear.
    """
    try:
        from collection.extractors.ledger_panel import CB2_ADDR
        from collection.extractors.ram import GBAState
        if GBAState(env=runner.env).u32(CB2_ADDR) not in _PROMPT_CB2:
            return False
        held = KEEP_MOVE in set((_mon(runner) or {}).get("moves") or ())
        from collection.heatz_adapter import _read_dialog_text
        t = (_read_dialog_text(runner.env) or "").lower()
        if _PROMPT not in t:
            return False
        # PROTECT: the keeper is held and something wants its slot. ACQUIRE (W34):
        # the keeper is the INCOMING move — Combusken's on-evolution Double Kick
        # with four junk moves held. The old held-only gate stood down exactly
        # then, and blind A's declined the learn.
        return held or _KEEP_NAME in t
    except Exception:
        return False


def _pending(runner) -> bool:
    """Either box of the decline exchange is still on screen."""
    try:
        from collection.heatz_adapter import _read_dialog_text
        t = (_read_dialog_text(runner.env) or "").lower()
        return _PROMPT in t or "stop learning" in t
    except Exception:
        return False


def resolve(runner) -> bool:
    """Decline the swap so KEEP_MOVE survives. True if it is still known afterwards.

    Runs the whole exchange itself rather than vetoing a single button: there is no one
    press that makes this safe, and leaving either box on screen hands it back to a
    caller whose next A deletes the move.
    """
    from collection import navigator as _nav

    def press(key):
        runner.perform_action(key, metadata={"src": "move_keeper"})
        _nav._hold(runner, [], 40, "move_keeper")

    held = KEEP_MOVE in set((_mon(runner) or {}).get("moves") or ())
    if not held:
        # ACQUIRE (W34): the prompt offers Double Kick and every held move is junk.
        # Accept, walk the forget-cursor off slot 0 onto slot 1, confirm — the same
        # redirect _battle_protected proved for the L17 Peck case, inverted.
        for _ in range(4):
            if KEEP_MOVE in set((_mon(runner) or {}).get("moves") or ()):
                break
            for k in ("A", "DOWN", "A", "A"):
                press(k)
            if not _pending(runner):
                break
        return KEEP_MOVE in set((_mon(runner) or {}).get("moves") or ())

    press("B")                            # "Delete a move to make room for X?" -> No
    for _ in range(12):                   # "Stop learning X?" -> Yes, then let it read out
        if not _pending(runner) or not runner.nav_state().in_battle:
            break
        press("A")
    return KEEP_MOVE in set((_mon(runner) or {}).get("moves") or ())
