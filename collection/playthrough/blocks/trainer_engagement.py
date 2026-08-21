"""TRAINER_ENGAGEMENT expedition block (task #46): scheduled per-trainer coverage.

Owner-measured evidence: scripted policies BYPASS optional trainers (treecko fought
11/18, torchic 6/18 tracked trainers) — fleet-wide per-trainer coverage must be
scheduled, not lucky. Given target specs [{map, trainer_flag}], this block navigates
to each trainer (v1 job_trainer_hunt's manifest-object targeting: ROM template coords
refined by the live ObjectEvent record, adjacent cells PLUS 4-tile sight lines as the
goal mask), triggers the battle by sightline or talk, FIGHTs to win through the heatz
battle machine (collect_behaviors._battle_one), and VERIFIES the defeat by re-reading
the trainer's flag from the SaveBlock1 flags slice (ledger_panel addresses — flag N
lives at byte N//8, bit N%8).

Trainer flag <-> script trainer id mapping (NO assumption — verified empirically):

  flag = 0x500 (TRAINER_FLAGS_START) + trainer_id
  where trainer_id is the u16 the manifest parsed out of the object's script
  `trainerbattle` opcode 0x5C (rom_manifest.walk_maps) — i.e. the SCRIPT id, the
  one the manifest stores per object under "trainer_id".

  Empirically verified 2026-08-21 on the live emulator by loading a storyline stage
  state, running THIS block for one trainer, and diffing the full 300-byte flags
  slice before/after (four trainers, four maps — every diff showed the predicted
  flag and no other unexplained trainer-range flip):
    * CALVIN  (id 318, map 0,17 Route 102, from ROUTE_102 state): won; exactly
      0x63E == 0x500+318 flipped in the trainer range (frames 3621).
    * JAMES   (id 621, map 24,11 Petalburg Woods, from TEAM_AQUA_GRUNT_DEFEATED
      state): won; 0x76D == 0x500+621 flipped — alongside 0x50A, the pending
      grunt-script tail completing on the block's first dialog clears (that state
      is saved a few frames before the script's setflag; see the GRUNT row below).
    * WINSTON (id 136, map 0,19 Route 104 north, from ROUTE_104_NORTH state): won;
      the ONLY flag flip in the whole slice was 0x588 == 0x500+136 (frames 2759).
    * GRUNT   (id 10, map 24,11, from the pre-grunt PETALBURG_WOODS state): the
      walk toward south 104 triggered the scripted grunt battle; the block fought
      and won it and the only trainer-range flip was 0x50A == 0x500+10 — live
      proof both of the grunt mapping AND of why woods trainer blocks must be
      scheduled AFTER TEAM_AQUA_GRUNT_DEFEATED, never at PETALBURG_WOODS.
  Negative case measured the same session: KAREN (id 280) and TOMMY (id 321)
  battles LOST with the lv-12 Mudkip lead (grass-/rock-type disadvantage under the
  battle machine's move selection) — flags correctly did NOT flip, the block
  recorded engaged won=False + ended=battle_lost, and no other flag moved: the
  flag is set on VICTORY only, confirming it as the win record, not a talk record.
  Corroborating chain evidence (storyline_wm flag probes, same session): the six
  trainers the scripted chain DOES fight en route (Cindy 114, Calvin 318, Allen 333,
  Tiana 603, Rick 615, Lyle 616) are exactly the six tracked flags set by
  TEAM_AQUA_GRUNT_DEFEATED (0x572, 0x63E, 0x64D, 0x75B, 0x767, 0x768); Josh's 0x640
  appears at TRAINER_JOSH_BATTLE; Tommy 0x641 / Marc 0x73B / Roxanne 0x609 appear at
  ROXANNE_BATTLE.

Preconditions/safety: heal-check like grind_evolve — when the lead drops under
`hp_floor` of max HP the block aborts the REMAINING targets with reason `lead_hp_low`
(the spine's heal logic owns recovery). A target whose flag is already set (the spine
or an earlier run leg fought it) is skipped-with-reason, never re-fought. Wild battles
en route are FOUGHT (exp is welcome; the flag diff tells trainer battles apart from
wild ones). Post-battle watch is grind_evolve's A-ONLY await_overworld: a trainer-exp
level-up can cross an evolution threshold, and B cancels an evolution in progress.

Summary: {engaged: [{flag, won}], skipped_with_reason: [...]} + battles/frames.
`engaged` carries every TRACKED flag this block's battles actually flipped (a
sightline engagement en route to another target is an engagement, and counted), plus
won=False rows for talk-initiated trainer battles whose flag did NOT flip (a lost
battle — the block then aborts its remaining targets). Phase "battle".
"""
from __future__ import annotations

import random

import numpy as np

from collection import navigator as nav
from collection.extractors.entities import npcs
from collection.extractors.ledger_panel import EWRAM, FLAGS_BYTES, SB1_FLAGS, SB1_PTR
from collection.extractors.ram import GBAState
from collection.playthrough.blocks.grind_evolve import await_overworld, read_lead

# pokeemerald TRAINER_FLAGS_START: trainer N defeated <=> flag 0x500 + N
TRAINER_FLAG_BASE = 0x500

# The 18 owner-tracked trainer flags (W33 fleet coverage targets).
TRACKED_TRAINER_FLAGS: tuple[int, ...] = tuple(
    TRAINER_FLAG_BASE + t for t in
    (10, 114, 136, 265, 280, 318, 319, 320, 321, 333, 337, 483, 571, 603, 604,
     615, 616, 621))

# Tracked flags the SPINE itself sets in every completed run — the aqua grunt and
# Roxanne are scripted battles with no sightline trainer object in the manifest
# (grunt: cutscene battle; Roxanne: talk-only leader, trainer_type 0 so the 0x5C
# parse never runs), and Josh IS a milestone (TRAINER_JOSH_BATTLE). None of the
# three is block-schedulable: a block must never pre-fight a spine milestone's
# trainer (the milestone's policy would find him already beaten).
SPINE_TRAINER_FLAGS: dict[int, str] = {
    TRAINER_FLAG_BASE + 10: "TEAM_AQUA_GRUNT_DEFEATED",   # GRUNT (Petalburg Woods)
    TRAINER_FLAG_BASE + 320: "TRAINER_JOSH_BATTLE",       # JOSH (Rustboro Gym)
    TRAINER_FLAG_BASE + 265: "ROXANNE_BATTLE",            # ROXANNE (Rustboro Gym)
}

_ATTEMPTS = 4                      # bounded goto/talk rounds per target (trainer_hunt)
_MAX_LOSSES = 2                 # losses tolerated before the block stops trying
_RETRY_WAIT = 240                  # frames between rounds: let a wanderer move on


def read_trainer_flag(runner, flag: int) -> bool | None:
    """One SaveBlock1 flag bit (byte N//8, bit N%8 of the flags slice); None when the
    SaveBlock pointer is out of EWRAM (reset/boot — never silently False)."""
    st = GBAState.from_env(runner.env)
    sb1 = st.u32(SB1_PTR)
    if not EWRAM <= sb1 <= EWRAM + 0x40000 - (SB1_FLAGS + FLAGS_BYTES):
        return None
    b = st.bytes(sb1 + SB1_FLAGS, FLAGS_BYTES)
    return bool((b[flag // 8] >> (flag % 8)) & 1)


def read_flags_slice(runner) -> bytes | None:
    """The whole 300-byte flags slice (the empirical-mapping diff instrument)."""
    st = GBAState.from_env(runner.env)
    sb1 = st.u32(SB1_PTR)
    if not EWRAM <= sb1 <= EWRAM + 0x40000 - (SB1_FLAGS + FLAGS_BYTES):
        return None
    return st.bytes(sb1 + SB1_FLAGS, FLAGS_BYTES)


class TrainerEngagement:
    name = "trainer_engagement"
    phase = "battle"

    def __init__(self, targets: list[dict], frames: int = 60000, seed: int = 0,
                 hp_floor: float = 0.30):
        self.targets = [dict(t) for t in targets]    # [{"map": "0,17", "trainer_flag": 0x63E}]
        self.frames = frames                         # whole-block frame budget
        self.seed = seed
        self.hp_floor = hp_floor

    # ---------------------------------------------------------------- plumbing

    def _skip(self, summary, flag, key, reason) -> None:
        summary["skipped_with_reason"].append(dict(flag=flag, map=key, reason=reason))

    def _watch_set(self, runner) -> set[int]:
        """Tracked flags currently set (None-safe: unreadable slice -> empty)."""
        out = set()
        for f in self._watch:
            if read_trainer_flag(runner, f):
                out.add(f)
        return out

    def _record_flips(self, runner, summary) -> set[int]:
        """Diff the tracked-flag set against the last snapshot; every NEW flag is a
        won engagement (the flag IS the defeat record). Returns the flips."""
        now = self._watch_set(runner)
        flips = now - self._flag_state
        self._flag_state = now
        for f in sorted(flips):
            if f not in self._engaged_flags:
                self._engaged_flags.add(f)
                summary["engaged"].append(dict(flag=f, won=True))
        return flips

    def _fight(self, runner, rng, summary) -> None:
        """One battle through the heatz machine, then the evolve-safe A-only watch
        (trainer exp can evolve the lead) and a leftover-box sweep."""
        from collection.collect_behaviors import _battle_one
        _battle_one(runner, rng, "fight")
        summary["battles"] += 1
        await_overworld(runner, phase=self.phase)
        if nav._dialog_open(runner):                 # post-battle speech leftovers
            nav._clear_dialog(runner, self.phase)

    def _lead_ok(self, runner, summary) -> bool:
        """grind_evolve's heal policy: abort-with-reason instead of risking a
        whiteout mid-block; the spine's heal logic owns recovery."""
        lead = read_lead(runner)
        if lead is None:
            summary["ended"] = "no_lead"
            return False
        if lead["max_hp"] and lead["hp"] / lead["max_hp"] < self.hp_floor:
            summary["ended"] = "lead_hp_low"
            return False
        return True

    def _ensure_map(self, runner, mk, key: str, rng, summary) -> bool:
        tries = 0
        while tries < 3 and runner.frame_idx < self._deadline:
            t, _, _ = nav._state(runner)
            if t is not None and f"{t.map_group},{t.map_num}" == key:
                return True
            r = nav.goto_map(runner, mk, key,
                             hop_budget=max(2000, min(9000, self._deadline - runner.frame_idx)))
            if r == "battle":
                # a wild/LoS battle must not consume a travel try (Petalburg Woods
                # measured 2026-08-21: three grass battles ate all three rounds)
                self._fight(runner, rng, summary)
                self._record_flips(runner, summary)
                continue
            if r == "arrived":
                return True
            tries += 1
        t, _, _ = nav._state(runner)
        return t is not None and f"{t.map_group},{t.map_num}" == key

    def _walk_to(self, runner, mk, cells_fn, *, budget: int) -> str:
        """interaction._goto_adjacent's denial walker, trainer-flavored: nav.goto's
        BFS walk minus its `_unstick` A-mash. A while facing a talkable NPC re-opens
        its chatter box and wedges the walk — measured 2026-08-21 in the Rustboro
        Gym: the beaten JOSH parks on the entrance pocket's only exit tile, goto's
        unstick re-talked him on every refused step, and all four Tommy rounds died
        'stuck' at the entrance. Here refusals transient-block the cell and route
        around (bfs_sweep's denial idea), en-route boxes are B-closed sparsely, and
        a battle returns 'battle' TO THE CALLER — a sightline engagement is the
        point of this block, never fled. The goal mask is rebuilt from cells_fn on
        every replan, so a wandering trainer is chased at its live position.
        Returns 'arrived' | 'battle' | 'stuck' | 'budget'."""
        from collection.playthrough.blocks.interaction import (
            _close_dialog, _dialog_open_live)
        blocked: dict[tuple[int, int], int] = {}
        misses = 0
        resets = 0
        it = 0
        start = runner.frame_idx
        while runner.frame_idx - start < budget:
            it += 1
            if runner.nav_state().in_battle:
                return "battle"
            if it % 8 == 1 and _dialog_open_live(runner):
                _close_dialog(runner, self.phase)
                continue
            t, x, y = nav._state(runner)
            if t is None:
                nav._hold(runner, [], 30, self.phase)
                continue
            walk = ((t.grid >> 10) & 3) == 0
            goals = np.zeros(t.grid.shape, bool)
            for cx, cy in cells_fn():
                if 0 <= cy + 7 < goals.shape[0] and 0 <= cx + 7 < goals.shape[1]:
                    goals[cy + 7, cx + 7] = True
            goals &= walk
            bx, by = x + 7, y + 7
            if not (0 <= by < goals.shape[0] and 0 <= bx < goals.shape[1]):
                nav._hold(runner, [], 30, self.phase)
                continue
            if goals[by, bx]:
                return "arrived"
            blocked = {c: f for c, f in blocked.items() if runner.frame_idx - f < 600}
            for cx, cy in blocked:
                if 0 <= cy < walk.shape[0] and 0 <= cx < walk.shape[1]:
                    walk[cy, cx] = False
            step = nav._bfs_step(walk, (bx, by), goals,
                                 elev=((t.grid >> 12) & 0xF).astype(np.uint8),
                                 beh=mk.behaviors(t))
            if step is None:
                if blocked and resets < 4:               # dead-ended by our own blocks
                    blocked.clear()
                    resets += 1
                    nav._hold(runner, [], 60, self.phase)
                    continue
                return "stuck"
            if nav._step(runner, nav.DIRS[step]):
                misses = 0
            else:
                misses += 1
                blocked[(bx + step[0], by + step[1])] = runner.frame_idx
                if misses >= 8:
                    return "stuck"
                nav._hold(runner, [], 10, self.phase)
        return "budget"

    # ------------------------------------------------------------------ engage

    def _engage_one(self, runner, mk, rng, key: str, flag: int, summary) -> None:
        tid = flag - TRAINER_FLAG_BASE
        pre = read_trainer_flag(runner, flag)
        if pre is None:
            self._skip(summary, flag, key, "SaveBlock flags unreadable")
            return
        if pre:
            self._skip(summary, flag, key, "flag already set (trainer already defeated)")
            return
        objs = [o for o in mk.objects.get(key, []) if o.get("trainer_id") == tid]
        if not objs:
            self._skip(summary, flag, key,
                       f"no manifest trainer object with trainer_id {tid} on map")
            return
        if not self._ensure_map(runner, mk, key, rng, summary):
            self._skip(summary, flag, key, "map unreached")
            return
        if read_trainer_flag(runner, flag):          # an en-route LoS battle was THIS one
            return                                   # (already recorded by _record_flips)

        def near(o, e):
            return e if e is not None and abs(e.x - o["x"]) + abs(e.y - o["y"]) <= 8 else None

        def pos(o):
            """Template coords refined by the live ObjectEvent record (they wander;
            connected maps' local_ids COLLIDE, so a live match must sit near its
            template — the job_trainer_hunt lesson)."""
            live = {e.local_id: e for e in npcs(GBAState.snapshot(runner.env))
                    if abs(e.x) < 200}
            le = near(o, live.get(o["local_id"]))
            return (le.x, le.y) if le is not None else (o["x"], o["y"])

        for attempt in range(_ATTEMPTS):
            if runner.frame_idx >= self._deadline:
                self._skip(summary, flag, key, "block frame budget expired")
                return
            if not self._lead_ok(runner, summary):
                self._skip(summary, flag, key, summary["ended"])
                return
            t, x, y = nav._state(runner)
            if t is None:
                nav._hold(runner, [], 30, self.phase)
                continue
            o = min(objs, key=lambda c: abs(pos(c)[0] - x) + abs(pos(c)[1] - y))

            def sight(o=o):                          # adjacent (talk) + 4-tile sight
                px, py = pos(o)                      # lines (gaze engages through
                return [(px + dx * r_, py + dy * r_)  # unwalkable neighbours)
                        for dx, dy in nav.DIRS for r_ in (1, 2, 3, 4)]

            def adj(o=o):
                px, py = pos(o)
                return [(px + dx, py + dy) for dx, dy in nav.DIRS]

            r = self._walk_to(runner, mk, sight,
                              budget=min(12000, max(2000, self._deadline - runner.frame_idx)))
            if r == "battle":                        # LoS engagement en route — or wild
                self._fight(runner, rng, summary)
                self._record_flips(runner, summary)
                if flag in self._engaged_flags:
                    return
                continue                             # wild/other trainer: re-target
            if r != "arrived":
                if attempt < _ATTEMPTS - 1:
                    nav._hold(runner, [], _RETRY_WAIT, self.phase)   # blocker wanders off
                continue
            nav._hold(runner, [], 45, self.phase)    # in the sightline now: give the
            if nav._in_battle(runner):               # gaze a beat to fire (free talk)
                self._fight(runner, rng, summary)
                self._record_flips(runner, summary)
                if flag in self._engaged_flags:
                    return
                continue
            t, x, y = nav._state(runner)
            tx, ty = pos(o)
            if abs(tx - x) + abs(ty - y) != 1:
                # sightline cell without a gaze engagement (the trainer faces another
                # way — Calvin, measured 2026-08-21): close to a TALK-adjacent cell.
                r2 = self._walk_to(runner, mk, adj,
                                   budget=min(6000, max(1500, self._deadline - runner.frame_idx)))
                if r2 == "battle":                   # walking INTO the line fired it
                    self._fight(runner, rng, summary)
                    self._record_flips(runner, summary)
                    if flag in self._engaged_flags:
                        return
                    continue
                t, x, y = nav._state(runner)
                tx, ty = pos(o)
                if abs(tx - x) + abs(ty - y) != 1:   # wanderer drifted mid-walk:
                    continue                         # re-target at the live position
            from collection.collect_behaviors import _tap_turn
            d = {(1, 0): "RIGHT", (-1, 0): "LEFT", (0, 1): "DOWN", (0, -1): "UP"}.get(
                (max(-1, min(1, tx - x)), max(-1, min(1, ty - y))), "UP")
            _tap_turn(runner, d)
            runner.perform_action("A", speed="normal", record_end_state=False,
                                  metadata={"block": self.name})
            nav._hold(runner, [], 60, self.phase)
            for _ in range(30):                      # intro text -> battle
                if nav._in_battle(runner):
                    break
                nav._hold(runner, ["A"], 4, self.phase)
                nav._hold(runner, [], 14, self.phase)
            if nav._in_battle(runner):               # talk-initiated: this IS the trainer
                self._fight(runner, rng, summary)
                flips = self._record_flips(runner, summary)
                if flag not in self._engaged_flags:
                    # trainer battle, flag did NOT flip: we lost (whiteout heals and
                    # teleports; the flag is the ground truth) — engaged, not won
                    self._engaged_flags.add(flag)
                    summary["engaged"].append(dict(flag=flag, won=False))
                    summary["ended"] = "battle_lost"
                return
            nav._clear_dialog(runner, self.phase)    # chatter without a battle
            if read_trainer_flag(runner, flag):      # (paranoia: scripted set)
                self._record_flips(runner, summary)
                return
        self._skip(summary, flag, key,
                   f"no battle after {_ATTEMPTS} goto/talk rounds")

    # ------------------------------------------------------------------- entry


    def run(self, runner, mk, ctx) -> dict:
        rng = random.Random(self.seed)
        summary = dict(engaged=[], skipped_with_reason=[], battles=0, ended="done",
                       frames=0)
        f0 = runner.frame_idx
        self._deadline = f0 + self.frames
        self._watch = set(TRACKED_TRAINER_FLAGS) | {
            int(t["trainer_flag"]) for t in self.targets}
        self._flag_state = self._watch_set(runner)
        self._engaged_flags: set[int] = set()
        for tgt in self.targets:
            key, flag = str(tgt["map"]), int(tgt["trainer_flag"])
            if runner.frame_idx >= self._deadline:
                self._skip(summary, flag, key, "block frame budget expired")
                continue
            if summary["ended"] in ("lead_hp_low", "no_lead"):
                self._skip(summary, flag, key, f"aborted: {summary['ended']}")
                continue
            # A LOSS IS NOT AN ABORT (W33 proof, measured): a lost trainer battle
            # whites out to the Center, which fully heals HP and PP — the very
            # resource the next target needs. Aborting the rest threw away two of
            # four scheduled trainers (and their XP) over one unwinnable fight.
            # Bounded so an out-of-depth schedule can't spend the block losing.
            if summary["ended"] == "battle_lost":
                if len([e for e in summary["engaged"] if not e["won"]]) >= _MAX_LOSSES:
                    self._skip(summary, flag, key, "aborted: loss budget spent")
                    continue
                summary["ended"] = "done"           # whiteout healed us: carry on
            self._engage_one(runner, mk, rng, key, flag, summary)
        summary["frames"] = runner.frame_idx - f0
        return summary
