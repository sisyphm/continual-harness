"""Continuous playthrough director: one session, all milestones, no state loads.

W33 §13 solve-then-record: every SPINE milestone is first solved UNRECORDED (recorder
+ sink detached, capture mode armed) with jittered savestate retries; the successful
attempt's exact per-frame button schedule is then replayed UNDER recording from the
same savestate and verified against the dry outcome (ledger-panel field vector +
(map,x,y) — divergence raises DeterminismError). Recorder streams stay append-only
and contain ONLY the successful trajectory; every recorded milestone doubles as a
continuous determinism check.

NOTE the block asymmetry: life/expedition BLOCKS still run RECORDED directly, as
before (they are robust by contract — bounded, battle/dialog-safe, anchor-returning —
and their summaries/phase savestates are expected downstream). Solve-then-record
applies to spine milestones only.

W33 §14 persona layer: per-run persona = {seed, tie_break, tic_rate, jitter}
(manifest-recorded) drives (1) seeded BFS tie-breaking in the navigator (route
diversity at zero correctness cost), (2) always-on milestone-start jitter (K0 idle
frames, recorded — they ARE the recording's content), (3) seeded micro-behavior tics
between spine actions (captured + replayed like everything else).
"""
from __future__ import annotations
import json
import random
import time
from pathlib import Path

from collection.catalog import MILESTONE_ORDER, discover_heatz_events
from collection.direct_runner import DirectEmulatorRunner
from collection.recorder import ChunkRecorder
from collection.world_model_sink import WorldModelSink
import collection.collect_events as ce
from collection.playthrough.spine import run_milestone


class DeterminismError(RuntimeError):
    """A recorded solve-then-record replay diverged from its dry solution (W33 §13)."""


# W33 §14.4 persona defaults; every field is overridable via run_playthrough(persona=...).
#   seed      — root of every persona-derived rng (tie-break, jitter, tics)
#   tie_break — seeded BFS equal-cost tie-breaking in the navigator (bool)
#   tic_rate  — per-action probability of a micro-behavior tic between spine actions
#   jitter    — max K0 idle frames prepended at every milestone start (always-on jitter)
DEFAULT_PERSONA = {"seed": 0, "tie_break": True, "tic_rate": 0.02, "jitter": 30}

# Frame-alignment shift added per retry attempt on top of K0 (§13: "K varies per
# attempt" — the fish-block decorrelation trick, promoted).
JITTER_PER_ATTEMPT = 17

_FACINGS = ("UP", "DOWN", "LEFT", "RIGHT")


def build_persona(seed: int, persona: dict | None = None) -> dict:
    """Materialize the run persona: defaults + overrides; seed falls back to the run seed."""
    p = {**DEFAULT_PERSONA, **(persona or {})}
    if persona is None or "seed" not in persona:
        p["seed"] = seed
    return p


def make_tic_fn(runner, rng: random.Random, rate: float):
    """W33 §14.3 micro-behavior noise: a persona-seeded closure for spine.run_milestone's
    `tic_fn` hook. With probability `rate` per action it emits a recorded human tic —
    a short pause (8-20 idle frames) or a facing flick (4-frame tap-turn sideways and
    back, the idle block's proven turn-in-place cadence). Bounded: never during battle
    or dialog (nav control mode + the BG0 window-mask check; a false-positive dialog
    read merely skips a tic), and never inside the Emerald clock-set screen (verified
    blind spot, W33 clock fix: hand-set mode reads free_overworld with no BG0 window,
    so only the visual clock predicate can veto there). The tic runs through
    step_frame, so during a DRY attempt it lands in the capture log and is replayed
    like everything else."""
    from collection import navigator as nav
    from collection.heatz_adapter import _visual_clock_ui

    def tic() -> None:
        if rate <= 0 or rng.random() >= rate:
            return
        n = runner.nav_state()
        if n.in_battle or n.control_mode != "free_overworld" or nav._dialog_open(runner):
            return
        if _visual_clock_ui(getattr(runner, "env", None)):
            return
        if rng.random() < 0.5:
            for _ in range(rng.randint(8, 20)):
                runner.step_frame([], phase="tic", metadata={"src": "tic", "kind": "pause"})
        else:
            cur = runner.facing if runner.facing in _FACINGS else "DOWN"
            side = rng.choice([d for d in _FACINGS if d != cur])
            for d in (side, cur):
                for _ in range(4):                 # < 8-frame turn window: turn, never step
                    runner.step_frame([d], phase="tic", metadata={"src": "tic", "kind": "turn"})
                for _ in range(8):
                    runner.step_frame([], phase="tic", metadata={"src": "tic", "kind": "turn"})

    return tic


def _outcome_vector(runner) -> dict:
    """The §13 verification vector: full ledger-panel field row + (map, x, y)."""
    from collection.extractors.ledger_panel import read_ledger
    from collection.extractors.ram import GBAState

    n = runner.nav_state()
    return {"pos": (n.map, n.x, n.y), "ledger": read_ledger(GBAState.snapshot(runner.env))}


def _vector_mismatch(dry: dict, rec: dict) -> str | None:
    """First mismatching component name between two outcome vectors, or None."""
    import numpy as np

    if dry["pos"] != rec["pos"]:
        return f"pos dry={dry['pos']} recorded={rec['pos']}"
    for name, va in dry["ledger"].items():
        vb = rec["ledger"][name]
        eq = np.array_equal(va, vb) if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray) else va == vb
        if not eq:
            return f"ledger[{name}] dry={va!r} recorded={vb!r}"
    return None


def solve_then_record_milestone(
    runner,
    *,
    event_id: str,
    policy_dir: str,
    expected_state,
    postcondition: str,
    start_money: int,
    max_actions: int,
    starter: str,
    persona: dict,
    max_attempts: int = 5,
    fail_injector=None,
    corrupt_replay=None,
) -> dict:
    """W33 §13 per-milestone loop on an already-initialized runner:

    (a) snapshot state_bytes at milestone start;
    (b) DRY attempt — recorder + frame_hook detached, capture mode armed — prepending
        K = K0 + attempt * 17 persona-seeded idle jitter frames (K0 always-on, §14.2);
    (c) dry failure: frame-neutral snapshot restore, retry with shifted jitter;
    (d) dry success: restore, REPLAY the captured schedule frame-by-frame under
        recording, verify the recorded outcome vector (ledger fields + (map,x,y))
        against the dry one — mismatch raises DeterminismError naming the milestone;
    (e) attempts exhausted: result flagged failed_persistent (policy defect — the
        pilot gate's signal; the director aborts the run on it).

    Returns run_milestone's result dict + {"retry": {attempts, jitters,
    dry_frames_wasted, dry_frames_solve, replay_frames}}.

    Test-only hooks (production None): `fail_injector(attempt)->kwargs-overrides` for
    deterministic dry failures, `corrupt_replay(log)->log` for determinism-path tests.

    With no recorder attached (record=False smoke runs) the retry/jitter machinery
    still runs, but the successful dry attempt IS the run — no replay, no verify.
    """
    snap = runner.save_state_bytes()
    assert snap is not None, "cannot snapshot for solve-then-record"
    f0, facing0 = runner.frame_idx, runner.facing
    jr = random.Random(f"{persona['seed']}:{event_id}:jitter")
    k0 = jr.randint(0, int(persona.get("jitter", 0)))
    tic_fn = make_tic_fn(runner, random.Random(f"{persona['seed']}:{event_id}:tic"),
                         float(persona.get("tic_rate", 0.0)))
    telemetry: dict = {"attempts": 0, "jitters": [], "dry_frames_wasted": 0,
                       "dry_frames_solve": 0, "replay_frames": 0}
    result: dict = {}
    # W33 DIRECT-RECORD (owner 08-25: "collection has to be clean: full speed, best
    # policy, stalls detected fast"). The dry+replay 2x tax was scaffolding for a
    # fragile policy; with the fix stack + trainer-routed plan the first attempt
    # succeeds ~99% of the time, so: record LIVE, single execution. On the rare
    # failure, restore the milestone snapshot as a LOGGED RECORDED restore (gate
    # provenance already accepts these) and append the failed span to the manifest's
    # blemish_spans -- the packaging stage excises those spans offline, which keeps
    # the TRAINING corpus clean without recorder surgery. Determinism stays proven by
    # sampled replay spot-checks in the verify battery instead of per-milestone 2x.
    import os as _os
    if _os.environ.get("W33_DIRECT_RECORD") == "1" and runner.recorder is not None:
        if _os.environ.get("W33_DIAG") == "1":
            max_attempts = 1                      # diagnostic: first verdict is THE verdict
        for attempt in range(max_attempts):
            k = k0 + attempt * JITTER_PER_ATTEMPT
            telemetry["attempts"] += 1
            telemetry["jitters"].append(k)
            fa = runner.frame_idx
            for _ in range(k):
                runner.step_frame([], phase="milestone_jitter",
                                  metadata={"event_id": event_id, "attempt": attempt})
            result = run_milestone(
                runner, event_id=event_id, policy_dir=policy_dir,
                expected_state=expected_state, postcondition=postcondition,
                start_money=start_money, starter=starter, tic_fn=tic_fn,
                max_actions=max_actions)
            if result["validation"] in ("passed", "skipped"):
                break
            span = [fa, runner.frame_idx]
            runner.load_state_bytes(snap, record=True)      # logged, recorded restore
            runner.facing = facing0
            runner.recorder.manifest.setdefault("blemish_spans", []).append(
                {"event_id": event_id, "attempt": attempt, "span": span,
                 "reason": result.get("failure_reason")})
            runner.recorder._write_manifest()
        else:
            result["failed_persistent"] = True
        result["retry"] = telemetry
        return result

    for attempt in range(max_attempts):
        k = k0 + attempt * JITTER_PER_ATTEMPT
        telemetry["attempts"] += 1
        telemetry["jitters"].append(k)
        overrides = fail_injector(attempt) if fail_injector is not None else {}
        rec, hook = runner.recorder, runner.frame_hook
        runner.recorder = None
        runner.frame_hook = None
        runner.capture_log = []
        try:
            for _ in range(k):
                runner.step_frame([], phase="milestone_jitter",
                                  metadata={"event_id": event_id, "attempt": attempt, "k": k})
            result = run_milestone(
                runner, event_id=event_id, policy_dir=policy_dir,
                expected_state=expected_state, postcondition=postcondition,
                start_money=start_money, starter=starter, tic_fn=tic_fn,
                **{"max_actions": max_actions, **overrides})
        finally:
            log = runner.capture_log
            runner.capture_log = None
            runner.recorder = rec
            runner.frame_hook = hook

        if result["validation"] in ("passed", "skipped"):
            dry_frames = runner.frame_idx - f0
            telemetry["dry_frames_solve"] = dry_frames
            if runner.recorder is None:
                break                                   # unrecorded smoke: dry run IS the run
            dry_vec = _outcome_vector(runner)
            dry_facing = runner.facing
            # frame-neutral rewind to the milestone-start snapshot, then the recorded replay
            runner.frame_idx, runner.facing = f0, facing0
            runner.load_state_bytes(snap, record=False)
            runner.restore_log[-1]["solve"] = event_id
            if corrupt_replay is not None:
                log = corrupt_replay(list(log))
            runner.replay_capture(log)
            telemetry["replay_frames"] = runner.frame_idx - f0
            rec_vec = _outcome_vector(runner)
            mismatch = _vector_mismatch(dry_vec, rec_vec)
            if mismatch is not None:
                raise DeterminismError(
                    f"milestone {event_id}: recorded replay diverged from dry solution "
                    f"({mismatch})")
            runner.facing = dry_facing                  # replay steps raw frames; resync tracker
            result["start_frame"], result["end_frame"] = f0, runner.frame_idx
            break

        # dry attempt failed: frame-neutral restore, retry with shifted jitter
        telemetry["dry_frames_wasted"] += runner.frame_idx - f0
        runner.frame_idx = f0
        runner.load_state_bytes(snap, record=False)
        runner.restore_log[-1]["solve"] = event_id
        runner.facing = facing0
    else:
        result["failed_persistent"] = True              # policy defect: never retried around

    result["retry"] = telemetry
    return result


def _verify_scheduled_blocks(schedule, results, block_log, executed_nav_phases, phases_path):
    """W33 blocks-dropped guard: every block scheduled after a COMPLETED milestone must
    have executed, and every nav block that ran must have stamped its phase into
    phases.jsonl (when recording). The W33 pilot bug — plan int milestone indices used
    verbatim as schedule keys, never matching the director's event_id (string) lookup —
    finished 51/51 green with all six expedition blocks silently dropped and an empty
    phases.jsonl; this turns that failure class into a hard error at run end. Blocks
    scheduled after milestones the run never completed (abort / stop_after) are excused."""
    from collections import Counter

    completed = {r["event_id"] for r in results if r["validation"] in ("passed", "skipped")}
    ran_after = Counter(b.get("after") for b in block_log)
    missing = [
        f"{mid}: scheduled={len(blks)} executed={ran_after.get(mid, 0)}"
        for mid, blks in schedule.items()
        if mid in completed and ran_after.get(mid, 0) != len(blks)
    ]
    if missing:
        raise RuntimeError("scheduled blocks did not execute: " + "; ".join(missing))
    if phases_path is not None and executed_nav_phases:
        rows = [json.loads(line) for line in Path(phases_path).read_text().splitlines() if line.strip()]
        seen = Counter(row.get("phase") for row in rows)
        for phase, want in Counter(executed_nav_phases).items():
            if seen.get(phase, 0) < want:
                raise RuntimeError(
                    f"phases.jsonl is missing block phase entries for {phase!r}: "
                    f"have {seen.get(phase, 0)}, executed {want}")


def run_playthrough(*, policy_dir: str, out_dir: str, rom_path: str = "Emerald-GBAdvance/rom.gba",
                    starter: str = "mudkip", seed: int = 0, record: bool = True,
                    stop_after: str | None = None, per_milestone_max: int = 18000,  # rescaled for condition-based pacing (W33 §3.5)
                    blocks: bool = False, expedition: list[dict] | None = None,
                    persona: dict | None = None, max_attempts: int = 5,
                    resume_state: str | None = None,
                    resume_after: str | None = None,
                    resume: dict | str | None = None,
                    savestate_every: int = 4000) -> dict:
    """...

    RESUME (W33): `resume_state` is a savestate written by an earlier attempt and
    `resume_after` the last milestone that attempt completed. A run that dies at
    milestone 46 after ninety minutes should not replay the whole game to test a
    one-line fix — it should carry on from just before where it broke. The seam is
    recorded in the manifest (resumed_from / resume_frame / resume_after) because a
    resumed run is frames A..B from one build and B..C from another, and that has to
    be visible to anything that reads the corpus rather than inferred later.

    STITCH (W33 §3.4): passing `resume` — the handshake dict/path written by
    collection.resume_stitch.prepare_resume, whose `savestate` IS `resume_state` —
    makes the tail APPEND to the rewound recording instead of opening a fresh one:
    recorder and sink continue the prefix's channels and counters, and the runner
    continues its FRAME NUMBERING from the cut (`frame_idx = cut_frame`), so the
    finished directory is one continuous, gate-valid stream instead of two runs."""
    from collection.playthrough.schedule import build_expedition_schedule, build_schedule
    from collection.playthrough.blocks.base import run_block, run_nav_block
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    persona_cfg = build_persona(seed, persona)
    schedule = build_schedule(seed) if blocks else {}
    # W33 §2: explicit expedition-block entries ([{after, block, **kwargs}]) merged in.
    for mid, blks in build_expedition_schedule(expedition or []).items():
        schedule.setdefault(mid, []).extend(blks)
    nav_mk = None
    block_log = []
    executed_nav_phases: list[str] = []
    ce.set_expected_starter(starter.capitalize())       # STARTER_CHOSEN gate holds THIS species
    events = discover_heatz_events(policy_dir)          # ordered, chained
    by_id = {e["event_id"]: e for e in events}
    order = [m for m in MILESTONE_ORDER if m in by_id]  # skip GAME_RUNNING (no policy row)

    resume_cfg = None
    if resume is not None:
        from collection.resume_stitch import load_resume

        resume_cfg = load_resume(resume)
        if resume_state is None:
            resume_state = resume_cfg["savestate"]
    recorder_cm = ChunkRecorder(str(out), run_id=f"playthrough_{starter}_s{seed}",
                                visual_fps=1000 if record else 30, backend="npz",
                                metadata={"kind": "playthrough", "starter": starter, "seed": seed,
                                          "persona": persona_cfg},
                                resume=resume_cfg if record else None)
    results = []
    aborted_milestone = None
    t_run = time.time()
    with recorder_cm as recorder:
        # Fine-grained savestates so a wedge can be resumed from just before it rather
        # than from the last block boundary. They are nearly free: a whole run's
        # savestates were 2.4 MB of a 1.3 GB run (0.2%) at the 4000-frame default, so
        # even an 8x tighter cadence stays under half a percent of the recording.
        runner = DirectEmulatorRunner(rom_path=rom_path, load_state=None,
                                      recorder=recorder if record else None,
                                      savestate_every=savestate_every)
        stitching = resume_cfg is not None and record
        if stitching:
            # initialize() re-points manifest["restores"] at the runner's log, so the
            # prefix's restores (and the seam entry the recorder just appended) have to
            # be IN that log or they are dropped from the manifest.
            runner.restore_log.extend(recorder.manifest.get("restores") or [])
        # the boot frame of a resumed run is thrown away at the seam: never record it
        runner.initialize(record_initial_frame=not stitching)
        if record:
            # W33 §14.4 persona plumbing: the persona is a first-class manifest field
            recorder.manifest["persona"] = persona_cfg
            recorder._write_manifest()
        # W33 regen FAST-RECORD: the live sink costs 3.78 ms/frame (86% of recording
        # overhead; raw 1,943 fps vs 141 live). The ledger is a pure function of the
        # recorded ppu blob stream, so labels are extracted OFFLINE after the gate
        # instead (collection/extract_ledger.py -- validated field-for-field on
        # exp_052, all 28 fields, 6.7k frames/sec). W33_FAST_RECORD=1 detaches it.
        import os as _os
        _fast = _os.environ.get("W33_FAST_RECORD") == "1"
        sink = (WorldModelSink(str(out), resume=resume_cfg)
                if (record and _os.environ.get("W33_NO_SINK") != "1") else None)
        if sink is not None and _fast:
            sink.skip_ledger = True      # blobs + semantic always; labels offline
        if sink is not None:
            sink.capture_mode = _os.environ.get("W33_CAPTURE_MODE", "full")
        if _os.environ.get("W33_DIRECT_RECORD") == "1":
            import threading, json as _json, os as _os2

            def _stall_watch(r=runner, outdir=str(out)):
                # Three tripwires (W34):
                #   frozen   — frame counter stuck 150s (the original: emulator wedged)
                #   semantic — frames FLOW but nothing happens: in_battle constant True
                #              at a constant position for 60k+ frames. Wave 1's B-loop
                #              wedge burned 850k frames this way and the frozen check,
                #              watching only frame_idx, saw a perfectly lively run.
                #   party    — gPlayerPartyCount != 1. The whole trainer/flee/whiteout
                #              policy rests on the single-mon invariant; a catch is a
                #              run-invalidating defect and must fail AT the catch, not
                #              an hour later in an unexplainable battle wedge.
                last, since = -1, 0
                sem_key, sem_frames = None, 0
                party_bad = 0                             # double-confirm: a poll racing
                while True:                               # a state load can read torn
                    import time as _t
                    _t.sleep(10)
                    f = r.frame_idx
                    if f == last:
                        since += 10
                        if since >= 150:                 # 150s frozen under direct
                            try:                          # record = genuinely wedged
                                Path(outdir, "STALL.json").write_text(_json.dumps(
                                    {"frame_idx": f, "kind": "frozen", "at": _t.time()}))
                            finally:
                                print(f"STALL-EXIT frame={f}", flush=True)
                                _os2._exit(86)
                        continue
                    try:
                        from collection.extractors.ram import GBAState as _GS
                        st = _GS(env=r.env)
                        pc = st.u8(0x020244E9)           # gPlayerPartyCount
                        n = r.nav_state()
                        key = (bool(n.in_battle), n.map, n.x, n.y)
                    except Exception:
                        last, since = f, 0
                        continue                          # mid-transition reads throw
                    party_bad = party_bad + 1 if pc not in (0, 1) else 0
                    if party_bad >= 2:                    # 0 = pre-starter boot
                        try:
                            Path(outdir, "STALL.json").write_text(_json.dumps(
                                {"frame_idx": f, "kind": "party_violation",
                                 "party_count": int(pc), "at": _t.time()}))
                        finally:
                            print(f"PARTY-EXIT frame={f} count={pc}", flush=True)
                            _os2._exit(87)
                    # No-progress tripwire: no milestone or block has COMPLETED for
                    # 300k frames. Catches loop classes the position check cannot —
                    # the acceptance run oscillated Oldale<->Center for 150k frames
                    # with the map key changing every cycle, feeding the position
                    # check fresh keys forever. Milestone/block completion is the
                    # one signal every healthy run emits continuously.
                    prog = getattr(r, "_progress_frame", 0)
                    if f - prog > 300_000:
                        try:
                            Path(outdir, "STALL.json").write_text(_json.dumps(
                                {"frame_idx": f, "kind": "no_progress",
                                 "last_progress_frame": prog, "at": _t.time()}))
                        finally:
                            print(f"NO-PROGRESS-EXIT frame={f} last={prog}", flush=True)
                            _os2._exit(86)
                    if key[0] and key == sem_key:
                        sem_frames += f - last
                        if sem_frames >= 60_000:
                            try:
                                Path(outdir, "STALL.json").write_text(_json.dumps(
                                    {"frame_idx": f, "kind": "semantic",
                                     "pos": list(key[1:]), "in_battle": True,
                                     "stuck_frames": sem_frames, "at": _t.time()}))
                            finally:
                                print(f"SEMANTIC-STALL-EXIT frame={f} key={key}", flush=True)
                                _os2._exit(86)
                    else:
                        sem_key, sem_frames = key, 0
                    last, since = f, 0

            threading.Thread(target=_stall_watch, daemon=True).start()
        if sink is not None:
            runner.frame_hook = sink.capture
        # Boot past the title screen: mash A/START until GAME_RUNNING (new game begins).
        if resume_state:
            import zlib
            raw = open(resume_state, "rb").read()
            if resume_cfg is not None:
                # STITCH: continue the prefix's frame numbering. Without this every
                # index the tail writes (action frame_idx, ppu frame, savestate name)
                # restarts at 0 in the middle of the stream.
                runner.frame_idx = int(resume_cfg["cut_frame"])
            runner.load_state_bytes(zlib.decompress(raw) if resume_state.endswith(".z") else raw,
                                    record=True)
            if stitching:
                # the seam restore is already logged (ChunkRecorder's resume entry);
                # a second recorded restore would claim a frame that does not exist
                # and break the gate's visual == actions + 1 + recorded_restores rule
                runner.restore_log.pop()
            for _ in range(60):
                runner.step_frame([], phase="resume")
            if record:
                recorder.manifest["resumed_from"] = resume_state
                recorder.manifest["resume_after"] = resume_after
                recorder.manifest["resume_frame"] = runner.frame_idx
                recorder._write_manifest()
            print(json.dumps({"RESUME": {"state": resume_state, "after": resume_after,
                                         "frame": runner.frame_idx}}), flush=True)
        else:
            # Boot is recorded live (pre-spine); solve-then-record starts with milestones.
            for _ in range(600):
                runner.step_frame(["a"], phase="boot")
                st = runner.state()
                if st.game_state not in ("title", "intro", None):
                    break
        start_money = 0
        try:
            _skip = bool(resume_after)
            for event_id in order:
                if _skip:                      # already done by the attempt we resumed
                    if event_id == resume_after:
                        _skip = False
                    results.append({"event_id": event_id, "validation": "skipped",
                                    "failure_reason": "resumed_past"})
                    continue
                exp = ce._load_expected_state(rom_path=rom_path,
                                              completed_state=by_id[event_id].get("completed_state"),
                                              event_id=event_id)
                post = by_id[event_id].get("postcondition", event_id)
                t0 = time.time()
                f0 = runner.frame_idx
                r = solve_then_record_milestone(
                    runner, event_id=event_id, policy_dir=policy_dir,
                    expected_state=exp, postcondition=post, start_money=start_money,
                    max_actions=per_milestone_max, starter=starter,
                    persona=persona_cfg, max_attempts=max_attempts)
                r.update(event_id=event_id, wall_s=round(time.time() - t0, 1),
                         frames=runner.frame_idx - f0)
                results.append(r)
                runner._progress_frame = runner.frame_idx     # watchdog: real progress
                st = runner.state()
                print(json.dumps({**r, "map": st.map, "gs": st.game_state}), flush=True)
                if r["validation"] not in ("passed", "skipped"):
                    # §13(e): N attempts exhausted -> failed-persistent, abort the run
                    r["FAILED_RUN_HERE"] = True
                    aborted_milestone = event_id
                    break
                # Life blocks scheduled after this milestone (diversity injection).
                # Blocks run RECORDED as today (robust by contract; summaries expected).
                for blk in schedule.get(event_id, []):
                    if hasattr(blk, "run"):          # navigator-driven expedition block (W33)
                        if nav_mk is None:
                            from collection.navigator import MapKnowledge
                            nav_mk = MapKnowledge(
                                rom_path=rom_path,
                                rng=(random.Random(f"{persona_cfg['seed']}:nav")
                                     if persona_cfg.get("tie_break") else None))
                        outcome = run_nav_block(runner, blk, mk=nav_mk)
                        if outcome.get("ran"):
                            executed_nav_phases.append(blk.phase)
                    else:
                        outcome = run_block(runner, blk)
                    outcome["after"] = event_id
                    block_log.append(outcome)
                    runner._progress_frame = runner.frame_idx  # watchdog: real progress
                    print(json.dumps({"BLOCK": outcome}), flush=True)
                if stop_after and event_id == stop_after:
                    break
        finally:
            if sink is not None:
                sink.close()
            total_frames = runner.frame_idx
            runner.close()
    # W33 blocks-dropped guard: scheduled-but-unexecuted blocks (or missing phase
    # stamps) RAISE here instead of letting the run finish green without its blocks.
    _verify_scheduled_blocks(schedule, results, block_log, executed_nav_phases,
                             (out / "phases.jsonl") if record else None)
    retry_rows = {r["event_id"]: r["retry"] for r in results if "retry" in r}
    summary = dict(starter=starter, seed=seed, persona=persona_cfg,
                   milestones_total=len(order),
                   milestones_passed=sum(1 for r in results if r["validation"] in ("passed", "skipped")),
                   # A resumed run PLAYED only the tail: the milestones before the
                   # resume point were inherited, not recorded here. Counting them as
                   # passed would let a partial recording be banked as a complete run,
                   # so say plainly how many were inherited and where the prefix lives.
                   resumed_from=resume_state,
                   resume_after=resume_after,
                   resume_cut_frame=(resume_cfg or {}).get("cut_frame"),
                   stitched=resume_cfg is not None,
                   milestones_inherited=sum(
                       1 for r in results if r.get("failure_reason") == "resumed_past"),
                   recording_is_end_to_end=not bool(resume_state),
                   total_frames=total_frames, wall_s=round(time.time() - t_run, 1), results=results,
                   blocks=block_log,
                   # §13 retry telemetry: per-milestone rows + fleet-audit aggregates
                   retry=retry_rows,
                   retry_total=dict(
                       attempts=sum(t["attempts"] for t in retry_rows.values()),
                       dry_frames_wasted=sum(t["dry_frames_wasted"] for t in retry_rows.values()),
                       dry_frames_solve=sum(t["dry_frames_solve"] for t in retry_rows.values())),
                   aborted_milestone=aborted_milestone)
    (out / "playthrough_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
