"""W33 corpus-v2 §11: the retroactive derivation kit — any future field, forever.

v2's contract is not "every field recorded" but "every field DERIVABLE": periodic
savestates (~4k frames + block boundaries) hold complete machine state, and replay is
deterministic under the pinned fixed RTC. So a brand-new field at tick t = load the
nearest stored state <= t, replay the recorded per-frame buttons, read RAM. This
module is that loop as an API:

    derive_field(run_dir, reader_fn, out=...)   # reader_fn(env) -> value/dict per tick

For a v2 run it iterates savestates/NNNNNNNN.state.z chronologically; for each gap
[state_k, state_{k+1}) it loads state_k FRAME-NEUTRALLY (run_settle_frame=False — the
machine is byte-exactly the stored state, §3.2), replays the recorded actions from
actions.jsonl frame by frame (`buttons_held` via audits.replay.load_actions — the
stream is homogeneous by the M1 contract), calls reader_fn(env) after every frame,
and emits per-tick arrays (npz when `out` is given). Ticks are stamped with the
recording's own emulator frame indices: gap [k, k') yields rows k+1..k' — the same
indexing the recorder's ledger chunks use (row = state AFTER the frame ran; frame 0
has no row, matching the accepted frame-0 sink gap).

Replay-fidelity verification (MANDATORY, per gap)
-------------------------------------------------
At the END of each gap the reproduced machine must equal the NEXT stored savestate.
The comparison is a RAM DIGEST: sha256 over the render-state six-block blob
(io+palette+oam+vram+ewram+iwram — `extract_full_ppu_state`/`serialize_ppu`, the
exact domain `audits/replay.py` proves bit-stable under replay). A full byte compare
of `save_raw_state` output is deliberately NOT used: mGBA's raw-savestate
serialization carries load-history-dependent header bytes, so byte equality needs the
save->load->save canonicalization + artifact-byte zeroing machinery of the markov
builder (attic/tools_state/build_savestate_markov_corpus.py `_canonicalize` +
`normalize_mgba_savestate` — read 2026-08-20); the six-block digest sidesteps the
serializer entirely while still covering every RAM/PPU domain a reader_fn can read.
(CPU registers are outside the digest; at a frame boundary a divergence that touches
nothing in 384 KB of RAM/VRAM/IO across a whole gap has no known mechanism, and the
NEXT gap starts from the stored state regardless — see below.)

A mismatch RAISES ReplayFidelityError naming the gap and the diverged blocks. After
a verified gap the emulator is re-seated on the stored next state (a semantic no-op
when the digests matched), so drift can never accumulate across gaps.

Preconditions checked (a violation raises, never silently degrades):
  * fixed RTC: the run manifest's provenance.fixed_rtc_value must equal the live
    emulator's — replay under a different RTC is silently divergent (v1 lesson).
  * dense action rows: every frame of a gap must have an action row. A recorded
    restore (manifest "restores") breaks button-replay for its gap; v2 blocks never
    restore mid-recording, so a hole is a real defect and raises ReplayGapError.
  * the leading gap [0, first_state): frame 0 = manifest metadata load_state + the
    one initialize() settle frame; replayed as frame-neutral load + one buttonless
    frame. Skipped (with the reason in the result) when the original load_state file
    is no longer on disk. Frames after the LAST savestate are derivable but
    UNVERIFIABLE (no terminal state to check against) — excluded unless
    include_tail=True.

The self-test (CLI --probe) derives the LEDGER fields (`ledger_reader`, wrapping
extractors.ledger_panel.read_ledger) and cross-checks every derived row against the
run's OWN recorded ledger chunks — the §11 contract proven end-to-end on real data:
  .venv/bin/python -m collection.derive RUN_DIR --probe [--max-gaps N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from collection.audits.replay import load_actions
from collection.render_state import BLOCK_SIZES, extract_full_ppu_state
from collection.world_model_sink import serialize_ppu


class ReplayGapError(RuntimeError):
    """A gap cannot be replayed (missing action rows — e.g. a recorded restore)."""


class ReplayFidelityError(RuntimeError):
    """Replay of a gap did not reproduce the next stored savestate."""


def _digest(env) -> tuple[str, bytes]:
    blob = serialize_ppu(extract_full_ppu_state(env))
    return hashlib.sha256(blob).hexdigest(), blob


def _diverged_blocks(a: bytes, b: bytes) -> list[str]:
    out, off = [], 0
    for name, sz in BLOCK_SIZES:
        if a[off:off + sz] != b[off:off + sz]:
            n = sum(x != y for x, y in zip(a[off:off + sz], b[off:off + sz]))
            out.append(f"{name}({n}B)")
        off += sz
    return out


def run_savestates(run_dir: str | Path) -> list[tuple[int, Path]]:
    """(frame_idx, path) of every stored periodic/boundary savestate, ascending."""
    d = Path(run_dir) / "savestates"
    if not d.is_dir():
        return []
    return sorted((int(p.name.split(".")[0]), p) for p in d.glob("*.state.z"))


def ledger_reader(env):
    """The built-in reader: one ledger row per tick (ledger_panel schema)."""
    from collection.extractors.ledger_panel import read_ledger
    from collection.extractors.ram import GBAState
    return read_ledger(GBAState.snapshot(env))


def derive_field(run_dir: str | Path, reader_fn, *, out: str | Path | None = None,
                 rom_path: str = "Emerald-GBAdvance/rom.gba", env=None,
                 max_gaps: int | None = None, include_tail: bool = False) -> dict:
    """Replay `run_dir` gap by gap, calling reader_fn(env) per tick.

    Returns {"frame_idx": (n,) array, <field>: (n, ...) arrays..., "gaps": [...]}
    where fields come from reader_fn's return (dict -> one array per key; scalar or
    ndarray -> a single "value" field). Writes the arrays to `out` (npz) if given.
    Raises ReplayGapError / ReplayFidelityError per the module docstring.
    """
    run = Path(run_dir)
    states = run_savestates(run)
    if not states:
        raise ReplayGapError(f"{run}: no savestates/ — not a v2 run (or savestate_every=0)")
    actions = load_actions(run)
    manifest = json.loads((run / "manifest.json").read_text())

    own_env = env is None
    if own_env:
        from pokemon_env.emulator import EmeraldEmulator
        env = EmeraldEmulator(rom_path=rom_path)
        env.initialize()
    pinned = (manifest.get("provenance") or {}).get("fixed_rtc_value")
    live_rtc = getattr(env, "_pokemon_wm_fixed_rtc_value", None)
    if pinned is not None and pinned != live_rtc:
        raise ReplayGapError(f"{run}: recorded under fixed RTC {pinned}, replayer has "
                             f"{live_rtc} — replay would silently diverge")

    frame_rows: list[int] = []
    rows: list = []
    gaps: list[dict] = []

    def replay_span(a: int, b: int) -> None:
        missing = [f for f in range(a, b) if f not in actions]
        if missing:
            raise ReplayGapError(
                f"{run}: gap [{a},{b}) has {len(missing)} frames without action rows "
                f"(first: {missing[0]}) — a recorded restore or a truncated "
                f"actions.jsonl; button replay cannot cross it")
        for f in range(a, b):
            env.run_frame_with_buttons([x.lower() for x in actions[f]])
            frame_rows.append(f + 1)
            rows.append(reader_fn(env))

    try:
        # ---- leading gap: original load_state + the initialize() settle frame
        first = states[0][0]
        load0 = (manifest.get("metadata") or {}).get("load_state")
        pending_verify_from = None
        if load0 and Path(load0).exists() and first > 0:
            env.load_state(str(load0), run_settle_frame=False)
            env.run_frame_with_buttons([])            # initialize()'s settle frame
            rows.append(reader_fn(env))               # tick 0's post-settle row is
            frame_rows.append(0)                      # frame 0 (the state row the
            replay_span(0, first)                     # recorder stamps as initial)
            pending_verify_from = 0
        else:
            gaps.append({"gap": [0, first], "verified": False,
                         "reason": "leading gap skipped: original load_state "
                                   "unavailable" if first > 0 else "empty"})

        for i, (k, path) in enumerate(states):
            raw = zlib.decompress(path.read_bytes())
            if pending_verify_from is not None:
                d_live, blob_live = _digest(env)
                env.load_state(state_bytes=raw, run_settle_frame=False)
                d_stored, blob_stored = _digest(env)
                if d_live != d_stored:
                    raise ReplayFidelityError(
                        f"{run}: replay of gap [{pending_verify_from},{k}) does not "
                        f"reproduce savestates/{path.name} — diverged blocks: "
                        f"{_diverged_blocks(blob_stored, blob_live)}")
                gaps.append({"gap": [pending_verify_from, k], "verified": True})
            else:
                env.load_state(state_bytes=raw, run_settle_frame=False)
            if max_gaps is not None and len([g for g in gaps if g.get("verified")]) >= max_gaps:
                break
            if i + 1 < len(states):
                nxt = states[i + 1][0]
                replay_span(k, nxt)
                pending_verify_from = k
            else:
                pending_verify_from = None
                if include_tail:
                    tail_end = max(actions) + 1 if actions else k
                    if tail_end > k:
                        replay_span(k, tail_end)
                        gaps.append({"gap": [k, tail_end], "verified": False,
                                     "reason": "tail after the last savestate — "
                                               "derivable, unverifiable"})
    finally:
        if own_env:
            env.stop()

    # ---- assemble arrays
    result: dict = {"frame_idx": np.asarray(frame_rows, np.uint32), "gaps": gaps}
    if rows:
        if isinstance(rows[0], dict):
            for key in rows[0]:
                result[key] = np.asarray([r[key] for r in rows])
        else:
            result["value"] = np.asarray(rows)
    if out is not None:
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrays)
        Path(str(out) + ".gaps.json").write_text(json.dumps(gaps, indent=1))
    return result


# ---- --probe: the §11 self-test against the run's own recorded ledger --------------

def load_run_ledger(run_dir: str | Path) -> dict[str, np.ndarray] | None:
    """Concatenate the run's recorded ledger chunks (ledger/index.json schema)."""
    d = Path(run_dir) / "ledger"
    idx_path = d / "index.json"
    if not idx_path.exists():
        return None
    idx = json.loads(idx_path.read_text())
    if not idx["chunks"]:
        return None
    parts = [np.load(d / fn) for fn, _, _ in idx["chunks"]]
    return {name: np.concatenate([p[name] for p in parts])
            for name in idx["fields"]}


def probe(run_dir: str | Path, *, max_gaps: int | None = None,
          rom_path: str = "Emerald-GBAdvance/rom.gba") -> dict:
    """Derive the ledger fields by replay and cross-check EVERY overlapping tick
    against the run's own recorded ledger chunks. Returns the comparison summary;
    raises on any replay-fidelity failure or field mismatch."""
    rec = load_run_ledger(run_dir)
    if rec is None:
        raise ReplayGapError(f"{run_dir}: no ledger/ chunks — probe needs a run "
                             f"recorded with the WorldModelSink (record_wm)")
    der = derive_field(run_dir, ledger_reader, rom_path=rom_path, max_gaps=max_gaps)
    rec_by_frame = {int(f): i for i, f in enumerate(rec["frame_idx"])}
    compared = mismatches = 0
    bad: list[str] = []
    fields = [k for k in rec if k != "frame_idx"]
    for j, f in enumerate(der["frame_idx"]):
        i = rec_by_frame.get(int(f))
        if i is None:
            continue
        compared += 1
        for name in fields:
            if not np.array_equal(rec[name][i], der[name][j]):
                mismatches += 1
                if len(bad) < 20:
                    bad.append(f"frame {int(f)} field {name}: recorded "
                               f"{rec[name][i]!r} != derived {der[name][j]!r}")
    summary = {
        "gaps_verified": sum(1 for g in der["gaps"] if g.get("verified")),
        "gaps_unverified": [g for g in der["gaps"] if not g.get("verified")],
        "ticks_derived": int(len(der["frame_idx"])),
        "ticks_compared": compared,
        "field_mismatches": mismatches,
        "fields": fields,
    }
    if mismatches:
        raise ReplayFidelityError(
            f"{run_dir}: derived ledger disagrees with the recorded ledger on "
            f"{mismatches} (frame, field) cells; first: {bad[:5]}")
    if compared == 0:
        raise ReplayGapError(f"{run_dir}: derived and recorded ledgers share no "
                             f"frames — nothing was actually cross-checked")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--probe", action="store_true",
                    help="derive ledger fields and cross-check the run's own chunks")
    ap.add_argument("--out", default=None, help="npz output for derive (non-probe)")
    ap.add_argument("--max-gaps", type=int, default=None)
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    args = ap.parse_args()
    if args.probe:
        s = probe(args.run_dir, max_gaps=args.max_gaps, rom_path=args.rom)
        print(json.dumps(s, indent=2, default=str))
        print(f"PROBE PASS: {s['gaps_verified']} gaps bit-verified, "
              f"{s['ticks_compared']} ticks field-equal to the recorded ledger")
    else:
        r = derive_field(args.run_dir, ledger_reader, out=args.out,
                         rom_path=args.rom, max_gaps=args.max_gaps)
        print(f"derived {len(r['frame_idx'])} ticks over "
              f"{sum(1 for g in r['gaps'] if g.get('verified'))} verified gaps"
              + (f" -> {args.out}" if args.out else ""))


if __name__ == "__main__":
    main()
