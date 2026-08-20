"""Phase-0 audit — replay-determinism check (finding recorded in ../pokemon-worldmodel/docs/PLAN.md §5).

The storage principle "savestate + input log => everything re-derivable" only holds if replay is
bit-deterministic. Test: load a storyline run's `initial.state`, re-apply its per-frame
`buttons_held` from `actions.jsonl`, re-extract the full PPU+WRAM condition per frame, and compare
byte-for-byte against the recorded `ppu_state.bin` chain. Reports match rate + first divergence
(and WHERE in the blob it diverged — io/palette/oam/vram/ewram/iwram — e.g. an RTC readout would
show as an isolated ewram/iwram drift).

Usage (no GPU needed):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python -m collection.audits.replay \
      --run ../pokemon-worldmodel/data/storyline_wm/CLOCK_INTERACT/attempt_000001
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from collection.render_state import BLOCK_SIZES, extract_full_ppu_state
from collection.world_model_sink import serialize_ppu


def load_actions(run_dir: Path) -> dict:
    """frame_idx -> buttons_held over ALL rows of actions.jsonl. The stream is
    homogeneous by contract (phase transitions live in phases.jsonl, never here),
    so every row must carry the action keys."""
    return {r["frame_idx"]: r["buttons_held"]
            for r in (json.loads(l) for l in (run_dir / "actions.jsonl").open())}


def recorded_blobs(run_dir: Path):
    """Yield (frame_idx, blob bytes) by streaming the recorded delta chain."""
    idx = json.loads((run_dir / "ppu_state.bin.idx.json").read_text())
    raw = (run_dir / "ppu_state.bin").read_bytes()
    cur = None
    for f, off, _ in idx["frames"]:
        ln = int.from_bytes(raw[off + 1:off + 5], "little")
        payload = np.frombuffer(zlib.decompress(raw[off + 5:off + 5 + ln]), np.uint8)
        cur = payload.copy() if raw[off:off + 1] == b"K" else (cur ^ payload)
        yield f, cur.tobytes()


def diverged_blocks(a: bytes, b: bytes) -> list[str]:
    out, off = [], 0
    for name, sz in BLOCK_SIZES:
        if a[off:off + sz] != b[off:off + sz]:
            n_diff = sum(x != y for x, y in zip(a[off:off + sz], b[off:off + sz]))
            out.append(f"{name}({n_diff}B)")
        off += sz
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="../pokemon-worldmodel/data/storyline_wm/CLOCK_INTERACT/attempt_000001")
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    run = Path(args.run)

    actions = load_actions(run)

    from pokemon_env.emulator import EmeraldEmulator
    env = EmeraldEmulator(rom_path=args.rom)
    env.initialize()
    env.load_state(str(run / "initial.state"))

    n_match = n_total = 0
    first_div = None
    for f, rec in recorded_blobs(run):
        if args.max_frames and f >= args.max_frames:
            break
        if f > 0:
            env.run_frame_with_buttons([b.lower() for b in actions.get(f - 1, [])])
        live = serialize_ppu(extract_full_ppu_state(env))
        n_total += 1
        if live == rec:
            n_match += 1
        elif first_div is None:
            first_div = (f, diverged_blocks(rec, live))
    print(f"replayed {n_total} frames: {n_match} bit-identical ({n_match / max(n_total,1):.2%})")
    if first_div:
        print(f"first divergence at frame {first_div[0]}: blocks {first_div[1]}")
    else:
        print("REPLAY IS BIT-DETERMINISTIC ✓")


if __name__ == "__main__":
    main()
