"""Phase-0 audit, Layer 2 — per-frame RAM fields via the canonical streaming walk (`iter_states`).

Extracts the small field set the mode taxonomy needs from every frame of every run, saved as
per-run arrays (`<kind>__<name>.v2.npz`):

  cb2          u32  gMain.callback2 (0x030022C4) — the game's own "which screen am I on" pointer.
                    Taxonomy method: CLUSTER frames by cb2 value, then label clusters empirically
                    from sample images + auto signals — no trusting unverified symbol lists.
  in_battle    u8   gMain.inBattle bit (0x030026F9 & 0x02). gMain base cross-checked: this
                    documented address sits at gMain+0x439, consistent with cb2 above.
  script       u8   sGlobalScriptContext bytes @0x02037A58 (+0..+3) — script/dialogue running.
  bldy,bldcnt  u16  io blend registers (fades).
  dispcnt      u16  io display control.
  species0/1   u16  gBattleMons[0/1].species candidates @0x02024084 (+0x58 stride) — validated
                    empirically downstream (plausible dex ids while in_battle), not assumed.
  battle_type  u32  gBattleTypeFlags @0x02022AAE (harness-documented).
  obj_*             gObjectEvents per-slot identity/coords. NOTE: this keeps the audit's original
                    NPC-only validity filter (gfx 1..239, localId<200 — EXCLUDES the player record)
                    so re-runs stay comparable to the archived campaign outputs; the canonical
                    extractor — player included, MAP_OFFSET applied — is `extractors.entities`.

All addresses are GBA bus addresses read through `extractors.ram.GBAState` (the offline=online
seam); the delta-chain walk is `iter_states` — this module owns no parsing of its own.

Usage:
  .venv/bin/python -m collection.audits.ram_fields --data_root ../pokemon-worldmodel/data \
      --out_dir ../pokemon-worldmodel/data/processed/audit/ram
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from collection.corpus import discover_runs
from collection.extractors.entities import OBJ_BASE, OBJ_N, OBJ_SIZE
from collection.extractors.ram import GBAState, iter_states

A_CB2 = 0x030022C4
A_INBATTLE = 0x030026F9
A_SCRIPT = 0x02037A58
A_BATTLE_TYPE = 0x02022AAE
A_BMON0 = 0x02024084                                     # gBattleMons[0] — VALIDATED (Mudkip/Zigzagoon)
BMON_STRIDE = 0x58
A_DIALOG = 0x020370B8                                    # DIALOG_STATE candidate — DEAD (constant 0x3FF)
A_DISPCNT, A_BLDCNT, A_BLDY = 0x04000000, 0x04000050, 0x04000054


def _obj_valid(st: GBAState, o: int) -> bool:
    """The audit's NPC-only validity (see module docstring; canonical rule lives in entities)."""
    if st.u8(o) == 0xFF or not (1 <= st.u8(o + 0x05) <= 239) or st.u8(o + 0x08) >= 200:
        return False
    x, y = st.s16(o + 0x10), st.s16(o + 0x12)
    return 0 <= x <= 999 and 0 <= y <= 999


def extract_run(args: tuple[str, str, str, str]) -> str:
    name, run_dir, kind, out_dir = args
    d = Path(run_dir)
    n = len(json.loads((d / "ppu_state.bin.idx.json").read_text())["frames"])

    fidx = np.zeros(n, np.int64)
    cb2 = np.zeros(n, np.uint32)
    in_battle = np.zeros(n, np.uint8)
    script = np.zeros((n, 4), np.uint8)
    bldcnt = np.zeros(n, np.uint16); bldy = np.zeros(n, np.uint16)
    dispcnt = np.zeros(n, np.uint16)
    species = np.zeros((n, 2), np.uint16)
    btype = np.zeros(n, np.uint32)
    dialog = np.zeros((n, 8), np.uint8)                  # DIALOG_STATE candidate bytes (validate later)
    obj_active = np.zeros(n, np.uint16)                  # bit i = slot i passed the validity filter
    obj_gfx = np.zeros((n, OBJ_N), np.uint8)             # graphicsId per slot (corrected +0x05)
    obj_local = np.zeros((n, OBJ_N), np.uint8)           # localId per slot (0xFF = the player)
    obj_xy = np.zeros((n, OBJ_N, 2), np.int16)           # currentCoords per slot (+0x10, RAW: no -7)

    for i, (f, st) in enumerate(iter_states(d)):
        fidx[i] = f
        cb2[i] = st.u32(A_CB2)
        in_battle[i] = (st.u8(A_INBATTLE) & 0x02) != 0
        script[i] = np.frombuffer(st.bytes(A_SCRIPT, 4), np.uint8)
        bldcnt[i] = st.u16(A_BLDCNT); bldy[i] = st.u16(A_BLDY)
        dispcnt[i] = st.u16(A_DISPCNT)
        species[i, 0] = st.u16(A_BMON0)
        species[i, 1] = st.u16(A_BMON0 + BMON_STRIDE)
        btype[i] = st.u32(A_BATTLE_TYPE)
        dialog[i] = np.frombuffer(st.bytes(A_DIALOG, 8), np.uint8)
        for s in range(OBJ_N):
            o = OBJ_BASE + s * OBJ_SIZE
            if _obj_valid(st, o):
                obj_active[i] |= 1 << s
                obj_gfx[i, s] = st.u8(o + 0x05)
                obj_local[i, s] = st.u8(o + 0x08)
                obj_xy[i, s, 0] = st.s16(o + 0x10)
                obj_xy[i, s, 1] = st.s16(o + 0x12)

    out = Path(out_dir) / f"{kind}__{name}.v2.npz"
    np.savez_compressed(out, fidx=fidx, cb2=cb2, in_battle=in_battle, script=script,
                        bldcnt=bldcnt, bldy=bldy, dispcnt=dispcnt, species=species, btype=btype,
                        dialog=dialog, obj_active=obj_active, obj_gfx=obj_gfx,
                        obj_local=obj_local, obj_xy=obj_xy)
    return f"{kind}/{name}: {n} frames -> {out.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out_dir", default="../pokemon-worldmodel/data/processed/audit/ram")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(Path(args.data_root))
    todo = [(name, str(d), kind, str(out_dir)) for name, d, kind in runs
            if not (out_dir / f"{kind}__{name}.v2.npz").exists()]
    print(f"{len(runs)} runs, {len(todo)} to extract…")
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(extract_run, todo):
            print(" ", msg)
    print("done")


if __name__ == "__main__":
    main()
