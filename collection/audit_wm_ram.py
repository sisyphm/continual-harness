"""Phase-0 audit, Layer 2 — per-frame RAM fields via a STREAMING walk of `ppu_state.bin`.

Walks each run's delta chain once (keyframe + XOR deltas; never materializes more than one frame)
and extracts the small field set the mode taxonomy needs, saved as per-run arrays:

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

Usage:
  .venv/bin/python -m collection.audit_wm_ram --data_root ../pokemon-worldmodel/data \
      --out_dir ../pokemon-worldmodel/data/processed/audit/ram
"""

from __future__ import annotations

import argparse
import json
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from collection.audit_wm import discover_runs

# block offsets inside the 396,288-byte blob (render_state.BLOCK_SIZES order)
OFF_IO, OFF_PAL, OFF_OAM, OFF_VRAM, OFF_EWRAM, OFF_IWRAM = 0, 1024, 2048, 3072, 101376, 363520
BLOB_SIZE = 396288

EW = lambda addr: OFF_EWRAM + (addr - 0x02000000)        # ewram address -> blob offset
IW = lambda addr: OFF_IWRAM + (addr - 0x03000000)        # iwram address -> blob offset

A_CB2 = IW(0x030022C4)
A_INBATTLE = IW(0x030026F9)
A_SCRIPT = EW(0x02037A58)
A_BATTLE_TYPE = EW(0x02022AAE)
A_BMON0 = EW(0x02024084)                                 # gBattleMons[0] — VALIDATED (Mudkip/Zigzagoon)
BMON_STRIDE = 0x58
A_DIALOG = EW(0x020370B8)                                # DIALOG_STATE candidate — DEAD (constant 0x3FF)
# gObjectEvents — CORRECTED layout (audit finding, validated on live frames): 16 records of
# 0x24=36 bytes (pokeemerald struct ObjectEvent), spriteId +0x04, graphicsId +0x05, localId +0x08,
# initial/current/previous Coords16 at +0x0C/+0x10/+0x14 in PLAIN map coords (no +7), facing-ish
# byte at +0x18. bit0-of-byte0 is NOT a usable active flag (0xFF-cleared slots read active) —
# validity is tested structurally instead. The old sink read stride 68 + gfx@+0x03 ->
# semantic.jsonl objects are garbage dataset-wide.
A_OBJ = EW(0x02037230)
OBJ_SIZE, OBJ_N = 0x24, 16


def _obj_valid(cur: np.ndarray, o: int) -> bool:
    """Live-NPC validity: not 0xFF-cleared, plausible gfx id, real localId, sane map coords."""
    if cur[o] == 0xFF or not (1 <= cur[o + 0x05] <= 239) or cur[o + 0x08] >= 200:
        return False
    x, y = _s16(cur, o + 0x10), _s16(cur, o + 0x12)
    return 0 <= x <= 999 and 0 <= y <= 999


def _u16(buf: np.ndarray, off: int) -> int:
    return int(buf[off]) | (int(buf[off + 1]) << 8)


def _u32(buf: np.ndarray, off: int) -> int:
    return _u16(buf, off) | (_u16(buf, off + 2) << 16)


def _s16(buf: np.ndarray, off: int) -> int:
    v = _u16(buf, off)
    return v - 0x10000 if v >= 0x8000 else v


def extract_run(args: tuple[str, str, str, str]) -> str:
    name, run_dir, kind, out_dir = args
    d = Path(run_dir)
    idx = json.loads((d / "ppu_state.bin.idx.json").read_text())
    raw = (d / "ppu_state.bin").read_bytes()

    n = len(idx["frames"])
    fidx = np.zeros(n, np.int64)
    cb2 = np.zeros(n, np.uint32)
    in_battle = np.zeros(n, np.uint8)
    script = np.zeros((n, 4), np.uint8)
    bldcnt = np.zeros(n, np.uint16); bldy = np.zeros(n, np.uint16)
    dispcnt = np.zeros(n, np.uint16)
    species = np.zeros((n, 2), np.uint16)
    btype = np.zeros(n, np.uint32)
    dialog = np.zeros((n, 8), np.uint8)                  # DIALOG_STATE candidate bytes (validate later)
    obj_active = np.zeros(n, np.uint16)                  # bit i = gObjectEvents[i].active
    obj_gfx = np.zeros((n, OBJ_N), np.uint8)             # graphicsId per slot (corrected +0x05)
    obj_local = np.zeros((n, OBJ_N), np.uint8)           # localId per slot (0xFF = the player)
    obj_xy = np.zeros((n, OBJ_N, 2), np.int16)           # currentCoords per slot (+0x10)

    cur: np.ndarray | None = None
    for i, (f, off, _kind) in enumerate(idx["frames"]):
        ln = int.from_bytes(raw[off + 1:off + 5], "little")
        payload = np.frombuffer(zlib.decompress(raw[off + 5:off + 5 + ln]), np.uint8)
        if raw[off:off + 1] == b"K":
            cur = payload.copy()
        else:
            cur ^= payload                                  # in-place XOR delta
        assert cur is not None and len(cur) == BLOB_SIZE, (name, f, None if cur is None else len(cur))
        fidx[i] = f
        cb2[i] = _u32(cur, A_CB2)
        in_battle[i] = (cur[A_INBATTLE] & 0x02) != 0
        script[i] = cur[A_SCRIPT:A_SCRIPT + 4]
        bldcnt[i] = _u16(cur, OFF_IO + 0x50); bldy[i] = _u16(cur, OFF_IO + 0x54)
        dispcnt[i] = _u16(cur, OFF_IO + 0x00)
        species[i, 0] = _u16(cur, A_BMON0)
        species[i, 1] = _u16(cur, A_BMON0 + BMON_STRIDE)
        btype[i] = _u32(cur, A_BATTLE_TYPE)
        dialog[i] = cur[A_DIALOG:A_DIALOG + 8]
        for s in range(OBJ_N):
            o = A_OBJ + s * OBJ_SIZE
            if _obj_valid(cur, o):
                obj_active[i] |= 1 << s
                obj_gfx[i, s] = cur[o + 0x05]
                obj_local[i, s] = cur[o + 0x08]
                obj_xy[i, s, 0] = _s16(cur, o + 0x10)
                obj_xy[i, s, 1] = _s16(cur, o + 0x12)

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
