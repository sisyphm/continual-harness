"""Phase-0 audit — final aggregation: joins all audit layers into the coverage-report numbers.

Per frame, joins: semantic (x/y/facing/map) + actions + brightness + textbox + framediff (RGB) +
RAM fields (cb2, in_battle, species, BLDY, corrected objects). Produces `audit_report.json` with:

  • MODE TAXONOMY mass: battle / dialogue / menu-or-other-cb2 / transition(dark|fade) / overworld
    (cb2 values auto-labeled by their battle/textbox/dark composition; sheets for visual confirm)
  • FREE IDLE: stationary-player runs EXCLUDING dialogue/battle/dark — the real idle number
  • identity support: NPC graphics_id (corrected), enemy species (gBattleMons[1]), per-map mass
  • transition support: fade (BLDY>0) frame mass
  • UNEXPLAINED DYNAMICS: frames where RGB changes (framediff>thresh) but the schema-v0 condition
    is static — the measured learned tail; samples dumped to a contact sheet for naming

Usage:
  .venv/bin/python -m collection.audits.report --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from collection.audits.l1 import DARK_THRESH, IDLE_BUCKETS
from collection.corpus import discover_runs

FD_THRESH = 2.0             # mean|Δ| above this = "pixels changed meaningfully"
PLAYER_GFX = (0, 1)         # Brendan/May overworld gfx ids (excluded from NPC support)


def _bucket(n: int) -> str:
    for lo, hi in IDLE_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < (1 << 29) else '∞'}"
    return "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--audit_dir", default="../pokemon-worldmodel/data/processed/audit")
    args = ap.parse_args()
    root, audit = Path(args.data_root), Path(args.audit_dir)

    runs = discover_runs(root)
    mode_mass = Counter()
    cb2_comp: dict[int, Counter] = defaultdict(Counter)          # cb2 -> {frames, battle, textbox, dark}
    idle_free_hist = Counter(); idle_free64 = 0
    npc_gfx = Counter(); enemy_species = Counter(); map_mass = Counter()
    fade_frames = 0
    unexplained = []                                             # (runkey, i, fidx, framediff)
    total = 0

    for name, d, kind in runs:
        key = f"{kind}__{name}"
        ram_f = audit / "ram" / f"{key}.v2.npz"
        if not ram_f.exists():
            print(f"  SKIP {key} (no RAM fields)", flush=True); continue
        z = np.load(ram_f)
        fidx_arr = z["fidx"]                  # materialize ONCE (lazy NpzFile re-decompresses per access)
        sem = [json.loads(l) for l in (d / "semantic.jsonl").open()]
        n = len(sem)
        bright = np.load(d / "brightness.npy")[:n]
        tbox = np.load(d / "textbox.npy")[:n]
        fdiff = np.load(d / "framediff.npy")[:n]
        cb2, ib = z["cb2"][:n], z["in_battle"][:n].astype(bool)
        bldy = z["bldy"][:n]
        sp1 = z["species"][:n, 1]
        gfx, act_bits = z["obj_gfx"][:n], z["obj_active"][:n]
        xy = z["obj_xy"][:n]

        # semantic columns -> arrays (the only per-frame python pass)
        px = np.fromiter((s["x"] for s in sem), np.int32, n)
        py = np.fromiter((s["y"] for s in sem), np.int32, n)
        fac = np.fromiter((hash(s["facing"]) & 0xFFFF for s in sem), np.int32, n)
        mp = np.fromiter((hash(s["map"]) & 0xFFFFFFF for s in sem), np.int64, n)
        for s in sem:
            map_mass[s["map"]] += 1

        dark = bright < DARK_THRESH
        total += n
        fade_frames += int(((bldy & 0x1F) > 0).sum())

        # per-cb2 composition (for auto-labeling)
        for v in np.unique(cb2):
            m = cb2 == v
            c = cb2_comp[int(v)]
            c["frames"] += int(m.sum()); c["battle"] += int((m & ib).sum())
            c["textbox"] += int((m & tbox).sum()); c["dark"] += int((m & dark).sum())

        # mode mass (precedence: battle > dark/fade > dialogue > overworld)
        dlg = tbox & ~ib & ~dark
        mode_mass["battle"] += int(ib.sum())
        mode_mass["transition_dark"] += int((dark & ~ib).sum())
        mode_mass["dialogue"] += int(dlg.sum())
        mode_mass["overworld_free"] += int((~ib & ~dark & ~dlg).sum())

        # identity support
        live = (gfx > 0)
        for g, c in zip(*np.unique(gfx[live], return_counts=True)):
            if int(g) not in PLAYER_GFX:
                npc_gfx[int(g)] += int(c)
        for v, c in zip(*np.unique(sp1[ib], return_counts=True)):
            if v > 0:
                enemy_species[int(v)] += int(c)

        # player-static per transition i-1 -> i (vectorized; index 0 = "changed")
        static = np.zeros(n, bool)
        static[1:] = (px[1:] == px[:-1]) & (py[1:] == py[:-1]) & (fac[1:] == fac[:-1]) & (mp[1:] == mp[:-1])

        # FREE idle: stationary player & free-overworld mode -> run lengths of `still`
        free = ~ib & ~dark & ~tbox
        still = free & static
        edges = np.flatnonzero(np.diff(np.concatenate(([0], still.view(np.int8), [0]))))
        for s0, e0 in zip(edges[::2], edges[1::2]):                  # [start, end) of each still-run
            L = int(e0 - s0)
            idle_free_hist[_bucket(L)] += 1
            if L >= 64:
                idle_free64 += L

        # unexplained dynamics: RGB changed, schema-v0 condition static (all vectorized)
        cond_static = static.copy()
        cond_static[1:] &= ((cb2[1:] == cb2[:-1]) & (bldy[1:] == bldy[:-1]) & (tbox[1:] == tbox[:-1])
                            & (ib[1:] == ib[:-1]) & (sp1[1:] == sp1[:-1]) & (act_bits[1:] == act_bits[:-1])
                            & (gfx[1:] == gfx[:-1]).all(1) & (xy[1:] == xy[:-1]).all((1, 2)))
        flag = cond_static & (fdiff > FD_THRESH) & ~dark
        flag[1:] &= ~dark[:-1]
        flag[0] = False
        for i in np.flatnonzero(flag):
            unexplained.append((key, int(i), int(fidx_arr[i]), float(fdiff[i])))
        print(f"  {key}: done", flush=True)

    # --- contact sheet of the most-changing unexplained frames ---
    run_dirs = {f"{kind}__{name}": d for name, d, kind in runs}
    unexplained.sort(key=lambda t: -t[3])
    picks = unexplained[:: max(1, len(unexplained) // 16)][:16]
    tiles = []
    for key, i, fidx, fd in picks:
        d = run_dirs[key]
        metas = [json.loads(l) for l in (d / "frames.jsonl").open()]
        m = metas[i]
        tiles.append(np.load(d / m["chunk"])["frames"][m["chunk_frame_idx"]])
    if tiles:
        rows = [np.concatenate(tiles[r:r+4] + [np.zeros_like(tiles[0])]*(4-len(tiles[r:r+4])), 1)
                for r in range(0, len(tiles), 4)]
        Image.fromarray(np.concatenate(rows, 0)).save(audit / "unexplained_dynamics.png")

    report = {
        "total_frames": total,
        "mode_mass": dict(mode_mass),
        "fade_frames_bldy": fade_frames,
        "free_idle_run_hist": dict(idle_free_hist),
        "free_idle64_frames": idle_free64,
        "free_idle64_fraction": idle_free64 / max(total, 1),
        "npc_gfx_support": {str(k): v for k, v in npc_gfx.most_common()},
        "distinct_npc_gfx": len(npc_gfx),
        "enemy_species_support": {str(k): v for k, v in enemy_species.most_common()},
        "distinct_enemy_species": len(enemy_species),
        "map_mass": dict(map_mass.most_common()),
        "unexplained_frames": len(unexplained),
        "unexplained_fraction": len(unexplained) / max(total, 1),
        "unexplained_samples": [{"run": k, "i": i, "fidx": f, "mae": d} for k, i, f, d in picks],
        "cb2_composition": {f"{v:#x}": dict(c) for v, c in
                            sorted(cb2_comp.items(), key=lambda kv: -kv[1]["frames"])},
    }
    (audit / "audit_report.json").write_text(json.dumps(report, indent=1, default=int))

    print(f"\n=== PHASE-0 AGGREGATE ({total:,} frames) ===")
    for k, v in mode_mass.most_common():
        print(f"  {k:18s} {v:8,d}  {v/total:6.1%}")
    print(f"  fade(BLDY>0)       {fade_frames:8,d}  {fade_frames/total:6.1%}")
    print(f"FREE idle ≥64f: {idle_free64:,} frames ({idle_free64/total:.3%})  hist {dict(idle_free_hist)}")
    print(f"NPC gfx ids: {len(npc_gfx)}  | enemy species: {len(enemy_species)} -> {dict(enemy_species.most_common(10))}")
    print(f"unexplained dynamics: {len(unexplained):,} frames ({len(unexplained)/total:.2%}) -> unexplained_dynamics.png")
    print(f"wrote {audit/'audit_report.json'}")


if __name__ == "__main__":
    main()
