"""Data-closure Phase D: channel-saturation audit — every condition channel vs its encoding.

The text-72 bug (the model conditioned on chars[0:72] of median-218-char messages — unreadable
generated dialogue) was found by EYE after 10 minutes of live play. This audit finds that whole
CLASS automatically: for every channel of the conditions corpus, measure the fraction of frames
where the value is clamped, saturated, out of the model encoder's vocabulary, or out of its
normalization range. Anything red here is a conditioning gap NO amount of data fixes.

Checked against the model-side constants (pokemon_worldmodel/v1/encoder.py):
  LC=72 chars · N_GFX=256 · N_SPECIES=448 · N_MOVES=384 · glob normalizers (bld/255,/255,/16).

Usage:
  .venv/bin/python -m collection.audits.saturation --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# model-side encoding constants (mirror of v1/encoder.py; the report breaks if these drift)
LC, N_GFX, N_SPECIES, N_MOVES, N_METATILE = 72, 256, 448, 384, 1025
BLD_NORM = (255.0, 255.0, 16.0)                       # glob divides BLDCNT/BLDALPHA/BLDY by these
SCREEN = (-64, 304, -64, 224)                         # plausible on/near-screen anchor box (px)


def audit(cond_dir: Path) -> dict:
    n_total = 0
    c: dict[str, float] = {k: 0 for k in (
        "text_frames", "text_reveal_gt_LC", "text_len_gt_LC", "text_reveal_u16_clamp",
        "textbox_without_text", "player_xy_ge_s16max", "player_screen_offbox",
        "ent_frames", "ent_screen_offbox", "ent_gfx_oov",
        "bat_frames", "bat_species_oov", "bat_moves_oov", "bat_level_gt100",
        "bld0_gt_norm", "bld1_gt_norm", "bld2_gt_norm",
        "terrain_masked", "map_unknown", "mode_other")}
    maxes: dict[str, int] = {k: 0 for k in ("player_xy", "player_screen", "ent_gfx",
                                            "bat_species", "bat_moves", "bld0", "bld1", "bld2",
                                            "text_reveal", "mode")}
    text_lens: list[int] = []
    for f in sorted(cond_dir.glob("*.npz")):
        z = np.load(f)
        side = json.loads(f.with_suffix(".json").read_text())
        n = len(z["mode"]); n_total += n
        lens = np.array([len(t) for t in side["texts"]], np.int64) if side["texts"] else np.zeros(1, np.int64)
        text_lens.extend(int(x) for x in lens if x)

        tid, rev = z["text_id"], z["text_reveal"]
        has = tid >= 0
        c["text_frames"] += int(has.sum())
        c["text_reveal_gt_LC"] += int((rev[has] > LC).sum())
        if side["texts"]:
            c["text_len_gt_LC"] += int((lens[tid[has]] > LC).sum())
        c["text_reveal_u16_clamp"] += int((rev == 65535).sum())
        wm = np.unpackbits(z["win_mask"], axis=1)[:, :600].reshape(-1, 20, 30)
        c["textbox_without_text"] += int((wm[:, 14:20].any(axis=(1, 2)) & ~has).sum())

        c["player_xy_ge_s16max"] += int((z["player_xy"] >= 32767).any(axis=1).sum())
        ps = z["player_screen"]
        c["player_screen_offbox"] += int(((ps[:, 0] < SCREEN[0]) | (ps[:, 0] > SCREEN[1]) |
                                          (ps[:, 1] < SCREEN[2]) | (ps[:, 1] > SCREEN[3])).sum())
        ev = z["ent_valid"] > 0
        c["ent_frames"] += int(ev.sum())
        es = z["ent_screen"]
        off = ((es[..., 0] < SCREEN[0]) | (es[..., 0] > SCREEN[1]) |
               (es[..., 1] < SCREEN[2]) | (es[..., 1] > SCREEN[3]))
        c["ent_screen_offbox"] += int((off & ev).sum())
        c["ent_gfx_oov"] += int(((z["ent_gfx"] >= N_GFX) & ev).sum())

        bv = z["bat_valid"] > 0
        c["bat_frames"] += int(bv.sum())
        c["bat_species_oov"] += int(((z["bat_species"] >= N_SPECIES) & bv).sum())
        c["bat_moves_oov"] += int(((z["bat_moves"] >= N_MOVES) & bv[..., None]).sum())
        c["bat_level_gt100"] += int(((z["bat_level"] > 100) & bv).sum())

        bld = z["bld"].astype(np.int64)
        for i in range(3):
            c[f"bld{i}_gt_norm"] += int((bld[:, i] > BLD_NORM[i]).sum())
            maxes[f"bld{i}"] = max(maxes[f"bld{i}"], int(bld[:, i].max()))

        c["terrain_masked"] += int((z["grid_idx"] < 0).sum())
        c["map_unknown"] += int(np.all(z["map_id"] == 255, axis=1).sum())
        c["mode_other"] += int((z["mode"] >= 4).sum())

        maxes["player_xy"] = max(maxes["player_xy"], int(z["player_xy"].max()))
        maxes["player_screen"] = max(maxes["player_screen"], int(np.abs(ps).max()))
        maxes["ent_gfx"] = max(maxes["ent_gfx"], int(z["ent_gfx"].max()))
        maxes["bat_species"] = max(maxes["bat_species"], int(z["bat_species"].max()))
        maxes["bat_moves"] = max(maxes["bat_moves"], int(z["bat_moves"].max()))
        maxes["text_reveal"] = max(maxes["text_reveal"], int(rev.max()))
        maxes["mode"] = max(maxes["mode"], int(z["mode"].max()))

    tl = np.array(text_lens) if text_lens else np.zeros(1)
    return {"frames": n_total, "counts": c, "maxes": maxes,
            "text_len_p50": int(np.percentile(tl, 50)), "text_len_p90": int(np.percentile(tl, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.data_root)
    rep = audit(root / "processed/conditions")
    out = Path(args.out) if args.out else root / "processed/audit/saturation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))

    n, c = rep["frames"], rep["counts"]
    den = {"text_reveal_gt_LC": "text_frames", "text_len_gt_LC": "text_frames",
           "ent_screen_offbox": "ent_frames", "ent_gfx_oov": "ent_frames",
           "bat_species_oov": "bat_frames", "bat_moves_oov": "bat_frames",
           "bat_level_gt100": "bat_frames"}
    print(f"=== CHANNEL SATURATION over {n:,} frames "
          f"(text len p50/p90 = {rep['text_len_p50']}/{rep['text_len_p90']}) ===")
    for k, v in c.items():
        if k in ("text_frames", "ent_frames", "bat_frames"):
            print(f"  {k:26s} {v:>12,}")
            continue
        d = c.get(den.get(k, ""), n) or n
        flag = " <-- RED" if v / d > 0.01 else ""
        print(f"  {k:26s} {v:>12,}  ({v / d:7.3%}){flag}")
    print("  maxes:", rep["maxes"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
