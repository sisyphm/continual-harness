"""M7 acceptance — re-measure UNEXPLAINED DYNAMICS against the REAL A′ conditions.

Phase-0's proxy condition left 44.6% of frames with unconditioned pixel change; the A′ schema was
designed to absorb the bulk (entity sub-tile motion via screen anchors; battle phase via comm bytes;
text reveal; window geometry; fades). This measures what remains: frames where pixels change
(framediff > thresh) while EVERY A′ condition field is static — the true learned tail
(ambient tile animation + move-animation unfolding).

Usage:
  .venv/bin/python -m collection.extractors.unexplained_v2 --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collection.corpus import discover_runs

FD_THRESH = 2.0
FIELDS_EQ = ("mode", "in_battle", "cam", "bld", "win_mask", "player_xy", "player_facing",
             "player_moving", "player_screen", "map_id", "grid_idx", "text_state", "text_id",
             "text_reveal", "bat_valid", "bat_species", "bat_hp", "bat_maxhp", "bat_status",
             "bat_moves", "bat_pp", "bat_type", "bat_comm",
             "ent_valid", "ent_gfx", "ent_facing", "ent_moving", "ent_xy", "ent_screen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--conditions", default="../pokemon-worldmodel/data/processed/conditions")
    args = ap.parse_args()
    cond_dir = Path(args.conditions)

    total = unexplained = changing = 0
    by_mode = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0]}    # mode -> [changing, unexpl]
    for name, d, kind in discover_runs(Path(args.data_root)):
        f = cond_dir / f"{kind}__{name}.npz"
        fd_p = d / "framediff.npy"
        if not f.exists() or not fd_p.exists():
            continue
        z = np.load(f)
        n = len(z["mode"])
        if n < 2:
            continue
        fd = np.load(fd_p)[:n]
        static = np.ones(n, bool)
        for k in FIELDS_EQ:
            a = z[k]
            eq = (a[1:] == a[:-1])
            static[1:] &= eq.reshape(n - 1, -1).all(1)
        bright_p = d / "brightness.npy"
        dark = np.load(bright_p)[:n] < 30 if bright_p.exists() else np.zeros(n, bool)
        chg = (fd > FD_THRESH) & ~dark
        chg[1:] &= ~dark[:-1]
        chg[0] = False
        un = chg & static
        total += n; changing += int(chg.sum()); unexplained += int(un.sum())
        for m in by_mode:
            sel = z["mode"] == m
            by_mode[m][0] += int((chg & sel).sum()); by_mode[m][1] += int((un & sel).sum())

    print(f"frames {total:,} | pixel-changing {changing:,} ({changing/total:.1%}) | "
          f"UNEXPLAINED under A′ conditions: {unexplained:,} ({unexplained/total:.2%} of corpus, "
          f"{unexplained/max(changing,1):.1%} of changing frames)")
    names = {0: "overworld", 1: "battle", 2: "transition", 3: "intro", 4: "other"}
    for m, (c, u) in by_mode.items():
        if c:
            print(f"  {names[m]:10s}: changing {c:8,d} -> unexplained {u:8,d} ({u/c:.1%})")


if __name__ == "__main__":
    main()
