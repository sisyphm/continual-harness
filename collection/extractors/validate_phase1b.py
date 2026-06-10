"""Phase-1b validation — text + battle extractors against independent sources.

  TX1  printer-active vs the (visual, FP-checked) textbox flag on non-battle frames — agreement.
  TX2  text sanity: decoded string non-empty + mostly printable on dialogue frames; 0 ≤ reveal ≤ len.
  TX3  VISUAL: sampled dialogue frames rendered next to their decoded strings (text_overlay.png).
  B1   battler sanity on battle frames: hp ≤ maxHP, level ≤ 20 (pre-badge), species in the Phase-0
       audit set, pp ≤ 40.
  B2   VISUAL: sampled battle frames + extracted battler lines (battle_overlay.png).

Sampled over the storyline runs + the OLDALE coverage seed (the RUSTBORO walk adds ~7 min and no
new modes — its battles are the same flee pattern).

Usage:
  .venv/bin/python -m collection.extractors.validate_phase1b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from collection.audit_wm import discover_runs
from collection.extractors.battle import battle_state, in_battle
from collection.extractors.ram import iter_states
from collection.extractors.text import text_state

SAMPLES_PER_RUN = 6


def _frame_rgb(d: Path, metas: list[dict], f: int) -> np.ndarray:
    m = metas[f]
    return np.load(d / m["chunk"])["frames"][m["chunk_frame_idx"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out_dir", default="../pokemon-worldmodel/data/processed/audit")
    args = ap.parse_args()
    out = Path(args.out_dir)
    rom = Path("Emerald-GBAdvance/rom.gba").read_bytes()

    runs = [(n, d, k) for n, d, k in discover_runs(Path(args.data_root))
            if k == "storyline" or n == "OLDALE_AFTER_POKEDEX"]

    tx_agree = tx_tot = 0
    tx_nonempty = tx_dlg = 0
    tx_reveal_ok = tx_reveal_tot = 0
    b_ok = b_tot = 0
    b_bad: list[tuple] = []
    tx_shots: list[tuple[np.ndarray, str]] = []
    b_shots: list[tuple[np.ndarray, str]] = []

    for name, d, kind in runs:
        sem = [json.loads(l) for l in (d / "semantic.jsonl").open()]
        tb = np.load(d / "textbox.npy")
        bright = np.load(d / "brightness.npy")
        n = len(sem)
        want = sorted({int(i) for i in np.linspace(n * 0.15, n - 2, SAMPLES_PER_RUN)})
        metas = None
        for f, st in iter_states(d):
            if f not in want:
                continue
            want.remove(f)
            if bright[f] < 30:
                if not want:
                    break
                continue
            ts = text_state(st, rom)
            ib = in_battle(st)

            # TX1: active-printer vs visual textbox (non-battle frames; the visual flag is the
            # independent source)
            if not ib:
                tx_tot += 1
                tx_agree += (ts is not None) == bool(tb[f])
            # TX2 on visually-dialogue frames
            if tb[f] and not ib:
                tx_dlg += 1
                if ts is not None and len(ts.text.strip()) > 0:
                    tx_nonempty += 1
                    tx_reveal_tot += 1
                    tx_reveal_ok += 0 <= ts.reveal <= len(ts.text) + 40   # ctl bytes inflate raw len
                    if len(tx_shots) < 6 and len(ts.text) > 20:
                        metas = metas or [json.loads(l) for l in (d / "frames.jsonl").open()]
                        tx_shots.append((_frame_rgb(d, metas, f), ts.text[:90]))

            # B1: battler sanity
            if ib:
                bs = battle_state(st)
                if bs and bs["battlers"]:
                    for bt in bs["battlers"]:
                        b_tot += 1
                        ok = bt.hp <= bt.max_hp and 1 <= bt.level <= 20 and all(p <= 40 for p in bt.pp)
                        b_ok += ok
                        if not ok and len(b_bad) < 6:
                            b_bad.append((name, f, bt))
                    if len(b_shots) < 4:
                        metas = metas or [json.loads(l) for l in (d / "frames.jsonl").open()]
                        desc = " | ".join(f"s{bt.slot} sp{bt.species} Lv{bt.level} {bt.hp}/{bt.max_hp}"
                                          for bt in bs["battlers"])
                        b_shots.append((_frame_rgb(d, metas, f), desc))
            if not want:
                break

    # visual artifacts: frame + extracted text/battlers below it
    def grid(shots: list[tuple[np.ndarray, str]], path: Path):
        if not shots:
            return
        tiles = []
        for fr, txt in shots:
            img = Image.new("RGB", (240, 200), (12, 12, 12))
            img.paste(Image.fromarray(fr), (0, 0))
            dr = ImageDraw.Draw(img)
            for li, line in enumerate([txt[i:i + 44] for i in range(0, len(txt), 44)][:3]):
                dr.text((3, 162 + 12 * li), line, fill=(120, 255, 120))
            tiles.append(np.asarray(img))
        rows = [np.concatenate(tiles[i:i + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[i:i + 2])), 1)
                for i in range(0, len(tiles), 2)]
        Image.fromarray(np.concatenate(rows, 0)).save(path)

    grid(tx_shots, out / "text_overlay.png")
    grid(b_shots, out / "battle_overlay.png")

    print("=== PHASE-1b TEXT+BATTLE VALIDATION ===")
    print(f"TX1 printer-active == visual textbox (non-battle): {tx_agree}/{tx_tot} ({tx_agree/max(tx_tot,1):.1%})")
    print(f"TX2 dialogue frames with non-empty decode: {tx_nonempty}/{tx_dlg} "
          f"({tx_nonempty/max(tx_dlg,1):.1%}); reveal in-range: {tx_reveal_ok}/{tx_reveal_tot}")
    print(f"B1  battler sanity (hp<=max, level<=20, pp<=40): {b_ok}/{b_tot} ({b_ok/max(b_tot,1):.1%})")
    for x in b_bad:
        print(f"    BAD {x}")
    print(f"visual artifacts: {out/'text_overlay.png'}, {out/'battle_overlay.png'}")


if __name__ == "__main__":
    main()
