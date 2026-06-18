"""Verification harness for the identity_grounded condition extractor (PLAN §8.3).

Overlays candidate condition fields on the RECORDED RGB so every field is validated against the
exact frame the model learns to reproduce (the project's 'trust nothing, verify on data' rule).
Currently overlays decoded OAM sprites (VERIFIED: boxes bound the on-screen sprites exactly) and
prints gBattleMons species. To be extended with glyph_grid + hp_fill overlays as those fields land.

VERIFIED (deep dive 2026-06-18):
  • OAM sprite decode correct (boxes bound the on-screen sprites exactly).
  • struct Sprite (gSprites @0x02020630, 0x44/slot, from pokeemerald): pos1 @+0x20, pos2 @+0x24,
    centerToCornerVec(s8) @+0x28/+0x29, invisible = byte 0x3E bit 2, OAM attr0/attr1 @+0x00/+0x02.
    On-screen top-left = pos1 + pos2 + centerToCornerVec. (battle has no overworld camera.)
  • gBattleMons species (player [0], enemy [1] in singles) @0x02024084 + b*0x58, validated on frames.
OPEN — robust battler→sprite labeling NEEDS gBattlerSpriteIds (the engine's battler→current-slot map):
  • position/role heuristics FAIL: the battle OPENING shows the player TRAINER (not the mon); sprites
    SLIDE during the intro; and gSprites SLOTS get REALLOCATED mid-battle (enemy was slot 3 at intro,
    then slot 4/6 later; slot 3 became the PLAYER). So no fixed slot and no position rule is robust.
  • blind ewram scans give FALSE POSITIVES (e.g. 0x020201CD passed a 460-frame track but read [3,3]
    off-screen elsewhere). Resolve gBattlerSpriteIds from the AUTHORITATIVE pokeemerald/retail address
    (gBattleMons matches the decomp, so decomp addrs apply) and validate on CALM (non-animation) frames.

Run (conda pokemon-wm or harness venv; recorded blobs need no mgba):
    python -m collection.extractors.verify_overlay <run_dir> [out_dir]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from collection.extractors.ram import iter_states
from collection.extractors.battle import in_battle

OAM = 0x07000000
GBATTLEMONS = 0x02024084
BAT_STRIDE = 0x58
_DIMS = {(0, 0): (8, 8), (0, 1): (16, 16), (0, 2): (32, 32), (0, 3): (64, 64),
         (1, 0): (16, 8), (1, 1): (32, 8), (1, 2): (32, 16), (1, 3): (64, 32),
         (2, 0): (8, 16), (2, 1): (8, 32), (2, 2): (16, 32), (2, 3): (32, 64)}


def oam_sprites(st, min_dim: int = 8):
    """Decode visible OBJ sprites from OAM -> list of (x, y, w, h). x is the 9-bit wrapping X."""
    out = []
    for i in range(128):
        a0 = st.u16(OAM + i * 8); a1 = st.u16(OAM + i * 8 + 2)
        if not (a0 >> 8) & 1 and (a0 >> 9) & 1:           # OBJ disabled (non-affine + disable bit)
            continue
        y = a0 & 0xFF; x = a1 & 0x1FF; x = x - 512 if x >= 512 else x
        w, h = _DIMS.get(((a0 >> 14) & 3, (a1 >> 14) & 3), (8, 8))
        if w >= min_dim and h >= min_dim and -w < x < 240 and -h < y < 160:
            out.append((x, y, w, h))
    return out


def battle_species(st):
    return [st.u16(GBATTLEMONS + b * BAT_STRIDE) for b in range(4)]


def _rgb_reader(run: Path):
    meta = {json.loads(l)["emulator_frame_idx"]: json.loads(l) for l in (run / "frames.jsonl").open()}
    cache: dict = {}
    def rgb(f):
        m = meta[f]; cp = str(run / m["chunk"])
        cache.setdefault(cp, np.load(cp)["frames"])
        return cache[cp][m["chunk_frame_idx"]]
    return rgb


def overlay_battles(run_dir: str, out_dir: str = "/tmp", n: int = 4):
    from PIL import Image, ImageDraw
    run = Path(run_dir); rgb = _rgb_reader(run); saved = 0
    for f, st in iter_states(run):
        if not in_battle(st):
            continue
        big = [s for s in oam_sprites(st, 32)]
        if len(big) < 2:
            continue
        im = Image.fromarray(rgb(f).astype(np.uint8)).resize((480, 320), Image.NEAREST)
        d = ImageDraw.Draw(im)
        for (x, y, w, h) in big:
            d.rectangle([x * 2, y * 2, (x + w) * 2, (y + h) * 2], outline=(255, 0, 0), width=2)
        p = Path(out_dir) / f"verify_battle_{f}.png"; im.save(p)
        print(f"frame {f}: sprites={sorted(big, key=lambda s: s[1])} species={battle_species(st)} -> {p}")
        saved += 1
        if saved >= n:
            break


if __name__ == "__main__":
    overlay_battles(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "/tmp")
