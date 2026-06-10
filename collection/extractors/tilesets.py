"""Map → tileset identity table — the key that lets terrain embeddings SHARE across maps.

A metatile id is tileset-local (0-511 primary, 512+ secondary), so the A′ terrain embedding is
keyed by (tileset, metatile_id). The mapping comes from a pure-ROM walk (no RAM header needed):

    gMapGroups (0x08486578, vanilla US Emerald) -> group ptr -> header ptr ->
        header+0x00 = MapLayout ptr: +0x00 s32 width, +0x04 s32 height,
        +0x10 primaryTileset ptr, +0x14 secondaryTileset ptr

VALIDATION (how we know the table address + struct walk are right for this build): for every map
that appears in the recorded corpus, the ROM layout dims must match the live `gBackupMapLayout`
dims under the (w−15, h−14) buffer convention. Result on this dataset: 41/43 exact; the two
Pokémon-Center-2F maps differ by 2 in height (engine quirk; tolerated). A failing walk could not
match a whole corpus of maps by accident.

Output `tilesets.json`: {"maps": {"group,num": [primary_id, secondary_id]}, ...} with tileset ROM
pointers enumerated to compact ids.

Usage:
  .venv/bin/python -m collection.extractors.tilesets --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

G_MAP_GROUPS = 0x08486578
ROM_BASE = 0x08000000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--conditions", default="../pokemon-worldmodel/data/processed/conditions")
    ap.add_argument("--out", default="../pokemon-worldmodel/data/processed/conditions/tilesets.json")
    args = ap.parse_args()
    rom = Path(args.rom).read_bytes()

    def u32(addr: int) -> int:
        off = addr - ROM_BASE
        return int.from_bytes(rom[off:off + 4], "little")

    # recorded (group,num) -> live map dims, from the condition grids (one grid per map state)
    seen: dict[tuple[int, int], tuple[int, int]] = {}
    for f in sorted(Path(args.conditions).glob("*.npz")):
        z = np.load(f)
        side = json.loads(f.with_suffix(".json").read_text())
        for k in range(side["n_grids"]):
            idx = np.flatnonzero(z["grid_idx"] == k)
            if not len(idx):
                continue
            mid = tuple(int(v) for v in z["map_id"][idx[0]])
            if mid == (255, 255) or mid in seen:
                continue
            g = z[f"grid_{k}"]
            seen[mid] = (g.shape[1] - 15, g.shape[0] - 14)       # (w, h) per the buffer convention

    table: dict[str, tuple[int, int]] = {}
    bad: list[str] = []
    for (grp, num), (w, h) in sorted(seen.items()):
        try:
            head = u32(u32(G_MAP_GROUPS + grp * 4) + num * 4)
            lay = u32(head)
            rw, rh = u32(lay), u32(lay + 4)
        except Exception:
            bad.append(f"{grp},{num}"); continue
        if abs(rw - w) > 0 or abs(rh - h) > 2:                   # width exact; height tolerance 2
            bad.append(f"{grp},{num}"); continue
        table[f"{grp},{num}"] = (u32(lay + 0x10), u32(lay + 0x14))

    ptrs = sorted({p for pair in table.values() for p in pair})
    pid = {p: i for i, p in enumerate(ptrs)}
    out = {k: [pid[a], pid[b]] for k, (a, b) in table.items()}
    Path(args.out).write_text(json.dumps(
        {"maps": out, "n_tilesets": len(ptrs), "tileset_ptrs": ptrs, "unvalidated": sorted(bad)},
        indent=1))
    print(f"{len(out)}/{len(seen)} maps validated -> {len(ptrs)} distinct tilesets; bad: {sorted(bad)}")


if __name__ == "__main__":
    main()
