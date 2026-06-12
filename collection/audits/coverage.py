"""Data-closure Phase B: coverage accounting — the corpus mapped onto the ROM manifest.

Every enumerable axis of the coverage manifest (rom_manifest.py) becomes rows with a support
floor; this audit measures the corpus support for each row and emits ONE report where every row
is green or red. Collection (Phase C) is driven by the red rows and stops when the report is
green — never again by noticing bugs.

Axes (and what "support" means):
  terrain   frames each (tileset, metatile) was ON SCREEN (camera-window replay, deduped by
            (grid, player-pos) state so 1.4M frames cost ~thousands of window slices)
  warps     traversals of each manifest warp/connection pair, per direction
  entities  visible frames per enumerated in-scope graphics id × facing
  battle    frames per species × side; outcome events (faint / level-up / battle-end)
  text      frames each charset glyph was on a revealed screen position
  menus     PARTIAL: distinct window-geometry patterns (the conditions are identity-free for UI
            by design — menu-screen identity is only measurable via the Phase-C crawler's labels)

Usage:
  .venv/bin/python -m collection.audits.coverage --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

# support floors (frames at 60 fps unless stated) — the definition of "covered"
FLOOR = {"terrain": 100, "warp_traversals": 5, "entity_gfx": 2000, "entity_gfx_facing": 300,
         "species_enemy": 3000, "species_player": 3000, "outcome_events": 10, "glyph": 2000}
SCREEN_MH, SCREEN_MW = 10, 15


def camera_topleft(px, py, mw, mh):
    return (min(max(px - 7, 0), max(0, mw - SCREEN_MW)),
            min(max(py - 5, 0), max(0, mh - SCREEN_MH)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.data_root)
    M = json.loads((root / "processed/coverage_manifest.json").read_text())
    tilesets = json.loads((root / "processed/conditions/tilesets.json").read_text())["maps"]
    starts = {r["run_key"]: set(r["clip_starts"])
              for r in json.loads((root / "processed/corpus_manifest.json").read_text())["runs"]}

    terrain, warp_n = Counter(), Counter()
    gfx_frames, gfx_face = Counter(), Counter()
    sp_player, sp_enemy = Counter(), Counter()
    outcomes = Counter()
    glyphs = Counter()
    win_patterns = Counter()

    for f in sorted((root / "processed/conditions").glob("*.npz")):
        z = np.load(f)
        side = json.loads(f.with_suffix(".json").read_text())
        grids = [z[f"grid_{k}"] for k in range(side["n_grids"])]
        texts = side["texts"]
        resets = starts.get(f.stem, set())
        n = len(z["mode"])

        # --- terrain: dedupe consecutive identical (grid_idx, player_xy) camera states
        gi, pxy, mid = z["grid_idx"], z["player_xy"], z["map_id"]
        state = np.stack([gi.astype(np.int64), pxy[:, 0], pxy[:, 1]], 1)
        change = np.ones(n, bool); change[1:] = np.any(state[1:] != state[:-1], 1)
        idxs = np.flatnonzero(change)
        counts = np.diff(np.append(idxs, n))
        for i, cnt in zip(idxs, counts):
            g = int(gi[i])
            if g < 0:
                continue
            key = f"{mid[i,0]},{mid[i,1]}"
            ts = tilesets.get(key)
            if ts is None:
                continue
            grid = grids[g]
            mh, mw = grid.shape[0] - 14, grid.shape[1] - 15
            cx, cy = camera_topleft(int(pxy[i, 0]), int(pxy[i, 1]), mw, mh)
            win = grid[cy + 7:cy + 7 + SCREEN_MH, cx + 7:cx + 7 + SCREEN_MW] & 0x3FF
            for loc in np.unique(win):
                t = ts[0] if loc < 512 else ts[1]
                terrain[(t, int(loc) if loc < 512 else int(loc) - 512)] += int(cnt)

        # --- warps: transition instances (reset±1 excluded)
        prev = None
        for i in range(n):
            if i in resets or (i - 1) in resets:
                prev = None
            a, b = int(mid[i, 0]), int(mid[i, 1])
            if (a, b) == (255, 255):
                continue
            k = f"{a},{b}"
            if prev is not None and prev != k:
                warp_n[(prev, k)] += 1
            prev = k

        # --- entities (battle frames excluded, as the model sees them)
        ev = (z["ent_valid"] > 0) & (z["in_battle"][:, None] == 0)
        gfx, fac = z["ent_gfx"], z["ent_facing"]
        for g_, c_ in zip(*np.unique(gfx[ev], return_counts=True)):
            if g_ > 0:
                gfx_frames[int(g_)] += int(c_)
        both = gfx.astype(np.int64) * 8 + fac
        for v, c_ in zip(*np.unique(both[ev], return_counts=True)):
            if v // 8 > 0:
                gfx_face[(int(v // 8), int(v % 8))] += int(c_)

        # --- battle: species frames + outcome events
        bv, bs = z["bat_valid"] > 0, z["bat_species"]
        for slot, ctr in ((0, sp_player), (2, sp_player), (1, sp_enemy), (3, sp_enemy)):
            v = bs[:, slot][bv[:, slot]]
            for s_, c_ in zip(*np.unique(v, return_counts=True)):
                if s_ > 0:
                    ctr[int(s_)] += int(c_)
        hp, lvl = z["bat_hp"], z["bat_level"]
        for slot in range(4):
            m = bv[:, slot]
            h = hp[:, slot].astype(np.int64)
            faint = m[1:] & m[:-1] & (h[1:] == 0) & (h[:-1] > 0)
            outcomes["faint"] += int(faint.sum())
            lv = lvl[:, slot].astype(np.int64)
            outcomes["level_up"] += int((m[1:] & m[:-1] & (lv[1:] == lv[:-1] + 1)).sum())
        ib = z["in_battle"].astype(bool)
        outcomes["battle_end"] += int((ib[:-1] & ~ib[1:]).sum())

        # --- text glyphs: revealed chars weighted by frames on screen
        tid, rev = z["text_id"], z["text_reveal"]
        has = np.flatnonzero(tid >= 0)
        for i in has:
            s = texts[int(tid[i])][:int(rev[i])]
            for ch in set(s):
                glyphs[ch] += 1

        # --- menus (proxy): distinct window geometries
        wm = z["win_mask"]
        for p, c_ in zip(*np.unique(wm, axis=0, return_counts=True)):
            if p.any():
                win_patterns[p.tobytes()] += int(c_)

    # ---- assemble the green/red report against the manifest -----------------------------------
    scope = set(M["scope_maps"])
    maps = M["maps"]
    rows: dict[str, list] = {}

    # terrain universe = every metatile present in the corpus map grids (the grid IS the full map)
    rows["terrain"] = [{"key": f"ts{t}/mt{m_}", "support": c, "ok": c >= FLOOR["terrain"]}
                       for (t, m_), c in sorted(terrain.items())]
    # warps: manifest pairs in scope, per direction
    wrows = []
    for a in scope:
        if a not in maps:
            continue
        dsts = {w["dst_map"] for w in maps[a]["warps"]} | {c["dst_map"] for c in maps[a]["connections"]}
        for b in sorted(dsts & scope):
            c = warp_n.get((a, b), 0)
            wrows.append({"key": f"{a}->{b}", "support": c, "ok": c >= FLOOR["warp_traversals"]})
    rows["warps"] = wrows
    enum_gfx = sorted({o["gfx"] for k in scope if k in maps for o in maps[k]["objects"]})
    rows["entities"] = [{"key": f"gfx{g}", "support": gfx_frames.get(g, 0),
                         "ok": gfx_frames.get(g, 0) >= FLOOR["entity_gfx"]} for g in enum_gfx]
    # enemy-species universe, ACCESS-AWARE (policy, 2026-06): 'water'/'rock' tables need Surf /
    # Rock Smash (post-badge), the Old Rod only draws fish slots 0-1 (Good/Super Rod are
    # post-badge), and Route 115 (0,30)'s grass sits behind a Surf-only elevation wall
    # (navigator-proven: the walkable reach from the Rustboro entrance is a 300-cell pocket
    # whose every frontier edge is a concrete 3->1 elevation mismatch). What a pre-badge player
    # cannot encounter, the model never has to render.
    surf_gated_land = {"0,30"}
    wild_sp = sorted({sp
                      for k in scope for kind, t in M["wild"].get(k, {}).items()
                      if kind not in ("water", "rock") and not (kind == "land" and k in surf_gated_land)
                      for sp, *_ in (t["mons"][:2] if kind == "fish" else t["mons"])})
    party_sp = sorted(set(M["party_space"]["starters"]) | set(M["party_space"]["catchable_in_scope"]))
    rows["species_enemy"] = [{"key": f"sp{s}", "support": sp_enemy.get(s, 0),
                              "ok": sp_enemy.get(s, 0) >= FLOOR["species_enemy"]} for s in wild_sp]
    rows["species_player"] = [{"key": f"sp{s}", "support": sp_player.get(s, 0),
                               "ok": sp_player.get(s, 0) >= FLOOR["species_player"]} for s in party_sp]
    # trainers: enemy (species, level) frames matched against enumerated party leads
    trainer_ids = sorted({o["trainer_id"] for k in scope if k in maps
                          for o in maps[k]["objects"] if o.get("trainer_id")})
    lead = {t["id"]: tuple(t["party"][0]) for t in M["trainers"] if t["party"]}
    sl_frames = Counter()
    for f in sorted((root / "processed/conditions").glob("*.npz")):
        z = np.load(f)
        bv, bs, bl = z["bat_valid"] > 0, z["bat_species"], z["bat_level"]
        m1 = bv[:, 1]
        for (s_, l_), c_ in zip(*[list(x) for x in np.unique(
                np.stack([bs[:, 1][m1], bl[:, 1][m1]], 1), axis=0, return_counts=True)] if m1.any() else ([], [])):
            sl_frames[(int(s_), int(l_))] += int(c_)
    rows["trainers"] = [{"key": f"tr{t}", "support": sl_frames.get(lead.get(t, (0, 0)), 0),
                         "ok": sl_frames.get(lead.get(t, (0, 0)), 0) >= 500} for t in trainer_ids]

    # menus: labeled frames from the crawler's sidecars (labels.jsonl per run)
    label_frames = Counter()
    for lab in (root / "behaviors").glob("*/labels.jsonl"):
        for line in lab.open():
            r_ = json.loads(line)
            label_frames[r_["label"]] += max(0, r_["end"] - r_["start"])
    rows["menus"] = [{"key": lb, "support": label_frames.get(lb, 0),
                      "ok": label_frames.get(lb, 0) >= 1500}
                     for lb in sorted(set(label_frames) |
                                      {"start_menu", "pokedex", "party", "summary_info",
                                       "bag_items", "trainer_card", "save_dialog", "options"})]

    charset = json.loads((root / "processed/conditions/charset.json").read_text())["charset"]
    rows["glyphs"] = [{"key": repr(ch), "support": glyphs.get(ch, 0),
                       "ok": glyphs.get(ch, 0) >= FLOOR["glyph"]} for ch in sorted(set(charset.values()))]
    oc = root / "processed/audit/outcomes_report.json"               # audits.outcomes (party
    measured = json.loads(oc.read_text()) if oc.exists() else {}     # walk; conditions can't see it)
    rows["outcomes"] = [{"key": k, "support": int(v), "ok": v >= FLOOR["outcome_events"]}
                        for k, v in sorted(outcomes.items())] + \
                       [{"key": "catch", "support": measured.get("catches", 0),
                         "ok": measured.get("catches", 0) >= FLOOR["outcome_events"]},
                        {"key": "evolution", "support": measured.get("evolutions", 0),
                         "ok": measured.get("evolutions", 0) >= 3}]

    report = {"floors": FLOOR,
              "summary": {ax: {"rows": len(rs), "red": sum(not r["ok"] for r in rs)}
                          for ax, rs in rows.items()},
              "menus_proxy_distinct_window_patterns": len(win_patterns),
              "rows": rows}
    out = Path(args.out) if args.out else root / "processed/audit/coverage_report.json"
    out.write_text(json.dumps(report, indent=1))
    print("=== COVERAGE vs MANIFEST (red = below floor) ===")
    for ax, s in report["summary"].items():
        print(f"  {ax:15s}: {s['rows']:>5} rows, {s['red']:>5} RED")
    print(f"  menus (proxy) : {len(win_patterns)} distinct window geometries")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
