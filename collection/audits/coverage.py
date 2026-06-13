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


def reachable_window_metatiles(grid: np.ndarray, beh: np.ndarray | None,
                               seeds: set[tuple[int, int]]) -> set[int]:
    """The access-aware terrain UNIVERSE: every metatile the live camera can show from a position
    the player ACTUALLY STOOD ON. Each observed seed contributes its clamped camera window over
    the de-bordered interior; the union is the universe.

    NOT a collision BFS — that walked through collision-open but SCRIPT-LOCKED doors (the
    Petalburg Gym guard blocks the inner rooms until 4 badges) and through Surf/elevation gates,
    inventing 'red forever' tiles the pre-badge player can never put on screen. Observed-seed
    dilation is exactly what the game rendered: after the exhaustive dwell pass the player stands
    on every reachable cell, so accessible terrain is fully covered and only genuinely
    unreachable rooms drop out. The 7-tile border is excluded (stale gBackupMapLayout from the
    prior map; reading it invented ~130 phantom metatiles per interior)."""
    h, w = grid.shape
    mh, mw = h - 14, w - 15
    on_screen = np.zeros((h, w), bool)
    for px, py in seeds:
        if not (0 <= py < mh and 0 <= px < mw):              # drop transition-frame garbage seeds
            continue                                          # (coords valid only on the real map)
        cx, cy = camera_topleft(px, py, mw, mh)
        on_screen[cy + 7:cy + 7 + SCREEN_MH, cx + 7:cx + 7 + SCREEN_MW] = True
    interior = np.zeros((h, w), bool)
    interior[7:7 + mh, 7:7 + mw] = True
    return {int(v) for v in np.unique(grid[on_screen & interior] & 0x3FF)}


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
    reach_grids: dict = {}                                    # map key -> one full grid
    reach_seeds: dict = {}                                    # map key -> observed player (x,y)

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
            # store ONE grid per map for the universe — but ONLY a frame whose de-bordered dims
            # match the manifest. Warps flip MAP_BANK/NUMBER a few frames before gBackupMapLayout
            # rebuilds, so a transition frame carries the PREVIOUS map's (wrong-size) layout under
            # the new map id; read through the new tileset it invents phantom metatiles (the
            # Rustboro Gym got a 40x60 outdoor grid → 229 foreign tiles vs its real 60).
            mfm = M["maps"].get(key)
            if key not in reach_grids and mfm and \
                    grid.shape[1] - 15 == mfm["width"] and grid.shape[0] - 14 == mfm["height"]:
                reach_grids[key] = grid.copy()
            reach_seeds.setdefault(key, set()).add((int(pxy[i, 0]), int(pxy[i, 1])))

        # --- warps: transition instances (reset±1 excluded); INTRA-map warps (e.g. the gym's
        # 8,1->8,1 chamber doors) never change the map id — detect them as same-map teleports
        # (player jumps >3 tiles in one frame)
        prev = None
        prev_xy = None
        for i in range(n):
            if i in resets or (i - 1) in resets:
                prev = None
                prev_xy = None
            a, b = int(mid[i, 0]), int(mid[i, 1])
            if (a, b) == (255, 255):
                continue
            k = f"{a},{b}"
            xy = (int(pxy[i, 0]), int(pxy[i, 1]))
            if prev is not None and prev != k:
                warp_n[(prev, k)] += 1
            elif (prev == k and prev_xy is not None
                  and abs(xy[0] - prev_xy[0]) + abs(xy[1] - prev_xy[1]) > 3):
                warp_n[(k, k)] += 1
            prev = k
            prev_xy = xy

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

    # terrain universe = metatiles VISIBLE FROM REACHABLE cells (access-aware: elevation+ledge
    # BFS from observed player positions, dilated by the camera window — see
    # reachable_window_metatiles). Anything actually seen on screen joins defensively.
    from types import SimpleNamespace
    from collection.navigator import MapKnowledge
    mk = MapKnowledge(manifest_json=str(root / "processed/coverage_manifest.json"))
    universe: set = set()
    for key, grid in reach_grids.items():
        ts = tilesets.get(key)
        if ts is None:
            continue
        g_, n_ = (int(v) for v in key.split(","))
        beh = mk.behaviors(SimpleNamespace(map_group=g_, map_num=n_, grid=grid))
        for loc in reachable_window_metatiles(grid, beh, reach_seeds[key]):
            universe.add((ts[0], loc) if loc < 512 else (ts[1], loc - 512))
    universe |= set(terrain)                                  # seen-on-screen is in by definition
    rows["terrain"] = [{"key": f"ts{t}/mt{m_}", "support": terrain.get((t, m_), 0),
                        "ok": terrain.get((t, m_), 0) >= FLOOR["terrain"]}
                       for (t, m_) in sorted(universe)]
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
    # entity-gfx universe, ACCESS-AWARE: ids 240-255 are OBJ_EVENT_GFX_VAR_* slots (resolved at
    # runtime from VARs — the live sprite is counted under its REAL id, so the placeholder rows
    # can never match). gfx whose only in-scope placements sit in Surf-gated areas (Route 103's
    # east bank trainers @ (67,9)/(36,6)/(36,13); Route 115 beyond the elevation wall @ (10,15)/
    # (29,50)) are out with their maps' gating. FLAG-GATED-ABSENT pre-badge, live-verified
    # (object in manifest, live table empty at the spot): 211 (Briney's cottage pair — he moves
    # in after the post-badge Peeko rescue) and 223 (an unbought house decoration on 2,3).
    gated_gfx = {42, 43, 66, 52, 86, 211, 223}
    enum_gfx = sorted({o["gfx"] for k in scope if k in maps for o in maps[k]["objects"]
                       if o["gfx"] < 240} - gated_gfx)
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
    # trainers: enemy (species, level) frames matched against enumerated party leads.
    # ACCESS-AWARE (same gating as species): Route 115 (0,30) and Route 103's east bank (0,18)
    # hold ALL their trainers behind Surf — the hunt collector BFS-proves each one unreachable
    # (instant clean-grid dead-ends from the pre-badge entrances). Petalburg Gym (8,1)'s seven
    # trainers are FLAG-GATED absent pre-badge (live ObjectEvent table holds only local_id 1;
    # trainer objects 2-8 never spawn before badge 5). All out of the pre-badge space.
    gated_trainer_maps = {"0,30", "0,18", "8,1"}
    # tr273 JERRY / tr605 JANICE (Route 116): live-verified ABSENT — standing adjacent to their
    # template cells, the ObjectEvent table holds the map's other NPCs but not locals 16/18
    # (they spawn after a later story flag). Every reachable 116 trainer battles fine.
    gated_trainer_ids = {273, 605}
    trainer_ids = sorted({o["trainer_id"] for k in scope if k in maps and k not in gated_trainer_maps
                          for o in maps[k]["objects"]
                          if o.get("trainer_id") and o["trainer_id"] not in gated_trainer_ids})
    party = {t["id"]: [tuple(p) for p in t["party"]] for t in M["trainers"] if t["party"]}
    sl_frames = Counter()
    for f in sorted((root / "processed/conditions").glob("*.npz")):
        z = np.load(f)
        bv, bs, bl = z["bat_valid"] > 0, z["bat_species"], z["bat_level"]
        m1 = bv[:, 1]
        for (s_, l_), c_ in zip(*[list(x) for x in np.unique(
                np.stack([bs[:, 1][m1], bl[:, 1][m1]], 1), axis=0, return_counts=True)] if m1.any() else ([], [])):
            sl_frames[(int(s_), int(l_))] += int(c_)
    # support = the best-covered PARTY MEMBER (species, level): lead-only matching marked
    # recorded battles red whenever the lead fell fast and most frames showed the second mon
    rows["trainers"] = [
        {"key": f"tr{t}",
         "support": (sup := max((sl_frames.get(p, 0) for p in party.get(t, [])), default=0)),
         "ok": sup >= 500} for t in trainer_ids]

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
