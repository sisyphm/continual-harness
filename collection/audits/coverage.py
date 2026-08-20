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
         "species_enemy": 3000, "species_player": 3000, "outcome_events": 10, "glyph": 2000,
         # W33 item 4 direction balance: a PAIR is green only when BOTH directions
         # clear the (already directional) warp floor — min(a->b, b->a) >= 5
         "connection_pair": 5}
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


# ======================================================================================
# W33 corpus-v2 item 4: audit extension over RECORDED v2 RUN DIRECTORIES
# (phases.jsonl + block summaries + per-tick ledger chunks) — the axes §5.1 adds on
# top of the conditions-based v1 axes above. Extend, don't break: everything below is
# additive; the conditions-based main() path is untouched unless --v2_runs is passed.
# ======================================================================================

# floors for the new axes (fleet-scale defaults; tests pass reduced floors)
V2_FLOORS = {
    "interaction_verb": 25,        # per verb, fleet-wide
    "mart_event": 5, "pc_event": 5, "item_use_event": 5,
    "mart_town": 1,                # each town's mart entered with >= 1 verified buy
    "sweep_cell_tiles": 20,        # tiles visited inside a (map, stage-window) cell
    "sweep_windows_per_map": 2,    # >= 2 anchor stages per map (capped by windows)
    "connection_pair": FLOOR["connection_pair"],
    "whiteout": FLOOR["outcome_events"],
    "evolutions": 3,               # all three starter evolution cutscenes (§1)
    "level_frames": 500,           # frames per observed lead level bucket
    "level_spread": 6,             # distinct lead levels across the fleet
    "low_hp_frames": 300,          # in-battle red-bar (<= 20%) frames
    "battle_level_up": 3,          # in-battle level-up events
    # task #46: runs whose FINAL ledger frame carries the trainer's defeated-flag
    # (flag = 0x500 + script trainer id, empirically verified — see
    # blocks/trainer_engagement.py); >= 2 runs per tracked trainer
    "trainer_runs": 2,
}

# DENSITY floors: events per 1,000 block frames per phase (§4/§5.2 — "a block that
# regresses into padding is a red row"). Provenance: MEASURED on the 2026-08-21
# item-4 acceptance recording (BACK_TO_ROUTE101_FROM_OLDALE state; legs + Route-101
# sweep + menus + encounter_farm + idle, real emulator — tests/test_item4.py
# re-records the same shape) plus the item-3 test pins; floors sit at ~30-50% of the
# measured density so a healthy block clears with margin while a padding regression
# (2x frames, same events) goes red:
#   bfs_sweep    MEASURED 16.4 tiles/1k (67 tiles / 4.1k block frames) -> floor 5
#                (the item-4 spec's suggested "sweep >= 5 tiles/1k")
#   encounter    MEASURED 0.87 battles/1k (7 / 8.0k); consistent with the v2 pin
#                "wild battle every ~0.5-1.5k paced frames"            -> 0.4
#   menus        MEASURED 2.33 views/1k (3 / 1.3k)                     -> 1.0
#   idle         MEASURED 823 dwell-frames/1k (1710 / 2.1k)            -> 500
#   interaction  v2b pin: ~14 verbs, approach/talk ~1.2k f/verb => ~0.8/1k -> 0.4
#   grind        battle-win cycle ~2x the flee cycle (v2b pin)         -> 0.2
#   mart_pc_item v2c pins: a verified transaction lands in <= ~10k frames
#                of block work (travel included)                       -> 0.1
DENSITY_FLOORS = {"bfs_sweep": 5.0, "interaction": 0.4, "encounter": 0.4,
                  "grind": 0.2, "mart_pc_item": 0.1, "menus": 1.0, "idle": 500.0}

# density events per block: block name -> (phase, events-from-summary)
_DENSITY_EVENTS = {
    "bfs_sweep": ("bfs_sweep", lambda s: sum(s.get("tiles_visited_per_map", {}).values())),
    "interaction": ("interaction", lambda s: sum(s.get("verbs", {}).values())),
    "encounter_farm": ("encounter", lambda s: s.get("battles", 0)),
    "grind_evolve": ("grind", lambda s: s.get("battles", 0)),
    "menus": ("menus", lambda s: s.get("party_menu_views", 0) + s.get("summary_views", 0)
              + s.get("bag_views", 0)),
    "idle": ("idle", lambda s: s.get("dwell_frames", 0)),
    "mart_buy": ("mart_pc_item", lambda s: sum(1 for p in s.get("purchases", []) if p.get("verified"))
                 + sum(1 for p in s.get("sells", []) if p.get("verified"))),
    "item_use": ("mart_pc_item", lambda s: s.get("overworld_uses", 0) + s.get("battle_uses", 0)),
    "pc_access": ("mart_pc_item", lambda s: sum(1 for p in s.get("deposits", []) if p.get("verified"))
                  + sum(1 for p in s.get("withdrawals", []) if p.get("verified"))),
}

INTERACTION_VERBS = ("npc_talk", "npc_retalk", "sign_read", "dialog_cancel",
                     "object_interact", "ledge_hop", "door_bounce", "save_dialog")


def _v2_stage_of(run_dir: Path) -> str | None:
    """The story stage a standalone block run started from — parsed from the manifest
    metadata's load_state path (…/storyline_wm/<STAGE>/attempt_*/final.state)."""
    mp = run_dir / "manifest.json"
    if not mp.exists():
        return None
    meta = (json.loads(mp.read_text()).get("metadata") or {})
    parts = Path(str(meta.get("load_state", ""))).parts
    if "storyline_wm" in parts:
        return parts[parts.index("storyline_wm") + 1]
    return None


def v2_block_summaries(run_dir: str | Path) -> list[tuple[dict, str | None]]:
    """(summary, stage) for every block outcome a run dir carries — single-block jobs
    (block_summary.json), multi-block jobs (block_summaries.json) and full director
    runs (playthrough_summary.json blocks[], whose entries carry 'after')."""
    run = Path(run_dir)
    out: list[tuple[dict, str | None]] = []
    stage = _v2_stage_of(run)
    p = run / "block_summary.json"
    if p.exists():
        out.append((json.loads(p.read_text()), stage))
    p = run / "block_summaries.json"
    if p.exists():
        out += [(s, stage) for s in json.loads(p.read_text())]
    p = run / "playthrough_summary.json"
    if p.exists():
        out += [(s, s.get("after")) for s in json.loads(p.read_text()).get("blocks", [])]
    return out


def v2_run_ledger(run_dir: str | Path):
    from collection.derive import load_run_ledger
    return load_run_ledger(run_dir)


def v2_phase_frames(run_dir: str | Path) -> dict[str, int]:
    """Recorded frames per phase from phases.jsonl spans (span end = next transition
    or the action-row count). Informational alongside the block-summary frames."""
    run = Path(run_dir)
    pp = run / "phases.jsonl"
    if not pp.exists():
        return {}
    trans = [json.loads(l) for l in pp.open()]
    total = sum(1 for _ in (run / "actions.jsonl").open()) if (run / "actions.jsonl").exists() else 0
    out: dict[str, int] = {}
    for i, t in enumerate(trans):
        end = trans[i + 1]["frame_idx"] if i + 1 < len(trans) else total
        if t["phase"] is not None:
            out[t["phase"]] = out.get(t["phase"], 0) + max(0, end - t["frame_idx"])
    return out


def _party_down_events(led) -> int:
    """Whiteout proxy from the per-tick ledger: the whole party's hp hits 0."""
    valid = (led["valid"] > 0) & (led["party_count"] > 0)
    have = led["species"] > 0
    hp_total = (led["hp"] * have).sum(axis=1)
    any_mon = have.any(axis=1)
    down = valid & any_mon & (hp_total == 0)
    up = valid & any_mon & (hp_total > 0)
    events = 0
    state_up = False
    for i in range(len(down)):
        if up[i]:
            state_up = True
        elif down[i] and state_up:
            events += 1
            state_up = False
    return events


def _evolution_events(led) -> Counter:
    """(from_species -> to_species) events: a party slot's species changes while its
    personality stays — the ledger-native evolution signal."""
    ev = Counter()
    sp, pid, valid = led["species"], led["personality"], led["valid"] > 0
    for slot in range(sp.shape[1]):
        s, p = sp[:, slot].astype(np.int64), pid[:, slot]
        m = valid[1:] & valid[:-1] & (s[1:] > 0) & (s[:-1] > 0) \
            & (s[1:] != s[:-1]) & (p[1:] == p[:-1]) & (p[1:] != 0)
        for i in np.flatnonzero(m):
            ev[(int(s[i]), int(s[i + 1]))] += 1
    return ev


def v2_axes(run_dirs, *, change_matrix: dict | None = None,
            floors: dict | None = None) -> dict[str, list]:
    """The new §5.1 axes over v2 run dirs. Every axis emits rows even at zero support
    (absence must be visible); 'deferred' rows document §11-derivable-later fields."""
    fl = dict(V2_FLOORS)
    fl.update(floors or {})
    runs = [Path(r) for r in run_dirs]
    rows: dict[str, list] = {}

    verbs, mart, pair_dir = Counter(), Counter(), Counter()
    towns_bought = Counter()
    sweep_cells = Counter()                       # (map, window) -> tiles
    sweep_windows: dict[str, set] = {}
    for run in runs:
        for s, stage in v2_block_summaries(run):
            for v, n in s.get("verbs", {}).items():
                verbs[v] += n
            for pu in s.get("purchases", []):
                if pu.get("verified"):
                    mart["mart_purchase"] += 1
            for se in s.get("sells", []):
                if se.get("verified"):
                    mart["mart_sell"] += 1
            for town, stock in s.get("stock", {}).items():
                if stock and any(p.get("verified") for p in s.get("purchases", [])):
                    towns_bought[town] += 1
            for d in s.get("deposits", []):
                if d.get("verified"):
                    mart["pc_deposit"] += 1
            for w in s.get("withdrawals", []):
                if w.get("verified"):
                    mart["pc_withdraw"] += 1
            mart["item_use_overworld"] += s.get("overworld_uses", 0)
            mart["item_use_battle"] += s.get("battle_uses", 0)
            for a, b in s.get("connections_crossed", []):
                pair_dir[(a, b)] += 1
            for pr in s.get("door_pairs", []):
                if isinstance(pr, (list, tuple)) and len(pr) == 2:
                    pair_dir[(pr[0], pr[1])] += 1
                    pair_dir[(pr[1], pr[0])] += 1
            if stage is not None and s.get("tiles_visited_per_map"):
                from collection.plan_change_matrix import window_of
                for k, tiles in s["tiles_visited_per_map"].items():
                    w = window_of(change_matrix, k, stage) if change_matrix else "*"
                    sweep_cells[(k, w)] += tiles
                    sweep_windows.setdefault(k, set()).add(w)

    rows["interaction_verbs"] = [
        {"key": v, "support": verbs.get(v, 0), "ok": verbs.get(v, 0) >= fl["interaction_verb"]}
        for v in INTERACTION_VERBS]
    rows["mart_pc_item"] = (
        [{"key": k, "support": mart.get(k, 0),
          "ok": mart.get(k, 0) >= fl["mart_event" if k.startswith("mart") else
                                     "pc_event" if k.startswith("pc") else
                                     "item_use_event"]}
         for k in ("mart_purchase", "mart_sell", "pc_deposit", "pc_withdraw",
                   "item_use_overworld", "item_use_battle")]
        + [{"key": f"mart@{t}", "support": n, "ok": n >= fl["mart_town"]}
           for t, n in sorted(towns_bought.items())])

    # per-(map, stage-window) sweep coverage — rows enumerate the CHANGE MATRIX's
    # cells (required universe) plus anything actually swept; per-map window counts
    # audit the ">= 2 anchor stages" rule
    cell_rows, win_rows = [], []
    if change_matrix is not None:
        from collection.plan_change_matrix import map_windows
        for k in sorted(change_matrix["matrix"]):
            wins = map_windows(change_matrix, k)
            for w in wins:
                sup = sweep_cells.get((k, w), 0)
                cell_rows.append({"key": f"{k}@{w}", "support": sup,
                                  "ok": sup >= fl["sweep_cell_tiles"]})
            n = len(sweep_windows.get(k, set()))
            want = min(fl["sweep_windows_per_map"], len(wins))
            win_rows.append({"key": k, "support": n, "ok": n >= want})
    else:
        for (k, w), sup in sorted(sweep_cells.items()):
            cell_rows.append({"key": f"{k}@{w}", "support": sup,
                              "ok": sup >= fl["sweep_cell_tiles"]})
    rows["sweep_cells"] = cell_rows
    rows["sweep_windows_per_map"] = win_rows

    pair_min: dict[tuple[str, str], int] = {}
    for (a, b), n in pair_dir.items():
        key = tuple(sorted((a, b)))
        rev = pair_dir.get((b, a), 0)
        pair_min[key] = min(n, rev) if key not in pair_min else pair_min[key]
    rows["connection_pairs"] = [
        {"key": f"{a}<->{b}", "support": n, "ok": n >= fl["connection_pair"]}
        for (a, b), n in sorted(pair_min.items())]

    # ---- ledger-derived axes (whiteouts / evolutions / levels / battle situations
    # / task #46 per-trainer engagement flags)
    from collection.playthrough.blocks.trainer_engagement import (
        TRACKED_TRAINER_FLAGS)
    whiteouts = 0
    evo = Counter()
    level_frames = Counter()
    low_hp = 0
    lvl_ups = 0
    trainer_runs = Counter()
    for run in runs:
        led = v2_run_ledger(run)
        if led is None:
            continue
        # task #46: a run supports trainer N iff its FINAL valid ledger frame has
        # flag 0x500+N set (byte N//8, bit N%8 of the recorded flags array) — the
        # end-of-run defeat record, exactly what the scheduler must guarantee twice
        vi = np.flatnonzero(led["valid"] > 0)
        if vi.size:
            frow = led["flags"][vi[-1]]
            for f in TRACKED_TRAINER_FLAGS:
                if (int(frow[f // 8]) >> (f % 8)) & 1:
                    trainer_runs[f] += 1
        whiteouts += _party_down_events(led)
        evo += _evolution_events(led)
        lead_ok = (led["valid"] > 0) & (led["species"][:, 0] > 0)
        for lv, n in zip(*np.unique(led["level"][:, 0][lead_ok], return_counts=True)):
            level_frames[int(lv)] += int(n)
        ib = (led["in_battle"] > 0) & (led["valid"] > 0)
        low_hp += int((ib & (led["active_max_hp"] > 0) & (led["active_hp"] > 0)
                       & (led["active_hp"] * 5 <= led["active_max_hp"])).sum())
        al, ap_ = led["active_level"].astype(np.int64), led["active_personality"]
        m = ib[1:] & ib[:-1] & (ap_[1:] == ap_[:-1]) & (ap_[1:] != 0) \
            & (al[1:] == al[:-1] + 1)
        lvl_ups += int(m.sum())

    rows["outcomes_v2"] = (
        [{"key": "whiteout", "support": whiteouts, "ok": whiteouts >= fl["whiteout"]},
         {"key": "evolution", "support": int(sum(evo.values())),
          "ok": sum(evo.values()) >= fl["evolutions"]}]
        + [{"key": f"evolution:{a}->{b}", "support": n, "ok": n >= 1}
           for (a, b), n in sorted(evo.items())])
    rows["level_hist"] = (
        [{"key": f"lv{lv}", "support": n, "ok": n >= fl["level_frames"]}
         for lv, n in sorted(level_frames.items())]
        + [{"key": "level_spread", "support": len(level_frames),
            "ok": len(level_frames) >= fl["level_spread"]}])
    # task #46: 18 per-trainer rows — support = runs whose final ledger frame has
    # the flag set, floor fl["trainer_runs"]; rows exist even at zero support
    rows["trainer_flags"] = [
        {"key": f"tr_flag_{f:#06x}", "support": trainer_runs.get(f, 0),
         "ok": trainer_runs.get(f, 0) >= fl["trainer_runs"]}
        for f in TRACKED_TRAINER_FLAGS]
    rows["battle_situation"] = [
        {"key": "low_hp_bar", "support": low_hp, "ok": low_hp >= fl["low_hp_frames"]},
        {"key": "battle_level_up", "support": lvl_ups,
         "ok": lvl_ups >= fl["battle_level_up"]},
        # no status/crit field in the v2 ledger schema (ledger_panel.FIELDS) —
        # documented DERIVABLE-LATER via §11: one derive.py reader over
        # gBattleMons.status1 (+0x4C) / battle result flags, scheduled pilot backfill
        *[{"key": k, "support": 0, "ok": False,
           "deferred": "derivable via §11 derive.py reader (gBattleMons.status1 "
                       "@ +0x4C / battle outcome flags) — pilot backfill"}
          for k in ("status_psn", "status_slp", "status_par", "crit")],
    ]
    return rows


def v2_density(run_dirs, *, floors: dict | None = None) -> list[dict]:
    """§5.2 density audit: events per 1k BLOCK frames per phase, from the block
    summaries (events and frames from the same source); phases.jsonl frame spans are
    reported alongside as `recorded_frames` (includes settle/return overhead)."""
    fl = dict(DENSITY_FLOORS)
    fl.update(floors or {})
    ev, fr = Counter(), Counter()
    rec = Counter()
    for run in run_dirs:
        for ph, n in v2_phase_frames(run).items():
            rec[ph] += n
        for s, _ in v2_block_summaries(Path(run)):
            name = s.get("block")
            if name not in _DENSITY_EVENTS or not s.get("ran"):
                continue
            phase, fn = _DENSITY_EVENTS[name]
            ev[phase] += fn(s)
            fr[phase] += s.get("frames", 0)
    rows = []
    for phase in sorted(set(ev) | set(fr)):
        frames = fr[phase]
        per_1k = round(ev[phase] * 1000.0 / frames, 2) if frames else 0.0
        rows.append({"phase": phase, "events": int(ev[phase]), "block_frames": int(frames),
                     "recorded_frames": int(rec.get(phase, 0)), "per_1k": per_1k,
                     "floor": fl.get(phase), "ok": (fl.get(phase) is None
                                                   or per_1k >= fl[phase])})
    return rows


def audit_v2(run_dirs, *, change_matrix: dict | None = None,
             floors: dict | None = None, density_floors: dict | None = None) -> dict:
    axes = v2_axes(run_dirs, change_matrix=change_matrix, floors=floors)
    density = v2_density(run_dirs, floors=density_floors)
    return {
        "runs": [str(r) for r in run_dirs],
        "summary": {ax: {"rows": len(rs), "red": sum(not r["ok"] for r in rs)}
                    for ax, rs in axes.items()},
        "density_red": sum(not r["ok"] for r in density),
        "rows": axes,
        "density": density,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--v2_runs", nargs="*", default=None,
                    help="v2 run dirs (or globs) — adds the item-4 axes/density audit")
    ap.add_argument("--v2_only", action="store_true",
                    help="skip the conditions pass; audit only the v2 run dirs")
    ap.add_argument("--change_matrix", default=None,
                    help="default: <data_root>/processed/w33_change_matrix.json")
    args = ap.parse_args()

    v2_report = None
    if args.v2_runs is not None:
        import glob as _glob
        dirs = sorted(d for pat in args.v2_runs for d in _glob.glob(pat))
        cm_path = Path(args.change_matrix) if args.change_matrix else \
            Path(args.data_root) / "processed/w33_change_matrix.json"
        cm = json.loads(cm_path.read_text()) if cm_path.exists() else None
        v2_report = audit_v2(dirs, change_matrix=cm)
        print("=== V2 RUN AUDIT (item-4 axes; red = below floor) ===")
        for ax, s in v2_report["summary"].items():
            print(f"  {ax:22s}: {s['rows']:>4} rows, {s['red']:>4} RED")
        print(f"  density: {len(v2_report['density'])} phases, "
              f"{v2_report['density_red']} RED")
        if args.v2_only:
            out = Path(args.out) if args.out else \
                Path(args.data_root) / "processed/audit/v2_coverage_report.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(v2_report, indent=1))
            print(f"wrote {out}")
            return
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
    # warps: manifest pairs in scope, per direction. VERIFIED directional (W33 item 4
    # review): warp_n keys on ordered (prev_map, cur_map) transitions, and the row set
    # enumerates a->b AND b->a separately (manifest warps/connections are symmetric in
    # scope — probe 2026-08-20: all 45 unordered pairs have both directed rows).
    wrows = []
    for a in scope:
        if a not in maps:
            continue
        dsts = {w["dst_map"] for w in maps[a]["warps"]} | {c["dst_map"] for c in maps[a]["connections"]}
        for b in sorted(dsts & scope):
            c = warp_n.get((a, b), 0)
            wrows.append({"key": f"{a}->{b}", "support": c, "ok": c >= FLOOR["warp_traversals"]})
    rows["warps"] = wrows
    # direction BALANCE (W33 item 4): pair rows — support = the weaker direction, so
    # a corpus that only ever crosses A->B stays red until the reverse leg exists too
    pair_dirs: dict[tuple[str, str], list[int]] = {}
    for r_ in wrows:
        a, b = r_["key"].split("->")
        pair_dirs.setdefault(tuple(sorted((a, b))), []).append(r_["support"])
    rows["connection_pairs"] = [
        {"key": f"{a}<->{b}", "support": min(v), "ok": min(v) >= FLOOR["connection_pair"]}
        for (a, b), v in sorted(pair_dirs.items())]
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
    if v2_report is not None:
        report["v2"] = v2_report
    out = Path(args.out) if args.out else root / "processed/audit/coverage_report.json"
    out.write_text(json.dumps(report, indent=1))
    print("=== COVERAGE vs MANIFEST (red = below floor) ===")
    for ax, s in report["summary"].items():
        print(f"  {ax:15s}: {s['rows']:>5} rows, {s['red']:>5} RED")
    print(f"  menus (proxy) : {len(win_patterns)} distinct window geometries")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
