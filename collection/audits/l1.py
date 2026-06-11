"""Phase-0 audit, Layer 1 — dataset coverage/sufficiency stats from the cheap streams.

Reads every run's `semantic.jsonl` + `actions.jsonl` (+ per-frame brightness, computed from the RGB
chunks once and cached as `brightness.npy`), and measures what the Rung-A′ plan needs to know BEFORE
training (`WORLD_MODEL_PLAN_RUNG_A_PRIME.md` §2):

  • mode mass (L1 proxy): battle / dark(warp/fade) / overworld — exact taxonomy is Layer 2 (RAM)
  • per-map frame mass
  • stationary-player run-length histogram  ← the idle gap, measured
  • input patterns: moving vs turning vs bump-ish, run(B)-while-moving, A/START presses, no-input
  • NPC graphics_id support (frames on screen)

Usage:
  .venv/bin/python -m collection.audits.l1 --data_root ../pokemon-worldmodel/data \
      --out ../pokemon-worldmodel/data/processed/audit/audit_l1.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from collection.corpus import clip_starts, discover_runs

DARK_THRESH = 30.0          # brightness below this = warp/fade black (validated in the v1 indexer)
BUMP_STATIC_FRAMES = 24     # direction held + facing aligned + coords static this long = bump-ish
DIRS = ("UP", "DOWN", "LEFT", "RIGHT")

# stationary run-length histogram buckets (frames @60fps); ≥64 ≈ ≥1s of standing still
IDLE_BUCKETS = ((1, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255), (256, 1 << 30))


def brightness_of(run_dir: Path, n_frames: int) -> np.ndarray:
    """Per-frame mean brightness; computed from the RGB chunks once and cached beside them."""
    cache = run_dir / "brightness.npy"
    if cache.exists():
        b = np.load(cache)
        if len(b) >= n_frames:
            return b[:n_frames]
    metas = [json.loads(l) for l in (run_dir / "frames.jsonl").open()]
    out = np.zeros(len(metas), np.float32)
    by_chunk: dict[str, list[tuple[int, int]]] = {}
    for i, m in enumerate(metas):
        by_chunk.setdefault(m["chunk"], []).append((i, m["chunk_frame_idx"]))
    for chunk, pairs in by_chunk.items():
        frames = np.load(run_dir / chunk)["frames"]                  # (N,160,240,3) uint8
        idx = np.array([p[1] for p in pairs])
        out[[p[0] for p in pairs]] = frames[idx].mean(axis=(1, 2, 3))
    np.save(cache, out)
    return out[:n_frames]


def _bucket(n: int) -> str:
    for lo, hi in IDLE_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi if hi < (1 << 29) else '∞'}"
    return "0"


def audit_run(run_dir: Path) -> dict:
    sem = [json.loads(l) for l in (run_dir / "semantic.jsonl").open()]
    held: dict[int, list[str]] = {r["frame_idx"]: r["buttons_held"]
                                  for r in (json.loads(l) for l in (run_dir / "actions.jsonl").open())}
    bright = brightness_of(run_dir, len(sem))
    resets = clip_starts(run_dir)

    maps, gids = Counter(), Counter()
    n_battle = n_dark = n_noinput = n_move = n_turn = 0
    n_moving_b = 0                                            # moving frames with B held (running)
    a_press = start_press = bumps = 0
    idle_hist: Counter = Counter()                            # stationary-run length buckets
    idle_total = 0                                            # frames inside ≥64-frame stationary runs

    stat_len = 0                                              # current stationary run (overworld only)
    held_dir_static = 0                                       # frames direction held while static+aligned
    prev = None
    prev_buttons: list[str] = []
    for i, s in enumerate(sem):
        b = held.get(s["frame"], [])
        battle, dark = s["in_battle"], bright[i] < DARK_THRESH
        n_battle += battle
        n_dark += (not battle) and dark
        maps[s["map"]] += 1
        for o in s["objects"]:
            if 0 <= o["x"] < 64 and 0 <= o["y"] < 64:         # sane-coord filter (raw gObjectEvents)
                gids[o["graphics_id"]] += 1
        if not b:
            n_noinput += 1
        if "A" in b and "A" not in prev_buttons:
            a_press += 1
        if "START" in b and "START" not in prev_buttons:
            start_press += 1

        overworld = not battle and not dark
        if prev is not None and overworld:
            moved = (s["x"], s["y"]) != (prev["x"], prev["y"]) or s["map"] != prev["map"]
            turned = (not moved) and s["facing"] != prev["facing"]
            n_move += moved
            n_turn += turned
            if moved and "B" in b:
                n_moving_b += 1
            # stationary runs (the idle measurement) — broken by movement, resets, mode changes
            if moved or s["frame"] in resets:
                if stat_len:
                    idle_hist[_bucket(stat_len)] += 1
                    if stat_len >= 64:
                        idle_total += stat_len
                stat_len = 0
            else:
                stat_len += 1
            # bump-ish: a direction held, facing already aligned, coords static long enough
            d = next((x for x in b if x in DIRS), None)
            if d and not moved and s["facing"] == d:
                held_dir_static += 1
                if held_dir_static == BUMP_STATIC_FRAMES:
                    bumps += 1
            else:
                held_dir_static = 0
        else:
            stat_len = 0
            held_dir_static = 0
        prev, prev_buttons = s, b

    if stat_len:
        idle_hist[_bucket(stat_len)] += 1
        if stat_len >= 64:
            idle_total += stat_len

    n = len(sem)
    return {
        "frames": n,
        "battle_frames": n_battle,
        "dark_frames": n_dark,
        "overworld_frames": n - n_battle - n_dark,
        "maps": dict(maps),
        "npc_gid_frames": {str(k): v for k, v in gids.items()},
        "move_frames": n_move,
        "turn_events": n_turn,
        "bump_events": bumps,
        "moving_with_B": n_moving_b,
        "no_input_frames": n_noinput,
        "a_presses": a_press,
        "start_presses": start_press,
        "stationary_run_hist": dict(idle_hist),
        "idle64_frames": idle_total,                          # frames in ≥64f (~1s+) stationary runs
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out", default="../pokemon-worldmodel/data/processed/audit/audit_l1.json")
    args = ap.parse_args()
    root = Path(args.data_root)

    runs = discover_runs(root)
    print(f"auditing {len(runs)} runs…")
    from collections import defaultdict
    per_run, agg = {}, defaultdict(Counter)
    for name, d, kind in runs:
        r = audit_run(d)
        per_run[f"{kind}/{name}"] = r
        for k in ("frames", "battle_frames", "dark_frames", "overworld_frames", "move_frames",
                  "turn_events", "bump_events", "moving_with_B", "no_input_frames",
                  "a_presses", "start_presses", "idle64_frames"):
            agg[kind][k] += r[k]
        print(f"  {kind:9s} {name:34s} {r['frames']:7d}f  battle {r['battle_frames']:6d}  "
              f"dark {r['dark_frames']:6d}  idle64 {r['idle64_frames']:5d}")

    # ---- merged views the report needs ----
    idle_hist, maps, gids = Counter(), Counter(), Counter()
    for r in per_run.values():
        idle_hist.update(r["stationary_run_hist"])
        maps.update(r["maps"])
        gids.update({int(k): v for k, v in r["npc_gid_frames"].items()})

    total = sum(a["frames"] for a in agg.values())
    summary = {
        "total_frames": total,
        "by_kind": {k: dict(v) for k, v in agg.items()},
        "stationary_run_hist": dict(idle_hist),
        "idle64_fraction": sum(a["idle64_frames"] for a in agg.values()) / max(total, 1),
        "distinct_maps": len(maps),
        "map_frames": dict(maps.most_common()),
        "distinct_npc_gids": len(gids),
        "npc_gid_frames": {str(k): v for k, v in gids.most_common()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_run": per_run}, indent=1, default=int))

    s = summary
    print(f"\n=== L1 SUMMARY ({total:,} frames, {len(per_run)} runs) ===")
    for kind, a in s["by_kind"].items():
        f = a["frames"]
        print(f"{kind:9s}: {f:7,d}f  battle {a['battle_frames']/f:6.1%}  dark {a['dark_frames']/f:6.1%}  "
              f"move {a['move_frames']/f:6.1%}  no-input {a['no_input_frames']/f:6.1%}")
        print(f"{'':9s}  turns {a['turn_events']:,}  bumps {a['bump_events']:,}  "
              f"run(B)-moving {a['moving_with_B']:,}  A-presses {a['a_presses']:,}  idle64f {a['idle64_frames']:,}")
    print(f"idle(≥64f stationary) fraction of corpus: {s['idle64_fraction']:.3%}")
    print(f"stationary-run histogram (frames): " +
          "  ".join(f"{k}:{v}" for k, v in sorted(idle_hist.items(), key=lambda kv: int(kv[0].split('-')[0]))))
    print(f"maps: {s['distinct_maps']}  npc gids: {s['distinct_npc_gids']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
