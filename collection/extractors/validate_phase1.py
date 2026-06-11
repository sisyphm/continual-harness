"""Phase-1 validation suite — entities + terrain extractors against INDEPENDENT sources, over
sampled frames of every recorded run. Nothing trains on these extractors until this passes.

  E1a blob-seam GATE: player x/y read offline must equal semantic.jsonl bit-for-bit (validates the
      recorded-blob backend against the live reads that produced semantic at collection time).
  E1b facing agreement (INFORMATIONAL): extractor facing (player object record) vs semantic.jsonl
      facing (the collector's input-tracker). Disagreements were pixel-adjudicated in favor of the
      RECORD (semantic goes stale after warps/forced turns), so this is reported, not gated.
  E2  OAM cross-check, CAMERA-FREE: anchored on the PLAYER's own OAM sprite (the 16x32 entry near
      screen center on unclamped frames), an extracted static NPC must have an OAM sprite at
      player_oam + 16*(tile delta). Restricted to interior frames (player ≥8 tiles from every map
      edge, both entities static) — near map connections the engine rebases coords/layout and any
      camera assumption breaks (measured: a Petalburg sample at x=width was a CONNECTION strip,
      not a bug). No camera model, no clamp logic, no dims dependence.
  E3  identity sanity: distinct NPC graphics ids across samples vs the Phase-0 audit set (67).
  T1  pinned-layout sanity: terrain() parses at 0x03005DC0 on every cb2-overworld sampled frame
      (cb2 gate via the Phase-0 audit fields: intro/naming/title frames are NOT overworld even
      though in_battle is false).
  T2  dims stability per (map_group,map_num) across visits; containment within the bordered
      buffer (map ±7 — connection strips are legitimate player positions).
  T3  grid stability: same map sampled at different times -> identical inner metatile grid.
  T4  walkability: the metatile word under the player has collision==0 on (nearly) all frames.

Usage:
  .venv/bin/python -m collection.extractors.validate_phase1 --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from collection.corpus import discover_runs
from collection.extractors.entities import entities as all_entities
from collection.extractors.entities import npcs, player_state
from collection.extractors.ram import GBAState, iter_states
from collection.extractors.terrain import terrain

OAM_BASE, OAM_N = 0x07000000, 128
SAMPLES_PER_RUN = 5


def _oam_positions(st: GBAState) -> list[tuple[int, int, int, int]]:
    """On-screen OBJ entries: (x, y, shape, size) of enabled OAM slots."""
    out = []
    for i in range(OAM_N):
        a0 = st.u16(OAM_BASE + i * 8)
        a1 = st.u16(OAM_BASE + i * 8 + 2)
        if (a0 >> 8) & 0x3 == 0b10:                       # rendering disabled (affine off + disable bit)
            continue
        x, y = a1 & 0x1FF, a0 & 0xFF
        if x < 240 and y < 160:
            out.append((x, y, a0 >> 14, a1 >> 14))
    return out


def _player_oam(oam: list[tuple[int, int, int, int]]) -> tuple[int, int] | None:
    """The player's sprite: the 16x32 entry (shape=2 vertical, size=2) nearest screen center.
    Only trusted within 24px of the canonical center (112, 56) — i.e. unclamped-camera frames."""
    best = None
    for x, y, shape, size in oam:
        if (shape, size) != (2, 2):
            continue
        d = abs(x - 112) + abs(y - 56)
        if d <= 24 and (best is None or d < best[0]):
            best = (d, x, y)
    return (best[1], best[2]) if best else None


def _sample_frames(run_dir: Path, n: int) -> list[int]:
    metas = sum(1 for _ in (run_dir / "frames.jsonl").open())
    return sorted({int(i) for i in np.linspace(metas * 0.1, metas - 2, n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    args = ap.parse_args()
    runs = discover_runs(Path(args.data_root))

    e1_xy_ok = e1_face_ok = e1_tot = 0
    e2_pairs: list[tuple[int, int, int, int, int]] = []    # (obs_id, sx, sy, oam_x, oam_y)
    e3_gfx = Counter()
    t1_bad = 0
    t2_dims: dict[tuple[int, int], tuple[str, tuple[int, int]]] = {}
    t2_stable = t2_unstable = t2_contained = t2_escaped = 0
    t3_grids: dict[tuple[int, int], tuple[str, np.ndarray]] = {}
    t3_ok = t3_bad = 0
    t4_ok = t4_tot = 0
    obs_id = 0

    audit_ram = Path(args.data_root) / "processed/audit/ram"
    CB2_OVERWORLD = {0x8085E5D, 0x8085E51}                  # Phase-0 cb2 taxonomy

    for name, d, kind in runs:
        sem = [json.loads(l) for l in (d / "semantic.jsonl").open()]
        bright = np.load(d / "brightness.npy") if (d / "brightness.npy").exists() else None
        npz = audit_ram / f"{kind}__{name}.v2.npz"
        cb2 = np.load(npz)["cb2"] if npz.exists() else None
        want = set(_sample_frames(d, SAMPLES_PER_RUN))
        for f, st in iter_states(d):
            if f not in want:
                continue
            want.discard(f)
            s = sem[f]
            # dark warp/load frames carry STALE coords + mid-load layout (the known v1 gotcha)
            if bright is not None and bright[f] < 30:
                if not want:
                    break
                continue
            # overworld gate: NOT in battle AND the game's own screen callback is overworld
            # (intro/naming/title frames are battle-false but have no valid overworld state)
            in_overworld = (not s["in_battle"]) and (cb2 is None or int(cb2[f]) in CB2_OVERWORLD)

            # E1: blob seam (x/y gate) + facing agreement (informational)
            p = player_state(st)
            e1_tot += 1
            e1_xy_ok += (p["x"], p["y"]) == (s["x"], s["y"])
            e1_face_ok += p["facing"] == s["facing"]

            t = terrain(st) if in_overworld else None
            if in_overworld:
                t1_bad += t is None

            if t is not None:
                player_rec = next((e for e in all_entities(st) if e.is_player), None)
                player_static = player_rec is None or not player_rec.mid_step
                interior = (8 <= p["x"] < t.map_width - 8 and 8 <= p["y"] < t.map_height - 8)
                oam = _oam_positions(st)
                anchor = _player_oam(oam) if (interior and player_static) else None

                # E2/E3: NPCs — camera-free, anchored on the player's own OAM sprite
                for e in npcs(st):
                    e3_gfx[e.graphics_id] += 1
                    if anchor is None or e.mid_step:
                        continue
                    dx, dy = e.x - p["x"], e.y - p["y"]
                    if abs(dx) <= 5 and abs(dy) <= 3 and (dx, dy) != (0, 0):
                        ex, ey = anchor[0] + 16 * dx, anchor[1] + 16 * dy
                        obs_id += 1
                        for ox, oy, _, _ in oam:
                            e2_pairs.append((obs_id, ex, ey, ox, oy))

                # T2: dims stability + containment-within-border per (group, num)
                key = (t.map_group, t.map_num)
                dims = (t.map_width, t.map_height)
                if key in t2_dims:
                    same = t2_dims[key][1] == dims
                    t2_stable += same
                    t2_unstable += not same
                else:
                    t2_dims[key] = (s["map"], dims)
                inside = -7 <= p["x"] < t.map_width + 7 and -7 <= p["y"] < t.map_height + 7
                t2_contained += inside
                t2_escaped += not inside

                # T3: grid stability per (group, num)
                inner = t.metatile_ids()[7:-8, 7:-8]
                if key in t3_grids:
                    prev_map, prev = t3_grids[key]
                    if prev.shape == inner.shape:
                        t3_ok += bool((prev == inner).all())
                        t3_bad += not (prev == inner).all()
                else:
                    t3_grids[key] = (s["map"], inner.copy())

                # T4: player tile passable
                if inside:
                    word = int(t.window(p["x"], p["y"], 1, 1)[0, 0])
                    t4_tot += 1
                    t4_ok += ((word >> 10) & 3) == 0
            if not want:
                break

    # E2: every OAM entry was paired with every static on-screen NPC, so most pairs are mismatches;
    # the TRUE constant screen->OAM offset is the residual MODE. Fit it, then count, per NPC
    # observation, whether at least one OAM entry lands within tolerance of the fitted offset.
    e2_fit = e2_obs_rate = None
    n_obs = len({p[0] for p in e2_pairs})
    if e2_pairs:
        arr = np.array(e2_pairs)                            # (N, 5): obs, sx, sy, ox, oy
        rx, ry = arr[:, 3] - arr[:, 1], arr[:, 4] - arr[:, 2]
        # the true offset is physically bounded (sprite anchor vs tile corner: 16x32 NPC sprite
        # ≈ (0,-16)); fit the residual mode within that window only
        phys = (np.abs(rx) <= 16) & (-32 <= ry) & (ry <= 16)
        cand = Counter(zip(rx[phys] // 4 * 4, ry[phys] // 4 * 4)).most_common(1)[0][0] if phys.any() else (0, -16)
        e2_fit = (int(cand[0]) + 2, int(cand[1]) + 2)
        good = (np.abs(rx - e2_fit[0]) <= 8) & (np.abs(ry - e2_fit[1]) <= 8)
        hit: dict[int, bool] = defaultdict(bool)
        for i in range(len(arr)):
            hit[int(arr[i, 0])] |= bool(good[i])
        e2_obs_rate = sum(hit.values()) / max(len(hit), 1)

    print("=== PHASE-1 EXTRACTOR VALIDATION ===")
    print(f"E1a GATE  blob-seam player x/y == semantic : {e1_xy_ok}/{e1_tot} ({e1_xy_ok/max(e1_tot,1):.1%})")
    print(f"E1b INFO  facing agreement vs semantic     : {e1_face_ok}/{e1_tot} ({e1_face_ok/max(e1_tot,1):.1%})"
          f"  (mismatches pixel-adjudicated in the extractor's favor)")
    if e2_pairs:
        print(f"E2 NPC->OAM: fitted offset {e2_fit}; observations matched: {e2_obs_rate:.1%} of {n_obs}")
    else:
        print("E2: no static on-screen NPC samples")
    top = dict(e3_gfx.most_common(8))
    print(f"E3 distinct NPC gfx ids in samples: {len(e3_gfx)} (audit full-corpus: 67); top {top}")
    print(f"T1 pinned layout unparsable on overworld frames: {t1_bad} (must be ~0)")
    print(f"T2 dims stable across visits: {t2_stable} ok / {t2_unstable} bad; "
          f"player inside dims: {t2_contained} / escaped {t2_escaped}")
    print(f"T3 grid stability across visits: ok={t3_ok} bad={t3_bad}")
    print(f"T4 player tile passable: {t4_ok}/{t4_tot} ({t4_ok/max(t4_tot,1):.1%})")


if __name__ == "__main__":
    main()
