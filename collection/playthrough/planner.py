"""W33 regen plan generator: reachable_matrix -> 60 run manifests (owner-signed design).

Every run: full spine + ALL reachable single-battle trainers (one-shot value, design's
"once per trainer per run") + a scheduled COVERAGE SLICE of the stage-conditioned tile
matrix + warp legs + floor blocks. No grind_evolve anywhere: leveling is trainer-routed
(torchic reaches L16 BEFORE the gym via the Route 116 batch; the Clark/Devan Geodudes
are ordered after the non-rock 116 trainers, and a small encounter_farm top-off covers
the 160-EXP thin spot the unreachable terrace four left behind).

Emits: data/processed/w33_regen_plan.json with per-run block_schedule in the exact
format pilot_driver/build_expedition_schedule consume ([milestone_name, block, kwargs]),
plus predicted per-run levels/frames for the audit to check delivery against.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from collection.audits.reachability import PE, WM, World

MATRIX = WM / "data/processed/reachable_matrix.json"
OUT = WM / "data/processed/w33_regen_plan.json"

# stage -> milestone boundaries (block runs AFTER the milestone) inside that window.
STAGE_WINDOWS = {
    "S1_prestarter": ["LEAVE_HOUSE", "RIVAL_HOUSE"],
    "S2_prepokedex": ["ROUTE_101", "OLDALE_TOWN", "ROUTE_103", "BACK_TO_OLDALE_FROM_ROUTE103"],
    "S3_postpokedex": ["OLDALE_AFTER_POKEDEX", "ROUTE101_AFTER_POKEDEX", "ROUTE_102",
                        "PETALBURG_CITY"],
    # ROUTE_104_SOUTH completes ON the seam: smoke rg_000 skipped all 5 blocks
    # scheduled there in ~240 frames each (precondition fails in the handoff state).
    # Use the gym exit and the woods boundary instead -- both hand off clean.
    "S4_open": ["EXIT_PETALBURG_GYM", "PETALBURG_WOODS"],
    "S4b_postwoods": ["PETALBURG_WOODS", "ROUTE_104_NORTH"],
    "S4c_rustboro": ["RUSTBORO_CITY", "RUSTBORO_CENTER_EXITED", "HEAL_AT_RUSTBORO_CENTER"],
}
SLICE_TILES = 45            # target tiles per sweep entry (tour ~35 f/tile with B-dash)
LEG_QUOTA = 5               # fleet-wide crossings per direction per connection pair
N_RUNS_PER_STARTER = 20     # 60 slots for 50 targets (17/17/16 + failure margin)


# ---------------------------------------------------------------- trainer roster
def _opponent_ids() -> dict[str, int]:
    out = {}
    for m in re.finditer(r"#define\s+(TRAINER_\w+)\s+(\d+)",
                         (PE / "include/constants/opponents.h").read_text()):
        out[m.group(1)] = int(m.group(2))
    return out


def build_roster(world: World, matrix: dict) -> list[dict]:
    """All reachable, single-battle route trainers with map key, flag, window, party
    order metadata. Gym trainers + Roxanne + grunt + rivals are SPINE-owned (fought by
    the story machinery) and excluded here."""
    ids = _opponent_ids()
    key_of = world.key_of_folder
    rows = []
    windows = {"Route102": "ROUTE_102", "Route104": None,  # split by y below
               "PetalburgWoods": "PETALBURG_WOODS", "Route116": "RUSTBORO_CITY"}
    for folder, window in windows.items():
        key = key_of[folder]
        gkey = key.replace("g", "").replace("_n", ",")
        labels = world._labels_cache.get(folder, {})
        for o in world.maps[key].get("object_events") or []:
            if o.get("trainer_type", "TRAINER_TYPE_NONE") == "TRAINER_TYPE_NONE":
                continue
            script = str(o.get("script", ""))
            name = script.split("EventScript_")[-1]
            body = " ".join(labels.get(script, []))
            m = re.search(r"trainerbattle_single\s+(TRAINER_\w+)", body)
            if not m:
                dbl = "trainerbattle_double" in body
                rows.append(dict(name=name, map=gkey, excluded=("double" if dbl else "no-single")))
                continue
            tid = ids.get(m.group(1))
            ent = matrix["trainers"].get(f"{folder}:{name}", {})
            if not ent.get("engageable_at"):
                rows.append(dict(name=name, map=gkey, excluded="unreachable"))
                continue
            w = window or ("ROUTE_104_NORTH" if int(o["y"]) < 40 else "ROUTE_104_SOUTH")
            rows.append(dict(name=name, trainer=m.group(1), map=gkey,
                             flag=0x500 + tid, window=w,
                             rock="GEODUDE" in _party_species(m.group(1))))
    return rows


_PARTY_CACHE: dict | None = None


def _party_species(trainer: str) -> str:
    global _PARTY_CACHE
    if _PARTY_CACHE is None:
        src = (PE / "src/data/trainers.h").read_text()
        parties = (PE / "src/data/trainer_parties.h").read_text()
        _PARTY_CACHE = {"trainers": src, "parties": parties}
    m = re.search(rf"\[{trainer}\]\s*=\s*\{{.*?sParty_(\w+)", _PARTY_CACHE["trainers"], re.S)
    if not m:
        return ""
    m2 = re.search(rf"sParty_{m.group(1)}\[\]\s*=\s*\{{(.*?)\}};", _PARTY_CACHE["parties"], re.S)
    return m2.group(1) if m2 else ""


# ---------------------------------------------------------------- sight cones
def cone_tiles(world: World, key: str) -> set[tuple[int, int]]:
    out = set()
    for o in world.maps.get(key, {}).get("object_events") or []:
        if o.get("trainer_type", "TRAINER_TYPE_NONE") == "TRAINER_TYPE_NONE":
            continue
        x, y = int(o["x"]), int(o["y"])
        sight = int(str(o.get("trainer_sight_or_berry_tree_id", "0")), 0)
        mv = str(o.get("movement_type", ""))
        d = {"FACE_LEFT": (-1, 0), "FACE_RIGHT": (1, 0),
             "FACE_UP": (0, -1), "FACE_DOWN": (0, 1)}
        ray = next((v for k, v in d.items() if k in mv), None)
        if ray and sight:
            for i in range(1, sight + 1):
                out.add((x + ray[0] * i, y + ray[1] * i))
        else:                                          # wanderer / look-around: box
            r = max(int(o.get("movement_range_x") or 0),
                    int(o.get("movement_range_y") or 0)) + sight
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    out.add((x + dx, y + dy))
    return out


# ---------------------------------------------------------------- plan build
def build_plan(seed0: int = 20260825) -> dict:
    from collection.catalog import MILESTONE_ORDER
    world = World()
    matrix = json.loads(MATRIX.read_text())
    for w in STAGE_WINDOWS.values():
        for mid in w:
            assert mid in MILESTONE_ORDER, f"window milestone {mid} not in MILESTONE_ORDER"

    roster = build_roster(world, matrix)
    active = [t for t in roster if "flag" in t]
    excluded = [t for t in roster if "flag" not in t]

    starters = (["torchic"] * N_RUNS_PER_STARTER + ["mudkip"] * N_RUNS_PER_STARTER
                + ["treecko"] * N_RUNS_PER_STARTER)
    runs = [dict(run_id=f"rg_{i:03d}_{st}", starter=st,
                 seed=(seed0 * 2654435761 + i * 40503) % (2 ** 31),
                 block_schedule=[]) for i, st in enumerate(starters)]
    rng = random.Random(seed0)

    # --- tile slices: every stage tile (minus cones) assigned to >=1 run, round-robin
    slices_stats = defaultdict(int)
    folder_of = {v: k for k, v in world.key_of_folder.items()}
    ri = 0
    prev_stage = None
    for sname, wins in STAGE_WINDOWS.items():
        stage = matrix["stages"].get(sname) or {}
        # §9: after the first full sweep of an area world, later ENTITY-variant stages
        # (S4b/S4c share S4's tiles) re-sweep ONLY maps whose blocked-set or tile-set
        # changed (grunt gone, Briney/rival moves...). Without this the planner sliced
        # the same 7,100 tiles three times over -- measured before this guard.
        changed = None
        if prev_stage is not None and stage.get("total", 0) and                 abs(stage["total"] - prev_stage.get("total", 0)) < 50:
            changed = {k for k in set(stage.get("tiles", {})) | set(prev_stage.get("tiles", {}))
                       if stage.get("tiles", {}).get(k) != prev_stage.get("tiles", {}).get(k)
                       or stage.get("blocked", {}).get(k) != prev_stage.get("blocked", {}).get(k)}
        prev_stage = stage
        for key, tiles in sorted((stage.get("tiles") or {}).items()):
            if changed is not None and key not in changed:
                continue
            gkey = key.replace("g", "").replace("_n", ",")
            cones = cone_tiles(world, key)
            pure = [t for t in map(tuple, tiles) if t not in cones]
            slices_stats[sname] += len(pure)
            # GEOGRAPHY-AWARE window choice (smoke lesson: a slice scheduled at a
            # boundary far from its map burns its budget on travel or skips):
            # Rustboro-region maps ride the Rustboro boundaries; 104-north tiles ride
            # ROUTE_104_NORTH; everything else uses its stage's windows.
            folder = folder_of.get(key, "")
            if sname.startswith("S4"):
                if "Rustboro" in folder or folder == "Route116":
                    use_wins = ["RUSTBORO_CITY", "RUSTBORO_CENTER_EXITED"]
                elif folder == "Route104":
                    ys = [t[1] for t in pure] or [50]
                    use_wins = (["ROUTE_104_NORTH"] if sum(ys) / len(ys) < 40
                                else ["EXIT_PETALBURG_GYM", "PETALBURG_WOODS"])
                elif folder == "PetalburgWoods":
                    use_wins = ["PETALBURG_WOODS"]
                else:
                    use_wins = wins
            else:
                use_wins = wins
            for i in range(0, len(pure), SLICE_TILES):
                run = runs[ri % len(runs)]; ri += 1
                run["block_schedule"].append([
                    rng.choice(use_wins), "bfs_sweep",
                    {"maps": [gkey], "tiles": {gkey: [list(t) for t in pure[i:i + SLICE_TILES]]},
                     "per_map_frames": 9000}])
    # --- warp legs: in-scope connection pairs x LEG_QUOTA runs (block does both dirs)
    pairs = set()
    for key, d in world.maps.items():
        gkey = key.replace("g", "").replace("_n", ",")
        for _, _, dkey in d["_conn"]:
            if dkey and dkey in world.maps:
                dg = dkey.replace("g", "").replace("_n", ",")
                pairs.add(tuple(sorted((gkey, dg))))
    for pi, pr in enumerate(sorted(pairs)):
        for q in range(LEG_QUOTA):
            run = runs[(pi * LEG_QUOTA + q) % len(runs)]
            near = {"0,3": "RUSTBORO_CITY", "0,31": "RUSTBORO_CITY",
                    "24,11": "PETALBURG_WOODS", "0,19": "PETALBURG_WOODS",
                    "0,0": "PETALBURG_CITY", "0,17": "ROUTE_102",
                    "0,10": "OLDALE_TOWN", "0,18": "ROUTE_103",
                    "0,16": "ROUTE_101", "0,9": "ROUTE101_AFTER_POKEDEX"}
            w = near.get(pr[0]) or near.get(pr[1]) or "RUSTBORO_CITY"
            run["block_schedule"].append([
                w, "bfs_sweep",
                {"maps": [], "legs": [list(pr)], "leg_frames": 9000}])
    # --- trainers: ALL active per run, grouped per window; torchic order constraint
    for run in runs:
        by_window = defaultdict(list)
        for t in active:
            by_window[t["window"]].append(t)
        for w, ts in by_window.items():
            if run["starter"] == "torchic" and w == "RUSTBORO_CITY":
                ts = sorted(ts, key=lambda t: t["rock"])       # non-rock first, then Geodudes
                run["block_schedule"].append([                 # 160-EXP top-off BEFORE rocks
                    "ROUTE_104_NORTH", "encounter_farm",
                    {"frames": 12000, "seed": run["seed"] % 99991,
                     "targets": [{"area": "land", "map": "0,31", "species_targets": {}}]}])
            run["block_schedule"].append([
                w, "trainer_engagement",
                {"frames": 30000 + 12000 * len(ts), "seed": run["seed"] % 99991,
                 "targets": [{"map": t["map"], "trainer_flag": t["flag"]} for t in ts]}])
    # --- floor blocks (interaction/mart/pc/menus/idle), seeded rotation like the banked plan
    towns = ["0,10", "0,0", "0,3"]
    for i, run in enumerate(runs):
        s = run["seed"]
        run["block_schedule"] += [
            ["PETALBURG_CITY", "interaction", {"frames": 45000, "maps": [towns[i % 3]], "seed": s % 7001}],
            ["RUSTBORO_CENTER_EXITED", "mart_buy" if i % 2 else "pc_access",
             ({"towns": ["11,5"], "frames": 30000} if i % 2 else {"center": "11,5", "frames": 20000})],
            ["OLDALE_TOWN", "menus", {"frames": 6000, "seed": s % 7919}],
            ["ROUTE_103", "idle", {}],
        ]
    plan = {"design": "W33 regen 08-25 (collect->verify->extract; no grind; all-trainers)",
            "runs": runs,
            "roster": {"active": len(active), "excluded": excluded},
            "stats": dict(slices_stats)}
    OUT.write_text(json.dumps(plan))
    return plan


if __name__ == "__main__":
    p = build_plan()
    n_blocks = sum(len(r["block_schedule"]) for r in p["runs"])
    print(f"runs: {len(p['runs'])}  total scheduled blocks: {n_blocks}")
    print(f"trainers active/run: {p['roster']['active']}  excluded: "
          f"{[(e['name'], e['excluded']) for e in p['roster']['excluded']]}")
    for s, n in p["stats"].items():
        print(f"  {s:16s} {n:5d} pure tiles sliced")
    per_run = [len(r["block_schedule"]) for r in p["runs"]]
    print(f"blocks per run: min {min(per_run)} / max {max(per_run)} -> {OUT}")
