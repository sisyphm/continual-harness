"""W33 corpus-v2 §2/§9/§10 rotation scheduler: seeded, RECORDED expedition plans.

No run does all the coverage work: this module assigns each of the fleet's N runs a
subset of sweep / farm / interaction / mart / grind targets such that the UNION over
the non-holdout runs closes the coverage requirements, then self-audits the plan
(--check) BEFORE any collection. The plan is a pure function of
(seed, N, floors, change matrix, coverage manifest): same inputs -> byte-identical
w33_expedition_plan.json, so the fleet is re-plannable (audit a partial fleet,
schedule the deficit into the remaining runs with a new seed).

Inputs
  * fleet size N (default 50) + seed
  * per-phase budget targets (spec §10.4): spine 45 / bfs_sweep 20 / encounter+fishing
    12 / interaction+mart_pc_item+menus 10 / grind+idle 8 / slack 5 (percent)
  * the change matrix (plan_change_matrix.py -> w33_change_matrix.json)
  * the coverage manifest (maps, wild tables, connections)
  * the audit floor constants (audits/coverage.py FLOOR) — connection_rounds is
    FLOOR["warp_traversals"]; species sightings derive from FLOOR["species_enemy"]

Union requirements closed by the plan (spec §2 rotation + item-4 acceptance):
  * every map swept at >= 2 anchor stages, + every changed (map, stage) cell of the
    change matrix swept inside its stage window
  * every in-scope connection pair crossed both directions >= connection_rounds
    (a bfs_sweep leg = one A->B->A round trip = +1 per direction)
  * every land wild-table species >= its floor sightings (encounter_farm targets)
  * every town's mart / pc / item verbs (mart_buy, pc_access, item_use blocks)
  * interaction verbs on every in-scope map (interaction blocks)
  * grind_evolve on >= grind_runs_per_starter non-holdout runs per starter
    (3 starters x evolutions; aggregate >= 3 planned evolution cutscenes)
  * idle / menus sprinkled across runs
  * task #46: every tracked OPTIONAL trainer assigned to >= trainer_runs_per_trainer
    distinct non-holdout runs (trainer_engagement blocks at reachable, spine-safe
    stages — see TRAINER_STAGE); grunt/Josh/Roxanne are spine milestones in every
    run and never block-scheduled
  * no block scheduled at an INELIGIBLE (scripted-tail) milestone boundary
    (INELIGIBLE_BLOCK_BOUNDARIES; pilot evidence: ROUTE_101 == index 14 has no
    free-overworld frame, so blocks placed there skip on precondition every run)

Documented exclusions (surfaced in the plan's "excluded" section, never silent):
  * fish-table species (Old Rod slots): the fishing block is the reserved registry
    slot in schedule.EXPEDITION_BLOCKS — pilot follow-up, not silently unplanned
  * water/rock tables + the Surf-gated Route 115 land table ("0,30" species), per
    the live-verified access policy in audits/coverage.py
  * "25,40" (intro truck interior): exists only during the intro cutscene; no
    block-safe stage can stand on it (every spine records it anyway)

Stage model / reachability (documented limits)
  * Block anchor stages = schedule._WANDER_OK (S1-proven overworld-safe hand-off
    milestones) + PILOT_VERIFY_ANCHORS, the extra milestones the change matrix
    REQUIRES for otherwise-unplannable windows (gym gauntlet cells, post-May
    Route 103/Oldale, post-grunt Woods). Pilot-verify anchors carry ONLY the cells
    that demand them (routine work stays on proven anchors); blocks skip cleanly
    via the precondition seam if the hand-off is not free-overworld, so a wrong
    guess costs coverage, never a run.
  * unlocked(map, stage): map was the spine's own map at some stage <= this one, is
    one warp/connection hop from such a map, or is an interior chained behind an
    unlocked map. Graph reachability is OPTIMISTIC about script blockers; outdoor
    maps beyond 1 hop from spine-visited maps stay locked to bound that optimism. A
    cell whose window contains no unlocked safe anchor is reported in "unplannable"
    (never silently dropped). Intro-corridor cells (Littleroot before free roam)
    are the expected residents of that list — the spine itself records them.

Frame-cost constants (planning estimates, provenance in comments; re-checked at the
pilot — the plan records them so a re-plan can swap in measured values).

Usage:
  .venv/bin/python -m collection.plan_expeditions --n 50 --seed 33 \
      --data_root ../pokemon-worldmodel/data
  .venv/bin/python -m collection.plan_expeditions --check \
      --plan ../pokemon-worldmodel/data/processed/w33_expedition_plan.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from collection.audits.coverage import FLOOR
from collection.catalog import MILESTONE_ORDER
from collection.plan_change_matrix import map_windows, window_of
from collection.playthrough.blocks.trainer_engagement import (
    SPINE_TRAINER_FLAGS, TRACKED_TRAINER_FLAGS, TRAINER_FLAG_BASE)
from collection.playthrough.schedule import _WANDER_OK

STARTERS = ("mudkip", "torchic", "treecko")

# §10.4 per-phase budget targets (percent of a run's frames)
PHASE_BUDGET_PCT = {"spine": 45, "bfs_sweep": 20, "encounter": 12,
                    "interaction_family": 10, "grind_idle": 8, "slack": 5}

# ---- planning cost constants (estimates; the pilot re-measures) -------------------
SPINE_FRAMES_EST = 130_000      # §4: ~250-300k/run post-pacing-cut x 45% spine share
RUN_FRAMES_TARGET = SPINE_FRAMES_EST * 100 // PHASE_BUDGET_PCT["spine"]
EST_FRAMES_PER_TILE = 40        # 17 f/tile condition-paced walk + replan/NPC overhead
EST_SWEEP_BASE = 2500           # travel to the map + settle
EST_FRAMES_PER_SIGHTING = 1200  # test pin: wild battle every ~0.5-1.5k paced frames
EST_LEG_FRAMES = 3000           # pin: 101<->Oldale round trip ~240 f + edge travel
EST_INTERACTION_BASE = 2000
EST_FRAMES_PER_TARGET = 1200    # talk/sign/dialog cycle incl. approach
WALKABLE_SHARE = 0.5            # of map area (planning heuristic)

# floors the PLAN must close (fleet defaults; tests pass a reduced set)
PLAN_FLOORS = {
    "anchor_stages_per_map": 2,                    # §9 full sweeps at anchor stages
    "connection_rounds": FLOOR["warp_traversals"],  # both directions >= 5
    # FLOOR["species_enemy"]=3000 frames / ~400 visible-foe frames per
    # battle-and-flee sighting => 8 sightings (documented derivation)
    "species_sightings": max(3, FLOOR["species_enemy"] // 400),
    "grind_runs_per_starter": 2,                   # >= 1 evolution each, x2 redundancy
    "mart_runs_per_town": 1,
    "pc_runs_per_center": 1,
    "item_use_runs": 2,                            # overworld + battle legs
    "idle_run_share": 4,                           # >= N//4 runs carry an idle block
    "menus_run_share": 4,
    # task #46: each tracked OPTIONAL trainer assigned to >= 2 distinct non-holdout
    # runs (the audit floor is 2 runs with the flag set at run end); the three
    # spine-guaranteed trainers are covered by every completing run's milestones
    "trainer_runs_per_trainer": 2,
}

# task #46 trainer-engagement stages: the earliest SPINE-SAFE milestone at which each
# tracked trainer's POSITION is reachable. Route 104 is ONE map key ("0,19") whose
# south and north halves are only joined through Petalburg Woods, so south trainers
# ride ROUTE_104_SOUTH and north trainers ROUTE_104_NORTH. Petalburg Woods trainers
# wait for TEAM_AQUA_GRUNT_DEFEATED (a block must never pre-trigger the grunt
# cutscene the spine owns), and the Rustboro Gym pair waits for TRAINER_JOSH_BATTLE
# (fighting gym trainers before the Josh milestone would strand its policy against
# an already-beaten trainer). Karen stands on Route 116 ("0,31"), one connection hop
# east of Rustboro. All stages are SAFE_ANCHORS members (proven or pilot_verify).
TRAINER_STAGE = {
    TRAINER_FLAG_BASE + 318: "ROUTE_102",               # CALVIN   (0,17)
    TRAINER_FLAG_BASE + 333: "ROUTE_102",               # ALLEN    (0,17)
    TRAINER_FLAG_BASE + 603: "ROUTE_102",               # TIANA    (0,17)
    TRAINER_FLAG_BASE + 615: "ROUTE_102",               # RICK     (0,17)
    TRAINER_FLAG_BASE + 114: "ROUTE_104_SOUTH",         # CINDY    (0,19 south, y=44)
    TRAINER_FLAG_BASE + 319: "ROUTE_104_SOUTH",         # BILLY    (0,19 south, y=67)
    TRAINER_FLAG_BASE + 616: "TEAM_AQUA_GRUNT_DEFEATED",  # LYLE   (24,11)
    TRAINER_FLAG_BASE + 621: "TEAM_AQUA_GRUNT_DEFEATED",  # JAMES  (24,11)
    TRAINER_FLAG_BASE + 136: "ROUTE_104_NORTH",         # WINSTON  (0,19 north, y=25)
    TRAINER_FLAG_BASE + 337: "ROUTE_104_NORTH",         # IVAN     (0,19 north, y=8)
    TRAINER_FLAG_BASE + 483: "ROUTE_104_NORTH",         # GINA&MIA (0,19 north, y=15)
    TRAINER_FLAG_BASE + 604: "ROUTE_104_NORTH",         # HALEY    (0,19 north, y=24)
    TRAINER_FLAG_BASE + 280: "RUSTBORO_CITY",           # KAREN    (0,31 Route 116)
    TRAINER_FLAG_BASE + 321: "TRAINER_JOSH_BATTLE",     # TOMMY    (11,3)
    TRAINER_FLAG_BASE + 571: "TRAINER_JOSH_BATTLE",     # MARC     (11,3)
}
EST_TRAINER_BASE = 3000         # travel to the map + settle (planning estimate)
EST_TRAINER_FRAMES = 9000       # approach + intro text + battle + outro per trainer
TRAINER_BLOCK_FRAMES = 60_000   # generous block deadline (est governs the load)

# Anchors beyond the S1-proven _WANDER_OK set, needed by change-matrix windows that
# would otherwise be unplannable (gym mid-gauntlet cells; post-May Route 103/Oldale;
# post-grunt Petalburg Woods). All end in free overworld by construction of their
# postconditions but are NOT S1-proven as block hand-offs -> flagged pilot_verify;
# a wrong guess skips cleanly (precondition seam) and shows up as an audit deficit.
PILOT_VERIFY_ANCHORS = {"TRAINER_JOSH_BATTLE", "ROXANNE_BATTLE",
                        "MAY_ROUTE103_INTERACTION", "BACK_TO_OLDALE_FROM_ROUTE103",
                        "TEAM_AQUA_GRUNT_DEFEATED",
                        # W33 grind leg 2: freshly healed boundary (the spine's
                        # HEAL_AT_RUSTBORO_CENTER just restored HP+PP), back in free
                        # overworld outside the Center — hosts the evolution grind.
                        "RUSTBORO_CENTER_EXITED"}

# Milestone boundaries INELIGIBLE for block placement (live pilot evidence): these
# milestones' completed checkpoints sit on a SCRIPTED TAIL with no free-overworld
# frame (ROUTE_101 == plan index 14 is the measured case — the Birch-rescue trigger
# fires the moment the player lands on the route), so any block scheduled there
# skips on the precondition seam every single run. The authoritative signal is
# collect_events._EVENTS_ACCEPTING_UNRESPONSIVE_TARGET — the exact milestones whose
# targets needed position-only acceptance BECAUSE memory reads free_overworld with a
# script/dialog still live. Imported (not copied) so a policy-side addition follows
# automatically. Block-side skip-with-reason stays as defense in depth; the planner
# simply never schedules into a boundary that cannot host a block.
import collection.collect_events as _ce
INELIGIBLE_BLOCK_BOUNDARIES = frozenset(_ce._EVENTS_ACCEPTING_UNRESPONSIVE_TARGET)

SAFE_ANCHORS = sorted((_WANDER_OK | PILOT_VERIFY_ANCHORS) - INELIGIBLE_BLOCK_BOUNDARIES)
EXCLUDED_MAPS = {"25,40"}       # intro truck interior (see module docstring)
SURF_GATED_LAND = {"0,30"}      # audits/coverage.py access policy
ENC_BLOCK_FRAMES = 35_000       # one encounter block ~= the per-run encounter budget
# Per-LEG grind budget (two legs around the Rustboro heal, each heal-capable — see
# the grind_evolve scheduling comment). Sized from the pilot's measured rates:
# ~1.9k frames per won battle, ~9-15 wins per full-HP/PP cycle, ~30 wins for
# 12->16, plus ~5-8k frames per nurse round trip — ~110k worst case across cycles;
# 100k per leg with two legs gives ample headroom. The block ends early on
# target_level, so the budget is a cap, not a cost.
GRIND_FRAMES = 520_000   # L16 from any start: measured ~90k/level top-end + two-floor grind (08-21)
# Center interior for mid-grind nurse trips (Rustboro's, adjacent to both legs'
# grass): restores the REAL sustain constraint, PP, along with HP.
GRIND_HEAL_CENTER = "11,5"
GRIND_TARGET_LEVEL = 16         # all three starters evolve at 16
IDLE_FRAMES, MENUS_FRAMES = 4000, 6000


# ---- shared derivations ------------------------------------------------------------

def _midx(stage: str) -> int:
    return MILESTONE_ORDER.index(stage)


def _neighbors(manifest: dict) -> dict[str, set[str]]:
    maps, scope = manifest["maps"], set(manifest["scope_maps"])
    nb: dict[str, set[str]] = {k: set() for k in scope}
    for a in scope:
        if a not in maps:
            continue
        for hop in list(maps[a]["warps"]) + list(maps[a]["connections"]):
            b = hop["dst_map"]
            if b in scope:
                nb[a].add(b)
                nb[b].add(a)
    return nb


def unlocked_at(matrix: dict, manifest: dict) -> dict[str, set[str]]:
    """stage -> maps unlocked: spine-visited by then, 1 hop away from a visited map
    (adjacent routes/interiors), or any INTERIOR (group != 0) chained behind an
    unlocked map (2F rooms, house-behind-house). Outdoor maps beyond 1 hop stay
    locked — that's where script blockers live (documented optimism bound)."""
    nb = _neighbors(manifest)
    scope = set(manifest["scope_maps"])
    visited: set[str] = set()
    out: dict[str, set[str]] = {}
    for s in matrix["stages"]:
        pm = matrix["player_map"].get(s)
        if pm:
            visited.add(pm)
        u = set(visited)
        changed = True
        while changed:
            changed = False
            for a in list(u):
                for b in nb.get(a, set()):
                    if b in u:
                        continue
                    if a in visited or not b.startswith("0,"):
                        u.add(b)
                        changed = True
        out[s] = u & scope
    return out


def build_requirements(manifest: dict, matrix: dict, floors: dict) -> dict:
    """The explicit requirement set both the scheduler and --check derive from."""
    scope = [k for k in manifest["scope_maps"] if k not in EXCLUDED_MAPS]
    unlocked = unlocked_at(matrix, manifest)
    anchors = [a for a in SAFE_ANCHORS if a in matrix["stages"]]

    def usable_anchors(key: str) -> list[str]:
        return sorted((a for a in anchors if key in unlocked[a]), key=_midx)

    # sweep cells: every post-init changed window needs a sweep planned inside it
    cells: dict[str, list[dict]] = {}
    unplannable = []
    for k in scope:
        wins = map_windows(matrix, k)
        ua = usable_anchors(k)
        rows = []
        for i, w in enumerate(wins[1:], start=1):     # changed windows only
            end = _midx(wins[i + 1]) if i + 1 < len(wins) else len(MILESTONE_ORDER)
            inside = [a for a in ua if _midx(w) <= _midx(a) < end]
            if inside:
                rows.append({"window": w, "anchors": inside})
            else:
                unplannable.append({"map": k, "window": w,
                                    "reason": "no unlocked safe anchor in window"})
        cells[k] = rows

    pairs = sorted({tuple(sorted((a, b)))
                    for a in scope if a in manifest["maps"]
                    for b in ({w["dst_map"] for w in manifest["maps"][a]["warps"]}
                              | {c["dst_map"] for c in manifest["maps"][a]["connections"]})
                    if b in scope and b != a})

    wild = manifest["wild"]
    land_maps = sorted(k for k in scope
                       if "land" in wild.get(k, {}) and k not in SURF_GATED_LAND)
    species_map: dict[int, str] = {}
    for k in land_maps:                                # first (earliest-key) map wins
        for sp, *_ in wild[k]["land"]["mons"]:
            species_map.setdefault(int(sp), k)

    towns = sorted(t for t in scope if t in manifest["maps"] and any(
        w["dst_map"] in manifest["maps"]
        and any(o.get("gfx") == 83 for o in manifest["maps"][w["dst_map"]]["objects"])
        and w["dst_map"] != t and w["dst_map"].split(",")[0] != "0"
        for w in manifest["maps"][t]["warps"]))
    centers = sorted(c for c in ("2,2", "8,4", "11,5") if c in scope)

    # task #46: tracked-trainer targets. flag -> manifest map (the object whose
    # parsed `trainerbattle` script id matches flag - 0x500 — the empirically
    # verified mapping, see blocks/trainer_engagement.py), grouped per (stage, map)
    # from TRAINER_STAGE. Spine-guaranteed flags are never block-scheduled; a flag
    # whose stage/map fails validation lands in "unplannable", never silently out.
    flag_map: dict[int, str] = {}
    for k in scope:
        if k not in manifest["maps"]:
            continue
        for o in manifest["maps"][k]["objects"]:
            tid = o.get("trainer_id")
            if tid and TRAINER_FLAG_BASE + tid in TRACKED_TRAINER_FLAGS:
                flag_map.setdefault(TRAINER_FLAG_BASE + tid, k)
    trainer_groups: dict[tuple[str, str], list[int]] = {}
    for f in TRACKED_TRAINER_FLAGS:
        if f in SPINE_TRAINER_FLAGS:
            continue
        k, stage = flag_map.get(f), TRAINER_STAGE.get(f)
        if k is None or stage is None or stage not in matrix["stages"] \
                or stage in INELIGIBLE_BLOCK_BOUNDARIES \
                or k not in unlocked.get(stage, set()):
            unplannable.append({"trainer_flag": f,
                                "reason": "no manifest object / stage missing, "
                                          "ineligible, or map locked at stage"})
            continue
        trainer_groups.setdefault((stage, k), []).append(f)

    fish_sp = sorted({int(sp) for k in scope
                      for sp, *_ in wild.get(k, {}).get("fish", {}).get("mons", [])[:2]})
    excluded = [
        {"what": f"species {sorted(set(fish_sp) - set(species_map))}",
         "reason": "fish tables need the fishing block — reserved registry slot "
                   "(schedule.EXPEDITION_BLOCKS), pilot follow-up"},
        {"what": "water/rock tables + '0,30' land table",
         "reason": "Surf/Rock Smash gated post-scope (audits/coverage.py policy)"},
        {"what": "map 25,40", "reason": "intro truck interior — cutscene-only"},
    ]
    return {"scope": scope, "unlocked": unlocked, "anchors": anchors,
            "usable_anchors": {k: usable_anchors(k) for k in scope},
            "cells": cells, "unplannable": unplannable, "pairs": pairs,
            "species_map": species_map, "land_maps": land_maps,
            "towns": towns, "centers": centers, "excluded": excluded,
            "trainer_groups": trainer_groups}


def est_sweep_frames(manifest: dict, key: str) -> int:
    m = manifest["maps"].get(key, {"width": 12, "height": 10})
    est = EST_SWEEP_BASE + int(m["width"] * m["height"] * WALKABLE_SHARE
                               * EST_FRAMES_PER_TILE)
    return min(est, 45_000)


# ---- the scheduler -----------------------------------------------------------------

def build_plan(*, n_runs: int = 50, seed: int = 33,
               data_root: str | Path = "../pokemon-worldmodel/data",
               manifest: dict | None = None, matrix: dict | None = None,
               floors: dict | None = None) -> dict:
    root = Path(data_root)
    if manifest is None:
        manifest = json.loads((root / "processed/coverage_manifest.json").read_text())
    if matrix is None:
        matrix = json.loads((root / "processed/w33_change_matrix.json").read_text())
    fl = dict(PLAN_FLOORS)
    fl.update(floors or {})
    req = build_requirements(manifest, matrix, fl)
    rng = random.Random(seed)

    runs = [{"run_id": f"exp_{i:03d}_{STARTERS[i % 3]}", "starter": STARTERS[i % 3],
             "seed": rng.getrandbits(30), "holdout": False, "blocks": []}
            for i in range(n_runs)]
    # §10.1 holdout: 3 full expeditions, one per starter, reserved at the SCHEDULER
    # level — excluded from every union requirement below.
    for st in STARTERS:
        cand = [r for r in runs if r["starter"] == st]
        if cand:
            rng.choice(cand)["holdout"] = True
    workers = [r for r in runs if not r["holdout"]] or runs

    load = {r["run_id"]: {"bfs_sweep": 0, "encounter": 0, "interaction_family": 0,
                          "grind_idle": 0} for r in runs}

    def pick(phase: str, exclude: set[str] = frozenset()) -> dict:
        """Least-loaded (in `phase` est-frames) worker; run_id tiebreak = determinism."""
        pool = [r for r in workers if r["run_id"] not in exclude] or workers
        return min(pool, key=lambda r: (load[r["run_id"]][phase], r["run_id"]))

    def add(run: dict, stage: str, name: str, args: dict, phase: str, est: int) -> None:
        run["blocks"].append({"stage": stage, "block": name, "args": args})
        load[run["run_id"]][phase] += est

    def routine_stage(stages: list[str]) -> str:
        """Latest S1-PROVEN anchor for routine work; pilot-verify anchors are only
        used by cells whose window demands them."""
        proven = [s for s in stages if s not in PILOT_VERIFY_ANCHORS]
        return (proven or stages)[-1]

    # -- sweeps: changed cells first (fixed anchors), then top-up to the anchor floor
    sweep_at: dict[tuple[str, str], list[str]] = {}   # (map, stage) -> run_ids
    for k in req["scope"]:
        for cell in req["cells"][k]:
            stage = cell["anchors"][0]                 # earliest anchor in the window
            r = pick("bfs_sweep")
            sweep_at.setdefault((k, stage), []).append(r["run_id"])
            add(r, stage, "bfs_sweep", {"maps": [k]}, "bfs_sweep",
                est_sweep_frames(manifest, k))
        ua = req["usable_anchors"][k]
        if not ua:
            req["unplannable"].append({"map": k, "reason": "sweep: no anchor"})
            continue
        want = min(fl["anchor_stages_per_map"], len(ua))
        have = {s for (m, s) in sweep_at if m == k}
        proven = [a for a in ua if a not in PILOT_VERIFY_ANCHORS] or ua
        order = [proven[0], proven[-1]] + proven[1:-1] + [a for a in ua
                                                          if a not in proven]
        for a in order:                                # earliest+latest PROVEN first
            if len(have) >= want:
                break
            if a in have:
                continue
            r = pick("bfs_sweep")
            sweep_at.setdefault((k, a), []).append(r["run_id"])
            add(r, a, "bfs_sweep", {"maps": [k]}, "bfs_sweep",
                est_sweep_frames(manifest, k))
            have.add(a)

    # -- direction-balance legs: each pair x connection_rounds, spread over runs
    for a, b in req["pairs"]:
        stages = [s for s in req["anchors"]
                  if a in req["unlocked"][s] and b in req["unlocked"][s]]
        if not stages:
            req["unplannable"].append({"pair": [a, b],
                                       "reason": "no anchor unlocks both endpoints"})
            continue
        used: set[str] = set()
        for _ in range(fl["connection_rounds"]):
            r = pick("bfs_sweep", exclude=used)
            used.add(r["run_id"])
            add(r, routine_stage(stages), "bfs_sweep", {"maps": [], "legs": [[a, b]]},
                "bfs_sweep", EST_LEG_FRAMES)

    # -- encounter farms: per land map, its assigned species split across blocks
    for k in req["land_maps"]:
        targets = {sp: fl["species_sightings"]
                   for sp, m in sorted(req["species_map"].items()) if m == k}
        if not targets:
            continue
        total = sum(targets.values())
        n_blocks = max(1, -(-total * EST_FRAMES_PER_SIGHTING // ENC_BLOCK_FRAMES))
        stages = [s for s in req["anchors"] if k in req["unlocked"][s]]
        used: set[str] = set()
        for bi in range(n_blocks):
            share = {str(sp): -(-n // n_blocks) for sp, n in sorted(targets.items())}
            r = pick("encounter", exclude=used)
            used.add(r["run_id"])
            add(r, routine_stage(stages), "encounter_farm",
                {"targets": [{"map": k, "area": "land", "species_targets": share}],
                 "frames": ENC_BLOCK_FRAMES, "seed": rng.getrandbits(20)},
                "encounter", min(ENC_BLOCK_FRAMES,
                                 sum(share.values()) * EST_FRAMES_PER_SIGHTING))

    # -- interaction: every map, grouped one map per item, late anchor (most actors)
    for k in req["scope"]:
        ua = req["usable_anchors"][k]
        if not ua:
            req["unplannable"].append({"map": k, "reason": "interaction: no anchor"})
            continue
        n_t = (len(manifest["maps"][k]["objects"]) + len(manifest["maps"][k]["signs"])
               + len(manifest["maps"][k]["warps"])) if k in manifest["maps"] else 6
        est = EST_INTERACTION_BASE + n_t * EST_FRAMES_PER_TARGET
        r = pick("interaction_family")
        add(r, routine_stage(ua), "interaction",
            {"maps": [k], "frames": min(60_000, est * 3), "seed": rng.getrandbits(20)},
            "interaction_family", est)

    # -- mart / pc / item verbs
    for town in req["towns"]:
        stages = [s for s in req["anchors"] if town in req["unlocked"][s]]
        used: set[str] = set()
        for _ in range(fl["mart_runs_per_town"]):
            r = pick("interaction_family", exclude=used)
            used.add(r["run_id"])
            add(r, routine_stage(stages), "mart_buy",
                {"towns": [town], "seed": rng.getrandbits(20)},
                "interaction_family", 20_000)
    for center in req["centers"]:
        stages = [s for s in req["anchors"] if center in req["unlocked"][s]]
        used = set()
        for _ in range(fl["pc_runs_per_center"]):
            r = pick("interaction_family", exclude=used)
            used.add(r["run_id"])
            add(r, routine_stage(stages), "pc_access",
                {"center": center, "seed": rng.getrandbits(20)},
                "interaction_family", 15_000)
    used = set()
    for _ in range(fl["item_use_runs"]):
        r = pick("interaction_family", exclude=used)
        used.add(r["run_id"])
        add(r, "OLDALE_AFTER_POKEDEX", "item_use",
            {"grass_map": "0,16", "town": "0,10", "seed": rng.getrandbits(20)},
            "interaction_family", 30_000)

    # -- trainer engagement (task #46): every tracked OPTIONAL trainer assigned to
    #    >= trainer_runs_per_trainer distinct non-holdout runs, grouped per
    #    (stage, map) so one block fights that map's whole tracked group; the
    #    spine-guaranteed three (grunt/Josh/Roxanne) are milestones in every run
    #    and never block-scheduled. Natural dodge-variance stays everywhere else —
    #    no run is forced to fight all 18.
    for (stage, key), t_flags in sorted(req["trainer_groups"].items()):
        t_flags = sorted(t_flags)
        used = set()
        for _ in range(fl["trainer_runs_per_trainer"]):
            r = pick("encounter", exclude=used)
            used.add(r["run_id"])
            add(r, stage, "trainer_engagement",
                {"targets": [{"map": key, "trainer_flag": f} for f in t_flags],
                 "frames": TRAINER_BLOCK_FRAMES, "seed": rng.getrandbits(20)},
                "encounter", EST_TRAINER_BASE + len(t_flags) * EST_TRAINER_FRAMES)

    # -- grind_evolve: per starter, on that starter's own non-holdout runs, in TWO
    # legs split across the Rustboro Center heal (W33 pilot, all measured). One
    # uninterrupted grind cannot reach the L16 evolution threshold: ~30+ won battles
    # on Route 104's L3-4 wilds exhausts attacking PP and drags HP down toward
    # hp_floor — two torchic runs stalled at L14 (45k ended=budget; 150k guard-cut
    # at hp 13) — and the higher-XP Route 116 alternative is not plannable from
    # Rustboro (goto_map burned 110k frames failing the hop;
    # ended=grass_map_unreachable). Leg 1 grinds Route 104 grass at the
    # ROUTE_104_NORTH boundary (12 -> ~14, ends honestly on budget/hp_floor); the
    # spine's HEAL_AT_RUSTBORO_CENTER milestone then restores HP+PP; leg 2 at
    # RUSTBORO_CENTER_EXITED hops back to the same grass with fresh resources
    # (14 -> 16 is ~15 wins, well inside one PP budget) and delivers the evolution
    # cutscene before the gym. The anchor maps must have land wild tables.
    _GRIND_LEGS = (              # (stage, grass map key) in spine order, heal between
        ("ROUTE_104_NORTH", "0,19"),
        ("RUSTBORO_CENTER_EXITED", "0,19"),
    )
    for _stage, _gmap in _GRIND_LEGS:
        assert _stage in SAFE_ANCHORS, f"grind leg stage {_stage} not an eligible anchor"
        assert "land" in manifest["wild"].get(_gmap, {}), \
            f"grind leg map {_gmap} has no land wild table"
    for st in STARTERS:
        cand = sorted((r for r in workers if r["starter"] == st),
                      key=lambda r: (load[r["run_id"]]["grind_idle"], r["run_id"]))
        for r in cand[:fl["grind_runs_per_starter"]]:
            for _stage, _gmap in _GRIND_LEGS:
                add(r, _stage, "grind_evolve",
                    {"target_level": GRIND_TARGET_LEVEL, "frames": GRIND_FRAMES,
                     "grass_map": _gmap, "heal_center": GRIND_HEAL_CENTER,
                     "seed": rng.getrandbits(20)}, "grind_idle", GRIND_FRAMES)

    # -- idle / menus sprinkled (deterministic fix-up guarantees the floor)
    want_idle = max(1, n_runs // fl["idle_run_share"])
    want_menus = max(1, n_runs // fl["menus_run_share"])
    idle_runs = [r for r in workers if rng.random() < 0.5][:max(want_idle, 1)]
    for r in (idle_runs + [r for r in workers if r not in idle_runs])[:want_idle]:
        add(r, "OLDALE_TOWN", "idle",
            {"frames": IDLE_FRAMES, "seed": rng.getrandbits(20)},
            "grind_idle", IDLE_FRAMES)
    menus_runs = [r for r in workers if rng.random() < 0.5][:max(want_menus, 1)]
    for r in (menus_runs + [r for r in workers if r not in menus_runs])[:want_menus]:
        add(r, "OLDALE_TOWN", "menus",
            {"frames": MENUS_FRAMES, "seed": rng.getrandbits(20)},
            "grind_idle", MENUS_FRAMES)

    # -- holdout runs: light seeded diversity schedule (never load-bearing)
    for r in runs:
        if not r["holdout"]:
            continue
        add(r, "OLDALE_AFTER_POKEDEX", "bfs_sweep", {"maps": ["0,10"]},
            "bfs_sweep", est_sweep_frames(manifest, "0,10"))
        add(r, "OLDALE_TOWN", "idle",
            {"frames": IDLE_FRAMES, "seed": rng.getrandbits(20)},
            "grind_idle", IDLE_FRAMES)
        add(r, "OLDALE_TOWN", "menus",
            {"frames": MENUS_FRAMES, "seed": rng.getrandbits(20)},
            "grind_idle", MENUS_FRAMES)

    # -- merge per (run, stage): one bfs_sweep block carrying maps+legs; emit sorted
    caps = {"bfs_sweep": PHASE_BUDGET_PCT["bfs_sweep"],
            "encounter": PHASE_BUDGET_PCT["encounter"],
            "interaction_family": PHASE_BUDGET_PCT["interaction_family"],
            "grind_idle": PHASE_BUDGET_PCT["grind_idle"]}
    out_runs = []
    for r in runs:
        merged: dict[tuple[str, str], dict] = {}
        rest = []
        for b in r["blocks"]:
            if b["block"] == "bfs_sweep":
                m = merged.setdefault((b["stage"], "bfs_sweep"),
                                      {"maps": [], "legs": []})
                m["maps"] += b["args"].get("maps", [])
                m["legs"] += b["args"].get("legs", [])
            else:
                rest.append(b)
        sched = []
        for (stage, _), args in merged.items():
            a = {"maps": sorted(dict.fromkeys(args["maps"]))}
            if args["legs"]:
                a["legs"] = sorted(args["legs"])
            sched.append([_midx(stage), "bfs_sweep", a])
        for b in rest:
            sched.append([_midx(b["stage"]), b["block"], b["args"]])
        sched.sort(key=lambda e: (e[0], e[1], json.dumps(e[2], sort_keys=True)))
        shares = {ph: round(v * 100 / RUN_FRAMES_TARGET, 1)
                  for ph, v in load[r["run_id"]].items()}
        warns = [f"{ph} est {shares[ph]}% > target {caps[ph]}%"
                 for ph in caps if shares[ph] > caps[ph] * 1.25]
        out_runs.append({
            "run_id": r["run_id"], "starter": r["starter"], "seed": r["seed"],
            "holdout": r["holdout"], "block_schedule": sched,
            "est_phase_share_pct": shares, "budget_warnings": warns,
        })

    return {
        "schema_version": 1,
        "seed": seed, "n_runs": n_runs,
        "floors": fl,
        "budget_targets_pct": PHASE_BUDGET_PCT,
        "constants": {"SPINE_FRAMES_EST": SPINE_FRAMES_EST,
                      "RUN_FRAMES_TARGET": RUN_FRAMES_TARGET,
                      "EST_FRAMES_PER_TILE": EST_FRAMES_PER_TILE,
                      "EST_FRAMES_PER_SIGHTING": EST_FRAMES_PER_SIGHTING,
                      "EST_LEG_FRAMES": EST_LEG_FRAMES},
        "pilot_verify_anchors": sorted(PILOT_VERIFY_ANCHORS),
        "excluded": req["excluded"],
        "unplannable": req["unplannable"],
        "runs": out_runs,
    }


def plan_run_to_expedition_entries(run: dict) -> list[dict]:
    """One plan run -> director `expedition=[...]` entries ({'after': milestone_id,
    'block': name, **args}) for run_playthrough / build_expedition_schedule."""
    return [{"after": MILESTONE_ORDER[idx], "block": name, **args}
            for idx, name, args in run["block_schedule"]]


# ---- --check: coverage recomputed FROM THE PLAN ------------------------------------

def check_plan(plan: dict, *, manifest: dict, matrix: dict) -> dict:
    fl = plan["floors"]
    req = build_requirements(manifest, matrix, fl)
    workers = [r for r in plan["runs"] if not r["holdout"]]
    rows: dict[str, list[dict]] = {}

    def blocks(name: str):
        for r in workers:
            for idx, bn, args in r["block_schedule"]:
                if bn == name:
                    yield r, MILESTONE_ORDER[idx], args

    # sweep cells + anchor-stage counts
    swept: dict[str, list[str]] = {}
    for r, stage, args in blocks("bfs_sweep"):
        for k in args.get("maps", []):
            swept.setdefault(k, []).append(stage)
    cell_rows, anchor_rows = [], []
    for k in req["scope"]:
        stages = swept.get(k, [])
        for cell in req["cells"][k]:
            w = cell["window"]
            n = sum(1 for s in stages if window_of(matrix, k, s) == w)
            cell_rows.append({"key": f"{k}@{w}", "support": n, "ok": n >= 1})
        want = min(fl["anchor_stages_per_map"], len(req["usable_anchors"][k]))
        n = len(set(stages))
        anchor_rows.append({"key": k, "support": n, "ok": n >= want, "want": want})
    rows["sweep_cells"] = cell_rows
    rows["sweep_anchor_stages"] = anchor_rows

    legs: dict[tuple[str, str], int] = {}
    for r, stage, args in blocks("bfs_sweep"):
        for a, b in args.get("legs", []):
            legs[tuple(sorted((a, b)))] = legs.get(tuple(sorted((a, b))), 0) + 1
    plannable_pairs = {tuple(p) for p in req["pairs"]} - {
        tuple(sorted(u["pair"])) for u in req["unplannable"] if "pair" in u}
    rows["connection_pairs"] = [
        {"key": f"{a}<->{b}", "support": legs.get((a, b), 0),
         "ok": legs.get((a, b), 0) >= fl["connection_rounds"]}
        for a, b in sorted(plannable_pairs)]

    sp_n: dict[int, int] = {}
    for r, stage, args in blocks("encounter_farm"):
        for t in args.get("targets", []):
            for sp, n in (t.get("species_targets") or {}).items():
                sp_n[int(sp)] = sp_n.get(int(sp), 0) + int(n)
    rows["species_sightings"] = [
        {"key": f"sp{sp}", "support": sp_n.get(sp, 0),
         "ok": sp_n.get(sp, 0) >= fl["species_sightings"]}
        for sp in sorted(req["species_map"])]

    town_n = {t: 0 for t in req["towns"]}
    for r, stage, args in blocks("mart_buy"):
        for t in args.get("towns", []):
            if t in town_n:
                town_n[t] += 1
    rows["marts"] = [{"key": t, "support": n, "ok": n >= fl["mart_runs_per_town"]}
                     for t, n in sorted(town_n.items())]
    c_n = {c: 0 for c in req["centers"]}
    for r, stage, args in blocks("pc_access"):
        if args.get("center") in c_n:
            c_n[args["center"]] += 1
    rows["pcs"] = [{"key": c, "support": n, "ok": n >= fl["pc_runs_per_center"]}
                   for c, n in sorted(c_n.items())]
    iu = sum(1 for _ in blocks("item_use"))
    rows["item_use"] = [{"key": "item_use", "support": iu,
                         "ok": iu >= fl["item_use_runs"]}]

    inter: dict[str, int] = {}
    for r, stage, args in blocks("interaction"):
        for k in args.get("maps", []):
            inter[k] = inter.get(k, 0) + 1
    plannable = [k for k in req["scope"] if req["usable_anchors"][k]]
    rows["interaction_maps"] = [
        {"key": k, "support": inter.get(k, 0), "ok": inter.get(k, 0) >= 1}
        for k in plannable]

    # task #46: 18 per-trainer rows. Optional trainers count DISTINCT assigned
    # non-holdout runs (floor trainer_runs_per_trainer); the three spine-guaranteed
    # trainers (grunt/Josh/Roxanne — scripted/milestone battles, never
    # block-schedulable) are fought by every completing run, so their support is the
    # worker count, annotated with the owning milestone.
    tr_floor = fl.get("trainer_runs_per_trainer", 2)
    tr_runs: dict[int, set] = {f: set() for f in TRACKED_TRAINER_FLAGS}
    for r, stage, args in blocks("trainer_engagement"):
        for t in args.get("targets", []):
            f = int(t.get("trainer_flag", -1))
            if f in tr_runs:
                tr_runs[f].add(r["run_id"])
    trows = []
    for f in TRACKED_TRAINER_FLAGS:
        if f in SPINE_TRAINER_FLAGS:
            trows.append({"key": f"tr_flag_{f:#06x}", "support": len(workers),
                          "ok": len(workers) >= tr_floor,
                          "spine": SPINE_TRAINER_FLAGS[f]})
        else:
            trows.append({"key": f"tr_flag_{f:#06x}", "support": len(tr_runs[f]),
                          "ok": len(tr_runs[f]) >= tr_floor})
    rows["trainers"] = trows

    # block placement eligibility (pilot evidence): NO block may be scheduled at a
    # scripted-tail milestone boundary — such a boundary has no free-overworld frame
    # and every block there skips on precondition, every run
    bad_bounds = sorted({
        f"{r['run_id']}@{MILESTONE_ORDER[idx]}"
        for r in plan["runs"] for idx, _bn, _args in r["block_schedule"]
        if MILESTONE_ORDER[idx] in INELIGIBLE_BLOCK_BOUNDARIES})
    rows["block_boundaries"] = [
        {"key": "blocks_at_ineligible_boundaries", "support": len(bad_bounds),
         "ok": not bad_bounds, "offenders": bad_bounds[:10]}]

    g_n = {st: 0 for st in STARTERS}
    for r, stage, args in blocks("grind_evolve"):
        g_n[r["starter"]] += 1
    rows["grind_evolve"] = (
        [{"key": st, "support": n, "ok": n >= min(
            fl["grind_runs_per_starter"],
            sum(1 for r in workers if r["starter"] == st))}
         for st, n in sorted(g_n.items())]
        + [{"key": "evolutions_planned", "support": sum(g_n.values()),
            "ok": sum(g_n.values()) >= 3}])

    n_idle = len({r["run_id"] for r, _, _ in blocks("idle")})
    n_menus = len({r["run_id"] for r, _, _ in blocks("menus")})
    rows["sprinkle"] = [
        {"key": "idle_runs", "support": n_idle,
         "ok": n_idle >= max(1, plan["n_runs"] // fl["idle_run_share"])},
        {"key": "menus_runs", "support": n_menus,
         "ok": n_menus >= max(1, plan["n_runs"] // fl["menus_run_share"])}]

    counts = {st: sum(1 for r in plan["runs"] if r["starter"] == st)
              for st in STARTERS}
    hold = [r for r in plan["runs"] if r["holdout"]]
    rows["fleet"] = [
        {"key": "starter_balance", "support": min(counts.values()),
         "ok": max(counts.values()) - min(counts.values()) <= 1},
        {"key": "holdouts", "support": len(hold),
         "ok": len(hold) == min(3, plan["n_runs"])
         and len({r["starter"] for r in hold}) == min(3, plan["n_runs"])},
    ]

    deficits = [{"axis": ax, **row} for ax, rs in rows.items()
                for row in rs if not row["ok"]]
    warnings = [{"run": r["run_id"], "warnings": r["budget_warnings"]}
                for r in plan["runs"] if r["budget_warnings"]]
    return {"rows": rows, "deficits": deficits, "budget_warnings": warnings,
            "unplannable": req["unplannable"], "green": not deficits}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out", default=None,
                    help="default: <data_root>/processed/w33_expedition_plan.json")
    ap.add_argument("--check", action="store_true",
                    help="audit an existing plan instead of building one")
    ap.add_argument("--plan", default=None, help="plan path for --check")
    args = ap.parse_args()
    root = Path(args.data_root)
    manifest = json.loads((root / "processed/coverage_manifest.json").read_text())
    matrix = json.loads((root / "processed/w33_change_matrix.json").read_text())
    out = Path(args.out) if args.out else root / "processed/w33_expedition_plan.json"

    if args.check:
        plan = json.loads(Path(args.plan or out).read_text())
        rep = check_plan(plan, manifest=manifest, matrix=matrix)
        print("=== PLAN SELF-AUDIT (coverage recomputed from the plan) ===")
        for ax, rs in rep["rows"].items():
            red = sum(not r["ok"] for r in rs)
            print(f"  {ax:20s}: {len(rs):>4} rows, {red:>3} DEFICIT")
        for d in rep["deficits"]:
            print(f"  DEFICIT {d['axis']}: {d['key']} support={d['support']}")
        if rep["budget_warnings"]:
            print(f"  budget warnings on {len(rep['budget_warnings'])} runs "
                  f"(advisory, coverage wins)")
        print("PLAN GREEN" if rep["green"] else "PLAN RED")
        raise SystemExit(0 if rep["green"] else 1)

    plan = build_plan(n_runs=args.n, seed=args.seed, data_root=args.data_root,
                      manifest=manifest, matrix=matrix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=1, sort_keys=True))
    rep = check_plan(plan, manifest=manifest, matrix=matrix)
    n_blocks = sum(len(r["block_schedule"]) for r in plan["runs"])
    print(f"plan: {args.n} runs ({sum(r['holdout'] for r in plan['runs'])} holdout), "
          f"{n_blocks} scheduled blocks -> {out}")
    print("self-audit:", "GREEN" if rep["green"] else
          f"RED ({len(rep['deficits'])} deficits — run --check for rows)")


if __name__ == "__main__":
    main()
