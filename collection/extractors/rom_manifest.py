"""Phase A of the data-closure plan: the ROM-derived COVERAGE MANIFEST (PLAN.md §8.1).

Enumerates the closed domain of everything the game can show — so data completeness is DEFINED
(manifest rows × support floors), not discovered through failures. Walks vanilla US Emerald ROM:

  maps         gMapGroups (0x08486578, validated by tilesets.py) → headers → layouts
  events       header+0x04 → MapEvents: object events (NPCs, trainer flags), warps, coord
               events (script triggers), bg events (signs/hidden items)
  connections  header+0x0C → map edge connections (the non-door transitions)
  wild tables  gWildMonHeaders found by STRUCTURE SCAN (20-B entries: (group,num,pad,4 ptrs)
               run terminated by 0xFF) — no symbol address is trusted untested (project rule)
  trainers     gTrainers found by structure scan (0x28-B entries: class/name/partySize/party)

Every section is cross-VALIDATED against the recorded corpus before the manifest is trusted:
corpus NPC graphics ids ⊆ enumerated object events; corpus enemy species ⊆ enumerated wild +
trainer species of in-scope maps; corpus map transitions explained by enumerated warps/connections.

Output: data/processed/coverage_manifest.json — the input to the coverage-accounting audit
(Phase B) and the closure collectors (Phase C).

Usage:
  .venv/bin/python -m collection.extractors.rom_manifest --data_root ../pokemon-worldmodel/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collection.extractors.text import decode

G_MAP_GROUPS = 0x08486578
ROM_BASE = 0x08000000
MAX_GROUPS, MAX_MAPS = 34, 80          # NUM_MAP_GROUPS = 34 (corpus groups all < 34; beyond the
                                       # array, adjacent ROM data masquerades as valid pointers)
N_SPECIES = 412                      # Hoenn internal ids 1..411 (corpus anchors: 283 Mudkip,
                                     # 288 Zigzagoon, 286 Poochyena — all consistent)
# the demo/menu screens are a finite, hand-enumerated tree (Phase-C menu crawler walks it)
MENU_TREE = ["start_menu", "bag_items", "bag_pokeballs", "bag_keyitems", "party", "summary_info",
             "summary_skills", "summary_moves", "trainer_card", "save_dialog", "options",
             "pc_boxes", "mart_buy", "mart_sell", "yesno_prompt", "naming_screen"]


class Rom:
    def __init__(self, data: bytes):
        self.b = data

    def ok(self, ptr: int) -> bool:
        return ROM_BASE <= ptr < ROM_BASE + len(self.b)

    def u8(self, a): return self.b[a - ROM_BASE]
    def s16(self, a):
        v = int.from_bytes(self.b[a - ROM_BASE:a - ROM_BASE + 2], "little")
        return v - 0x10000 if v >= 0x8000 else v
    def u16(self, a): return int.from_bytes(self.b[a - ROM_BASE:a - ROM_BASE + 2], "little")
    def u32(self, a): return int.from_bytes(self.b[a - ROM_BASE:a - ROM_BASE + 4], "little")


# ---- maps / events / warps / connections ------------------------------------------------------

def walk_maps(r: Rom) -> dict[str, dict]:
    """Every map header reachable from gMapGroups, with its events and connections."""
    maps: dict[str, dict] = {}
    # group arrays are PACKED back-to-back in ROM: group g's true map count is the gap to the
    # next group's array (walking past it would read the next group's headers as phantom extras)
    gps = []
    for g in range(MAX_GROUPS):
        gp = r.u32(G_MAP_GROUPS + g * 4)
        if not r.ok(gp):
            break
        gps.append(gp)
    bounds = sorted(gps)
    counts = {}
    for i, gp in enumerate(bounds):
        counts[gp] = min((bounds[i + 1] - gp) // 4, MAX_MAPS) if i + 1 < len(bounds) else MAX_MAPS
    for g, gp in enumerate(gps):
        for n in range(counts[gp]):
            hp = r.u32(gp + n * 4)
            if not r.ok(hp):
                break
            lay = r.u32(hp)
            if not r.ok(lay):
                break
            w, h = r.u32(lay), r.u32(lay + 4)
            if not (1 <= w <= 300 and 1 <= h <= 300):
                break
            m: dict = {"width": w, "height": h, "weather": r.u8(hp + 0x16),
                       "map_type": r.u8(hp + 0x17), "battle_type": r.u8(hp + 0x1B),
                       "objects": [], "warps": [], "signs": [], "coord_events": [],
                       "connections": []}
            ev = r.u32(hp + 0x04)
            if r.ok(ev):
                n_obj, n_warp, n_coord, n_bg = (r.u8(ev), r.u8(ev + 1), r.u8(ev + 2), r.u8(ev + 3))
                op, wp, cp, bp = (r.u32(ev + 4), r.u32(ev + 8), r.u32(ev + 12), r.u32(ev + 16))
                if r.ok(op):
                    for i in range(n_obj):                       # ObjectEventTemplate, 24 B
                        o = op + i * 24
                        obj = {"local_id": r.u8(o), "gfx": r.u8(o + 1),
                               "x": r.s16(o + 4), "y": r.s16(o + 6),
                               "movement": r.u8(o + 9), "trainer_type": r.u16(o + 12),
                               "has_script": int(r.ok(r.u32(o + 16)))}
                        sp_ = r.u32(o + 16)
                        if obj["trainer_type"] and r.ok(sp_):    # parse `trainerbattle` (0x5C):
                            raw = r.b[sp_ - ROM_BASE:sp_ - ROM_BASE + 96]
                            for j in range(len(raw) - 4):        # opcode, type u8, trainer u16
                                if raw[j] == 0x5C and raw[j + 1] <= 12:
                                    tid = raw[j + 2] | (raw[j + 3] << 8)
                                    if 0 < tid < 855:
                                        obj["trainer_id"] = tid
                                        break
                        m["objects"].append(obj)
                if r.ok(wp):
                    for i in range(n_warp):                      # WarpEvent, 8 B
                        o = wp + i * 8
                        m["warps"].append({
                            "x": r.s16(o), "y": r.s16(o + 2), "warp_id": r.u8(o + 5),
                            "dst_map": f"{r.u8(o + 7)},{r.u8(o + 6)}"})
                if r.ok(cp):
                    for i in range(n_coord):                     # CoordEvent, 16 B (script triggers)
                        o = cp + i * 16
                        m["coord_events"].append({"x": r.s16(o), "y": r.s16(o + 2)})
                if r.ok(bp):
                    for i in range(n_bg):                        # BgEvent, 12 B (signs etc.)
                        o = bp + i * 12
                        m["signs"].append({"x": r.s16(o), "y": r.s16(o + 2), "kind": r.u8(o + 5)})
            cn = r.u32(hp + 0x0C)
            if r.ok(cn):
                cnt, cl = r.u32(cn), r.u32(cn + 4)
                if 0 < cnt <= 8 and r.ok(cl):
                    for i in range(cnt):                         # MapConnection, 12 B
                        o = cl + i * 12
                        m["connections"].append({                # MapConnection: group +8, num +9
                            "direction": r.u8(o), "offset": r.u32(o + 4),   # (order A/B-tested
                            "dst_map": f"{r.u8(o + 8)},{r.u8(o + 9)}"})     # against the corpus)
            maps[f"{g},{n}"] = m
    return maps


# ---- wild encounter tables (structure scan) ---------------------------------------------------

def _wild_entry_ok(r: Rom, a: int, known: set[str]) -> bool:
    g, n = r.u8(a), r.u8(a + 1)
    if g == 0xFF and n == 0xFF:
        return False
    if g > 40 or n > 70 or r.u16(a + 2) != 0:
        return False
    ptrs = [r.u32(a + 4 + 4 * k) for k in range(4)]
    return all(p == 0 or r.ok(p) for p in ptrs) and any(p for p in ptrs)


def scan_wild_headers(r: Rom, known_maps: set[str]) -> tuple[int, dict[str, dict]]:
    """Find gWildMonHeaders by scanning for its 20-byte entry structure; parse the tables.
    Probes every 4-byte offset for a RUN START (no skip-ahead: a rejected run must not be allowed
    to step over the real table with the wrong 20-byte phase — the bug in the first version)."""
    best, best_len = 0, 0
    end = ROM_BASE + len(r.b) - 24
    a = ROM_BASE
    while a < end:
        if _wild_entry_ok(r, a, known_maps) and \
                (a - 20 < ROM_BASE or not _wild_entry_ok(r, a - 20, known_maps)):
            run = 0
            while _wild_entry_ok(r, a + run * 20, known_maps):
                run += 1
            # no terminator-sentinel requirement: this ROM's table simply ends (next entry fails
            # the structural test). Longest structurally-valid run containing scope maps wins.
            ids = {f"{r.u8(a + i * 20)},{r.u8(a + i * 20 + 1)}" for i in range(run)}
            if run > best_len and len(ids & known_maps) >= 3:
                best, best_len = a, run
        a += 4

    def mons(info_ptr: int, count: int) -> dict | None:
        if not info_ptr:
            return None
        rate, mp = r.u8(info_ptr), r.u32(info_ptr + 4)
        if not r.ok(mp):
            return None
        out = []
        for i in range(count):
            mn, mx, sp = r.u8(mp + i * 4), r.u8(mp + i * 4 + 1), r.u16(mp + i * 4 + 2)
            if not (1 <= sp < N_SPECIES and 1 <= mn <= mx <= 100):
                return None
            out.append([sp, mn, mx])
        return {"rate": rate, "mons": out}

    tables: dict[str, dict] = {}
    for i in range(best_len):
        a = best + i * 20
        key = f"{r.u8(a)},{r.u8(a + 1)}"
        t = {}
        for name, count, off in (("land", 12, 4), ("water", 5, 8), ("rock", 5, 12), ("fish", 10, 16)):
            m = mons(r.u32(a + off), count)
            if m:
                t[name] = m
        if t:
            tables[key] = t
    return best, tables


# ---- trainers (structure scan) -----------------------------------------------------------------

def _trainer_ok(r: Rom, a: int) -> bool:
    cls, pic = r.u8(a + 1), r.u8(a + 3)
    size, party = r.u32(a + 0x20), r.u32(a + 0x24)
    if not (cls < 70 and pic < 100 and 0 < size <= 6 and r.ok(party)):
        return False
    name = r.b[a + 4 - ROM_BASE:a + 16 - ROM_BASE]
    return 0xFF in name                                           # gen-3 terminated name

def scan_trainers(r: Rom) -> tuple[int, list[dict]]:
    """Find gTrainers by scanning for runs of 0x28-byte Trainer entries (entry 0 is a dummy)."""
    best, best_len = 0, 0
    a, end = ROM_BASE, ROM_BASE + len(r.b) - 0x28
    while a < end:
        if _trainer_ok(r, a):
            run = 0
            while _trainer_ok(r, a + run * 0x28):
                run += 1
            if run > best_len:
                best, best_len = a, run
            a += max(run, 1) * 0x28
        else:
            a += 4
    trainers = []
    for i in range(best_len):
        a = best + i * 0x28
        flags, size, party = r.u8(a), r.u32(a + 0x20), r.u32(a + 0x24)
        stride = 8 if not (flags & 1) else 16                     # custom-moves variants are wider
        mons = []
        for k in range(size):
            o = party + k * stride
            lvl, sp = r.u8(o + 2), r.u16(o + 4)
            if not (1 <= sp < N_SPECIES and 1 <= lvl <= 100):
                mons = []; break
            mons.append([sp, lvl])
        if mons:
            trainers.append({"id": i, "class": r.u8(a + 1),
                             "name": decode(r.b[a + 4 - ROM_BASE:a + 16 - ROM_BASE]),
                             "party": mons})
    return best, trainers


# ---- corpus cross-validation -------------------------------------------------------------------

def corpus_observations(cond_dir: Path, manifest_path: Path) -> dict:
    """What the recorded corpus actually contains: NPC gfx ids, enemy species, map transitions.
    Transitions are broken at recording resets (`clip_starts` from the corpus manifest) — a reset
    teleports the player without a game warp and must not count as one."""
    starts = {r["run_key"]: set(r["clip_starts"])
              for r in json.loads(manifest_path.read_text())["runs"]}
    gfx, species, transitions, maps = set(), set(), set(), set()
    for f in sorted(cond_dir.glob("*.npz")):
        z = np.load(f)
        v = z["ent_gfx"][z["ent_valid"] > 0]
        gfx.update(int(x) for x in np.unique(v) if x > 0)
        for slot in (1, 3):
            sp = z["bat_species"][:, slot][z["bat_valid"][:, slot] > 0]
            species.update(int(x) for x in np.unique(sp) if x > 0)
        mid = z["map_id"]
        resets = starts.get(f.stem, set())
        prev = None
        for i in range(len(mid)):
            if i in resets or (i - 1) in resets:                  # reset: not a game transition
                prev = None                                       # (±1: the RAM map id settles one
                                                                  # frame after the recorded start)
            a, b = int(mid[i, 0]), int(mid[i, 1])
            if (a, b) == (255, 255):
                continue
            key = f"{a},{b}"
            maps.add(key)
            if prev is not None and prev != key:
                transitions.add((prev, key))
            prev = key
    return {"npc_gfx": sorted(gfx), "enemy_species": sorted(species),
            "transitions": sorted(transitions), "maps": sorted(maps)}


def validate(manifest: dict, obs: dict) -> dict:
    scope = set(obs["maps"])
    maps = manifest["maps"]
    enum_gfx = {o["gfx"] for k in scope if k in maps for o in maps[k]["objects"]}
    gfx_missing = [g for g in obs["npc_gfx"] if g not in enum_gfx]

    wild_sp = {sp for k in scope for t in manifest["wild"].get(k, {}).values()
               for sp, *_ in t["mons"]}
    trainer_sp = {sp for t in manifest["trainers"] for sp, _ in t["party"]}
    sp_missing = [s for s in obs["enemy_species"] if s not in wild_sp | trainer_sp]

    def explained(a: str, b: str) -> bool:
        ma = maps.get(a)
        if ma is None:
            return False
        return (any(w["dst_map"] == b for w in ma["warps"])
                or any(c["dst_map"] == b for c in ma["connections"]))
    tr_missing = [t for t in obs["transitions"] if not explained(*t)]

    return {"corpus_gfx_not_enumerated": gfx_missing,
            "corpus_species_not_enumerated": sp_missing,
            "corpus_transitions_unexplained": tr_missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.data_root)
    r = Rom(Path(args.rom).read_bytes())

    maps = walk_maps(r)
    obs = corpus_observations(root / "processed/conditions", root / "processed/corpus_manifest.json")
    wild_base, wild = scan_wild_headers(r, set(obs["maps"]))
    tr_base, trainers = scan_trainers(r)
    scope = obs["maps"]                                          # start→badge-1 = the corpus maps

    starters = [277, 280, 283]                                   # Treecko/Torchic/Mudkip (internal)
    evos = [278, 279, 281, 282, 284, 285]
    catchable = sorted({sp for k in scope for t in wild.get(k, {}).values() for sp, *_ in t["mons"]})

    manifest = {
        "schema_version": 1,
        "rom_tables": {"gMapGroups": hex(G_MAP_GROUPS), "gWildMonHeaders": hex(wild_base),
                       "gTrainers": hex(tr_base)},
        "scope_maps": scope,
        "maps": maps,
        "wild": wild,
        "trainers": trainers,
        "party_space": {"starters": starters, "starter_evos": evos, "catchable_in_scope": catchable},
        "menus": MENU_TREE,
        "corpus_observations": obs,
    }
    manifest["validation"] = validate(manifest, obs)

    out = Path(args.out) if args.out else root / "processed/coverage_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    v = manifest["validation"]
    n_warps = sum(len(maps[k]["warps"]) for k in scope if k in maps)
    n_obj = sum(len(maps[k]["objects"]) for k in scope if k in maps)
    print(f"maps {len(maps)} total / {len(scope)} in scope | warps(in-scope) {n_warps} | "
          f"objects(in-scope) {n_obj} | wild tables {len(wild)} @ {hex(wild_base)} | "
          f"trainers {len(trainers)} @ {hex(tr_base)}")
    print(f"catchable in scope: {len(catchable)} species | corpus enemy species seen: "
          f"{len(obs['enemy_species'])}")
    print(f"VALIDATION — gfx missing: {v['corpus_gfx_not_enumerated']} | species missing: "
          f"{v['corpus_species_not_enumerated']} | unexplained transitions: "
          f"{len(v['corpus_transitions_unexplained'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
