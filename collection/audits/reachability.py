"""Story-stage x reachable-tile probe: the coverage denominator W33 was missing.

Both prior corpora plateaued near 55-58% of STATIC walkable tiles (v1: 4,998 of 8,538
across 32.9M frames; v2 fleet: 3,090 of the same set, a near-subset adding 3 tiles),
because the static denominator is a fiction: reachability is stage-conditioned. The
owner's three gates, all confirmed in the ROM's own event data:

  Route 101   LittlerootTown coord (10,1)/(11,1)  VAR_LITTLEROOT_TOWN_STATE gates the
              north exit until the Birch interaction (NeedPokemon / GoSaveBirch scripts)
  Route 102   OldaleTown coord (0,10)             VAR_OLDALE_TOWN_STATE=0 -> BlockedPath
  Route 104   PetalburgCity coord x=8, y=10..13   VAR_PETALBURG_CITY_STATE=0 ->
              ShowGymToPlayer (the Wally scene seals the west until the Dad meeting)

Model (static, no emulator stepping): a tile is blocked at stage S when
  * an object_event stands on it whose hide-flag is CLEAR in S's SaveBlock1 flags
    (item balls included -- they block movement), or
  * a coord_event on it has var == var_value in S's SaveBlock1 vars (conservative:
    an armed trigger is a barrier -- exactly right for the three gates above).
Reachable(S) = multi-map BFS from the player's position over the collision table,
crossing map connections and warp pairs, on the 44 in-scope episode maps.

Acceptance (owner-set): the probe must reproduce all three gates from savestate
flags/vars alone -- it is not told about them.

Inputs: expert milestone savestates (<POL>/<EVENT>/<EVENT>_completed.state),
references/pokeemerald map jsons + constants, data/state_catalog/collision_table.npz.
"""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import numpy as np

PE = Path("/root/code/proj-minhyuk-2026/references/pokeemerald")
WM = Path("/root/code/proj-minhyuk-2026/pokemon-worldmodel")
SAVE_BLOCK1_PTR = 0x03005D8C
SB1_FLAGS, SB1_VARS = 0x1270, 0x139C


# ------------------------------------------------------------------ ROM-side tables
def _defines(path: Path, prefix: str) -> dict[str, int]:
    out = {}
    for m in re.finditer(rf"#define\s+({prefix}\w+)\s+(0x[0-9A-Fa-f]+|\d+)", path.read_text()):
        out[m.group(1)] = int(m.group(2), 0)
    return out


def _resolve(names: dict[str, int], raw: str) -> int | None:
    """map.json flag/var fields are names, '0', or occasionally arithmetic-free ids."""
    if raw in names:
        return names[raw]
    try:
        return int(raw, 0) or None
    except ValueError:
        return None


class World:
    """The 44 in-scope maps: grids, per-map events, connection/warp graph."""

    def __init__(self):
        self.flags = _defines(PE / "include/constants/flags.h", "FLAG_")
        self.vars = _defines(PE / "include/constants/vars.h", "VAR_")
        ct = dict(np.load(WM / "data/state_catalog/collision_table.npz"))
        self.grids = {k: v for k, v in ct.items() if not k.endswith("_elev")}
        # ELEVATION IS PART OF WALKABILITY. Gen-3 refuses a step onto a different
        # elevation (0 and 15 are wildcards), and elevation 1 is WATER -- collision
        # marks surf tiles walkable, so a foot-only denominator must drop them.
        # Ground truth: the Petalburg->Route104 "leak" was a single elev 3->1 step at
        # (7,20)->(6,20) the emulator refuses; with this rule the probe refuses it too.
        self.elev = {k[:-5]: v for k, v in ct.items() if k.endswith("_elev")}

        groups = json.loads((PE / "data/maps/map_groups.json").read_text())
        order = groups["group_order"]
        self.key_of_folder, folder_of_key = {}, {}
        for g, gname in enumerate(order):
            for n, folder in enumerate(groups[gname]):
                self.key_of_folder[folder] = f"g{g}_n{n}"
                folder_of_key[f"g{g}_n{n}"] = folder

        self.maps: dict[str, dict] = {}
        self.overrides: dict[str, list] = {}   # key -> [(cond, localid, x, y)]
        self.parse_warnings: list[str] = []
        id_to_key = {}
        for key in self.grids:
            folder = folder_of_key.get(key)
            if folder is None:
                continue
            mj = PE / "data/maps" / folder / "map.json"
            if not mj.exists():
                continue
            d = json.loads(mj.read_text())
            d["_folder"] = folder
            id_to_key[d["id"]] = key
            self.maps[key] = d
            self.overrides[key] = self._transition_overrides(mj.parent / "scripts.inc")
        self.id_to_key = id_to_key
        # second pass: resolve connection/warp destinations (ids of OUT-of-scope maps
        # simply produce no edge -- the episode boundary)
        for key, d in self.maps.items():
            d["_conn"] = [(c["direction"], int(c["offset"]), id_to_key.get(c["map"]))
                          for c in (d.get("connections") or [])]
            _ws = []
            for w in d.get("warp_events") or []:
                try:                                     # WARP_ID_DYNAMIC (e.g. the
                    wid = int(str(w["dest_warp_id"]), 0)  # Secret Base door) has no
                except ValueError:                        # static destination: keep the
                    wid = -1                              # tile enterable, no edge
                _ws.append((int(w["x"]), int(w["y"]), id_to_key.get(w["dest_map"]), wid))
            d["_warps"] = _ws

    def _script_labels(self, inc: Path) -> dict[str, list[str]]:
        labels: dict[str, list[str]] = {}
        cur = None
        for line in inc.read_text().splitlines():
            m = re.match(r"^(\w+):+", line)
            if m:
                cur = m.group(1)
                labels[cur] = []
            elif cur is not None:
                labels[cur].append(line.strip())
        return labels

    def _common_labels(self) -> dict:
        """Labels from the shared script files (Common_Movement_*, Common_EventScript_*):
        Scott's classifier failure was `Common_Movement_WalkInPlaceFasterRight` -- defined
        outside the map file, so it read as unresolved -> conservative BLOCKING, when it
        is a walk-IN-PLACE: the player turns to face Scott and never leaves the tile."""
        if getattr(self, "_common", None) is None:
            self._common = {}
            for inc in (PE / "data/scripts").glob("*.inc"):
                self._common.update(self._script_labels(inc))
        return self._common

    def _trigger_blocks(self, labels: dict, script: str, depth: int = 0) -> bool:
        """Does this coord-event script BLOCK, or is it a pass-through scene?

        Owner's observation (Scott at Petalburg, S4): "some npc comes and does some
        conversation, and he's gone. He doesn't block going to route 104." The armed-
        trigger-equals-wall rule over-sealed exactly these. The discriminator is in the
        scripts themselves: a turn-back gate MUST displace the player --
            BlockedPath:  applymovement LOCALID_PLAYER, ..._PlayerStepBack   (walk_*)
            Scott0..2:    addobject/setobjectxy SCOTT only, player never moved
        So: BLOCKING iff the script (followed through goto/call, same file) applies a
        movement containing walk/jump steps to the player, or warps. Unresolvable
        scripts stay BLOCKING (conservative), listed in parse_warnings.
        """
        if depth > 4:
            return True
        body = labels.get(script) or self._common_labels().get(script)
        if body is None:
            self.parse_warnings.append(f"unresolved trigger script {script}")
            return True
        for ln in body:
            if re.match(r"warp", ln):
                return True
            m = re.match(r"applymovement\s+(LOCALID_PLAYER|OBJ_EVENT_ID_PLAYER),\s*(\w+)", ln)
            if m:
                mv = labels.get(m.group(2)) or self._common_labels().get(m.group(2))
                if mv is None:
                    return True
                # displacing steps only: walk_in_place / jump_in_place turn the player
                # without moving them (Scott), face_* and delays never move anyone
                if any(x.startswith(("walk_", "jump_", "slide_", "player_run"))
                       and "in_place" not in x for x in mv):
                    return True
            # follow EVERY branch, conditional or not: the Petalburg gym boy hides his
            # player-displacement inside `call_if_eq VAR_0x8008, N, LeadPlayerToGymN` --
            # plain goto/call recursion classified the whole scene pass-through and the
            # world flooded to Rustboro. If ANY branch displaces the player, it blocks.
            m = re.match(r"(?:goto|call)(?:_if_\w+)?\s+(?:[\w.]+,\s*)*(\w+)\s*$", ln)
            if m and m.group(1) != script and self._trigger_blocks(labels, m.group(1), depth + 1):
                return True
        return False

    def _transition_overrides(self, inc: Path) -> list:
        """ON_TRANSITION `setobjectxyperm` moves, with their flag/var conditions.

        The map.json object positions are only where NPCs stand AFTER the story moves
        them: Oldale's footprints man is drawn at his template tile mid-town, but on
        every map load while FLAG_ADVENTURE_STARTED is unset the transition script
        teleports him to (1,11) -- the west-exit tile the emulator walk was refused on,
        and exactly where the owner remembered him standing. 49 maps use this pattern
        (Littleroot 5x, Rustboro 8x), so it is parsed, not special-cased.
        """
        if not inc.exists():
            return []
        labels = self._script_labels(inc)
        self._labels_cache = getattr(self, "_labels_cache", {})
        self._labels_cache[inc.parent.name] = labels
        out = []

        def collect(label: str, cond):
            for ln in labels.get(label, ()):
                m = re.match(r"setobjectxyperm\s+(\w+),\s*(\d+),\s*(\d+)", ln)
                if m:
                    out.append((cond, m.group(1), int(m.group(2)), int(m.group(3))))

        for name, body in labels.items():
            if not name.endswith("_OnTransition"):
                continue
            for ln in body:
                m = re.match(r"call_if_(set|unset)\s+(FLAG_\w+),\s*(\w+)", ln)
                if m:
                    collect(m.group(3), ("flag", m.group(2), m.group(1) == "set"))
                    continue
                m = re.match(r"call_if_eq\s+(VAR_\w+),\s*(\d+),\s*(\w+)", ln)
                if m:
                    collect(m.group(3), ("var_eq", m.group(1), int(m.group(2))))
                    continue
                m = re.match(r"call\s+(\w+)$", ln)
                if m:
                    collect(m.group(1), ("always",))
                    continue
                if "call_if" in ln:
                    self.parse_warnings.append(f"{inc.parent.name}: unparsed {ln}")
            collect(name, ("always",))
        return out

    def blocked(self, key: str, flag_get, var_get) -> set[tuple[int, int]]:
        d = self.maps.get(key) or {}
        moved: dict[str, tuple[int, int]] = {}
        for cond, lid, x, y in self.overrides.get(key, ()):
            live = (cond[0] == "always"
                    or (cond[0] == "flag" and flag_get(self.flags.get(cond[1], -1) or 0) == cond[2])
                    or (cond[0] == "var_eq" and var_get(self.vars.get(cond[1], 0)) == cond[2]))
            if live:
                moved[lid] = (x, y)
        out: set[tuple[int, int]] = set()
        for o in d.get("object_events") or []:
            fl = _resolve(self.flags, str(o.get("flag", "0")))
            if fl is None or not flag_get(fl):          # flag CLEAR -> object visible
                pos = moved.get(str(o.get("local_id", "")), (int(o["x"]), int(o["y"])))
                out.add(pos)
        labels = getattr(self, "_labels_cache", {}).get(
            Path(d.get("_folder", "")).name if d.get("_folder") else "", None)
        for c in d.get("coord_events") or []:
            v = _resolve(self.vars, str(c.get("var", "0")))
            if v is not None and var_get(v) == int(str(c.get("var_value", "0")), 0):
                script = str(c.get("script", ""))
                if labels is not None and not self._trigger_blocks(labels, script):
                    continue                            # pass-through scene: Scott etc.
                out.add((int(c["x"]), int(c["y"])))     # armed turn-back = barrier
        return out


# ------------------------------------------------------------------ savestate side
def state_readers(st):
    """(flag_get, var_get) closures over a loaded GBAState."""
    sb1 = st.u32(SAVE_BLOCK1_PTR)

    def flag_get(n: int) -> bool:
        return bool(st.u8(sb1 + SB1_FLAGS + n // 8) & (1 << (n % 8)))

    def var_get(v: int) -> int:
        off = sb1 + SB1_VARS + 2 * (v - 0x4000)
        return st.u8(off) | (st.u8(off + 1) << 8)

    return flag_get, var_get


# ------------------------------------------------------------------ the BFS
def reachable(world: World, start_key: str, start_xy, flag_get, var_get):
    """{map_key: set[(x,y)]} reachable from start under stage-S blockers."""
    blocked = {k: world.blocked(k, flag_get, var_get) for k in world.maps}
    grids = world.grids
    warp_at = {}                                        # (key,x,y) -> (dkey,dx,dy)
    for key, d in world.maps.items():
        for x, y, dkey, dwid in d["_warps"]:
            if dkey is None:
                continue
            dw = world.maps[dkey]["_warps"]
            if dwid >= 0 and 0 <= dwid < len(dw):
                warp_at[(key, x, y)] = (dkey, dw[dwid][0], dw[dwid][1])
    warp_tiles = {(k, x, y) for (k, x, y) in warp_at}

    def ok(key, x, y):
        g = grids.get(key)
        if g is None or not (0 <= y < g.shape[0] and 0 <= x < g.shape[1]):
            return False
        if (x, y) in blocked.get(key, ()):
            return False
        e = world.elev.get(key)
        if e is not None and e[y, x] == 1:                 # water: not foot-reachable
            return (key, x, y) in warp_tiles
        return bool(g[y, x]) or (key, x, y) in warp_tiles   # door mats: enterable

    def elev_ok(key, x, y, nx, ny):
        e = world.elev.get(key)
        if e is None:
            return True
        a, b = int(e[y, x]), int(e[ny, nx])
        return a == b or a in (0, 15) or b in (0, 15)

    seen: dict[str, set] = {}
    q = deque()

    def push(key, x, y):
        s = seen.setdefault(key, set())
        if (x, y) not in s and ok(key, x, y):
            s.add((x, y))
            q.append((key, x, y))

    push(start_key, *start_xy)
    while q:
        key, x, y = q.popleft()
        w = warp_at.get((key, x, y))
        if w:
            push(*w)
        g = grids[key]
        h, wd = g.shape
        for dx, dy, side in ((1, 0, "right"), (-1, 0, "left"), (0, 1, "down"), (0, -1, "up")):
            nx, ny = x + dx, y + dy
            if 0 <= nx < wd and 0 <= ny < h:
                if elev_ok(key, x, y, nx, ny):
                    push(key, nx, ny)
                continue
            for cdir, off, dkey in world.maps[key]["_conn"]:   # cross the seam
                if cdir != side or dkey is None or dkey not in grids:
                    continue
                dg = grids[dkey]
                if side in ("left", "right"):
                    tx = dg.shape[1] - 1 if side == "left" else 0
                    push(dkey, tx, y - off)
                else:
                    ty = dg.shape[0] - 1 if side == "up" else 0
                    push(dkey, x - off, ty)
    return seen


# ------------------------------------------------------------------ driver
def probe_stage(world: World, runner, state_path: Path):
    import collection.navigator as nav
    from collection.extractors.ram import GBAState
    runner.env.load_state(str(state_path))
    nav._hold(runner, [], 45, "probe")
    t, x, y = nav._state(runner)
    st = GBAState(env=runner.env)
    flag_get, var_get = state_readers(st)
    key = f"g{t.map_group}_n{t.map_num}"
    return key, (x, y), reachable(world, key, (x, y), flag_get, var_get)


# ------------------------------------------------------------------ matrix freeze CLI
STAGE_REPS = [("S0_intro", "INTRO_CUTSCENE_COMPLETE"), ("S1_prestarter", "LEAVE_HOUSE"),
              ("S2_prepokedex", "OLDALE_TOWN"), ("S3_postpokedex", "PETALBURG_CITY"),
              ("S4_open", "GYM_CUTSCENE_OUTSIDE"), ("S4b_postwoods", "PETALBURG_WOODS"),
              ("S4c_rustboro", "RUSTBORO_CITY")]

def freeze_matrix(pol_dir: str, out_path: str) -> dict:
    """Emit reachable_matrix.json: per-stage reachable tiles + trainer cones + blockers.

    S4/S4b/S4c share one AREA world (post-gym-scene everything opens; verified vs owner
    knowledge 08-25) but differ in NPC/object state -- all three are kept because the
    ENTITY axes are stage-conditioned even where tiles are not.
    """
    import collection.navigator as nav
    from collection.direct_runner import DirectEmulatorRunner
    from collection.extractors.ram import GBAState
    world = World()
    r = DirectEmulatorRunner(rom_path="Emerald-GBAdvance/rom.gba", savestate_every=0)
    r.initialize()
    out = {"stages": {}, "trainers": {}, "warnings": world.parse_warnings}
    for sname, ev in STAGE_REPS:
        p = Path(pol_dir) / ev / f"{ev}_completed.state"
        r.env.load_state(str(p)); nav._hold(r, [], 40, "probe")
        t, x, y = nav._state(r)
        st = GBAState(env=r.env); fg, vg = state_readers(st)
        seen = reachable(world, f"g{t.map_group}_n{t.map_num}", (x, y), fg, vg)
        blocked = {k: sorted(world.blocked(k, fg, vg)) for k in world.maps}
        out["stages"][sname] = {
            "rep_state": ev,
            "total": sum(len(s) for s in seen.values()),
            "tiles": {k: sorted(v) for k, v in seen.items() if v},
            "blocked": {k: v for k, v in blocked.items() if v},
        }
        # trainer engageability at this stage (objects with trainer_type + sight)
        for key, d in world.maps.items():
            reach = seen.get(key, set())
            for o in d.get("object_events") or []:
                if o.get("trainer_type", "TRAINER_TYPE_NONE") == "TRAINER_TYPE_NONE":
                    continue
                name = str(o.get("script", "")).split("EventScript_")[-1]
                tid = f"{d['_folder']}:{name}"
                sight = int(str(o.get("trainer_sight_or_berry_tree_id", "0")), 0)
                ox, oy = int(o["x"]), int(o["y"])
                near = min((abs(ox-a)+abs(oy-b) for a, b in reach), default=999)
                ent = out["trainers"].setdefault(tid, {
                    "map": d["_folder"], "x": ox, "y": oy, "sight": sight,
                    "facing": o.get("movement_type", ""), "engageable_at": []})
                if near <= max(sight, 1):
                    ent["engageable_at"].append(sname)
    Path(out_path).write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    import sys
    pol = sys.argv[1] if len(sys.argv) > 1 else \
        "/root/code/proj-minhyuk-2026/pokeagent-solution/expert_policies_by_llm"
    dst = sys.argv[2] if len(sys.argv) > 2 else \
        str(WM / "data/processed/reachable_matrix.json")
    m = freeze_matrix(pol, dst)
    for s, d in m["stages"].items():
        print(f"  {s:16s} {d['total']:6d} tiles / {len(d['tiles'])} maps")
    eng = sum(1 for t in m["trainers"].values() if t["engageable_at"])
    print(f"  trainers: {len(m['trainers'])} known, {eng} engageable somewhere -> {dst}")
