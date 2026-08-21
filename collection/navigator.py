"""Navigator — deliberate movement over the LIVE map state (data-closure Phase C).

Random wandering was the closure bottleneck: independent 14k-frame walks repeatedly found zero
grass from some spawns, warp pairs need door visits, and some farming maps have no nearby base.
This module navigates ON PURPOSE, using only state we already extract + the ROM:

  walkability   collision bits 10-11 of the live `gBackupMapLayout` words (terrain extractor)
  grass         metatile behavior byte 0x02 from the tileset attribute tables — attributes ptr
                PINNED at Tileset+0x10 by the corpus itself (behavior(battle-start tile) == 0x02
                on 247/266 recorded battle starts; the rest are scripted/trainer starts)
  doors/warps   the coverage manifest's per-map warp events (ROM-enumerated)

Movement is BFS over the walkable mask with per-step REPLANNING: a step that doesn't move
(wandering NPC, ledge from the wrong side) transiently blocks that cell and replans; a battle
interrupts and returns to the caller (battles are usually the point). All reads go through
`GBAState.snapshot` — the same seam as everything else.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from collection.extractors.entities import player_state
from collection.extractors.ram import GBAState
from collection.extractors.terrain import Terrain, terrain

GRASS = 0x02                                # MB_TALL_GRASS (corpus-pinned, see module docstring)
ATTR_PTR_OFF = 0x10                         # attributes ptr inside the Tileset struct (pinned)
DIRS = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
ROM_BASE = 0x08000000

ROM_PATH = "Emerald-GBAdvance/rom.gba"
TILESETS_JSON = "../pokemon-worldmodel/data/processed/conditions/tilesets.json"
MANIFEST_JSON = "../pokemon-worldmodel/data/processed/coverage_manifest.json"


class MapKnowledge:
    """ROM-side per-map knowledge: metatile behaviors (via tileset attr tables) + warp events.

    `rng` (W33 §14.1 persona layer): optional random.Random used by `goto`'s per-step BFS to
    shuffle equal-cost neighbor expansion order. BFS stays level-order, so every returned step
    still lies on A shortest path — the seed only picks WHICH of the equally-short paths a run
    walks (same seed -> same route; different seeds -> measurably different routes; zero
    correctness cost). None keeps the historical fixed expansion order."""

    def __init__(self, rom_path: str = ROM_PATH, tilesets_json: str = TILESETS_JSON,
                 manifest_json: str = MANIFEST_JSON, rng=None):
        self.rng = rng
        self.rom = Path(rom_path).read_bytes()
        ts = json.loads(Path(tilesets_json).read_text())
        self.ptrs = ts["tileset_ptrs"]
        self.maps = {k: tuple(v) for k, v in ts["maps"].items()}
        self.warps = {}
        self.signs = {}
        self.connections = {}
        self.objects = {}
        mf = Path(manifest_json)
        if mf.exists():
            m = json.loads(mf.read_text())["maps"]
            self.warps = {k: v["warps"] for k, v in m.items()}
            self.signs = {k: v["signs"] for k, v in m.items()}
            self.connections = {k: v["connections"] for k, v in m.items()}
            self.objects = {k: v["objects"] for k, v in m.items()}
        self._attr_cache: dict[int, np.ndarray] = {}

    def _attrs(self, tileset_id: int) -> np.ndarray:
        if tileset_id not in self._attr_cache:
            ap = int.from_bytes(self.rom[self.ptrs[tileset_id] + ATTR_PTR_OFF - ROM_BASE:][:4], "little")
            off = ap - ROM_BASE
            self._attr_cache[tileset_id] = np.frombuffer(self.rom[off:off + 512 * 2], "<u2")
        return self._attr_cache[tileset_id]

    def behaviors(self, t: Terrain) -> np.ndarray | None:
        """Behavior byte per buffer cell (None for unknown maps)."""
        key = f"{t.map_group},{t.map_num}"
        if key not in self.maps:
            return None
        prim, sec = self.maps[key]
        mids = t.grid & 0x3FF
        beh = np.where(mids < 512,
                       self._attrs(prim)[np.minimum(mids, 511)] & 0xFF,
                       self._attrs(sec)[np.minimum(mids - 512, 511)] & 0xFF)
        return beh.astype(np.uint8)


def _state(runner) -> tuple[Terrain | None, int, int]:
    st = GBAState.snapshot(runner.env)
    t = terrain(st)
    p = player_state(st)
    return t, p["x"], p["y"]


def _in_battle(runner) -> bool:
    return bool(runner.nav_state().in_battle)


def _hold(runner, buttons, frames, phase="nav"):
    from collection.collect_behaviors import _hold as h
    h(runner, buttons, frames, phase)


def _step(runner, d: str, max_frames: int = 48) -> bool:
    """One held step in direction d; True if the player's tile changed. Polls while holding: a
    fixed-length hold is unreliable (turning first consumes ~8 frames, so 26 frames sometimes
    only turns — and a 'missed' step falsely marks a FREE cell blocked, collapsing the route;
    the first nav test failed exactly this way)."""
    _, x0, y0 = _state(runner)
    for _ in range(max_frames // 4):
        _hold(runner, [d], 4)
        _, x1, y1 = _state(runner)
        if (x1, y1) != (x0, y0):
            _hold(runner, [d], 8)                    # let the tile-walk animation commit
            return True
    return False


# Jump-ledge behaviors: collision-1 tiles passable ONE WAY (entering in the jump direction
# hops the walker 2 cells). The descent from Route 115's entrance plateau to its beach is
# ONLY possible through these.
_JUMP = {0x38: (1, 0), 0x39: (-1, 0), 0x3A: (0, -1), 0x3B: (0, 1)}


def _bfs_step(walk: np.ndarray, start: tuple[int, int], goals: np.ndarray,
              elev: np.ndarray | None = None,
              beh: np.ndarray | None = None,
              rng=None) -> tuple[int, int] | None:
    """First step DIRECTION (dx, dy) of a shortest path from start to any True cell of `goals`
    over the walkable mask (buffer coords). None if unreachable.

    `rng` (W33 §14.1): shuffles the per-node neighbor expansion order. The queue stays FIFO,
    so expansion remains strictly level-order (BFS optimality untouched) — the shuffle only
    tie-breaks which equal-cost parent claims a cell first, i.e. which shortest path wins.

    With `elev` (grid bits 12-15) the search is ELEVATION-AWARE: a cliff-top cell is collision-0
    yet unenterable from below (Route 115 burned 40k frames walking UP into one). pokeemerald's
    rule: a move is blocked when both elevations are concrete (not 0=transition / 15=bridge) and
    differ; stepping onto a tile ADOPTS its elevation — including 0, which is how stairs connect
    two levels (keeping the old concrete value walls off every ramp: instant dead-end on Route
    115). Bridges (15) keep the walker's elevation. So the search state is (x, y, elev).
    With `beh`, jump ledges add one-way 2-cell edges (see _JUMP)."""
    h, w = walk.shape
    if elev is None:
        elev = np.zeros((h, w), np.uint8)                     # all-transition: plain BFS
    e0 = int(elev[start[1], start[0]])
    s3 = (start[0], start[1], e0)
    prev: dict = {s3: None}
    q = deque([s3])
    hit = None
    dirs = list(DIRS)
    while q:
        x, y, e = q.popleft()
        if goals[y, x] and (x, y) != start:
            hit = (x, y, e)
            break
        for dx, dy in (rng.sample(dirs, len(dirs)) if rng is not None else dirs):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if walk[ny, nx] or goals[ny, nx]:
                te = int(elev[ny, nx])
                if te not in (0, 15) and e not in (0, 15) and te != e:
                    continue                                  # concrete elevation mismatch
                n3 = (nx, ny, e if te == 15 else te)
            elif beh is not None and _JUMP.get(int(beh[ny, nx])) == (dx, dy):
                lx, ly = x + 2 * dx, y + 2 * dy               # ledge: hop over to the landing
                if not (0 <= lx < w and 0 <= ly < h) or not walk[ly, lx]:
                    continue
                le = int(elev[ly, lx])
                n3 = (lx, ly, e if le == 15 else le)
            else:
                continue
            if n3 not in prev:
                prev[n3] = (x, y, e)
                q.append(n3)
    if hit is None:
        return None
    while prev[hit] != s3:
        hit = prev[hit]
    dx, dy = hit[0] - start[0], hit[1] - start[1]
    return max(-1, min(1, dx)), max(-1, min(1, dy))           # a first-step jump is 2 cells


def _dialog_open(runner) -> bool:
    """A textbox is on screen (same BG0 bottom-band signal the conditions use)."""
    from collection.extractors.ui import window_mask
    wm = window_mask(GBAState.snapshot(runner.env))
    return bool(wm[14:20].any())


def _clear_dialog(runner, phase: str = "nav", max_cycles: int = 60) -> None:
    """Advance an OPEN dialogue to its end. The cycle is A, A, B — pure A-mash LOOPS on scripts
    that end in a YES/NO prompt (the Center-2F attendant's wireless pitch: A accepts and re-enters
    the script; B declines and actually ends it — established by screenshot after 60 A-presses)."""
    for i in range(max_cycles):
        if not _dialog_open(runner):
            return
        _hold(runner, ["B" if i % 3 == 2 else "A"], 3, phase)
        _hold(runner, [], 14, phase)


def _unstick(runner, phase: str = "nav") -> None:
    """Clear input locks. Two classes, both found by screenshot: an OPEN dialogue (advance with A
    until gone) and boxless script locks (the May-interaction base: scripted NPC movement /
    pending triggers — mash B then A, then give the script time to run its course)."""
    _clear_dialog(runner, phase)
    for b in ("B", "B", "B", "A", "A", "A", "B"):
        _hold(runner, [b], 3, phase)
        _hold(runner, [], 12, phase)
    _hold(runner, [], 25, phase)                              # a box the A-mash reopened (e.g.
    _clear_dialog(runner, phase)                              # re-talking to an adjacent NPC)
    _hold(runner, [], 40, phase)                              # renders ~20f late — wait, THEN
                                                              # clear; let scripts finish


def goto(runner, mk: MapKnowledge, goal_fn, *, budget: int = 8000, phase: str = "nav",
         stop_fn=None, visit_fn=None, avoid_fn=None, miss_fn=None, rng=None) -> str:
    """Walk toward the nearest goal cell; returns 'arrived' | 'battle' | 'stuck' | 'budget'.
    `goal_fn(t, beh) -> bool mask over buffer cells`. Replans every step; the first refused step
    triggers an unstick (script locks), repeated refusals transiently block the cell (NPCs,
    wrong-side ledges) and route around. Blocks EXPIRE (~10 s — a wandering NPC moves on; a
    stale block in a 1-2 cell choke like Route 115's ledge gap otherwise dead-ends the route),
    and a dead-ended BFS clears them and retries: 'stuck' now means stuck on a CLEAN grid.

    W33 expedition hooks (both optional, both per-replan i.e. per settled tile):
      visit_fn(t, x, y)      — observe every tile the walk lands on (bfs_sweep's visited set;
                               a goal_fn that shrinks as visit_fn marks tiles turns one goto
                               call into a full nearest-unvisited tile tour)
      avoid_fn(t, beh)->mask — cells the PLAN may never enter (True = forbidden), subtracted
                               from the walk mask: keeps a sweep inside its map (border strips
                               are the neighbour's tiles — entering one crosses the connection)
                               and off warp mats/stairs that fire on step-on.
      miss_fn(x, y)          — a refused step's TARGET tile (map coords), fired when the miss
                               gets transiently blocked. Needed because _bfs_step may enter
                               GOAL cells regardless of the walk mask, so a goal tile occupied
                               by a parked NPC (Brendan-house mom, measured 2026-08-20: 12
                               straight refusals, ~9.8k frames) can only be routed around by
                               the CALLER dropping it from its goal mask.
      rng                    — (W33 §14.1) equal-cost tie-break rng for the per-step BFS;
                               defaults to the MapKnowledge's persona rng (mk.rng) so every
                               goto through a persona-seeded mk is route-diversified without
                               each call site plumbing it."""
    if rng is None:
        rng = getattr(mk, "rng", None)
    blocked: dict[tuple[int, int], int] = {}                  # cell -> frame of the miss
    refusals: dict[tuple[int, int], int] = {}                 # cell -> times it refused us
    start_frame = runner.frame_idx
    misses = 0
    resets = 0
    while runner.frame_idx - start_frame < budget:
        if _in_battle(runner):
            return "battle"
        t, x, y = _state(runner)
        if t is None:
            _hold(runner, [], 30, phase)
            continue
        if stop_fn is not None and (r := stop_fn(t)):
            return r
        if visit_fn is not None:
            visit_fn(t, x, y)
        beh = mk.behaviors(t)
        goals = goal_fn(t, beh)
        if goals is None:
            return "stuck"
        bx, by = x + 7, y + 7
        if not (0 <= by < goals.shape[0] and 0 <= bx < goals.shape[1]):
            _hold(runner, [], 30, phase)                      # transition window: the layout
            continue                                          # rebased before the player coords
        if goals[by, bx]:
            return "arrived"
        walk = ((t.grid >> 10) & 3) == 0
        if avoid_fn is not None and (av := avoid_fn(t, beh)) is not None:
            walk &= ~av
        # a cell that refused us TWICE is terrain-like, not a passer-by: hold it far
        # longer. The flat 600-frame expiry made the walk forget the immovable Gina &
        # Mia tiles between replans and propose them again forever (measured: 117k
        # frames of UP/LEFT refusals at (27,16), never once re-routing).
        blocked = {c: f for c, f in blocked.items()
                   if runner.frame_idx - f < (6000 if refusals.get(c, 0) >= 2 else 600)}
        for cx, cy in blocked:
            if 0 <= cy < walk.shape[0] and 0 <= cx < walk.shape[1]:
                walk[cy, cx] = False
        step = _bfs_step(walk, (bx, by), goals,
                         elev=((t.grid >> 12) & 0xF).astype(np.uint8), beh=beh, rng=rng)
        if step is None:
            if blocked and resets < 4:                        # dead-ended by our own blocks
                blocked = {c: f for c, f in blocked.items()   # (keep the immovable ones)
                           if refusals.get(c, 0) >= 2}
                resets += 1
                _hold(runner, [], 60, phase)                  # let the blocking NPC wander off
                continue
            # ELEVATION FALLBACK (measured 2026-08-21): the elevation rule is a MODEL,
            # and when it says "unreachable" while a plain walk says reachable, the game
            # is the authority — retry blind and let REAL step refusals blacklist cells.
            # Route 104's north crossing is only reachable this way once the Gina & Mia
            # double parks on the bridge (a one-mon party can never battle them, so they
            # never move): strict BFS dead-ended at (27,16) on every attempt and burned
            # 117k frames, while the spine's pathfinder — which has no elevation model —
            # crosses there in every clean run.
            step = _bfs_step(walk, (bx, by), goals, beh=beh, rng=rng)
            if step is None:
                return "stuck"
        if _step(runner, DIRS[step]):
            misses = 0
        else:
            misses += 1
            _cell = (bx + step[0], by + step[1])
            refusals[_cell] = refusals.get(_cell, 0) + 1
            if misses % 3 == 1:                               # script lock? clear before blaming
                _unstick(runner, phase)                       # the cell (re-fires: an unstick
                continue                                      # can itself reopen a dialog)
            blocked[_cell] = runner.frame_idx                 # NPC / ledge: route around
            if miss_fn is not None:
                miss_fn(bx + step[0] - 7, by + step[1] - 7)
            if misses >= 8:
                return "stuck"
            _hold(runner, [], 10, phase)
    return "budget"


_MIN_PATCH = 6          # tiles; below this a "patch" cannot hold a walk


def grass_goal(t: Terrain, beh: np.ndarray | None, min_patch: int = _MIN_PATCH):
    """Grass tiles worth walking to — patches big enough to PACE inside.

    Nearest-tile targeting fails on sparse routes. Route 116 measured: a run enters
    from Rustboro at (0,12) and the closest grass is a SINGLE tile at (5,12) walled
    in on both sides, so goto_grass reports 'arrived' in 84 frames and pace_grass
    reports 'left' in 16 — step on, step off, no encounter, forever. The real field
    is 30+ tiles further south. Dropping patches smaller than `min_patch` sends the
    walk to grass it can actually move around in."""
    if beh is None:
        return None
    walk = ((t.grid >> 10) & 3) == 0
    g = (beh == GRASS) & walk
    if not g.any():
        return g
    # 4-connected components, iterative flood fill (no scipy dependency)
    seen = np.zeros_like(g)
    keep = np.zeros_like(g)
    ys, xs = np.where(g)
    for sy, sx in zip(ys, xs):
        if seen[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        comp = []
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < g.shape[0] and 0 <= nx < g.shape[1] \
                        and g[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) >= min_patch:
            for cy, cx in comp:
                keep[cy, cx] = True
    return keep if keep.any() else g          # never strand a grind with no goal


WATER = {0x10, 0x11, 0x14, 0x15, 0x16, 0x17}          # pond/sea behavior bytes (bite-validated)


def water_adjacent_goal(t: Terrain, beh: np.ndarray | None):
    """Walkable land cells with a water 4-neighbour (where a rod can be cast)."""
    if beh is None:
        return None
    walk = ((t.grid >> 10) & 3) == 0
    water = np.isin(beh, list(WATER))
    near = np.zeros_like(water)
    near[1:, :] |= water[:-1, :]; near[:-1, :] |= water[1:, :]
    near[:, 1:] |= water[:, :-1]; near[:, :-1] |= water[:, 1:]
    return walk & near & ~water


def goto_grass(runner, mk: MapKnowledge, budget: int = 8000) -> str:
    return goto(runner, mk, grass_goal, budget=budget, phase="nav_grass")


def pace_grass(runner, mk: MapKnowledge, rng, budget: int = 4000) -> str:
    """Walk randomly WITHIN grass cells until a battle starts; 'battle' | 'budget' | 'left'."""
    start = runner.frame_idx
    _last_xy = None
    _still = 0
    while runner.frame_idx - start < budget:
        if _in_battle(runner):
            return "battle"
        t, x, y = _state(runner)
        if t is None:
            _hold(runner, [], 30, "nav_grass")
            continue
        g = grass_goal(t, mk.behaviors(t))
        if g is None:
            return "left"
        if not g[y + 7, x + 7]:
            # Standing NEXT TO the grass, not in it — goto_grass lands here routinely,
            # and bailing out just re-routes to the same spot. Measured on Route 116:
            # the walker parked one tile off a patch and burned lap after lap without
            # a single encounter. Step in; only give up when there is nothing to step
            # into.
            for dx, dy in DIRS:
                if g[y + 7 + dy, x + 7 + dx] and _step(runner, DIRS[(dx, dy)]):
                    break
            else:
                return "left"
        # A pace that stops MOVING triggers nothing: encounters fire on a step, so a
        # walker whose steps are refused burns its whole budget standing still.
        # Measured on Route 116: 10 of 13 laps returned 'budget' after ~15,000 frames
        # with the player parked on (13,15) the entire time, which is what makes a
        # grind rep win zero battles. Give up quickly and let goto_grass re-route to
        # another patch rather than paying the full budget for nothing.
        if (x, y) == _last_xy:
            _still += 1
            if _still >= 12:
                return "left"
        else:
            _still = 0
        _last_xy = (x, y)
        opts = [(dx, dy) for (dx, dy) in DIRS if g[y + 7 + dy, x + 7 + dx]]
        if not opts:
            # No grass neighbour: the OLD code fell back to any direction, which
            # walks straight out of the grass and ends the pace on the next lap.
            # On a sparse route that is the whole failure — Route 116 measured
            # 'arrived' in 84 frames then 'left' in 16, over and over, with no
            # encounter. Hand back to goto_grass, which now targets real patches.
            return "left"
        dx, dy = rng.choice(opts)
        _step(runner, DIRS[(dx, dy)])
    return "budget"


# MapConnection directions (pokeemerald): 1=south 2=north 3=west 4=east
_CONN_EDGE = {1: ("DOWN", lambda t: t.map_height - 1, "y"), 2: ("UP", 0, "y"),
              3: ("LEFT", 0, "x"), 4: ("RIGHT", lambda t: t.map_width - 1, "x")}


def _settle_seam(runner, phase: str) -> None:
    """Post-crossing stabilization: at a connection seam the map ID flips BEFORE the
    coordinate re-base (measured live: map=(0,3) with a still-104-based y=59), so an
    immediate re-route or warp-watch reads a half-updated state — goto_warp captured a
    stale src, its map-watch fired instantly, and _unstick wandered back across the
    seam (exp_001 wave-3 heal trip). Hold until two consecutive reads agree on
    (map, x, y), ~180 frames max."""
    prev = None
    for _ in range(6):
        _hold(runner, [], 30, phase)
        t, x, y = _state(runner)
        cur = (None if t is None else (t.map_group, t.map_num), x, y)
        if t is not None and cur == prev:
            return
        prev = cur


def cross_connection(runner, mk: MapKnowledge, direction: int, *, budget: int = 8000) -> str:
    """Walk off the map edge in `direction` (a manifest connection); 'crossed' | failures."""
    t0, _, _ = _state(runner)
    if t0 is None or direction not in _CONN_EDGE:
        return "stuck"
    src = (t0.map_group, t0.map_num)
    d, edge, axis = _CONN_EDGE[direction]

    def crossed(t):
        if (t.map_group, t.map_num) != src:
            _settle_seam(runner, "nav_conn")
            return "crossed"
        return None

    # A reachable edge tile is NOT always crossable: the beyond-edge buffer can be
    # void at that column (measured on ROUTE_104 north: columns 11/17 dead, 16 live —
    # 32/32 real crossings use 16). Nearest-edge choice therefore deadlocks
    # deterministically. Blacklist failed columns and retry the crossing at the next
    # candidate instead of walking back to the same dead tile forever.
    failed_cols: set = set()

    def goal(t, beh):
        e = edge(t) if callable(edge) else edge
        m = np.zeros(t.grid.shape, bool)
        if axis == "y":
            m[e + 7, 7:7 + t.map_width] = True
        else:
            m[7:7 + t.map_height, e + 7] = True
        m &= ((t.grid >> 10) & 3) == 0
        for fc in failed_cols:
            if axis == "y":
                m[e + 7, fc + 7] = False
            else:
                m[fc + 7, e + 7] = False
        return m

    for _attempt in range(6):
        r = goto(runner, mk, goal, budget=budget, phase="nav_conn", stop_fn=crossed)
        if r != "arrived":
            return r
        for _ in range(4):                                    # step off the edge, map-watched
            _step(runner, d)
            t, _, _ = _state(runner)
            if t is not None and (c := crossed(t)):
                return c
        t2, px, py = _state(runner)                           # step-off refused: dead column —
        if t2 is not None:                                    # blacklist it and re-route
            failed_cols.add(px if axis == "y" else py)
    return "stuck"


def map_route(mk: MapKnowledge, src: str, dst: str) -> list[tuple] | None:
    """BFS over the map graph -> [(hop_kind, hop, on_map_key), ...]; skips link rooms (policy)
    and counter maps (the Center-2F trap)."""
    def hazardous(k):
        return any(w["dst_map"].startswith("25,") for w in mk.warps.get(k, []))
    prev: dict[str, tuple] = {src: None}
    q = deque([src])
    while q:
        k = q.popleft()
        if k == dst:
            hops = []
            while prev[k] is not None:
                pk, hop = prev[k]
                hops.append(hop + (pk,))
                k = pk
            return hops[::-1]
        for w in mk.warps.get(k, []):
            e = w["dst_map"]
            if e not in prev and not e.startswith("25,") and (e == dst or not hazardous(e)):
                prev[e] = (k, ("warp", w))
                q.append(e)
        for c in mk.connections.get(k, []):
            e = c["dst_map"]
            if e not in prev and (e == dst or not hazardous(e)):
                prev[e] = (k, ("conn", c))
                q.append(e)
    return None


def goto_map(runner, mk: MapKnowledge, dst: str, *, hop_budget: int = 9000,
             max_hops: int = 12) -> str:
    """Navigate ACROSS maps to `dst` (warp + connection hops, re-routed per hop)."""
    for _ in range(max_hops):
        t, _, _ = _state(runner)
        if t is None:
            _hold(runner, [], 30, "nav_map")
            continue
        cur = f"{t.map_group},{t.map_num}"
        if cur == dst:
            return "arrived"
        route = map_route(mk, cur, dst)
        if not route:
            return "stuck"
        kind, hop, _on = route[0]
        if kind == "warp":
            r = goto_warp(runner, mk, hop["x"], hop["y"], budget=hop_budget)
        else:
            r = cross_connection(runner, mk, hop["direction"], budget=hop_budget)
        if r == "battle":
            return "battle"
        if r not in ("crossed",):
            _unstick(runner, "nav_map")                       # try re-routing from wherever we are
    return "stuck"


def goto_warp(runner, mk: MapKnowledge, wx: int, wy: int, *, budget: int = 8000) -> str:
    """Walk onto the warp tile at map coords (wx, wy) and trigger it; 'crossed' | goto's failures.
    Mat/stair warps fire the moment you STEP onto them — so the map watch runs INSIDE the walk
    (the first version checked only after arrival and, post-crossing, navigated the NEW map toward
    OLD-map coordinates). Doors that need a push get one in each direction, map-watched."""
    t0, x0, y0 = _state(runner)
    if t0 is None:                                            # transient seam/fade read —
        _hold(runner, [], 45, "nav_warp")                     # settle once before giving up
        t0, x0, y0 = _state(runner)
    if t0 is None:
        return "stuck"
    src = (t0.map_group, t0.map_num)

    def crossed(t):
        if (t.map_group, t.map_num) != src:
            _settle_seam(runner, "nav_warp")                  # settle the fade-in
            return "crossed"
        return None

    def goal(t, beh):
        m = np.zeros(t.grid.shape, bool)
        if 0 <= wy + 7 < m.shape[0] and 0 <= wx + 7 < m.shape[1]:
            m[wy + 7, wx + 7] = True
        return m

    if (x0, y0) == (wx, wy):                                  # already ON it (e.g. we just arrived
        for d in ("DOWN", "UP", "LEFT", "RIGHT"):             # through it): step off, then re-enter
            if _step(runner, d):
                break
    r = goto(runner, mk, goal, budget=budget, phase="nav_warp", stop_fn=crossed)
    if r != "arrived":
        return r
    for d in ("UP", "DOWN", "LEFT", "RIGHT"):                 # door push, map-watched
        t, _, _ = _state(runner)
        if t is not None and (c := crossed(t)):
            return c
        if _dialog_open(runner):                              # arrival can trigger a greeting
            _clear_dialog(runner, "nav_warp")
        _hold(runner, [d], 26, "nav_warp")
        _hold(runner, [], 40, "nav_warp")
    t, _, _ = _state(runner)
    return crossed(t) or "stuck" if t is not None else "stuck"
