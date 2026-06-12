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
    """ROM-side per-map knowledge: metatile behaviors (via tileset attr tables) + warp events."""

    def __init__(self, rom_path: str = ROM_PATH, tilesets_json: str = TILESETS_JSON,
                 manifest_json: str = MANIFEST_JSON):
        self.rom = Path(rom_path).read_bytes()
        ts = json.loads(Path(tilesets_json).read_text())
        self.ptrs = ts["tileset_ptrs"]
        self.maps = {k: tuple(v) for k, v in ts["maps"].items()}
        self.warps = {}
        self.signs = {}
        mf = Path(manifest_json)
        if mf.exists():
            m = json.loads(mf.read_text())["maps"]
            self.warps = {k: v["warps"] for k, v in m.items()}
            self.signs = {k: v["signs"] for k, v in m.items()}
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


def _bfs_step(walk: np.ndarray, start: tuple[int, int], goals: np.ndarray) -> tuple[int, int] | None:
    """First step (dx, dy) of a shortest path from start to any True cell of `goals` over the
    walkable mask (buffer coords). None if unreachable."""
    h, w = walk.shape
    prev = -np.ones((h, w, 2), np.int32)
    seen = np.zeros((h, w), bool)
    q = deque([start])
    seen[start[1], start[0]] = True
    hit = None
    while q:
        x, y = q.popleft()
        if goals[y, x] and (x, y) != start:
            hit = (x, y)
            break
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and (walk[ny, nx] or goals[ny, nx]):
                seen[ny, nx] = True
                prev[ny, nx] = (x, y)
                q.append((nx, ny))
    if hit is None:
        return None
    x, y = hit
    while tuple(prev[y, x]) != start:
        x, y = prev[y, x]
    return x - start[0], y - start[1]


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
    _clear_dialog(runner, phase)                              # the mash may have opened one
    _hold(runner, [], 40, phase)                              # let scripted movement finish


def goto(runner, mk: MapKnowledge, goal_fn, *, budget: int = 8000, phase: str = "nav",
         stop_fn=None) -> str:
    """Walk toward the nearest goal cell; returns 'arrived' | 'battle' | 'stuck' | 'budget'.
    `goal_fn(t, beh) -> bool mask over buffer cells`. Replans every step; the first refused step
    triggers an unstick (script locks), repeated refusals transiently block the cell (NPCs,
    wrong-side ledges) and route around."""
    blocked: set[tuple[int, int]] = set()
    start_frame = runner.frame_idx
    misses = 0
    while runner.frame_idx - start_frame < budget:
        if _in_battle(runner):
            return "battle"
        t, x, y = _state(runner)
        if t is None:
            _hold(runner, [], 30, phase)
            continue
        if stop_fn is not None and (r := stop_fn(t)):
            return r
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
        for cx, cy in blocked:
            if 0 <= cy < walk.shape[0] and 0 <= cx < walk.shape[1]:
                walk[cy, cx] = False
        step = _bfs_step(walk, (bx, by), goals)
        if step is None:
            return "stuck"
        if _step(runner, DIRS[step]):
            misses = 0
        else:
            misses += 1
            if misses == 1:
                _unstick(runner, phase)                       # script lock? clear before blaming
                continue                                      # the cell
            blocked.add((bx + step[0], by + step[1]))         # NPC / ledge: route around it
            if misses >= 8:
                return "stuck"
            _hold(runner, [], 10, phase)
    return "budget"


def grass_goal(t: Terrain, beh: np.ndarray | None):
    if beh is None:
        return None
    walk = ((t.grid >> 10) & 3) == 0
    return (beh == GRASS) & walk


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
    while runner.frame_idx - start < budget:
        if _in_battle(runner):
            return "battle"
        t, x, y = _state(runner)
        if t is None:
            _hold(runner, [], 30, "nav_grass")
            continue
        g = grass_goal(t, mk.behaviors(t))
        if g is None or not g[y + 7, x + 7]:
            return "left"
        opts = [(dx, dy) for (dx, dy) in DIRS
                if g[y + 7 + dy, x + 7 + dx]] or list(DIRS)
        dx, dy = rng.choice(opts)
        _step(runner, DIRS[(dx, dy)])
    return "budget"


def goto_warp(runner, mk: MapKnowledge, wx: int, wy: int, *, budget: int = 8000) -> str:
    """Walk onto the warp tile at map coords (wx, wy) and trigger it; 'crossed' | goto's failures.
    Mat/stair warps fire the moment you STEP onto them — so the map watch runs INSIDE the walk
    (the first version checked only after arrival and, post-crossing, navigated the NEW map toward
    OLD-map coordinates). Doors that need a push get one in each direction, map-watched."""
    t0, x0, y0 = _state(runner)
    if t0 is None:
        return "stuck"
    src = (t0.map_group, t0.map_num)

    def crossed(t):
        if (t.map_group, t.map_num) != src:
            _hold(runner, [], 90, "nav_warp")                 # settle the fade-in
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
