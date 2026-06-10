"""Unified coverage collector — one module, two phases.

Replaces collect_explore (graph) + collect_rollout (tour) + collect_world (cross-map). From
a single checkpoint:

  Phase 1 — MAP: a restore-based BFS discovers the COMPLETE cross-map walkable graph (tries
    every direction at every tile, follows warps/doors into every connected map). Fast, uses
    save/restore, records NO video — so the graph is complete and reliable (both directions
    of every walkable edge are known), without bloating the dataset.

  Phase 2 — TOUR: walks that complete graph as clean continuous video, recording every frame
    with the full world-model condition (RGB + action + PPU + semantic, via WorldModelSink).
    Greedy "go to nearest unvisited tile"; one-way ledges -> reset to checkpoint = new clip;
    wild battles -> flee + resume; transient NPC blocks -> mark edge failed and re-plan.

Pipeline is now just:  collect_events -> checkpoints -> collect_coverage.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

from collection.direct_runner import DirectEmulatorRunner
from collection.recorder import ChunkRecorder
from collection.world_model_sink import WorldModelSink

DIRS = ("UP", "DOWN", "LEFT", "RIGHT")


def _flee_battle(runner, max_actions: int = 40) -> bool:
    """Escape a wild battle (RUN = bottom-right, then confirm)."""
    for _ in range(max_actions):
        for act in ("B", "DOWN", "RIGHT", "A"):
            runner.perform_action(act, speed="fast", record_end_state=False)
        if not runner.nav_state().in_battle:
            return True
    return not runner.nav_state().in_battle


def _tile(runner) -> tuple:
    n = runner.nav_state(); return (n.map, n.x, n.y)


def _settle(runner, *, stats=None, max_frames: int = 120) -> tuple:
    """Release all input and wait for the tile to stop changing. With no button held the avatar
    cannot start a new step, so this lets the current move (or a door/connection fade) finish
    WITHOUT chaining into a second tile, and filters mid-step transient coords."""
    last, stable = _tile(runner), 0
    for _ in range(max_frames):
        runner.step_frame([], phase="coverage", record_state=False)
        if runner.nav_state().in_battle:
            if stats is not None:
                stats["battles"] += 1
            _flee_battle(runner)
        t = _tile(runner)
        stable = stable + 1 if t == last else 0
        last = t
        if stable >= 4:
            break
    return last


def _step_once(runner, action: str, *, stats=None, max_frames: int = 90) -> tuple:
    """Move EXACTLY one tile in `action` (turning in place first if the avatar faces elsewhere),
    then release+settle so the move can't chain into a second tile.

    This is the single source of truth for movement, used by BOTH Phase-1 mapping and the Phase-2
    tour, so the recorded graph is reproduced faithfully. The tile coord flips to a step's
    DESTINATION on the first held frame and leads the sprite by a whole step; holding a fixed
    window (the old `perform_action(hold=12)`) therefore moves 1 OR 2 tiles depending on arrival
    facing/momentum -> non-reproducible phantom 2-tile edges with missing middle nodes. Pressing
    only until the tile changes, then releasing, is deterministic: one press = one tile (a real
    ledge hop still resolves to its true landing during settle)."""
    start = _tile(runner)
    runner.facing = action
    for _ in range(max_frames):
        runner.step_frame([action], phase="coverage", record_state=False)
        if runner.nav_state().in_battle:
            if stats is not None:
                stats["battles"] += 1
            _flee_battle(runner)
            break                                  # fled -> let settle land us on the resting tile
        if _tile(runner) != start:                 # step committed -> stop pressing, don't chain
            break
    return _settle(runner, stats=stats)


def _build_graph(runner, start, max_tiles: int) -> tuple[dict, set, set]:
    """Phase 1: restore-based BFS -> complete cross-map adjacency {tile: {action: neighbour}}.
    Probes every direction at every tile with the SAME single-tile move the tour uses, so every
    edge is exactly one tile (or one true ledge hop / map transition) and every standable tile
    becomes a node -> the Phase-2 tour reproduces the graph faithfully."""
    edges: dict = defaultdict(dict)
    seen = {start}; maps = {start[0]}
    queue = deque([(start, runner.save_state_bytes())])
    while queue and not (max_tiles and len(seen) >= max_tiles):
        tile, sb = queue.popleft()
        for d in DIRS:
            runner.load_state_bytes(sb, record=False)
            t = _step_once(runner, d)
            if t != tile:
                edges[tile][d] = t
                if t not in seen:
                    seen.add(t); maps.add(t[0])
                    queue.append((t, runner.save_state_bytes()))
    return edges, seen, maps


def collect_coverage(*, load_state, output_dir, rom_path, backend, max_step_frames=90,
                     max_clips=4096, max_tiles=0, story_bucket="coverage") -> dict:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 1: MAP (restore-based, no recording) ----------
    mapper = DirectEmulatorRunner(rom_path=rom_path, load_state=load_state, story_bucket=story_bucket)
    mapper.initialize()
    mapper.settle_to_free_overworld()
    s = mapper.state(); start = (s.map, s.x, s.y)
    edges, tiles, maps = _build_graph(mapper, start, max_tiles)
    mapper.close()

    # ---------- Phase 2: TOUR (continuous, recording) ----------
    with ChunkRecorder(out, run_id=f"coverage_{story_bucket}", emulator_fps=60, visual_fps=60,
                       backend=backend, lean=True, metadata={"story_bucket": story_bucket}) as rec:
        sink = WorldModelSink(out)
        runner = DirectEmulatorRunner(rom_path=rom_path, load_state=load_state, story_bucket=story_bucket,
                                      recorder=rec, emulator_fps=60, frame_hook=sink.capture)
        runner.initialize(); sink.capture(runner); runner.settle_to_free_overworld()
        start_bytes = runner.save_state_bytes(); start_facing = runner.facing
        stats = {"battles": 0}
        visited = {start}; failed: set = set(); pos = [start]
        clip_starts = [runner.frame_idx]
        # checkpoints[tile] = (state_bytes, facing): resume points sampled along the tour (every
        # CKPT_EVERY tiles, plus each first-map-entry). On reset we jump to the checkpoint whose
        # tile is GRAPH-NEAREST to a still-unvisited tile -> a short re-walk straight to the next
        # pocket, instead of re-descending one big ledge-map from its single top every clip. One
        # saved tile-state is ~0.4 MB; CKPT_EVERY=6 keeps it bounded (~a few hundred MB worst case).
        CKPT_EVERY = 6
        checkpoints: dict = {start: (start_bytes, start_facing)}
        seen_maps = {start[0]}; mv = [0]

        def walk_edge(frm: tuple, action: str) -> tuple:
            """Walk one tile with the SAME single-tile move Phase-1 used to map the graph, so the
            tour lands exactly on graph nodes. Returns the settled tile (== frm if blocked)."""
            t = _step_once(runner, action, stats=stats, max_frames=max_step_frames)
            if t != frm:                                 # a real move -> maybe snapshot a resume point
                mv[0] += 1
                if (t[0] not in seen_maps or mv[0] % CKPT_EVERY == 0) and t not in checkpoints:
                    seen_maps.add(t[0])
                    checkpoints[t] = (runner.save_state_bytes(), runner.facing)
            return t

        def path_to_unvisited() -> list | None:
            """BFS over the complete graph (skipping failed edges) -> [(from, action, to), ...]
            to the nearest not-yet-toured tile. None if none reachable from pos."""
            src = pos[0]; prev = {src: None}; q = deque([src])
            while q:
                n = q.popleft()
                if n not in visited:
                    steps, c = [], n
                    while prev[c] is not None:
                        p, a = prev[c]; steps.append((p, a, c)); c = p
                    return steps[::-1]
                for a, nb in edges.get(n, {}).items():
                    if (n, a) in failed or nb in prev:
                        continue
                    prev[nb] = (n, a); q.append(nb)
            return None

        def greedy():
            """Re-plan after EVERY single step from the player's ACTUAL tile, always taking the
            first hop of the BFS shortest path to the nearest unvisited tile. Mark an edge failed
            only when the step doesn't move at all (transient block / divergent ledge) so BFS
            routes around it; a step that merely lands off-target is accepted and we re-plan.

            Two stuck-guards, both world-size independent: `no_move` ends the clip when every exit
            thrashes; `stuck` ends it only when the walker is neither touring new tiles NOR closing
            the distance to a frontier (true oscillation). Crucially we do NOT abort just because
            many steps pass without a new tile — re-walking visited tiles from `start` to a distant
            frontier in a big multi-map world is legitimate progress (each step shrinks the path)."""
            no_move = 0; stuck = 0; best = None; last_n = len(visited)
            while True:
                visited.add(pos[0])                          # current tile is toured
                if len(visited) > last_n:                    # reached a new tile -> progressing
                    last_n = len(visited); stuck = 0; best = None
                path = path_to_unvisited()
                if not path:                                 # nothing unvisited reachable from here
                    return
                if best is None or len(path) < best:         # getting closer to a frontier -> progressing
                    best = len(path); stuck = 0
                else:
                    stuck += 1
                    if stuck >= 300:                         # not converging at all -> oscillation, end clip
                        return
                frm, a, _to = path[0]                        # take only the first hop, then re-plan
                t = walk_edge(pos[0], a)
                if t == pos[0]:                              # didn't move -> this exit is blocked now
                    failed.add((pos[0], a)); no_move += 1
                    if no_move >= 8:                         # every exit thrashing -> end this clip
                        return
                else:
                    pos[0] = t; no_move = 0

        import os as _os
        _verbose = bool(_os.environ.get("COV_DEBUG"))

        def _pick_reset(banned: set):
            """Choose where the next clip resumes: the checkpoint whose tile is GRAPH-NEAREST to a
            still-unvisited tile (multi-source forward BFS from all checkpoint tiles — first frontier
            reached wins). That lands the walker one short hop from fresh territory, so big ledge
            maps get descended from many different tops instead of re-walking one entry. `banned`
            skips checkpoints that just failed to progress. `start` is always a checkpoint, so the
            whole graph stays reachable -> completeness preserved."""
            src = {}; q = deque()
            for tile in checkpoints:
                if tile in banned:
                    continue
                src[tile] = tile; q.append(tile)
            while q:
                n = q.popleft()
                if n not in visited:                         # reached a frontier
                    root = src[n]; sb, fac = checkpoints[root]
                    return root, sb, fac
                for a, nb in edges.get(n, {}).items():
                    if nb not in src:
                        src[nb] = src[n]; q.append(nb)
            return start, start_bytes, start_facing           # fallback (shouldn't happen pre-100%)

        greedy()
        stall = 0; banned: set = set()
        # One-way ledges strand the walker -> each reset starts a fresh clip near the nearest unvisited
        # pocket. A no-progress reset bans that checkpoint for the round (so we try the next-nearest);
        # only when banning has exhausted useful checkpoints AND a fresh round still can't progress do
        # we stop. stall counts consecutive no-progress clips overall as a hard safety break.
        while len(clip_starts) <= max_clips and any(n not in visited for n in tiles):
            before = len(visited)
            tile, sb, fac = _pick_reset(banned)
            runner.load_state_bytes(sb, record=False)
            runner.facing = fac; pos[0] = tile
            failed.clear()                                       # NPCs respawn -> retry blocked edges
            clip_starts.append(runner.frame_idx)
            greedy()
            if len(visited) == before:                           # no new tiles this clip
                banned.add(tile); stall += 1
            else:
                stall = 0; banned.clear()                        # progress -> re-enable all checkpoints
            if _verbose:
                print(f"clip {len(clip_starts)}: visited {len(visited)}/{len(tiles)} "
                      f"({100 * len(visited) / max(1, len(tiles)):.1f}%) stall={stall} "
                      f"reset={tile[0]} ckpts={len(checkpoints)}", flush=True)
            if stall >= 12:
                break
        for _ in range(12):
            runner.step_frame([], phase="coverage", record_state=False)
        sink.close(); runner.close()

    coverage = {
        "story_bucket": story_bucket, "graph_tiles": len(tiles), "visited": len(visited),
        "coverage_pct": round(100 * len(visited) / max(1, len(tiles)), 1),
        "map_count": len(maps), "maps": sorted(maps), "clips": len(clip_starts),
        "clip_start_frames": clip_starts, "frames": runner.frame_idx, "battles": stats["battles"],
        "ppu_bytes": sink.ppu.bytes_written,
    }
    (out / "coverage_summary.json").write_text(json.dumps(coverage, indent=2))
    return coverage


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--load-state", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rom-path", default="Emerald-GBAdvance/rom.gba")
    p.add_argument("--backend", default="npz", choices=["auto", "ffv1", "npz"])
    p.add_argument("--max-clips", type=int, default=4096, help="safety cap; stall-break ends earlier")
    p.add_argument("--max-tiles", type=int, default=0, help="0 = whole connected world")
    p.add_argument("--story-bucket", default="coverage")
    a = p.parse_args()
    print(json.dumps(collect_coverage(load_state=a.load_state, output_dir=a.output, rom_path=a.rom_path,
                                      backend=a.backend, max_clips=a.max_clips, max_tiles=a.max_tiles,
                                      story_bucket=a.story_bucket), indent=2))


if __name__ == "__main__":
    main()
