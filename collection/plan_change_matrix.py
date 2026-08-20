"""W33 corpus-v2 §9: the (map x stage) CHANGE MATRIX — a PLANNING artifact.

The world is flag-conditioned; sweeping every map at every story stage would be ~10x
coverage cost. This probe loads each milestone savestate on a SCRATCH emulator
(recorder=None, frame-neutral loads — legal for planning: nothing is recorded, no
frame ever elapses) and fingerprints every in-scope map's OBJECT population per stage.
Diffing fingerprints across stages yields {map: [stages where the fingerprint
changed]} — the input the rotation scheduler (plan_expeditions.py) uses to target
re-sweeps at exactly the cells where the world really changed.

Fingerprint method (CHOSEN + DOCUMENTED, per the item-4 spec)
-------------------------------------------------------------
Per (map, stage) the fingerprint is sha256 over one record per ROM ObjectEventTemplate
of the map:

    (local_id, gfx, x, y, movement, flag_id, FLAG VALUE at the stage,
     [VAR value for runtime-resolved gfx slots])

* The static template fields come from the ROM (same header walk as
  `extractors.rom_manifest.walk_maps`, extended with the template's `flagId` at
  offset +20 — pokeemerald `struct ObjectEventTemplate`; probe-validated 2026-08-20:
  object counts match the coverage manifest exactly on all 46 in-scope maps, 129/326
  in-scope objects carry a nonzero flagId, every flagId < 2400 i.e. inside the ledger
  panel's 300-byte flags slice).
* The stage-varying part is exactly **the flags that gate spawns**: for each object,
  the value of ITS OWN gating flag read from the stage state's SaveBlock1 flags slice
  (ledger-panel addresses — `SB1_PTR + SB1_FLAGS`, the RamPanel-cross-checked read).
  pokeemerald hides a template object iff `FlagGet(flagId)`, so this bit IS the spawn
  table for the map, without loading the map.
* gfx ids >= 240 are OBJ_EVENT_GFX_VAR_* slots resolved from VARs at runtime; for
  those the var value (VAR_OBJ_GFX_ID_0 = 0x4010 + slot) joins the record, so a
  var-driven costume/actor swap changes the fingerprint too.
* For the map the stage state is CURRENTLY ON, the live gObjectEvents table
  (`extractors.entities`) is additionally hashed (SEPARATELY, as `live`) as the set
  of present (local_id, graphics_id) pairs — the "live RAM where cheaply possible"
  upgrade; it catches stage-spawned actors the ROM template list can't know.
  Positions are deliberately EXCLUDED from the live part: wander movement makes NPC
  coordinates stage-noise (the template x/y already sits in the static part). The
  live hash is only DIFFED between stages that both have one (the player was on the
  map at both) — otherwise every spine arrival/departure would read as world change.

Known limits (documented, accepted for a planning artifact)
-----------------------------------------------------------
* Script-driven differences that are not object-spawn-gated (invisible coord-event
  triggers, scripted NPC repositioning within a map, level scripts) do NOT move the
  fingerprint. §9's suggested augmentation — map-entry-triggered flags mined from the
  storyline recordings' ledger — is NOT cheaply derivable today: the storyline_wm
  recordings predate the per-tick ledger (no ledger/ chunks on disk), so deriving
  per-map-entry flag writes would need a full §11 replay pass. Recorded here as a
  PILOT FOLLOW-UP; the scheduler compensates by sweeping every map at >= 2 anchor
  stages regardless of the matrix.
* Every map "changes" at the first in-game stage (world init sets the initial hide
  flags from their pre-game zeros). Consumers treating the first game stage as the
  baseline should read `changes_at` entries AFTER the first stage; the scheduler does.
* Perfect spawn-table fidelity is NOT claimed — the matrix targets re-sweeps; a
  missed cell costs coverage efficiency, never correctness (the audit still counts
  what was actually recorded).

Plausibility pin (probe run 2026-08-20, all 51 storyline stage states, 0.4 s):
Littleroot ("0,9") changes at LITTLEROOT_TOWN / PLAYER_HOUSE_ENTERED /
RECEIVED_POKEDEX / ROUTE101_AFTER_POKEDEX (mom, rival, Birch actors); late routes
Route 116 ("0,31") and Route 115 ("0,30") are stable after world init.

Usage:
  .venv/bin/python -m collection.plan_change_matrix \
      --data_root ../pokemon-worldmodel/data \
      [--out data/processed/w33_change_matrix.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from collection.catalog import MILESTONE_ORDER
from collection.extractors.ledger_panel import (
    FLAGS_BYTES,
    SB1_FLAGS,
    SB1_PTR,
    SB1_VARS,
    VARS_BYTES,
)
from collection.extractors.rom_manifest import G_MAP_GROUPS, MAX_GROUPS, MAX_MAPS, Rom

OBJ_TEMPLATE_SIZE = 24
OBJ_FLAG_OFF = 20                  # ObjectEventTemplate.flagId (u16) — probe-validated
VAR_OBJ_GFX_ID_0 = 0x4010          # vars backing OBJ_EVENT_GFX_VAR_0.. (gfx ids 240+)
GFX_VAR_BASE = 240


def rom_object_templates(rom_path: str) -> dict[str, list[dict]]:
    """map key -> ObjectEventTemplates incl. the spawn-gating flagId. Same gMapGroups
    walk as rom_manifest.walk_maps (bounds logic copied verbatim), plus flagId@+20."""
    r = Rom(Path(rom_path).read_bytes())
    gps = []
    for g in range(MAX_GROUPS):
        gp = r.u32(G_MAP_GROUPS + g * 4)
        if not r.ok(gp):
            break
        gps.append(gp)
    bounds = sorted(gps)
    counts = {gp: (min((bounds[i + 1] - gp) // 4, MAX_MAPS) if i + 1 < len(bounds)
                   else MAX_MAPS) for i, gp in enumerate(bounds)}
    out: dict[str, list[dict]] = {}
    for g, gp in enumerate(gps):
        for n in range(counts[gp]):
            hp = r.u32(gp + n * 4)
            if not r.ok(hp):
                break
            lay = r.u32(hp)
            if not r.ok(lay):
                break
            if not (1 <= r.u32(lay) <= 300 and 1 <= r.u32(lay + 4) <= 300):
                break
            key = f"{g},{n}"
            ev = r.u32(hp + 0x04)
            if not r.ok(ev):
                out[key] = []
                continue
            n_obj, op = r.u8(ev), r.u32(ev + 4)
            objs = []
            if r.ok(op):
                for i in range(n_obj):
                    o = op + i * OBJ_TEMPLATE_SIZE
                    objs.append({
                        "local_id": r.u8(o), "gfx": r.u8(o + 1),
                        "x": r.s16(o + 4), "y": r.s16(o + 6),
                        "movement": r.u8(o + 9),
                        "flag": r.u16(o + OBJ_FLAG_OFF),
                    })
            out[key] = objs
    return out


def _flag_bit(flags: bytes, flag: int) -> int:
    return (flags[flag // 8] >> (flag % 8)) & 1


def map_fingerprint(templates: list[dict], flags: bytes, vars_: bytes,
                    live_pairs: list[tuple[int, int]] | None = None) -> dict:
    """{"static": sha256 of the template+gating-flag records, "live": sha256 of the
    present (local_id, gfx) pairs or None} — see the module docstring."""
    rec = []
    for t in templates:
        var_val = None
        if t["gfx"] >= GFX_VAR_BASE:
            vo = (VAR_OBJ_GFX_ID_0 - 0x4000 + (t["gfx"] - GFX_VAR_BASE)) * 2
            if vo + 1 < len(vars_):
                var_val = vars_[vo] | (vars_[vo + 1] << 8)
        rec.append((t["local_id"], t["gfx"], t["x"], t["y"], t["movement"],
                    t["flag"], _flag_bit(flags, t["flag"]) if t["flag"] else 0,
                    var_val))
    static = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    live = None
    if live_pairs is not None:
        live = hashlib.sha256(
            json.dumps(sorted(live_pairs)).encode()).hexdigest()
    return {"static": static, "live": live}


def map_windows(matrix: dict, key: str) -> list[str]:
    """Stage-window starts for a map: baseline + every change stage (post-init; the
    first two stages are the pre-game/world-init baseline). Shared by the scheduler
    (plan_expeditions) and the v2 coverage audit — the ONE definition of a cell."""
    stages = matrix["stages"]
    init = set(stages[:2])
    return [stages[0]] + [s for s in matrix["matrix"].get(key, []) if s not in init]


def window_of(matrix: dict, key: str, stage: str) -> str:
    """The window (start stage) a sweep at `stage` falls into for `key`."""
    idx = MILESTONE_ORDER.index(stage)
    cur = None
    for w in map_windows(matrix, key):
        if MILESTONE_ORDER.index(w) <= idx:
            cur = w
    return cur if cur is not None else map_windows(matrix, key)[0]


def stage_states(storyline_root: Path) -> list[tuple[str, Path]]:
    """(milestone, final.state) for every stage on disk, in MILESTONE_ORDER."""
    out = []
    for m in MILESTONE_ORDER:
        p = storyline_root / m / "attempt_000001" / "final.state"
        if p.exists():
            out.append((m, p))
    return out


def build_change_matrix(*, data_root: str | Path = "../pokemon-worldmodel/data",
                        rom_path: str = "Emerald-GBAdvance/rom.gba",
                        stages: list[tuple[str, Path]] | None = None,
                        env=None) -> dict:
    """The full probe: scratch emulator over the stage states -> matrix dict.
    `stages`/`env` injectable for tests; env (if given) must be an initialized
    EmeraldEmulator the caller owns."""
    from collection.extractors.entities import entities
    from collection.extractors.ram import GBAState
    from collection.extractors.terrain import terrain

    root = Path(data_root)
    manifest = json.loads((root / "processed/coverage_manifest.json").read_text())
    scope = list(manifest["scope_maps"])
    templates = rom_object_templates(rom_path)

    if stages is None:
        stages = stage_states(root / "storyline_wm")
    if not stages:
        raise RuntimeError(f"no stage states under {root / 'storyline_wm'}")

    own_env = env is None
    if own_env:
        from pokemon_env.emulator import EmeraldEmulator
        env = EmeraldEmulator(rom_path=rom_path)
        env.initialize()
    fps: dict[str, dict[str, str]] = {k: {} for k in scope}
    player_map: dict[str, str | None] = {}
    try:
        for stage, path in stages:
            # frame-neutral: the scratch emulator never advances a frame (planning-only)
            env.load_state(str(path), run_settle_frame=False)
            st = GBAState.from_env(env)
            sb1 = st.u32(SB1_PTR)
            flags = st.bytes(sb1 + SB1_FLAGS, FLAGS_BYTES)
            vars_ = st.bytes(sb1 + SB1_VARS, VARS_BYTES)
            snap = GBAState.snapshot(env)
            t = terrain(snap)
            here = f"{t.map_group},{t.map_num}" if t is not None else None
            player_map[stage] = here
            live = None
            if here in scope:
                live = [(e.local_id, e.graphics_id)
                        for e in entities(snap) if not e.is_player]
            for k in scope:
                fps[k][stage] = map_fingerprint(
                    templates.get(k, []), flags, vars_,
                    live_pairs=live if k == here else None)
    finally:
        if own_env:
            env.stop()

    stage_names = [s for s, _ in stages]
    matrix: dict[str, list[str]] = {}
    for k in scope:
        prev, changes = None, []
        for s in stage_names:
            v = fps[k][s]
            if prev is not None and (
                    v["static"] != prev["static"]
                    or (v["live"] is not None and prev["live"] is not None
                        and v["live"] != prev["live"])):
                changes.append(s)
            prev = v
        matrix[k] = changes
    return {
        "schema_version": 1,
        "method": "sha256(ObjectEventTemplates + per-object gating-flag values "
                  "+ gfx-var values; live gObjectEvents presence for the loaded map) "
                  "— see collection/plan_change_matrix.py docstring for limits",
        "stages": stage_names,
        "player_map": player_map,
        "matrix": matrix,
        "fingerprints": fps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--out", default=None,
                    help="default: <data_root>/processed/w33_change_matrix.json")
    args = ap.parse_args()
    m = build_change_matrix(data_root=args.data_root, rom_path=args.rom)
    out = Path(args.out) if args.out else Path(args.data_root) / "processed/w33_change_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1, sort_keys=True))
    init = set(m["stages"][:2])            # pre-game + world-init stages = baseline
    post = {k: [s for s in v if s not in init] for k, v in m["matrix"].items()}
    print(f"stages: {len(m['stages'])} | maps: {len(m['matrix'])} | "
          f"maps with post-init changes: {sum(1 for v in post.values() if v)}")
    for k, v in sorted(post.items()):
        if v:
            print(f"  {k:6s} changes at: {v}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
