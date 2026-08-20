"""ENCOUNTER_FARM expedition block (W33 corpus-v2 §2): foe-species coverage by
battle-and-flee in each target map's grass until the assigned species sightings are met.

Targets: [{"map": "0,16", "area": "land", "species_targets": {species_id: min_sightings}}].
Per target: goto the map, navigator-goal into grass (behavior 0x02 mask), pace within
the patch until a wild battle starts, read the foe species from the LIVE gBattleMons[1]
battle copy (ledger-panel addresses), count the sighting, FLEE, repeat. Empty
species_targets = budget-bound farming (every sighting still counted).

Foe read timing, measured 2026-08-20 on Route 101: by the frame pace_grass/goto report
'battle' (memory_reader.is_in_battle), gBattleMons[1] is ALREADY populated with the
current encounter (species/level/hp all sane) — the flag flips after battle setup. The
struct is stale-persistent from the PREVIOUS battle outside battles (measured: species
277 leftovers pre-battle), so reads are gated on the gMain inBattle bit + a short
stability window + sanity bounds, never taken from the overworld.

Water/fish areas: NOT farmable in scope. Surf is gated far past the Stone Badge (the
corpus scope) and the harness's Surf-gated exclusions are live-verified policy; rods
are the fishing block's job (W33 §2). Such targets are skipped with a documented note
in the summary — never silently.
"""
from __future__ import annotations

import random

from collection import navigator as nav
from collection.extractors.ledger_panel import IN_BATTLE_ADDR, _battle_mon
from collection.extractors.ram import GBAState
from collection.playthrough.blocks.base import flee_battle

MAX_SPECIES = 411                # Emerald internal species ids end at 411


def read_foe_species(runner, *, min_wait: int = 30, cap: int = 900) -> int | None:
    """Current battle's foe species from the live gBattleMons copy, or None.
    Gated on the gMain inBattle bit; accepts a value only after it reads back
    identically 3 polls running with sane hp/level (stale-struct insurance)."""
    st = GBAState.from_env(runner.env)
    nav._hold(runner, [], min_wait, "encounter")
    last, stable = 0, 0
    for _ in range(max(1, (cap - min_wait) // 8)):
        if not ((st.u8(IN_BATTLE_ADDR) >> 1) & 1):
            break
        foe = _battle_mon(st, 1)
        ok = (0 < foe["species"] <= MAX_SPECIES and 0 < foe["hp"] <= foe["max_hp"]
              and 1 <= foe["level"] <= 100)
        if ok and foe["species"] == last:
            stable += 1
            if stable >= 3:
                return last
        else:
            stable = 0
            last = foe["species"] if ok else 0
        nav._hold(runner, [], 8, "encounter")
    # never count a value that failed the 3-poll stability gate (stale-struct insurance)
    return None


class EncounterFarm:
    name = "encounter_farm"
    phase = "encounter"

    def __init__(self, targets: list[dict], frames: int = 60000, seed: int = 0):
        self.targets = list(targets)
        self.frames = frames                        # whole-block frame budget
        self.seed = seed

    def _sight(self, runner, summary: dict) -> None:
        species = read_foe_species(runner)
        summary["battles"] += 1
        if species:
            k = str(species)                        # str keys: summary must survive json
            summary["sightings"][k] = summary["sightings"].get(k, 0) + 1
        flee_battle(runner)

    def _met(self, want: dict[int, int], summary: dict) -> bool:
        return bool(want) and all(summary["sightings"].get(str(s), 0) >= n
                                  for s, n in want.items())

    def _farm(self, runner, mk, key: str, want: dict[int, int], rng, deadline: int,
              summary: dict) -> str:
        while runner.frame_idx < deadline:
            if self._met(want, summary):
                return "met"
            budget = deadline - runner.frame_idx
            t, _, _ = nav._state(runner)
            g = nav.grass_goal(t, mk.behaviors(t)) if t is not None else None
            if g is None or not g.any():
                return "no_grass"
            r = nav.goto_grass(runner, mk, budget=budget)
            if r == "battle":
                self._sight(runner, summary)        # en-route encounters count too
                continue
            if r == "stuck":
                return "grass_unreachable"
            if r != "arrived":
                return "budget"
            r = nav.pace_grass(runner, mk, rng, budget=min(6000, budget))
            if r == "battle":
                self._sight(runner, summary)
            # 'left' → re-goto grass; 'budget' → next slice (deadline still governs)
        return "budget"

    def run(self, runner, mk, ctx) -> dict:
        rng = random.Random(self.seed)
        summary = dict(sightings={}, battles=0, frames=0, skipped=[], results=[])
        f0 = runner.frame_idx
        deadline = f0 + self.frames
        for tgt in self.targets:
            key, area = tgt["map"], tgt.get("area", "land")
            want = {int(s): int(n) for s, n in (tgt.get("species_targets") or {}).items()}
            if area != "land":
                summary["skipped"].append(dict(
                    map=key, area=area,
                    reason="water/fish farming needs Surf/rods — Surf is post-scope "
                           "(start→Stone Badge) per the live-verified Surf-gated "
                           "exclusions; rods belong to the fishing block"))
                continue
            if "land" not in mk_wild(mk, key):
                summary["skipped"].append(dict(map=key, area=area,
                                              reason="no land wild table in the manifest"))
                continue
            for _ in range(3):                      # reach the map, fleeing en-route battles
                t, _, _ = nav._state(runner)
                if t is not None and f"{t.map_group},{t.map_num}" == key:
                    break
                r = nav.goto_map(runner, mk, key, hop_budget=max(2000, deadline - runner.frame_idx))
                if r == "battle":
                    self._sight(runner, summary)
            else:
                summary["results"].append(dict(map=key, outcome="map_unreached"))
                continue
            outcome = self._farm(runner, mk, key, want, rng, deadline, summary)
            summary["results"].append(dict(map=key, outcome=outcome))
        summary["frames"] = runner.frame_idx - f0
        return summary


def mk_wild(mk, key: str) -> dict:
    """The manifest's per-map wild tables ({'land'|'water'|'fish': {rate, mons}}), {} when
    absent. MapKnowledge doesn't load the wild section (built for warps/behaviors), so read
    it lazily from the same manifest file and cache on the instance."""
    tables = getattr(mk, "_wild_tables", None)
    if tables is None:
        import json
        from pathlib import Path

        from collection.navigator import MANIFEST_JSON
        mf = Path(MANIFEST_JSON)
        tables = json.loads(mf.read_text()).get("wild", {}) if mf.exists() else {}
        mk._wild_tables = tables
    return tables.get(key, {})
