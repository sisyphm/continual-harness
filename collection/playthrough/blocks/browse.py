"""BROWSE life block: open the START menu, flash BAG (pocket cycle) and PARTY, close.

Purpose is DATA DIVERSITY — the corpus lacks menu/bag/party screens; this generates
those pixels. It does NOT perform verified transactions (menu-mode RAM statics are
stale-persistent on this build, so mid-menu mode detection isn't reliable — see the
menu-block wall notes). Safety comes from the base wrapper: bounded, force-closes,
returns to the anchor tile, can never break the spine.

Navigation uses the calibrated runner.perform_action cadence (raw frame-stepping drops
inputs). Start-menu order in Emerald is fixed; BAG/PARTY indices shift by whether the
Pokédex is present, so we detect that cheaply from party/pokedex context passed in ctx.
"""
from __future__ import annotations


class Browse:
    name = "browse"
    battle_strategy = "run"

    def __init__(self, seed: int = 0):
        # phase sequence of (button) actions; reactive stepping via base wrapper.
        self._plan: list[str] | None = None
        self._i = 0

    def setup(self, state, ctx):
        # Menu order: [POKéDEX?, POKéMON, BAG, ...]. Pokédex present once received.
        has_dex = bool((state.get("game") or {}).get("pokedex_owned")
                       or (state.get("game") or {}).get("has_pokedex"))
        party_idx = 1 if has_dex else 0
        bag_idx = 2 if has_dex else 1
        seq: list[str] = ["start"]
        # BAG: down to bag, A, cycle pockets, B out
        seq += ["down"] * bag_idx + ["a", "right", "right", "down", "down", "left", "b"]
        # PARTY: from start menu, up to party, A, B
        seq += ["start", "down"] * 0  # menu still open after B; realign
        seq += ["up"] * (bag_idx - party_idx) + ["a", "b"]
        # close start menu
        seq += ["b", "b"]
        self._plan = seq
        self._i = 0

    def act(self, state, ctx):
        if self._plan is None or self._i >= len(self._plan):
            return None
        btn = self._plan[self._i]
        self._i += 1
        return btn
