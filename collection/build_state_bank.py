"""Data-closure Phase C: build the constructed-state BANK from the coverage report's red rows.

Reads `audit/coverage_report.json` (what's missing) + `coverage_manifest.json` (what exists) and
emits one verified savestate per closure target via the StateConstructor:

  species_<id>_l<lvl>   a red player-side species as the lead (+20 Poké Balls) — seeded on a
                        grassy post-starter base so battle jobs farm player-side frames for it
  evolve_<id>_l15       starter at level 15 ONE FIGHT from leveling (evolution cutscenes at 16)
  catch_<base>          unmodified party + 30 Poké Balls on each grassy base (the catch axis)

Bases are the storyline checkpoints (`final.state`), auto-classified: post-starter (non-empty,
checksum-valid party) and grassy (their map has a wild table in the manifest). Every construction
re-runs the constructor's pins; every output carries a manifest. `bank.json` indexes the bank for
the closure orchestrator (collect_closure).

Usage:
  CUDA_VISIBLE_DEVICES= .venv/bin/python -m collection.build_state_bank \
      --data_root ../pokemon-worldmodel/data [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from collection.constructor import ITEM_POKE_BALL, StateConstructor, decrypt_box
from collection.extractors.ram import GBAState

G_PLAYER_PARTY = 0x020244EC
STARTERS = (283, 280, 277)                  # Mudkip, Torchic, Treecko (evolve at 16)
SPECIES_LEVEL = 12                          # constructed leads: survivable on pre-badge routes


def classify_bases(con: StateConstructor, data_root: Path, wild_maps: set[str]) -> list[dict]:
    """Storyline checkpoints -> [{name, state, map, grassy}] for post-starter bases only."""
    bases = []
    for seg in sorted((data_root / "storyline_wm").iterdir()):
        st_path = seg / "attempt_000001" / "final.state"
        if not st_path.exists():
            continue
        con.env.load_state(str(st_path))
        con.env.run_frame_with_buttons([])
        st = GBAState.snapshot(con.env)
        try:
            d = decrypt_box(st.bytes(G_PLAYER_PARTY, 80))
        except Exception:
            continue
        if not (d["checksum_ok"] and 0 < d["species"] < 412):
            continue                                          # pre-starter: empty party
        key = f"{st.u8(0x020322E4)},{st.u8(0x020322E5)}"
        bases.append({"name": seg.name, "state": str(st_path), "map": key,
                      "grassy": key in wild_maps})
    return bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--data_root", default="../pokemon-worldmodel/data")
    ap.add_argument("--limit", type=int, default=0, help="construct at most N entries (pilot)")
    args = ap.parse_args()
    root = Path(args.data_root)
    bank_dir = root / "processed/state_bank"
    report = json.loads((root / "processed/audit/coverage_report.json").read_text())
    manifest = json.loads((root / "processed/coverage_manifest.json").read_text())
    wild_maps = {k for k, t in manifest["wild"].items() if "land" in t}

    con = StateConstructor(args.rom)
    bases = classify_bases(con, root, wild_maps)
    grassy = [b for b in bases if b["grassy"]]
    assert grassy, "no grassy post-starter base found"
    print(f"bases: {len(bases)} post-starter ({len(grassy)} grassy)")

    red_species = [int(r["key"][2:]) for r in report["rows"]["species_player"] if not r["ok"]]
    targets: list[dict] = []
    for i, sp in enumerate(red_species):
        targets.append({"name": f"species_{sp}_l{SPECIES_LEVEL}", "intent": "battle",
                        "base": grassy[i % len(grassy)], "species": sp,
                        "level": SPECIES_LEVEL, "near_levelup": False, "balls": 20})
    for sp in STARTERS:
        targets.append({"name": f"evolve_{sp}_l15", "intent": "evolve",
                        "base": grassy[len(targets) % len(grassy)], "species": sp,
                        "level": 15, "near_levelup": True, "balls": 10})
    for b in grassy:
        targets.append({"name": f"catch_{b['name']}", "intent": "catch", "base": b,
                        "species": None, "level": 0, "near_levelup": False, "balls": 30})
    if args.limit:
        targets = targets[:args.limit]

    index = []
    for t in targets:
        out = bank_dir / t["name"]
        if (out / "constructed.state").exists():
            print(f"  skip (exists): {t['name']}")
        else:
            con.load_base(t["base"]["state"])
            if t["species"] is not None:
                con.set_party_slot(0, t["species"], t["level"], near_levelup=t["near_levelup"])
            con.give_item(ITEM_POKE_BALL, t["balls"], "balls")
            con.finalize(out)
            print(f"  constructed: {t['name']}  (base {t['base']['name']})")
        index.append({**{k: v for k, v in t.items() if k != "base"},
                      "base": t["base"]["name"], "map": t["base"]["map"],
                      "state": str(out / "constructed.state")})
    (bank_dir / "bank.json").write_text(json.dumps(
        {"bases": bases, "entries": index}, indent=1))
    print(f"bank: {len(index)} entries -> {bank_dir / 'bank.json'}")


if __name__ == "__main__":
    main()
