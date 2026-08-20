"""W33 corpus-v2 §3.3 per-tick ledger + §10.2 ui_state. Real emulator, real states.

The acceptance bar is the CROSS-CHECK: `collection.extractors.ledger_panel` must agree
field-for-field with the model repo's `RamPanel` (tools/state/label_corpus.py — the
verified Σ-label extractor) on the SAME live emulator state. RamPanel emits panel buckets
(hp as ceil(4*hp/max_hp), exp//256 capped, -1 sentinels, fused battle-copy series); the
ledger emits raw unsigned values with separated party/battle series — the comparison
applies RamPanel's own transforms to the ledger row, so equality is semantic, not
representational.

ui_state: only sources verified live are asserted here (none/dialog/start_menu/bag/
party/summary — battle_menu has no verified source and is never emitted; see the
ledger_panel module docstring). The mart (6) and pc (7) carriers were verified in the
W33 item 3c work and are probed live during the real mart/PC flows by
tests/test_blocks_v2c.py — no static fixture reaches those UIs. Menu drives use
START-cursor RAM feedback (collect_behaviors' seek pattern), not blind press counts.
"""

import importlib.util
import json
import math
import sys

import numpy as np
import pytest

from collection.extractors.ledger_panel import (
    START_MENU_WINDOW_ID,
    UI_BAG,
    UI_DIALOG,
    UI_NONE,
    UI_PARTY_MENU,
    UI_START_MENU,
    UI_SUMMARY,
    read_ledger,
    ui_state,
)
from collection.extractors.ram import GBAState
from pokemon_env.emulator import EmeraldEmulator

MODEL_REPO = "/root/code/proj-minhyuk-2026/pokemon-worldmodel"
START_MENU_CURSOR = 0x0203760E                  # collect_behaviors.START_MENU_CURSOR


@pytest.fixture(scope="module")
def emu():
    e = EmeraldEmulator(rom_path="Emerald-GBAdvance/rom.gba", headless=True, sound=False)
    e.initialize()
    yield e
    e.stop()


@pytest.fixture(scope="module")
def ram_panel_cls():
    """The model repo's RamPanel, loaded by file path (the harness has its own `tools`
    package, so a plain `import tools.state.label_corpus` would collide)."""
    if MODEL_REPO not in sys.path:
        sys.path.insert(0, MODEL_REPO)          # for its pokemon_worldmodel imports
    spec = importlib.util.spec_from_file_location(
        "w33_ref_label_corpus", MODEL_REPO + "/tools/state/label_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RamPanel


def _row(emu):
    return read_ledger(GBAState.snapshot(emu))


def _run(emu, buttons, n):
    for _ in range(n):
        emu.run_frame_with_buttons(buttons)


def _poll_ui(emu, want, frames=180, settle=30):
    """Poll for a ui_state, then let the screen finish initializing — a press landing on
    init frames is dropped (the reason blind press counts are brittle here)."""
    for _ in range(frames):
        if ui_state(GBAState.snapshot(emu)) == want:
            _run(emu, [], settle)
            return True
        emu.run_frame_with_buttons([])
    return False


def _seek_start_slot(emu, slot):
    for _ in range(10):
        if GBAState.snapshot(emu).u8(START_MENU_CURSOR) == slot:
            return True
        _run(emu, ["down"], 4)
        _run(emu, [], 14)
    return False


# ---------------------------------------------------------------- cross-check (the bar)

CROSS_STATES = ["tests/states/torchic.state", "tests/states/wild_battle.state",
                "tests/states/house.state", "tests/states/dialog.state",
                "tests/states/truck.state"]


@pytest.mark.parametrize("state_path", CROSS_STATES)
def test_cross_check_vs_ram_panel(emu, ram_panel_cls, state_path):
    emu.load_state(state_path)
    ours = _row(emu)                            # snapshot; no frame runs before ref reads
    ref = ram_panel_cls(emu).read()

    assert ours["valid"] == 1
    assert ours["in_battle"] == ref["in_battle"]
    assert ours["money"] == ref["money"]
    assert ours["party_count"] == ref["party_count"]

    # flag/var archives byte-equal, badges recomputed from the reference's own archive
    assert ref["_flags_slice"] is not None and ours["flags"].tobytes() == ref["_flags_slice"]
    assert ref["_vars_slice"] is not None and ours["vars"].tobytes() == ref["_vars_slice"]
    ref_badges = 0
    for i in range(8):
        f = 0x867 + i
        ref_badges |= ((ref["_flags_slice"][f // 8] >> (f % 8)) & 1) << i
    assert ours["badges"] == ref_badges
    assert (ours["badges"] & 1) == ref["flag_badge01"]

    for s in range(3):
        if ref[f"species{s}"] == -1:
            assert ours["species"][s] == 0      # ledger absent-slot convention
            continue
        # RamPanel's slot series is FUSED (battle copy while this slot is active);
        # rebuild the same fusion from the ledger's separated series.
        if (ours["in_battle"] and ours["active_species"]
                and ours["active_personality"] == ours["personality"][s]):
            live = {"species": int(ours["active_species"]), "level": int(ours["active_level"]),
                    "hp": int(ours["active_hp"]), "max_hp": int(ours["active_max_hp"]),
                    "exp": int(ours["active_exp"]), "moves": ours["active_moves"].tolist(),
                    "pp": ours["active_pp"].tolist()}
        else:
            live = {"species": int(ours["species"][s]), "level": int(ours["level"][s]),
                    "hp": int(ours["hp"][s]), "max_hp": int(ours["max_hp"][s]),
                    "exp": int(ours["exp"][s]), "moves": ours["moves"][s].tolist(),
                    "pp": ours["pp"][s].tolist()}
        assert live["species"] == ref[f"species{s}"]
        assert live["level"] == ref[f"level{s}"]
        assert ref[f"hp{s}"] == (math.ceil(4 * live["hp"] / live["max_hp"]) if live["max_hp"] else -1)
        assert ref[f"exp{s}"] == min(live["exp"] // 256, 255)
        for i in range(4):
            assert live["moves"][i] == ref[f"move{s}_{i}"]
            assert live["pp"][i] == ref[f"pp{s}_{i}"]
        if s == 0:
            assert (int(ours["personality"][0]) & 0xFF) == ref["neg_personality_lo0"]

    if ours["in_battle"]:
        assert ours["foe_species"] == ref["foe_species"]
        assert ours["foe_level"] == ref["foe_level"]
        assert ref["foe_hp"] == (math.ceil(4 * int(ours["foe_hp"]) / int(ours["foe_max_hp"]))
                                 if ours["foe_max_hp"] else -1)
    else:
        assert ref["foe_species"] == -1 and ours["foe_species"] == 0


def test_known_values_torchic(emu):
    """Fixture ground truth, independent of RamPanel (both extractors being wrong the same
    way would slip the cross-check; a lv-5 Torchic with Scratch/Growl and 3000 starting
    money would not)."""
    emu.load_state("tests/states/torchic.state")
    r = _row(emu)
    assert r["valid"] == 1 and r["in_battle"] == 0 and r["party_count"] == 1
    assert r["species"][0] == 280 and r["level"][0] == 5           # SPECIES_TORCHIC
    assert r["hp"][0] == r["max_hp"][0] == 19
    assert r["moves"][0].tolist() == [10, 45, 0, 0]                # Scratch, Growl
    assert r["pp"][0].tolist() == [35, 40, 0, 0]
    assert r["money"] == 3000 and r["badges"] == 0


def test_known_values_wild_battle(emu):
    emu.load_state("tests/states/wild_battle.state")
    r = _row(emu)
    assert r["in_battle"] == 1
    assert r["foe_species"] == 290 and r["foe_level"] == 3         # SPECIES_WURMPLE (screen-checked)
    assert r["foe_hp"] == 16 and r["foe_max_hp"] == 16
    assert r["active_species"] == 280 and r["active_hp"] == 19
    assert r["active_personality"] == r["personality"][0]          # the fusion key matches slot 0
    # battle_menu (8) has no verified source — battle frames read 0 (module docstring)
    assert r["ui_state"] == UI_NONE


# ---------------------------------------------------------------------- sink integration

def test_sink_writes_ledger_chunks(tmp_path, monkeypatch):
    from collection.direct_runner import DirectEmulatorRunner
    from collection.recorder import ChunkRecorder
    import collection.world_model_sink as wms
    from collection.world_model_sink import WorldModelSink

    # sharp minor 4: the PPU blob is serialized ONCE per frame and shared between the
    # delta writer and the ledger read (was serialized twice)
    calls = {"n": 0}
    orig_serialize = wms.serialize_ppu

    def counting(state):
        calls["n"] += 1
        return orig_serialize(state)

    monkeypatch.setattr(wms, "serialize_ppu", counting)

    rec = ChunkRecorder(output_dir=tmp_path, run_id="ledger_test", backend="npz",
                        emulator_fps=60, visual_fps=60)
    sink = WorldModelSink(tmp_path)
    runner = DirectEmulatorRunner(load_state="tests/states/torchic.state", recorder=rec,
                                  frame_hook=sink.capture, savestate_every=0)
    runner.initialize()
    for _ in range(50):
        runner.step_frame([], phase="action")
    sink.close()
    rec.close()
    runner.close()

    # pre-existing sink outputs untouched (hook fires per step_frame -> 50 rows)
    assert (tmp_path / "ppu_state.bin").exists()
    assert (tmp_path / "ppu_state.bin.idx.json").exists()
    assert sum(1 for _ in open(tmp_path / "semantic.jsonl")) == 50
    assert calls["n"] == 50                                        # once per frame, not twice

    idx = json.loads((tmp_path / "ledger" / "index.json").read_text())
    assert idx["chunk_frames"] == 4096
    assert idx["frames"] == 50
    assert idx["chunks"] == [["chunk_000000.npz", 0, 50]]          # partial flushed on close
    with np.load(tmp_path / "ledger" / "chunk_000000.npz") as z:
        assert set(z.files) == set(idx["fields"])
        for name, spec in idx["fields"].items():
            assert z[name].dtype == np.dtype(spec["dtype"]), name
            assert z[name].shape == (50, *spec["shape"]), name
        assert z["flags"].shape == (50, 300) and z["vars"].shape == (50, 512)
        assert z["frame_idx"].tolist() == list(range(1, 51))
        assert (z["valid"] == 1).all()
        assert (z["species"][:, 0] == 280).all() and (z["level"][:, 0] == 5).all()
        assert (z["money"] == 3000).all() and (z["in_battle"] == 0).all()
        assert (z["ui_state"] == UI_NONE).all()


def test_ledger_writer_chunk_cadence(tmp_path, emu):
    from collection.world_model_sink import LedgerWriter

    emu.load_state("tests/states/torchic.state")
    row = _row(emu)
    w = LedgerWriter(tmp_path / "ledger", chunk_frames=8)
    for i in range(20):
        w.add(i, row)
    w.close()
    idx = json.loads((tmp_path / "ledger" / "index.json").read_text())
    assert idx["chunks"] == [["chunk_000000.npz", 0, 8],
                             ["chunk_000001.npz", 8, 8],
                             ["chunk_000002.npz", 16, 4]]
    assert idx["frames"] == 20
    with np.load(tmp_path / "ledger" / "chunk_000002.npz") as z:
        assert z["flags"].shape == (4, 300) and z["moves"].shape == (4, 3, 4)


def test_ledger_index_written_at_first_flush_and_refreshed(tmp_path, emu):
    """Sharp minor 4: a crashed run (no close) must still leave a valid index.json —
    written at the FIRST chunk flush and refreshed on every later flush."""
    from collection.world_model_sink import LedgerWriter

    emu.load_state("tests/states/torchic.state")
    row = _row(emu)
    w = LedgerWriter(tmp_path / "ledger", chunk_frames=8)
    for i in range(8):
        w.add(i, row)                       # first flush fires here
    idx = json.loads((tmp_path / "ledger" / "index.json").read_text())
    assert idx["chunks"] == [["chunk_000000.npz", 0, 8]]
    assert idx["frames"] == 8
    for i in range(8, 16):
        w.add(i, row)                       # second flush refreshes it
    idx = json.loads((tmp_path / "ledger" / "index.json").read_text())
    assert idx["chunks"][1] == ["chunk_000001.npz", 8, 8]
    assert idx["frames"] == 16
    # no close() ever ran: the on-disk schema is already valid for everything flushed
    with np.load(tmp_path / "ledger" / "chunk_000001.npz") as z:
        assert set(z.files) == set(idx["fields"])


# --------------------------------------------------------------------------- ui_state

def test_ui_state_overworld_and_start_menu(emu):
    emu.load_state("tests/states/torchic.state")
    assert ui_state(GBAState.snapshot(emu)) == UI_NONE
    _run(emu, ["start"], 3)
    assert _poll_ui(emu, UI_START_MENU, frames=40)
    assert GBAState.snapshot(emu).u8(START_MENU_WINDOW_ID) != 0xFF
    # close: the start-menu signal must drop; BG0 residue may read DIALOG (documented)
    _run(emu, ["b"], 3)
    _run(emu, [], 30)
    assert ui_state(GBAState.snapshot(emu)) in (UI_NONE, UI_DIALOG)
    assert GBAState.snapshot(emu).u8(START_MENU_WINDOW_ID) == 0xFF


def test_ui_state_dialog_fixture(emu):
    emu.load_state("tests/states/dialog.state")
    assert ui_state(GBAState.snapshot(emu)) == UI_DIALOG


def test_ui_state_party_menu_and_summary(emu):
    emu.load_state("tests/states/torchic.state")
    _run(emu, ["start"], 3)
    _run(emu, [], 30)
    assert _seek_start_slot(emu, 0)             # POKeMON
    _run(emu, ["a"], 3)
    assert _poll_ui(emu, UI_PARTY_MENU)
    _run(emu, ["a"], 3)                         # select the mon -> action popup (same cb2)
    _run(emu, [], 60)
    assert ui_state(GBAState.snapshot(emu)) == UI_PARTY_MENU
    _run(emu, ["a"], 3)                         # SUMMARY
    assert _poll_ui(emu, UI_SUMMARY)
    _run(emu, ["b"], 3)
    assert _poll_ui(emu, UI_PARTY_MENU)


def test_ui_state_bag(emu):
    emu.load_state("tests/states/torchic.state")
    _run(emu, ["start"], 3)
    _run(emu, [], 30)
    assert _seek_start_slot(emu, 1)             # BAG
    _run(emu, ["a"], 3)
    assert _poll_ui(emu, UI_BAG)
    _run(emu, ["b"], 3)                         # back out to the start menu
    assert _poll_ui(emu, UI_START_MENU)
