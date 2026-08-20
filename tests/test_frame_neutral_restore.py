"""W33 corpus-v2 §3.2: mid-recording restores must be FRAME-NEUTRAL.

v1's replay killer: EmeraldEmulator.load_state ran one hidden emulator frame after
every raw load ("settle frame"), so every silent BFS/simulation restore left reality
one unrecorded frame ahead of the recording. These tests pin the fix:

- run_settle_frame=False makes load->save byte-stable (nothing elapses), and
- a simulate-then-restore round trip returns the live emulator to the EXACT
  pre-simulation state bytes,
- while the default path (run_settle_frame=True) still runs its one video-refresh
  frame, preserving v1 caller behavior.
"""

import glob

import pytest

from pokemon_env.emulator import EmeraldEmulator

STATES = sorted(glob.glob("tests/states/*.state"))


@pytest.fixture(scope="module")
def emu():
    e = EmeraldEmulator(rom_path="Emerald-GBAdvance/rom.gba", headless=True, sound=False)
    e.initialize()
    yield e
    e.stop()


def test_frame_neutral_load_is_byte_stable(emu):
    raw = open(STATES[0], "rb").read()
    emu.load_state(state_bytes=raw, run_settle_frame=False)
    first = bytes(emu.core.save_raw_state())
    emu.load_state(state_bytes=raw, run_settle_frame=False)
    second = bytes(emu.core.save_raw_state())
    assert first == second


def test_simulate_restore_roundtrip_is_exact(emu):
    raw = open(STATES[0], "rb").read()
    emu.load_state(state_bytes=raw, run_settle_frame=False)
    pre = bytes(emu.core.save_raw_state())
    # simulate: arbitrary frames with inputs (what _simulate_action does)
    for _ in range(12):
        emu.run_frame_with_buttons(["a"])
    for _ in range(24):
        emu.run_frame_with_buttons([])
    assert bytes(emu.core.save_raw_state()) != pre  # simulation really moved the world
    emu.load_state(state_bytes=pre, run_settle_frame=False)
    post = bytes(emu.core.save_raw_state())
    assert post == pre  # restore is EXACT: no hidden frame, no drift


def test_default_load_still_settles_one_frame(emu):
    raw = open(STATES[0], "rb").read()
    emu.load_state(state_bytes=raw, run_settle_frame=False)
    neutral = bytes(emu.core.save_raw_state())
    emu.load_state(state_bytes=raw)  # default: settle frame runs (video refresh)
    settled = bytes(emu.core.save_raw_state())
    assert settled != neutral  # documents the intentional difference


def test_ram_reads_valid_without_settle_frame(emu):
    raw = open(STATES[0], "rb").read()
    emu.load_state(state_bytes=raw, run_settle_frame=False)
    r = emu.memory_reader
    coords_a, money_a = r.read_coordinates(), r.read_money()
    emu.core.run_frame()
    assert (coords_a, money_a) == (r.read_coordinates(), r.read_money())


def _direct_ewram(emu) -> bytes:
    """Cache-free EWRAM ground truth straight from the mgba core (region 2)."""
    from pokemon_env.emulator import ffi

    mem_core = emu.core.memory.u8._core
    size = ffi.new("size_t *")
    ptr = ffi.cast("uint8_t *", mem_core.getMemoryBlock(mem_core, 2, size))
    return bytes(ffi.buffer(ptr, size[0])[:])


def test_frame_neutral_load_refreshes_env_read_cache(emu):
    """M2: the emulator-level _mem_cache is only cleared by the frame callback, which a
    frame-neutral load never runs — load_state must clear it itself, or env.read_u8
    (the seam GBAState.from_env / read_memory consumers use) serves the PREVIOUS
    state's bytes. Two different fixtures force a real difference."""
    a = open(STATES[0], "rb").read()
    b = open("tests/states/torchic.state", "rb").read()
    assert a != b

    emu.load_state(state_bytes=a, run_settle_frame=False)
    ewram_a = _direct_ewram(emu)
    _ = emu.read_u8(0x02000000)                  # populates _mem_cache with A's EWRAM

    emu.load_state(state_bytes=b, run_settle_frame=False)
    ewram_b = _direct_ewram(emu)
    off = next(i for i in range(len(ewram_a)) if ewram_a[i] != ewram_b[i])

    got = emu.read_u8(0x02000000 + off)          # env-level read, no frame ran
    assert got == ewram_b[off]                   # LOADED state's value...
    assert got != ewram_a[off]                   # ...not the stale pre-load one


def test_load_state_failure_raises(emu):
    """M3: a failed restore must never silently continue — garbage bytes and empty
    payloads raise, and the emulator stays usable afterwards (mgba rejects the bad
    state before touching the core)."""
    with pytest.raises(RuntimeError):
        emu.load_state(state_bytes=b"garbage-not-a-savestate")
    with pytest.raises(RuntimeError):
        emu.load_state(state_bytes=b"")
    raw = open(STATES[0], "rb").read()
    emu.load_state(state_bytes=raw, run_settle_frame=False)   # still healthy
    first = bytes(emu.core.save_raw_state())
    emu.load_state(state_bytes=raw, run_settle_frame=False)
    assert bytes(emu.core.save_raw_state()) == first


def test_provenance_block_complete(emu):
    from collection.provenance import collect_provenance

    pin = collect_provenance(emu, "Emerald-GBAdvance/rom.gba")
    assert len(pin["rom_sha256"]) == 64
    assert pin["fixed_rtc_value"] is not None          # fixed RTC active by default
    assert pin["harness_commit"] not in ("", None)
    assert pin["python"].count(".") == 2
    # M6: uncommitted code + installed-package set are pinned too
    assert isinstance(pin["harness_dirty"], bool)
    if pin["harness_dirty"]:
        assert len(pin["harness_diff_sha256"]) == 64   # dirty runs stay identifiable
    else:
        assert pin["harness_diff_sha256"] is None
    assert len(pin["env_hash"]) == 64                  # sha256 of sorted pip freeze
