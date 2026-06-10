"""UI geometry + effects + mode — the remaining A′ condition groups.

WINDOW MASK — which screen cells are covered by UI windows (textboxes, menus, popups). Emerald
draws all of these on BG0; occupancy of the BG0 tilemap (vs its dominant filler tile) gives a
binary mask at 8-px cells — a (20, 30) grid that aligns 1:1 with the model's latent cells. This is
identity-free geometry ("a window covers these cells"), not content: text content rides in the
text condition; menu chrome appearance is learned.

EFFECTS — the blend/fade registers (BLDCNT/BLDALPHA/BLDY) straight from the io block: the whole
fade/flash vocabulary of the GBA in three u16s.

MODE — the game's own screen id (gMain.callback2), mapped to the Phase-0 audit taxonomy.
"""

from __future__ import annotations

import numpy as np

from collection.extractors.ram import GBAState

VRAM_BASE, IO_BASE = 0x06000000, 0x04000000
SCREEN_ROWS, SCREEN_COLS = 20, 30            # 8-px cells on the 160x240 screen == the latent grid

# cb2 -> mode label (Phase-0 empirical taxonomy; see audit cb2 contact sheets)
MODE_OVERWORLD, MODE_BATTLE, MODE_TRANSITION, MODE_INTRO, MODE_OTHER = 0, 1, 2, 3, 4
_CB2_MODE = {0x8085E5D: MODE_OVERWORLD,
             0x8085E51: MODE_TRANSITION,                 # overworld during battle-wipe / map load
             0x8038421: MODE_BATTLE,
             0x802F6B1: MODE_INTRO}
_CB2_BATTLE_AUX = {0x81AAD5D, 0x8036FAD, 0x80A933D, 0x813E3A5, 0x81AAD8D}   # black init/teardown
CB2_ADDR = 0x030022C4


def mode(st: GBAState) -> int:
    cb2 = st.u32(CB2_ADDR)
    if cb2 in _CB2_MODE:
        return _CB2_MODE[cb2]
    if cb2 in _CB2_BATTLE_AUX:
        return MODE_BATTLE
    return MODE_OTHER


def window_mask(st: GBAState) -> np.ndarray:
    """(20, 30) bool — BG0 cells covered by a UI window. BG0 scroll is applied (it is 0 in all
    UI modes we condition; applying it keeps the mask correct regardless)."""
    io = np.frombuffer(st.bytes(IO_BASE, 0x60), "<u2")
    bg0cnt = int(io[0x08 // 2])
    hofs, vofs = int(io[0x10 // 2]) & 0x1FF, int(io[0x12 // 2]) & 0x1FF
    screenbase = ((bg0cnt >> 8) & 0x1F) * 0x800
    tm = np.frombuffer(st.bytes(VRAM_BASE + screenbase, 0x800), "<u2").reshape(32, 32)
    # filler = the dominant entry (0x0000 or a blank tile id depending on mode)
    vals, counts = np.unique(tm, return_counts=True)
    filler = vals[counts.argmax()]
    occ = tm != filler
    rows = (np.arange(SCREEN_ROWS) + (vofs >> 3)) & 31
    cols = (np.arange(SCREEN_COLS) + (hofs >> 3)) & 31
    return occ[np.ix_(rows, cols)]


def effects(st: GBAState) -> tuple[int, int, int]:
    """(BLDCNT, BLDALPHA, BLDY) — the fade/blend state."""
    io = np.frombuffer(st.bytes(IO_BASE, 0x60), "<u2")
    return int(io[0x50 // 2]), int(io[0x52 // 2]), int(io[0x54 // 2])
