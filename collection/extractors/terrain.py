"""Terrain — the live metatile grid (`gBackupMapLayout`) for identity-based terrain conditioning.

The game keeps the CURRENT map (with a 7-tile border margin all around) as a u16 buffer in ewram,
described by an iwram struct `{ s32 width; s32 height; u16 *map; }` — `gBackupMapLayout`, pinned for
this build at **0x03005DC0** by cross-frame consensus scan (present on 38/38 sampled overworld
frames with map-consistent dims; the other scan candidates were constant-dim stack noise).
`locate_layout` remains as the discovery/confirmation tool; `terrain()` reads the pinned struct.

Buffer semantics (validated in Phase 0 + ROM cross-validation of all recorded maps):
    width  = map_width + 15,  height = map_height + 14      (margin; height is NOT symmetric —
        41/43 recorded maps match their ROM layout dims exactly under (w−15, h−14))
    index(x, y) = (x + 7) + (y + 7) * width                 (x, y in PLAYER map coords)
    word: bits 0-9 metatile id · 10-11 collision · 12-15 elevation

Metatile ids are tileset-local identities (0-511 primary, 512+ secondary) — the embedding keys of
the A′ terrain condition; the map→tileset table comes from the ROM map-groups walk (`tilesets.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from collection.extractors.ram import GBAState

IWRAM_BASE, IWRAM_SIZE = 0x03000000, 0x8000
EWRAM_BASE, EWRAM_END = 0x02000000, 0x02040000
BORDER = 7                                   # margin tiles on each side
SCREEN_W, SCREEN_H = 15, 10                  # visible metatiles (240×160 px / 16)

MAP_BANK, MAP_NUMBER = 0x020322E4, 0x020322E5   # validated-by-data (source of semantic map names)
BACKUP_LAYOUT = 0x03005DC0                       # gBackupMapLayout (pinned; see module docstring)


@dataclass(frozen=True)
class Terrain:
    """The current map's live metatile grid (border included) + identity."""
    map_group: int
    map_num: int
    width: int                               # buffer width  = map width  + 15
    height: int                              # buffer height = map height + 14 (ROM-cross-validated:
    grid: np.ndarray                         #   41/43 recorded maps match ROM layouts exactly under
                                             #   (w-15, h-14); the old -15 height was off by one)

    @property
    def map_width(self) -> int:
        return self.width - 15

    @property
    def map_height(self) -> int:
        return self.height - 14

    def metatile_ids(self) -> np.ndarray:
        return self.grid & 0x3FF

    def window(self, tl_x: int, tl_y: int, w: int = SCREEN_W, h: int = SCREEN_H) -> np.ndarray:
        """(h, w) raw words at a top-left given in PLAYER map coords (border handled internally).
        The border margin makes edge windows valid without clamping for tl ≥ -7."""
        x0, y0 = tl_x + BORDER, tl_y + BORDER
        assert 0 <= x0 and x0 + w <= self.width and 0 <= y0 and y0 + h <= self.height, \
            f"window ({tl_x},{tl_y}) out of buffer {self.width}x{self.height}"
        return self.grid[y0:y0 + h, x0:x0 + w]


def locate_layout(st: GBAState) -> list[tuple[int, int, int]]:
    """Scan iwram for the BackupMapLayout struct -> [(width, height, buffer_addr), ...] candidates.
    Filters hard (plausible dims, pointer into ewram, buffer in bounds); the validation suite
    asserts exactly ONE candidate survives across the dataset before anyone trusts `terrain()`."""
    iw = np.frombuffer(st.bytes(IWRAM_BASE, IWRAM_SIZE), np.uint8)
    u32 = iw.view("<u4")                                       # iwram is 4-aligned
    out = []
    for i in range(len(u32) - 2):
        w, h, ptr = int(u32[i]), int(u32[i + 1]), int(u32[i + 2])
        if not (15 <= w <= 250 and 15 <= h <= 250):
            continue
        if not (EWRAM_BASE <= ptr and ptr + 2 * w * h <= EWRAM_END):
            continue
        out.append((w, h, ptr))
    return out


def terrain(st: GBAState) -> Terrain | None:
    """The live map grid via the pinned gBackupMapLayout; None when the struct is not in a sane
    state (title/intro/teardown frames)."""
    w, h = st.u32(BACKUP_LAYOUT), st.u32(BACKUP_LAYOUT + 4)
    ptr = st.u32(BACKUP_LAYOUT + 8)
    if not (15 <= w <= 250 and 15 <= h <= 250 and
            EWRAM_BASE <= ptr and ptr + 2 * w * h <= EWRAM_END):
        return None
    grid = st.array_u16(ptr, w * h).reshape(h, w).copy()
    return Terrain(map_group=st.u8(MAP_BANK), map_num=st.u8(MAP_NUMBER),
                   width=w, height=h, grid=grid)
