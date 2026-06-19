"""On-screen text -> glyph grid (identity_grounded, PLAN §8.3).

The grid carries WHICH glyph sits at each screen position so the model renders text/numbers from
the condition instead of its prior. Built from game STATE + decomp-exact layout (advance == font
glyph width, letterSpacing 0; glyphId == raw charmap byte — both verified against pokeemerald
src/text.c), NOT VRAM OCR.

RESOLUTION — why 2x (4px sub-cells), not the 8px latent grid:
  FONT_NORMAL advances ~6px/glyph, FONT_SMALL ~5px; the VAE latent cell is 8px. Placing glyphs at
  `x//8` collides ~1-in-6 consecutive chars into one cell (a char dropped -> "PROF"->"PRF"), so an
  8px grid is structurally lossy for proportional text. At 4px sub-cells, glyphs of width >=4 land
  in DISTINCT cells (only punctuation runs like "..." collide, ~3%). The encoder sums each 2x2
  block's glyph embeddings (padding_idx 0), so the 8px latent cell still sees the (up to ~2) glyphs
  it actually contains — consistent cell->pixel mapping AND near-lossless. Grid is (2H, 2W)=(40,60).

Sources combined onto one grid (OR; later sources win a contested cell):
  • dialogue / message box   — text.text_state (typing reveal) + finished-box (last_message)   [DONE]
  • battle HUD healthboxes    — nick / level / HP numbers (gHealthboxSpriteIds + layout)        [TODO]
  • FIGHT move menu           — the 4 move names (+ action menu)                                 [TODO]
"""
from __future__ import annotations

import numpy as np

from collection.extractors.font import NARROW_GLYPH_WIDTHS, NORMAL_GLYPH_WIDTHS, SMALL_GLYPH_WIDTHS
from collection.extractors.ram import GBAState
from collection.extractors.text import (G_DISPLAYED_STRING_BATTLE, GSTRINGVAR4, S_TEXT_PRINTERS,
                                        OFF_ACTIVE, OFF_CURRENT_CHAR, _read_span, text_state)
from collection.extractors.ui import window_mask

# latent grid 20x30 @ 8px; glyph grid at SUB x that = 4px sub-cells (see module docstring).
H, W, SUB = 20, 30, 2
GH, GW, CELL = H * SUB, W * SUB, 8 // SUB            # (40, 60), 4px per sub-cell

# charmap control bytes (pokeemerald include/constants/characters.h), needed for reveal-accurate
# placement: the printer's currentChar (== reveal) advances past EVERY byte incl. control codes.
EXT_CTRL = 0xFC          # EXT_CTRL_CODE_BEGIN: 0xFC <subcode> <args...>
PLACEHOLDER = 0xFD       # PLACEHOLDER_BEGIN: 0xFD <id>  (already expanded in composed buffers)
KEYPAD_ICON, EXTRA_SYMBOL = 0xF8, 0xF9              # both: <code> <id>
NEWLINE, PROMPT_SCROLL, PROMPT_CLEAR, EOS = 0xFE, 0xFA, 0xFB, 0xFF

# bytes consumed AFTER the subcode, per ext-ctrl subcode (decomp src/text.c:1256-1289 fall-through).
_EXT_ARGS = {0x04: 3, 0x0B: 2, 0x10: 2}             # COLOR_HIGHLIGHT_SHADOW / PLAY_BGM / PLAY_SE
for _c in (0x01, 0x02, 0x03, 0x05, 0x06, 0x08, 0x0C, 0x0D, 0x0E, 0x11, 0x12, 0x13, 0x14):
    _EXT_ARGS[_c] = 1                               # COLOR/HIGHLIGHT/SHADOW/PALETTE/FONT/PAUSE/...
# all other subcodes (RESET_FONT/PAUSE_UNTIL_PRESS/WAIT_SE/FILL_WINDOW/JPN/ENG/...) take 0 args.

# dialogue (field + battle) message box: FONT_NORMAL, glyph top-left (12, 112) px, lines 16px apart;
# 2 visible lines + scroll. Y0=112 fixed by an OBJECTIVE ink-overlap Y-sweep (field 100% @112-118
# collapsing at 120; battle 96.9% @112-114 vs 89.6% @116+) — an earlier eyeballed 118 was ~1.5 cells
# too low. DLG_X0=12 from the same recorded Birch-lab dialogue (content matches RAM exactly).
DLG_X0, DLG_Y0, DLG_LINE_PX, DLG_VISIBLE_LINES = 12, 112, 16, 2


def _printable(b: int) -> bool:
    """A byte that advances the cursor by a glyph width (i.e. occupies a cell)."""
    return b not in (NEWLINE, PROMPT_SCROLL, PROMPT_CLEAR, EOS, EXT_CTRL, PLACEHOLDER,
                     KEYPAD_ICON, EXTRA_SYMBOL)


def _token_len(raw: bytes, i: int) -> int:
    """Byte length of the token starting at raw[i] (so reveal byte-accounting matches the printer)."""
    b = raw[i]
    if b == EXT_CTRL:
        sub = raw[i + 1] if i + 1 < len(raw) else 0
        return 2 + _EXT_ARGS.get(sub, 0)
    if b in (PLACEHOLDER, KEYPAD_ICON, EXTRA_SYMBOL):
        return 2
    return 1


def _layout_lines(raw: bytes, widths, x0: int, reveal: int | None) -> list[list[tuple[int, int]]]:
    """Walk a charmap string into laid-out lines of (x_px, glyph_byte), honouring proportional
    advance + control codes, stopping at `reveal` bytes (None = whole string). NEWLINE starts a new
    line. PROMPT_SCROLL/PROMPT_CLEAR are PAUSE points: the box keeps showing the current page until
    the player presses on, so we STOP there unless the typewriter has already advanced PAST them
    (reveal beyond) — only then do we scroll (new line) / clear (fresh page)."""
    lines: list[list[tuple[int, int]]] = [[]]
    x, pos, i = x0, 0, 0
    n = len(raw)
    while i < n:
        b = raw[i]
        if b == EOS:
            break
        if reveal is not None and pos >= reveal:
            break
        ln = _token_len(raw, i)
        if b == PROMPT_SCROLL or b == PROMPT_CLEAR:
            if reveal is None or pos + ln >= reveal:    # paused here -> current page is visible
                break
            lines = [[]] if b == PROMPT_CLEAR else lines + [[]]
            x = x0
        elif b == NEWLINE:
            lines.append([]); x = x0
        elif _printable(b):
            lines[-1].append((x, b)); x += widths[b]
        i += ln; pos += ln
    return lines


def _stamp(grid: np.ndarray, lines, y0: int, line_px: int, visible_lines: int) -> None:
    """Render the last `visible_lines` laid-out lines into the 4px-cell grid (top-left anchor).
    Trailing empty lines (a message ending in a newline) are dropped first so a 2-line page shows
    both its lines rather than scrolling the first one off."""
    lines = list(lines)
    while len(lines) > 1 and not lines[-1]:
        lines.pop()
    vis = lines[-visible_lines:] if visible_lines else lines
    for row, line in enumerate(vis):
        cy = (y0 + row * line_px) // CELL
        if not (0 <= cy < GH):
            continue
        for x, b in line:
            cx = x // CELL
            if 0 <= cx < GW and grid[cy, cx] == 0:      # keep-first on sub-cell collision
                grid[cy, cx] = b


TEXTBOX_ROWS = slice(14, 20)                          # bottom band of the BG window mask = msg box


def _box_visible(st: GBAState) -> bool:
    return bool(window_mask(st)[TEXTBOX_ROWS].any())


def _place_dialogue(raw: bytes, reveal: int | None, grid: np.ndarray) -> None:
    lines = _layout_lines(raw, NORMAL_GLYPH_WIDTHS, DLG_X0, reveal)
    _stamp(grid, lines, DLG_Y0, DLG_LINE_PX, DLG_VISIBLE_LINES)


def _has_text(raw: bytes) -> bool:
    end = raw.index(0xFF) if 0xFF in raw else len(raw)
    return any(b not in (0x00, 0xFF) for b in raw[:end])


def _field_message(st: GBAState, rom, grid: np.ndarray) -> None:
    ts = text_state(st, rom)
    if ts is not None and ts.text.strip():            # active printer: full string + typing reveal
        raw = _read_span(st, ts.source, 1000, rom)    # ts.source may be ROM (signs/fishing) — rom-aware
        if raw is not None:
            _place_dialogue(raw, ts.reveal, grid)
        return
    raw = st.bytes(GSTRINGVAR4, 1000)                 # finished box awaiting input: full message
    if _box_visible(st) and _has_text(raw):
        _place_dialogue(raw, None, grid)


def _battle_message(st: GBAState, grid: np.ndarray) -> None:
    """In battle, text_state's back-scan is unreliable (the gDisplayedStringBattle buffer holds
    spurious 0xFF, and healthbox window printers race for the first active slot). Read the battle
    buffer from its KNOWN start; reveal from slot-0's currentChar when it points into the buffer
    (else the box shows a finished message -> full). Gate on the message box being on screen."""
    if st.u16(BG0_Y_ADDR) in (SC_ACTION, SC_MOVE):    # bottom shows the action/move MENU, not a msg
        return
    if not _box_visible(st):
        return
    raw = st.bytes(G_DISPLAYED_STRING_BATTLE, 1000)
    if not _has_text(raw):
        return
    p0 = S_TEXT_PRINTERS
    off = st.u32(p0 + OFF_CURRENT_CHAR) - G_DISPLAYED_STRING_BATTLE
    reveal = off if (st.u8(p0 + OFF_ACTIVE) == 1 and 0 <= off < 1000) else None
    _place_dialogue(raw, reveal, grid)


def _dialogue(st: GBAState, rom, grid: np.ndarray, in_battle: bool) -> None:
    if in_battle:
        _battle_message(st, grid)
    else:
        _field_message(st, rom, grid)


# --- battle HUD healthboxes -------------------------------------------------------------------
# Box origin = gSprites[gHealthboxSpriteIds[battler]] top-left (pos1+pos2+centerToCornerVec, the
# same OAM corner as battle.battle_sprites). VERIFIED by overlay (battle__PETALBURG_WOODS, f1076):
# corner+(16,10) lands on the player nick "MUDKIP", HP "28/ 35" at the tuned x's below, etc. The
# per-element offsets follow decomp src/battle_interface.c (UpdateLvlInHealthbox/UpdateHpText...)
# corrected against the rendered frame (the decomp tile→pixel math for the split HP copy is off).
GHB_SPRITE_IDS = 0x03005D70                          # gHealthboxSpriteIds u8[4] (IWRAM)
_GSPRITES, _STRIDE = 0x02020630, 0x44
CHAR_SLASH = 0xBA
# (nick_dx, lvl_right_dx, text_dy) per side; player also has HP (cur_right, slash, max_right, hp_dy)
_HUD_PLAYER = dict(nick_dx=16, lvl_right=94, dy=10, hp_cur_right=74, hp_slash=76, hp_max_right=101, hp_dy=26)
_HUD_ENEMY = dict(nick_dx=8, lvl_right=86, dy=10)


def _s8(v: int) -> int:
    return v - 256 if v >= 128 else v


def _hb_corner(st: GBAState, sid: int) -> tuple[int, int] | None:
    if sid >= 64:
        return None
    b = _GSPRITES + sid * _STRIDE
    if (st.u16(b + 0x3E) >> 2) & 1:                  # healthbox sprite invisible -> not on screen
        return None
    return (st.s16(b + 0x20) + st.s16(b + 0x24) + _s8(st.u8(b + 0x28)),
            st.s16(b + 0x22) + st.s16(b + 0x26) + _s8(st.u8(b + 0x29)))


def _put_run(grid: np.ndarray, byts, x_px: int, y_px: int, widths) -> None:
    """Place charmap bytes left-to-right from (x_px, y_px), advancing by proportional width."""
    cy = y_px // CELL
    if not (0 <= cy < GH):
        return
    x = x_px
    for b in byts:
        if b not in (0x00, 0xFF):
            cx = x // CELL
            if 0 <= cx < GW and grid[cy, cx] == 0:
                grid[cy, cx] = b
        x += widths[b] if b != 0xFF else 0


def _put_right(grid: np.ndarray, byts, right_px: int, y_px: int, widths) -> None:
    """Right-align a run so it ENDS at right_px (level/HP numbers are right-aligned in-game)."""
    total = sum(widths[b] for b in byts)
    _put_run(grid, byts, right_px - total, y_px, widths)


def _nick(st: GBAState, mon_base: int) -> list[int]:
    raw = st.bytes(mon_base + 0x30, 11)              # BattlePokemon.nickname @0x30, 11 bytes
    return list(raw[:raw.index(0xFF)]) if 0xFF in raw else list(raw)


def _digits(n: int) -> list[int]:
    return [0xA1 + (ord(c) - ord("0")) for c in str(n)]      # charmap '0'=0xA1


def _hud(st: GBAState, grid: np.ndarray) -> None:
    from collection.extractors.battle import (G_BATTLE_MONS, G_BATTLE_TYPE, MON_SIZE, MAX_SPECIES,
                                              N_BATTLERS, in_battle)
    if not in_battle(st):
        return
    if st.u32(G_BATTLE_TYPE) & 1:                    # BATTLE_TYPE_DOUBLE: different layout — skip
        return
    for battler in range(N_BATTLERS):
        cfg = _HUD_PLAYER if battler % 2 == 0 else _HUD_ENEMY
        mon = G_BATTLE_MONS + battler * MON_SIZE
        species = st.u16(mon)
        if not (0 < species <= MAX_SPECIES):
            continue
        corner = _hb_corner(st, st.u8(GHB_SPRITE_IDS + battler))
        if corner is None:
            continue
        cx, cy = corner
        _put_run(grid, _nick(st, mon), cx + cfg["nick_dx"], cy + cfg["dy"], SMALL_GLYPH_WIDTHS)
        _put_right(grid, _digits(st.u8(mon + 0x2A)), cx + cfg["lvl_right"], cy + cfg["dy"], SMALL_GLYPH_WIDTHS)
        if "hp_cur_right" in cfg:                    # singles: only player shows HP numbers
            hp, mxhp = st.u16(mon + 0x28), st.u16(mon + 0x2C)
            _put_right(grid, _digits(hp), cx + cfg["hp_cur_right"], cy + cfg["hp_dy"], SMALL_GLYPH_WIDTHS)
            _put_run(grid, [CHAR_SLASH], cx + cfg["hp_slash"], cy + cfg["hp_dy"], SMALL_GLYPH_WIDTHS)
            _put_right(grid, _digits(mxhp), cx + cfg["hp_max_right"], cy + cfg["hp_dy"], SMALL_GLYPH_WIDTHS)


# --- FIGHT move-select menu -------------------------------------------------------------------
# gMoveNames @ ROM 0x0831977C, 13 bytes/entry (located by searching the ROM for "POUND"=move 1;
# [2]="KARATE CHOP" matches the canonical Gen-3 list). Move-select is detected by the move-menu's
# own type box: gDisplayedStringBattle = "TYPE/<type>" for the whole selection. The 4 move-name
# windows render with FONT_NARROW at fixed on-screen px (decomp battle_bg.c templates + the
# BG0_Y=320 move-select scroll): TL/TR/BL/BR. Verified by overlay on recorded move-select frames.
GMOVENAMES = 0x0831977C
MOVE_NAME_STRIDE = 13
# gBattle_BG0_Y: the battle BG scroll. 0=message, 160=action-select (FIGHT/BAG/...), 320=move-select.
# RELIABLE menu-state signal (gDisplayedStringBattle goes STALE — a finished move-select leaves
# "TYPE/x" there even on the action menu). Found empirically (only EWRAM u16 with value set
# {0,160,320}); ==160 at the action menu, ==320 one frame after picking FIGHT.
BG0_Y_ADDR = 0x02022E16
SC_ACTION, SC_MOVE = 160, 320
MOVE_MENU_POS = ((16, 120), (88, 120), (16, 136), (88, 136))     # move 1..4 top-left px (BG0_Y=320)
# action menu (FONT_NORMAL, BG0_Y=160): FIGHT/BAG top row, POKéMON/RUN bottom; left col x136 right x192
ACTION_ITEMS = (("FIGHT", 136, 120), ("BAG", 192, 120), ("POKéMON", 136, 136), ("RUN", 192, 136))
_TYPE_PREFIX = bytes([0xCE, 0xD3, 0xCA, 0xBF, 0xBA])    # "TYPE/" charmap
TYPE_BOX_POS = (168, 136)                                # move-menu type box (FONT_NARROW)


def _move_name_bytes(st: GBAState, rom, move_id: int) -> list[int]:
    addr = GMOVENAMES + move_id * MOVE_NAME_STRIDE
    if st._env is not None:
        raw = st.bytes(addr, MOVE_NAME_STRIDE)           # live mgba reads ROM directly
    elif rom is not None:
        off = addr - 0x08000000
        raw = rom[off:off + MOVE_NAME_STRIDE]
    else:
        return []
    return list(raw[:raw.index(0xFF)]) if 0xFF in raw else list(raw)


def _encode(s: str) -> list[int]:
    from collection.extractors.text import CHARSET
    inv = {v: k for k, v in CHARSET.items()}
    return [inv[c] for c in s]


def _menu(st: GBAState, rom, grid: np.ndarray) -> None:
    from collection.extractors.battle import G_BATTLE_MONS, in_battle
    if not in_battle(st):
        return
    scroll = st.u16(BG0_Y_ADDR)
    if scroll == SC_MOVE:                                # move-select: player's 4 move names
        for i, (x, y) in enumerate(MOVE_MENU_POS):       # singles -> battler 0
            mv = st.u16(G_BATTLE_MONS + 0x0C + 2 * i)
            if mv != 0:
                _put_run(grid, _move_name_bytes(st, rom, mv), x, y, NARROW_GLYPH_WIDTHS)
        tr = st.bytes(G_DISPLAYED_STRING_BATTLE, 16)     # type box "TYPE/<type>" (the highlighted move)
        if bytes(tr[:5]) == _TYPE_PREFIX:
            end = tr.index(0xFF) if 0xFF in tr else len(tr)
            _put_run(grid, list(tr[:end]), *TYPE_BOX_POS, NARROW_GLYPH_WIDTHS)
    elif scroll == SC_ACTION:                            # action-select: FIGHT/BAG/POKéMON/RUN
        for label, x, y in ACTION_ITEMS:
            _put_run(grid, _encode(label), x, y, NORMAL_GLYPH_WIDTHS)


# --- HP bar fill channel ----------------------------------------------------------------------
# The colored HP bar (48px = 6 tiles). Geometry VERIFIED by pixel-mapping the fill vs RAM hp/maxhp
# (player fill x=corner+48, enemy x=corner+40, both width 48, y=corner+19; fill width == frac*48 to
# ±1px across frames). hp_fill is at the LATENT 20x30 (8px) res: each bar-overlapping cell gets the
# fraction of itself that is filled (frac from RAM, exact) — a spatial green-length the model renders.
BAR_W, BAR_Y_REL = 48, 19
BAR_X_REL = (48, 40)                                  # player (even battler) / enemy (odd)


def hp_fill_grid(st: GBAState) -> np.ndarray:
    """(H, W)=(20, 30) float HP-bar fill (0 = empty/none). Per-cell fraction-filled at the bar cells."""
    from collection.extractors.battle import (G_BATTLE_MONS, G_BATTLE_TYPE, MON_SIZE, MAX_SPECIES,
                                              N_BATTLERS, in_battle)
    grid = np.zeros((H, W), np.float32)
    if not in_battle(st) or (st.u32(G_BATTLE_TYPE) & 1):     # none / double (different layout)
        return grid
    for battler in range(N_BATTLERS):
        mon = G_BATTLE_MONS + battler * MON_SIZE
        species, mxhp = st.u16(mon), st.u16(mon + 0x2C)
        if not (0 < species <= MAX_SPECIES) or mxhp == 0:
            continue
        corner = _hb_corner(st, st.u8(GHB_SPRITE_IDS + battler))
        if corner is None:
            continue
        frac = max(0.0, min(st.u16(mon + 0x28) / mxhp, 1.0))
        bx = corner[0] + BAR_X_REL[battler % 2]
        fr = bx + frac * BAR_W                               # right edge of the filled region
        cy = (corner[1] + BAR_Y_REL) // 8
        if not (0 <= cy < H):
            continue
        for c in range(max(0, bx // 8), min(W, (bx + BAR_W + 7) // 8)):
            cov = max(0.0, min(c * 8 + 8, fr) - max(c * 8, bx))     # filled px inside this 8px cell
            if cov > 0:
                grid[cy, c] = max(grid[cy, c], cov / 8.0)
    return grid


def glyph_grid(st: GBAState, rom: bytes | None = None) -> np.ndarray:
    """The (GH, GW)=(40, 60) uint8 glyph grid (charmap glyph id per 4px sub-cell, 0 = empty).
    Single source of truth for on-screen text; the encoder sums each 2x2 block to the 20x30 latent."""
    from collection.extractors.battle import in_battle
    grid = np.zeros((GH, GW), np.uint8)
    ib = bool(in_battle(st))
    _dialogue(st, rom, grid, in_battle=ib)
    _hud(st, grid)
    _menu(st, rom, grid)
    return grid
