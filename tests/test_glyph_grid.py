"""Pure-logic tests for the identity_grounded glyph grid (collection.extractors.glyph).

These guard the text-layout semantics that an objective ink-overlap audit on recorded frames pinned
down (dialogue 89-100%, HUD 97%, move menu 100%). They run without recorded data — the per-frame
extraction is validated separately against RAM ground truth in the audit, but the control-code /
page-scroll logic below is where the subtle bugs live (e.g. PROMPT_CLEAR was once a fatal box-wipe)."""
import numpy as np

from collection.extractors.font import NORMAL_GLYPH_WIDTHS
from collection.extractors import glyph as G

CM = {" ": 0x00, "A": 0xBB, "B": 0xBC, "C": 0xBD}     # a few charmap bytes
NL, CLEAR, SCROLL, EOS, EXT = 0xFE, 0xFB, 0xFA, 0xFF, 0xFC


_INV = {v: k for k, v in CM.items()}


def _flat(lines):
    return "".join(_INV.get(b, "?") for line in lines for _x, b in line)


def test_token_len_ext_ctrl():
    # EXT_CTRL_CODE_BEGIN + subcode(+args): COLOR(0x01)=1 arg, PLAY_SE(0x10)=2, CHS(0x04)=3, FONT done
    assert G._token_len(bytes([EXT, 0x01, 0x02]), 0) == 3        # 0xFC 01 <arg>
    assert G._token_len(bytes([EXT, 0x10, 0, 0]), 0) == 4        # 0xFC 10 <2 args>
    assert G._token_len(bytes([EXT, 0x04, 0, 0, 0]), 0) == 5     # 0xFC 04 <3 args>
    assert G._token_len(bytes([EXT, 0x07]), 0) == 2              # RESET_FONT: 0 args
    assert G._token_len(bytes([0xBB]), 0) == 1                   # plain glyph
    assert G._token_len(bytes([0xFD, 0x01]), 0) == 2            # placeholder


def test_layout_basic_and_newline():
    raw = bytes([CM["A"], CM["B"], NL, CM["C"], EOS])
    lines = G._layout_lines(raw, NORMAL_GLYPH_WIDTHS, 0, None)
    assert _flat(lines) == "ABC"
    assert len(lines) == 2 and len(lines[0]) == 2 and len(lines[1]) == 1
    # proportional advance: 'A','B' are width 6 -> second glyph at x=6
    assert lines[0][1][0] == NORMAL_GLYPH_WIDTHS[CM["A"]]


def test_reveal_gates_typing():
    raw = bytes([CM["A"], CM["B"], CM["C"], EOS])
    assert _flat(G._layout_lines(raw, NORMAL_GLYPH_WIDTHS, 0, 2)) == "AB"   # 2 bytes shown
    assert _flat(G._layout_lines(raw, NORMAL_GLYPH_WIDTHS, 0, 0)) == ""


def test_prompt_clear_is_a_pause_not_a_wipe():
    # the regression: a finished message ending in NEWLINE + PROMPT_CLEAR must STILL show its page,
    # not be wiped to empty. reveal=None => paused at the clear => current page visible.
    raw = bytes([CM["A"], CM["B"], NL, CM["C"], CLEAR, EOS])
    assert _flat(G._layout_lines(raw, NORMAL_GLYPH_WIDTHS, 0, None)) == "ABC"
    # but if typing has advanced PAST the clear, the box clears to a fresh page
    raw2 = bytes([CM["A"], CLEAR, CM["B"], EOS])
    assert _flat(G._layout_lines(raw2, NORMAL_GLYPH_WIDTHS, 0, 4)) == "B"


def test_stamp_drops_trailing_empty_line():
    # 2-line page + trailing newline -> both real lines visible (no scroll-off of line 0)
    lines = [[(0, CM["A"])], [(0, CM["B"])], []]
    grid = np.zeros((G.GH, G.GW), np.uint8)
    G._stamp(grid, lines, 112, 16, 2)
    assert int((grid > 0).sum()) == 2


def test_put_right_aligns():
    grid = np.zeros((G.GH, G.GW), np.uint8)
    G._put_right(grid, G._digits(11), 96, 40, NORMAL_GLYPH_WIDTHS)   # ends at px 96
    cols = sorted(c for c in range(G.GW) if grid[40 // G.CELL, c])
    assert cols and cols[-1] * G.CELL < 96 and cols[-1] * G.CELL >= 96 - 16
