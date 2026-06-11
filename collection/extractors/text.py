"""Dialogue text — the game's text-printer state, decoded.

All addresses are VANILLA US Emerald (this build matches vanilla for every struct validated so
far) and were re-proven on recorded data before use:

  gStringVar4    0x02021FC4 — the composed-message buffer; decoding it at a known dialogue frame
                 reproduced the on-screen sentence verbatim (charset + address proven together).
  sTextPrinters  0x020201B0 — printer array, 0x24 bytes each:
                   +0x00 u32  printerTemplate.currentChar (ADVANCES while typing — the reveal)
                   +0x1B u8   active (empirically: 1 on every textbox-visible frame, 0 otherwise)
  sTempTextPrinter 0x0202018C — holds the last-added printer's ORIGINAL string start.

The string a printer is typing may live in ewram (composed text) or in ROM (static text, signs).
`text_state` decodes from the string START (back-scan to the previous 0xFF terminator) so the
condition carries the full text plus `reveal` = how much the typewriter has shown — exactly the
(string, reveal-count) pair the A′ text condition needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from collection.extractors.ram import GBAState

S_TEXT_PRINTERS = 0x020201B0
PRINTER_SIZE, N_PRINTERS = 0x24, 4         # slots beyond 3 unused in our modes
OFF_CURRENT_CHAR, OFF_ACTIVE = 0x00, 0x1B
GSTRINGVAR4 = 0x02021FC4
EWRAM_BASE, EWRAM_END = 0x02000000, 0x02040000
ROM_BASE, ROM_END = 0x08000000, 0x0A000000
MAX_TEXT = 1000                            # longest message we ever decode/back-scan

# Gen-3 western charset (validated: decodes recorded dialogue to the on-screen text verbatim).
CHARSET = {0x00: " ", 0x1B: "é", 0xAB: "!", 0xAC: "?", 0xAD: ".", 0xAE: "-", 0xB0: "…",
           0xB1: "“", 0xB2: "”", 0xB3: "‘", 0xB4: "'", 0xB5: "♂", 0xB6: "♀", 0xB8: ",",
           0xBA: "/", 0xF0: ":"}
CHARSET.update({0xA1 + i: chr(ord("0") + i) for i in range(10)})
CHARSET.update({0xBB + i: chr(ord("A") + i) for i in range(26)})
CHARSET.update({0xD5 + i: chr(ord("a") + i) for i in range(26)})
CTRL = {0xFE: "\n", 0xFB: "\n", 0xFA: "\n", 0xFC: "", 0xFD: "?"}   # newline-ish / ctl / var marker


def decode(raw: bytes) -> str:
    """Gen-3 bytes -> str (stops at the 0xFF terminator)."""
    out = []
    for c in raw:
        if c == 0xFF:
            break
        out.append(CHARSET.get(c) or CTRL.get(c, ""))
    return "".join(out)


@dataclass(frozen=True)
class TextState:
    text: str                              # the full message being shown/typed
    reveal: int                            # bytes of it consumed by the typewriter so far
    source: int                            # address of the string start (ewram or ROM)


def _read_span(st: GBAState, addr: int, n: int, rom: bytes | None) -> bytes | None:
    if EWRAM_BASE <= addr and addr + n <= EWRAM_END:
        return st.bytes(addr, n)
    if ROM_BASE <= addr < ROM_END:
        if st._env is not None:
            return st.bytes(addr, n)                       # live mgba reads ROM directly
        if rom is not None:
            off = addr - ROM_BASE
            return rom[off:off + n]
    return None


G_DISPLAYED_STRING_BATTLE = 0x02022E2C     # the battle UI's own message buffer (validated: holds
                                           # the exact on-screen battle text while gStringVar4
                                           # holds a STALE overworld string during battles)


def last_message(st: GBAState, in_battle: bool = False) -> str:
    """The most recent composed message for the finished-box state (no active printer).
    Overworld dialogue persists in gStringVar4; BATTLE text lives in gDisplayedStringBattle —
    using gStringVar4 in battle conditions ~30% of the corpus on wrong text (caught pre-training)."""
    addr = G_DISPLAYED_STRING_BATTLE if in_battle else GSTRINGVAR4
    return decode(st.bytes(addr, MAX_TEXT))


def text_state(st: GBAState, rom: bytes | None = None) -> TextState | None:
    """The active printer's full string + reveal position, or None when no printer is running
    (no textbox, or a finished box awaiting input — see `last_message` for the latter).
    `rom` (the ROM file's bytes) enables decoding ROM-resident static text on recorded blobs."""
    for slot in range(N_PRINTERS):
        base = S_TEXT_PRINTERS + slot * PRINTER_SIZE
        if st.u8(base + OFF_ACTIVE) != 1:
            continue
        cur = st.u32(base + OFF_CURRENT_CHAR)
        # back-scan to the string start: the byte before it is a 0xFF terminator (or buffer base)
        lo = max(cur - MAX_TEXT, EWRAM_BASE if cur < EWRAM_END else ROM_BASE)
        span = _read_span(st, lo, cur - lo, rom)
        if span is None:
            continue
        term = span.rfind(b"\xff")
        start = lo + term + 1 if term >= 0 else lo
        body = _read_span(st, start, MAX_TEXT, rom)
        if body is None:
            continue
        txt = decode(body)
        # normalize away buffer fill: 0x00 doubles as 'space', so zero-fill (plus the occasional
        # junk byte) decodes as leading whitespace — no real message opens with it
        lead = len(txt) - len(txt.lstrip(" "))
        if lead:
            txt, start = txt[lead:], start + lead
        if not txt.strip():
            continue                                      # printer alive but no content (UI clears)
        return TextState(text=txt, reveal=max(min(cur - start, len(txt)), 0), source=start)
    return None
