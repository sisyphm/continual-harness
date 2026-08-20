"""Entities (player + NPCs) — the corrected `gObjectEvents` extractor.

Phase-0 established the TRUE record layout for this build (validated against live frames; the
legacy `world_model_sink.extract_objects` used stride 68 / gfx@+0x03 and produced garbage
dataset-wide). BASE CORRECTION (2026-07-02): Phase-0's base 0x02037230 was **8 slots (0x120)
low** — it probed 8 phantom slots inside `sBackupMapData`'s tail (always filtered by the
structural validity check; no phantoms observed corpus-wide) and could NOT see true slots 8–15,
so real NPCs were missing from conditions on busy maps (e.g. Rustboro: an NPC absent from
conditions on ~44% of sampled tour frames). The true base 0x02037350 is the SHA-exact decomp
catalog address, verified empirically (player record at true slot 0, coords == saveblock).
Conditions extracted BEFORE this fix under-report entities: re-extract
(`precompute_conditions --force`) and re-train the renderer at the next collection round.

    gObjectEvents @ 0x02037350 — 16 records × 0x24 (36) bytes (pokeemerald `struct ObjectEvent`):
      +0x00  u32  bitfield        (bit0 NOT usable as `active`: 0xFF-cleared slots read 1)
      +0x04  u8   spriteId
      +0x05  u8   graphicsId      (the identity the model learns appearance for)
      +0x06  u8   movementType
      +0x08  u8   localId         (0xFF = the player's record)
      +0x09  u8   mapNum   +0x0A u8 mapGroup   +0x0B elevation nibbles
      +0x0C/+0x10/+0x14  Coords16 initial/current/previous — map tile coords **+ MAP_OFFSET (7)**
                  (the pokeemerald convention; established by visual overlay against rendered
                  frames — NPC boxes land exactly on sprites only after subtracting 7. The
                  saveblock player position has NO such offset.)
      +0x18  u8   facingDirection (low nibble) | movementDirection (high nibble)
                  direction constants: 1=DOWN(S) 2=UP(N) 3=LEFT(W) 4=RIGHT(E)

Slot validity is STRUCTURAL (no trustworthy active bit): not 0xFF-cleared, plausible graphicsId,
sane coords. Source-of-truth split for the PLAYER (each validated independently):
  • position  — the saveblock path (bit-exact vs every recorded semantic x/y);
  • facing    — the player OBJECT RECORD's facing nibble. NOT the saveblock byte (goes stale) and
    NOT semantic.jsonl's facing (the collector's input-tracker: pixel-adjudicated WRONG after
    warps/cutscene turns — door exits, forced turns; a v1 data-stream bug found in Phase 1).
"""

from __future__ import annotations

from dataclasses import dataclass

from collection.extractors.ram import GBAState

OBJ_BASE, OBJ_SIZE, OBJ_N = 0x02037350, 0x24, 16   # true gObjectEvents (base corrected 2026-07-02)
SAVEBLOCK1_PTR = 0x03005D8C            # iwram pointer -> SaveBlock1 (DMA-shifted; validated Phase 0)
PLAYER_LOCALID = 0xFF
MAP_OFFSET = 7                         # object coords carry the border offset; player saveblock doesn't

# Sub-tile screen positions (discovered by differential scan, verified on data):
#   gSprites @0x02020630 (0x44/sprite): pos1 = s16 (x @+0x20, y @+0x22) in WORLD pixels
#   camera offset: X @0x03005DEC, Y @0x03005DE8 (s16, next to gBackupMapLayout)
#   screen anchor = pos1 + cam — for the player this is EXACTLY (120, 112) on every frame
#   (OAM top-left of a 16x32 char sprite = anchor + (-8, -56), constant per sprite shape).
# This carries mid-step sub-tile motion — the largest chunk of the Phase-0 "unexplained" tail.
GSPRITES, SPRITE_SIZE = 0x02020630, 0x44
CAM_X, CAM_Y = 0x03005DEC, 0x03005DE8

# GBA overworld direction constants (facing nibble); 0 = none/unset
DIRECTIONS = {1: "DOWN", 2: "UP", 3: "LEFT", 4: "RIGHT"}


@dataclass(frozen=True)
class Entity:
    """One on-map object (NPC or player record): identity + tile coords + sub-tile screen anchor."""
    slot: int
    graphics_id: int
    local_id: int
    x: int
    y: int
    prev_x: int
    prev_y: int
    facing: str | None                 # None if the nibble is 0/garbage
    moving_dir: str | None             # direction currently being walked (None = standing)
    movement_type: int                 # the game's wander/look-around behavior id
    sprite_id: int
    screen_x: int                      # screen anchor = gSprites.pos1 + camera (player ≡ (120,112));
    screen_y: int                      # pixel-accurate, carries mid-step sub-tile motion

    @property
    def is_player(self) -> bool:
        return self.local_id == PLAYER_LOCALID

    @property
    def mid_step(self) -> bool:
        """True while walking between tiles (current != previous) — the sub-tile animation phase
        exists during these frames; exact pixel offset comes from the gSprites follow-up."""
        return (self.x, self.y) != (self.prev_x, self.prev_y)


def _valid(st: GBAState, o: int) -> bool:
    """Slot validity at the TRUE base (2026-07-02): the decomp `ObjectEvent.active` bit (bit0 of
    the leading u32 bitfield) IS trustworthy here — the old "bit0 unusable" Phase-0 rule was an
    artifact of probing `sBackupMapData` bytes at the shifted base. The active gate is required:
    despawned NPCs keep plausible bytes (ghost splats under the structural-only rule — observed:
    Rustboro stale slots), and zero-cleared slots fake the player pattern (gfx=0, local=0xFF).
    Structural sanity kept as belt-and-braces."""
    if not (st.u32(o) & 1):                                 # ObjectEvent.active
        return False
    gfx = st.u8(o + 0x05)
    if not (0 <= gfx <= 239):
        return False
    x, y = st.s16(o + 0x10), st.s16(o + 0x12)
    if not (0 <= x <= 999 and 0 <= y <= 999):
        return False
    return gfx > 0 or st.u8(o + 0x08) == PLAYER_LOCALID     # gfx 0 only meaningful on the player rec


def camera_px(st: GBAState) -> tuple[int, int]:
    """The global sprite/world camera offset (screen = world_pos1 + camera). Its fractional part
    is the sub-tile scroll phase the terrain condition needs."""
    return st.s16(CAM_X), st.s16(CAM_Y)


def entities(st: GBAState) -> list[Entity]:
    """All valid object records (NPCs + the player's record if its coords pass validity)."""
    cam_x, cam_y = camera_px(st)
    out = []
    for slot in range(OBJ_N):
        o = OBJ_BASE + slot * OBJ_SIZE
        if not _valid(st, o):
            continue
        nib = st.u8(o + 0x18)
        spr = st.u8(o + 0x04)
        sb = GSPRITES + spr * SPRITE_SIZE
        out.append(Entity(
            slot=slot,
            graphics_id=st.u8(o + 0x05),
            local_id=st.u8(o + 0x08),
            x=st.s16(o + 0x10) - MAP_OFFSET, y=st.s16(o + 0x12) - MAP_OFFSET,
            prev_x=st.s16(o + 0x14) - MAP_OFFSET, prev_y=st.s16(o + 0x16) - MAP_OFFSET,
            facing=DIRECTIONS.get(nib & 0xF),
            moving_dir=DIRECTIONS.get(nib >> 4),
            movement_type=st.u8(o + 0x06),
            sprite_id=spr,
            screen_x=st.s16(sb + 0x20) + cam_x,
            screen_y=st.s16(sb + 0x22) + cam_y,
        ))
    return out


def npcs(st: GBAState) -> list[Entity]:
    return [e for e in entities(st) if not e.is_player]


def player_pace_read(env) -> tuple[bool, str | None] | None:
    """Per-frame poll for condition-based pacing (W33 §3.5): the player record's
    (mid_step, facing) in ONE bus read of the whole gObjectEvents block — no Entity
    construction, no gSprites/camera reads (the pacing loop runs this every frame).
    mid_step is Entity.mid_step verbatim: current != previous tile coords while the
    sub-tile walk animation runs. Returns None when no valid player record exists
    (map-load blackout / intro frames) — callers treat that as `not mid-step`."""
    raw = GBAState.from_env(env).bytes(OBJ_BASE, OBJ_SIZE * OBJ_N)
    for slot in range(OBJ_N):
        o = slot * OBJ_SIZE
        if not (raw[o] & 1) or raw[o + 0x08] != PLAYER_LOCALID:   # active + player localId
            continue
        cur = raw[o + 0x10:o + 0x14]                              # Coords16 current (x, y)
        prev = raw[o + 0x14:o + 0x18]                             # Coords16 previous (x, y)
        return cur != prev, DIRECTIONS.get(raw[o + 0x18] & 0xF)
    return None


def player_state(st: GBAState) -> dict:
    """Player position from the saveblock (bit-exact vs recorded semantics); facing + movement from
    the player's object record (the rendered truth — see module docstring). Saveblock facing is the
    fallback when the record is absent (intro/teardown frames)."""
    sb = st.u32(SAVEBLOCK1_PTR)
    x, y = st.u16(sb + 0x00), st.u16(sb + 0x02)
    rec = next((e for e in entities(st) if e.is_player), None)
    if rec is not None and rec.facing is not None:
        facing = rec.facing
    else:
        raw = st.u8(sb + 0x04)                             # 0=S 1=N 2=W 3=E (saveblock encoding)
        facing = ("DOWN", "UP", "LEFT", "RIGHT")[raw] if raw < 4 else "DOWN"
    return {"x": x, "y": y, "facing": facing,
            "moving_dir": rec.moving_dir if rec else None,
            "movement_type": rec.movement_type if rec else None}
