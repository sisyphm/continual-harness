"""Full-PPU render-state extraction + a software renderer used as a *completeness
verifier*: if rendering the extracted state reproduces the emulator's frame, the state is
a sufficient statistic for the pixels ("nothing could be added"). The same extractor is
the maximal dataset we store; every conditioning rung is an offline view/mask of it.

First pass: GBA video mode 0 (tiled), 4bpp BG + 1D-mapped 4bpp sprites, BG size 0.
Window / blend / mosaic effects are captured (registers) but not yet applied in the
renderer — the diff localizes where they matter.
"""

from __future__ import annotations

import numpy as np
from mgba._pylib import ffi


# ---- extraction -------------------------------------------------------------

# PPU I/O registers (byte offset within 0x0400_0000). Read from the io shadow because
# scroll/BLDY are write-only on the bus.
_REGS = {
    "DISPCNT": 0x00, "BG0CNT": 0x08, "BG1CNT": 0x0A, "BG2CNT": 0x0C, "BG3CNT": 0x0E,
    "BG0HOFS": 0x10, "BG0VOFS": 0x12, "BG1HOFS": 0x14, "BG1VOFS": 0x16,
    "BG2HOFS": 0x18, "BG2VOFS": 0x1A, "BG3HOFS": 0x1C, "BG3VOFS": 0x1E,
    "WIN0H": 0x40, "WIN1H": 0x42, "WIN0V": 0x44, "WIN1V": 0x46,
    "WININ": 0x48, "WINOUT": 0x4A, "MOSAIC": 0x4C, "BLDCNT": 0x50,
    "BLDALPHA": 0x52, "BLDY": 0x54,
}


# Full machine render-state blocks, in serialization order (name, byte size). This is the
# maximal sufficient statistic: VRAM/OAM/palette + ALL I/O registers + WRAM (EWRAM/IWRAM).
# WRAM holds HDMA source buffers (so per-scanline effects are reconstructable) and all
# semantic state (object events/party/flags) for the reduced rungs. Everything else (the
# regs dict, the per-scanline HDMA overrides) is *derived* from these blocks offline.
BLOCK_SIZES = [("io", 1024), ("palette", 1024), ("oam", 1024),
               ("vram", 98304), ("ewram", 262144), ("iwram", 32768)]


def _regs_from_io(io_bytes: bytes) -> dict:
    u = np.frombuffer(io_bytes, "<u2")
    return {name: int(u[off >> 1]) for name, off in _REGS.items()}


def _hblank_from_io(io_bytes: bytes, ewram: bytes, iwram: bytes) -> dict:
    """Per-scanline BG scroll from HBlank-timed DMA (e.g. the battle-intro wipe): the
    end-of-frame register is reset, so we read the DMA's source buffer (one value per
    scanline) from WRAM. Returns {reg_offset(0x10..0x1E): [160 per-scanline u16]}."""
    u = np.frombuffer(io_bytes, "<u2")
    overrides = {}
    for ch in (0xB0, 0xBC, 0xC8, 0xD4):          # DMA0-3 control blocks in I/O space
        cnt = int(u[(ch + 0xA) >> 1])
        if not ((cnt >> 15) & 1) or ((cnt >> 12) & 3) != 2:   # enabled + HBlank timing
            continue
        dst = (int(u[(ch + 4) >> 1]) | (int(u[(ch + 6) >> 1]) << 16)) - 0x04000000
        if 0x10 <= dst <= 0x1E and dst % 2 == 0:              # a BG scroll register
            src = int(u[ch >> 1]) | (int(u[(ch + 2) >> 1]) << 16)
            region, mask = (ewram, 0x3FFFF) if (src >> 24) == 2 else (iwram, 0x7FFF)
            b = src & mask
            buf = region[b:b + 320]
            overrides[dst] = [buf[i] | (buf[i + 1] << 8) for i in range(0, min(320, len(buf) - 1), 2)]
    return overrides


def state_from_blocks(blocks: dict) -> dict:
    """Build a render-ready state dict (raw blocks + derived regs/hblank) — used by both
    live extraction and offline reconstruction from the stored stream."""
    state = dict(blocks)
    state["regs"] = _regs_from_io(blocks["io"])
    state["hblank_scroll"] = _hblank_from_io(blocks["io"], blocks["ewram"], blocks["iwram"])
    return state


def extract_full_ppu_state(env) -> dict:
    gba = ffi.cast("struct GBA *", env.core._core.board)
    blocks = {
        "io": bytes(ffi.buffer(gba.memory.io, 1024)),    # ALL I/O registers (PPU+DMA+timers+…)
        "palette": bytes(env._get_memory_region(0x5)),   # 1 KB: 512 BGR555 colors
        "oam": bytes(env._get_memory_region(0x7)),       # 1 KB: 128 sprites
        "vram": bytes(env._get_memory_region(0x6)),      # 96 KB: tilemaps + BG/OBJ tiles
        "ewram": bytes(env._get_memory_region(0x2)),     # 256 KB: HDMA buffers + game state
        "iwram": bytes(env._get_memory_region(0x3)),     # 32 KB
    }
    return state_from_blocks(blocks)


# ---- helpers ----------------------------------------------------------------

def _s16(v: int) -> int:
    return v - 65536 if v >= 32768 else v


def palette_rgb(palette: bytes) -> np.ndarray:
    """512 BGR555 colors -> (512,3) uint8 RGB (5->8 bit with replication)."""
    p = np.frombuffer(palette, dtype="<u2").astype(np.uint32)
    r = (p & 0x1F); g = (p >> 5) & 0x1F; b = (p >> 10) & 0x1F
    to8 = lambda c: ((c << 3) | (c >> 2)).astype(np.uint8)
    return np.stack([to8(r), to8(g), to8(b)], axis=-1)


def _decode_tiles_4bpp(vram: bytes, base: int, count: int) -> np.ndarray:
    """Decode `count` 4bpp 8x8 tiles starting at byte `base` -> (count,8,8) uint8 indices."""
    raw = np.frombuffer(vram[base:base + count * 32], dtype=np.uint8)
    if raw.size < count * 32:
        raw = np.concatenate([raw, np.zeros(count * 32 - raw.size, np.uint8)])
    d = raw.reshape(count, 8, 4)
    out = np.empty((count, 8, 8), np.uint8)
    out[:, :, 0::2] = d & 0xF
    out[:, :, 1::2] = d >> 4
    return out


_SPRITE_DIM = {  # (shape,size) -> (w_tiles, h_tiles)
    (0, 0): (1, 1), (0, 1): (2, 2), (0, 2): (4, 4), (0, 3): (8, 8),
    (1, 0): (2, 1), (1, 1): (4, 1), (1, 2): (4, 2), (1, 3): (8, 4),
    (2, 0): (1, 2), (2, 1): (1, 4), (2, 2): (2, 4), (2, 3): (4, 8),
}


# ---- renderer ---------------------------------------------------------------

def render_frame(state: dict) -> np.ndarray:
    """Reconstruct a (160,240,3) RGB frame from the extracted PPU state (mode 0 path)."""
    regs, vram, pal = state["regs"], state["vram"], palette_rgb(state["palette"])
    W, H = 240, 160
    out = np.zeros((H, W, 3), np.uint8)
    out[:] = pal[0]                       # backdrop = BG palette color 0
    win_prio = np.full((H, W), 5, np.int16)   # priority of current top BG pixel (5 = backdrop)

    dispcnt = regs["DISPCNT"]
    # BG layers: draw from lowest priority-on-top last. Collect (bg, prio).
    layers = []
    for bg in range(4):
        if not (dispcnt >> (8 + bg)) & 1:
            continue
        cnt = regs[f"BG{bg}CNT"]
        layers.append((bg, cnt & 0x3))
    # higher priority number is further back; equal prio -> lower bg index on top.
    for bg, prio in sorted(layers, key=lambda t: (-t[1], -t[0])):
        cnt = regs[f"BG{bg}CNT"]
        screen_base = ((cnt >> 8) & 0x1F) * 0x800
        char_base = ((cnt >> 2) & 0x3) * 0x4000
        hofs = regs[f"BG{bg}HOFS"] & 0x1FF
        vofs = regs[f"BG{bg}VOFS"] & 0x1FF
        size = (cnt >> 14) & 3
        map_w, map_h = {0: (32, 32), 1: (64, 32), 2: (32, 64), 3: (64, 64)}[size]
        tiles = _decode_tiles_4bpp(vram, char_base, 1024)
        cam_tx, cam_ty = hofs // 8, vofs // 8
        sub_x, sub_y = hofs % 8, vofs % 8
        for cy in range(21):
            gy = (cam_ty + cy) % map_h
            sby, iy = divmod(gy, 32)
            for cx in range(31):
                gx = (cam_tx + cx) % map_w
                sbx, ix = divmod(gx, 32)
                sc = {1: sbx, 2: sby, 3: sby * 2 + sbx}.get(size, 0)  # multi-screenblock layout
                eoff = screen_base + sc * 0x800 + (iy * 32 + ix) * 2
                entry = vram[eoff] | (vram[eoff + 1] << 8)
                tile = entry & 0x3FF
                palbank = (entry >> 12) & 0xF
                idx = tiles[tile]
                if entry & 0x400:  # hflip
                    idx = idx[:, ::-1]
                if entry & 0x800:  # vflip
                    idx = idx[::-1, :]
                px0 = cx * 8 - sub_x
                py0 = cy * 8 - sub_y
                for j in range(8):
                    y = py0 + j
                    if y < 0 or y >= H:
                        continue
                    for i in range(8):
                        x = px0 + i
                        if x < 0 or x >= W:
                            continue
                        ci = idx[j, i]
                        if ci == 0 or prio >= win_prio[y, x]:
                            continue
                        out[y, x] = pal[palbank * 16 + ci]
                        win_prio[y, x] = prio

    # Sprites (OBJ): 4bpp, 1D mapping (DISPCNT bit6). OBJ tiles at vram 0x10000.
    if (dispcnt >> 12) & 1:
        oam = np.frombuffer(state["oam"], dtype="<u2")
        obj_tiles = _decode_tiles_4bpp(vram, 0x10000, 1024)
        for s in range(127, -1, -1):       # lower index drawn on top -> iterate high->low
            a0, a1, a2 = int(oam[s * 4]), int(oam[s * 4 + 1]), int(oam[s * 4 + 2])
            rot = (a0 >> 8) & 1
            if not rot and (a0 & 0x0300) == 0x0200:   # non-affine "disabled"
                continue
            if a0 == 0 and a1 == 0 and a2 == 0:
                continue
            shape = (a0 >> 14) & 3; size = (a1 >> 14) & 3
            wt, ht = _SPRITE_DIM[(shape, size)]
            sw, sh = wt * 8, ht * 8
            base_tile = a2 & 0x3FF; palbank = (a2 >> 12) & 0xF; prio = (a2 >> 10) & 3
            # sprite texture (sh,sw) of color indices, 1D tile mapping
            tex = np.zeros((sh, sw), np.uint8)
            for tj in range(ht):
                for ti in range(wt):
                    tex[tj*8:tj*8+8, ti*8:ti*8+8] = obj_tiles[(base_tile + tj*wt + ti) & 0x3FF]
            y = a0 & 0xFF; x = a1 & 0x1FF
            if rot:
                # affine: box (double-size if bit9), sample texture via the 8.8 matrix
                dbl = (a0 >> 9) & 1
                bw, bh = (sw*2, sh*2) if dbl else (sw, sh)
                p = (a1 >> 9) & 0x1F
                PA, PB = _s16(int(oam[16*p+3])), _s16(int(oam[16*p+7]))
                PC, PD = _s16(int(oam[16*p+11])), _s16(int(oam[16*p+15]))
                x0 = x - 512 if x >= 256 else x
                y0 = y - 256 if y >= 160 else y
                cx, cy = bw/2.0, bh/2.0
                for by in range(bh):
                    yy = y0 + by
                    if yy < 0 or yy >= H: continue
                    dy = by - cy
                    for bx in range(bw):
                        xx = x0 + bx
                        if xx < 0 or xx >= W: continue
                        dx = bx - cx
                        tx = int((PA*dx + PB*dy)/256.0 + sw/2.0)
                        ty = int((PC*dx + PD*dy)/256.0 + sh/2.0)
                        if 0 <= tx < sw and 0 <= ty < sh:
                            ci = tex[ty, tx]
                            if ci and prio <= win_prio[yy, xx]:
                                out[yy, xx] = pal[256 + palbank*16 + ci]
                                win_prio[yy, xx] = prio
            else:
                if (a1 >> 12) & 1: tex = tex[:, ::-1]   # hflip
                if (a1 >> 13) & 1: tex = tex[::-1, :]   # vflip
                x0 = x - 512 if x >= 256 else x
                y0 = y - 256 if y >= 160 else y
                for by in range(sh):
                    yy = y0 + by
                    if yy < 0 or yy >= H: continue
                    for bx in range(sw):
                        xx = x0 + bx
                        if xx < 0 or xx >= W: continue
                        ci = tex[by, bx]
                        if ci and prio <= win_prio[yy, xx]:
                            out[yy, xx] = pal[256 + palbank*16 + ci]
                            win_prio[yy, xx] = prio
    return out


if __name__ == "__main__":
    import sys
    from pokemon_env.emulator import EmeraldEmulator
    ck = sys.argv[1] if len(sys.argv) > 1 else "/tmp/coll_chainN/RUSTBORO_CITY/attempt_000001/final.state"
    env = EmeraldEmulator(rom_path="Emerald-GBAdvance/rom.gba")
    env.initialize(); env.load_state(ck)
    for _ in range(6): env.core.run_frame()
    state = extract_full_ppu_state(env)
    recon = render_frame(state)
    real = np.asarray(env.get_screenshot().convert("RGB"), dtype=np.uint8)
    diff = np.abs(recon.astype(int) - real.astype(int)).sum(-1)
    exact = float((diff == 0).mean()) * 100
    close = float((diff <= 24).mean()) * 100
    print(f"regs DISPCNT=0x{state['regs']['DISPCNT']:04X} BLDCNT=0x{state['regs']['BLDCNT']:04X} "
          f"WIN0en={(state['regs']['DISPCNT']>>13)&1}")
    print(f"pixel match: exact={exact:.1f}%  close(<=24/255)={close:.1f}%   mean|Δ|={diff.mean():.1f}")
    try:
        from PIL import Image
        Image.fromarray(recon).save("/tmp/recon.png")
        Image.fromarray(real).save("/tmp/real.png")
        Image.fromarray((np.clip(diff,0,255)).astype(np.uint8)).save("/tmp/diff.png")
        print("wrote /tmp/recon.png /tmp/real.png /tmp/diff.png")
    except Exception as e:
        print("save err", e)
    env.stop()
