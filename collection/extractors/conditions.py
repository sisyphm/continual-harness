"""The assembled A′ per-frame condition record — every validated extractor group in one schema.

`frame_condition(st)` is the SINGLE source of truth, usable both live (demo) and offline; the
`RunConditionWriter` packs a recorded run's frames into compact arrays (one npz per run) — the
training-side input. Conditions are PER GAME FRAME (60 fps); aggregation to latent timing
(1 latent = 4 frames; representative frame + 4 sub-frame scalars) happens in the model repo.

Per-run npz schema (N = frames):
  mode u8 · in_battle u8 · cam (N,2) s16 · bld (N,3) u16 · win_mask (N,75) u8 packed bits (20x30)
  player_xy (N,2) s16 · player_facing u8 (0-3,255) · player_moving u8 · player_screen (N,2) s16
  ent_valid/gfx/local/facing/moving/mvt (N,16) u8 · ent_xy,ent_screen (N,16,2) s16
  map_id (N,2) u8 · grid_idx (N,) i16  (+ grid_<k> u16 arrays, the unique metatile grids)
  text_state u8 (0 none / 1 typing / 2 finished-box) · text_id i32 · text_reveal u16 (+ texts json)
  bat_valid (N,4) u8 · bat_species u16 · bat_level u8 · bat_hp,bat_maxhp u16 · bat_status u32
  bat_moves (N,4,4) u16 · bat_pp (N,4,4) u8 · bat_type u32 · bat_comm (N,8) u8
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from collection.extractors.battle import battle_state, in_battle
from collection.extractors.entities import DIRECTIONS, camera_px, entities, player_state
from collection.extractors.ram import GBAState, iter_states
from collection.extractors.terrain import terrain
from collection.extractors.text import last_message, text_state
from collection.extractors.ui import effects, mode, window_mask

_FACING_ID = {"DOWN": 0, "UP": 1, "LEFT": 2, "RIGHT": 3}
_DIR_ID = {None: 0, "DOWN": 1, "UP": 2, "LEFT": 3, "RIGHT": 4}
N_ENT, N_BAT = 16, 4
TEXTBOX_ROWS = slice(14, 20)                 # bottom band of the window mask = message box


class RunConditionWriter:
    """Accumulates per-frame conditions for one run and saves the npz + sidecar json."""

    def __init__(self, rom: bytes | None = None):
        self.rom = rom
        self.rows: list[dict] = []
        self.texts: dict[str, int] = {}      # string -> id
        self.grids: list[np.ndarray] = []
        self._grid_hash: dict[bytes, int] = {}

    def _text_id(self, s: str) -> int:
        if s not in self.texts:
            self.texts[s] = len(self.texts)
        return self.texts[s]

    def _grid_id(self, grid: np.ndarray) -> int:
        key = grid.tobytes()
        if key not in self._grid_hash:
            self._grid_hash[key] = len(self.grids)
            self.grids.append(grid)
        return self._grid_hash[key]

    def add(self, st: GBAState) -> None:
        r: dict = {}
        r["mode"] = mode(st)
        r["in_battle"] = int(in_battle(st))
        r["cam"] = camera_px(st)
        r["bld"] = effects(st)
        wm = window_mask(st)
        r["win"] = np.packbits(wm)

        p = player_state(st)
        r["player_xy"] = (p["x"], p["y"])
        r["player_facing"] = _FACING_ID.get(p["facing"], 255)
        r["player_moving"] = _DIR_ID.get(p["moving_dir"], 0)

        ents = entities(st)
        ev = np.zeros(N_ENT, np.uint8); eg = np.zeros(N_ENT, np.uint8)
        el = np.zeros(N_ENT, np.uint8); ef = np.zeros(N_ENT, np.uint8)
        em = np.zeros(N_ENT, np.uint8); et = np.zeros(N_ENT, np.uint8)
        exy = np.zeros((N_ENT, 2), np.int16); esc = np.zeros((N_ENT, 2), np.int16)
        pscreen = (0, 0)
        for e in ents:
            s = e.slot
            ev[s], eg[s], el[s] = 1, e.graphics_id, e.local_id
            ef[s] = _DIR_ID.get(e.facing, 0); em[s] = _DIR_ID.get(e.moving_dir, 0)
            et[s] = e.movement_type
            exy[s] = (e.x, e.y); esc[s] = (e.screen_x, e.screen_y)
            if e.is_player:
                pscreen = (e.screen_x, e.screen_y)
        r["player_screen"] = pscreen
        r["ent"] = (ev, eg, el, ef, em, et, exy, esc)

        t = terrain(st)
        if t is not None:
            r["map_id"] = (t.map_group, t.map_num)
            r["grid_idx"] = self._grid_id(t.grid)
        else:
            r["map_id"] = (255, 255)
            r["grid_idx"] = -1

        ts = text_state(st, self.rom)
        if ts is not None and ts.text.strip():
            r["text"] = (1, self._text_id(ts.text), min(ts.reveal, 65535))
        elif wm[TEXTBOX_ROWS].any():                          # finished box awaiting input
            msg = last_message(st, in_battle=bool(r["in_battle"]))
            r["text"] = (2, self._text_id(msg), min(len(msg), 65535)) if msg.strip() else (0, -1, 0)
        else:
            r["text"] = (0, -1, 0)

        bv = np.zeros(N_BAT, np.uint8); bs = np.zeros(N_BAT, np.uint16)
        blv = np.zeros(N_BAT, np.uint8); bhp = np.zeros(N_BAT, np.uint16)
        bmx = np.zeros(N_BAT, np.uint16); bst = np.zeros(N_BAT, np.uint32)
        bmv = np.zeros((N_BAT, 4), np.uint16); bpp = np.zeros((N_BAT, 4), np.uint8)
        btype, bcomm = 0, b"\0" * 8
        b = battle_state(st)
        if b is not None:
            btype, bcomm = b["battle_type"], b["comm"]
            for bt in b["battlers"]:
                s = bt.slot
                bv[s], bs[s], blv[s] = 1, bt.species, bt.level
                bhp[s], bmx[s], bst[s] = bt.hp, bt.max_hp, bt.status1
                bmv[s] = bt.moves; bpp[s] = bt.pp
        r["bat"] = (bv, bs, blv, bhp, bmx, bst, bmv, bpp, btype, np.frombuffer(bcomm, np.uint8))
        self.rows.append(r)

    def _merge_streamed_texts(self) -> tuple[list[str], dict[int, tuple[int, int]]]:
        """Some UI paths STREAM text char-by-char, producing one table entry per typed prefix.
        Collapse: a string that is a strict prefix of another maps to the longest superstring,
        with reveal = the prefix length. Returns (final table, old_id -> (new_id, reveal_cap))."""
        by_id = {i: s for s, i in self.texts.items()}
        ordered = sorted(self.texts, key=len, reverse=True)     # longest first
        finals: list[str] = []
        remap: dict[str, int] = {}
        for s in ordered:
            host = next((f for f in finals if f.startswith(s)), None)
            if host is None:
                finals.append(s)
                remap[s] = len(finals) - 1
            else:
                remap[s] = finals.index(host)
        return finals, {i: (remap[s], len(s)) for i, s in by_id.items()}

    def save(self, out: Path) -> None:
        n = len(self.rows)
        R = self.rows
        finals, tmap = self._merge_streamed_texts()
        for r in R:
            state, tid, reveal = r["text"]
            if tid >= 0:
                new_id, cap = tmap[tid]
                r["text"] = (state, new_id, min(reveal, cap) if state == 1 else cap)
        self.texts = {s: i for i, s in enumerate(finals)}
        arrs: dict[str, np.ndarray] = {
            "mode": np.array([r["mode"] for r in R], np.uint8),
            "in_battle": np.array([r["in_battle"] for r in R], np.uint8),
            "cam": np.array([r["cam"] for r in R], np.int16),
            "bld": np.array([r["bld"] for r in R], np.uint16),
            "win_mask": np.stack([r["win"] for r in R]),
            "player_xy": np.array([r["player_xy"] for r in R], np.int16),
            "player_facing": np.array([r["player_facing"] for r in R], np.uint8),
            "player_moving": np.array([r["player_moving"] for r in R], np.uint8),
            "player_screen": np.array([r["player_screen"] for r in R], np.int16),
            "map_id": np.array([r["map_id"] for r in R], np.uint8),
            "grid_idx": np.array([r["grid_idx"] for r in R], np.int16),
            "text_state": np.array([r["text"][0] for r in R], np.uint8),
            "text_id": np.array([r["text"][1] for r in R], np.int32),
            "text_reveal": np.array([r["text"][2] for r in R], np.uint16),
            "bat_type": np.array([r["bat"][8] for r in R], np.uint32),
            "bat_comm": np.stack([r["bat"][9] for r in R]),
        }
        ent_names = ("ent_valid", "ent_gfx", "ent_local", "ent_facing", "ent_moving", "ent_mvt",
                     "ent_xy", "ent_screen")
        for i, nm in enumerate(ent_names):
            arrs[nm] = np.stack([r["ent"][i] for r in R])
        bat_names = ("bat_valid", "bat_species", "bat_level", "bat_hp", "bat_maxhp", "bat_status",
                     "bat_moves", "bat_pp")
        for i, nm in enumerate(bat_names):
            arrs[nm] = np.stack([r["bat"][i] for r in R])
        for k, g in enumerate(self.grids):
            arrs[f"grid_{k}"] = g
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arrs)
        sidecar = {"frames": n, "n_grids": len(self.grids),
                   "texts": [s for s, _ in sorted(self.texts.items(), key=lambda kv: kv[1])]}
        out.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False))


def precompute_run(run_dir: str | Path, out: str | Path, rom: bytes | None) -> str:
    w = RunConditionWriter(rom)
    for _f, st in iter_states(run_dir):
        w.add(st)
    w.save(Path(out))
    return f"{Path(out).name}: {len(w.rows)} frames, {len(w.grids)} grids, {len(w.texts)} texts"
