"""Offline views over the collected world-model data.

Every rung of the conditioning ladder, and every frame rate, is a *pure transform* of the
single stored full-state stream — no re-collection. This module provides:
  • clips_at_fps  — continuous clips (split at resets) subsampled to a target fps
  • rung_condition — the per-rung conditioning mask derived from the full PPU+WRAM state
  • RolloutReader — ties frames(RGB) + actions + PPU + semantic together by frame index
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

from collection.world_model_sink import deserialize_ppu

# Conditioning-reduction ladder: each rung selects what information the model is given.
# All are derived from the stored full machine state (Rung A) — "collect the superset once".
RUNGS = ("A_full_ppu", "B_map_coord_party", "C_coord_only", "Z_action_only")


def clips_at_fps(rollout_dir: str | Path, target_fps: int = 30, src_fps: int = 60) -> list[list[int]]:
    """Continuous clips (split at checkpoint resets) of emulator_frame_idx, subsampled to
    target_fps. Each returned clip is a continuous walk (no teleports) at the target rate."""
    d = Path(rollout_dir)
    summ_path = next((p for p in (d / "coverage_summary.json", d / "rollout_summary.json") if p.exists()), None)
    starts = json.loads(summ_path.read_text()).get("clip_start_frames", [0]) if summ_path else [0]
    frames = [json.loads(l)["emulator_frame_idx"] for l in (d / "frames.jsonl").open()]
    stride = max(1, round(src_fps / target_fps))
    clips = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else (1 << 30)
        seg = [f for f in frames if s <= f < e]
        if seg:
            clips.append(seg[::stride])
    return clips


def rung_condition(rung: str, *, state: dict | None = None, semantic: dict | None = None) -> dict | None:
    """The conditioning the model receives at a given rung, derived from the full state.
    state = deserialized PPU+WRAM dict (Rung A); semantic = the per-frame semantic record."""
    if rung == "A_full_ppu":
        return state                                   # full machine render-state
    if rung == "B_map_coord_party":                    # map + coordinate + party (+ NPCs)
        return {"map": semantic["map"], "x": semantic["x"], "y": semantic["y"],
                "facing": semantic["facing"], "objects": semantic["objects"]}
        # party/flags also derivable offline from state["ewram"] via memory_reader offsets
    if rung == "C_coord_only":
        return {"map": semantic["map"], "x": semantic["x"], "y": semantic["y"], "facing": semantic["facing"]}
    if rung == "Z_action_only":
        return {}                                      # nothing but the action (pure-network end)
    raise ValueError(f"unknown rung {rung}")


class RolloutReader:
    """Random-access reader joining all four streams by emulator_frame_idx."""

    def __init__(self, rollout_dir: str | Path):
        self.d = Path(rollout_dir)
        self.frame_meta = {json.loads(l)["emulator_frame_idx"]: json.loads(l)
                           for l in (self.d / "frames.jsonl").open()}
        self.actions = {r["frame_idx"]: r["buttons_held"]
                        for r in (json.loads(l) for l in (self.d / "actions.jsonl").open())}
        self.semantic = {r["frame"]: r for r in (json.loads(l) for l in (self.d / "semantic.jsonl").open())}
        self._ppu_blobs = self._load_ppu()
        self._chunks: dict = {}

    def _load_ppu(self) -> dict:
        idx = json.loads((self.d / "ppu_state.bin.idx.json").read_text())
        raw = (self.d / "ppu_state.bin").read_bytes()
        blobs, prev = {}, None
        for fidx, off, _kind in idx["frames"]:
            ln = int.from_bytes(raw[off + 1:off + 5], "little")
            payload = np.frombuffer(zlib.decompress(raw[off + 5:off + 5 + ln]), np.uint8)
            blob = payload if raw[off:off + 1] == b"K" else np.bitwise_xor(payload, prev)
            prev = blob
            blobs[fidx] = blob.tobytes()
        return blobs

    def rgb(self, frame_idx: int) -> np.ndarray:
        m = self.frame_meta[frame_idx]
        cp = str(self.d / m["chunk"])
        if cp not in self._chunks:
            self._chunks[cp] = np.load(cp)["frames"]
        return self._chunks[cp][m["chunk_frame_idx"]]

    def state(self, frame_idx: int) -> dict:
        return deserialize_ppu(self._ppu_blobs[frame_idx])

    def sample(self, frame_idx: int, rung: str = "A_full_ppu") -> dict:
        sem = self.semantic.get(frame_idx)
        st = self.state(frame_idx) if rung == "A_full_ppu" else None
        return {
            "frame_idx": frame_idx,
            "rgb": self.rgb(frame_idx),
            "action": self.actions.get(frame_idx, []),
            "condition": rung_condition(rung, state=st, semantic=sem),
        }


if __name__ == "__main__":
    import sys
    from collection.render_state import render_frame
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rollout_all_test/ROUTE_103"
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    clips = clips_at_fps(d, target_fps=fps)
    print(f"{d}: {len(clips)} continuous clips @ {fps}fps; lengths {[len(c) for c in clips][:8]}{'...' if len(clips)>8 else ''}")
    r = RolloutReader(d)
    # demo a Rung-A sample and a Rung-B sample on one frame; verify Rung-A condition renders
    f = clips[0][len(clips[0]) // 2]
    a = r.sample(f, "A_full_ppu")
    b = r.sample(f, "B_map_coord_party")
    d_rgb = np.abs(render_frame(a["condition"]).astype(int) - a["rgb"].astype(int)).sum(-1)
    print(f"frame {f}  action={a['action']}")
    print(f"  Rung A (full): blocks={list(a['condition']['regs'])[:3]}...  renders-to-RGB exact={100*(d_rgb==0).mean():.1f}%")
    print(f"  Rung B (reduced): {b['condition']}")
