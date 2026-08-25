"""RESUME/STITCH: cut a recording back to a savestate boundary so a killed run can be
continued into the SAME channels instead of being replayed from the title screen.

`prepare_resume(run_dir, cut_frame)` rewinds every channel of `run_dir` to the newest
savestate K <= cut_frame and returns the handshake (`<run>/resume.json`) that
`ChunkRecorder(resume=...)` / `WorldModelSink(resume=...)` / `run_playthrough(resume=...)`
use to APPEND the tail. The result is one continuous, gate-valid stream — not two runs
glued together.

WHERE THE CUT FALLS (the whole correctness argument, one paragraph):

  A savestate written at frame K IS the emulator state at frame K, saved after that
  frame was recorded. So the prefix keeps the world up to and INCLUDING K:

      state-indexed rows (frames/states/phases/segments/semantic, ppu records)  idx <= K
      transition rows   (actions.jsonl: frame_idx -> next_frame_idx)            idx <  K

  i.e. everything that describes a frame the recording still has, and no transition
  that leaves the boundary. That keeps the truncated prefix gate-valid ON ITS OWN
  (`visual == actions + 1 + recorded_restores == ppu + 1 + recorded_restores`), which
  is the property the appended tail then extends: the resumed director sets
  `runner.frame_idx = K`, loads the savestate as a RECORDED restore (its settle frame
  becomes recorded frame K+1, explained by the resume entry in manifest["restores"]),
  and steps on. No hidden frames, no renumbering, no duplicate indices.

  Cutting the state rows at `< K` instead would leave the prefix's last action row
  (K-1 -> K) pointing at a frame nobody recorded, and the restored settle frame would
  then be the second unrecorded frame at the seam — exactly the v1 replay leak that
  W33 §3.2 closed. Hence `<= K`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Which key holds the frame index differs per channel (schema drift across corpus
# generations), so it is read off each file's FIRST row rather than hard-coded.
# Order matters: frames.jsonl carries both an emulator and a visual index and the cut
# is defined in emulator frames.
FRAME_FIELDS = ("emulator_frame_idx", "frame_idx", "frame", "visual_frame_idx")

# actions.jsonl rows are transitions; everything else is a per-frame state row (see
# module docstring for why the two get different comparisons).
CHANNELS = ("frames.jsonl", "actions.jsonl", "states.jsonl", "phases.jsonl",
            "segments.jsonl", "semantic.jsonl")
TRANSITION_CHANNELS = {"actions.jsonl"}


class ResumeError(RuntimeError):
    """prepare_resume could not produce a usable seam."""


# --------------------------------------------------------------------------- helpers

def savestate_frames(run_dir: Path) -> list[int]:
    """Frame indices of the run's savestates, ascending (`savestates/%08d.state.z`)."""
    out = []
    for p in (run_dir / "savestates").glob("*.state*"):
        stem = p.name.split(".")[0]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def _frame_field(path: Path) -> str | None:
    """The frame key of a jsonl channel, read off its first row (None if empty)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for name in FRAME_FIELDS:
                if name in row:
                    return name
            raise ResumeError(f"{path.name}: no frame field in {sorted(row)[:8]}")
    return None


def _truncate_jsonl(path: Path, limit: int, *, transition: bool,
                    on_keep: Callable[[dict], None] | None = None) -> dict:
    """Streaming rewrite + atomic rename: keep rows at/below the boundary.

    `transition` rows (actions) are kept while idx < limit, state rows while idx <= limit.
    """
    if not path.exists():
        return {"field": None, "kept": 0, "dropped": 0, "missing": True}
    field = _frame_field(path)
    if field is None:
        return {"field": None, "kept": 0, "dropped": 0, "empty": True}
    tmp = path.with_name(path.name + ".tmp")
    kept = dropped = 0
    with open(path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # a KILLED recorder can leave one torn line: that is exactly the case
                # this module exists for, so drop it instead of refusing to resume
                dropped += 1
                continue
            idx = row.get(field)
            keep = idx is not None and (idx < limit if transition else idx <= limit)
            if keep:
                dst.write(line if line.endswith("\n") else line + "\n")
                kept += 1
                if on_keep is not None:
                    on_keep(row)
            else:
                dropped += 1
    os.replace(tmp, path)
    return {"field": field, "kept": kept, "dropped": dropped}


def _scan_ppu_records(path: Path, run_dir: Path) -> list[list]:
    """Rebuild a ppu index by sequential scan when the .idx.json is missing/stale.

    Records are `tag(1) + u32 len + payload`, so offsets are exact; the frame numbers
    are not stored in the stream, so they are taken from semantic.jsonl (written by the
    same sink hook, one line per ppu record) and fall back to ordinal+1 (the sink's
    first record is frame 1 — the frame_hook runs after frame_idx increments).
    """
    offsets: list[tuple[int, str]] = []
    size = path.stat().st_size
    with open(path, "rb") as f:
        while True:
            off = f.tell()
            head = f.read(5)
            if len(head) < 5:
                break
            ln = int.from_bytes(head[1:5], "little")
            if off + 5 + ln > size:
                break                                   # torn tail from a killed writer
            f.seek(ln, os.SEEK_CUR)
            offsets.append((off, head[0:1].decode()))
    sem = run_dir / "semantic.jsonl"
    frames: list[int] | None = None
    if sem.exists():
        got = []
        with open(sem, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    got.append(json.loads(line).get("frame"))
        if len(got) >= len(offsets) and all(isinstance(v, int) for v in got[:len(offsets)]):
            frames = got[:len(offsets)]
    if frames is None:
        frames = [i + 1 for i in range(len(offsets))]
    return [[frames[i], off, kind] for i, (off, kind) in enumerate(offsets)]


def _cut_ppu(run_dir: Path, limit: int) -> dict:
    """Truncate ppu_state.bin at the first record with frame > limit; rewrite the index."""
    from collection.render_state import BLOCK_SIZES

    bin_path = run_dir / "ppu_state.bin"
    idx_path = run_dir / "ppu_state.bin.idx.json"
    if not bin_path.exists():
        return {"kept": 0, "dropped": 0, "missing": True}
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        frames = idx["frames"]
        block_sizes = idx.get("block_sizes", BLOCK_SIZES)
    else:
        frames = _scan_ppu_records(bin_path, run_dir)
        block_sizes = BLOCK_SIZES
    cut = len(frames)
    for i, rec in enumerate(frames):
        if rec[0] > limit:
            cut = i
            break
    if cut < len(frames):
        end = frames[cut][1]
    elif frames:
        # nothing to drop by frame number, but a killed writer can leave records past
        # the last INDEXED one (the index is only written on close): cut to the end of
        # the last indexed record so stream and index always agree
        off = frames[-1][1]
        with open(bin_path, "rb") as f:
            f.seek(off + 1)
            end = off + 5 + int.from_bytes(f.read(4), "little")
        end = min(end, bin_path.stat().st_size)
    else:
        end = 0
    dropped = len(frames) - cut
    os.truncate(bin_path, end)
    tmp = idx_path.with_name(idx_path.name + ".tmp")
    tmp.write_text(json.dumps({"block_sizes": block_sizes, "frames": frames[:cut]}))
    os.replace(tmp, idx_path)
    return {"kept": cut, "dropped": dropped, "bytes": end,
            "last_frame": frames[cut - 1][0] if cut else None}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomic compressed-npz write (savez would append its own .npz to a tmp name)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def _cut_chunks(run_dir: Path, kept: dict[str, int]) -> dict:
    """`kept` maps the chunk name recorded in frames.jsonl -> highest kept chunk_frame_idx.

    Chunks no kept row references are deleted outright; the boundary chunk is rewritten
    with its surviving frames only.
    """
    chunks_dir = run_dir / "chunks"
    deleted, rewritten = [], []
    if not chunks_dir.is_dir():
        return {"deleted": deleted, "rewritten": rewritten, "last_index": 0}
    for p in sorted(chunks_dir.glob("chunk_*")):
        rel = f"chunks/{p.name}"
        if rel not in kept:
            p.unlink()
            deleted.append(p.name)
            continue
        want = kept[rel] + 1
        if p.suffix != ".npz":
            continue                    # ffv1: boundary chunk cannot be losslessly cut here
        with np.load(p) as z:
            key = z.files[0]
            arr = z[key]
        if len(arr) > want:
            _save_npz(p, {key: arr[:want]})
            rewritten.append([p.name, int(len(arr)), int(want)])
    idxs = [int(name.split("_")[1].split(".")[0]) for name in kept]
    return {"deleted": deleted, "rewritten": rewritten, "last_index": max(idxs) if idxs else 0}


def _cut_ledger(run_dir: Path, limit: int) -> dict:
    """Truncate the live per-tick ledger (`ledger/chunk_*.npz` + index.json) to frame<=K.

    Not read by the gate, but leaving rows for frames the recording no longer contains
    would hand the packaging stage a silently wrong channel.
    """
    d = run_dir / "ledger"
    ip = d / "index.json"
    if not ip.exists():
        return {"missing": True}
    idx = json.loads(ip.read_text())
    chunks, total, dropped = [], 0, 0
    for fn, _start, n_rows in idx.get("chunks", []):
        p = d / fn
        if not p.exists():
            continue
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        fi = arrays.get("frame_idx")
        n_keep = int(np.count_nonzero(np.asarray(fi) <= limit)) if fi is not None else n_rows
        if n_keep == 0:
            p.unlink()
            dropped += n_rows
            continue
        if n_keep < len(next(iter(arrays.values()))):
            _save_npz(p, {k: v[:n_keep] for k, v in arrays.items()})
            dropped += n_rows - n_keep
        chunks.append([fn, total, n_keep])
        total += n_keep
    idx["chunks"] = chunks
    idx["frames"] = total
    tmp = ip.with_name(ip.name + ".tmp")
    tmp.write_text(json.dumps(idx))
    os.replace(tmp, ip)
    return {"kept": total, "dropped": dropped, "chunks": len(chunks)}


def _cut_manifest(run_dir: Path, limit: int) -> dict:
    """Drop manifest bookkeeping that refers to frames past the cut.

    `restores` matters to the gate: every RECORDED restore is one visual frame with no
    action row, so an entry for a frame that no longer exists breaks the channel
    arithmetic. blemish_spans are trimmed for the same reason (packaging reads them).
    """
    mp = run_dir / "manifest.json"
    if not mp.exists():
        return {"missing": True}
    man = json.loads(mp.read_text())
    restores = man.get("restores") or []
    keep = [x for x in restores if not isinstance(x.get("frame_idx"), int) or x["frame_idx"] <= limit]
    man["restores"] = keep
    spans = man.get("blemish_spans")
    if spans:
        man["blemish_spans"] = [s for s in spans if (s.get("span") or [0, 0])[1] <= limit]
    tmp = mp.with_name(mp.name + ".tmp")
    tmp.write_text(json.dumps(man, indent=2, sort_keys=True))
    os.replace(tmp, mp)
    return {"restores_kept": len(keep), "restores_dropped": len(restores) - len(keep),
            "recorded_restores": sum(1 for x in keep if x.get("recorded"))}


# ------------------------------------------------------------------------------ main

def prepare_resume(run_dir: str | Path, cut_frame: int) -> dict:
    """Rewind `run_dir` to the newest savestate K <= cut_frame; return/write resume.json.

    Every channel is cut in place (streaming rewrite + atomic rename for the jsonl
    channels, byte truncation for ppu_state.bin), savestates past K are deleted, and
    the handshake is written to `<run>/resume.json`:

        {cut_frame, savestate, chunk_index, visual_count, actions_count, ...}

    `chunk_index` is the recorder's counter value to restore (the last surviving chunk),
    so the resumed recorder's next chunk is chunk_index + 1.
    """
    run = Path(run_dir)
    if not run.is_dir():
        raise ResumeError(f"not a run directory: {run}")
    frames = savestate_frames(run)
    if not frames:
        raise ResumeError(f"no savestates in {run/'savestates'} — nothing to resume from")
    eligible = [f for f in frames if f <= int(cut_frame)]
    if not eligible:
        raise ResumeError(f"no savestate at or before frame {cut_frame} "
                          f"(earliest is {frames[0]})")
    k = eligible[-1]
    state_path = next(iter(sorted((run / "savestates").glob(f"{k:08d}.state*"))), None)
    if state_path is None:
        raise ResumeError(f"savestate for frame {k} vanished")

    kept_chunks: dict[str, int] = {}

    def _on_frame_row(row: dict) -> None:
        name = row.get("chunk")
        if name is None:
            return
        cfi = int(row.get("chunk_frame_idx", 0))
        if cfi > kept_chunks.get(name, -1):
            kept_chunks[name] = cfi

    # ppu FIRST: a killed run has no .idx.json, and the scan fallback recovers the
    # record->frame mapping from semantic.jsonl, which must still be complete.
    ppu = _cut_ppu(run, k)

    channels: dict[str, Any] = {}
    for name in CHANNELS:
        channels[name] = _truncate_jsonl(
            run / name, k, transition=name in TRANSITION_CHANNELS,
            on_keep=_on_frame_row if name == "frames.jsonl" else None)

    chunks = _cut_chunks(run, kept_chunks)
    ledger = _cut_ledger(run, k)
    manifest = _cut_manifest(run, k)

    dropped_states = 0
    for f in frames:
        if f > k:
            for p in (run / "savestates").glob(f"{f:08d}.state*"):
                p.unlink()
                dropped_states += 1

    info = {
        "cut_frame": k,
        "savestate": str(state_path),
        "chunk_index": chunks["last_index"],
        "visual_count": channels["frames.jsonl"]["kept"],
        "actions_count": channels["actions.jsonl"]["kept"],
        "requested_cut_frame": int(cut_frame),
        "run_dir": str(run),
        "ppu_records": ppu.get("kept"),
        "recorded_restores": manifest.get("recorded_restores"),
        "channels": channels,
        "ppu": ppu,
        "chunks": chunks,
        "ledger": ledger,
        "manifest": manifest,
        "savestates_dropped": dropped_states,
    }
    (run / "resume.json").write_text(json.dumps(info, indent=2))
    return info


def load_resume(resume: dict | str | Path) -> dict:
    """Accept a resume.json dict, a path to one, or a run dir containing one."""
    if isinstance(resume, dict):
        return resume
    p = Path(resume)
    if p.is_dir():
        p = p / "resume.json"
    return json.loads(p.read_text())


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("cut_frame", type=int)
    args = ap.parse_args()
    print(json.dumps(prepare_resume(args.run_dir, args.cut_frame), indent=2))


if __name__ == "__main__":
    main()
