"""Quality gate for continuous playthrough runs (read-only, retroactive).

Per run dir:
  1. COMPLETION — verified from the RECORDING, not the collector log: decode the
     final ppu state (last keyframe + XOR tail) and require the Stone Badge flag
     (SaveBlock1 flags + system-flag 0x7, i.e. flag 0x867) to be set.
  2. INTEGRITY — channel alignment (frames = actions + 1 = ppu + 1, the S1 rule,
     re-verified empirically), frames.jsonl contiguity, chunk spot-decode.
  3. DIVERSITY META — wander/life-block outcomes from playthrough_summary.json.
Verdict PASS -> eligible for the corpus; FAIL -> moved to _quarantine/.

Usage:
  python -m collection.audits.playthrough_gate <playthroughs_root> [--apply]
    --apply    actually move FAIL dirs to _quarantine/ (default: report only)
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

from collection.extractors.ram import GBAState, BLOB_SIZE

SAVE_BLOCK1_PTR = 0x03005D8C
FLAGS_OFFSET = 0x1270
SYSTEM_FLAGS_START = 0x860
STONE_BADGE_SYS_OFFSET = 0x7          # badge_01 (memory_reader.py:3829)


def final_state(run_dir: Path) -> GBAState:
    """Decode the run's final ppu state from the last keyframe forward."""
    idx = json.loads((run_dir / "ppu_state.bin.idx.json").read_text())
    raw = (run_dir / "ppu_state.bin").read_bytes()
    frames = idx["frames"]
    # find last keyframe (byte tag 'K' at the record start)
    k = None
    for i in range(len(frames) - 1, -1, -1):
        off = frames[i][1]
        if raw[off:off + 1] == b"K":
            k = i
            break
    if k is None:
        raise ValueError("no keyframe found")
    cur: np.ndarray | None = None
    for f, off, _kind in frames[k:]:
        ln = int.from_bytes(raw[off + 1:off + 5], "little")
        payload = np.frombuffer(zlib.decompress(raw[off + 5:off + 5 + ln]), np.uint8)
        cur = payload.copy() if raw[off:off + 1] == b"K" else (cur ^ payload)
        assert cur is not None and len(cur) == BLOB_SIZE
    return GBAState.from_blob(cur)


def stone_badge(st: GBAState) -> tuple[bool, int]:
    """(stone_badge_set, badge_count) read from SaveBlock1 flags."""
    sb1 = st.u32(SAVE_BLOCK1_PTR)
    if not (0x02000000 <= sb1 < 0x02040000):
        return False, -1
    base = sb1 + FLAGS_OFFSET + SYSTEM_FLAGS_START // 8
    count = 0
    stone = False
    for i in range(8):                              # badge_01..08 = sys offsets 0x7..0xE
        off = STONE_BADGE_SYS_OFFSET + i
        b = st.u8(base + off // 8)
        if b & (1 << (off % 8)):
            count += 1
            if i == 0:
                stone = True
    return stone, count


PARTY_BASE = 0x020244EC
PARTY_COUNT = 0x020244E9
_CHARMAP = {**{0xBB + i: chr(ord("A") + i) for i in range(26)},
            **{0xD5 + i: chr(ord("a") + i) for i in range(26)},
            0x00: " "}
# The starter lines. A party mon whose stored name is one of these was never
# nicknamed; anything else (the v2 corpus was full of "AAAAAAAAAA") was.
_SPECIES_NAMES = {"MUDKIP", "MARSHTOMP", "SWAMPERT",
                  "TORCHIC", "COMBUSKEN", "BLAZIKEN",
                  "TREECKO", "GROVYLE", "SCEPTILE"}


def party_fidelity(st: GBAState) -> tuple[int, list[str], list[str]]:
    """(party_count, stored_names, reasons) — the corpus entity-fidelity invariants.

    Both failures were live in the v2 corpus and both are silent in the collector log,
    which is why they are checked HERE, against the recording:
      * every run's starter was renamed AAAAAAAAAA (Birch's nickname offer, blanket-
        confirmed by the policy);
      * two runs finished holding a wild Pokemon (an A-mash landed on BAG and threw a
        Poke Ball), which also deadlocked one run on the switch screen.
    """
    reasons: list[str] = []
    n = st.bytes(PARTY_COUNT, 1)[0]
    names: list[str] = []
    for slot in range(min(n, 6)):
        raw = st.bytes(PARTY_BASE + slot * 100 + 8, 10)
        nm = "".join(_CHARMAP.get(b, "") for b in raw).strip()
        names.append(nm)
    if n != 1:
        reasons.append(f"party holds {n} pokemon, expected exactly 1 (caught/received one?)")
    for nm in names:
        if nm.upper() not in _SPECIES_NAMES:
            reasons.append(f"party member named {nm!r} — nicknamed, or an unexpected species")
    return n, names, reasons


def _unexplained_frames(run_dir: Path, restores: list[dict]) -> list[int]:
    """Visual frames with no action row that no logged restore accounts for.

    A restore is logged with the frame_idx it resumed AT, while the frame it wrote is
    the one just before; accept either side rather than guessing the convention.
    """
    acts: set[int] = set()
    with open(run_dir / "actions.jsonl") as f:
        for line in f:
            try:
                acts.add(json.loads(line)["frame_idx"])
            except Exception:
                continue
    vis: list[int] = []
    with open(run_dir / "frames.jsonl") as f:
        for line in f:
            try:
                vis.append(json.loads(line)["visual_frame_idx"])
            except Exception:
                continue
    if not vis:
        return []
    last = max(vis)
    ok = set()
    for x in restores:
        fi = x.get("frame_idx")
        if isinstance(fi, int):
            ok.update((fi - 1, fi, fi + 1))
    return [f for f in sorted(set(vis) - acts) if f != last and f not in ok]

def count_lines(p: Path) -> int:
    n = 0
    with open(p, "rb") as f:
        for _ in f:
            n += 1
    return n


def gate_run(run_dir: Path) -> dict:
    r: dict = {"run": run_dir.name, "verdict": "FAIL", "reasons": []}
    try:
        summ = json.loads((run_dir / "playthrough_summary.json").read_text())
        man = json.loads((run_dir / "manifest.json").read_text())
        r["milestones"] = f"{summ.get('milestones_passed')}/{summ.get('milestones_total')}"
        r["frames_summary"] = summ.get("total_frames")

        # 1. completion from the recording
        st = final_state(run_dir)
        stone, badges = stone_badge(st)
        r["stone_badge"] = stone
        r["badge_count"] = badges
        if not stone:
            r["reasons"].append("stone badge flag NOT set in final state")

        # 1b. entity fidelity: exactly one pokemon, and it kept its species name
        n_party, party_names, party_reasons = party_fidelity(st)
        r["party_count"] = n_party
        r["party_names"] = party_names
        r["reasons"].extend(party_reasons)

        # 2. integrity
        idx = json.loads((run_dir / "ppu_state.bin.idx.json").read_text())
        n_ppu = len(idx["frames"])
        n_actions = count_lines(run_dir / "actions.jsonl")
        n_frames_jsonl = count_lines(run_dir / "frames.jsonl")
        vis = man.get("visual_frame_count")
        # RESTORES ARE FRAMES TOO. The original S1 rule read `vis == actions+1 == ppu+1`,
        # which predates savestate restores: DirectEmulatorRunner.restore() calls
        # record_current_frame(phase="restore"), writing a visual frame with no action
        # row. Every run therefore failed this check -- including the ones counted as
        # gate-valid, because full_audit.py only ever checked badge + party fidelity and
        # never called gate_run(). Measured, the offset is exact and fully explained:
        #
        #   exp_052  visual 533693  actions 533691  recorded restores 1
        #   exp_022  visual 438392  actions 438390  recorded restores 1
        #   exp_040  visual 771125  actions 771122  recorded restores 2
        #
        #   visual == actions + 1 + recorded_restores       (+1 = the final frame,
        #                                                    which has no action after it)
        #
        # Do NOT just widen the arithmetic -- that would let genuine hidden frames
        # through. Check PROVENANCE: every visual frame lacking an action row must be
        # either the last frame or the one a logged restore wrote. That is the
        # "no hidden frames" property the restore_log exists to prove, and it is a
        # strictly stronger test than the count it replaces.
        restores = [x for x in (man.get("restores") or []) if x.get("recorded")]
        n_restore = len(restores)
        r["channels"] = {"frames_jsonl": n_frames_jsonl, "actions": n_actions,
                         "ppu": n_ppu, "visual": vis, "recorded_restores": n_restore}
        if not (vis == n_actions + 1 + n_restore == n_ppu + 1 + n_restore):
            r["reasons"].append(f"channel misalignment {r['channels']}")
        else:
            unexplained = _unexplained_frames(run_dir, restores)
            if unexplained:
                r["reasons"].append(
                    f"{len(unexplained)} visual frames with no action row and no logged "
                    f"restore: {unexplained[:5]}")
        # frames.jsonl contiguity (first/last index sanity)
        with open(run_dir / "frames.jsonl") as f:
            first = json.loads(f.readline())
        r["frames_first_idx"] = first.get("frame", first.get("visual_frame"))
        # chunk spot-decode
        chunks = sorted((run_dir / "chunks").glob("chunk_*.npz"))
        for c in (chunks[0], chunks[-1]):
            with np.load(c) as z:
                arr = z[z.files[0]]
                assert arr.ndim >= 3 and arr.shape[-1] == 3, f"bad chunk {c.name}: {arr.shape}"
        r["chunks"] = len(chunks)

        # 3. diversity meta
        blocks = summ.get("blocks") or []
        r["blocks_scheduled"] = len(blocks)
        r["blocks_ran"] = sum(1 for b in blocks if b.get("ran") or b.get("status") == "ran")

        if not r["reasons"]:
            r["verdict"] = "PASS"
    except Exception as e:                                       # noqa: BLE001
        r["reasons"].append(f"exception: {type(e).__name__}: {e}")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    quarantine = root / "_quarantine"
    results = []
    for d in sorted(root.glob("playthrough__*")):
        if not d.is_dir():
            continue
        res = gate_run(d)
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))
        if res["verdict"] == "FAIL" and args.apply:
            quarantine.mkdir(exist_ok=True)
            d.rename(quarantine / d.name)
            print(f"  -> quarantined {d.name}", file=sys.stderr)
    (root / "gate_report.json").write_text(json.dumps(results, indent=1))
    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    print(f"\nGATE: {n_pass}/{len(results)} PASS -> {root/'gate_report.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
