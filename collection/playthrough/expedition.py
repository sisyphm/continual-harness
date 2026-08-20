"""Standalone runner for W33 expedition blocks — the job-style entry point.

Runs ONE registered expedition block (schedule.EXPEDITION_BLOCKS) from a savestate on
a fresh recorded runner, through the same `run_nav_block` safety wrapper a playthrough
uses, and writes the outcome to <out>/block_summary.json. For block testing and
deficit-filling jobs; inside a full expedition the director carries the same blocks
via `run_playthrough(expedition=[...])`.

    .venv/bin/python -m collection.playthrough.expedition \
        --state data/storyline_wm/OLDALE_TOWN/attempt_000001/final.state \
        --out /tmp/sweep_test --block bfs_sweep \
        --kwargs '{"maps": ["0,10"], "legs": [["0,10", "0,16"]]}'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from collection.direct_runner import DirectEmulatorRunner
from collection.playthrough.blocks.base import run_nav_block
from collection.playthrough.schedule import EXPEDITION_BLOCKS
from collection.recorder import ChunkRecorder


def run_expedition_block(*, load_state: str, out_dir: str, block: str, kwargs: dict,
                         rom_path: str = "Emerald-GBAdvance/rom.gba", record: bool = True,
                         visual_fps: int = 60) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blk = EXPEDITION_BLOCKS[block](**kwargs)
    recorder_cm = ChunkRecorder(str(out), run_id=f"block_{block}", backend="npz",
                                visual_fps=visual_fps,
                                metadata={"kind": "expedition_block", "block": block,
                                          "kwargs": kwargs, "load_state": str(load_state)})
    with recorder_cm as recorder:
        runner = DirectEmulatorRunner(rom_path=rom_path, load_state=load_state,
                                      recorder=recorder if record else None)
        runner.initialize()
        try:
            runner.settle_to_free_overworld()
            outcome = run_nav_block(runner, blk)
        finally:
            runner.close()
    (out / "block_summary.json").write_text(json.dumps(outcome, indent=2))
    return outcome


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", required=True, choices=sorted(EXPEDITION_BLOCKS))
    ap.add_argument("--kwargs", default="{}", help="JSON constructor kwargs for the block")
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()
    outcome = run_expedition_block(load_state=args.state, out_dir=args.out, block=args.block,
                                   kwargs=json.loads(args.kwargs), rom_path=args.rom,
                                   record=not args.no_record)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
