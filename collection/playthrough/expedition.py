"""Standalone runner for W33 expedition blocks — the job-style entry point.

Runs registered expedition blocks (schedule.EXPEDITION_BLOCKS) from a savestate on
a fresh recorded runner, through the same `run_nav_block` safety wrapper a playthrough
uses, and writes the outcome to <out>/block_summary.json. For block testing and
deficit-filling jobs; inside a full expedition the director carries the same blocks
via `run_playthrough(expedition=[...])`.

    .venv/bin/python -m collection.playthrough.expedition \
        --state data/storyline_wm/OLDALE_TOWN/attempt_000001/final.state \
        --out /tmp/sweep_test --block bfs_sweep \
        --kwargs '{"maps": ["0,10"], "legs": [["0,10", "0,16"]]}'

Multi-block jobs (item 4: audit/derivation-shaped runs) pass `blocks=[{"block": ...,
"kwargs": {...}}, ...]` (CLI: --blocks JSON) — the blocks run in sequence on ONE
runner/recording and the outcomes land in <out>/block_summaries.json (the
single-block path keeps writing block_summary.json byte-compatibly).

`record_wm=True` (CLI: --wm) attaches the WorldModelSink frame hook, so the job run
carries the full v2 payload (per-tick ledger chunks + PPU stream + semantic rows)
like a director run does — required for any job whose output feeds the corpus or the
v2 audits/derivation kit. Default off: block TESTS don't pay the sink cost.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from collection.direct_runner import DirectEmulatorRunner
from collection.playthrough.blocks.base import run_nav_block
from collection.playthrough.schedule import EXPEDITION_BLOCKS
from collection.recorder import ChunkRecorder


def _recover_to_overworld(runner, *, dialog_tries: int = 30) -> None:
    """Bounded battle-flee + dialog-close so the NEXT block's precondition seam sees
    free overworld (see the call site for why the job runner needs this)."""
    from collection.heatz_adapter import is_dialog_open
    from collection.playthrough.blocks.base import _hstate, flee_battle

    if runner.nav_state().in_battle:
        flee_battle(runner)
    for _ in range(dialog_tries):
        if not is_dialog_open(_hstate(runner)):
            break
        runner.perform_action("b", metadata={"src": "job_recover"})


def run_expedition_block(*, load_state: str, out_dir: str, block: str | None = None,
                         kwargs: dict | None = None, blocks: list[dict] | None = None,
                         rom_path: str = "Emerald-GBAdvance/rom.gba", record: bool = True,
                         visual_fps: int = 60, record_wm: bool = False,
                         savestate_every: int = 4000,
                         return_budget: int = 30000) -> dict | list:
    single = blocks is None
    if single:
        blocks = [{"block": block, "kwargs": kwargs or {}}]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blks = [EXPEDITION_BLOCKS[b["block"]](**b.get("kwargs", {})) for b in blocks]
    meta = ({"kind": "expedition_block", "block": block, "kwargs": kwargs or {},
             "load_state": str(load_state)} if single else
            {"kind": "expedition_blocks", "blocks": blocks, "load_state": str(load_state)})
    run_id = f"block_{block}" if single else f"blocks_{len(blocks)}"
    recorder_cm = ChunkRecorder(str(out), run_id=run_id, backend="npz",
                                visual_fps=visual_fps, metadata=meta)
    outcomes: list[dict] = []
    with recorder_cm as recorder:
        runner = DirectEmulatorRunner(rom_path=rom_path, load_state=load_state,
                                      recorder=recorder if record else None,
                                      savestate_every=savestate_every)
        runner.initialize()
        sink = None
        if record and record_wm:
            from collection.world_model_sink import WorldModelSink
            sink = WorldModelSink(str(out))
            runner.frame_hook = sink.capture
        mk = None
        try:
            runner.settle_to_free_overworld()
            for blk in blks:
                if mk is None:
                    from collection.navigator import MapKnowledge
                    mk = MapKnowledge(rom_path=rom_path)
                # Between-blocks recovery: a previous block may legitimately end
                # inside a wild battle or with a dialog box open (anchor-return
                # budget expiry mid-grass / mid-unstick) — the director's spine
                # machinery absorbs this between milestones; the job runner must
                # too, or every later block skips on precondition. Battle: flee.
                # Dialog: vision-validated check + B-only closer (the A,A,B clearer
                # re-talks a faced NPC — the interaction block's documented lesson).
                _recover_to_overworld(runner)
                outcomes.append(run_nav_block(runner, blk, mk=mk,
                                              return_budget=return_budget))
        finally:
            if sink is not None:
                sink.close()
            runner.close()
    if single:
        (out / "block_summary.json").write_text(json.dumps(outcomes[0], indent=2))
        return outcomes[0]
    (out / "block_summaries.json").write_text(json.dumps(outcomes, indent=2))
    return outcomes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", default=None, choices=sorted(EXPEDITION_BLOCKS))
    ap.add_argument("--kwargs", default="{}", help="JSON constructor kwargs for the block")
    ap.add_argument("--blocks", default=None,
                    help='JSON [{"block": name, "kwargs": {...}}, ...] multi-block job')
    ap.add_argument("--rom", default="Emerald-GBAdvance/rom.gba")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--wm", action="store_true",
                    help="attach the WorldModelSink (ledger/PPU/semantic per frame)")
    ap.add_argument("--savestate_every", type=int, default=4000)
    args = ap.parse_args()
    if (args.block is None) == (args.blocks is None):
        ap.error("exactly one of --block / --blocks")
    outcome = run_expedition_block(
        load_state=args.state, out_dir=args.out, block=args.block,
        kwargs=json.loads(args.kwargs),
        blocks=json.loads(args.blocks) if args.blocks else None,
        rom_path=args.rom, record=not args.no_record, record_wm=args.wm,
        savestate_every=args.savestate_every)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
