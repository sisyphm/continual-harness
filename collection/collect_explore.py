"""Bounded BFS explore-transition collector."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

from collection.actions import EXPLORE_ACTIONS
from collection.direct_runner import DirectEmulatorRunner
from collection.recorder import ChunkRecorder
from collection.state import result_label


def collect_explore(
    *,
    load_state: str,
    story_bucket: str,
    output_dir: str,
    rom_path: str,
    backend: str,
    visual_fps: int,
    max_states: int,
    actions: tuple[str, ...] = EXPLORE_ACTIONS,
    settle: bool = True,
    settle_actions: int = 24,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with ChunkRecorder(out, run_id=f"explore_{story_bucket}", visual_fps=visual_fps, backend=backend, metadata={"story_bucket": story_bucket}) as recorder:
        runner = DirectEmulatorRunner(rom_path=rom_path, load_state=load_state, story_bucket=story_bucket, recorder=recorder)
        runner.initialize()
        failures: list[dict] = []

        settle_state = runner.state()
        if settle:
            settle_start = runner.frame_idx
            settle_pre = settle_state
            settle_state = runner.settle_to_free_overworld(
                max_actions=settle_actions,
                metadata={"story_bucket": story_bucket, "load_state": load_state},
            )
            if runner.frame_idx != settle_start or settle_state.control_mode != settle_pre.control_mode:
                recorder.record_segment(
                    {
                        "segment_type": "explore_settle",
                        "story_bucket": story_bucket,
                        "load_state": load_state,
                        "frame_range": [settle_start, runner.frame_idx],
                        "start_state": settle_pre.to_dict(),
                        "end_state": settle_state.to_dict(),
                        "validation": "passed" if settle_state.control_mode == "free_overworld" else "failed",
                    }
                )
        if settle_state.control_mode != "free_overworld":
            failures.append(
                {
                    "prefix": settle_state.explore_key_prefix,
                    "reason": "start_not_free_overworld",
                    "control_mode": settle_state.control_mode,
                }
            )

        start_bytes = runner.save_state_bytes()
        if start_bytes is None:
            raise RuntimeError("Could not save initial emulator state")

        queue = deque([(start_bytes, runner.facing)])
        visited_prefixes: set[str] = set()
        seen_keys: set[str] = set()

        while queue and len(visited_prefixes) < max_states:
            state_bytes, facing = queue.popleft()
            runner.facing = facing
            runner.load_state_bytes(state_bytes, record=False)
            pre = runner.state()
            if pre.control_mode != "free_overworld":
                continue
            if pre.explore_key_prefix in visited_prefixes:
                continue
            visited_prefixes.add(pre.explore_key_prefix)
            pre_bytes = runner.save_state_bytes()
            if pre_bytes is None:
                failures.append({"prefix": pre.explore_key_prefix, "reason": "save_state_failed"})
                continue

            for action in actions:
                runner.load_state_bytes(pre_bytes, record=False)
                runner.facing = facing
                pre_action_state = runner.state()
                key = pre_action_state.explore_key(action)
                if key in seen_keys:
                    continue
                segment_start = runner.frame_idx
                try:
                    runner.perform_action(action, metadata={"segment_type": "explore", "key": key})
                    post = runner.wait_until_stable(metadata={"segment_type": "explore", "key": key})
                    label = result_label(pre_action_state, post)
                    recorder.record_segment(
                        {
                            "segment_type": "explore_transition",
                            "key": key,
                            "story_bucket": story_bucket,
                            "action": action,
                            "result": label,
                            "frame_range": [segment_start, runner.frame_idx],
                            "pre_state_hash": pre_action_state.raw_state_hash,
                            "post_state_hash": post.raw_state_hash,
                            "pre_state": pre_action_state.to_dict(),
                            "post_state": post.to_dict(),
                            "validation": "passed",
                        }
                    )
                    seen_keys.add(key)
                    if post.control_mode == "free_overworld" and label in {"move", "turn_only"}:
                        post_bytes = runner.save_state_bytes()
                        if post_bytes and post.explore_key_prefix not in visited_prefixes and len(visited_prefixes) + len(queue) < max_states:
                            queue.append((post_bytes, post.facing))
                except Exception as exc:
                    failures.append({"key": key, "reason": str(exc)})
                    recorder.record_segment(
                        {
                            "segment_type": "explore_transition",
                            "key": key,
                            "story_bucket": story_bucket,
                            "action": action,
                            "result": "failed",
                            "frame_range": [segment_start, runner.frame_idx],
                            "validation": "failed",
                            "error": str(exc),
                        }
                    )
        runner.close()

    coverage = {
        "story_bucket": story_bucket,
        "visited_state_count": len(visited_prefixes),
        "transition_count": len(seen_keys),
        "failure_count": len(failures),
        "failures": failures,
    }
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True), encoding="utf-8")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-state", required=True)
    parser.add_argument("--story-bucket", required=True)
    parser.add_argument("--map", default=None, help="Accepted for CLI compatibility; pilot BFS starts from the load-state map")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rom-path", default="Emerald-GBAdvance/rom.gba")
    parser.add_argument("--backend", default="auto", choices=["auto", "ffv1", "npz"])
    parser.add_argument("--visual-fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=32)
    parser.add_argument("--settle-actions", type=int, default=24)
    parser.add_argument("--no-settle", action="store_true")
    args = parser.parse_args()
    result = collect_explore(
        load_state=args.load_state,
        story_bucket=args.story_bucket,
        output_dir=args.output,
        rom_path=args.rom_path,
        backend=args.backend,
        visual_fps=args.visual_fps,
        max_states=args.max_states,
        settle=not args.no_settle,
        settle_actions=args.settle_actions,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
