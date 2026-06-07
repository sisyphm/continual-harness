"""Audit Heatz event checkpoints before collecting event data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from collection.actions import EXPLORE_ACTIONS
from collection.catalog import discover_heatz_events
from collection.collect_events import collect_one_event
from collection.direct_runner import DirectEmulatorRunner
from collection.recorder import ChunkRecorder
from collection.state import result_label


def _target_reached(state, expected_state) -> bool:
    if expected_state is None or state.control_mode != "free_overworld":
        return False
    if state.map != expected_state.map:
        return False
    if state.x is None or state.y is None or expected_state.x is None or expected_state.y is None:
        return True
    return abs(state.x - expected_state.x) + abs(state.y - expected_state.y) <= 1


def _load_expected_state(*, rom_path: str, completed_state: str | None, event_id: str):
    if not completed_state:
        return None
    runner = DirectEmulatorRunner(rom_path=rom_path, load_state=completed_state, story_bucket=event_id)
    try:
        runner.initialize()
        return runner.state()
    finally:
        runner.close()


def _probe_actions(runner: DirectEmulatorRunner, base_state_bytes: bytes, base_facing: str, pre_state) -> list[dict[str, Any]]:
    probes = []
    for action in [*EXPLORE_ACTIONS, "A", "B", "WAIT"]:
        runner.load_state_bytes(base_state_bytes, record=False)
        runner.facing = base_facing
        segment_start = runner.frame_idx
        try:
            runner.perform_action(action, metadata={"audit_probe": True, "action": action})
            post = runner.wait_until_stable(max_frames=64, stable_frames=4, metadata={"audit_probe": True, "action": action})
            probes.append(
                {
                    "action": action,
                    "result": result_label(pre_state, post),
                    "frame_range": [segment_start, runner.frame_idx],
                    "post_state": post.to_dict(),
                }
            )
        except Exception as exc:
            probes.append({"action": action, "result": "error", "error": str(exc), "frame_range": [segment_start, runner.frame_idx]})
    runner.load_state_bytes(base_state_bytes, record=False)
    runner.facing = base_facing
    return probes


def _recommendation(*, start_state, expected_state, probes: list[dict[str, Any]], policy_validation: str | None) -> tuple[str, str]:
    if start_state.control_mode not in {"free_overworld", "dialogue", "battle"}:
        return "skip", f"unsupported_start_mode:{start_state.control_mode}"
    if _target_reached(start_state, expected_state):
        return "skip", "already_complete_start"
    directional = [probe for probe in probes if probe.get("action") in EXPLORE_ACTIONS]
    ui = [probe for probe in probes if probe.get("action") in {"A", "B", "WAIT"}]
    directional_progress = any(probe.get("result") in {"move", "warp", "battle_start", "dialogue_open", "event_trigger"} for probe in directional)
    any_input_effect = any(probe.get("result") not in {"blocked", "error"} for probe in probes)
    only_turns_or_blocks = all(probe.get("result") in {"turn_only", "blocked", "error"} for probe in directional)
    ui_no_progress = all(probe.get("result") in {"blocked", "turn_only", "error"} for probe in ui)
    if not any_input_effect:
        return "skip", "no_input_effect"
    if not directional_progress:
        if only_turns_or_blocks and ui_no_progress:
            return "skip", "script_or_dialog_control_lock"
        return "skip", "directionally_stuck"
    if policy_validation == "passed":
        return "collect", "policy_validated"
    if policy_validation in {"failed", "skipped"}:
        return "skip", f"policy_{policy_validation}"
    return "candidate", "precheck_only"


def audit_one_event(
    *,
    event: dict,
    policy_dir: str,
    rom_path: str,
    output_dir: str,
    backend: str,
    visual_fps: int,
    run_policy: bool,
    max_actions: int,
    min_actions: int,
) -> dict[str, Any]:
    event_id = event["event_id"]
    row: dict[str, Any] = {
        "event_id": event_id,
        "pre_milestone": event.get("pre_milestone"),
        "pre_state": event.get("pre_state"),
        "completed_state": event.get("completed_state"),
        "postcondition": event.get("postcondition"),
    }
    if not event.get("pre_state") or not event.get("completed_state"):
        row.update({"recommendation": "skip", "reason": "missing_state"})
        return row

    expected_state = _load_expected_state(rom_path=rom_path, completed_state=event.get("completed_state"), event_id=event_id)
    runner = DirectEmulatorRunner(rom_path=rom_path, load_state=event["pre_state"], story_bucket=event_id)
    try:
        runner.initialize()
        start_state = runner.state()
        row["start_state"] = start_state.to_dict()
        row["expected_state"] = expected_state.to_dict() if expected_state else None
        row["already_complete_start"] = _target_reached(start_state, expected_state)
        base_state_bytes = runner.save_state_bytes()
        if base_state_bytes is None:
            row.update({"recommendation": "skip", "reason": "save_state_failed"})
            return row
        probes = _probe_actions(runner, base_state_bytes, runner.facing, start_state)
        row["action_probes"] = probes
    finally:
        runner.close()

    policy_validation = None
    policy_result = None
    if run_policy:
        policy_result = collect_one_event(
            event_id=event_id,
            policy_dir=policy_dir,
            pre_state=event["pre_state"],
            completed_state=event.get("completed_state"),
            output_dir=str(Path(output_dir) / "policy_smoke"),
            rom_path=rom_path,
            backend=backend,
            visual_fps=visual_fps,
            max_actions=max_actions,
            min_actions=min_actions,
        )
        policy_validation = policy_result.get("validation")
        row["policy_result"] = policy_result

    recommendation, reason = _recommendation(
        start_state=start_state,
        expected_state=expected_state,
        probes=row["action_probes"],
        policy_validation=policy_validation,
    )
    row["recommendation"] = recommendation
    row["reason"] = reason
    return row


def write_audit(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "event_audit.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "event_count": len(rows),
        "collect_count": sum(1 for row in rows if row.get("recommendation") == "collect"),
        "candidate_count": sum(1 for row in rows if row.get("recommendation") == "candidate"),
        "skip_count": sum(1 for row in rows if row.get("recommendation") == "skip"),
        "reasons": {},
        "audit_jsonl": str(jsonl_path),
    }
    for row in rows:
        reason = row.get("reason") or "unknown"
        summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heatz-policy-dir", required=True)
    parser.add_argument("--event", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rom-path", default="Emerald-GBAdvance/rom.gba")
    parser.add_argument("--backend", default="npz", choices=["auto", "ffv1", "npz"])
    parser.add_argument("--visual-fps", type=int, default=10)
    parser.add_argument("--run-policy", action="store_true")
    parser.add_argument("--max-actions", type=int, default=120)
    parser.add_argument("--min-actions", type=int, default=1)
    args = parser.parse_args()

    events = discover_heatz_events(args.heatz_policy_dir)
    if args.event:
        events = [event for event in events if event["event_id"] == args.event]
    rows = []
    for event in events:
        row = audit_one_event(
            event=event,
            policy_dir=args.heatz_policy_dir,
            rom_path=args.rom_path,
            output_dir=args.output,
            backend=args.backend,
            visual_fps=args.visual_fps,
            run_policy=args.run_policy,
            max_actions=args.max_actions,
            min_actions=args.min_actions,
        )
        print(json.dumps({"event_id": row.get("event_id"), "recommendation": row.get("recommendation"), "reason": row.get("reason")}, sort_keys=True))
        rows.append(row)
    print(json.dumps(write_audit(rows, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
