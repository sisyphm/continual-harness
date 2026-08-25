"""runsup — the run supervisor: PID-file process control for collection runs.

Six times in one week a pkill pattern matched the shell that typed it and killed the
session (exit 144); twice the opposite — an awk-kill missed two A/B arms, which then
triple-loaded the box for an hour and contaminated a bisect round. Both failure modes
are pattern-matching on cmdlines. This tool removes the pattern: every run gets a PID
file at launch, and stop/ls/reap operate on those exact PIDs only.

PID reuse is guarded: the launch records /proc/<pid>/stat field 22 (starttime in
clock ticks, unique per boot per pid incarnation) and stop refuses to signal a pid
whose starttime no longer matches.

    runsup launch --name bv2_sweep [--log path] -- env FOO=1 cmd args...
    runsup ls
    runsup stop <name> [--sig KILL]
    runsup stop --all
    runsup reap            # drop dead entries

State: data/runsup/<name>.json ; default log: data/runsup/<name>.log
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[2] / "data" / "runsup"


def _starttime(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # field 22, counting from 1, AFTER the parenthesised comm (which may contain
        # spaces) — split on the closing paren, not on whitespace alone.
        return int(stat.rsplit(")", 1)[1].split()[19])
    except Exception:
        return None


def _load(name: str) -> dict | None:
    p = STATE_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _alive(rec: dict) -> bool:
    st = _starttime(rec["pid"])
    return st is not None and st == rec["starttime"]


def launch(args) -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    rec = _load(args.name)
    if rec and _alive(rec):
        print(f"refusing: '{args.name}' is already running (pid {rec['pid']})")
        return 1
    log = Path(args.log) if args.log else STATE_DIR / f"{args.name}.log"
    lf = open(log, "ab", buffering=0)
    proc = subprocess.Popen(
        args.cmd, stdout=lf, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True, cwd=args.cwd or None)
    st = _starttime(proc.pid)
    (STATE_DIR / f"{args.name}.json").write_text(json.dumps({
        "name": args.name, "pid": proc.pid, "starttime": st,
        "cmd": args.cmd, "log": str(log), "cwd": args.cwd or os.getcwd(),
        "launched": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1))
    print(f"launched '{args.name}' pid={proc.pid} log={log}")
    return 0


def ls(args) -> int:
    rows = []
    for p in sorted(STATE_DIR.glob("*.json")):
        rec = json.loads(p.read_text())
        alive = _alive(rec)
        tail = ""
        try:
            with open(rec["log"], "rb") as f:
                f.seek(max(0, os.path.getsize(rec["log"]) - 400))
                lines = [l for l in f.read().decode(errors="replace").splitlines() if l.strip()]
                tail = lines[-1][-90:] if lines else ""
        except Exception:
            pass
        rows.append((rec["name"], rec["pid"], "RUN " if alive else "dead", rec["launched"], tail))
    if not rows:
        print("(no supervised runs)")
        return 0
    w = max(len(r[0]) for r in rows)
    for name, pid, st, when, tail in rows:
        print(f"{name:<{w}}  {pid:>7}  {st}  {when}  {tail}")
    return 0


def stop(args) -> int:
    names = ([p.stem for p in STATE_DIR.glob("*.json")] if args.all else [args.name])
    if not names or names == [None]:
        print("stop: give a name or --all")
        return 1
    rc = 0
    for name in names:
        rec = _load(name)
        if rec is None:
            print(f"{name}: no record")
            rc = 1
            continue
        if not _alive(rec):
            print(f"{name}: already dead (pid {rec['pid']})")
            continue
        sig = getattr(signal, f"SIG{args.sig}")
        # signal the whole session (start_new_session at launch): children included,
        # nothing outside the group ever matched — the anti-pkill contract.
        try:
            os.killpg(rec["pid"], sig)
        except ProcessLookupError:
            os.kill(rec["pid"], sig)
        for _ in range(50):
            if not _alive(rec):
                break
            time.sleep(0.1)
        print(f"{name}: sent SIG{args.sig} to pgid {rec['pid']} "
              f"({'exited' if not _alive(rec) else 'STILL ALIVE'})")
    return rc


def reap(args) -> int:
    for p in sorted(STATE_DIR.glob("*.json")):
        rec = json.loads(p.read_text())
        if not _alive(rec):
            p.unlink()
            print(f"reaped {rec['name']} (pid {rec['pid']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="runsup")
    sub = ap.add_subparsers(dest="op", required=True)
    l = sub.add_parser("launch")
    l.add_argument("--name", required=True)
    l.add_argument("--log")
    l.add_argument("--cwd")
    l.add_argument("cmd", nargs=argparse.REMAINDER)
    s = sub.add_parser("stop")
    s.add_argument("name", nargs="?")
    s.add_argument("--all", action="store_true")
    s.add_argument("--sig", default="TERM", choices=["TERM", "KILL", "INT"])
    sub.add_parser("ls")
    sub.add_parser("reap")
    args = ap.parse_args()
    if args.op == "launch":
        args.cmd = [c for c in args.cmd if c != "--"] and (
            args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd)
        if not args.cmd:
            print("launch: nothing to run after --")
            return 1
        return launch(args)
    return {"ls": ls, "stop": stop, "reap": reap}[args.op](args)


if __name__ == "__main__":
    sys.exit(main())
