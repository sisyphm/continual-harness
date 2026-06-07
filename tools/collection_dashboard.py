#!/usr/bin/env python3
"""Lightweight realtime dashboard for parallel Pokemon collection runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def tail_last_line(path: Path, block_size: int = 131072) -> str | None:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - block_size))
            lines = f.read().splitlines()
        for line in reversed(lines):
            if line.strip():
                return line.decode("utf-8", "replace")
    except OSError:
        return None
    return None


def tail_text(path: Path, block_size: int = 131072) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - block_size))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def load_json(path: Path) -> dict:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def load_jsonl_tail(path: Path) -> dict:
    line = tail_last_line(path)
    if not line:
        return {}
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {}


def is_port_open(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.12)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def parse_log_time(line: str) -> str | None:
    match = TS_RE.match(line)
    return match.group(1) if match else None


def latest_press_command(log_path: Path) -> dict:
    text = tail_text(log_path, 262144)
    if not text:
        return {}
    lines = text.splitlines()
    for line in reversed(lines):
        if "[codex:mcp_tool_call] press_buttons" not in line:
            continue
        result = {"time": parse_log_time(line), "raw": line[-500:]}
        marker = "press_buttons"
        idx = line.find(marker)
        payload = line[idx + len(marker):].strip() if idx >= 0 else ""
        if payload.startswith("{"):
            try:
                data = json.loads(payload)
                result["buttons"] = data.get("buttons")
                result["speed"] = data.get("speed")
                reason = data.get("reasoning") or ""
                result["reasoning"] = reason[:180]
            except json.JSONDecodeError:
                result["raw_payload"] = payload[:220]
        return result
    return {}


def latest_log_health(log_path: Path) -> dict:
    text = tail_text(log_path, 131072)
    low = text.lower()
    return {
        "timeout_warning": "read timed out" in low,
        "fatal": any(
            needle in low
            for needle in ("traceback", "fatal", "uncaught exception", "model_not_found", "invalid model")
        ),
    }


def summarize_party(party: object) -> str:
    if not isinstance(party, list):
        return ""
    names: list[str] = []
    for mon in party[:3]:
        if not isinstance(mon, dict):
            continue
        species = mon.get("species") or mon.get("name") or "?"
        level = mon.get("level")
        hp = mon.get("hp")
        max_hp = mon.get("max_hp")
        if level is not None and hp is not None and max_hp is not None:
            names.append(f"{species} L{level} {hp}/{max_hp}")
        elif level is not None:
            names.append(f"{species} L{level}")
        else:
            names.append(str(species))
    return ", ".join(names)


class Dashboard:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = (ROOT / args.data_dir).resolve()
        self.logs_dir = (ROOT / args.logs_dir).resolve()

    def run_dirs(self) -> list[Path]:
        return [self.data_dir / f"run_{idx:02d}" / "episode_000001" for idx in range(1, self.args.runs + 1)]

    def status(self) -> dict:
        rows = []
        location_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}

        for idx, episode_dir in enumerate(self.run_dirs(), 1):
            run_id = f"run_{idx:02d}"
            port = self.args.base_port + (idx - 1) * self.args.port_stride
            manifest = load_json(episode_dir / "manifest.json")
            state_row = load_jsonl_tail(episode_dir / "states.jsonl")
            action_row = load_jsonl_tail(episode_dir / "actions.jsonl")
            state = state_row.get("state") if isinstance(state_row.get("state"), dict) else state_row
            if not isinstance(state, dict):
                state = {}

            log_matches = sorted(self.logs_dir.glob(f"{self.args.log_prefix}_{idx:02d}_*.log"))
            log_path = log_matches[-1] if log_matches else None
            press = latest_press_command(log_path) if log_path else {}
            log_health = latest_log_health(log_path) if log_path else {}

            frame_idx = state.get("frame_idx") or action_row.get("frame_idx")
            fps_target = manifest.get("fps_target") or 80
            elapsed_game_seconds = None
            if isinstance(frame_idx, (int, float)) and fps_target:
                elapsed_game_seconds = frame_idx / fps_target

            location = state.get("location") or state.get("location_name") or state.get("map") or "unknown"
            status = manifest.get("status") or "unknown"
            location_counts[location] = location_counts.get(location, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1

            rows.append(
                {
                    "run": run_id,
                    "port": port,
                    "stream_url": f"http://127.0.0.1:{port}/stream",
                    "server_live": is_port_open(port),
                    "frame_live": is_port_open(port + 1),
                    "status": status,
                    "end_reason": manifest.get("end_reason"),
                    "success": manifest.get("success"),
                    "frame_idx": frame_idx,
                    "action_frame_idx": action_row.get("frame_idx"),
                    "elapsed_game_seconds": elapsed_game_seconds,
                    "location": location,
                    "milestone": state.get("milestone"),
                    "badges": state.get("badges", state.get("badge_count")),
                    "game_state": state.get("game_state"),
                    "in_battle": state.get("in_battle"),
                    "dialogue": state.get("dialogue"),
                    "coords": [state.get("x"), state.get("y")],
                    "money": state.get("money"),
                    "party": summarize_party(state.get("party") or state.get("party_pokemon")),
                    "current_buttons": action_row.get("buttons_held") or [],
                    "phase": action_row.get("phase"),
                    "queue_length": action_row.get("queue_length"),
                    "last_press": press,
                    "log_health": log_health,
                }
            )

        return {
            "batch": self.args.batch_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runs": rows,
            "location_counts": location_counts,
            "status_counts": status_counts,
            "watch_url": f"http://127.0.0.1:{self.args.base_port}/stream",
        }

    def html(self) -> str:
        return HTML.replace("__BATCH__", self.args.batch_name).replace(
            "__DEFAULT_STREAM__", f"http://127.0.0.1:{self.args.base_port}/stream"
        )


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pokemon Collection Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f7;
      --panel: #ffffff;
      --line: #d5dcde;
      --text: #1d282b;
      --muted: #5c6a70;
      --green: #16855d;
      --amber: #a56a00;
      --red: #b3263a;
      --blue: #1f6fb2;
      --row: #eef5f6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }
    header {
      height: 54px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 17px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .meta {
      display: flex;
      align-items: center;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--green);
      display: inline-block;
      margin-right: 5px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(410px, 42vw) minmax(560px, 1fr);
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 54px);
    }
    .viewer, .table-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
    }
    .viewer-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 42px;
      padding: 0 12px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
    }
    #selectedRun {
      color: var(--text);
      font-weight: 700;
    }
    .stream-frame {
      width: 100%;
      height: calc(100vh - 112px);
      min-height: 520px;
      border: 0;
      display: block;
      background: #101417;
    }
    .summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      background: #fbfcfc;
      color: var(--muted);
      font-size: 12px;
    }
    .table-wrap {
      overflow: auto;
      max-height: calc(100vh - 120px);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1060px;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #edf2f3;
      color: #334248;
      font-size: 12px;
      font-weight: 700;
    }
    tbody tr {
      cursor: pointer;
    }
    tbody tr:hover, tbody tr.selected {
      background: var(--row);
    }
    .run {
      font-weight: 700;
      color: var(--blue);
      white-space: nowrap;
    }
    .ok { color: var(--green); font-weight: 700; }
    .warn { color: var(--amber); font-weight: 700; }
    .bad { color: var(--red); font-weight: 700; }
    .muted { color: var(--muted); }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    .reason {
      max-width: 290px;
      color: var(--muted);
      line-height: 1.35;
    }
    button, a.button {
      border: 1px solid var(--line);
      background: #ffffff;
      border-radius: 6px;
      padding: 6px 9px;
      color: var(--text);
      text-decoration: none;
      font: inherit;
      cursor: pointer;
    }
    button:hover, a.button:hover {
      border-color: #98aab0;
    }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      .stream-frame { height: 520px; }
      .table-wrap { max-height: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1>__BATCH__</h1>
    <div class="meta">
      <span><span class="dot" id="liveDot"></span><span id="healthText">loading</span></span>
      <span id="updatedAt">not updated</span>
      <button id="refreshButton" type="button">Refresh</button>
    </div>
  </header>
  <main>
    <section class="viewer">
      <div class="viewer-head">
        <div>Selected stream: <span id="selectedRun">run_01</span></div>
        <a class="button" id="openStream" href="__DEFAULT_STREAM__" target="_blank" rel="noreferrer">Open</a>
      </div>
      <iframe id="streamFrame" class="stream-frame" src="__DEFAULT_STREAM__"></iframe>
    </section>
    <section class="table-panel">
      <div class="summary" id="summary"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th>Live</th>
              <th>Status</th>
              <th>Location</th>
              <th>Frame</th>
              <th>Game Time</th>
              <th>State</th>
              <th>Party</th>
              <th>Last Command</th>
              <th>Log</th>
            </tr>
          </thead>
          <tbody id="runsBody"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    let selectedRun = localStorage.getItem("selectedRun") || "run_01";
    let rows = [];

    function fmtInt(value) {
      if (value === null || value === undefined) return "";
      return Number(value).toLocaleString();
    }

    function fmtTime(seconds) {
      if (seconds === null || seconds === undefined) return "";
      const total = Math.floor(seconds);
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
      }[ch]));
    }

    function setSelected(row) {
      selectedRun = row.run;
      localStorage.setItem("selectedRun", selectedRun);
      document.getElementById("selectedRun").textContent = row.run;
      document.getElementById("streamFrame").src = row.stream_url;
      document.getElementById("openStream").href = row.stream_url;
      renderRows();
    }

    function renderSummary(data) {
      const summary = document.getElementById("summary");
      const locs = Object.entries(data.location_counts || {}).sort((a, b) => b[1] - a[1]);
      const statuses = Object.entries(data.status_counts || {}).sort((a, b) => b[1] - a[1]);
      const parts = [];
      for (const [status, count] of statuses) parts.push(`${count} ${status}`);
      for (const [loc, count] of locs) parts.push(`${count} ${loc}`);
      summary.innerHTML = parts.map(item => `<span class="pill">${esc(item)}</span>`).join("");
    }

    function renderRows() {
      const body = document.getElementById("runsBody");
      body.innerHTML = rows.map(row => {
        const liveClass = row.server_live && row.frame_live ? "ok" : "bad";
        const liveText = row.server_live && row.frame_live ? "up" : "down";
        const badges = row.badges ?? 0;
        const coords = Array.isArray(row.coords) && row.coords.some(v => v !== null && v !== undefined)
          ? ` @ ${row.coords[0] ?? "?"},${row.coords[1] ?? "?"}` : "";
        const press = row.last_press || {};
        const buttons = Array.isArray(press.buttons) ? press.buttons.join(" ") : "";
        const command = buttons ? `${buttons}${press.speed ? ` (${press.speed})` : ""}` : "";
        const health = row.log_health || {};
        const logStatus = health.fatal ? "fatal" : health.timeout_warning ? "timeout" : "ok";
        const logClass = health.fatal ? "bad" : health.timeout_warning ? "warn" : "ok";
        const selected = row.run === selectedRun ? "selected" : "";
        return `<tr class="${selected}" data-run="${esc(row.run)}">
          <td><span class="run">${esc(row.run)}</span><br><span class="muted mono">:${row.port}</span></td>
          <td><span class="${liveClass}">${liveText}</span></td>
          <td>${esc(row.status)}<br><span class="muted">badges ${esc(badges)}</span></td>
          <td>${esc(row.location)}<br><span class="muted">${esc(row.milestone || "")}${esc(coords)}</span></td>
          <td class="mono">${fmtInt(row.frame_idx)}</td>
          <td class="mono">${fmtTime(row.elapsed_game_seconds)}</td>
          <td>${esc(row.game_state || "")}<br><span class="muted">battle ${row.in_battle ? "yes" : "no"} / dialogue ${row.dialogue ? "yes" : "no"}</span></td>
          <td>${esc(row.party || "")}</td>
          <td><span class="mono">${esc(command)}</span><div class="reason">${esc(press.reasoning || "")}</div></td>
          <td><span class="${logClass}">${logStatus}</span><br><span class="muted mono">${esc(press.time || "")}</span></td>
        </tr>`;
      }).join("");
      for (const tr of body.querySelectorAll("tr")) {
        tr.addEventListener("click", () => {
          const row = rows.find(item => item.run === tr.dataset.run);
          if (row) setSelected(row);
        });
      }
    }

    async function refresh() {
      try {
        const response = await fetch(`/api/status?ts=${Date.now()}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        rows = data.runs || [];
        renderSummary(data);
        renderRows();
        const selected = rows.find(row => row.run === selectedRun) || rows[0];
        if (selected && document.getElementById("streamFrame").src !== selected.stream_url) {
          document.getElementById("selectedRun").textContent = selected.run;
          document.getElementById("openStream").href = selected.stream_url;
        }
        const updated = new Date(data.generated_at);
        document.getElementById("updatedAt").textContent = `updated ${updated.toLocaleTimeString()}`;
        document.getElementById("healthText").textContent = "live";
        document.getElementById("liveDot").style.background = "var(--green)";
      } catch (error) {
        document.getElementById("healthText").textContent = `error: ${error.message}`;
        document.getElementById("liveDot").style.background = "var(--red)";
      }
    }

    document.getElementById("refreshButton").addEventListener("click", refresh);
    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


def make_handler(dashboard: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            if self.path.startswith("/api/status"):
                return
            super().log_message(fmt, *args)

        def send_json(self, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/status":
                self.send_json(dashboard.status())
                return
            if path in ("/", "/dashboard"):
                self.send_html(dashboard.html())
                return
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime dashboard for collection runs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--data-dir", default="data/emerald_codex_16_gpt54")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--batch-name", default="emerald_codex_16_gpt54")
    parser.add_argument("--log-prefix", default="emerald_codex_16_gpt54")
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--port-stride", type=int, default=10)
    parser.add_argument("--runs", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard = Dashboard(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dashboard))
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print(f"Data dir: {dashboard.data_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
