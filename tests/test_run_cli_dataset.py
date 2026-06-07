from types import SimpleNamespace

import run_cli


class _FakeProcess:
    pid = 12345


def test_start_server_forwards_dataset_and_termination_flags(monkeypatch):
    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(run_cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_cli.time, "sleep", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        port=8123,
        game="emerald",
        record=False,
        collect_dataset=True,
        dataset_output_dir="data/cli_episode",
        dataset_state_interval=3,
        dataset_max_seconds=123.5,
        dataset_frame_writer_workers=8,
        dataset_png_compress_level=0,
        dataset_compact_state_mode="fast",
        terminate_on_first_badge=True,
        load_checkpoint=False,
        load_state="Emerald-GBAdvance/truck_start.state",
        no_ocr=True,
        direct_objectives="categorized_full_game",
        direct_objectives_start=2,
        direct_objectives_battling_start=4,
    )

    process = run_cli.start_server(args, run_id="run_cli_dataset_test")

    assert process is not None
    cmd = captured["cmd"]
    assert cmd[:4] == [run_cli.sys.executable, "-m", "server.app", "--port"]
    assert "--collect-dataset" in cmd
    assert cmd[cmd.index("--dataset-output-dir") + 1] == "data/cli_episode"
    assert cmd[cmd.index("--dataset-state-interval") + 1] == "3"
    assert cmd[cmd.index("--dataset-max-seconds") + 1] == "123.5"
    assert cmd[cmd.index("--dataset-frame-writer-workers") + 1] == "8"
    assert cmd[cmd.index("--dataset-png-compress-level") + 1] == "0"
    assert cmd[cmd.index("--dataset-compact-state-mode") + 1] == "fast"
    assert "--terminate-on-first-badge" in cmd
    assert cmd[cmd.index("--load-state") + 1] == "Emerald-GBAdvance/truck_start.state"
    assert cmd[cmd.index("--direct-objectives") + 1] == "categorized_full_game"
    assert cmd[cmd.index("--direct-objectives-start") + 1] == "2"
    assert cmd[cmd.index("--direct-objectives-battling-start") + 1] == "4"
    assert "--no-ocr" in cmd

    env = captured["kwargs"]["env"]
    assert env["RUN_DATA_ID"] == "run_cli_dataset_test"
    assert env["POKEAGENT_CLI_MODE"] == "1"
    assert env["GAME_TYPE"] == "emerald"
    assert env["HAS_DIRECT_OBJECTIVES"] == "1"
    assert env["LLM_METRICS_WRITE_ENABLED"] == "true"
    assert env["POKEMON_STATE_API_KEY"]


def test_start_services_no_container_skips_mcp_sse(monkeypatch):
    calls = {"mcp": 0}

    class _RunManager:
        run_id = "run_cli_local_test"

        def get_run_directory(self):
            return "run_data/run_cli_local_test"

    def fake_start_server(args, run_id):
        assert run_id == "run_cli_local_test"
        return _FakeProcess()

    def fake_start_frame_server(port):
        return _FakeProcess()

    def fail_start_mcp(*_args, **_kwargs):
        calls["mcp"] += 1
        raise AssertionError("MCP SSE server should not start in --no-container mode")

    monkeypatch.setattr(run_cli, "start_server", fake_start_server)
    monkeypatch.setattr(run_cli, "start_frame_server", fake_start_frame_server)
    monkeypatch.setattr(run_cli, "start_mcp_sse_server", fail_start_mcp)

    args = SimpleNamespace(port=8123, mcp_sse_port=None, no_container=True)
    services = run_cli._start_services(args, _RunManager())

    assert services is not None
    assert services.mcp_process is None
    assert services.server_url == "http://localhost:8123"
    assert calls["mcp"] == 0


def test_codex_local_launch_uses_absolute_workdir(tmp_path):
    from utils.agent_infrastructure.cli_agent_backends import CodexCliBackend

    directive = tmp_path / "directive.md"
    directive.write_text("Act in the Pokemon environment.")
    working_dir = tmp_path / "scratch"
    memory_dir = tmp_path / "codex_home"

    backend = CodexCliBackend()
    cmd, env, _bootstrap, _temp = backend.build_launch_cmd(
        str(directive),
        "http://localhost:8123",
        str(working_dir),
        project_root="/repo/continual-harness",
        containerized=False,
        agent_memory_dir=str(memory_dir),
        agent_model="gpt-5.4-mini",
    )

    assert cmd[:2] == ["sh", "-c"]
    assert f"-C {working_dir.resolve()}" in cmd[2]
    assert "-m gpt-5.4-mini" in cmd[2]
    assert env["CODEX_HOME"] == str(memory_dir.resolve())
    assert (memory_dir / "config.toml").exists()


def test_codex_local_resume_passes_continue_prompt(tmp_path):
    from utils.agent_infrastructure.cli_agent_backends import CodexCliBackend

    directive = tmp_path / "directive.md"
    directive.write_text("Act in the Pokemon environment.")
    working_dir = tmp_path / "scratch"
    memory_dir = tmp_path / "codex_home"

    backend = CodexCliBackend()
    cmd, env, _bootstrap, _temp = backend.build_launch_cmd(
        str(directive),
        "http://localhost:8123",
        str(working_dir),
        project_root="/repo/continual-harness",
        containerized=False,
        agent_memory_dir=str(memory_dir),
        resume_session_id="019e9810-5e80-7220-89e0-0db1c3c65525",
        thinking_effort="low",
        agent_model="gpt-5.4-mini",
    )

    assert cmd[:2] == ["sh", "-c"]
    assert "printf '%s\\n'" in cmd[2]
    assert "Continue autonomously" in cmd[2]
    assert "codex exec resume 019e9810-5e80-7220-89e0-0db1c3c65525" in cmd[2]
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd[2]
    assert "-m gpt-5.4-mini" in cmd[2]
    assert "-c model_reasoning_effort=low" in cmd[2]
    assert cmd[2].endswith(" -")
    assert env["CODEX_HOME"] == str(memory_dir.resolve())
