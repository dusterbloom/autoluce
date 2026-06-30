"""
Tests for the unified autoggml CLI.

The CLI is a thin router: it maps a subcommand to one of the existing scripts and
dispatches it as a subprocess (each script keeps its own argparse; nothing is
reimplemented). resolve() is the pure, tested routing logic; main() is the thin
dispatcher with an injectable runner so the wiring is verified without spawning real
builds/benchmarks.
"""

import pytest

import cli
from cli import COMMANDS, CliError, CliHelp, main, resolve


def test_resolve_known_command_passthrough_args():
    script, args = resolve(["ideas", "--bound", "memory"])
    assert script == "ideas.py"
    assert args == ["--bound", "memory"]


def test_resolve_baseline_injects_default_flag_then_passthrough():
    script, args = resolve(["baseline", "--simulate"])
    assert script == "harness.py"
    assert args == ["--baseline", "--simulate"]  # default flag first, user args after


def test_resolve_run_with_passthrough():
    script, args = resolve(["run", "--dry-run"])
    assert script == "agent_loop.py"
    assert args == ["--dry-run"]


def test_resolve_setup_maps_to_prepare():
    script, _ = resolve(["setup"])
    assert script == "prepare.py"


@pytest.mark.parametrize("cmd", sorted(COMMANDS))
def test_every_advertised_command_resolves(cmd):
    script, _ = resolve([cmd])
    assert script.endswith(".py")


def test_resolve_empty_raises_help():
    with pytest.raises(CliHelp):
        resolve([])


def test_resolve_help_flags_raise_help():
    for token in ("help", "--help", "-h"):
        with pytest.raises(CliHelp):
            resolve([token])


def test_resolve_unknown_raises_error():
    with pytest.raises(CliError):
        resolve(["bogus"])


# --- main wiring (injected runner, no real subprocess) ------------------------

class _Result:
    def __init__(self, returncode):
        self.returncode = returncode


def test_main_dispatches_to_resolved_script_and_propagates_returncode(monkeypatch):
    calls = []

    def fake_runner(invocation):
        calls.append(invocation)
        return _Result(7)

    rc = main(["ideas", "--bound", "compute"], runner=fake_runner)
    assert rc == 7
    assert len(calls) == 1
    assert calls[0][1].endswith("ideas.py")   # [python, script, *args]
    assert calls[0][2:] == ["--bound", "compute"]


def test_main_unknown_command_returns_nonzero(capsys):
    rc = main(["bogus"], runner=lambda inv: _Result(0))
    assert rc != 0


def test_main_help_returns_zero(capsys):
    rc = main([], runner=lambda inv: _Result(0))
    assert rc == 0


def test_scripts_dir_constant_present():
    # Ensure the dispatcher actually knows where the scripts live.
    assert cli.SCRIPTS is not None
