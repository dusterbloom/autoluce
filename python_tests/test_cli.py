"""
Tests for the unified autoluce CLI.

The CLI is a thin router: it maps a subcommand to a dotted module path inside the
autoluce package and dispatches it as a subprocess (`python -m <module>`; each module
keeps its own argparse; nothing is reimplemented). resolve() is the pure, tested routing
logic; main() is the thin dispatcher with an injectable runner so the wiring is verified
without spawning real builds/benchmarks.
"""

import importlib.util

import pytest

from cli import COMMANDS, CliError, CliHelp, main, resolve


def test_resolve_known_command_passthrough_args():
    module, args = resolve(["ideas", "--bound", "memory"])
    assert module == "autoluce.ideation.ideas"
    assert args == ["--bound", "memory"]


def test_resolve_baseline_injects_default_flag_then_passthrough():
    module, args = resolve(["baseline", "--simulate"])
    assert module == "autoluce.bench.harness"
    assert args == ["--baseline", "--simulate"]  # default flag first, user args after


def test_resolve_run_with_passthrough():
    module, args = resolve(["run", "--dry-run"])
    assert module == "autoluce.loop.agent_loop"
    assert args == ["--dry-run"]


def test_resolve_setup_maps_to_prepare():
    module, _ = resolve(["setup"])
    assert module == "autoluce.prepare"


def test_resolve_web_maps_to_web_server():
    module, args = resolve(["web", "--listen", "0.0.0.0", "--port", "8000"])
    assert module == "autoluce.web.server"
    assert args == ["--listen", "0.0.0.0", "--port", "8000"]


@pytest.mark.parametrize("cmd", sorted(COMMANDS))
def test_every_advertised_command_resolves_to_package_module(cmd):
    module, _ = resolve([cmd])
    assert module.startswith("autoluce")
    assert not module.endswith(".py")  # dotted module path, not a script filename


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


def test_main_dispatches_module_via_dash_m_and_propagates_returncode(monkeypatch):
    calls = []

    def fake_runner(invocation):
        calls.append(invocation)
        return _Result(7)

    rc = main(["ideas", "--bound", "compute"], runner=fake_runner)
    assert rc == 7
    assert len(calls) == 1
    assert calls[0][1] == "-m"                      # [python, -m, module, *args]
    assert calls[0][2] == "autoluce.ideation.ideas"
    assert calls[0][3:] == ["--bound", "compute"]


def test_main_unknown_command_returns_nonzero(capsys):
    rc = main(["bogus"], runner=lambda inv: _Result(0))
    assert rc != 0


def test_main_help_returns_zero(capsys):
    rc = main([], runner=lambda inv: _Result(0))
    assert rc == 0


def test_every_command_module_is_importable():
    # Replaces the old SCRIPTS-dir check: the dispatcher must point at real modules.
    for module, _, _ in COMMANDS.values():
        assert importlib.util.find_spec(module) is not None, module
