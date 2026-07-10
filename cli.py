"""
Unified autoggml CLI: one entry point routing to the runnable modules.

Each subcommand maps to one of the runnable modules in the autoggml package
(autoggml.prepare / autoggml.bench.harness / autoggml.loop.agent_loop / ...).
The CLI reimplements nothing -- it resolves the subcommand to a dotted module path and
dispatches it as a subprocess (`python -m <module>`), so every module keeps its own
argparse and the read-only contract on harness / agent_loop / etc. is preserved.
resolve() is pure and tested; the subprocess call is an injected seam.

Registered as `autoggml` via [project.scripts] in pyproject.toml -> `uv run autoggml <cmd>`.
"""

from __future__ import annotations

import subprocess
import sys

# command -> (dotted module path, default_args, one-line help)
COMMANDS: dict[str, tuple[str, list[str], str]] = {
    "source":    ("autoggml.source_cli",       [], "inspect Lucebox ownership, vendor provenance, and drift"),
    "agent":     ("autoggml.agent_cli",        [], "choose, build, review, and combine research tasks"),
    "join":      ("autoggml.fleet_cli",        ["join"], "join this machine to the team"),
    "submit":    ("autoggml.fleet_cli",        ["submit"], "submit a candidate to an available machine"),
    "status":    ("autoggml.fleet_cli",        ["status"], "show machines and experiments in plain language"),
    "pause":     ("autoggml.fleet_cli",        ["pause"], "stop assigning new work to this machine"),
    "resume":    ("autoggml.fleet_cli",        ["resume"], "allow this machine to receive work"),
    "leave":     ("autoggml.fleet_cli",        ["leave"], "remove this machine from the team"),
    "worker":    ("autoggml.fleet_cli",        ["worker"], "run an assigned experiment through the safe pipeline"),
    "coordinator": ("autoggml.coordinator_server", [], "run the restricted shared team coordinator"),
    "doctor":    ("autoggml.doctor",            [], "inspect and fingerprint a local or remote target"),
    "onboard":   ("autoggml.onboard",           [], "install a user-local launcher on an SSH target"),
    "test-drive": ("autoggml.test_drive",        [], "check readiness or run a short live V4 canary"),
    "consult":   ("autoggml.consult",           [], "create a machine-aware research contract"),
    "freeze":    ("autoggml.freeze",            [], "freeze exact and KL quality references on a target"),
    "profile-report": ("autoggml.profile_report", [], "summarize a rocprofv3 kernel capture"),
    "verify":    ("autoggml.verify_remote",      [], "run interleaved remote A/B verification"),
    "setup":     ("autoggml.prepare",           [], "clone + build + download models (one-time)"),
    "baseline":  ("autoggml.bench.harness",     ["--baseline"], "measure the baseline score"),
    "kl-base":   ("autoggml.bench.kl",          [], "generate the KL reference logits from the baseline build"),
    "run":       ("autoggml.loop.agent_loop",   [], "one keep/revert experiment (the agent loop)"),
    "shadow":    ("autoggml.shadow",            [], "shadow bench from your own local traffic (proxy|build)"),
    "ideas":     ("autoggml.ideation.ideas",    [], "list/rank untried ROADMAP ideas (--bound)"),
    "propose":   ("autoggml.ideation.propose",  [], "ask the LLM for the next idea (needs OPENAI_BASE_URL)"),
    "harness":   ("autoggml.bench.harness",     [], "raw benchmark harness"),
    "report":    ("autoggml.report",            [], "aggregate / diff results"),
    "reproduce": ("autoggml.reproduce",         [], "reproducibility suite"),
}


class CliError(Exception):
    """Unknown command."""


class CliHelp(Exception):
    """User asked for help / gave no command."""


def resolve(argv: list[str]) -> tuple[str, list[str]]:
    """Map a CLI invocation to (dotted module path, full_args). Default args precede user args."""
    if not argv or argv[0] in ("help", "--help", "-h"):
        raise CliHelp()
    cmd, *rest = argv
    if cmd not in COMMANDS:
        raise CliError(f"unknown command '{cmd}'")
    module, default_args, _ = COMMANDS[cmd]
    return module, [*default_args, *rest]


def _help_text() -> str:
    width = max(len(c) for c in COMMANDS)
    lines = ["autoggml <command> [args...]", "", "Commands:"]
    for cmd, (_, _, desc) in COMMANDS.items():
        lines.append(f"  {cmd:<{width}}  {desc}")
    lines.append("")
    lines.append("Run `autoggml <command> --help` for a command's own options.")
    return "\n".join(lines)


def main(argv: list[str] | None = None, runner=subprocess.run) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        module, args = resolve(argv)
    except CliHelp:
        print(_help_text())
        return 0
    except CliError as e:
        print(f"autoggml: {e}", file=sys.stderr)
        print(_help_text(), file=sys.stderr)
        return 2
    invocation = [sys.executable, "-m", module, *args]
    return runner(invocation).returncode


if __name__ == "__main__":
    sys.exit(main())
