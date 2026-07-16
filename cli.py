"""
Unified autoluce CLI: one entry point routing to the runnable modules.

Each subcommand maps to one of the runnable modules in the autoluce package
(autoluce.prepare / autoluce.bench.harness / autoluce.loop.agent_loop / ...).
The CLI reimplements nothing -- it resolves the subcommand to a dotted module path and
dispatches it as a subprocess (`python -m <module>`), so every module keeps its own
argparse and the read-only contract on harness / agent_loop / etc. is preserved.
resolve() is pure and tested; the subprocess call is an injected seam.

Registered as `autoluce` via [project.scripts] in pyproject.toml -> `uv run autoluce <cmd>`.
"""

from __future__ import annotations

import subprocess
import sys

# command -> (dotted module path, default_args, one-line help)
COMMANDS: dict[str, tuple[str, list[str], str]] = {
    "source":    ("autoluce.source_cli",       [], "inspect Lucebox ownership, vendor provenance, and drift"),
    "workspace": ("autoluce.workspace_cli",    [], "inventory workspaces and enforce safe lifecycle rules"),
    "agent":     ("autoluce.agent_cli",        [], "choose, build, review, and combine research tasks"),
    "join":      ("autoluce.fleet_cli",        ["join"], "join this machine to the team"),
    "submit":    ("autoluce.fleet_cli",        ["submit"], "submit a candidate to an available machine"),
    "status":    ("autoluce.fleet_cli",        ["status"], "show machines and experiments in plain language"),
    "pause":     ("autoluce.fleet_cli",        ["pause"], "stop assigning new work to this machine"),
    "resume":    ("autoluce.fleet_cli",        ["resume"], "allow this machine to receive work"),
    "leave":     ("autoluce.fleet_cli",        ["leave"], "remove this machine from the team"),
    "worker":    ("autoluce.fleet_cli",        ["worker"], "run an assigned experiment through the safe pipeline"),
    "coordinator": ("autoluce.coordinator_server", [], "run the restricted shared team coordinator"),
    "doctor":    ("autoluce.doctor",            [], "inspect and fingerprint a local or remote target"),
    "onboard":   ("autoluce.onboard",           [], "install a user-local launcher on an SSH target"),
    "test-drive": ("autoluce.test_drive",        [], "check readiness or run a short live V4 canary"),
    "consult":   ("autoluce.consult",           [], "create a machine-aware research contract"),
    "research":  ("autoluce.research",          [], "run a machine-aware campaign with an optional reference"),
    "freeze":    ("autoluce.freeze",            [], "freeze exact quality references on a target"),
    "profile-report": ("autoluce.profile_report", [], "summarize a rocprofv3 kernel capture"),
    "verify":    ("autoluce.verify_remote",      [], "run interleaved remote A/B verification"),
    "setup":     ("autoluce.prepare",           [], "clone + build + download models (one-time)"),
    "baseline":  ("autoluce.bench.harness",     ["--baseline"], "measure the baseline score"),
    "kl-base":   ("autoluce.bench.kl",          [], "generate the KL reference logits from the baseline build"),
    "run":       ("autoluce.loop.agent_loop",   [], "one keep/revert experiment (the agent loop)"),
    "shadow":    ("autoluce.shadow",            [], "shadow bench from your own local traffic (proxy|build)"),
    "ideas":     ("autoluce.ideation.ideas",    [], "list/rank untried ROADMAP ideas (--bound)"),
    "propose":   ("autoluce.ideation.propose",  [], "ask the LLM for the next idea (needs OPENAI_BASE_URL)"),
    "harness":   ("autoluce.bench.harness",     [], "raw benchmark harness"),
    "report":    ("autoluce.report",            [], "aggregate / diff results"),
    "reproduce": ("autoluce.reproduce",         [], "reproducibility suite"),
    "nvfp4":     ("autoluce.nvfp4",             [], "test or benchmark the CUDA NVFP4 operator"),
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
    lines = ["autoluce <command> [args...]", "", "Commands:"]
    for cmd, (_, _, desc) in COMMANDS.items():
        lines.append(f"  {cmd:<{width}}  {desc}")
    lines.append("")
    lines.append("Run `autoluce <command> --help` for a command's own options.")
    return "\n".join(lines)


def main(argv: list[str] | None = None, runner=subprocess.run) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        module, args = resolve(argv)
    except CliHelp:
        print(_help_text())
        return 0
    except CliError as e:
        print(f"autoluce: {e}", file=sys.stderr)
        print(_help_text(), file=sys.stderr)
        return 2
    invocation = [sys.executable, "-m", module, *args]
    return runner(invocation).returncode


if __name__ == "__main__":
    sys.exit(main())
