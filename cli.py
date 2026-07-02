"""
Unified autoggml CLI: one entry point routing to the existing scripts.

Each subcommand maps to one of the runnable scripts (prepare / harness / agent_loop / ...).
The CLI reimplements nothing -- it resolves the subcommand to a script and dispatches it as
a subprocess, so every script keeps its own argparse and the read-only contract on
harness.py / agent_loop.py / etc. is preserved. resolve() is pure and tested; the subprocess
call is an injected seam.

Registered as `autoggml` via [project.scripts] in pyproject.toml -> `uv run autoggml <cmd>`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT

# command -> (script, default_args, one-line help)
COMMANDS: dict[str, tuple[str, list[str], str]] = {
    "setup":     ("prepare.py",    [], "clone + build + download models (one-time)"),
    "baseline":  ("harness.py",    ["--baseline"], "measure the baseline score"),
    "kl-base":   ("kl.py",         [], "generate the KL reference logits from the baseline build"),
    "run":       ("agent_loop.py", [], "one keep/revert experiment (the agent loop)"),
    "shadow":    ("shadow.py",     [], "shadow bench from your own local traffic (proxy|build)"),
    "ideas":     ("ideas.py",      [], "list/rank untried ROADMAP ideas (--bound)"),
    "propose":   ("propose.py",    [], "ask the LLM for the next idea (needs OPENAI_BASE_URL)"),
    "harness":   ("harness.py",    [], "raw benchmark harness"),
    "report":    ("report.py",     [], "aggregate / diff results"),
    "reproduce": ("reproduce.py",  [], "reproducibility suite"),
}


class CliError(Exception):
    """Unknown command."""


class CliHelp(Exception):
    """User asked for help / gave no command."""


def resolve(argv: list[str]) -> tuple[str, list[str]]:
    """Map a CLI invocation to (script, full_args). Default args precede user args."""
    if not argv or argv[0] in ("help", "--help", "-h"):
        raise CliHelp()
    cmd, *rest = argv
    if cmd not in COMMANDS:
        raise CliError(f"unknown command '{cmd}'")
    script, default_args, _ = COMMANDS[cmd]
    return script, [*default_args, *rest]


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
        script, args = resolve(argv)
    except CliHelp:
        print(_help_text())
        return 0
    except CliError as e:
        print(f"autoggml: {e}", file=sys.stderr)
        print(_help_text(), file=sys.stderr)
        return 2
    invocation = [sys.executable, str(SCRIPTS / script), *args]
    return runner(invocation).returncode


if __name__ == "__main__":
    sys.exit(main())
