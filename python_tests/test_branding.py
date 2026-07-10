"""Keep the public and internal project name consistent."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LEGACY_NAME = "auto" + "ggml"


def test_tracked_text_files_do_not_use_the_previous_project_name():
    tracked = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    offenders = []
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if LEGACY_NAME in text.lower() or LEGACY_NAME in relative.lower():
            offenders.append(relative)
    assert offenders == []


def test_packaging_exposes_only_the_autoluce_command_and_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert project["project"]["name"] == "autoluce"
    assert project["project"]["scripts"] == {"autoluce": "cli:main"}
    assert all(package == "autoluce" or package.startswith("autoluce.") for package in project["tool"]["setuptools"]["packages"])
