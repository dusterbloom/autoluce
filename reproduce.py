"""
Read-only reproducibility suite.

Records the environment, runs the harness, and writes a provenance bundle.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from harness import run_harness

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def capture_environment() -> dict[str, any]:
    """Capture machine and dependency versions."""
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }

    # Git commit of autoggml
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT)
        env["autoggml_commit"] = r.stdout.strip()
    except Exception:
        env["autoggml_commit"] = None

    # Tool versions
    for tool in ["cmake", "gcc", "g++", "nvcc"]:
        path = shutil.which(tool)
        if path:
            try:
                r = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
                env[f"{tool}_version"] = r.stdout.splitlines()[0] if r.stdout else "unknown"
            except Exception as e:
                env[f"{tool}_version"] = str(e)
        else:
            env[f"{tool}_version"] = None

    # Python packages
    try:
        import pkg_resources
        env["python_packages"] = {d.key: d.version for d in pkg_resources.working_set}
    except Exception:
        env["python_packages"] = {}

    return env


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RESULTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    env = capture_environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))

    summary = run_harness(baseline=True)
    (out_dir / "baseline.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"Reproducibility bundle written to {out_dir}")


if __name__ == "__main__":
    main()
