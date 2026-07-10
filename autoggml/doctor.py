"""Machine-aware target inventory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from autoggml.models import load_catalog, resolve_entry_files
from autoggml.profiles import MachineProfile
from autoggml.remote import SSHWorker
from autoggml.targets import TargetConfig


REMOTE_PROBE = r'''
import datetime, json, os, pathlib, platform, shutil, subprocess, sys

def run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return (p.stdout or p.stderr).strip()
    except Exception:
        return ""

def first_line(cmd):
    value = run(cmd)
    return value.splitlines()[0] if value else None

def read(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except Exception:
        return None

mem = {}
for line in (read("/proc/meminfo") or "").splitlines():
    key, _, value = line.partition(":")
    if value:
        mem[key] = int(value.strip().split()[0]) * 1024

rocminfo = run(["rocminfo"]) if shutil.which("rocminfo") else ""
gfx = None
marketing = None
for line in rocminfo.splitlines():
    value = line.split(":", 1)[-1].strip()
    if value.startswith("gfx11"):
        gfx = value
    if "Marketing Name:" in line and "Graphics" in value:
        marketing = value

active = []
needles = ("llama-bench", "llama-server", "llama-cli", "dflash_server", "test_dflash", "rocprof", "hipcc", "cmake --build", "ninja")
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        cmdline = entry.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except Exception:
        continue
    rss = 0
    try:
        rss = int(entry.joinpath("statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    if cmdline and (rss >= 1024**3 or any(needle in cmdline for needle in needles)):
        active.append({"pid": int(entry.name), "rss_bytes": rss, "command": cmdline})

model_paths = sys.argv[1:]
models = []
for value in model_paths:
    path = pathlib.Path(value)
    try:
        st = path.stat()
        models.append({"path": value, "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "readable": os.access(path, os.R_OK)})
    except Exception:
        models.append({"path": value, "missing": True})

observed = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": platform.node(),
    "os": platform.system(),
    "kernel": platform.release(),
    "machine": platform.machine(),
    "cpu": first_line(["bash", "-lc", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"]),
    "ram_total_bytes": mem.get("MemTotal"),
    "mem_available_bytes": mem.get("MemAvailable"),
    "swap_total_bytes": mem.get("SwapTotal"),
    "swap_free_bytes": mem.get("SwapFree"),
    "gpu_arch": gfx,
    "gpu_name": marketing,
    "uma": bool(gfx and gfx.startswith("gfx115")),
    "ttm_pages_limit": int(read("/sys/module/ttm/parameters/pages_limit") or 0),
    "hipcc": shutil.which("hipcc"),
    "hipcc_version": first_line(["hipcc", "--version"]) if shutil.which("hipcc") else None,
    "rocprofv3": shutil.which("rocprofv3"),
    "vulkaninfo": shutil.which("vulkaninfo"),
    "cmake_version": first_line(["cmake", "--version"]) if shutil.which("cmake") else None,
    "ninja_version": first_line(["ninja", "--version"]) if shutil.which("ninja") else None,
    "ccache_version": first_line(["ccache", "--version"]) if shutil.which("ccache") else None,
    "python_version": platform.python_version(),
    "load_average": list(os.getloadavg()),
    "active_processes": active,
    "busy_reasons": (["available_memory_below_12_gib"] if mem.get("MemAvailable", 0) < 12 * 1024**3 else []) +
                    (["accelerator_or_build_process_active"] if active else []),
    "models": models,
}
print(json.dumps(observed))
'''.strip()


def _local_observed(model_paths: list[Path]) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-", *(str(path) for path in model_paths)],
        input=REMOTE_PROBE,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def build_profile(target: TargetConfig, model_id: str = "deepseek-v4-flash") -> MachineProfile:
    entry = load_catalog()[model_id]
    model_root = Path(target.model_root) if target.model_root else None
    paths = resolve_entry_files(entry, model_root)
    if target.transport == "ssh":
        result = SSHWorker(target).run_python(REMOTE_PROBE, [str(path) for path in paths])
        observed = json.loads(result.stdout)
    else:
        observed = _local_observed(paths)

    inferred = []
    unknown = []
    if observed.get("gpu_arch") == "gfx1151" and observed.get("uma"):
        inferred.append({"fact": "unified_memory_apu", "confidence": 1.0, "evidence": ["gpu_arch", "uma"]})
    if observed.get("hipcc"):
        inferred.append({"fact": "hip_available", "confidence": 1.0, "evidence": ["hipcc"]})
    if not observed.get("vulkaninfo"):
        unknown.append("vulkan_backend_runtime")
    if not observed.get("ccache_version"):
        unknown.append("ccache")
    if observed.get("busy_reasons"):
        inferred.append({
            "fact": "target_busy",
            "confidence": 1.0,
            "evidence": list(observed["busy_reasons"]),
        })
    return MachineProfile(observed, inferred, unknown)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a local or remote research target")
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = build_profile(TargetConfig.load(args.target), args.model)
    payload = profile.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2) if args.json or args.output is None else profile.fingerprint)


if __name__ == "__main__":
    main()
