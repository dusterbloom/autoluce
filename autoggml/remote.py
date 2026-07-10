"""SSH transport with fail-fast accelerator leasing."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from autoggml import ROOT
from autoggml.targets import TargetConfig


class RemoteBusyError(RuntimeError):
    pass


_ACTIVITY_CHECK = r'''
import os, pathlib, sys
mem_available = 0
try:
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available = int(line.split()[1]) * 1024
            break
except Exception:
    pass
if mem_available < 12 * 1024**3:
    print(f"remote target has only {mem_available / 1024**3:.2f} GiB available", file=sys.stderr)
    raise SystemExit(73)
me = os.getpid()
ancestors = {me}
pid = me
while pid > 1:
    try:
        fields = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
        pid = int(fields[3])
        ancestors.add(pid)
    except Exception:
        break
needles = ("llama-bench", "llama-server", "llama-cli", "dflash_server", "test_dflash", "rocprof", "hipcc", "cmake --build", "ninja")
busy = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors:
        continue
    try:
        cmd = entry.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except Exception:
        continue
    rss = 0
    try:
        rss = int(entry.joinpath("statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    if rss >= 1024**3 or any(needle in cmd for needle in needles):
        busy.append(f"{entry.name}:{cmd.strip()}")
if busy:
    print("remote target is busy: " + "; ".join(busy), file=sys.stderr)
    raise SystemExit(73)
'''.strip()


@dataclass
class SSHResult:
    stdout: str
    stderr: str
    returncode: int


class SSHWorker:
    def __init__(self, target: TargetConfig, runner=subprocess.run) -> None:
        if target.transport != "ssh" or not target.host or not target.root:
            raise ValueError("SSHWorker requires an ssh target with host and root")
        self.target = target
        self.runner = runner

    def _ssh(self, remote_args: list[str], **kwargs) -> subprocess.CompletedProcess:
        remote_command = shlex.join(remote_args)
        return self.runner(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.target.host, remote_command],
            capture_output=True, text=True, **kwargs,
        )

    def run_python(self, source: str, args: list[str] | None = None, check: bool = True) -> SSHResult:
        proc = self._ssh(["python3", "-", *(args or [])], input=source, check=False)
        if check and proc.returncode:
            raise RuntimeError(proc.stderr.strip() or f"remote python failed with {proc.returncode}")
        return SSHResult(proc.stdout, proc.stderr, proc.returncode)

    def run(self, command: list[str], *, lease: bool = False, timeout: int | None = None) -> SSHResult:
        quoted = shlex.join(command)
        if lease:
            check = shlex.quote(_ACTIVITY_CHECK)
            inner = f"python3 -c {check} && exec {quoted}"
            remote = ["flock", "-n", self.target.lock_path, "bash", "-lc", inner]
        else:
            remote = ["bash", "-lc", quoted]
        proc = self._ssh(remote, check=False, timeout=timeout)
        if lease and proc.returncode in {1, 73}:
            raise RemoteBusyError(proc.stderr.strip() or "remote accelerator lease is held")
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip() or f"remote command failed with {proc.returncode}")
        return SSHResult(proc.stdout, proc.stderr, proc.returncode)

    def sync_repo(self) -> None:
        if shutil.which("rsync") is None:
            raise RuntimeError("rsync is required for remote workers")
        destination = f"{self.target.host}:{self.target.root.rstrip('/')}/"
        proc = self.runner(
            [
                "rsync", "-az", "--delete",
                "--exclude=.git/", "--exclude=.venv/", "--exclude=work/",
                "--exclude=results/", "--exclude=.pytest_cache/", "--exclude=__pycache__/",
                str(ROOT) + "/", destination,
            ],
            capture_output=True, text=True,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip() or "rsync failed")

    def ensure_remote_uv(self) -> None:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("local uv binary not found")
        remote_bin = f"{self.target.root.rstrip('/')}/.tools/uv"
        probe = self._ssh(["test", "-x", remote_bin], check=False)
        if probe.returncode == 0:
            return
        self.run(["mkdir", "-p", f"{self.target.root.rstrip('/')}/.tools"], lease=True)
        proc = self.runner(
            ["rsync", "-az", uv, f"{self.target.host}:{remote_bin}"],
            capture_output=True, text=True,
        )
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip() or "failed to provision uv")

    def run_harness(self, args: list[str], env: dict[str, str] | None = None, timeout: int = 7200) -> dict[str, Any]:
        self.sync_repo()
        root = self.target.root.rstrip("/")
        command = [f"{root}/.tools/uv", "run", "python", "-m", "autoggml.bench.harness", *args, "--json"]
        env_args = ["env", "AUTOGGML_REMOTE_WORKER=1", f"AUTOGGML_BUILD_JOBS={self.target.build_jobs}"]
        for key, value in sorted((env or {}).items()):
            env_args.append(f"{key}={value}")
        result = self.run([*env_args, *command], lease=True, timeout=timeout)
        payload = json.loads(result.stdout)
        self.fetch_results()
        return payload

    def fetch_results(self) -> None:
        destination = ROOT / "results" / "remote" / self.target.name
        destination.mkdir(parents=True, exist_ok=True)
        source = f"{self.target.host}:{self.target.root.rstrip('/')}/results/"
        proc = self.runner(
            ["rsync", "-az", source, str(destination) + "/"],
            capture_output=True, text=True,
        )
        # A first run may legitimately have no results directory yet.
        if proc.returncode not in {0, 23}:
            raise RuntimeError(proc.stderr.strip() or "failed to retrieve remote results")
