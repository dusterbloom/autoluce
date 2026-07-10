"""Restricted worker lifecycle for assigned coordination jobs."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from autoggml import ROOT
from autoggml.coordination import Claim, Job
from autoggml.source_layout import SourceLayout
from autoggml.test_drive import _lease


class WorkerGateway(Protocol):
    def claim(self, worker_id: str) -> Claim | None: ...

    def finish(self, job_id: str, status: str, result: dict[str, Any]) -> Job: ...


@dataclass
class TeamWorker:
    gateway: WorkerGateway
    worker_id: str
    executor: Callable[[Claim], dict[str, Any]]

    def run_once(self) -> dict[str, Any]:
        claim = self.gateway.claim(self.worker_id)
        if claim is None:
            return {"status": "idle"}
        try:
            result = self.executor(claim)
        except Exception as error:
            failure = {"error": str(error), "error_type": type(error).__name__}
            self.gateway.finish(claim.job.job_id, "failed", failure)
            return {"status": "failed", **failure}
        self.gateway.finish(claim.job.job_id, "completed", result)
        return {"status": "completed", "result": result}


class LocalExperimentExecutor:
    """Execute only the fixed autoggml experiment pipeline, never queued shell."""

    def __init__(self, *, simulate: bool = False, lock_path: str = "/tmp/autoggml-gpu.lock") -> None:
        self.simulate = simulate
        self.lock_path = lock_path

    def __call__(self, claim: Claim) -> dict[str, Any]:
        if self.simulate:
            from autoggml.bench.harness import run_harness

            summary = run_harness(simulate=True)
            return {"mode": "simulation", "correctness": summary.get("correctness"), "score": summary.get("score")}

        patch_name = f"team-{claim.candidate.candidate_id}.patch"
        patch_path = ROOT / "patches" / patch_name
        patch_path.write_bytes(claim.patch)
        results = []
        try:
            layout = SourceLayout.resolve()
            layout.require_capability("product-benchmark")
            with _lease(self.lock_path):
                for backend in claim.candidate.backends:
                    if backend not in layout.manifest.supported_backends:
                        raise ValueError(f"Lucebox product does not support backend '{backend}'")
                    env = os.environ.copy()
                    for variable in ("GGML_CUDA", "GGML_HIP", "GGML_VULKAN"):
                        env.pop(variable, None)
                    env.update({
                        {"cuda": "GGML_CUDA", "hip": "GGML_HIP"}[backend]: "ON",
                        "AUTOGGML_BENCHMARKS": claim.candidate.model,
                        "AUTOGGML_BUILD_JOBS": str(min(4, int(env.get("AUTOGGML_BUILD_JOBS", "4")))),
                        "AUTOGGML_BUILD_SUBDIR": f"build-{backend}",
                        "AUTOGGML_EXPERIMENT_PATCH": patch_name,
                    })
                    process = subprocess.run(
                        [sys.executable, "-m", "autoggml.loop.agent_loop", "--dry-run"],
                        cwd=ROOT, env=env, text=True, capture_output=True, check=False,
                    )
                    if process.returncode:
                        message = process.stderr.strip() or process.stdout.strip() or f"{backend} experiment failed"
                        raise RuntimeError(message)
                    results.append({"backend": backend, "output": process.stdout[-8000:]})
            return {"mode": "live", "backends": results, "patch_sha256": claim.candidate.patch_sha256}
        finally:
            patch_path.unlink(missing_ok=True)
