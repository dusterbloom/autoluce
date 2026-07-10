"""Freeze deterministic and KL quality references on a contracted target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoggml.contracts import ResearchContract
from autoggml.remote import SSHWorker
from autoggml.targets import TargetConfig


def contract_namespace(contract: ResearchContract, backend: str) -> str:
    return f"{contract.machine_fingerprint[:16]}-{contract.model_fingerprint[:16]}-{backend}"


def freeze(target: TargetConfig, contract: ResearchContract, backend: str = "hip") -> dict:
    if backend not in contract.backends:
        raise ValueError(f"backend '{backend}' is not allowed by the research contract")
    worker = SSHWorker(target)
    worker.sync_repo()
    worker.ensure_remote_uv()
    root = target.root.rstrip("/")
    namespace = contract_namespace(contract, backend)
    state = f"{root}/work/state/{namespace}"
    backend_var = {"cuda": "GGML_CUDA", "hip": "GGML_HIP", "vulkan": "GGML_VULKAN"}[backend]
    env = [
        "env", "AUTOGGML_REMOTE_WORKER=1", f"{backend_var}=ON",
        f"AUTOGGML_MODEL_ROOT={target.model_root or root + '/work/models'}",
        f"AUTOGGML_BUILD_SUBDIR=build-{backend}",
        f"AUTOGGML_STATE_DIR={state}", f"AUTOGGML_GOLDEN_DIR={state}/golden",
        "AUTOGGML_BENCHMARKS=deepseek-v4-flash",
    ]
    uv = f"{root}/.tools/uv"
    worker.run(
        [*env, uv, "run", "python", "scripts/generate_golden.py", "--benchmark", contract.model, "--overwrite"],
        lease=True, timeout=7200,
    )
    worker.run(
        [*env, uv, "run", "python", "-m", "autoggml.bench.kl", contract.model],
        lease=True, timeout=7200,
    )
    output = Path("results") / "remote" / target.name / "state" / namespace
    output.mkdir(parents=True, exist_ok=True)
    proc = worker.runner(
        ["rsync", "-az", f"{target.host}:{state}/", str(output) + "/"],
        capture_output=True, text=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "failed to retrieve frozen references")
    manifest = {
        "target": target.name,
        "contract": contract.to_dict(),
        "backend": backend,
        "namespace": namespace,
        "files": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
    }
    (output / "freeze-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze quality references on a remote target")
    parser.add_argument("--target", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--backend", choices=["cuda", "hip", "vulkan"], default="hip")
    args = parser.parse_args()
    result = freeze(TargetConfig.load(args.target), ResearchContract.read(args.contract), args.backend)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
