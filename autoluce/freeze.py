"""Freeze deterministic exact-quality references on a contracted target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoluce.contracts import ResearchContract
from autoluce.remote import SSHWorker
from autoluce.source_layout import SourceLayout
from autoluce.targets import TargetConfig


def contract_namespace(contract: ResearchContract, backend: str) -> str:
    return f"{contract.machine_fingerprint[:16]}-{contract.model_fingerprint[:16]}-{backend}"


def freeze(target: TargetConfig, contract: ResearchContract, backend: str = "hip") -> dict:
    SourceLayout.resolve().require_capability("product-quality-exact")
    if backend not in contract.backends:
        raise ValueError(f"backend '{backend}' is not allowed by the research contract")
    worker = SSHWorker(target)
    worker.sync_repo()
    worker.ensure_remote_uv()
    root = target.root.rstrip("/")
    namespace = contract_namespace(contract, backend)
    state = f"{root}/work/state/{namespace}"
    backend_var = {"cuda": "GGML_CUDA", "hip": "GGML_HIP"}[backend]
    env = [
        "env", "AUTOLUCE_REMOTE_WORKER=1", f"{backend_var}=ON",
        f"AUTOLUCE_MODEL_ROOT={target.model_root or root + '/work/models'}",
        f"AUTOLUCE_BUILD_SUBDIR=build-{backend}",
        f"AUTOLUCE_STATE_DIR={state}", f"AUTOLUCE_GOLDEN_DIR={state}/golden",
        "AUTOLUCE_BENCHMARKS=deepseek-v4-flash",
    ]
    uv = f"{root}/.tools/uv"
    worker.run(
        [*env, uv, "run", "python", "scripts/generate_golden.py", "--benchmark", contract.model, "--overwrite"],
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
        "quality_oracles": ["exact"],
        "unavailable_oracles": ["kl: dflash_server has no token-logits API"],
        "files": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
    }
    (output / "freeze-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze quality references on a remote target")
    parser.add_argument("--target", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--backend", choices=["cuda", "hip"], default="hip")
    args = parser.parse_args()
    result = freeze(TargetConfig.load(args.target), ResearchContract.read(args.contract), args.backend)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
