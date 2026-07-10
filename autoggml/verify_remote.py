"""Interleaved remote A/B verification under one accelerator lease."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from autoggml import ROOT
from autoggml.contracts import ResearchContract
from autoggml.freeze import contract_namespace
from autoggml.loop.verify import ab_compare
from autoggml.parallel.concurrency import LockedFrontier
from autoggml.remote import SSHWorker
from autoggml.targets import TargetConfig


REMOTE_AB = r'''
import json, os, pathlib, shutil, subprocess, sys
root, uv, patch, rounds, backend, state, model, jobs = sys.argv[1:9]
rounds = int(rounds)
jobs = min(4, max(1, int(jobs)))
backend_var = {"cuda": "GGML_CUDA", "hip": "GGML_HIP", "vulkan": "GGML_VULKAN"}[backend]
common = os.environ.copy()
common.update({
    "AUTOGGML_REMOTE_WORKER": "1",
    backend_var: "ON",
    "AUTOGGML_STATE_DIR": state,
    "AUTOGGML_GOLDEN_DIR": state + "/golden",
    "AUTOGGML_BENCHMARKS": model,
    "AUTOGGML_RESULT_BUNDLE": root + "/results/runs/" + os.path.basename(state),
})

# Validate the patch's reference/backend op before loading the full model.
source = root + "/work/lucebox-ggml"
test_build = source + "/build-" + backend + "-optest"
prep_env = common.copy()
prep_env["AUTOGGML_EXPERIMENT_PATCH"] = patch
prep = subprocess.run(
    [uv, "run", "python", "-c", "from experiment import reset_lucebox,apply_experiment; reset_lucebox(); apply_experiment()"],
    cwd=root, env=prep_env, capture_output=True, text=True,
)
if prep.returncode:
    print(prep.stdout + prep.stderr, file=sys.stderr)
    raise SystemExit(prep.returncode)
cmake = ["cmake", "-G", "Ninja", "-S", source, "-B", test_build, "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=ON"]
cmake.append({"cuda": "-DGGML_CUDA=ON", "hip": "-DGGML_HIP=ON", "vulkan": "-DGGML_VULKAN=ON"}[backend])
if shutil.which("ccache"):
    cmake += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
for cmd in (cmake, ["cmake", "--build", test_build, "-j", str(jobs), "--target", "test-backend-ops"]):
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    if proc.returncode:
        print(proc.stdout + proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
op_test = subprocess.run(
    [test_build + "/bin/test-backend-ops", "test", "-o", "SINKHORN_NORM"],
    cwd=root, capture_output=True, text=True,
)
if op_test.returncode:
    print(op_test.stdout + op_test.stderr, file=sys.stderr)
    raise SystemExit(op_test.returncode)

results = []
for index in range(rounds):
    for label in ("baseline", "candidate"):
        env = common.copy()
        env["AUTOGGML_BUILD_SUBDIR"] = "build-" + backend + "-" + label
        if label == "candidate":
            env["AUTOGGML_EXPERIMENT_PATCH"] = patch
        else:
            env.pop("AUTOGGML_EXPERIMENT_PATCH", None)
        cmd = [uv, "run", "python", "-m", "autoggml.bench.harness", "--json", "--significance", "1.0"]
        if label == "baseline":
            cmd.append("--baseline")
        proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        if proc.returncode:
            print(proc.stderr, file=sys.stderr)
            raise SystemExit(proc.returncode)
        results.append({"round": index + 1, "label": label, "summary": json.loads(proc.stdout)})
print(json.dumps({"op_test": {"status": "pass", "output": op_test.stdout[-4000:]}, "rounds": results}))
'''.strip()


def _combine(results: list[dict], label: str) -> dict:
    summaries = [item["summary"] for item in results if item["label"] == label]
    count = len(summaries)
    return {
        "score": sum(item["score"] for item in summaries) / count,
        "score_stddev": math.sqrt(sum(item.get("score_stddev", 0.0) ** 2 for item in summaries)) / count,
        "decode_tok_s": sum(item["decode_tok_s"] for item in summaries) / count,
        "prefill_tok_s": sum(item["prefill_tok_s"] for item in summaries) / count,
        "peak_mem_GiB": max(item["peak_mem_GiB"] for item in summaries),
        "correctness": "pass" if all(item.get("correctness") == "pass" for item in summaries) else "FAIL",
    }


def verify(target: TargetConfig, contract: ResearchContract, patch: str, backend: str, rounds: int, k: float) -> dict:
    if backend not in contract.backends:
        raise ValueError(f"backend '{backend}' is not allowed by the research contract")
    if rounds < 2:
        raise ValueError("remote A/B verification requires at least two rounds")
    worker = SSHWorker(target)
    worker.sync_repo()
    worker.ensure_remote_uv()
    root = target.root.rstrip("/")
    namespace = contract_namespace(contract, backend)
    state = f"{root}/work/state/{namespace}"
    result = worker.run(
        [
            "python3", "-c", REMOTE_AB, root, f"{root}/.tools/uv", patch, str(rounds), backend,
            state, contract.model, str(target.build_jobs),
        ],
        lease=True, timeout=21600,
    )
    remote_result = json.loads(result.stdout)
    rounds_data = remote_result["rounds"]
    baseline = _combine(rounds_data, "baseline")
    candidate = _combine(rounds_data, "candidate")
    verdict = ab_compare(baseline, candidate, k)
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target": target.name,
        "namespace": namespace,
        "backend": backend,
        "patch": patch,
        "rounds": rounds_data,
        "op_test": remote_result["op_test"],
        "baseline": baseline,
        "candidate": candidate,
        "verdict": verdict,
        "contract": contract.to_dict(),
    }
    output = ROOT / "results" / "remote" / target.name / "verification" / namespace
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    worker.fetch_results()
    if verdict.get("verified"):
        frontier = LockedFrontier(ROOT / "results" / "frontiers" / namespace, k=k)
        frontier.claim_best_if_significant(patch, candidate, f"verified remote patch: {patch}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interleaved remote A/B verification")
    parser.add_argument("--target", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--experiment-patch", required=True)
    parser.add_argument("--backend", choices=["cuda", "hip", "vulkan"], default="hip")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--significance", type=float, default=1.0)
    args = parser.parse_args()
    payload = verify(
        TargetConfig.load(args.target), ResearchContract.read(args.contract), args.experiment_patch,
        args.backend, args.rounds, args.significance,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
