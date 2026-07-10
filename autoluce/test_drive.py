"""Safe local smoke and optional live DeepSeek V4 canary."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from autoluce.doctor import build_profile
from autoluce.source_layout import SourceLayout
from autoluce.targets import TargetConfig


@contextmanager
def _lease(path: str):
    lock = Path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another autoluce worker holds the accelerator lease") from exc
        yield


def _patch_ready(root: Path) -> bool:
    patch = root / "patches" / "deepseek-v4-sinkhorn-77bccaa.patch"
    provenance = patch.with_suffix(".json")
    if not patch.is_file() or not provenance.is_file():
        return False
    metadata = json.loads(provenance.read_text())
    manifest = SourceLayout.resolve(root=root).manifest
    return (
        metadata.get("compatibility") == "ready"
        and metadata.get("target_layout") == manifest.layout
        and metadata.get("target_luce_commit") == manifest.ref
    )


def safe_test_drive(target: TargetConfig) -> dict:
    from autoluce.bench.harness import run_harness

    profile = build_profile(target)
    model = profile.observed.get("models", [{}])[0]
    simulation = run_harness(baseline=True, simulate=True)
    busy = bool(profile.observed.get("busy_reasons"))
    model_ready = bool(model.get("readable")) and not model.get("missing")
    hip_ready = bool(profile.observed.get("hipcc"))
    runtime_ready = "product-benchmark" in SourceLayout.resolve().manifest.capabilities
    patch_ready = _patch_ready(Path(__file__).resolve().parent.parent)
    if busy:
        status, next_step = "busy", "retry when the host is idle"
    elif not model_ready:
        status, next_step = "needs-model", "check the configured model_root"
    elif not hip_ready:
        status, next_step = "needs-hip", "install or repair the HIP toolchain"
    elif not runtime_ready:
        status, next_step = "needs-runtime-adapter", "product source/build are ready; product benchmark adapter is pending"
    else:
        status, next_step = "ready", "autoluce test-drive --live"
    return {
        "status": status,
        "host": profile.observed.get("hostname"),
        "gpu": profile.observed.get("gpu_arch"),
        "memory_available_gib": round(float(profile.observed.get("mem_available_bytes") or 0) / 1024**3, 2),
        "model": {"path": model.get("path"), "size_gib": round(float(model.get("size_bytes") or 0) / 1024**3, 2),
                  "readable": model.get("readable", False)},
        "patch_ready": patch_ready,
        "patch_status": "ready" if patch_ready else "requires-port",
        "simulated_loop": "pass" if simulation.get("correctness") == "pass" else "FAIL",
        "busy_reasons": profile.observed.get("busy_reasons", []),
        "next": next_step,
    }


def live_test_drive(target: TargetConfig) -> dict:
    SourceLayout.resolve().require_capability("product-benchmark")
    with _lease(target.lock_path):
        before = build_profile(target)
        if before.observed.get("busy_reasons"):
            raise RuntimeError("target is busy: " + ", ".join(before.observed["busy_reasons"]))
        old_env = os.environ.copy()
        try:
            os.environ.update({
                "AUTOLUCE_BENCHMARKS": "deepseek-v4-test-drive",
                "AUTOLUCE_MODEL_ROOT": target.model_root or str(Path.home() / "models"),
                "AUTOLUCE_BUILD_JOBS": str(target.build_jobs),
                "AUTOLUCE_BUILD_SUBDIR": "build-hip",
                "AUTOLUCE_STATE_DIR": str(Path(target.root or Path.cwd()) / "work" / "state" / "test-drive"),
                "GGML_HIP": "ON",
            })
            from autoluce.bench.harness import run_harness
            from autoluce.prepare import build_lucebox, clone_lucebox, download_models

            clone_lucebox()
            download_models()
            build_lucebox()
            summary = run_harness(baseline=True)
        finally:
            os.environ.clear()
            os.environ.update(old_env)
    return {
        "status": "pass" if summary.get("correctness") == "pass" else "FAIL",
        "decode_tok_s": summary.get("decode_tok_s"),
        "prefill_tok_s": summary.get("prefill_tok_s"),
        "peak_rss_gib": summary.get("peak_mem_GiB"),
        "note": "canary only; not eligible for the research frontier",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check autoluce readiness or run a short live V4 canary")
    parser.add_argument("--target", default=os.environ.get("AUTOLUCE_DEFAULT_TARGET", "lucebox3"))
    parser.add_argument("--live", action="store_true", help="Build if needed and run one 4K DeepSeek V4 canary")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    target = TargetConfig.load(args.target)
    result = live_test_drive(target) if args.live else safe_test_drive(target)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("autoluce test drive")
    print(f"  host:       {result.get('host', 'lucebox3')}")
    if "gpu" in result:
        print(f"  accelerator:{' ' if result.get('gpu') else ''}{result.get('gpu') or 'not detected'}")
    model = result.get("model")
    if model:
        state = "readable" if model.get("readable") else "missing"
        print(f"  model:      {model.get('size_gib', 0):.2f} GiB, {state}")
    if "patch_ready" in result:
        print(f"  experiment: {result.get('patch_status', 'unknown')}")
    if "simulated_loop" in result:
        print(f"  harness:    {result['simulated_loop'].lower()}")
    if "decode_tok_s" in result:
        print(f"  decode:     {result['decode_tok_s']:.2f} tok/s")
    print(f"\n{result['status'].upper()}")
    print(f"Next: {result.get('next') or result.get('note')}")


if __name__ == "__main__":
    main()
