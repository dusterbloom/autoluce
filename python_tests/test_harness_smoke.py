"""
Smoke tests for the autoluce v2 harness.

These tests run the harness in simulation mode (no real Lucebox product source
required) and verify that it produces a valid, reproducible result bundle.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoluce.bench.harness import run_harness
from autoluce.reproduce import capture_environment


def test_baseline_returns_required_keys():
    summary = run_harness(baseline=True, simulate=True)
    required = {"score", "decode_tok_s", "prefill_tok_s", "acceptance_rate", "peak_mem_GiB", "correctness", "experiment"}
    assert required.issubset(summary.keys())
    assert summary["correctness"] == "pass"
    assert summary["score"] > 0.0


def test_environment_capture_is_json_serializable():
    env = capture_environment()
    # Should not raise.
    text = json.dumps(env, indent=2, default=str)
    assert "timestamp_utc" in text
    assert "python_version" in text


def test_reproducibility_bundle(tmp_path: Path):
    summary = run_harness(baseline=True, simulate=True)
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "baseline.json").write_text(json.dumps(summary, indent=2, default=str))
    env = capture_environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))
    assert (out_dir / "baseline.json").exists()
    assert (out_dir / "environment.json").exists()


def test_simulated_summary_carries_score_stddev(monkeypatch):
    # Pin to one benchmark so the aggregate is independent of how many exist.
    monkeypatch.setenv("AUTOLUCE_BENCHMARKS", "smoke")
    summary = run_harness(baseline=True, simulate=True)
    assert "score_stddev" in summary
    # Sim fixture: decode=120±4. Score IS decode_tok_s; constraints are separate.
    assert summary["score"] == pytest.approx(120.0)
    assert summary["score_stddev"] == pytest.approx(4.0)


def test_summary_records_inherited_product_environment(monkeypatch):
    monkeypatch.setenv("AUTOLUCE_BENCHMARKS", "smoke")
    monkeypatch.setenv("DFLASH_PREFILL_UBATCH", "1024")
    monkeypatch.setenv("GGML_CUDA_GRAPH_OPT", "1")

    summary = run_harness(baseline=True, simulate=True)

    assert summary["experiment"]["runtime_env"] == {
        "DFLASH_PREFILL_UBATCH": "1024",
        "DFLASH27B_PREFILL_UBATCH": "1024",
        "GGML_CUDA_GRAPH_OPT": "1",
    }


def test_constraint_violation_zeroes_score():
    # smoke.json caps peak_mem_GiB at 8.0; the simulated fixture uses 18.0, so an
    # enforced run must zero the score exactly like a correctness failure.
    from autoluce.bench.harness import run_single_benchmark

    result = run_single_benchmark(
        "smoke", 1.0, {}, simulate=True,
        baseline_metrics={"smoke": {"prefill_tok_s": 2500.0}},
        k=1.0, enforce_constraints=True,
    )
    assert result["constraint_violations"]
    assert result["score"] == 0.0


def test_vendored_product_declares_http_benchmark_and_exact_quality_capabilities():
    from autoluce.source_layout import SourceLayout

    capabilities = SourceLayout.resolve().manifest.capabilities
    assert "product-benchmark" in capabilities
    assert "product-quality-exact" in capabilities


def test_real_run_source_evidence_records_backend_contract_and_content(monkeypatch):
    from autoluce.bench import harness

    evidence = SimpleNamespace(
        product_revision="a" * 40,
        product_digest="b" * 64,
        vendor_digest="c" * 64,
        binary_sha256=None,
        dirty_paths=["server/src/candidate.cpp"],
    )
    layout = SimpleNamespace(
        manifest=SimpleNamespace(
            product_backends=["cuda", "hip"],
            vendor_backends=["cpu", "cuda", "hip", "vulkan"],
        ),
        binary=lambda target, backend: Path(f"/build/{backend}/{target}"),
        evidence=lambda: evidence,
    )
    monkeypatch.setattr(harness.SourceLayout, "resolve", lambda: layout)
    monkeypatch.setattr(harness, "detect_backend", lambda: "cuda")
    closure = SimpleNamespace(
        executable=SimpleNamespace(sha256="d" * 64),
        dependencies=(SimpleNamespace(path="/build/libggml-cuda.so.0", sha256="e" * 64),),
    )
    monkeypatch.setattr(harness, "capture_runtime_artifact_closure", lambda *args, **kwargs: closure)
    monkeypatch.setattr(harness, "asdict", lambda value: {
        "executable": {"sha256": value.executable.sha256},
        "dependencies": [
            {"path": item.path, "sha256": item.sha256} for item in value.dependencies
        ],
    })

    captured = harness.capture_source_evidence({"DFLASH_PREFILL_UBATCH": "1024"})

    assert captured["backend"] == "cuda"
    assert captured["binary_sha256"] == "d" * 64
    assert captured["runtime_artifacts"]["dependencies"] == [
        {"path": "/build/libggml-cuda.so.0", "sha256": "e" * 64},
    ]
    assert captured["product_backends"] == ["cuda", "hip"]
    assert captured["vendor_backends"] == ["cpu", "cuda", "hip", "vulkan"]


def test_run_rejects_source_or_binary_changes_after_build():
    from autoluce.bench.harness import require_stable_source_evidence

    built = {"product_digest": "a", "vendor_digest": "b", "binary_sha256": "c"}
    require_stable_source_evidence(built, dict(built))

    changed = {**built, "vendor_digest": "different"}
    with pytest.raises(RuntimeError, match="changed after the build"):
        require_stable_source_evidence(built, changed)


def test_mutated_runtime_artifacts_are_rejected_before_baseline_is_saved(monkeypatch):
    from autoluce.bench import harness

    monkeypatch.setenv("AUTOLUCE_BENCHMARKS", "smoke")
    layout = SimpleNamespace(require_capability=lambda capability: None)
    monkeypatch.setattr(harness.SourceLayout, "resolve", lambda: layout)
    monkeypatch.setattr(harness, "reset_lucebox", lambda: None)
    monkeypatch.setattr(harness, "build", lambda: 1.0)
    monkeypatch.setattr(harness, "get_cmake_flags", lambda: [])
    monkeypatch.setattr(harness, "get_runtime_flags", lambda: {})
    evidence = iter([
        {"runtime_artifacts": {"dependencies": [{"sha256": "before"}]}},
        {"runtime_artifacts": {"dependencies": [{"sha256": "after"}]}},
    ])
    monkeypatch.setattr(harness, "capture_source_evidence", lambda runtime_env: next(evidence))
    monkeypatch.setattr(harness, "run_single_benchmark", lambda *args, **kwargs: {
        "benchmark": "smoke",
        "score": 1.0,
        "score_stddev": 0.0,
        "decode_tok_s": 1.0,
        "prefill_tok_s": 1.0,
        "peak_mem_GiB": 1.0,
        "correctness": "pass",
    })
    saved = []
    monkeypatch.setattr(harness, "save_baseline_metrics", lambda results: saved.append(results))

    with pytest.raises(RuntimeError, match="changed after the build"):
        harness._run_harness_unlocked(True, False, False, 1.0)

    assert saved == []


def test_source_run_lease_rejects_a_second_local_worker(tmp_path):
    from autoluce.bench.harness import source_run_lease

    lock = tmp_path / "source.lock"
    with source_run_lease(lock):
        with pytest.raises(RuntimeError, match="source/build lease"):
            with source_run_lease(lock):
                pass


def test_default_frontier_excludes_the_non_scoring_test_drive_canary():
    from autoluce.bench.harness import list_benchmarks

    assert "deepseek-v4-test-drive" not in list_benchmarks()


def test_target_only_benchmark_never_loads_the_manifest_draft(monkeypatch):
    from autoluce.bench import harness

    monkeypatch.setattr(harness, "resolve_model", lambda entry, role: Path(f"/{entry}-{role}.gguf"))

    target, draft = harness.resolve_benchmark_models({
        "manifest_entry": "qwen36-27b",
        "spec_type": "target-only",
    })

    assert target == Path("/qwen36-27b-target.gguf")
    assert draft is None


def test_target_only_golden_freeze_never_loads_the_manifest_draft(monkeypatch):
    from scripts import generate_golden

    monkeypatch.setattr(generate_golden, "resolve_model", lambda entry, role: Path(f"/{entry}-{role}.gguf"))

    target, draft = generate_golden.resolve_benchmark_models({
        "manifest_entry": "qwen36-27b",
        "spec_type": "target-only",
    })

    assert target == Path("/qwen36-27b-target.gguf")
    assert draft is None
