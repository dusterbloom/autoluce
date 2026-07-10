"""
Smoke tests for the autoluce v2 harness.

These tests run the harness in simulation mode (no real Lucebox product source
required) and verify that it produces a valid, reproducible result bundle.
"""

import json
from pathlib import Path

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


def test_default_frontier_excludes_the_non_scoring_test_drive_canary():
    from autoluce.bench.harness import list_benchmarks

    assert "deepseek-v4-test-drive" not in list_benchmarks()
