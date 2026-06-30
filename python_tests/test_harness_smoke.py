"""
Smoke tests for the autoggml v2 harness.

These tests run the harness in simulation mode (no real lucebox-ggml source
required) and verify that it produces a valid, reproducible result bundle.
"""

import json
from pathlib import Path

import pytest

from harness import run_harness
from reproduce import capture_environment


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


def test_simulated_summary_carries_score_stddev():
    summary = run_harness(baseline=True, simulate=True)
    assert "score_stddev" in summary
    # Sim fixture: decode=120±4, prefill=2500±100, acceptance=1.0, mem=18, build_time excluded.
    # score = 120*2500*1/18; sigma = score*sqrt((4/120)^2+(100/2500)^2)
    assert summary["score"] == pytest.approx(16666.6667, rel=1e-4)
    assert summary["score_stddev"] == pytest.approx(867.8055, rel=1e-4)
