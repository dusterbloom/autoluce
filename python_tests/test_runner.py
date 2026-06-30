"""
Tests for the parallel experiment fan-out: dispatch (concurrent execution via an
injected run_fn) and screen (significance-based candidate selection).

The run_fn is injected so these tests need no GPU/VM — a fake stands in for the
local-subprocess / SSH / sky-exec worker.
"""

import time

from runner import ExperimentSpec, dispatch, screen


def _spec(i: int) -> ExperimentSpec:
    return ExperimentSpec(id=f"exp-{i}", description=f"idea {i}")


def test_dispatch_runs_all_and_returns_results():
    def run_fn(spec):
        return {"score": float(spec.id[-1]) * 10.0, "score_stddev": 1.0, "correctness": "pass"}

    results = dispatch([_spec(1), _spec(2), _spec(3)], run_fn, max_parallel=3)
    assert {r[0].id for r in results} == {"exp-1", "exp-2", "exp-3"}
    assert all(r[1]["correctness"] == "pass" for r in results)


def test_dispatch_completes_in_parallel_not_serially():
    # 4 specs x 0.10s: parallel ~0.10s, serial ~0.40s. Wide margin avoids flake.
    def run_fn(spec):
        time.sleep(0.10)
        return {"score": 100.0, "score_stddev": 1.0, "correctness": "pass"}

    specs = [_spec(i) for i in range(4)]
    t0 = time.time()
    results = dispatch(specs, run_fn, max_parallel=4)
    elapsed = time.time() - t0
    assert elapsed < 0.30  # serial would be ~0.40
    assert len(results) == 4


def test_dispatch_yields_in_completion_order():
    def run_fn(spec):
        time.sleep({"fast": 0.02, "slow": 0.20}[spec.id])
        return {"score": 1.0, "score_stddev": 0.0, "correctness": "pass"}

    specs = [ExperimentSpec(id="slow", description="s"), ExperimentSpec(id="fast", description="f")]
    results = dispatch(specs, run_fn, max_parallel=2)
    assert results[0][0].id == "fast"  # faster one completes first


def test_dispatch_turns_exceptions_into_crash_results():
    def run_fn(spec):
        raise RuntimeError("boom")

    results = dispatch([_spec(1)], run_fn, max_parallel=1)
    assert results[0][1]["correctness"] == "FAIL"
    assert results[0][1]["crash"] == "boom"


def test_screen_keeps_only_significant_winners_sorted_desc():
    catalog = {"exp-1": (90.0, 1.0), "exp-2": (110.0, 1.0), "exp-3": (150.0, 1.0), "exp-4": (105.0, 1.0)}

    def run_fn(spec):
        score, sigma = catalog[spec.id]
        return {"score": score, "score_stddev": sigma, "correctness": "pass"}

    specs = [_spec(i) for i in (1, 2, 3, 4)]
    # baseline 100 ± 5, k=1.0 -> noise ~= sqrt(sigma^2 + 5^2).
    # exp-2 (110, d=10) and exp-3 (150) clear it; exp-4 (105, d=5) is within noise.
    winners = screen(specs, run_fn, baseline_score=100.0, baseline_sigma=5.0, max_parallel=4, k=1.0)
    ids = [s.id for s, _ in winners]
    assert ids == ["exp-3", "exp-2"]  # exp-4 within noise; exp-1 a regression
    assert winners[0][1]["score"] == 150.0


def test_screen_drops_correctness_failures():
    def run_fn(spec):
        return {"score": 999.0, "score_stddev": 0.0, "correctness": "FAIL"}

    winners = screen([_spec(1)], run_fn, baseline_score=1.0, baseline_sigma=0.0, max_parallel=1)
    assert winners == []
