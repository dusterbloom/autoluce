"""
Tests for the parallel experiment fan-out: dispatch (concurrent execution via an
injected run_fn) and screen (significance-based candidate selection).

The run_fn is injected so these tests need no GPU/VM — a fake stands in for the
local-subprocess / SSH / sky-exec worker.
"""

import json
import subprocess
import time
from pathlib import Path

from autoggml.parallel.concurrency import LockedFrontier
from autoggml.parallel.runner import ExperimentSpec, _harness_command, _parse_harness_json, dispatch, local_run, run_parallel, screen


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


# --- run_parallel: live-frontier funnel (vs screen's fixed-snapshot filter) -----

def _result(score: float, correctness: str = "pass", sigma: float = 0.01) -> dict:
    return {
        "score": score, "score_stddev": sigma, "correctness": correctness,
        "decode_tok_s": 1.0, "prefill_tok_s": 1.0, "acceptance_rate": 1.0, "peak_mem_GiB": 1.0,
    }


def test_run_parallel_skips_correctness_failures(tmp_path):
    frontier = LockedFrontier(tmp_path)
    catalog = {"exp-1": _result(100.0), "exp-2": _result(999.0, "FAIL")}
    results = run_parallel([_spec(1), _spec(2)], lambda s: catalog[s.id], frontier, max_parallel=2)
    assert {r[0].id for r in results} == {"exp-1"}  # exp-2 skipped, never claimed
    assert frontier.read_best()["score"] == 100.0


def test_run_parallel_advancing_frontier_ends_at_the_best(tmp_path):
    frontier = LockedFrontier(tmp_path)
    catalog = {"exp-1": _result(50.0), "exp-2": _result(200.0)}
    results = run_parallel([_spec(1), _spec(2)], lambda s: catalog[s.id], frontier, max_parallel=2)
    # Order-independent: the live frontier's best ends at the max score.
    assert frontier.read_best()["score"] == 200.0
    assert "exp-2" in {s.id for s, c in results if c.claimed}


def test_run_parallel_empty_specs_is_noop(tmp_path):
    frontier = LockedFrontier(tmp_path)
    assert run_parallel([], lambda s: {}, frontier, max_parallel=2) == []
    assert frontier.read_best() == {}


def test_run_parallel_records_real_commit_from_result_not_spec_id(tmp_path):
    # The frontier's `commit` must be a git ref (agent_loop.git_reset consumes it),
    # not the spec's arbitrary id label. A run_fn that measures a real commit must
    # see it land in .best_score.json.
    frontier = LockedFrontier(tmp_path)

    def run_fn(spec):
        return {**_result(100.0), "commit": "abc1234"}

    run_parallel([_spec(1)], run_fn, frontier, max_parallel=1)
    assert frontier.read_best()["commit"] == "abc1234"


# --- local_run: the local run_fn that consumes worktree + gives run_parallel a body —

def test_harness_command_is_uv_run_python_harness_json():
    cmd = _harness_command(Path("/tmp/wt"))
    assert cmd[:3] == ["uv", "run", "python"]
    assert "autoggml.bench.harness" in cmd
    assert "--json" in cmd


def test_parse_harness_json_happy_path():
    stdout = json.dumps({"score": 42.0, "score_stddev": 1.5, "correctness": "pass"})
    assert _parse_harness_json(stdout, commit="abc1234", spec_id="s1") == {
        "score": 42.0, "score_stddev": 1.5, "correctness": "pass", "commit": "abc1234",
    }


def test_parse_harness_json_malformed_returns_crash_dict():
    r = _parse_harness_json("not json", commit="abc1234", spec_id="s1")
    assert r["correctness"] == "FAIL"
    assert r["score"] == 0.0
    assert r["commit"] == "abc1234"
    assert "crash" in r


def test_local_run_ensures_worktree_and_returns_real_commit(tmp_path):
    # A real git repo so ensure_worktree succeeds; a fake transport so no real build runs.
    repo = tmp_path / "repo"
    repo.mkdir()
    for c in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.t"], ["git", "config", "user.name", "t"]):
        subprocess.run(c, cwd=repo, check=True)
    (repo / "harness.py").write_text("# stub")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()

    class _Proc:
        stdout = json.dumps({"score": 99.0, "score_stddev": 2.0, "correctness": "pass"})

    result = local_run(ExperimentSpec(id="w1", description="t"), repo, run_subprocess=lambda cmd, **kw: _Proc())
    assert result["score"] == 99.0
    assert result["correctness"] == "pass"
    assert result["commit"] == head  # the worktree's real HEAD, not "w1"
    assert (repo / ".worktrees" / "w1" / "harness.py").exists()  # worktree actually created
