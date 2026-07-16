"""
Tests for the coordination layer: file_lock + LockedFrontier.

The concurrent-claim test is the one that proves the design. Two workers that both
measured a 'significant' win against the same starting snapshot race to commit; the
frontier must end at the better score (monotonic, never a regression) with both
attempts recorded, regardless of scheduling.
"""

import json
import threading
from pathlib import Path

from autoluce.parallel.concurrency import LockedFrontier, file_lock


def _summary(score: float, sigma: float = 0.01) -> dict:
    return {
        "score": score,
        "score_stddev": sigma,
        "decode_tok_s": 1.0,
        "prefill_tok_s": 1.0,
        "acceptance_rate": 1.0,
        "peak_mem_GiB": 1.0,
    }


def _best_score(tmp_path: Path) -> float:
    return json.loads((tmp_path / ".best_score.json").read_text())["score"]


def _data_rows(tmp_path: Path) -> list[str]:
    lines = (tmp_path / "results.tsv").read_text().splitlines()
    return lines[1:] if lines and lines[0].startswith("commit") else lines


# --- file_lock ----------------------------------------------------------------

def test_file_lock_is_a_reusable_context_manager(tmp_path):
    lock = tmp_path / "coord.lock"
    with file_lock(lock):
        pass
    with file_lock(lock):
        pass
    assert lock.exists()


# --- claim semantics (sequential) ---------------------------------------------

def test_claim_on_empty_frontier_becomes_best(tmp_path):
    f = LockedFrontier(tmp_path)
    result = f.claim_best_if_significant("c1", _summary(100.0), "first")
    assert result.claimed is True
    assert _best_score(tmp_path) == 100.0


def test_significant_claim_replaces_best(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _summary(100.0, 0.01), "first")
    result = f.claim_best_if_significant("c2", _summary(200.0, 0.01), "better")
    assert result.claimed is True
    assert _best_score(tmp_path) == 200.0


def test_non_significant_claim_does_not_beat_best(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _summary(100.0, 5.0), "first")
    # delta=1.0 vs noise = 1.0*sqrt(5^2+5^2) ~= 7.07 -> within noise, not significant.
    result = f.claim_best_if_significant("c2", _summary(101.0, 5.0), "within noise")
    assert result.claimed is False
    assert _best_score(tmp_path) == 100.0


def test_lower_score_never_lands_frontier_is_monotonic(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _summary(100.0, 0.01), "first")
    result = f.claim_best_if_significant("c2", _summary(50.0, 0.01), "regression")
    assert result.claimed is False
    assert _best_score(tmp_path) == 100.0


def test_claim_records_row_with_status(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _summary(100.0, 5.0), "first")
    f.claim_best_if_significant("c2", _summary(101.0, 5.0), "within noise")
    rows = _data_rows(tmp_path)
    assert len(rows) == 2
    assert "keep" in rows[0] and "discard" in rows[1]


# --- the race the design exists to survive ------------------------------------

def test_concurrent_claims_keep_monotonic_frontier_and_record_both(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("seed", _summary(10.0, 0.01), "seed")  # baseline best

    outcomes: list[tuple[str, bool]] = []
    barrier = threading.Barrier(2)

    def worker(commit: str, score: float) -> None:
        barrier.wait()  # release both together so they race
        r = f.claim_best_if_significant(commit, _summary(score, 0.01), f"worker-{commit}")
        outcomes.append((commit, r.claimed))

    t_a = threading.Thread(target=worker, args=("A", 15.0))
    t_b = threading.Thread(target=worker, args=("B", 12.0))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    # Monotonic, never a regression: final best is the maximum score seen.
    best = json.loads((tmp_path / ".best_score.json").read_text())
    assert best["score"] == 15.0
    assert best["commit"] == "A"
    # The 15-score worker always clears the gate (15 beats anything <= 12 it raced against).
    assert ("A", True) in outcomes
    # Every attempt is recorded exactly once -- no lost or duplicated rows.
    assert len(_data_rows(tmp_path)) == 3  # seed + A + B


# --- baseline force-set + crash log (no gate, no best change) ------------------

def test_set_best_overrides_regardless_of_existing(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _summary(100.0, 0.01), "high")
    # set_best writes 50 even though 50 < 100 (baseline semantics: set the reference).
    f.set_best("base", _summary(50.0, 0.01), "baseline")
    assert _best_score(tmp_path) == 50.0
    assert _data_rows(tmp_path)[-1].split("\t")[0] == "base"


def test_log_result_appends_without_changing_best(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _summary(100.0, 0.01), "best")
    f.log_result("crash1", _summary(0.0, 0.0), "crash", "boom")
    assert _best_score(tmp_path) == 100.0  # frontier untouched
    assert "crash" in _data_rows(tmp_path)[-1]


# --- sample-backed claims (Welch gate) ------------------------------------------


def _sampled_summary(score: float, samples: list[float], sigma: float = 0.01) -> dict:
    summary = _summary(score, sigma)
    summary["score_samples"] = samples
    return summary


def test_sample_backed_clear_win_claims_and_persists_samples(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _sampled_summary(100.0, [99.0, 100.0, 101.0]), "first")
    result = f.claim_best_if_significant("c2", _sampled_summary(110.0, [109.0, 110.0, 111.0]), "better")
    assert result.claimed is True
    best = json.loads((tmp_path / ".best_score.json").read_text())
    assert best["score_samples"] == [109.0, 110.0, 111.0]


def test_sample_backed_overlapping_claim_is_rejected(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _sampled_summary(100.0, [99.0, 100.0, 101.0]), "first")
    # Mean 100.23 with overlapping spread: a point-estimate bump inside the noise.
    noisy = _sampled_summary(100.23, [100.2, 101.0, 99.5], sigma=0.76)
    result = f.claim_best_if_significant("c2", noisy, "within noise")
    assert result.claimed is False
    assert _best_score(tmp_path) == 100.0


def test_mixed_sigma_candidate_against_sampled_best_uses_legacy_gate(tmp_path):
    f = LockedFrontier(tmp_path)
    f.claim_best_if_significant("c1", _sampled_summary(100.0, [99.0, 100.0, 101.0]), "first")
    # Sigma-only candidate: no samples on one side -> legacy k*sigma comparison.
    result = f.claim_best_if_significant("c2", _summary(150.0, 1.0), "big win")
    assert result.claimed is True


def test_set_best_persists_samples_for_later_welch_claims(tmp_path):
    f = LockedFrontier(tmp_path)
    f.set_best("base", _sampled_summary(100.0, [99.0, 100.0, 101.0]), "baseline")
    best = json.loads((tmp_path / ".best_score.json").read_text())
    assert best["score_samples"] == [99.0, 100.0, 101.0]
    result = f.claim_best_if_significant("c2", _sampled_summary(110.0, [109.0, 110.0, 111.0]), "better")
    assert result.claimed is True
