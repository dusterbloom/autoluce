"""
Coordination layer for safe parallel workers sharing one leaderboard.

LockedFrontier serializes access to .best_score.json + results.tsv. Its coordinated
op, claim_best_if_significant, re-reads the live best UNDER THE LOCK and re-tests
significance before writing -- the fix for the race where a worker measures a win
against a stale snapshot and then commits a regression once the frontier has moved.
This is the local equivalent of the 're-verify on a quiet worker' rule verify.py
already warns screening candidates about, enforced atomically at commit time.
"""

from __future__ import annotations

import csv
import fcntl
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from autoluce.bench.uncertainty import is_significant_improvement, is_significant_improvement_samples

RESULTS_HEADER = (
    "commit\tscore\tscore_stddev\tdecode_tok_s\t"
    "prefill_tok_s\tacceptance_rate\tpeak_mem_GiB\tstatus\tdescription"
)
# NOTE: the single source of truth for the results.tsv schema. agent_loop and any
# parallel runner must format rows through LockedFrontier so the column layout is
# stated exactly once.


@contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    """Exclusive (blocking) advisory file lock via fcntl.flock. Creates the lockfile."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@dataclass
class ClaimResult:
    claimed: bool           # True iff this claim became the new best
    best_commit: str        # commit the caller should be at (its own if claimed)


class LockedFrontier:
    """Serialized read/claim access to the shared frontier + results log."""

    def __init__(self, root: Path, k: float = 1.0, alpha: float = 0.05) -> None:
        self.root = root
        self.best_path = root / ".best_score.json"
        self.results_path = root / "results.tsv"
        self.lock_path = root / ".autoluce.lock"
        self.k = k
        self.alpha = alpha

    def read_best(self) -> dict:
        with file_lock(self.lock_path):
            return self._read_best_unlocked()

    def set_best(self, commit: str, summary: dict, description: str) -> None:
        """Force-set the frontier (baseline semantics): write best + log 'keep',
        ignoring the significance gate. Serialized so concurrent baselines don't race.
        """
        with file_lock(self.lock_path):
            self._append_result_unlocked(commit, summary, "keep", description)
            self._write_best_unlocked(
                float(summary.get("score", 0.0)),
                float(summary.get("score_stddev", 0.0)),
                commit,
                summary.get("score_samples"),
            )

    def log_result(self, commit: str, summary: dict, status: str, description: str) -> None:
        """Append a results.tsv row with an explicit status without touching the best
        frontier (used for crashes and other non-gated records)."""
        with file_lock(self.lock_path):
            self._append_result_unlocked(commit, summary, status, description)

    def claim_best_if_significant(self, commit: str, summary: dict, description: str) -> ClaimResult:
        """Atomically: re-read live best, re-test significance, write if it holds.

        Always appends a results.tsv row (keep if claimed, else discard). Returns a
        ClaimResult whose best_commit is the commit the caller should be at after:
        its own commit if it claimed, otherwise the current best to reset to.
        """
        with file_lock(self.lock_path):
            best = self._read_best_unlocked()
            best_score = float(best.get("score", 0.0))
            best_sigma = float(best.get("score_stddev", 0.0))
            score = float(summary.get("score", 0.0))
            sigma = float(summary.get("score_stddev", 0.0))
            new_samples = summary.get("score_samples")
            best_samples = best.get("score_samples")
            if new_samples and best_samples:
                # Both sides carry per-repetition samples: real Welch t-test.
                claimed = is_significant_improvement_samples(new_samples, best_samples, self.alpha)
            else:
                claimed = is_significant_improvement(score, sigma, best_score, best_sigma, self.k)
            self._append_result_unlocked(commit, summary, "keep" if claimed else "discard", description)
            if claimed:
                self._write_best_unlocked(score, sigma, commit, new_samples)
            return ClaimResult(claimed=claimed, best_commit=commit if claimed else best.get("commit", ""))

    def _read_best_unlocked(self) -> dict:
        if not self.best_path.exists():
            return {}
        return json.loads(self.best_path.read_text())

    def _write_best_unlocked(self, score: float, score_stddev: float, commit: str, score_samples=None) -> None:
        record = {
            "score": score,
            "score_stddev": score_stddev,
            "commit": commit,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if score_samples:
            record["score_samples"] = [float(sample) for sample in score_samples]
        self.best_path.write_text(json.dumps(record, indent=2))

    def _append_result_unlocked(self, commit: str, summary: dict, status: str, description: str) -> None:
        is_new = not self.results_path.exists()
        with open(self.results_path, "a", newline="") as fh:
            if is_new:
                fh.write(RESULTS_HEADER + "\n")
            csv.writer(fh, delimiter="\t", lineterminator="\n").writerow([
                commit,
                f"{summary.get('score', 0.0):.6f}",
                f"{summary.get('score_stddev', 0.0):.6f}",
                f"{summary.get('decode_tok_s', 0.0):.2f}",
                f"{summary.get('prefill_tok_s', 0.0):.2f}",
                f"{summary.get('acceptance_rate', 0.0):.4f}",
                f"{summary.get('peak_mem_GiB', 0.0):.2f}",
                status,
                description,
            ])
