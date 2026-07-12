"""Numerical quality evidence for matched product first-token logits."""

from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LogitQualityMetrics:
    vocabulary_size: int
    kl_divergence: float
    max_abs_error: float
    mean_abs_error: float
    rmse: float
    top_k: int
    top_k_overlap: float
    reference_argmax: int
    candidate_argmax: int
    argmax_changed: bool
    reference_margin: float
    candidate_margin: float


@dataclass(frozen=True)
class AggregateLogitQuality:
    sample_count: int
    mean_kl_divergence: float
    max_kl_divergence: float
    max_abs_error: float
    mean_abs_error: float
    min_top_k_overlap: float
    argmax_changes: int


def _validated(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError("logit vectors must be non-empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} logits must be finite")
    return result


def _log_softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(values)
    log_normalizer = maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))
    return tuple(value - log_normalizer for value in values)


def _top_indices(values: tuple[float, ...], count: int) -> tuple[int, ...]:
    return tuple(heapq.nsmallest(count, range(len(values)), key=lambda index: (-values[index], index)))


def compare_logits(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    top_k: int = 20,
) -> LogitQualityMetrics:
    ref = _validated(reference, "reference")
    cand = _validated(candidate, "candidate")
    if len(ref) != len(cand):
        raise ValueError("reference and candidate logits must have the same shape")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    effective_k = min(top_k, len(ref))

    differences = tuple(candidate_value - reference_value for reference_value, candidate_value in zip(ref, cand))
    ref_logp = _log_softmax(ref)
    cand_logp = _log_softmax(cand)
    kl = math.fsum(
        math.exp(reference_logp) * (reference_logp - candidate_logp)
        for reference_logp, candidate_logp in zip(ref_logp, cand_logp)
    )
    ref_top = _top_indices(ref, effective_k)
    cand_top = _top_indices(cand, effective_k)

    return LogitQualityMetrics(
        vocabulary_size=len(ref),
        kl_divergence=max(0.0, kl),
        max_abs_error=max(abs(value) for value in differences),
        mean_abs_error=math.fsum(abs(value) for value in differences) / len(differences),
        rmse=math.sqrt(math.fsum(value * value for value in differences) / len(differences)),
        top_k=effective_k,
        top_k_overlap=len(set(ref_top) & set(cand_top)) / effective_k,
        reference_argmax=ref_top[0],
        candidate_argmax=cand_top[0],
        argmax_changed=ref_top[0] != cand_top[0],
        reference_margin=ref[ref_top[0]] - ref[ref_top[1]] if len(ref_top) > 1 else math.inf,
        candidate_margin=cand[cand_top[0]] - cand[cand_top[1]] if len(cand_top) > 1 else math.inf,
    )


def logit_quality_gate(
    metrics: LogitQualityMetrics,
    *,
    kl_tau: float = 0.01,
    min_top_k_overlap: float = 0.9,
) -> list[str]:
    if kl_tau < 0:
        raise ValueError("kl_tau must be non-negative")
    if not 0 <= min_top_k_overlap <= 1:
        raise ValueError("min_top_k_overlap must be in [0, 1]")
    violations: list[str] = []
    if metrics.kl_divergence > kl_tau:
        violations.append(
            f"kl_divergence: {metrics.kl_divergence:.6g} > tau {kl_tau:.6g}"
        )
    if metrics.top_k_overlap < min_top_k_overlap:
        violations.append(
            f"top_k_overlap: {metrics.top_k_overlap:.6g} < {min_top_k_overlap:.6g}"
        )
    return violations


def aggregate_logit_quality(metrics: Sequence[LogitQualityMetrics]) -> AggregateLogitQuality:
    samples = tuple(metrics)
    if not samples:
        raise ValueError("aggregate logit quality requires at least one sample")
    return AggregateLogitQuality(
        sample_count=len(samples),
        mean_kl_divergence=math.fsum(sample.kl_divergence for sample in samples) / len(samples),
        max_kl_divergence=max(sample.kl_divergence for sample in samples),
        max_abs_error=max(sample.max_abs_error for sample in samples),
        mean_abs_error=math.fsum(sample.mean_abs_error for sample in samples) / len(samples),
        min_top_k_overlap=min(sample.top_k_overlap for sample in samples),
        argmax_changes=sum(sample.argmax_changed for sample in samples),
    )


def aggregate_logit_quality_gate(
    aggregate: AggregateLogitQuality,
    *,
    kl_tau: float = 0.01,
    min_top_k_overlap: float = 0.9,
) -> list[str]:
    if kl_tau < 0:
        raise ValueError("kl_tau must be non-negative")
    if not 0 <= min_top_k_overlap <= 1:
        raise ValueError("min_top_k_overlap must be in [0, 1]")
    violations: list[str] = []
    if aggregate.mean_kl_divergence > kl_tau:
        violations.append(
            f"mean_kl_divergence: {aggregate.mean_kl_divergence:.6g} > tau {kl_tau:.6g}"
        )
    if aggregate.max_kl_divergence > 10 * kl_tau:
        violations.append(
            f"max_kl_divergence: {aggregate.max_kl_divergence:.6g} > 10*tau {10 * kl_tau:.6g}"
        )
    if aggregate.min_top_k_overlap < min_top_k_overlap:
        violations.append(
            f"min_top_k_overlap: {aggregate.min_top_k_overlap:.6g} < {min_top_k_overlap:.6g}"
        )
    return violations
