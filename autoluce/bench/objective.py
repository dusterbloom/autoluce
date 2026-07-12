"""Constrained, contract-selected objective for AutoLuce.

The score is a single maximized throughput metric; everything else is a
constraint declared in the benchmark JSON's optional "objective" block:

    {"objective": {"maximize": "decode_tok_s",
                   "constraints": {"peak_mem_GiB": {"max": 22.0},
                                   "prefill_tok_s": {"min_frac_of_baseline": 0.95}}}}

Pure module (no IO): the harness loads metrics/baseline and passes dicts in,
so every edge case is testable with synthetic data.
"""

from __future__ import annotations


SUPPORTED_OBJECTIVES = frozenset({"decode_tok_s", "prefill_tok_s"})


def objective_metric(spec: dict) -> str:
    """Return the measured throughput axis selected by a benchmark contract."""
    metric = str(spec.get("objective", {}).get("maximize", "decode_tok_s"))
    if metric not in SUPPORTED_OBJECTIVES:
        supported = ", ".join(sorted(SUPPORTED_OBJECTIVES))
        raise ValueError(f"unsupported objective '{metric}'; choose one of: {supported}")
    return metric


def context_regression_metrics(spec: dict) -> tuple[str, ...]:
    """Return the throughput axes the context-depth regression floor should guard.

    Prefill-only contracts (objective "prefill_tok_s", typically n_predict=1)
    have no meaningful decode signal, so only prefill is guarded. Every other
    contract restores decode benchmarks' prior floor on both axes.
    """
    if objective_metric(spec) == "prefill_tok_s":
        return ("prefill_tok_s",)
    return ("decode_tok_s", "prefill_tok_s")


def check_constraints(metrics: dict, baseline: dict | None, spec: dict, k: float) -> list[str]:
    """Return human-readable constraint violations (empty list = pass).

    Each bound must hold with a k*stddev significance margin: for an upper bound
    `value + k*stddev <= bound`; for a lower bound `value - k*stddev >= bound`.
    A missing `<metric>_stddev` counts as 0. "min_frac_of_baseline" derives a
    lower bound from the baseline metrics and raises if baseline is None.
    """
    violations: list[str] = []
    constraints = spec.get("objective", {}).get("constraints", {})
    for metric, forms in constraints.items():
        value = metrics.get(metric)
        if value is None:
            violations.append(f"{metric}: constrained but not measured")
            continue
        margin = k * metrics.get(f"{metric}_stddev", 0.0)
        for form, bound in forms.items():
            if form == "max":
                if value + margin > bound:
                    violations.append(f"{metric}: {value:.4g} + {margin:.4g} (k*sigma) > max {bound:.4g}")
            elif form == "min":
                if value - margin < bound:
                    violations.append(f"{metric}: {value:.4g} - {margin:.4g} (k*sigma) < min {bound:.4g}")
            elif form == "min_frac_of_baseline":
                if baseline is None:
                    raise ValueError(f"{metric}.min_frac_of_baseline requires baseline metrics; run `autoluce baseline` first")
                floor = bound * baseline[metric]
                if value - margin < floor:
                    violations.append(
                        f"{metric}: {value:.4g} - {margin:.4g} (k*sigma) < {bound}*baseline({baseline[metric]:.4g}) = {floor:.4g}"
                    )
            else:
                raise ValueError(f"unknown constraint form '{form}' on {metric}")
    return violations
