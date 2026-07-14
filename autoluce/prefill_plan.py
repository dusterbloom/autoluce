"""Deterministic planning policy for Qwen3.6 normal-KV prefill research.

This module does not execute a benchmark.  It defines the small amount of policy
that must be settled before a GPU is leased: explicit KV-cache lanes, compatible
campaign identities, context eligibility, isolated hypotheses, and the minimum
evidence required for promotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

from autoluce.research_contract import Campaign


FRONTIER_CONTEXTS = (1024, 8192, 16384, 65536)


@dataclass(frozen=True)
class KVCacheLane:
    """One explicit, symmetric normal-quant KV-cache lane."""

    name: str
    key_type: str
    value_type: str
    fit_probe_contexts: tuple[int, ...] = ()

    @property
    def runtime_env(self) -> dict[str, str]:
        return {
            "DFLASH27B_KV_K": self.key_type,
            "DFLASH27B_KV_V": self.value_type,
        }

    @property
    def quantization(self) -> dict[str, str]:
        return {"key": self.key_type, "value": self.value_type}


NORMAL_KV_LANES = (
    KVCacheLane("f16-f16", "f16", "f16"),
    KVCacheLane("q8_0-q8_0", "q8_0", "q8_0", fit_probe_contexts=(131072,)),
    KVCacheLane("q4_0-q4_0", "q4_0", "q4_0", fit_probe_contexts=(131072,)),
)
_LANES_BY_NAME = {lane.name: lane for lane in NORMAL_KV_LANES}


@dataclass(frozen=True)
class MeasurementCell:
    context: int
    frontier_eligible: bool
    fit_probe: bool = False


@dataclass(frozen=True)
class PrefillHypothesis:
    """A bounded experiment which changes one product policy at a time."""

    name: str
    question: str
    controls: tuple[str, ...]
    requires_profile: bool = False


PREFILL_HYPOTHESES = (
    PrefillHypothesis(
        "explicit-baseline",
        "Capture a clean local baseline with both KV types explicitly selected.",
        (),
    ),
    PrefillHypothesis(
        "attention-dispatch",
        "Verify which flash-attention implementation each batched normal-quant lane actually dispatches.",
        ("fattn_kernel",),
        requires_profile=True,
    ),
    PrefillHypothesis(
        "gdn-chunking",
        "Remove the serial-token GDN prefill bottleneck while preserving the final recurrent state.",
        ("gdn_chunk_size",),
        requires_profile=True,
    ),
    PrefillHypothesis(
        "causal-mask",
        "Avoid materializing or uploading a full causal mask when the kernel can derive it.",
        ("causal_mask_materialization",),
        requires_profile=True,
    ),
    PrefillHypothesis(
        "quantized-attention",
        "Keep Q8_0/Q4_0 K and V compressed through the batched attention path.",
        ("quantized_fattn_path",),
        requires_profile=True,
    ),
    PrefillHypothesis(
        "depth-schedule",
        "Choose prefill chunk and ubatch sizes from context depth without changing attention semantics.",
        ("prefill_ubatch",),
    ),
    PrefillHypothesis(
        "graph-cleanup",
        "Remove graph outputs and synchronizations proven unnecessary for target-only prefill.",
        ("graph_output_policy",),
        requires_profile=True,
    ),
)


def normal_kv_lane(name: str) -> KVCacheLane:
    try:
        return _LANES_BY_NAME[name.lower()]
    except KeyError as error:
        supported = ", ".join(_LANES_BY_NAME)
        raise ValueError(f"'{name}' is not a normal KV campaign lane; choose {supported}") from error


def _weight_quantization(value: Any) -> Any:
    if isinstance(value, Mapping):
        if "weights" not in value:
            raise ValueError("structured quantization requires a weights identity")
        return value["weights"]
    return value


def campaign_for_kv_lane(template: Campaign, lane_name: str) -> Campaign:
    """Derive one campaign without conflating model weights and KV-cache types."""

    lane = normal_kv_lane(lane_name)
    if template.workload.get("mode") != "prefill":
        raise ValueError("normal KV campaign template must use prefill mode")
    if tuple(int(value) for value in template.workload["contexts"]) != FRONTIER_CONTEXTS:
        raise ValueError(f"normal KV campaign contexts must be {list(FRONTIER_CONTEXTS)}")

    system = dict(template.system)
    system["quantization"] = {
        "weights": _weight_quantization(system["quantization"]),
        "kv_cache": lane.quantization,
    }
    return replace(
        template,
        name=f"{template.name}-{lane.name}",
        system=system,
    )


def measurement_cells(lane_name: str) -> tuple[MeasurementCell, ...]:
    lane = normal_kv_lane(lane_name)
    frontier = tuple(MeasurementCell(context, frontier_eligible=True) for context in FRONTIER_CONTEXTS)
    probes = tuple(
        MeasurementCell(context, frontier_eligible=False, fit_probe=True)
        for context in lane.fit_probe_contexts
    )
    return frontier + probes


def _require_numeric(mapping: Mapping[str, Any], name: str, *, positive: bool = False) -> float:
    value = mapping.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"promotion measurement requires finite numeric {name}")
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f"promotion measurement requires positive {name}")
    if not positive and number < 0:
        raise ValueError(f"promotion measurement requires non-negative {name}")
    return number


def require_promotion_measurement(
    measurement: Mapping[str, Any],
    lane_name: str,
    *,
    context: int,
    prompt_tolerance: float = 0.05,
) -> None:
    """Fail closed unless a result is complete enough to enter the frontier.

    Exploratory captures may remain partial.  Promotion evidence must include
    dispersion, exact-quality status, immutable artifact identities, the resolved
    KV pair, and proof that the prompt represented the requested context cell.
    """

    lane = normal_kv_lane(lane_name)
    cell = next((item for item in measurement_cells(lane_name) if item.context == context), None)
    if cell is None:
        raise ValueError(f"context {context} is not part of lane {lane.name}")
    if not cell.frontier_eligible:
        raise ValueError(f"context {context} is a fit probe and is not frontier eligible")
    if not 0 <= prompt_tolerance < 1:
        raise ValueError("prompt tolerance must be in [0, 1)")

    metrics = measurement.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("promotion measurement requires metrics")
    _require_numeric(metrics, "prefill_tok_s", positive=True)
    _require_numeric(metrics, "prefill_tok_s_stddev")
    _require_numeric(metrics, "peak_mem_GiB", positive=True)

    quality = measurement.get("quality")
    correctness = quality.get("correctness") if isinstance(quality, Mapping) else None
    if not (
        isinstance(correctness, Mapping)
        and correctness.get("kind") == "exact"
        and correctness.get("passed") is True
    ):
        raise ValueError("promotion measurement requires passing exact correctness evidence")

    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("promotion measurement requires provenance")
    for name in (
        "binary_sha256",
        "product_digest",
        "model_fingerprint",
        "machine_fingerprint",
    ):
        if not provenance.get(name):
            raise ValueError(f"promotion measurement requires provenance.{name}")
    if provenance.get("resolved_kv_cache") != lane.quantization:
        raise ValueError(f"promotion measurement resolved KV cache does not match {lane.quantization}")
    if provenance.get("context_depth") != context:
        raise ValueError("promotion measurement context depth does not match the requested cell")

    prompt_tokens = _require_numeric(provenance, "prompt_tokens", positive=True)
    lower, upper = context * (1 - prompt_tolerance), context * (1 + prompt_tolerance)
    if not lower <= prompt_tokens <= upper:
        raise ValueError("promotion measurement prompt tokens do not represent the requested context cell")
    samples = metrics.get("prefill_tok_s_samples")
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("promotion measurement requires at least three ordered samples")
    for sample in samples:
        if not isinstance(sample, (int, float)) or isinstance(sample, bool) or not math.isfinite(float(sample)):
            raise ValueError("promotion measurement samples must be finite numbers")


def promotion_bundle_violations(raw: Mapping[str, Any], campaign: Campaign) -> list[str]:
    """Validate every frontier context in a normal-KV result bundle."""

    quantization = campaign.system.get("quantization")
    kv_cache = quantization.get("kv_cache") if isinstance(quantization, Mapping) else None
    lane = next((item for item in NORMAL_KV_LANES if item.quantization == kv_cache), None)
    if lane is None:
        return ["campaign quantization does not identify a supported normal KV lane"]

    cells: dict[int, Mapping[str, Any]] = {}
    for benchmark in raw.get("benchmarks", []):
        if not isinstance(benchmark, Mapping):
            continue
        for cell in benchmark.get("context_metrics", []):
            if isinstance(cell, Mapping) and cell.get("context_depth") is not None:
                cells[int(cell["context_depth"])] = cell

    source = raw.get("provenance", raw.get("source_evidence", {}))
    source = dict(source) if isinstance(source, Mapping) else {}
    source.setdefault("model_fingerprint", campaign.system.get("model_fingerprint"))
    source.setdefault("machine_fingerprint", campaign.system.get("machine"))
    quality = raw.get("quality")
    if not isinstance(quality, Mapping):
        passed = raw.get("correctness") is True or str(raw.get("correctness", "")).lower() == "pass"
        quality = {"correctness": {"kind": "exact", "passed": passed}}

    violations = []
    for context in FRONTIER_CONTEXTS:
        cell = cells.get(context)
        if cell is None:
            violations.append(f"context {context}: missing measurement cell")
            continue
        provenance = {
            **source,
            "context_depth": cell.get("context_depth"),
            "prompt_tokens": cell.get("prompt_tokens"),
        }
        try:
            require_promotion_measurement(
                {"metrics": cell, "quality": quality, "provenance": provenance},
                lane.name,
                context=context,
            )
        except ValueError as error:
            violations.append(f"context {context}: {error}")
    return violations
