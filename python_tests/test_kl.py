"""
Synthetic-data tests for the KL-to-baseline quality oracle (kl.py).

No GPU, no subprocess: pure parsing/gating/command-builder logic against
fixtures shaped like llama-perplexity's real --kl-divergence stdout
(verified against the historical standalone tools/perplexity/perplexity.cpp:
'Mean    KLD:', 'Maximum KLD:', '99.9%   KLD:').
"""

import pytest

from autoggml.bench.kl import (
    DEFAULT_KL_TAU,
    build_kl_base_cmd,
    build_kl_check_cmd,
    check_kl,
    kl_base_path,
    kl_gate,
    parse_kl_output,
    resolve_kl_tau,
)

# %10.6f-formatted summary block as printed by llama-perplexity.
KL_STDOUT = """
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Same top p
====== Perplexity statistics ======
Mean PPL(Q)                   :   7.834323 ±   0.104382
====== KL divergence statistics ======
Mean    KLD:   0.002331 ±   0.000123
Maximum KLD:   0.049919
99.9%   KLD:   0.030293
99.0%   KLD:   0.015615
95.0%   KLD:   0.008551
Median  KLD:   0.001034
Minimum KLD:   0.000000
"""


# ---------------------------------------------------------------------------
# parse_kl_output
# ---------------------------------------------------------------------------


def test_parse_kl_output_present():
    kl = parse_kl_output(KL_STDOUT)
    assert kl["mean_kld"] == pytest.approx(0.002331)
    assert kl["max_kld"] == pytest.approx(0.049919)
    assert kl["p999_kld"] == pytest.approx(0.030293)


def test_parse_kl_output_absent_raises():
    with pytest.raises(RuntimeError, match="mean_kld"):
        parse_kl_output("perplexity: calculating perplexity over 32 chunks\n[1]4.2\n")


def test_parse_kl_output_partial_raises():
    # Mean present but Maximum missing -> measure or raise, no silent defaults.
    with pytest.raises(RuntimeError, match="max_kld"):
        parse_kl_output("Mean    KLD:   0.002331 ±   0.000123\n")


def test_parse_kl_output_p999_optional():
    # p999 is a logged diagnostic, not gated; its absence must not fail the run.
    kl = parse_kl_output("Mean    KLD:   0.002331 ±   0.000123\nMaximum KLD:   0.049919\n")
    assert kl["mean_kld"] == pytest.approx(0.002331)
    assert "p999_kld" not in kl


def test_parse_kl_output_malformed_value_raises():
    with pytest.raises(RuntimeError):
        parse_kl_output("Mean    KLD: N/A\nMaximum KLD: N/A\n99.9%   KLD: N/A\n")


# ---------------------------------------------------------------------------
# kl_gate
# ---------------------------------------------------------------------------


def _kl(mean, mx, p999=0.0):
    return {"mean_kld": mean, "max_kld": mx, "p999_kld": p999}


def test_kl_gate_passes_at_exactly_tau_and_10tau():
    assert kl_gate(_kl(0.01, 0.1), tau=0.01) == []


def test_kl_gate_mean_just_above_tau_fails():
    violations = kl_gate(_kl(0.0101, 0.0), tau=0.01)
    assert len(violations) == 1
    assert "mean_kld" in violations[0]


def test_kl_gate_max_just_above_10tau_fails():
    violations = kl_gate(_kl(0.0, 0.101), tau=0.01)
    assert len(violations) == 1
    assert "max_kld" in violations[0]


def test_kl_gate_both_fail():
    assert len(kl_gate(_kl(1.0, 1.0), tau=0.01)) == 2


def test_kl_gate_default_tau_is_0_01():
    assert DEFAULT_KL_TAU == 0.01
    assert kl_gate(_kl(0.01, 0.1)) == []
    assert kl_gate(_kl(0.011, 0.0)) != []


def test_resolve_kl_tau_default_and_override():
    assert resolve_kl_tau({}) == DEFAULT_KL_TAU
    assert resolve_kl_tau({"objective": {"kl_tau": 0.05}}) == 0.05


# ---------------------------------------------------------------------------
# command builders
# ---------------------------------------------------------------------------


def test_build_kl_base_cmd():
    cmd = build_kl_base_cmd("bin/llama-perplexity", "m.gguf", "wiki.txt", "base.bin", ["--flash-attn", "1"])
    assert cmd == [
        "bin/llama-perplexity", "-m", "m.gguf", "-f", "wiki.txt",
        "--kl-divergence-base", "base.bin", "--flash-attn", "1",
    ]


def test_build_kl_check_cmd_reads_tokens_from_base_file():
    # The check run takes no -f: eval tokens are embedded in the base file.
    cmd = build_kl_check_cmd("bin/llama-perplexity", "m.gguf", "base.bin")
    assert cmd == ["bin/llama-perplexity", "-m", "m.gguf", "--kl-divergence-base", "base.bin", "--kl-divergence"]
    assert "-f" not in cmd


def test_kl_base_path_derived_from_benchmark_name():
    path = kl_base_path("smoke")
    assert path.name == "smoke.kl_base.bin"
    assert path.parent.name == "golden"


def test_check_kl_missing_base_file_tells_user_to_generate_it():
    with pytest.raises(RuntimeError, match="kl-base"):
        check_kl("no-such-benchmark-xyz", "m.gguf")


# ---------------------------------------------------------------------------
# CLI routing
# ---------------------------------------------------------------------------


def test_cli_routes_kl_base():
    from cli import resolve

    module, args = resolve(["kl-base", "smoke"])
    assert module == "autoggml.bench.kl"
    assert args == ["smoke"]
