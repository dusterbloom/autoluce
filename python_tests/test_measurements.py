"""
Tests for real metric parsing and the (fabrication-free) score formula.
"""

from harness import compute_score, parse_acceptance_rate, parse_llama_bench_output, parse_peak_memory


def test_parse_llama_bench_output_captures_stddev():
    line = "| qwen2 1.5B Q4_0 | 885.97 MiB | 1.54 B | CPU | 16 | tg128 | 132.45 ± 6.23 |"
    metrics = parse_llama_bench_output(line)
    assert metrics["decode_tok_s"] == 132.45
    assert metrics["decode_tok_s_stddev"] == 6.23


def test_parse_llama_bench_output_stddev_optional():
    line = "| qwen2 1.5B Q4_0 | 885.97 MiB | 1.54 B | CPU | 16 | pp512 | 5765.41 |"
    metrics = parse_llama_bench_output(line)
    assert metrics["prefill_tok_s"] == 5765.41
    assert "prefill_tok_s_stddev" not in metrics


def test_parse_peak_memory_converts_kib_to_gib():
    stderr = "\n        Maximum resident set size (kbytes): 19005440\n"
    assert parse_peak_memory(stderr) == 19005440 / (1024 * 1024)


def test_parse_peak_memory_raises_on_missing_metric():
    import pytest
    with pytest.raises(RuntimeError):
        parse_peak_memory("Command exited with non-zero status 1\n")


def test_parse_acceptance_rate_extracts_decimal():
    assert parse_acceptance_rate("... acc: 0.6543 ...") == 0.6543
    assert parse_acceptance_rate("... acceptance=0.70 ...") == 0.70


def test_parse_acceptance_rate_returns_none_when_absent():
    assert parse_acceptance_rate("| qwen2 1.5B Q4_0 | pp512 | 5765.41 |") is None


def test_compute_score_is_honest_ratio():
    metrics = {
        "decode_tok_s": 100.0,
        "prefill_tok_s": 2000.0,
        "acceptance_rate": 0.5,
        "peak_mem_GiB": 16.0,
        "build_time_s": 100.0,
    }
    assert compute_score(metrics, correct=True) == (100 * 2000 * 0.5) / 16
    assert compute_score(metrics, correct=False) == 0.0
