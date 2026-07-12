import pytest

from autoluce.bench.harness import _http_benchmark_prompts, apply_benchmark_overrides, server_context_capacity


def test_context_prompt_scales_with_requested_token_depth():
    spec = {"prompts": ["Inspect this code."], "prompt_token_reserve": 64}

    short = _http_benchmark_prompts(spec, context_depth=1024)[0]
    long = _http_benchmark_prompts(spec, context_depth=16384)[0]

    assert short.startswith("AutoLuce sample 0. Inspect this code.")
    assert short.count(" x") == 960
    assert long.count(" x") == 16320


def test_server_context_capacity_keeps_output_outside_the_prompt_cell():
    spec = {"context_headroom": 256}

    assert server_context_capacity(spec, context_depth=65536, max_tokens=1) == 65792


def test_context_headroom_must_cover_requested_output():
    with pytest.raises(ValueError, match="context_headroom"):
        server_context_capacity({"context_headroom": 8}, context_depth=1024, max_tokens=16)


def test_diagnostic_overrides_do_not_mutate_the_research_contract():
    original = {"contexts": [1024, 65536], "llama_bench_args": {"--repetitions": 3}}

    selected = apply_benchmark_overrides(original, contexts=[65536], repetitions=1)

    assert selected["contexts"] == [65536]
    assert selected["llama_bench_args"]["--repetitions"] == 1
    assert original == {"contexts": [1024, 65536], "llama_bench_args": {"--repetitions": 3}}
