from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import sys

from autoluce import ROOT


module_name = "bench_bonsai_prompt_benchmark"
_spec = importlib.util.spec_from_file_location(
    module_name,
    str(ROOT / "scripts" / "bench_bonsai_prompt_benchmark.py"),
)
_bench = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules[module_name] = _bench
_spec.loader.exec_module(_bench)


def test_parse_timings_supports_top_level_timings_and_usage_timings():
    top_level = {
        "choices": [{"message": {"content": "alpha"}}],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 13,
            "timings": {
                "prefill_ms": 50.0,
                "decode_ms": 120.0,
                "decode_tokens_per_sec": 108.0,
                "draft_n": 2,
                "draft_n_accepted": 1,
            },
        },
        "timings": {
            "predict": "ignored",
            "prefill_ms": 40.0,
            "decode_ms": 100.0,
            "decode_tokens_per_second": 110.0,
            "predicted_per_second": 220.0,
            "draft_n": 3,
            "draft_n_accepted": 2,
            "accept_rate": 0.9,
        },
    }

    parsed = _bench.parse_timings(top_level, mode="dspark-ddtree")

    assert parsed is not None
    assert parsed.mode == "dspark-ddtree"
    assert parsed.prefill_ms == 40.0
    assert parsed.decode_ms == 100.0
    assert parsed.decode_tokens_per_sec == 110.0
    assert parsed.predicted_per_second == 220.0
    assert parsed.draft_n == 3
    assert parsed.draft_n_accepted == 2
    assert parsed.accept_rate == 0.9


def test_parse_timings_falls_back_to_usage_timings():
    usage_timing_only = {
        "model": "m",
        "choices": [{"message": {"content": "beta"}}],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 11,
            "timings": {
                "prefill_ms": 60.0,
                "decode_ms": 130.0,
                "decode_tokens_per_sec": 130.0,
                "spec_draft_n": 4,
                "spec_draft_n_accepted": 3,
                "accept_rate": 0.75,
            },
        },
    }

    parsed = _bench.parse_timings(usage_timing_only)

    assert parsed is not None
    assert parsed.mode == "unknown"
    assert parsed.prefill_ms == 60.0
    assert parsed.decode_ms == 130.0
    assert parsed.decode_tokens_per_sec == 130.0
    assert parsed.draft_n == 4
    assert parsed.draft_n_accepted == 3
    assert parsed.accept_rate == 0.75
    assert parsed.prompt_tokens == 9
    assert parsed.completion_tokens == 11


def test_parse_timings_requires_any_timing_block():
    assert _bench.parse_timings({"usage": {"prompt_tokens": 1}}) is None


def test_sample_payload_preserves_metrics_text_and_raw_response():
    response = {
        "choices": [{"message": {"content": "Paris"}}],
        "usage": {
            "prompt_tokens": 18,
            "completion_tokens": 1,
            "timings": {
                "decode_tokens_per_sec": 120.5,
                "decode_ms": 8.3,
                "accept_rate": 0.75,
            },
        },
    }
    sample = _bench.parse_timings(response, mode="dspark-no-ddtree")

    assert sample is not None
    assert _bench._sample_payload(sample) == {
        "decode_tokens_per_sec": 120.5,
        "predicted_per_second": 120.5,
        "draft_n": None,
        "draft_n_accepted": None,
        "accept_rate": 0.75,
        "prefill_ms": None,
        "decode_ms": 8.3,
        "completion_tokens": 1,
        "prompt_tokens": 18,
        "text": "Paris",
        "response": response,
    }


def test_write_json_payload_creates_replayable_artifact(tmp_path: Path):
    destination = tmp_path / "nested" / "result.json"

    _bench._write_json_payload({"samples": [1, 2]}, destination)

    assert json.loads(destination.read_text()) == {"samples": [1, 2]}
    assert destination.read_text().endswith("\n")


def test_build_server_command_disables_caches_and_includes_mode_args(tmp_path: Path):
    cfg = _bench.RunConfig(
        host="127.0.0.1",
        port=18800,
        target=str(tmp_path / "target.gguf"),
        draft=str(tmp_path / "draft.gguf"),
        max_ctx=16384,
        max_tokens=64,
        seed=42,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetitions=1,
        warmup=1,
        mode="dspark-ddtree",
        use_ddtree=True,
        server_bin=str(tmp_path / "dflash_server"),
        server_args=("--target-device", "cuda:0"),
        ddtree_budget=24,
        model_name="bonsai",
        chunk=512,
    )

    command = _bench._build_server_command(cfg, include_draft=True)

    assert str(tmp_path / "dflash_server") == command[0]
    assert str(tmp_path / "target.gguf") == command[1]
    assert "--model-name" in command
    assert command[command.index("--model-name") + 1] == "bonsai"
    assert command[command.index("--chunk") + 1] == "512"
    assert command[command.index("--draft") + 1] == str(tmp_path / "draft.gguf")
    assert "--ddtree" in command
    assert command[command.index("--ddtree-budget") + 1] == "24"
    assert command[command.index("--prefill-cache-slots") + 1] == "0"
    assert command[command.index("--prefix-cache-slots") + 1] == "0"
    assert "--target-device" in command


def test_cli_ddtree_toggle_changes_dspark_mode(tmp_path: Path):
    args = _bench.parse_args(
        [
            "--server-bin",
            str(tmp_path / "dflash_server"),
            "--target",
            str(tmp_path / "target.gguf"),
            "--draft",
            str(tmp_path / "draft.gguf"),
            "--modes",
            "dspark-ddtree",
            "--ddtree",
            "off",
        ]
    )

    assert args.ddtree == "off"
