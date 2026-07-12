from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from autoluce.runtime.dflash_http import (
    DflashHttpClient,
    build_server_command,
    parse_completion,
    product_environment_overrides,
    server_environment,
    validate_prompt_depth,
)


def _response(text: str = "Paris") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 24,
            "total_tokens": 144,
            "accept_rate": 0.75,
            "timings": {
                "prefill_ms": 60.0,
                "decode_ms": 200.0,
                "decode_tokens_per_sec": 120.0,
            },
        },
    }


def test_parse_completion_uses_server_timings_as_authoritative_metrics():
    sample = parse_completion(_response())

    assert sample.text == "Paris"
    assert sample.prompt_tokens == 120
    assert sample.completion_tokens == 24
    assert sample.prefill_tok_s == pytest.approx(2000.0)
    assert sample.decode_tok_s == pytest.approx(120.0)
    assert sample.acceptance_rate == pytest.approx(0.75)
    assert sample.first_token_logits is None


def test_parse_completion_accepts_finite_opt_in_first_token_logits():
    body = _response()
    body["diagnostics"] = {
        "first_token_logits": {
            "dtype": "float32",
            "axis": "token_id",
            "values": [1.25, -0.5, 0.0],
        }
    }

    sample = parse_completion(body)

    assert sample.first_token_logits == pytest.approx((1.25, -0.5, 0.0))


@pytest.mark.parametrize("logits", [[], [1.0, float("nan")], "not-an-array"])
def test_parse_completion_rejects_invalid_opt_in_logits(logits):
    body = _response()
    body["diagnostics"] = {
        "first_token_logits": {
            "dtype": "float32",
            "axis": "token_id",
            "values": logits,
        }
    }
    with pytest.raises(ValueError, match="first_token_logits"):
        parse_completion(body)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda body: body.pop("usage"), "usage"),
        (lambda body: body["usage"].pop("timings"), "timings"),
        (lambda body: body["usage"]["timings"].pop("decode_tokens_per_sec"), "decode_tokens_per_sec"),
    ],
)
def test_parse_completion_fails_closed_when_measurements_are_missing(mutation, message):
    body = _response()
    mutation(body)
    with pytest.raises(ValueError, match=message):
        parse_completion(body)


class _FakeLuceboxHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self):  # noqa: N802 - stdlib callback name
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).requests.append(body)
        encoded = json.dumps(_response(text=body["messages"][0]["content"])).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        pass


@pytest.fixture
def fake_lucebox():
    _FakeLuceboxHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLuceboxHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", _FakeLuceboxHandler.requests
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_http_client_benchmarks_and_checks_frozen_quality(fake_lucebox):
    base_url, requests = fake_lucebox
    client = DflashHttpClient(base_url, timeout_s=2)

    metrics = client.benchmark(["one", "two"], repetitions=2, max_tokens=24)
    quality = client.compare_golden({
        "parameters": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "seed": 42, "n_predict": 24},
        "outputs": [{"prompt": "Paris", "text": "Paris"}],
    })

    assert metrics["decode_tok_s"] == pytest.approx(120.0)
    assert metrics["prefill_tok_s"] == pytest.approx(2000.0)
    assert metrics["prompt_tokens"] == pytest.approx(120.0)
    assert metrics["prompt_tokens_min"] == 120
    assert metrics["prompt_tokens_max"] == 120
    assert metrics["acceptance_rate"] == pytest.approx(0.75)
    assert metrics["measurement_source"] == "dflash_server.usage.timings"
    assert quality == (True, [{"prompt": "Paris", "passed": True, "expected": "Paris", "actual": "Paris"}])
    assert all(request["stream"] is False for request in requests)
    assert all(request["prefix_cache"] == {"scope": "off"} for request in requests)


def test_http_client_requests_logits_only_when_explicitly_enabled(fake_lucebox):
    base_url, requests = fake_lucebox
    client = DflashHttpClient(base_url, timeout_s=2)

    client.complete("quality", {"capture_first_token_logits": True, "n_predict": 1})
    client.complete("normal", {"n_predict": 1})

    assert requests[0]["diagnostics"] == {"first_token_logits": True}
    assert "diagnostics" not in requests[1]


def test_prompt_depth_gate_accepts_a_close_measurement():
    validate_prompt_depth(measured=995, requested=1024, tolerance=0.05)


@pytest.mark.parametrize("measured", [900, 1100])
def test_prompt_depth_gate_rejects_a_mislabeled_context_cell(measured):
    with pytest.raises(RuntimeError, match="context cell requested 1024"):
        validate_prompt_depth(measured=measured, requested=1024, tolerance=0.05)


def test_exact_quality_rejects_an_empty_reference_set(fake_lucebox):
    client = DflashHttpClient(fake_lucebox[0], timeout_s=2)
    with pytest.raises(ValueError, match="at least one reference"):
        client.compare_golden({"generated_at": "2026-07-10T00:00:00Z", "outputs": []})


def test_build_server_command_disables_caches_and_preserves_runtime_flags(tmp_path: Path):
    binary = tmp_path / "dflash_server"
    model = tmp_path / "model.gguf"
    draft = tmp_path / "draft.gguf"

    command = build_server_command(
        binary=binary,
        model=model,
        draft=draft,
        host="127.0.0.1",
        port=8123,
        max_context=32768,
        runtime_flags={"chunk": "512", "fast-rollback": True, "spark": False},
    )

    assert command[:2] == [str(binary), str(model)]
    assert command[command.index("--draft") + 1] == str(draft)
    assert command[command.index("--max-ctx") + 1] == "32768"
    assert command[command.index("--prefix-cache-slots") + 1] == "0"
    assert command[command.index("--prefill-cache-slots") + 1] == "0"
    assert command[command.index("--chunk") + 1] == "512"
    assert "--fast-rollback" in command
    assert "--spark" not in command


def test_server_environment_preserves_product_native_overrides():
    # server_environment consumes an already-resolved mapping (the output of
    # product_environment_overrides); this composes the pipeline the same way
    # the harness -> DflashServer chain does.
    base = {"PATH": "/usr/bin", "DFLASH_PREFILL_UBATCH": "512"}
    declared = {
        "DFLASH27B_PREFILL_UBATCH": 1024,
        "DFLASH_FORCE_MMQ": True,
        "GGML_CUDA_GRAPH_OPT": 1,
    }
    resolved = product_environment_overrides(declared, base=base)
    environment = server_environment(resolved, base=base)

    assert environment == {
        "PATH": "/usr/bin",
        "DFLASH_PREFILL_UBATCH": "512",
        "DFLASH27B_PREFILL_UBATCH": "1024",
        "DFLASH_FORCE_MMQ": "1",
        "GGML_CUDA_GRAPH_OPT": "1",
    }


def test_server_environment_maps_generic_prefill_controls_to_lucebox_names():
    base = {
        "DFLASH_PREFILL_UBATCH": "1024",
        "DFLASH_CHUNKED_Q_BATCH": "2048",
        "DFLASH_CHUNKED_CHUNK": "8192",
    }
    resolved = product_environment_overrides({}, base=base)
    environment = server_environment(resolved, base=base)

    assert environment["DFLASH27B_PREFILL_UBATCH"] == "1024"
    assert environment["DFLASH27B_CHUNKED_Q_BATCH"] == "2048"
    assert environment["DFLASH27B_CHUNKED_CHUNK"] == "8192"


def test_canonical_product_control_wins_over_generic_alias():
    base = {
        "DFLASH_PREFILL_UBATCH": "1024",
        "DFLASH27B_PREFILL_UBATCH": "3072",
    }
    resolved = product_environment_overrides({}, base=base)
    environment = server_environment(resolved, base=base)

    assert environment["DFLASH27B_PREFILL_UBATCH"] == "3072"


def test_product_environment_overrides_discovers_inherited_native_settings():
    overrides = product_environment_overrides(
        {},
        base={
            "PATH": "/usr/bin",
            "DFLASH_PREFILL_UBATCH": "1024",
            "DFLASH27B_KV_K": "q4_0",
            "GGML_CUDA_GRAPH_OPT": "1",
            "LUCE_MMQ_DP_MAX_NE1": "2048",
        },
    )

    assert overrides == {
        "DFLASH_PREFILL_UBATCH": "1024",
        "DFLASH27B_PREFILL_UBATCH": "1024",
        "DFLASH27B_KV_K": "q4_0",
        "GGML_CUDA_GRAPH_OPT": "1",
        "LUCE_MMQ_DP_MAX_NE1": "2048",
    }


def test_product_environment_overrides_applies_declared_values_and_unsets():
    overrides = product_environment_overrides(
        {
            "DFLASH_PREFILL_UBATCH": 2048,
            "DFLASH27B_KV_K": None,
            "GGML_CUDA_GRAPH_OPT": False,
        },
        base={
            "DFLASH_PREFILL_UBATCH": "512",
            "DFLASH27B_KV_K": "q4_0",
            "GGML_CUDA_GRAPH_OPT": "1",
        },
    )

    assert overrides == {
        "DFLASH_PREFILL_UBATCH": "2048",
        "DFLASH27B_PREFILL_UBATCH": "2048",
    }


def test_declared_generic_control_overrides_inherited_product_name():
    base = {"DFLASH27B_PREFILL_UBATCH": "3072"}
    resolved = product_environment_overrides({"DFLASH_PREFILL_UBATCH": 2048}, base=base)
    environment = server_environment(resolved, base=base)

    assert environment["DFLASH27B_PREFILL_UBATCH"] == "2048"


def test_declared_generic_unset_removes_inherited_product_name():
    base = {"DFLASH27B_PREFILL_UBATCH": "3072"}
    resolved = product_environment_overrides({"DFLASH_PREFILL_UBATCH": None}, base=base)
    environment = server_environment(resolved, base=base)

    assert "DFLASH_PREFILL_UBATCH" not in environment
    assert "DFLASH27B_PREFILL_UBATCH" not in environment


@pytest.mark.parametrize("name", ["PATH", "LD_PRELOAD", "not_dflash"])
def test_server_environment_rejects_non_dflash_overrides(name):
    with pytest.raises(ValueError, match="DFLASH"):
        server_environment({name: "value"}, base={})


def test_declared_unset_does_not_leak_back_via_server_environment():
    """Regression test: a variable the experiment declares unset must not
    reappear in the server environment just because the process it inherits
    from (or an unrelated `base`) still has it set. The resolved mapping from
    product_environment_overrides is the complete authority."""
    base_env = {"PATH": "/usr/bin"}
    resolved = product_environment_overrides({"DFLASH_FOO": None}, base={**base_env, "DFLASH_FOO": "1"})

    server_env = server_environment(resolved, base={**base_env, "DFLASH_FOO": "1"})

    assert "DFLASH_FOO" not in server_env


def test_server_environment_end_to_end_generic_alias_overrides_inherited_and_preserves_non_product_vars():
    """base env inherits DFLASH27B_PREFILL_UBATCH=512 (product spelling); the
    experiment declares the generic alias DFLASH_PREFILL_UBATCH=1024, which
    must win over the inherited product spelling. Non-product vars (PATH)
    from base must be preserved untouched."""
    base = {"PATH": "/usr/bin", "DFLASH27B_PREFILL_UBATCH": "512"}
    resolved = product_environment_overrides({"DFLASH_PREFILL_UBATCH": "1024"}, base=base)

    server_env = server_environment(resolved, base=base)

    assert server_env["DFLASH_PREFILL_UBATCH"] == "1024"
    assert server_env["DFLASH27B_PREFILL_UBATCH"] == "1024"
    assert server_env["PATH"] == "/usr/bin"
