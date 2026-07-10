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
    assert metrics["acceptance_rate"] == pytest.approx(0.75)
    assert metrics["measurement_source"] == "dflash_server.usage.timings"
    assert quality == (True, [{"prompt": "Paris", "passed": True, "expected": "Paris", "actual": "Paris"}])
    assert all(request["stream"] is False for request in requests)
    assert all(request["prefix_cache"] == {"scope": "off"} for request in requests)


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
