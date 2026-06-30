"""
Tests for the OpenAI-compatible LLM client.

The HTTP transport is injected, so the pure logic (env->config, request building,
response parsing, disabled-when-unconfigured) is tested with no network. One client
covers cloud OpenAI and any local backend (llama-server / Ollama / vLLM / LM Studio)
because they all speak /v1/chat/completions -- the only difference is OPENAI_BASE_URL.
"""

import pytest

from llm import LLMConfig, build_request, complete, config_from_env, parse_completion


# --- config resolution ---------------------------------------------------------

def test_config_disabled_when_base_url_unset():
    assert config_from_env({}) is None
    # A key alone must not enable LLM calls -- base_url is the explicit opt-in.
    assert config_from_env({"OPENAI_API_KEY": "sk-x"}) is None


def test_config_enabled_with_base_url_uses_local_defaults():
    cfg = config_from_env({"OPENAI_BASE_URL": "http://localhost:8080/v1"})
    assert cfg.base_url == "http://localhost:8080/v1"
    assert cfg.api_key  # some default (local backends ignore it)
    assert cfg.model    # some default model


def test_config_reads_all_overrides():
    cfg = config_from_env({
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_KEY": "sk-cloud",
        "AUTOGGML_MODEL": "gpt-4o",
    })
    assert cfg.base_url == "https://api.openai.com/v1"
    assert cfg.api_key == "sk-cloud"
    assert cfg.model == "gpt-4o"


# --- request building ----------------------------------------------------------

def test_build_request_appends_path_and_bearer_header():
    cfg = LLMConfig(base_url="https://api.openai.com/v1", api_key="sk-x", model="m")
    url, headers, body = build_request(cfg, [{"role": "user", "content": "hi"}])
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-x"
    assert body["model"] == "m"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_build_request_normalizes_trailing_slash():
    cfg = LLMConfig(base_url="http://localhost:8080/v1/", api_key="k", model="m")
    url, _, _ = build_request(cfg, [])
    assert url == "http://localhost:8080/v1/chat/completions"


# --- response parsing ----------------------------------------------------------

def test_parse_completion_extracts_content():
    payload = {"choices": [{"message": {"content": "do the thing"}}]}
    assert parse_completion(payload) == "do the thing"


# --- complete: wiring (build -> injected post -> parse) ------------------------

class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_complete_raises_when_disabled(monkeypatch):
    # No OPENAI_BASE_URL -> the harness must NOT fire a network call; it refuses clearly.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        complete([{"role": "user", "content": "hi"}])


def test_complete_happy_path_uses_injected_post():
    cfg = LLMConfig(base_url="https://x/v1", api_key="k", model="m")
    captured = {}

    def fake_post(url, headers=None, json=None):
        captured["url"] = url
        captured["body"] = json
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    out = complete([{"role": "user", "content": "hi"}], config=cfg, post=fake_post)
    assert out == "ok"
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == "m"
