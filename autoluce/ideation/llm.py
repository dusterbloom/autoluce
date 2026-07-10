"""
OpenAI-compatible LLM client for the harness (cloud or local, one client).

Anything speaking the /v1/chat/completions dialect works: OpenAI cloud, llama.cpp /
lucebox `llama-server`, Ollama, vLLM, LM Studio. The backend is chosen purely by env
(OPENAI_BASE_URL + OPENAI_API_KEY + AUTOLUCE_MODEL), so 'local vs cloud' is a config
switch, not a code change.

Disabled by default: OPENAI_BASE_URL is the explicit opt-in, so the harness never silently
fires network calls. The HTTP transport is injected into complete(), keeping the pure
config/build/parse logic testable offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

DEFAULT_MODEL = "gpt-4o-mini"   # cloud-friendly default; override via AUTOLUCE_MODEL for local
_DUMMY_KEY = "not-needed"       # local backends ignore the bearer token


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str


def config_from_env(env: dict[str, str] | None = None) -> LLMConfig | None:
    """Resolve LLM config from env, or return None when disabled.

    OPENAI_BASE_URL is the single opt-in: absent -> None (LLM features refuse rather than
    fire). When present, key defaults to a dummy (local backends ignore it) and model to a
    cloud-friendly default; set OPENAI_API_KEY / AUTOLUCE_MODEL to override.
    """
    env = env if env is not None else os.environ
    base_url = env.get("OPENAI_BASE_URL")
    if not base_url:
        return None
    return LLMConfig(
        base_url=base_url,
        api_key=env.get("OPENAI_API_KEY", _DUMMY_KEY),
        model=env.get("AUTOLUCE_MODEL", DEFAULT_MODEL),
    )


def build_request(
    config: LLMConfig, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024,
) -> tuple[str, dict[str, str], dict]:
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    body = {"model": config.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    return url, headers, body


def parse_completion(payload: dict) -> str:
    return payload["choices"][0]["message"]["content"]


def complete(
    messages: list[dict],
    config: LLMConfig | None = None,
    post=requests.post,
    **params,
) -> str:
    """Call the configured chat-completions endpoint and return the message content.

    `config` defaults to config_from_env(); `post` is the transport seam (injectable so
    the build->post->parse wiring is testable without a network). Raises RuntimeError if
    the LLM is disabled (no OPENAI_BASE_URL).
    """
    if config is None:
        config = config_from_env()
    if config is None:
        raise RuntimeError(
            "LLM disabled: set OPENAI_BASE_URL to enable "
            "(OPENAI_API_KEY for cloud, AUTOLUCE_MODEL to override the model)."
        )
    url, headers, body = build_request(config, messages, **params)
    resp = post(url, headers=headers, json=body)
    resp.raise_for_status()
    return parse_completion(resp.json())
