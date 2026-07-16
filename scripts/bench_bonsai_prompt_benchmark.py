#!/usr/bin/env python3
"""Run quicksort speculative-decoding benchmark in AR and DSpark modes.

This script mirrors the public Prism `SPECULATIVE.md` prompt benchmark shape:

* Prompt: "Implement quicksort in Python."
* max_tokens default: 400
* JSON fields exposed:
  * predicted/t decode throughput
  * draft width and accepted draft width

It supports both response shapes we see in the wild:

* top-level `timings` (llama-server style)
* `usage.timings` (`dflash_server` style)

It is intentionally minimal: one request payload path, one warm-up request,
and one or more timed repetitions per mode.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
import shlex
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

DEFAULT_PROMPT = "Implement quicksort in Python."


@dataclass(frozen=True)
class RunConfig:
    host: str
    port: int
    target: str
    draft: str | None
    max_ctx: int
    max_tokens: int
    seed: int
    temperature: float
    top_k: int
    top_p: float
    repetitions: int
    warmup: int
    mode: str
    use_ddtree: bool
    server_bin: str
    server_args: tuple[str, ...]
    ddtree_budget: int = 22
    model_name: str = "autoluce-benchmark"
    chunk: int = 512


@dataclass(frozen=True)
class BenchResult:
    mode: str
    predicted_per_second: float | None
    decode_tokens_per_sec: float | None
    prefill_ms: float | None
    decode_ms: float | None
    prompt_tokens: float | None
    completion_tokens: float | None
    draft_n: float | None
    draft_n_accepted: float | None
    accept_rate: float | None
    text: str
    response_raw: Mapping[str, Any]


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)) and (not math.isfinite(float(value))):
            continue
        return value
    return None


def parse_timings(response: Mapping[str, Any], mode: str = "unknown") -> BenchResult | None:
    """Parse speculative and throughput metrics from both llama-server and dflash forms."""

    usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else {}
    timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
    usage_timings = usage.get("timings") if isinstance(usage.get("timings"), Mapping) else {}
    raw = timings or usage_timings

    if not raw:
        return None

    decode_tokens_per_sec = _coalesce(
        raw.get("decode_tokens_per_sec"),
        raw.get("decode_tokens_per_second"),
        usage_timings.get("decode_tokens_per_sec"),
    )
    predicted_per_second = _coalesce(
        raw.get("predicted_per_second"),
        decode_tokens_per_sec,
    )

    return BenchResult(
        mode=mode,
        predicted_per_second=_coalesce(predicted_per_second, None),
        decode_tokens_per_sec=_coalesce(decode_tokens_per_sec, None),
        prefill_ms=_coalesce(raw.get("prefill_ms"), None),
        decode_ms=_coalesce(raw.get("decode_ms"), None),
        prompt_tokens=_coalesce(usage.get("prompt_tokens"), response.get("prompt_tokens"), None),
        completion_tokens=_coalesce(usage.get("completion_tokens"), response.get("completion_tokens"), None),
        draft_n=_coalesce(raw.get("draft_n"), raw.get("spec_draft_n"), usage.get("draft_n"), usage.get("spec_draft_n"), None),
        draft_n_accepted=_coalesce(raw.get("draft_n_accepted"), raw.get("spec_draft_n_accepted"), usage.get("draft_n_accepted"), usage.get("spec_draft_n_accepted"), None),
        accept_rate=_coalesce(
            raw.get("accept_rate"),
            usage.get("accept_rate"),
            response.get("accept_rate"),
            None,
        ),
        text=response.get("choices", [{}])[0].get("message", {}).get("content", "") if response.get("choices") else "",
        response_raw=response,
    )


def _build_request_body(prompt: str, cfg: RunConfig) -> dict[str, Any]:
    return {
        "model": cfg.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "seed": cfg.seed,
        "stream": False,
    }


def _send_completion(prompt: str, cfg: RunConfig, timeout_s: float) -> Mapping[str, Any]:
    url = f"http://{cfg.host}:{cfg.port}/v1/chat/completions"
    response = requests.post(url, json=_build_request_body(prompt, cfg), timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def _mean_and_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if not values:
        return None, None
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, stddev


def _mean_or_none(values: Iterable[float]) -> float | None:
    mean, _ = _mean_and_std(values)
    return mean


def _std_or_none(values: Iterable[float]) -> float | None:
    _, std = _mean_and_std(values)
    return std


def _sample_payload(sample: BenchResult) -> dict[str, Any]:
    """Return one measured response without losing evidence needed for replay."""

    return {
        "decode_tokens_per_sec": sample.decode_tokens_per_sec,
        "predicted_per_second": sample.predicted_per_second,
        "draft_n": sample.draft_n,
        "draft_n_accepted": sample.draft_n_accepted,
        "accept_rate": sample.accept_rate,
        "prefill_ms": sample.prefill_ms,
        "decode_ms": sample.decode_ms,
        "completion_tokens": sample.completion_tokens,
        "prompt_tokens": sample.prompt_tokens,
        "text": sample.text,
        "response": sample.response_raw,
    }


def _write_json_payload(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")


class _ServerRunner(AbstractContextManager["_ServerRunner"]):
    def __init__(self, cmd: list[str], host: str, port: int, timeout_s: float, log_path: Path | None = None):
        self.cmd = cmd
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.log_path = log_path
        self.proc: subprocess.Popen[str] | None = None
        self.log_handle: subprocess._FILE | int | None = None

    def _wait_for_healthy(self) -> None:
        deadline = time.time() + self.timeout_s
        base = f"http://{self.host}:{self.port}"
        health_endpoints = ("/health", "/v1/health")
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"server exited early; log={self.log_path}")
            for endpoint in health_endpoints:
                try:
                    resp = requests.get(base + endpoint, timeout=1.0)
                except requests.RequestException:
                    time.sleep(0.5)
                    continue
                if resp.ok:
                    return
            time.sleep(0.5)
        raise TimeoutError(f"server did not become healthy within {self.timeout_s}s on port {self.port}")

    def __enter__(self) -> "_ServerRunner":
        if self.log_path is not None:
            self.log_handle = self.log_path.open("w")
        else:
            self.log_handle = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_for_healthy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc = None
        if self.log_handle not in (None, subprocess.DEVNULL):
            self.log_handle.close()


def _build_server_command(cfg: RunConfig, include_draft: bool) -> list[str]:
    cmd = [cfg.server_bin, cfg.target, "--host", cfg.host, "--port", str(cfg.port), "--max-ctx", str(cfg.max_ctx)]
    cmd.extend(["--model-name", cfg.model_name])
    cmd.extend(["--chunk", str(cfg.chunk)])
    if include_draft and cfg.draft:
        cmd.extend(["--draft", cfg.draft])
    if include_draft and cfg.use_ddtree:
        cmd.extend(["--ddtree", "--ddtree-budget", str(cfg.ddtree_budget)])
    # keep a stable memory profile for apples-to-apples
    cmd.extend(["--prefill-cache-slots", "0", "--prefix-cache-slots", "0"])
    cmd.extend(cfg.server_args)
    return cmd


def run_mode(prompt: str, cfg: RunConfig, include_draft: bool) -> tuple[BenchResult, list[BenchResult]]:
    server_cmd = _build_server_command(cfg, include_draft)
    mode = cfg.mode
    log_path = Path(f"/tmp/lucebox-bonsai-{mode}.log")
    with _ServerRunner(server_cmd, cfg.host, cfg.port, timeout_s=180.0, log_path=log_path):
        # Warm-up is excluded from timing
        for _ in range(cfg.warmup):
            _send_completion(prompt, cfg, timeout_s=180.0)

        samples: list[BenchResult] = []
        for _ in range(cfg.repetitions):
            payload = _send_completion(prompt, cfg, timeout_s=180.0)
            parsed = parse_timings(payload, cfg.mode)
            if parsed is None:
                raise ValueError(f"missing timings in response for mode {mode}")
            samples.append(parsed)

    agg = BenchResult(
        mode=mode,
        predicted_per_second=_mean_or_none(
            sample.predicted_per_second for sample in samples if isinstance(sample.predicted_per_second, (int, float))
        ),
        decode_tokens_per_sec=_mean_or_none(
            sample.decode_tokens_per_sec for sample in samples if isinstance(sample.decode_tokens_per_sec, (int, float))
        ),
        prefill_ms=_mean_or_none(
            sample.prefill_ms for sample in samples if isinstance(sample.prefill_ms, (int, float))
        ),
        decode_ms=_mean_or_none(
            sample.decode_ms for sample in samples if isinstance(sample.decode_ms, (int, float))
        ),
        prompt_tokens=_mean_or_none(
            sample.prompt_tokens for sample in samples if isinstance(sample.prompt_tokens, (int, float))
        ),
        completion_tokens=_mean_or_none(
            sample.completion_tokens for sample in samples if isinstance(sample.completion_tokens, (int, float))
        ),
        draft_n=_mean_or_none(
            sample.draft_n for sample in samples if isinstance(sample.draft_n, (int, float))
        ),
        draft_n_accepted=_mean_or_none(
            sample.draft_n_accepted for sample in samples if isinstance(sample.draft_n_accepted, (int, float))
        ),
        accept_rate=_mean_or_none(
            sample.accept_rate for sample in samples if isinstance(sample.accept_rate, (int, float))
        ),
        text=samples[-1].text if samples else "",
        response_raw=samples[-1].response_raw if samples else {},
    )
    return agg, samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark exact quicksort prompt in AR vs DSpark")
    parser.add_argument("--server-bin", required=True, help="Path to dflash_server binary")
    parser.add_argument("--target", required=True, help="Target GGUF path")
    parser.add_argument("--draft", default="", help="Draft GGUF path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=18690)
    parser.add_argument("--max-ctx", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--modes",
        default="ar,dspark-ddtree",
        help="Comma-separated modes (ar, dspark-ddtree, dspark-no-ddtree)",
    )
    parser.add_argument(
        "--server-args",
        default="",
        help="Extra server args passed to each launch (shell-style quoting not supported)",
    )
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--model-name", default="autoluce-benchmark")
    parser.add_argument("--ddtree-budget", type=int, default=22)
    parser.add_argument("--ddtree", default="on", choices=["on", "off"])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    parser.add_argument(
        "--json-output",
        default="",
        help="Write machine-readable JSON to this path (also enables JSON mode)",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Include each measured API response in --json output for reproducible evidence",
    )
    return parser.parse_args(argv)


def _parse_modes(modes_text: str) -> list[str]:
    return [mode.strip() for mode in modes_text.split(",") if mode.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modes = _parse_modes(args.modes)
    if not modes:
        raise SystemExit("--modes must contain at least one valid mode")

    base_args = tuple(shlex.split(args.server_args))
    results = []

    for index, mode in enumerate(modes):
        port = args.base_port + index
        if mode == "ar":
            include_draft = False
            use_ddtree = False
        elif mode == "dspark-ddtree":
            include_draft = bool(args.draft)
            use_ddtree = args.ddtree == "on"
        elif mode == "dspark-no-ddtree":
            include_draft = bool(args.draft)
            use_ddtree = False
        else:
            raise SystemExit(f"unsupported mode: {mode}")

        if include_draft and not args.draft:
            raise SystemExit(f"mode {mode} requires --draft")
        cfg = RunConfig(
            host=args.host,
            port=port,
            target=args.target,
            draft=args.draft or None,
            max_ctx=args.max_ctx,
            max_tokens=args.max_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetitions=args.repetitions,
            warmup=args.warmup,
            mode=mode,
            use_ddtree=use_ddtree,
            server_bin=args.server_bin,
            server_args=base_args,
            ddtree_budget=args.ddtree_budget,
            model_name=args.model_name,
            chunk=args.chunk,
        )
        result, samples = run_mode(args.prompt, cfg, include_draft)
        results.append((mode, result, samples))

    # Keep output compact and stable.
    ar_row = next((item for item in results if item[0].startswith("ar")), None)
    ar_speed = ar_row[1].decode_tokens_per_sec if ar_row else None

    if args.json or args.json_output:
        payload = {
            "prompt": args.prompt,
            "params": {
                "max_tokens": args.max_tokens,
                "max_ctx": args.max_ctx,
                "seed": args.seed,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetitions": args.repetitions,
                "warmup": args.warmup,
            },
            "modes": [],
        }
        for mode, result, samples in results:
            speedup = None
            if ar_speed and result.decode_tokens_per_sec and not mode.startswith("ar"):
                speedup = result.decode_tokens_per_sec / ar_speed
            mode_payload = {
                "mode": mode,
                "decode_tokens_per_sec": result.decode_tokens_per_sec,
                "decode_tokens_per_sec_std": _std_or_none(
                    sample.decode_tokens_per_sec for sample in samples if isinstance(sample.decode_tokens_per_sec, (int, float))
                ),
                "predicted_per_second": result.predicted_per_second,
                "draft_n": result.draft_n,
                "draft_n_accepted": result.draft_n_accepted,
                "accept_rate": result.accept_rate,
                "prefill_ms": result.prefill_ms,
                "decode_ms": result.decode_ms,
                "completion_tokens": result.completion_tokens,
                "prompt_tokens": result.prompt_tokens,
                "speedup_vs_ar": speedup,
            }
            if args.include_samples:
                mode_payload["samples"] = [_sample_payload(sample) for sample in samples]
            payload["modes"].append(mode_payload)
        if args.json_output:
            _write_json_payload(payload, Path(args.json_output))
        if args.json:
            print(json.dumps(payload, indent=2))
        return 0

    print("mode\tdecode_tok_s\tpredict\tdraft_n\taccept\tprompt_toks\tgen_toks\tspeedup_vs_ar")
    for mode, result, _samples in results:
        speedup = ""
        if ar_speed and result.decode_tokens_per_sec and not mode.startswith("ar"):
            speedup = f"{result.decode_tokens_per_sec / ar_speed:.2f}x"
        print(
            "\t".join(
                [
                    mode,
                    f"{result.decode_tokens_per_sec:.2f}" if result.decode_tokens_per_sec is not None else "N/A",
                    f"{result.predicted_per_second:.2f}" if result.predicted_per_second is not None else "N/A",
                    str(int(result.draft_n)) if isinstance(result.draft_n, (int, float)) else "N/A",
                    f"{result.accept_rate:.3f}" if isinstance(result.accept_rate, (int, float)) else "N/A",
                    f"{result.prompt_tokens:.0f}" if isinstance(result.prompt_tokens, (int, float)) else "N/A",
                    f"{result.completion_tokens:.0f}" if isinstance(result.completion_tokens, (int, float)) else "N/A",
                    speedup,
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
