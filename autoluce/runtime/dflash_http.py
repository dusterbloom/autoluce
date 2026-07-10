"""Benchmark and exact-quality adapter for Lucebox's dflash_server HTTP API."""

from __future__ import annotations

import os
import socket
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from autoluce import ROOT
from autoluce.bench.profiling import profile_command
from autoluce.bench.telemetry import TelemetryCollector
from autoluce.source_layout import SourceLayout


MEASUREMENT_SOURCE = "dflash_server.usage.timings"


@dataclass(frozen=True)
class CompletionSample:
    text: str
    prompt_tokens: int
    completion_tokens: int
    prefill_tok_s: float
    decode_tok_s: float
    prefill_ms: float
    decode_ms: float
    acceptance_rate: float | None


def _required(mapping: dict[str, Any], key: str, parent: str) -> Any:
    if key not in mapping:
        raise ValueError(f"dflash response is missing {parent}.{key}")
    return mapping[key]


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content or "")


def parse_completion(body: dict[str, Any]) -> CompletionSample:
    """Parse one non-streaming completion and reject unmeasured responses."""
    usage = _required(body, "usage", "response")
    if not isinstance(usage, dict):
        raise ValueError("dflash response usage must be an object")
    timings = _required(usage, "timings", "usage")
    if not isinstance(timings, dict):
        raise ValueError("dflash response usage.timings must be an object")
    choices = _required(body, "choices", "response")
    if not isinstance(choices, list) or not choices:
        raise ValueError("dflash response choices must be a non-empty array")
    message = choices[0].get("message", {})

    prompt_tokens = int(_required(usage, "prompt_tokens", "usage"))
    completion_tokens = int(_required(usage, "completion_tokens", "usage"))
    prefill_ms = float(_required(timings, "prefill_ms", "usage.timings"))
    decode_ms = float(_required(timings, "decode_ms", "usage.timings"))
    decode_tok_s = float(_required(timings, "decode_tokens_per_sec", "usage.timings"))
    if prefill_ms <= 0 or decode_ms < 0 or prompt_tokens <= 0 or completion_tokens < 0:
        raise ValueError("dflash response contains invalid timing or token measurements")
    acceptance = usage.get("accept_rate")
    return CompletionSample(
        text=_message_text(message),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prefill_tok_s=prompt_tokens / (prefill_ms / 1000.0),
        decode_tok_s=decode_tok_s,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        acceptance_rate=float(acceptance) if acceptance is not None else None,
    )


class DflashHttpClient:
    def __init__(self, base_url: str, timeout_s: float = 3600.0, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def healthy(self) -> bool:
        try:
            return self.session.get(f"{self.base_url}/health", timeout=2).ok
        except requests.RequestException:
            return False

    def complete(self, prompt: str, parameters: dict[str, Any] | None = None) -> CompletionSample:
        params = parameters or {}
        body = {
            "model": params.get("model", "autoluce-benchmark"),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": int(params.get("n_predict", params.get("max_tokens", 64))),
            "temperature": float(params.get("temperature", 0.0)),
            "top_p": float(params.get("top_p", 1.0)),
            "top_k": int(params.get("top_k", 1)),
            "seed": int(params.get("seed", 42)),
            "prefix_cache": {"scope": "off"},
        }
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return parse_completion(response.json())

    def benchmark(self, prompts: list[str], repetitions: int, max_tokens: int) -> dict[str, float | str]:
        if not prompts:
            raise ValueError("benchmark requires at least one prompt")
        if repetitions < 1:
            raise ValueError("benchmark repetitions must be positive")
        # Exclude one-time graph/allocation work from the measured repetitions.
        self.complete(prompts[0], {"n_predict": max_tokens})
        samples = [
            self.complete(prompts[index % len(prompts)], {"n_predict": max_tokens})
            for index in range(repetitions)
        ]

        def mean(name: str) -> float:
            return statistics.fmean(float(getattr(sample, name)) for sample in samples)

        def deviation(name: str) -> float:
            values = [float(getattr(sample, name)) for sample in samples]
            return statistics.stdev(values) if len(values) > 1 else 0.0

        metrics: dict[str, float | str] = {
            "decode_tok_s": mean("decode_tok_s"),
            "decode_tok_s_stddev": deviation("decode_tok_s"),
            "prefill_tok_s": mean("prefill_tok_s"),
            "prefill_tok_s_stddev": deviation("prefill_tok_s"),
            "measurement_source": MEASUREMENT_SOURCE,
        }
        rates = [sample.acceptance_rate for sample in samples if sample.acceptance_rate is not None]
        if rates:
            metrics["acceptance_rate"] = statistics.fmean(rates)
        return metrics

    def compare_golden(self, golden: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
        if golden.get("generated_at") == "NOT_YET_GENERATED":
            raise RuntimeError("golden outputs are placeholders; run `autoluce freeze`")
        parameters = dict(golden.get("parameters", {}))
        outputs = golden.get("outputs", [])
        if not outputs:
            raise ValueError("golden outputs must contain at least one reference")
        results = []
        passed_all = True
        for expected in outputs:
            actual = self.complete(str(expected["prompt"]), parameters).text
            passed = actual == expected["text"]
            passed_all = passed_all and passed
            results.append({
                "prompt": expected["prompt"],
                "passed": passed,
                "expected": expected["text"],
                "actual": actual,
            })
        return passed_all, results


def build_server_command(
    binary: Path,
    model: Path,
    draft: Path | None,
    host: str,
    port: int,
    max_context: int,
    runtime_flags: dict[str, Any] | None = None,
) -> list[str]:
    command = [
        str(binary), str(model),
        "--host", host,
        "--port", str(port),
        "--max-ctx", str(max_context),
        "--prefix-cache-slots", "0",
        "--prefill-cache-slots", "0",
    ]
    if draft is not None:
        command += ["--draft", str(draft)]
    for key, value in (runtime_flags or {}).items():
        flag = f"--{key}"
        if value is False or value is None:
            continue
        if value is True:
            if flag not in command:
                command.append(flag)
            continue
        if flag in command:
            index = command.index(flag)
            command[index + 1] = str(value)
        else:
            command += [flag, str(value)]
    return command


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class _ProcessMemoryMonitor:
    def __init__(self, process: subprocess.Popen, interval_s: float = 0.1) -> None:
        self.process = process
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        status = Path(f"/proc/{self.process.pid}/status")
        while not self._stop.wait(self.interval_s):
            try:
                for line in status.read_text().splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        self.peak_bytes = max(self.peak_bytes, int(line.split()[1]) * 1024)
            except (OSError, ValueError):
                if self.process.poll() is not None:
                    return

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak_bytes / (1024 ** 3)


class DflashServer:
    def __init__(
        self,
        command: list[str],
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout_s: float = 900.0,
        log_dir: Path | None = None,
    ) -> None:
        self.host = host
        self.port = port or _free_port(host)
        self.command = list(command)
        self.startup_timeout_s = startup_timeout_s
        self.log_dir = log_dir or ROOT / "work" / "runtime-logs"
        self.process: subprocess.Popen | None = None
        self.monitor: _ProcessMemoryMonitor | None = None
        self._log_handle = None
        self.peak_rss_gib = 0.0

    def __enter__(self) -> DflashHttpClient:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (self.log_dir / f"dflash-{self.port}.log").open("w")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            self.monitor = _ProcessMemoryMonitor(self.process)
            self.monitor.start()
            client = DflashHttpClient(f"http://{self.host}:{self.port}")
            deadline = time.monotonic() + self.startup_timeout_s
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"dflash_server exited during startup; see {self._log_handle.name}")
                if client.healthy():
                    return client
                time.sleep(0.25)
            raise TimeoutError(f"dflash_server did not become healthy; see {self._log_handle.name}")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.monitor is not None:
            self.peak_rss_gib = self.monitor.stop()
        if self._log_handle is not None:
            self._log_handle.close()


class DflashHttpRuntime:
    """Product runtime factory used by the benchmark harness and golden freezer."""

    def __init__(
        self,
        layout: SourceLayout,
        backend: str,
        model: Path,
        draft: Path | None,
        runtime_flags: dict[str, Any] | None = None,
        profile_path: str | None = None,
    ) -> None:
        if layout.runtime != "dflash-server-http":
            raise ValueError(f"unsupported Lucebox runtime: {layout.runtime}")
        self.layout = layout
        self.backend = backend
        self.model = model
        self.draft = draft if draft is not None and draft.exists() else None
        self.runtime_flags = runtime_flags or {}
        self.profile_path = profile_path

    def session(self, max_context: int) -> _RuntimeSession:
        return _RuntimeSession(self, max_context)


class _RuntimeSession:
    def __init__(self, runtime: DflashHttpRuntime, max_context: int) -> None:
        self.runtime = runtime
        self.max_context = max_context
        self.server: DflashServer | None = None
        self.telemetry = TelemetryCollector()
        self.final_metrics: dict[str, float] = {}

    def __enter__(self) -> tuple[DflashHttpClient, DflashServer, TelemetryCollector]:
        port = _free_port()
        binary = self.runtime.layout.binary("dflash_server", self.runtime.backend)
        if not binary.exists():
            raise RuntimeError(f"{binary} not found; run `autoluce setup`")
        command = build_server_command(
            binary, self.runtime.model, self.runtime.draft, "127.0.0.1", port,
            self.max_context, self.runtime.runtime_flags,
        )
        if self.runtime.profile_path:
            command = profile_command(command, self.runtime.backend, self.runtime.profile_path)
        self.server = DflashServer(command, port=port)
        client = self.server.__enter__()
        self.telemetry.start()
        return client, self.server, self.telemetry

    def __exit__(self, kind, value, traceback) -> None:
        telemetry = self.telemetry.stop()
        if self.server is not None:
            self.server.__exit__(kind, value, traceback)
            memory_values = [self.server.peak_rss_gib]
            memory_values += [telemetry[key] for key in ("peak_vram_used_GiB", "peak_gtt_used_GiB") if key in telemetry]
            telemetry["peak_mem_GiB"] = max(memory_values)
        self.final_metrics = telemetry
