"""
Shadow bench for autoggml v2: the benchmark workload is the user's OWN local
traffic, and the quality oracle is KL divergence on their own prompts.

Flow:
  1. `autoggml shadow proxy`  -- reverse proxy in front of llama-server; tees the
     prompt of every /completion | /chat/completions POST to a local JSONL log.
  2. `autoggml shadow build`  -- turns the recent log into benchmarks/shadow.json
     (smoke.json schema) + a kl_text file under the shadow dir.
  3. `autoggml kl-base shadow` then `AUTOGGML_BENCHMARKS=shadow autoggml run`.

Privacy: everything stays on-disk local. Prompts live under ~/.autoggml/shadow
(NOT under benchmarks/, so they are never committable); benchmarks/shadow.json
and the shadow dir are gitignored. Nothing leaves the box.

Pure logic (extract_prompt, select_shadow_prompts, build_shadow_benchmark) is
separated from the IO wrappers so every edge case is testable with synthetic data.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from autoggml import ROOT
BENCHMARKS_DIR = ROOT / "benchmarks"
DEFAULT_SHADOW_DIR = Path.home() / ".autoggml" / "shadow"
TEMPLATE_BENCHMARK = "smoke"


def shadow_dir() -> Path:
    """Prompt log + kl_text live here; overridable via AUTOGGML_SHADOW_DIR."""
    return Path(os.environ.get("AUTOGGML_SHADOW_DIR", DEFAULT_SHADOW_DIR))


def _now() -> datetime:
    return datetime.now()


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def extract_prompt(path: str, body_bytes: bytes) -> str | None:
    """Extract the user prompt from an OpenAI-style request body, or None.

    Chat: concatenated message contents; completion: the 'prompt' field.
    Never raises on malformed input — capture must not break the proxied request.
    """
    if "/completion" not in path and "/chat/completions" not in path:
        return None
    try:
        body = json.loads(body_bytes)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    messages = body.get("messages")
    if isinstance(messages, list):
        parts = [m["content"] for m in messages
                 if isinstance(m, dict) and isinstance(m.get("content"), str)]
        return "\n".join(parts) or None
    prompt = body.get("prompt")
    return prompt if isinstance(prompt, str) and prompt else None


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    return ts.replace(tzinfo=None)  # compare naive-local; log entries are local


def select_shadow_prompts(
    entries: list[dict], now: datetime, window_days: int = 1,
    max_prompts: int = 32, max_chars: int = 4000,
) -> list[str]:
    """Last window only, exact dedup, truncated, most-recent-first."""
    cutoff = now - timedelta(days=window_days)
    dated = []
    for e in entries:
        ts = _parse_ts(e.get("ts"))
        prompt = e.get("prompt")
        if ts is None or ts < cutoff or not isinstance(prompt, str) or not prompt:
            continue
        dated.append((ts, prompt[:max_chars]))
    dated.sort(key=lambda pair: pair[0], reverse=True)
    seen: set[str] = set()
    result: list[str] = []
    for _, prompt in dated:
        if prompt in seen:
            continue
        seen.add(prompt)
        result.append(prompt)
        if len(result) == max_prompts:
            break
    return result


def build_shadow_benchmark(prompts: list[str], template_bench: dict, kl_text: str | None = None) -> dict:
    """Clone the template benchmark, point it at the user's prompts, mark KL-only quality.

    Relative constraints (min_frac_of_baseline) are dropped: the shadow bench must
    run without a baseline_metrics entry of its own.
    """
    bench = copy.deepcopy(template_bench)
    bench["name"] = "shadow"
    bench["prompts"] = list(prompts)
    bench["kl_text"] = kl_text if kl_text is not None else str(shadow_dir() / "shadow.kl_text.txt")
    bench["quality"] = "kl"
    constraints = bench.get("objective", {}).get("constraints", {})
    for metric in list(constraints):
        constraints[metric] = {f: b for f, b in constraints[metric].items() if f != "min_frac_of_baseline"}
        if not constraints[metric]:
            del constraints[metric]
    return bench


# ---------------------------------------------------------------------------
# Prompt log
# ---------------------------------------------------------------------------


def append_prompt_log(prompt: str) -> None:
    directory = shadow_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "prompts.jsonl").open("a") as f:
        f.write(json.dumps({"ts": _now().isoformat(), "prompt": prompt}) + "\n")


def read_prompt_log() -> list[dict]:
    log = shadow_dir() / "prompts.jsonl"
    if not log.exists():
        return []
    entries = []
    for line in log.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            entries.append(e)
    return entries


# ---------------------------------------------------------------------------
# Capture proxy (stdlib only)
# ---------------------------------------------------------------------------

_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailers", "transfer-encoding", "upgrade"}


class _ProxyHandler(BaseHTTPRequestHandler):
    """Forward every request verbatim to upstream; stream the response back in
    chunks (flushing each, so SSE/chunked replies flow through). Prompt capture
    failure must never break the proxied request."""

    def _handle(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.send_error(411, "chunked request bodies not supported")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.send_error(400, "bad Content-Length")
            return
        body = self.rfile.read(length) if length else b""
        if self.command == "POST":
            try:
                prompt = extract_prompt(self.path, body)
                if prompt:
                    append_prompt_log(prompt)
            except Exception as e:  # never break the proxied request
                print(f"shadow proxy: prompt capture failed: {e}", file=sys.stderr)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in _HOP_BY_HOP | {"host", "content-length"}}
        req = urllib.request.Request(
            self.server.upstream + self.path, data=body or None,
            headers=headers, method=self.command,
        )
        try:
            # timeout is per blocking read, not total: long generations are fine
            # as long as tokens keep arriving.
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            resp = e  # non-2xx is still a response to forward verbatim
        except (urllib.error.URLError, OSError) as e:
            self.send_error(502, f"upstream unreachable: {e}")
            return
        with resp:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP_BY_HOP:
                    self.send_header(k, v)
            # upstream framing (chunked) is stripped, so the body is
            # close-delimited — must not be combined with keep-alive.
            self.close_connection = True
            self.end_headers()
            while True:
                # read1: return per upstream chunk instead of blocking until
                # 8KiB accumulate — SSE tokens must flow as they arrive.
                chunk = resp.read1(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _handle

    def log_message(self, *args):
        pass  # quiet: one local user, prompts already go to the JSONL


def make_proxy_server(port: int, upstream: str) -> ThreadingHTTPServer:
    # ponytail: stdlib thread-per-request proxy, fine for one local user;
    # real streaming perf work only if someone complains.
    server = ThreadingHTTPServer(("127.0.0.1", port), _ProxyHandler)
    server.upstream = upstream.rstrip("/")
    return server


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_proxy(port: int, upstream: str) -> None:
    server = make_proxy_server(port, upstream)
    print(f"shadow proxy: http://127.0.0.1:{server.server_port} -> {upstream}")
    print(f"prompt log:   {shadow_dir() / 'prompts.jsonl'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_build() -> None:
    prompts = select_shadow_prompts(read_prompt_log(), _now())
    if not prompts:
        print("shadow build: no prompts in the last window; run `autoggml shadow proxy` "
              "in front of your llama-server and send some traffic first", file=sys.stderr)
        sys.exit(2)
    template = json.loads((BENCHMARKS_DIR / f"{TEMPLATE_BENCHMARK}.json").read_text())
    kl_text = shadow_dir() / "shadow.kl_text.txt"
    bench = build_shadow_benchmark(prompts, template, kl_text=str(kl_text))
    kl_text.parent.mkdir(parents=True, exist_ok=True)
    kl_text.write_text("\n".join(prompts) + "\n")  # user prompts: shadow dir, never benchmarks/
    (BENCHMARKS_DIR / "shadow.json").write_text(json.dumps(bench, indent=2) + "\n")
    print(f"wrote {BENCHMARKS_DIR / 'shadow.json'} ({len(prompts)} prompts)")
    print(f"wrote {kl_text}")
    print("next:")
    print("  uv run autoggml kl-base shadow")
    print("  AUTOGGML_BENCHMARKS=shadow uv run autoggml run")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow bench: benchmark on your own local traffic (all on-disk local)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    proxy = sub.add_parser("proxy", help="capture proxy in front of your llama-server")
    proxy.add_argument("--port", type=int, default=8091)
    proxy.add_argument("--upstream", default="http://127.0.0.1:8080")
    sub.add_parser("build", help="build benchmarks/shadow.json from the prompt log")
    args = parser.parse_args()
    if args.cmd == "proxy":
        cmd_proxy(args.port, args.upstream)
    else:
        cmd_build()


if __name__ == "__main__":
    main()
