"""
Tests for shadow.py: shadow bench built from the user's own local traffic.

All pure logic (prompt extraction, selection, benchmark construction) is tested
with synthetic data — no network, no GPU. The proxy gets one in-process smoke
test against a stdlib dummy upstream.
"""

import json
import threading
import urllib.request
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import shadow
from shadow import build_shadow_benchmark, extract_prompt, select_shadow_prompts

NOW = datetime(2026, 7, 2, 12, 0, 0)


def entry(prompt, hours_ago=0.0):
    return {"ts": (NOW - timedelta(hours=hours_ago)).isoformat(), "prompt": prompt}


# --- select_shadow_prompts ----------------------------------------------------


def test_select_filters_outside_window():
    entries = [entry("old", hours_ago=48), entry("fresh", hours_ago=1)]
    assert select_shadow_prompts(entries, NOW) == ["fresh"]


def test_select_dedups_exact():
    entries = [entry("same"), entry("same"), entry("other")]
    result = select_shadow_prompts(entries, NOW)
    assert sorted(result) == ["other", "same"]
    assert len(result) == 2


def test_select_truncates_to_max_chars():
    entries = [entry("x" * 100)]
    assert select_shadow_prompts(entries, NOW, max_chars=10) == ["x" * 10]


def test_select_most_recent_first_and_caps_count():
    entries = [entry(f"p{i}", hours_ago=i) for i in range(10)]  # p0 newest
    result = select_shadow_prompts(entries, NOW, max_prompts=3)
    assert result == ["p0", "p1", "p2"]


def test_select_skips_malformed_entries():
    entries = [{"prompt": "no ts"}, {"ts": "not-a-date", "prompt": "bad"},
               {"ts": NOW.isoformat()}, entry(""), entry("good")]
    assert select_shadow_prompts(entries, NOW) == ["good"]


def test_select_handles_aware_timestamps_against_naive_now():
    entries = [{"ts": "2026-07-02T11:00:00+00:00", "prompt": "aware"}]
    assert select_shadow_prompts(entries, NOW) == ["aware"]


# --- build_shadow_benchmark ---------------------------------------------------


TEMPLATE = {
    "name": "smoke",
    "manifest_entry": "smoke",
    "spec_type": None,
    "n_draft": 0,
    "prompts": ["template prompt"],
    "llama_bench_args": {"-p": 128, "-n": 32, "--repetitions": 3},
    "expected": {"min_decode_tok_s": 10.0},
    "objective": {
        "maximize": "decode_tok_s",
        "constraints": {
            "peak_mem_GiB": {"max": 8.0},
            "prefill_tok_s": {"min_frac_of_baseline": 0.95},
        },
    },
}


def test_build_shadow_benchmark_schema():
    bench = build_shadow_benchmark(["a", "b"], TEMPLATE, kl_text="/tmp/x/shadow.kl_text.txt")
    assert bench["name"] == "shadow"
    assert bench["prompts"] == ["a", "b"]
    assert bench["kl_text"] == "/tmp/x/shadow.kl_text.txt"
    assert bench["quality"] == "kl"
    assert bench["manifest_entry"] == "smoke"  # model resolution follows the template
    assert bench["llama_bench_args"] == TEMPLATE["llama_bench_args"]


def test_build_shadow_benchmark_drops_relative_constraints():
    bench = build_shadow_benchmark(["a"], TEMPLATE, kl_text="k.txt")
    constraints = bench["objective"]["constraints"]
    assert constraints == {"peak_mem_GiB": {"max": 8.0}}  # no baseline needed to run


def test_build_shadow_benchmark_does_not_mutate_template():
    before = json.dumps(TEMPLATE, sort_keys=True)
    build_shadow_benchmark(["a"], TEMPLATE, kl_text="k.txt")
    assert json.dumps(TEMPLATE, sort_keys=True) == before


def test_build_shadow_benchmark_default_kl_text_under_shadow_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOGGML_SHADOW_DIR", str(tmp_path))
    bench = build_shadow_benchmark(["a"], TEMPLATE)
    assert bench["kl_text"] == str(tmp_path / "shadow.kl_text.txt")


# --- extract_prompt -----------------------------------------------------------


def test_extract_prompt_chat_concatenates_messages():
    body = json.dumps({"messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]}).encode()
    assert extract_prompt("/v1/chat/completions", body) == "sys\nhello"


def test_extract_prompt_completion_field():
    body = json.dumps({"prompt": "complete me"}).encode()
    assert extract_prompt("/completion", body) == "complete me"


def test_extract_prompt_non_matching_path_returns_none():
    body = json.dumps({"prompt": "x"}).encode()
    assert extract_prompt("/v1/models", body) is None


@pytest.mark.parametrize("body", [b"not json", b"[1,2]", b"{}",
                                  json.dumps({"messages": "nope"}).encode(),
                                  json.dumps({"prompt": 42}).encode()])
def test_extract_prompt_malformed_returns_none(body):
    assert extract_prompt("/v1/chat/completions", body) is None


# --- proxy smoke test ---------------------------------------------------------


class _Upstream(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        reply = b"echo:" + body
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


def test_proxy_forwards_and_tees_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOGGML_SHADOW_DIR", str(tmp_path))
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    proxy = shadow.make_proxy_server(0, f"http://127.0.0.1:{upstream.server_port}")
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        payload = json.dumps({"messages": [{"role": "user", "content": "hi there"}]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
            data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.read() == b"echo:" + payload
        lines = (tmp_path / "prompts.jsonl").read_text().splitlines()
        assert len(lines) == 1
        logged = json.loads(lines[0])
        assert logged["prompt"] == "hi there"
        assert "ts" in logged
    finally:
        proxy.shutdown()
        upstream.shutdown()


# --- build IO -----------------------------------------------------------------


def test_cmd_build_writes_benchmark_and_kl_text(monkeypatch, tmp_path):
    shadow_dir = tmp_path / "shadow"
    bench_dir = tmp_path / "benchmarks"
    bench_dir.mkdir()
    (bench_dir / "smoke.json").write_text(json.dumps(TEMPLATE))
    monkeypatch.setenv("AUTOGGML_SHADOW_DIR", str(shadow_dir))
    monkeypatch.setattr(shadow, "BENCHMARKS_DIR", bench_dir)
    shadow_dir.mkdir(parents=True)
    log = shadow_dir / "prompts.jsonl"
    log.write_text(json.dumps(entry("my prompt")) + "\n")
    monkeypatch.setattr(shadow, "_now", lambda: NOW)

    shadow.cmd_build()

    bench = json.loads((bench_dir / "shadow.json").read_text())
    assert bench["name"] == "shadow"
    assert bench["prompts"] == ["my prompt"]
    kl_text = shadow_dir / "shadow.kl_text.txt"
    assert bench["kl_text"] == str(kl_text)
    assert kl_text.read_text() == "my prompt\n"


def test_cmd_build_no_prompts_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOGGML_SHADOW_DIR", str(tmp_path))
    monkeypatch.setattr(shadow, "BENCHMARKS_DIR", tmp_path / "benchmarks")
    with pytest.raises(SystemExit):
        shadow.cmd_build()


# --- harness integration: golden optional when quality == "kl" ----------------


def test_harness_skips_golden_when_quality_kl(monkeypatch, tmp_path):
    import harness
    spec = dict(TEMPLATE, name="shadow", quality="kl", kl_text=str(tmp_path / "k.txt"))

    def fail(*a, **k):
        raise AssertionError("check_correctness must not be called")

    monkeypatch.setattr(harness, "check_correctness", fail)
    monkeypatch.setattr(harness, "load_golden", lambda name: None)
    correct, details = harness.resolve_correctness("shadow", spec, None, None, 0, {})
    assert correct is True
    assert details == []


def test_harness_still_requires_golden_without_quality_kl(monkeypatch):
    import harness
    monkeypatch.setattr(harness, "load_golden", lambda name: None)
    with pytest.raises(RuntimeError, match="golden"):
        harness.resolve_correctness("smoke", dict(TEMPLATE), None, None, 0, {})
