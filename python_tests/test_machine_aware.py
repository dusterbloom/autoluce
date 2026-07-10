import json
import os
import subprocess
import sys
import pytest

from autoggml.bench.harness import aggregate_context_metrics, check_context_regressions, parse_llama_bench_json
from autoggml.bench.telemetry import summarize
from autoggml.contracts import ResearchContract
from autoggml.models import ModelEntry, profile_model, resolve_entry_files
from autoggml.onboard import INSTALL
from autoggml.profiles import MachineProfile
from autoggml.remote import RemoteBusyError, SSHWorker
from autoggml.targets import TargetConfig
from autoggml.test_drive import _lease, safe_test_drive


def test_target_config_loads_named_ssh_target(monkeypatch, tmp_path):
    config = tmp_path / "targets.toml"
    config.write_text(
        "[targets.strix-halo]\n"
        'transport = "ssh"\n'
        'host = "user@example"\n'
        'root = "/home/user/autoggml"\n'
        'model_root = "/opt/models"\n'
        "build_jobs = 4\n"
    )
    monkeypatch.delenv("AUTOGGML_TARGET_HOST", raising=False)
    target = TargetConfig.load("strix-halo", config)
    assert target.host == "user@example"
    assert target.model_root == "/opt/models"
    assert target.build_jobs == 4


def test_target_config_rejects_unsafe_build_parallelism(monkeypatch, tmp_path):
    config = tmp_path / "targets.toml"
    config.write_text('[targets.x]\ntransport="ssh"\nhost="u@h"\nroot="/tmp/x"\nbuild_jobs=8\n')
    with pytest.raises(ValueError, match="build_jobs"):
        TargetConfig.load("x", config)


def test_machine_fingerprint_ignores_volatile_state_and_models():
    a = MachineProfile({"hostname": "x", "gpu_arch": "gfx1151", "mem_available_bytes": 1, "swap_free_bytes": 1, "models": [{"mtime": 1}]})
    b = MachineProfile({"hostname": "x", "gpu_arch": "gfx1151", "mem_available_bytes": 2, "swap_free_bytes": 2, "models": [{"mtime": 2}]})
    assert a.fingerprint == b.fingerprint


def test_contract_round_trip_and_validation(tmp_path):
    path = tmp_path / "contract.yaml"
    contract = ResearchContract("strix", "machine", "model", "digest")
    contract.write(path)
    assert ResearchContract.read(path) == contract
    contract.build_jobs = 8
    with pytest.raises(ValueError, match="build_jobs"):
        contract.validate()


def test_external_sharded_model_resolution_and_profile(monkeypatch, tmp_path):
    first = tmp_path / "model-00001-of-00002.gguf"
    second = tmp_path / "model-00002-of-00002.gguf"
    first.write_bytes(b"abc")
    second.write_bytes(b"defg")
    entry = ModelEntry("m", "q", [first.name, second.name], expected_size_bytes=7, path_env="MODEL_OVERRIDE")
    monkeypatch.setenv("MODEL_OVERRIDE", str(first))
    paths = resolve_entry_files(entry)
    assert paths == [first, second]
    profile = profile_model(entry, paths, "sha")
    assert profile.size_bytes == 7
    assert profile.files[0].endswith("00001-of-00002.gguf")


def test_parse_llama_bench_json_depth_rows():
    output = json.dumps([
        {"n_prompt": 512, "n_gen": 0, "n_depth": 8192, "avg_ts": 100.0, "stddev_ts": 2.0},
        {"n_prompt": 0, "n_gen": 128, "n_depth": 8192, "avg_ts": 12.5, "stddev_ts": 0.4},
    ])
    assert parse_llama_bench_json(output) == {
        "prefill_tok_s": 100.0,
        "prefill_tok_s_stddev": 2.0,
        "decode_tok_s": 12.5,
        "decode_tok_s_stddev": 0.4,
    }


def test_context_aggregation_uses_primary_geometric_mean_and_max_memory():
    cells = [
        {"context_depth": 8192, "decode_tok_s": 16.0, "decode_tok_s_stddev": 1.0, "prefill_tok_s": 100.0,
         "prefill_tok_s_stddev": 2.0, "peak_mem_GiB": 90.0},
        {"context_depth": 32768, "decode_tok_s": 9.0, "decode_tok_s_stddev": 1.0, "prefill_tok_s": 64.0,
         "prefill_tok_s_stddev": 2.0, "peak_mem_GiB": 95.0},
        {"context_depth": 131072, "decode_tok_s": 4.0, "decode_tok_s_stddev": 1.0, "prefill_tok_s": 25.0,
         "prefill_tok_s_stddev": 2.0, "peak_mem_GiB": 100.0},
    ]
    result = aggregate_context_metrics(cells)
    assert result["decode_tok_s"] == pytest.approx(12.0)
    assert result["prefill_tok_s"] == pytest.approx(80.0)
    assert result["peak_mem_GiB"] == 100.0


def test_context_regression_reports_matching_depth():
    cells = [{"context_depth": 8192, "decode_tok_s": 8.0, "prefill_tok_s": 100.0}]
    baseline = {"context_metrics": [{"context_depth": 8192, "decode_tok_s": 10.0, "prefill_tok_s": 100.0}]}
    violations = check_context_regressions(cells, baseline, 0.95)
    assert len(violations) == 1
    assert "8192 decode_tok_s" in violations[0]


def test_telemetry_summary_reports_headroom_swap_and_faults():
    samples = [
        {"mem_available_bytes": 20 * 1024**3, "swap_used_bytes": 0, "major_faults": 2},
        {"mem_available_bytes": 12 * 1024**3, "swap_used_bytes": 1024**3, "major_faults": 5},
    ]
    result = summarize(samples)
    assert result["min_mem_available_GiB"] == 12.0
    assert result["swap_growth_GiB"] == 1.0
    assert result["major_faults_delta"] == 3.0


class _FakeRunner:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


def _ssh_target():
    return TargetConfig("strix", "ssh", "user@host", "/home/user/autoggml", "/opt/models")


def test_ssh_worker_quotes_remote_command_as_one_argument():
    runner = _FakeRunner(stdout="ok")
    worker = SSHWorker(_ssh_target(), runner=runner)
    result = worker.run(["printf", "%s", "value with spaces"])
    invocation = runner.calls[0][0]
    assert invocation[-1] == "bash -lc 'printf %s '\"'\"'value with spaces'\"'\"''"
    assert result.stdout == "ok"


def test_ssh_worker_maps_nonblocking_lease_failure_to_busy():
    runner = _FakeRunner(returncode=1, stderr="busy")
    worker = SSHWorker(_ssh_target(), runner=runner)
    with pytest.raises(RemoteBusyError, match="busy"):
        worker.run(["true"], lease=True)


def test_onboard_installs_launcher_and_preserves_other_targets(tmp_path):
    home = tmp_path / "home"
    config = home / ".config" / "autoggml" / "targets.toml"
    config.parent.mkdir(parents=True)
    config.write_text('[targets.other]\ntransport = "local"\n')
    env = os.environ.copy()
    env["HOME"] = str(home)
    subprocess.run(
        [sys.executable, "-", "/srv/autoggml", "/opt/models", "lucebox3", "/tmp/gpu.lock", "4"],
        input=INSTALL, text=True, capture_output=True, check=True, env=env,
    )
    text = config.read_text()
    assert "[targets.other]" in text
    assert "[targets.lucebox3]" in text
    launcher = home / ".local" / "bin" / "autoggml"
    assert launcher.stat().st_mode & 0o111
    assert "AUTOGGML_DEFAULT_TARGET=lucebox3" in launcher.read_text()
    assert '.local/bin:$PATH' in (home / ".profile").read_text()


def test_test_drive_safe_mode_reports_busy_without_live_work(monkeypatch):
    profile = MachineProfile({
        "hostname": "lucebox3",
        "gpu_arch": "gfx1151",
        "mem_available_bytes": 1024**3,
        "busy_reasons": ["available_memory_below_12_gib"],
        "models": [{"path": "/opt/models/model.gguf", "size_bytes": 80 * 1024**3, "readable": True}],
    })
    monkeypatch.setattr("autoggml.test_drive.build_profile", lambda target: profile)
    monkeypatch.setattr("autoggml.bench.harness.run_harness", lambda **kwargs: {"correctness": "pass"})
    result = safe_test_drive(TargetConfig("local", "local"))
    assert result["status"] == "busy"
    assert result["simulated_loop"] == "pass"
    assert "retry" in result["next"]


def test_local_test_drive_lease_is_nonblocking(tmp_path):
    path = str(tmp_path / "gpu.lock")
    with _lease(path):
        with pytest.raises(RuntimeError, match="holds the accelerator lease"):
            with _lease(path):
                pass
