import subprocess

import pytest


def _runner(output: str, returncode: int = 0):
    def run(args, **kwargs):
        return subprocess.CompletedProcess(args, returncode, output, "")

    return run


def test_closure_hashes_executable_and_resolved_dependency_bytes(tmp_path):
    from autoluce.runtime.artifacts import capture_runtime_artifact_closure

    binary = tmp_path / "dflash_server"
    library = tmp_path / "libggml-cuda.so.0.9.11"
    binary.write_bytes(b"server")
    library.write_bytes(b"cuda-a")
    output = f"linux-vdso.so.1 (0x1)\nlibggml-cuda.so.0 => {library} (0x2)\n"

    closure = capture_runtime_artifact_closure(
        binary, env={"LD_LIBRARY_PATH": str(tmp_path)}, runner=_runner(output),
    )

    assert closure.executable.path == str(binary.resolve())
    assert len(closure.executable.sha256) == 64
    assert [(item.name, item.path) for item in closure.dependencies] == [
        ("libggml-cuda.so.0", str(library.resolve())),
    ]


def test_dependency_mutation_changes_closure_without_executable_change(tmp_path):
    from autoluce.runtime.artifacts import (
        capture_runtime_artifact_closure,
        require_stable_runtime_artifacts,
    )

    binary = tmp_path / "dflash_server"
    library = tmp_path / "libggml-cuda.so.0"
    binary.write_bytes(b"server")
    library.write_bytes(b"cuda-a")
    runner = _runner(f"libggml-cuda.so.0 => {library} (0x2)\n")
    built = capture_runtime_artifact_closure(binary, env={}, runner=runner)
    library.write_bytes(b"cuda-b")
    measured = capture_runtime_artifact_closure(binary, env={}, runner=runner)

    assert built.executable == measured.executable
    with pytest.raises(RuntimeError, match="libggml-cuda"):
        require_stable_runtime_artifacts(built, measured)


def test_runtime_closure_fails_when_a_dependency_is_missing(tmp_path):
    from autoluce.runtime.artifacts import capture_runtime_artifact_closure

    binary = tmp_path / "dflash_server"
    binary.write_bytes(b"server")
    with pytest.raises(RuntimeError, match="missing runtime dependencies"):
        capture_runtime_artifact_closure(
            binary,
            env={},
            runner=_runner("libggml-cuda.so.0 => not found\n"),
        )


def test_runtime_closure_is_deterministically_sorted(tmp_path):
    from autoluce.runtime.artifacts import capture_runtime_artifact_closure

    binary = tmp_path / "dflash_server"
    first = tmp_path / "liba.so"
    second = tmp_path / "libz.so"
    for path in (binary, first, second):
        path.write_bytes(path.name.encode())
    output = f"libz.so => {second} (0x2)\nliba.so => {first} (0x3)\n"

    closure = capture_runtime_artifact_closure(binary, env={}, runner=_runner(output))

    assert [item.name for item in closure.dependencies] == ["liba.so", "libz.so"]
