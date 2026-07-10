"""Build and run the standalone NVFP4 CUDA oracle, operator test, and microbenchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from autoluce import ROOT


def native_source_dir() -> Path:
    return ROOT / "native" / "nvfp4"


def default_build_dir() -> Path:
    return ROOT / "work" / "nvfp4-sm86-build"


def cuda_compiler() -> Path:
    override = os.environ.get("CUDACXX")
    preferred = Path(override) if override else Path("/usr/local/cuda/bin/nvcc")
    if preferred.is_file():
        return preferred
    discovered = shutil.which("nvcc")
    if discovered:
        return Path(discovered)
    raise RuntimeError("nvcc is required for the NVFP4 CUDA operator")


def cuda_host_compiler() -> Path | None:
    override = os.environ.get("CUDAHOSTCXX")
    if override:
        return Path(override)
    for candidate in ("/usr/bin/g++-12", "/usr/bin/g++-11"):
        if Path(candidate).is_file():
            return Path(candidate)
    return None


def build_commands(build_dir: Path, architecture: str = "86", jobs: int = 4) -> tuple[list[str], list[str]]:
    jobs = min(4, max(1, jobs))
    configure = [
        "cmake", "-S", str(native_source_dir()), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={architecture}",
        f"-DCMAKE_CUDA_COMPILER={cuda_compiler()}",
    ]
    host_compiler = cuda_host_compiler()
    if host_compiler is not None:
        configure.append(f"-DCMAKE_CUDA_HOST_COMPILER={host_compiler}")
    build = ["cmake", "--build", str(build_dir), "-j", str(jobs)]
    return configure, build


def build(build_dir: Path, architecture: str, jobs: int) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    configure, compile_command = build_commands(build_dir, architecture, jobs)
    subprocess.run(configure, check=True)
    subprocess.run(compile_command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and benchmark the Ampere NVFP4 W4A16 CUDA operator")
    parser.add_argument("action", choices=["build", "test", "bench"], nargs="?", default="test")
    parser.add_argument("--build-dir", type=Path, default=default_build_dir())
    parser.add_argument("--architecture", default=os.environ.get("AUTOLUCE_CUDA_ARCH", "86"))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("AUTOLUCE_BUILD_JOBS", "4")))
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    build(args.build_dir, args.architecture, args.jobs)
    if args.action == "build":
        print(json.dumps({"build_dir": str(args.build_dir), "architecture": args.architecture}))
        return
    binary = args.build_dir / ("test_nvfp4" if args.action == "test" else "bench_nvfp4")
    command = [str(binary)]
    if args.action == "bench":
        command += ["--rows", str(args.rows), "--cols", str(args.cols), "--iterations", str(args.iterations)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
