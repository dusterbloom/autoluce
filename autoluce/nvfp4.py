"""Build and run the standalone NVFP4 CUDA oracle, operator test, and microbenchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from autoluce import ROOT


LLAMA_CPP_CONVERTER_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_CONVERTER_REVISION = "4f37f519722aa3242eecb7649466b4a4a2d6d6da"


def native_source_dir() -> Path:
    return ROOT / "native" / "nvfp4"


def default_build_dir() -> Path:
    return ROOT / "work" / "nvfp4-sm86-build"


def default_converter_dir() -> Path:
    return ROOT / "work" / "llama.cpp-converter"


def converter_patch() -> Path:
    return ROOT / "patches" / "llama.cpp" / "mixed-compressed-nvfp4-fp8.patch"


def is_mixed_nvfp4_fp8_config(config: dict[str, Any]) -> bool:
    """Return whether compressed-tensors mixes channel FP8 with packed NVFP4."""
    if config.get("quant_method") != "compressed-tensors" or config.get("format") != "mixed-precision":
        return False
    formats = {
        group.get("format")
        for group in config.get("config_groups", {}).values()
        if isinstance(group, dict)
    }
    return "nvfp4-pack-quantized" in formats and "float-quantized" in formats


def conversion_command(converter: Path, model: Path, output: Path) -> list[str]:
    """Build the reproducible low-memory conversion command for Ampere."""
    return [
        str(converter / "convert_hf_to_gguf.py"),
        str(model),
        "--outfile", str(output),
        "--outtype", "q8_0",
        "--fp8-as-q8",
        "--no-mtp",
        "--use-temp-file",
    ]


def prepare_converter(checkout: Path) -> None:
    """Checkout the tested upstream converter revision and apply our narrow compatibility patch."""
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", LLAMA_CPP_CONVERTER_REPOSITORY, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", LLAMA_CPP_CONVERTER_REVISION], check=True)
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != LLAMA_CPP_CONVERTER_REVISION:
        raise RuntimeError(f"converter checkout is {revision}; expected {LLAMA_CPP_CONVERTER_REVISION}")

    patch = converter_patch()
    forward = subprocess.run(
        ["git", "-C", str(checkout), "apply", "--check", str(patch)], capture_output=True, text=True,
    )
    if forward.returncode == 0:
        subprocess.run(["git", "-C", str(checkout), "apply", str(patch)], check=True)
        return
    reverse = subprocess.run(
        ["git", "-C", str(checkout), "apply", "--reverse", "--check", str(patch)], capture_output=True, text=True,
    )
    if reverse.returncode != 0:
        raise RuntimeError(f"converter compatibility patch does not apply:\n{forward.stderr.strip()}")


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
    parser.add_argument("action", choices=["build", "test", "bench", "convert"], nargs="?", default="test")
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("--build-dir", type=Path, default=default_build_dir())
    parser.add_argument("--converter-dir", type=Path, default=default_converter_dir())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--architecture", default=os.environ.get("AUTOLUCE_CUDA_ARCH", "86"))
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("AUTOLUCE_BUILD_JOBS", "4")))
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if args.action == "convert":
        if args.model is None or args.output is None:
            parser.error("convert requires MODEL and --output")
        prepare_converter(args.converter_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, *conversion_command(args.converter_dir, args.model, args.output)], check=True)
        return

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
