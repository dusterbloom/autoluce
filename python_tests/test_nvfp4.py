from pathlib import Path

from autoluce.nvfp4 import build_commands, native_source_dir


def test_nvfp4_native_sources_are_owned_by_autoluce():
    source = native_source_dir()
    assert (source / "CMakeLists.txt").is_file()
    assert (source / "nvfp4.cuh").is_file()
    assert (source / "nvfp4.cu").is_file()
    assert (source / "test_nvfp4.cu").is_file()
    assert (source / "bench_nvfp4.cu").is_file()


def test_nvfp4_build_defaults_to_ampere_and_caps_parallelism(tmp_path: Path):
    configure, build = build_commands(tmp_path, architecture="86", jobs=99)

    assert configure[:3] == ["cmake", "-S", str(native_source_dir())]
    assert "-DCMAKE_CUDA_ARCHITECTURES=86" in configure
    assert any(flag.startswith("-DCMAKE_CUDA_COMPILER=") for flag in configure)
    assert build == ["cmake", "--build", str(tmp_path), "-j", "4"]
