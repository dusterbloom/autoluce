from pathlib import Path

from autoluce.nvfp4 import (
    LLAMA_CPP_CONVERTER_REVISION,
    build_commands,
    conversion_command,
    is_mixed_nvfp4_fp8_config,
    native_source_dir,
)


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


def test_unsloth_mixed_nvfp4_fp8_config_is_recognized():
    config = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "weights": {"type": "float", "num_bits": 8, "strategy": "channel"},
            },
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "weights": {"type": "float", "num_bits": 4, "group_size": 16},
            },
        },
    }

    assert is_mixed_nvfp4_fp8_config(config)
    assert not is_mixed_nvfp4_fp8_config({**config, "quant_method": "modelopt"})


def test_nvfp4_conversion_uses_pinned_upstream_and_q8_fallback(tmp_path: Path):
    checkout = tmp_path / "llama.cpp"
    model = tmp_path / "model"
    output = tmp_path / "model.gguf"

    command = conversion_command(checkout, model, output)

    assert len(LLAMA_CPP_CONVERTER_REVISION) == 40
    assert command[:2] == [str(checkout / "convert_hf_to_gguf.py"), str(model)]
    assert command[command.index("--outfile") + 1] == str(output)
    assert "--fp8-as-q8" in command
    assert "--no-mtp" in command
    assert "--use-temp-file" in command
    assert command[command.index("--outtype") + 1] == "q8_0"
