# NVFP4 CUDA Operator

This directory is the device oracle and kernel laboratory for bringing Unsloth's
`nvfp4-pack-quantized` tensors into Lucebox on pre-Blackwell CUDA GPUs.

The current ABI consumes:

- Two E2M1 weights per byte, low nibble first.
- One E4M3FN decode scale per 16 consecutive weights.
- One FP32 global decode scale per tensor.
- FP16 activations and FP32 accumulators/output.
- Row-major weights with the inner dimension divisible by 16.

The dequantization identity is:

```text
weight = e2m1_value * e4m3_block_scale * fp32_global_scale
```

`test_nvfp4` checks all 16 E2M1 codes, known E4M3 boundary values, invalid shapes,
and a deterministic CUDA GEMV against an independent CPU calculation.
`bench_nvfp4` compares the fused packed kernel with the same naive FP16 GEMV.

The RTX 3090 has no native FP4 tensor-core path. This kernel therefore performs
software unpack/decode fused with W4A16 GEMV; it is not a claim of native NVFP4
execution. The Lucebox loader, safetensors metadata mapping, padded scale layouts,
and graph dispatch remain separate product work.
