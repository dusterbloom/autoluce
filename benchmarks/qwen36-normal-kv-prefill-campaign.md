# Qwen3.6 normal-KV prefill campaign

Status: baseline frontier archived through 64K; the first automatic-policy
Q4_K_M/llama.cpp comparison is archived separately for 1K, 8K, and 16K.

This campaign holds the Qwen3.6-27B Q4_K_M weight artifact fixed and varies only the
KV-cache representation. `F16/F16`, `Q8_0/Q8_0`, and `Q4_0/Q4_0` are independent
campaigns. TQ3 is deliberately outside this campaign until the normal paths are
correct and fast.

The lanes never rely on a product default. The Lucebox README describes Q8_0 as the
default while the checked-out resolver currently initializes Q4_0, so every run sets
`DFLASH27B_KV_K` and `DFLASH27B_KV_V` explicitly. AutoLuce now records the resolved
pair in source evidence; an implicit pair is recorded as `unknown:runtime-default`.

## Evidence scope

| Lane | Frontier contexts | Additional fit probe | Initial purpose |
|---|---:|---:|---|
| F16/F16 | 1K, 8K, 16K, 64K | none | unquantized correctness and speed control |
| Q8_0/Q8_0 | 1K, 8K, 16K, 64K | 128K | normal product-quality cache |
| Q4_0/Q4_0 | 1K, 8K, 16K, 64K | 128K | lower-memory normal cache |

The 128K cells are capacity probes, not frontier evidence. They become eligible only
after a separate campaign contract explicitly adds that workload. Evidence is never
compared across KV lanes: the structured quantization identity contains both the fixed
weight quantization and the K/V pair. An upstream llama.cpp result may be compared only
within the same lane and workload, with `runtime` as the sole allowed variation.

Earlier IQ4_XS F16 parity and MMQ results are useful hypothesis evidence, but they are
not numerically comparable with this Q4_K_M campaign.

The 2026-07-14 automatic grouped-GDN result is recorded in
[`q4km-prefill-upstream-20260714`](q4km-prefill-upstream-20260714/README.md). It is a
compatible F16/F16 performance comparison against llama.cpp, but remains outside the
quality-constrained frontier because the cross-runtime response capture did not prove
exact generated-output equality.

## GPU execution order

When the GPU is free, use one common exact-quality oracle frozen under F16 and separate
state/result directories for each lane:

```bash
export GGML_CUDA=ON
export AUTOLUCE_BUILD_JOBS=4
export AUTOLUCE_BUILD_SUBDIR=build-cuda-sm86
export AUTOLUCE_BENCHMARKS=qwen36-27b-prefill
export AUTOLUCE_GOLDEN_DIR="$PWD/work/state/normal-kv-prefill/golden"

export DFLASH27B_KV_K=f16
export DFLASH27B_KV_V=f16
uv run autoluce freeze --benchmark qwen36-27b-prefill --overwrite
```

For each lane, set its pair and a lane-specific archive before taking the clean
baseline. This example is Q8_0; substitute `f16` or `q4_0` for the other lanes.

```bash
export DFLASH27B_KV_K=q8_0
export DFLASH27B_KV_V=q8_0
export AUTOLUCE_STATE_DIR="$PWD/work/state/normal-kv-prefill/q8_0-q8_0"
export AUTOLUCE_RESULT_BUNDLE="$PWD/results/runs/normal-kv-prefill/q8_0-q8_0"

uv run autoluce baseline \
  --contexts 1024,8192,16384,65536 \
  --repetitions 3 --json
```

Run the Q8_0 and Q4_0 128K capacity probes separately so they cannot enter the baseline
frontier accidentally:

```bash
export AUTOLUCE_STATE_DIR="$PWD/work/state/normal-kv-prefill/q8_0-q8_0-fit-probe"
export AUTOLUCE_RESULT_BUNDLE="$PWD/results/runs/normal-kv-prefill/q8_0-q8_0-fit-probe"
uv run autoluce baseline --contexts 131072 --repetitions 1 --json
```

After all three clean baselines pass exact correctness, capture one profiled 8K run and
one profiled 64K run per lane. Keep clean/candidate ordering interleaved and retain the
individual ordered samples; do not combine a code change with a chunk-size or dispatch
override.

## Hypothesis ladder

1. `explicit-baseline`: establish the three uncontaminated lane curves.
2. `attention-dispatch`: confirm the kernel selected at 8K and 64K. Static inspection
   shows batched Q8_0/Q4_0 on SM86 reaches the MMA-F16 path, which allocates full F16
   K/V temporaries and dequantizes the cache before attention. Measure its share before
   changing it.
3. `gdn-chunking`: parallelize the serial-token GDN prefill path while preserving its
   exact final recurrent state.
4. `causal-mask`: derive causality in-kernel instead of materializing/uploading a full
   mask, if the profile confirms that cost.
5. `quantized-attention`: consume Q8_0/Q4_0 directly in batched attention and remove the
   full-cache F16 temporary.
6. `depth-schedule`: tune one context-dependent ubatch/chunk policy after kernel paths
   are correct.
7. `graph-cleanup`: remove only graph outputs or synchronizations proven unnecessary by
   the target-only trace.

Advance one hypothesis only when exact correctness passes and the improvement exceeds
the measured noise. Promotion evidence requires at least three ordered samples,
`prefill_tok_s` dispersion, peak memory, exact-quality evidence, machine/model/product/
binary identities, the resolved K/V pair, and a measured prompt depth within 5% of the
requested cell. Reject a candidate that regresses any eligible context by more than the
benchmark's 2% floor, changes the recurrent result, or depends on an unrecorded runtime
default.

The executable campaign contracts are in
[`examples/normal-kv-prefill`](../examples/normal-kv-prefill); replace their illustrative
machine, model, and environment fingerprints with observed values before recording
evidence.
