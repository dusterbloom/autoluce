# Bonsai-27B Q1 quicksort AR vs DSpark — 2026-07-15

Same-session matched run on one RTX 3090 (WSL2, CUDA 12.6).

- Engine (lucebox): worktree `lucebox-bonsai-dspark` HEAD `b19b95e`
  (`fix(bonsai): preserve qwen35 speculative target semantics`), built
  build-cuda-sm86, CUDA arch 86. NOTE: native DSpark on this HEAD is a
  four-proposal chain; `--ddtree` is NOT supported (superseded the 3ade7bc
  DDTree-budget-22 frontier path).
- Prism: `/tmp/prism-b9591-bin/.../llama-server` (b9591-62061f9).
- Target: Bonsai-27B-Q1_0.gguf, sha256 17ef84..aa0 (verified).
- Draft: Bonsai-27B-dspark-Q4_1.gguf, sha256 25e73f..b1b (verified).
- Workload: prompt "Implement quicksort in Python.", n_predict=400, ctx=16384,
  temp=0, top_k=1, top_p=1, seed=42, caching disabled both arms (lucebox
  prefix/prefill-cache-slots=0 -> prefix_len=0; prism cache_prompt=false).
- 1 warmup + 5 measured reps per arm.

## Results (decode tok/s)

| Arm | tok/s | note |
|---|---:|---|
| lucebox AR | 65.8 (std 1.5) | |
| Prism AR | 72.81 median / 72.85 mean (std 0.28) | |
| lucebox DSpark (4-chain) | 118.12 (std 2.6) | 75.5% accept |

- lucebox DSpark vs own AR: **1.80x**.
- lucebox AR vs Prism AR: **0.904x (-9.6%)** — AR parity NOT met on this workload.
- lucebox DSpark vs Prism AR: 1.622x (spec-vs-nonspec; Prism has no dspark head).

## Caveats
- 118 exceeds last night's DDTree-22 headline (108.5) and the committed frontier
  N=1 diagnostic (98.9, different generic prompts).
- Yesterday's Prism baseline JSONs (/tmp/prism-final-rep*.json) used n_predict=256,
  prompt_n=37 — a DIFFERENT workload; not parity-comparable to this run.
- DSpark exactness (output == AR output) NOT yet asserted for quicksort; frozen
  golden currently covers only the 3 generic campaign prompts.

## (a) AR gap root cause — profiler pass (nsys, cuda-graph-trace=node)

Decode runs under CUDA graphs (plain nsys undercounts kernels; must use
`--cuda-graph-trace=node`). With graph nodes captured:

- GPU is **~89% busy during decode** -> lucebox AR is **GPU-compute-bound, NOT
  host/launch-bound**. The `tok_embd` Q1_0 CPU-only placement is a red herring:
  per-token embedding H2D copies (20 KB each) total ~1.6 ms.
- Decode GPU time breakdown: **`mul_mat_vec_q` (Q1_0 GEMV, ggml_type 41) ~70%**,
  `quantize_q8_1` ~4.5%, qwen35 GDN kernels (`k_turbo_wht`+`gated_delta_net`+
  `l2_norm`) ~7%, rms_norm/concat/rope ~14%.
- Conclusion: the 9.6% AR deficit vs Prism lives in the Q1_0 `mul_mat_vec_q`
  kernel (both engines read identical weights). Next: `ncu` the GEMV for achieved
  vs peak DRAM bandwidth; fix is a lucebox CUDA-kernel campaign gated by the
  harness significance test.

## (c) DSpark vs AR exactness gate — FAILS (documented, not a logic bug)

Same quicksort prompt, temp0/seed42, AR vs DSpark outputs diffed:

- **RESULT: DIVERGED.** 858/1332 chars identical (64% prefix): the quicksort
  function is byte-identical; they split in the `__main__` example at a close-logit
  tie (`unsorted` vs `sample_list`). Both outputs are valid, correct code.
- Cause: width-5 DSpark verify uses different CUDA reduction shapes than width-1
  AR, so the target argmax differs at near-ties (as the frontier doc predicted).
  DSpark is deterministic/self-consistent but NOT bit-identical to AR.
- Implication: a strict `DSpark==AR` gate cannot pass on long prompts. The
  shippable gate is a frozen DSpark golden (self-consistency), matching the
  existing `benchmarks/golden/` pattern.

Evidence: exact-ar-output.txt, exact-dspark-output.txt, exact-gate-result.txt.
