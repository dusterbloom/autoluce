# RTX 3090 Qwen3.6-27B Frontier

Last reviewed: 2026-07-10.

This is the external target for one RTX 3090 (24 GiB), batch one. No reproducible
public context-depth curve was found for the exact
`unsloth/Qwen3.6-27B-NVFP4` checkpoint. Until AutoLuce measures it, use the best
comparable Qwen3.6-27B results on the same GPU as cross-quant targets, not as an
NVFP4 baseline.

| Context/workload | Prefill target | Decode target |
|---|---:|---:|
| 1K | >1,575 tok/s | >82 tok/s sustained |
| 16K | >1,432 tok/s | >73 tok/s |
| 64K | >1,111 tok/s | >66 tok/s |
| 128K | >852 tok/s | >50-66 tok/s |
| Short-context stretch | n/a | >100 tok/s, correct output |

Evidence:

- The prefill curve used Qwen3.6-27B IQ4_XS, flash attention, and one 3090:
  1K 1,575 tok/s; 16K 1,432; 64K 1,111; 128K 852.
  <https://dev.to/sysoft/the-prefill-wall-why-mtps-2x-barely-moves-long-context-latency-qwen36-27b-rtx-3090-185i>
- A sustained real workload with a 5.9K prompt and 1K output measured 1,260.95
  prefill tok/s and 72.93 decode tok/s using IQ4_KS plus MTP.
  <https://www.reddit.com/r/LocalLLaMA/comments/1tgis7s/qwen_36_27b_on_24gb_vram_setup_backend/>
- An INT4 plus MTP vLLM setup reported 82.4 tok/s for 100 output tokens, 82.1 for
  400, and 71.3 for 800, with correct output favored over a faster broken graph mode.
  <https://www.reddit.com/r/LocalLLaMA/comments/1t07su1/followup_qwen3627b_on_1_rtx_3090_pushing_to_218k/>
- vLLM's current compatibility table lists Marlin FP4 support on Ampere. That is a
  software/fallback route; the RTX 3090 has no native FP4 tensor cores.
  <https://docs.vllm.ai/en/stable/features/quantization/>

## Measurement Contract

The first AutoLuce NVFP4 baseline must report separate prompt processing and decode
cells at 1K, 8K, 16K, 64K, and the largest resident depth. Record warm and cold TTFT,
at least 256 decoded tokens, exact-quality results, peak VRAM, power, KV type, MTP
configuration, CUDA/driver versions, and the model revision. Do not use prefix-cache
hits, first-request graph capture, aggregate multi-user throughput, or Blackwell
results as the single-user RTX 3090 score.

The downloaded checkpoint target is HF revision
`dd75bc44c47d033c2a34234d020632fee738b18a` (21.83 GiB). Its tight fit means 64K and
128K may require quantized KV, CPU placement for non-language components, or another
memory reduction. A cell that does not fit is recorded as such, never extrapolated.
