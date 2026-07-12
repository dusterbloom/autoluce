# RTX 3090 Qwen3.6 GDN broadcast candidate

Measured 2026-07-11 with target-only Qwen3.6-27B IQ4_XS, F16/F16 KV, CUDA 12.6,
SM86, `DFLASH_PREFILL_UBATCH=512`, one warmup, seven measured repetitions, and a
one-token completion. The clean and candidate arms ran in both orders. A second,
lower-variance ABBA pass is the decision evidence:

- `rtx3090-qwen36-27b-gdn-broadcast-stable-ab.json`
- `rtx3090-qwen36-27b-gdn-broadcast-stable-ba.json`

The earlier noisier pass is retained in `rtx3090-qwen36-27b-gdn-broadcast-{ab,ba}.json`.

## Hypothesis

The product graph repeated GDN Q/K tensors from 16 to 48 heads in each of 48 GDN
layers even though the fused CUDA kernel already broadcasts K/Q heads with modulo
indexing. `patches/lucebox/qwen35-gdn-broadcast.patch` removes those repeat nodes.

## Result

| Prompt depth | Clean mean | Candidate mean | Delta | Exact output | AutoLuce raw-SD gate |
|---|---:|---:|---:|---|---|
| 1K | 1,424.1 tok/s | 1,436.8 tok/s | +0.89% | pass | fail |
| 8K | 1,436.6 tok/s | 1,453.4 tok/s | **+1.17%** | pass | **pass** |

Each cell contains 14 clean and 14 candidate repetitions combined across the two
orders. Both cells pass a standard-error comparison. The 8K cell also clears
AutoLuce's intentionally conservative `delta > hypot(raw_stddevs)` keep rule; 1K does
not. Combined 8K medians moved from 1,435.7 to 1,452.1 tok/s.

## Verdict

Retain the patch as an 8K prefill winner. It is correctness-preserving and closes about
1.17 percentage points of the current same-session 3.36% Luce-to-upstream deficit. The
1K effect remains positive but inconclusive under the raw-standard-deviation gate.
