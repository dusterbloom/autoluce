# Bonsai-27B Q1 native DSpark frontier

Validated 2026-07-14 on one RTX 3090 under WSL2 with CUDA 12.6.

## Reproducibility

- Lucebox source: `dusterbloom/lucebox-hub` commit
  `3ade7bc8d792f6d907606f0c00997b6a275080ed`
- Hugging Face revision:
  `0cf7e3d21581b169b4df1de8bf01316000e2fbb7`
- Target: `Bonsai-27B-Q1_0.gguf`, SHA-256
  `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`
- Draft: `Bonsai-27B-dspark-Q4_1.gguf`, SHA-256
  `25e73f9f7ab5d1f7f1336711496dbc12da674e639ec88d579dc8683045befb1b`

## Frontier diagnostic

One measured repetition at the 512-token context cell, followed by an exact
golden check across all three campaign prompts:

| Metric | Result |
|---|---:|
| Decode | 98.9 tok/s |
| Prefill | 1,388.5 tok/s |
| Accepted verify positions | 89.63% |
| Peak VRAM | 7.11 GiB |
| Exact golden prompts | 3/3 pass |
| Constraint violations | none |

This is a frontier bring-up diagnostic, not a multi-context/multi-repetition
headline. A separate 256-token code prompt measured 80.4 tok/s DSpark versus
66.5 tok/s target-only AR (1.21x) at 78.3% accepted verify positions. The
server also completed two consecutive requests with request-scoped draft
unload/reload at a configured 16K context, covering the persistent draft-KV
lifecycle.

Greedy DSpark and target-only AR can diverge at close logits because width-5
verify uses different CUDA reduction shapes than width-1 AR. The frozen golden
therefore protects deterministic product behavior for this exact source and
artifact pair; it does not claim universal bit identity with token-at-a-time AR.
