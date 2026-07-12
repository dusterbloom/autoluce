# RTX 3090 Qwen3.6-27B lazy rollback-cache validation

Measured 2026-07-12 with target-only Qwen3.6-27B IQ4_XS, F16/F16 KV, CUDA 12.6,
SM86, an 8K prompt, one-token greedy completion, caches disabled, and an isolated
product build at Lucebox commit `5e302cbb483819cd21e72f5dd8becaa609eca8cf`.

## Change

`patches/lucebox/qwen35-lazy-rollback-cache.patch` removes rollback-cache migration
from `do_prefill()`. The cache is promoted only after `do_spec_decode()` establishes
`can_spec`. Promotion is required for every speculative mode, including plain chain
verification with fast rollback and DDTree disabled, because that path also snapshots
recurrent state.

## Result

| Measurement | Existing path | Lazy candidate | Delta |
|---|---:|---:|---:|
| Synchronous tensor sets before first graph compute | 3,182 | 1,790 | **-1,392** |
| Whole-process synchronous tensor sets, two requests | 3,275 | 1,883 | **-1,392** |
| Estimated unused rollback allocation | about 1.31 GiB | 0 in target-only AR | about **-1.31 GiB** |
| Salted 8K greedy completion | `Based` | `Based` | exact |

The 1,392-call reduction equals 29 one-MiB zeroing chunks across each of 48 recurrent
layers. The remaining 1,790 pre-compute sets belong to other model/cache initialization
and are outside this candidate.

The isolated candidate executable SHA-256 was
`bf6e26ee14ca1c5c6e29019dd057b1b32297fb957214bf41b64d40cf0eeea65e`.
The E2E run resolved the clean GGML CUDA library SHA-256
`76c2aa020e85af58bfef26b19f8c989936483dec704566585c6d508dfc40cbc1`.

## Provenance finding

The first attempted run resolved a different shared `libggml-cuda.so` from a clean-source
checkout's preserved build directory and failed in an incompatible SSM-conv call. The
executable itself had been relinked correctly. This directly reproduced why AutoLuce must
hash the resolved shared-library closure rather than only `dflash_server`.

## Promotion gate

Target-only behavior and compilation pass. A BF16 Qwen3.5 DFlash, SWA-2048,
ring-2048, `--no-fast-rollback` plain-chain gate was attempted against both the clean and
final part-1 vendor libraries. Both fail before rollback is exercised in the unsynced
central CUDA dispatcher at `ggml_cuda_cpy_tensor_2d(...): invalid pitch argument`.
Fable5 part 2 targets that excluded dispatcher. After that pre-existing failure is fixed,
rerun plain chain and DDTree/tree verification and require exact output parity before
promoting this product patch.
