# RTX 3090 Qwen3.6-27B IQ4_XS prefill gap forensics

Measured 2026-07-11 with target-only prefill, F16/F16 KV, CUDA 12.6, SM86,
`DFLASH_PREFILL_UBATCH=512`, and exact greedy output checks.

## Corrected gap

The original sequential 8K comparison reported Luce 1,337.7 versus upstream 1,471.0
tok/s (-9.1%). A later stable same-session control measured clean Luce at 1,447.5 and
upstream at 1,497.9 tok/s (-3.36%). At 1K the engines were effectively tied: 1,439.0
versus 1,441.5 (-0.17%). Most of the original 8K gap was machine-state drift, not code.

The exact-output GDN broadcast patch then improved 8K by 1.17%, reducing the estimated
residual from 3.36% to approximately 2.2%.

## Candidate matrix

| Candidate | Depth | Delta | Quality | Decision |
|---|---:|---:|---|---|
| Remove redundant GDN Q/K repeats | 8K | **+1.17%** | exact | retain; raw-SD gate passes |
| Remove redundant GDN Q/K repeats | 1K | +0.89% | exact | inconclusive; raw-SD gate fails |
| Skip target-only feature captures | 8K | +0.44% | exact | do not retain; raw-SD gate fails |
| Write GDN state in place | 8K | -0.53% median | exact | reject |
| Upstream-style KV `set_rows` | 8K | -0.85% median | exact | reject |
| Remove F16 FWHT | 8K | +0.90% noisy | exact output changed | reject |

## CUDA API attribution

Nsight Systems cannot expose GPU kernel timing in this WSL environment, but CUDA API
backtraces are available. Whole-process counts for two 8K requests are in
`rtx3090-qwen36-27b-prefill-cuda-api.csv`.

The large Luce synchronization count is dominated by first-request initialization:

- `do_prefill()` unconditionally migrates the prefill-only recurrent cache to rollback
  storage, even for target-only AR.
- Four tensors across 48 recurrent layers are allocated and zeroed in 1 MiB chunks.
- This contributes 1,392 synchronous uploads and about 1.31 GiB of unnecessary
  target-only rollback storage.
- Before Luce's first graph compute, the trace records 3,182 synchronous tensor sets.

The lazy-migration candidate in `patches/lucebox/qwen35-lazy-rollback-cache.patch`
was validated in an isolated build. At 8K it preserved the exact salted completion and
reduced synchronous tensor sets before the first graph compute from 3,182 to 1,790:
exactly the predicted 1,392-call removal. It remains a first-request latency and memory
fix, not a steady-state throughput improvement. Speculative E2E must pass before the
patch is promoted because rollback allocation now occurs at the `can_spec` transition.

On the measured second request Luce performs 16 graph synchronizations, 45 tensor
uploads, and 16 readbacks. Upstream performs 67, 110, and 130 respectively and is still
faster. Kernel count is also higher upstream. Synchronization or launch count alone is
therefore ruled out as the remaining cause.

The GDN in-place candidate reinforces that conclusion: it removed 1,536 memcpy calls
over two requests, but introduced 3,072 kernel launches, left synchronization unchanged,
and slowed 8K prefill.

## Remaining leading causes

1. **MMQ stream-k scheduling.** A one-file scheduler/fixup slice from the vendor sync
   reproduced the tensor closure's exact candidate-specific output sequence and improved
   paired prefill by 6.02% at 1K and 5.71% at 8K. This is enough to cover the residual
   engine gap without any FAttn, shared tensor-core, quantization, dispatcher, or KV
   changes. The quantitative gate now passes: five CPU-reference IQ4_XS operator cases,
   34 matched full-logit samples (mean KL 0.002729, maximum 0.016672, top-20 overlap at
   least 90%), bit-exact repeat captures, and 20 longer generation canaries found no
   corruption. The candidate is ready to package as an upstream-backport PR.
2. **Custom Luce graph and KV layout.** Luce and upstream reach different attention
   graphs and physical cache writes. The rejected `set_rows` transplant shows that an
   isolated upstream graph idiom is not automatically faster inside Luce's layout.
3. **Per-chunk graph reconstruction.** Luce rebuilds and reallocates graph metadata for
   every ubatch. CUDA graphs and graph identity differ from upstream, but a structural
   fix must avoid changing `ggml_cgraph` layout in the vendored ABI.

## Next experiments

1. Submit the MMQ stream-k scheduler as an upstream-backport PR with the missing SM86
   IQ4_XS scheduler-transition operator cases and the content-addressed performance and
   quality evidence in `rtx3090-qwen36-27b-mmq-quality.md`.
2. Add product phase timers around graph construction/allocation and GPU compute.
3. Run the lazy rollback patch through correctly configured BF16 DFlash chain and tree
   verification before promotion. Score first-request latency and peak memory separately.
4. After the MMQ numerical gate, measure a FAttn-only closure only as an independent
   opportunity. It is no longer required to explain the current IQ4_XS prefill gap.

No result here establishes the NVFP4 frontier; this IQ4_XS campaign is an engine-parity
control used to remove product overhead before NVFP4 kernel work.
