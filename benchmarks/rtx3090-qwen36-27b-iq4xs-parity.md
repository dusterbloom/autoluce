# RTX 3090 IQ4_XS engine parity: lucebox dflash_server vs llama.cpp

Measured 2026-07-11. Full raw arrays and provenance are retained in
`rtx3090-qwen36-27b-iq4xs-parity-raw.json`.

## Contract (all axes held; caching symmetry proven per-rep)

- Model: `~/models/qwen36-27b-unsloth-gguf/Qwen3.6-27B-IQ4_XS.gguf`
  (unsloth, sha256 `8a3365...53e5dc`), same file both arms.
- Target-only (no draft), f16/f16 KV, ctx 8448 both arms (llama: `--parallel 1` —
  default slot-splitting would silently quarter the context; caught in smoke test).
- temp 0, top_k 1, seed 42, n_predict 128; 1 warmup + 5 measured per cell.
- Distinct salted prompts per rep; llama `timings.cache_n == 0` on all 20 reps,
  dflash `prefix_len=0` on all 24 requests. Flash attention on both. GPU idle-checked,
  arms sequential.
- Arms: llama.cpp build 9970 (`00f5442cc`, vanilla, 2026-07-11 master) vs
  dflash_server @ clean `5e302cbb` (binary md5 `5467429a...`).

## Results (median of 5; dflash Δ vs llama)

| Depth | dflash prefill | llama prefill | Δ | dflash decode | llama decode | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 1K | 1,357.0 | 1,360.5 | −0.3% | 45.8 | 46.0 | −0.4% |
| 2K | 1,388.6 | 1,450.6 | −4.3% | 45.5 | 47.3 | −3.8% |
| 4K | 1,379.7 | 1,460.3 | −5.5% | 43.7 | 48.2 | −9.3% |
| 8K | 1,337.7 | 1,471.0 | −9.1% | 43.2 | 47.4 | −8.9% |

llama 1K session-to-session sanity: +2.5% prefill / +2.6% decode (noise band).

## Same-session correction

A later prefill-only control ran the unchanged clean Luce binary and upstream in the
same stable session (seven repetitions, one-token completion):

| Depth | clean Luce median | upstream median | Luce delta |
|---|---:|---:|---:|
| 1K | 1,439.0 | 1,441.5 | -0.17% |
| 8K | 1,447.5 | 1,497.9 | -3.36% |

The original 9.1% 8K result is not a stable engine delta. Most of it was session-level
RTX 3090 clock/power/thermal drift. It remains useful as evidence that sequential,
unpaired GPU measurements are unsafe, not as the optimization baseline. Raw controls:
`rtx3090-qwen36-27b-gdn-inplace-rejected.json` (clean arm) and
`rtx3090-qwen36-27b-llama-prefill-control.json`.

The retained GDN broadcast candidate improves 8K by 1.17%, reducing the estimated
same-session residual to roughly 2.2%. See
`rtx3090-qwen36-27b-prefill-gap-forensics.md`.

## Findings

1. **Engine parity at 1K; llama.cpp pulls ahead with depth.** The original session
   showed about 9% at 8K, but the controlled same-session prefill residual is 3.36%.
   Subsequent source forensics did not find a post-fork SM86/IQ4_XS CUDA commit that
   explains the gap by itself: NVFP4 MMVQ fusion is irrelevant to this control and the
   normal IQ4 loader is byte-identical. The product graph does perform redundant GDN
   Q/K head repeats, always rotates F16 KV through FWHT, copies recurrent state, rebuilds
   each chunk graph, and uses a custom KV/attention path. These are now separate,
   correctness-gated hypotheses. A vendor-sync control remains useful, but is no longer
   assumed to be the root cause before measurement.
2. **F16 FWHT is not the missing cost.** A direct candidate that bypassed F16/BF16
   K/Q rotation changed exact greedy output, slowed the 1K median by 0.6%, and improved
   the 8K median by only 0.9%. It was rejected before reverse-order measurement.
3. **The `qwen35-prefill-skip-intermediate-output` patch is a global regression.**
   First run unknowingly measured a build with it applied: 3–6% slower on EVERY cell,
   both prefill and decode, vs clean (prefill +6.3/+5.9/+4.3/+3.0%, decode
   +6.0/+4.6/+3.1/+5.9% clean-over-dirty at 1K/2K/4K/8K). Its keep decision should be
   re-examined under the k·sigma gate.
4. A 2026-06 survey of all 51 open lucebox PRs found none that attack this gap
   (#471/#473 mechanisms already vendored locally, author-reported deltas ~1–1.6%
   flat with depth — wrong shape). #390 (speculative fast-rollback, +30% decode
   claim) is the one to test separately in speculative mode after a rebase.

## Vendor-sync experiment result (2026-07-11 evening)

Measured: `patches/llama.cpp/vendor-ggml-cuda-sync-9970.patch` (61 files,
+2739/−1345) — 3-way merge of upstream ggml-cuda from true ancestor `fae3a28070`
(b8796, 2026-04-14) to build 9970, onto the `Luce-Org/lucebox-ggml` `luce-dflash`
fork. Synced: fattn (all variants), mmq/mmf/mmvf, partial cuBLAS refactor, misc ops.
Excluded (too entangled with lucebox custom ops): `ggml-cuda.cu` dispatcher,
`mmvq.cu/.cuh`, `gated_delta_net.cu`, core ggml.c/h. `test_deepseek4_unit` clean.
Same contract as the baseline; medians of 5:

| Depth | prefill Δ vs clean dflash | prefill Δ vs llama 9970 | decode Δ vs clean | decode Δ vs llama |
|---|---:|---:|---:|---:|
| 1K | +5.1% | **+4.9% (ahead)** | +1.1% (noise) | +0.7% (noise) |
| 2K | +4.2% | −0.2% (parity) | +0.2% (noise) | −3.6% |
| 4K | +6.1% | +0.2% (parity) | +3.4% | −6.1% |
| 8K | +7.8% | −2.0% (≈noise) | +1.4% (noise) | −7.7% |

**Revised verdict:** this run makes stale vendored attention kernels a strong candidate,
but it did not interleave a same-session clean arm. Its apparent 7.8% 8K gain was
computed against the older, drift-affected baseline and therefore cannot be credited as
a code delta. Re-run the vendor patch as clean/candidate ABBA under AutoLuce's source,
binary, and machine-state checks before promotion. The raw result and patch remain
valuable hypothesis evidence, not a frontier result.

## Content-addressed ABBA rerun (2026-07-12)

The required rerun held the GDN broadcast candidate and product executable constant,
changing only the resolved `libggml-cuda.so.0.9.11`. Four seven-repetition blocks ran
in clean/vendor/vendor/clean order. The executable SHA-256 was
`001d7cc850682ebe9ea3d2bc2ecd50d92839cddd846eae4ad89ddcf4739b51dc`.
Clean and vendor CUDA-library SHA-256 values were respectively `76c2aa...cbc1` and
`8f2829...4320`.

| Depth | Clean combined mean | Vendor combined mean | Delta | Raw-SD gate | Exact salted output |
|---|---:|---:|---:|---|---|
| 1K | 1,366.8 | 1,424.8 | **+4.24%** | pass | **fail** |
| 8K | 1,379.9 | 1,443.4 | **+4.60%** | pass | **fail** |

The vendor arm won in both directions: +3.55%/+4.95% at 1K and +4.02%/+5.19% at
8K. However, each arm produced its own reproducible greedy-token sequence for identical
salted prompts. For example, 8K repetition 4 returned `Based` on clean and `It` on
vendor in both vendor blocks; the clean sequence repeated in both clean blocks.

**Decision: reject the 61-file patch as a whole.** It confirms that the stale vendor
layer contains more than enough performance to cover the residual prefill gap, but it
does not satisfy exact behavioral preservation. Decompose the patch into buildable
tensor-core, KV/data-path, and generic families and retain only correctness-preserving
subsets. Raw evidence: `rtx3090-qwen36-27b-vendorsync-abba-*.json`.

## Tensor-core closure ABBA (2026-07-12)

The first decomposition retained only the smallest buildable tensor-core closure:
FlashAttention, MMQ/MMF/MMVF, quantization/vecdot, their template instances, and the
shared `mma.cuh`/`common.cuh` interfaces. It excluded the broad patch's dispatcher,
KV/data-path, and generic-operation changes. The exact patch is
`patches/llama.cpp/vendor-ggml-cuda-sync-9970-tensor-core.patch` (31 files,
+1770/-920; SHA-256 `b0f589...8a7`).

The same immutable executable was used for a new clean/subset/subset/clean run. Only
`libggml-cuda.so.0.9.11` changed: clean `76c2aa...cbc1`, subset `d235b8...78f9`.

| Depth | Clean combined mean | Subset combined mean | Delta | Directional deltas | Exact salted output |
|---|---:|---:|---:|---:|---|
| 1K | 1,370.0 | 1,432.4 | **+4.55%** | +4.07% / +5.05% | **fail** |
| 8K | 1,383.7 | 1,447.2 | **+4.58%** | +4.21% / +4.96% | **fail** |

Each subset block reproduced the same candidate-specific output sequence, while both
clean blocks reproduced the clean sequence. The behavior change is therefore localized
to this tensor-core closure, as is essentially all of the broad patch's prefill gain.

**Decision: reject the combined tensor-core closure.** The next useful decomposition is
not another whole-file family: FlashAttention and MMQ share refactored tensor-core
interfaces. Their interface hunks must be separated surgically, then built and tested as
an FAttn closure and an MMQ/quantization closure under the same content-addressed ABBA
contract. Raw evidence: `rtx3090-qwen36-27b-vendorsync-tensor-abba-*.json`.

## MMQ stream-k isolation (2026-07-12)

Source forensics identified a one-file candidate inside the rejected tensor closure:
the upstream MMQ stream-k scheduler and fixup rewrite. It changes the launch from a
fixed SM-count grid to one block per destination tile when tile efficiency is at least
90%; otherwise it repartitions fixup work. It requires no FAttn, `mma.cuh`,
`common.cuh`, quantization, or dispatcher changes. The exact 138-line patch is
`patches/llama.cpp/vendor-ggml-cuda-sync-9970-mmq-streamk.patch` (SHA-256
`9de9ad...946f`).

A content-addressed candidate/clean/candidate/clean sequence used the same immutable
product executable and base/CPU libraries. CUDA-library SHA-256 values were clean
`76c2aa...cbc1` and MMQ `42c09f...9d91`.

| Depth | Clean combined mean | MMQ combined mean | Delta | Directional deltas | Exact salted output |
|---|---:|---:|---:|---:|---|
| 1K | 1,368.1 | 1,450.4 | **+6.02%** | +5.98% / +6.05% | **fail** |
| 8K | 1,382.9 | 1,461.9 | **+5.71%** | +5.73% / +5.69% | **fail** |

Both MMQ blocks exactly reproduced the combined tensor closure's candidate-specific
token sequence at both depths. Both clean blocks reproduced the clean sequence. This
one scheduler rewrite therefore owns the observed output divergence and more than the
combined closure's measured throughput gain; stale FAttn is no longer the leading
explanation for this IQ4_XS prefill gap.

**Initial decision: hold for quantitative quality evidence.** That follow-up now passes.
Five targeted IQ4_XS CUDA-vs-CPU operator cases passed the `NMSE <= 5e-4` oracle, while
34 matched full-vocabulary samples measured mean KL 0.002729, maximum KL 0.016672, and
at least 90% top-20 overlap. Repeat captures were bit-exact and 20 longer diverse
generation canaries showed no corruption. The scheduler is promoted as a Lucebox Hub
PR candidate; see `rtx3090-qwen36-27b-mmq-quality.md`. Raw performance evidence:
`rtx3090-qwen36-27b-vendorsync-mmq-streamk-*.json`.

## Evidence warning

The original manual run exposed that source state and a previously built binary can
diverge. Real AutoLuce result bundles now content-address both trees and the executable,
then fail if any of them changes between build completion and measurement completion.
