# RTX 3090 Qwen3.6-27B IQ4_XS MMQ quality gate

Measured 2026-07-12 for the one-file MMQ stream-k scheduler patch in
`patches/llama.cpp/vendor-ggml-cuda-sync-9970-mmq-streamk.patch`.

## Provenance

The candidate is a surgical backport of upstream llama.cpp PR #22298,
`CUDA: reduce MMQ stream-k overhead`, commit `9725a313be0528214c4a02fed906ddaf7b3f712e`.
It is already present in the former full `Luce-Org/lucebox-ggml` tree but absent from
the GGML snapshot vendored into Lucebox Hub. The exact matching preimage is full-tree
commit `ab50058c8`; its `mmq.cuh` blob matches the Hub vendor preimage.

## Artifact contract

- Model SHA-256: `8a3365759dc1b33b52c4e7d91d5a67d5ee1418e8408aa54196f04a98da53e5dc`
- Diagnostic executable SHA-256: `2780101a88073c5cc4b3eb7d0f51b82944bad98caddfab5e771a1d3762dbc47f`
- Clean CUDA library SHA-256: `76c2aa020e85af58bfef26b19f8c989936483dec704566585c6d508dfc40cbc1`
- Candidate CUDA library SHA-256: `42c09f1552d3fa8e41947be2dd9a1a3ba9dfcc168bf5b6a1ab5fd2266f749d91`
- Raw logits: F32, final prompt position, vocabulary axis, before sampler transforms
- Vocabulary size: 248,320

Only the resolved CUDA library changed between arms. The product executable and all
request parameters were identical. Prefix and disk-cache restore were disabled for
diagnostic requests.

## Operator oracle

The clean and candidate libraries were each tested against GGML's CPU reference using
`test-backend-ops`, whose `MUL_MAT` gate rejects non-finite values and requires
`NMSE <= 5e-4`. Five explicit SM86 IQ4_XS shapes covered:

- 97%-efficient one-block-per-tile scheduling with no fixup.
- Partial output rows.
- The below-threshold stream-k fixup path.
- The exact two-wave scheduler boundary.
- A deeper K dimension.

Both clean and candidate passed **5/5** cases.

## Full-logit results

The project policy is mean KL `<= 0.01`, maximum KL `<= 0.1`, and minimum top-20
overlap `>= 0.90`.

| Corpus | Samples | Mean KL | Maximum KL | Min top-20 overlap | Argmax changes | Gate |
|---|---:|---:|---:|---:|---:|---|
| Synthetic 1K depth | 7 | 0.003961 | 0.012618 | 0.90 | 1 | pass |
| Synthetic 8K depth | 7 | 0.007234 | 0.016672 | 0.90 | 2 | pass |
| Diverse task prompts | 20 | 0.000721 | 0.004083 | 0.95 | 0 | pass |
| **Combined** | **34** | **0.002729** | **0.016672** | **0.90** | **3** | **pass** |

Clean-to-clean and candidate-to-candidate repeat captures were bit-exact across all 14
depth probes. The three argmax changes are deterministic and confined to synthetic
prompts with small top-two margins. All 20 diverse prompts preserve the first token.

Detailed per-prompt metrics and raw-evidence hashes are retained in
`rtx3090-qwen36-27b-mmq-logit-quality.json`.

## Generation canaries

Twenty diverse prompts were also extended to as many as 128 greedy tokens. Eleven
outputs were byte-identical. The other nine were coherent paraphrases or alternate
explanations; manual inspection found no factual or structural corruption. One
deliberately mixed sentiment prompt changed from `Negative` to `Neutral`, both defensible
labels for the input. The retained raw responses are
`rtx3090-qwen36-27b-mmq-generation-{clean,candidate}.json`.

## Decision

**Pass the quality promotion gate.** The patch changes deterministic floating-point
reduction order, so byte-identical generation is not expected at every near-tied token.
It passes the CPU-reference operator oracle, the existing aggregate KL policy, top-k
stability, repeat determinism, and diverse generation canaries while retaining the
measured +6.02% 1K and +5.71% 8K prefill improvement.

This is suitable for a Lucebox Hub pull request as an upstream GGML backport. The PR
should include the missing IQ4_XS stream-k scheduler cases because upstream's stock
`other_types` coverage does not exercise this transition.
