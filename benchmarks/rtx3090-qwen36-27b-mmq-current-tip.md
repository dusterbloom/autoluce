# RTX 3090 Qwen3.6-27B IQ4_XS MMQ current-tip validation

Measured 2026-07-13 against Lucebox Hub `main` commit
`0e0023649131a23f45d58be71f2bfc60d6cd25a0`. This record validates the
one-file upstream llama.cpp #22298 backport on the current Hub source; it does
not relabel or replace the earlier evidence collected at `5e302cbb`.

## Content-addressed evidence

- Evidence ID: `evidence-sha256-be8ec967d917b0a1110e55fc5571b23756f77228739a5a374b00c6898a565964`
- Evidence file SHA-256: `5805ded9110250a1b178f5952e125dd2bda6eec211074931f33017209009d143`
- Raw record: `evidence-sha256-be8ec967d917b0a1110e55fc5571b23756f77228739a5a374b00c6898a565964.json`
- Model SHA-256: `8a3365759dc1b33b52c4e7d91d5a67d5ee1418e8408aa54196f04a98da53e5dc`
- Clean MMQ Git blob: `d8df400fbc3a1eee2049a810089f395e5ba2d7a2`
- Candidate MMQ Git blob: `b867d39653c19cf4efd3bb7809e8b43572a79a6e`
- Clean executable SHA-256: `8ad8a9fa53e54ef640f1503527c1990455811fa19bae8b8f0fc736004d693ad0`
- Candidate executable SHA-256: `dddb7db95cd7c54d0a2d0c84b8ab0f3dc30cd5daf7f5153217d5b42c1669b232`

The evidence ID is SHA-256 over canonical JSON after removing the
`evidence_id` field. The raw record retains the exact machine identity,
toolchain, workload, request parameters, per-repetition observations, source
identities, and artifact hashes.

## Performance result

The measurement used A-B-B-A ordering on the same RTX 3090, with one warmup
and five measured repetitions per phase and context. Prefix and prefill caches
were disabled at both server and request levels.

| Context | Clean pooled median | Candidate pooled median | Relative change |
|---:|---:|---:|---:|
| 1,024 | 1300.68 tok/s | 1365.68 tok/s | **+5.00%** |
| 8,192 | 1312.16 tok/s | 1382.97 tok/s | **+5.40%** |

## Correctness result

The current-tip candidate passed all five focused IQ4_XS CPU-vs-CUDA
`MUL_MAT` oracle cases on the RTX 3090. The cases cover high-efficiency tiling,
partial output rows, stream-k fixup, the exact two-wave transition, and a
deeper K dimension. Every output was finite and every NMSE was below GGML's
`5e-4` operator threshold.

The earlier full-logit, generation-canary, repeat-determinism, and operator
evidence remains content-addressed by its own artifacts in
`rtx3090-qwen36-27b-mmq-quality.md`. It was collected at `5e302cbb` and is
supporting historical evidence, not a silently substituted current-tip
comparison.
