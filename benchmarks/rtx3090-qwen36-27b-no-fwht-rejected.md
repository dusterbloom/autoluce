# RTX 3090 F16 KV no-FWHT candidate: rejected

Measured 2026-07-11 with the same target-only Qwen3.6-27B IQ4_XS prefill contract
as the GDN broadcast candidate. Raw evidence is in
`rtx3090-qwen36-27b-no-fwht-rejected.json`; the candidate source is retained as
`patches/lucebox/qwen35-unquantized-kv-no-fwht.patch`.

The hypothesis was that F16 K/Q FWHT work explained the context-dependent parity
gap. The candidate bypassed graph rotation only for unquantized F16/BF16 K caches;
quantized Q4/Q8 and TQ3 behavior was unchanged.

| Prompt depth | Clean median | Candidate median | Delta | Exact output |
|---|---:|---:|---:|---|
| 1K | 1,369.6 tok/s | 1,361.3 tok/s | -0.6% | fail |
| 8K | 1,330.6 tok/s | 1,342.6 tok/s | +0.9% | fail |

The first 1K greedy token changed from `It` to `The`. Because exact quality failed
and the performance effect was small, AutoLuce stopped before the reverse-order arm.
FWHT overhead is not the missing 9%; the next forensic targets are the custom
attention/KV layout and per-chunk graph construction lifecycle.
