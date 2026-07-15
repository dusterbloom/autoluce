# Bonsai-27B Q1 quicksort AR vs native DSpark — 2026-07-15

RTX 3090 / WSL2 diagnostic using the exact public prompt `Implement quicksort
in Python.`, 400 generated tokens, 16,384 context, temperature zero, top-k 1,
top-p 1, seed 42, no prefix reuse, and thinking disabled.

## Runtime and model contract

- Lucebox source: `b19b95e` on `feat/bonsai27b-dspark-frontier`.
- Prism binary: `b9591-62061f9`, SHA-256 recorded in
  `prism-nothink/binary.sha256`.
- Target: `Bonsai-27B-Q1_0.gguf`, SHA-256
  `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`.
- Drafter: `Bonsai-27B-dspark-Q4_1.gguf`, SHA-256
  `25e73f9f7ab5d1f7f1336711496dbc12da674e639ec88d579dc8683045befb1b`.
- The published draft metadata declares four proposals. Both engines therefore
  verify five target rows per round: one anchor plus four proposals. Lucebox
  native DSpark does not support `--ddtree` on this revision.

Prism used its documented CUDA flags: `-ngl 999 -fa on`, plus `-md`,
`--spec-type draft-dspark`, `--spec-draft-n-max 4`, and `-ngld 999` for the
speculative arm. The committed request explicitly sets
`chat_template_kwargs.enable_thinking=false` and `cache_prompt=false`.

## Matched diagnostic

All statistics are population means and standard deviations over five measured
responses after warmup.

| Engine / arm | Decode tok/s | Spec telemetry |
|---|---:|---|
| Prism AR | 72.81 ± 1.96 | — |
| Prism native DSpark | 115.75 ± 1.05 | 293 / 420 proposals accepted (69.76%) |
| Lucebox AR, original aggregate | 65.80 ± 1.52 | — |
| Lucebox native DSpark, original aggregate | 118.12 ± 2.65 | 106 rounds; 3.77 tokens/round |

The Prism raw responses and logs are under `prism-nothink/`. The original
Lucebox aggregate is `lucebox-ar-dspark.json`. A later raw Lucebox refresh
(`lucebox-h4-matched-raw.json`) measured AR at 65.58 ± 0.83 but DSpark at
111.56 ± 2.50; an isolated two-warmup refresh measured 112.42 ± 2.81. Its
acceptance trajectory and output were unchanged while throughput drifted from
108.2 to 115.7 tok/s. Consequently, the roughly two-percent point difference
between the original Lucebox and Prism DSpark aggregates is a parity diagnostic,
not evidence of a Lucebox lead.

The acceptance fields also need normalization. Prism reports accepted draft
proposals (293 / 420). Lucebox's displayed 400 / 530 includes the always-valid
anchor row; subtracting 106 anchors gives 294 / 424 proposals (69.34%). The two
engines therefore have effectively the same proposal acceptance and round
economics: about 3.8 committed tokens per target verification.

## Output relationship

- Prism AR and Lucebox AR produce the same 400-token text.
- Both speculative engines match AR for the first 239 generated tokens, then
  choose `sample_list` where AR chooses `unsorted`.
- Prism and Lucebox speculative outputs later differ slightly in whitespace and
  truncation, while each arm is deterministic across its five repetitions.

This alignment is strong evidence against a Lucebox-specific proposal-index or
rollback bug. Source inspection also finds shape-dependent target arithmetic:
width-one and width-five Q1 MMVQ use different reduction configurations, and
FlashAttention dispatches different query tiles. That is consistent with the
shared divergence, but no first-mismatch logit margins were captured, so this
bundle does not claim a proven close-logit tie or exact distribution parity.
The frozen output remains a determinism/regression gate, not an algorithmic
losslessness proof.

## Why this is slower than the 238 tok/s Qwen3.6 DFlash run

The historical Qwen3.6-27B `UD-Q4_K_XL` width-16 run committed 13.83 tokens per
target forward and reached 237.68 tok/s. Native Bonsai commits only 3.77 tokens
per target forward. Its Q1 target step is much faster (about 32 ms versus 58 ms),
but it amortizes that step over 3.7 times fewer tokens. At the native maximum of
five committed tokens, a 32 ms step has a theoretical ceiling near 156 tok/s
even with perfect acceptance.

A shape-compatible Qwen3.6 width-16 drafter was tested as a no-code escape hatch.
It loaded successfully but accepted only 18.75% as a chain (63.2 tok/s); DDTree-22
accepted 16.51% (62.1 tok/s). It is not a usable Bonsai drafter.

The research-only metadata extrapolation in `horizon-sweep/` found a 3.30%
quicksort lead at five proposals, but it lost every existing Bonsai golden prompt.
Horizons six through eight also failed to beat the native default. The published
four-proposal contract remains the correct default; a broadly faster frontier
needs a Bonsai-trained wider checkpoint or a new composed drafting algorithm.

## Evidence map

- `prism-nothink/`: commands, request, raw warmup/repetition responses, server
  logs, properties, GPU snapshots, hashes, and summaries for Prism AR + DSpark.
- `lucebox-h4-matched-raw.json`: fresh Lucebox AR + DSpark per-response payloads.
- `lucebox-binary.sha256`: exact Lucebox server binary used for the refreshes.
- `lucebox-h4-isolated-raw.json` and `lucebox-h4-isolated-server.log`: isolated
  native DSpark drift control.
- `exact-ar-output.txt`, `exact-dspark-output.txt`, `exact-gate-result.txt`:
  deterministic output comparison from the original Lucebox run.
- `horizon-sweep/`: raw ABBA logs and the rejected wider-horizon research result.
