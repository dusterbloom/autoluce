# Native DSpark horizon extrapolation

This is a research-only probe of whether the published Bonsai drafter can
extrapolate beyond its trained four-proposal block. A temporary copy of the
draft GGUF changed only `dspark.dspark.block_size`; no tensor was edited. After
restoring the value to four, `cmp` confirmed the copy was byte-identical to the
published file.

Horizon five improved the 400-token quicksort diagnostic by 3.30% in an ABBA
run (122.10 vs 118.21 tok/s) by reducing target forwards from 106 to 96. It was
not a general improvement: it lost all three existing Bonsai golden prompts,
including the 128-token Fibonacci case. Horizons six through eight also failed
to beat the native horizon-four baseline.

The result is not promoted. The published GGUF contract remains four proposals
and a five-row target verification batch (anchor plus four proposals). Any
future wider checkpoint or composed algorithm needs a new model/algorithm
contract and corpus-level gates.

`summary.json` contains the measured samples and prompt-breadth decision. The
four server logs preserve the ABBA quicksort telemetry; the first request in
each log is the excluded warmup.
