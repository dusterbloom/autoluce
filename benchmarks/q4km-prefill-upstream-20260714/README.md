# RTX 3090 Q4_K_M prefill comparison with llama.cpp

Measured on 2026-07-14 for Lucebox draft PR
[#524](https://github.com/Luce-Org/lucebox/pull/524), stacked on MMQ PR #518.
The tested Lucebox source is current upstream `main` (`0e002364`) plus the MMQ
commit (`506cd972`) and the automatic grouped-GDN/QK commit (`3c0be1f3`).

## Result

The final candidate uses the product default prefill ubatch of 512. No GDN force,
GDN disable, ubatch, or chunked-GDN environment override was present.

| Requested depth | Lucebox tok/s | llama.cpp tok/s | Delta | Combined-SEM multiple |
|---:|---:|---:|---:|---:|
| 1,024 | 1341.47 | 1278.72 | +4.91% | 7.0x |
| 8,192 | 1393.73 | 1359.53 | +2.52% | 7.3x |
| 16,384 | 1332.63 | 1306.86 | +1.97% | 5.5x |

Each value pools ten measurements from an A-B-B-A run: one warmup and five
measured requests per phase. Prompt caches were disabled for both runtimes, each
request generated one token, K/V were F16/F16, the server context was 17,408, and
the fixed Q4_K_M model SHA-256 was
`5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`.

This closes the observed prefill gap for the tested MMQ+GDN stack. It does not claim
universal leadership across models, quantizations, GPUs, or longer contexts.

## Compatibility and quality boundary

Both runtimes received the same prompt payload and fixed model artifact. DFlash
reported two more prompt tokens in every cell: 988/986, 8,156/8,154, and
16,349/16,347. That accounting difference is too small to explain the measured
lead, but it means the comparison is not bit-identical at the tokenizer contract.

The upstream response parser also did not retain generated text even though the
server reported one completion token. Therefore this bundle is strong comparative
performance evidence, not an end-to-end exact-output promotion bundle. Correctness
for the changed computation is supplied separately by the Lucebox CUDA test: classic
and grouped GDN output and final recurrent state passed `NMSE <= 1e-6` at 1, 221,
477, 512, and 768 tokens with observed NMSE around `1e-14`. The graph test also
proves the fused path removes both Q/K repeats while the optional chunked fallback
retains them.

## Provenance

- Machine: RTX 3090, SM86, GPU UUID
  `GPU-3307b546-fd93-b443-8e4d-27a437ad0082`.
- Candidate commit: `3c0be1f3be018a5a1b7a3e48d3e2a88c77419d3b`.
- Candidate executable SHA-256:
  `87268b1f7b33474bc2ea485bcfd1b7cd4fd3130557d5719ac7ffa6da2355901a`.
- Candidate `libggml-cuda` SHA-256 captured immediately after the run:
  `5f664844130e17b980cec82010e71bf44b0439617c6fbf49c8fe3d99df3c2dcb`.
- llama.cpp commit: `00f5442cc4e805293280c8f85d21d8f9d4aad206` (version 9970).
- llama-server SHA-256:
  `559b37b2c755b8446dd7819bf79b4982f85f4c660d9defd66b0f9ed5e678e0dd`.
- Environment: WSL2 5.15.167.4, NVIDIA driver 610.62, CUDA 12.6.85,
  GCC/G++ 11.5.0.
- Raw JSON byte SHA-256:
  `06d1da6353ed17728cd139378adb44253c3647a79715380abfc354870c27a819`.

The immutable raw A-B-B-A result is archived as
[`measurement-792d2c28b337ed544dc8c1e3e3bd699c65d8c2b4f182813dfac272c1359fb047.json`](evidence/measurement-792d2c28b337ed544dc8c1e3e3bd699c65d8c2b4f182813dfac272c1359fb047.json).
Its filename is the AutoLuce canonical JSON content ID, independently verified after
copying the ignored campaign output into this tracked archive. The campaign contract
is under `campaigns/` and attaches llama.cpp as an optional runtime reference without
changing the earlier normal-KV baseline archive.
