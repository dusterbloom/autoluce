# RTX 3090 Qwen3.6-27B normal-KV prefill baseline

Measured on 2026-07-13 with Lucebox `5e302cbb483819cd21e72f5dd8becaa609eca8cf`
and its vendored GGML, using the Qwen3.6-27B Q4_K_M model. This archive establishes
the clean F16, Q8_0, and Q4_0 KV-cache frontier before product optimization. TQ3 is
deliberately out of scope.

## Observed result

All replicated cells use three ordered measurements and the same F16-frozen exact
one-token oracle. Rates below are the arithmetic means reported by `dflash_server`.

| Context | F16/F16 tok/s | Q8_0/Q8_0 tok/s | Q8 vs F16 | Q4_0/Q4_0 tok/s | Q4 vs F16 |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1201.4 | 1240.8 | +3.28% | 1250.8 | +4.11% |
| 8,192 | 1248.0 | 1248.9 | +0.07% | 1250.8 | +0.23% |
| 16,384 | 1199.4 | 1188.9 | -0.87% | 1198.6 | -0.07% |
| 65,536 | 941.6 | 933.3 | -0.88% | 937.7 | -0.41% |

At 64K, F16 remains the raw-speed reference, but both normal quantized lanes stay
within the campaign's 2% performance gate. Q4_0 is the useful capacity frontier:
it gives up only 0.41% versus F16 while reducing measured peak VRAM from 21.91 GiB
to 19.26 GiB. Q8_0 reaches 20.26 GiB and is dominated by Q4_0 at this depth.

The non-replicated 128K fit probes are diagnostic only and are not frontier-eligible:

| KV cache | Prefill tok/s | Peak VRAM | Exact oracle |
|---|---:|---:|---:|
| Q8_0/Q8_0 | 718.4 | 22.64 GiB | pass |
| Q4_0/Q4_0 | 720.5 | 20.63 GiB | pass |

Q4_0 therefore also dominates Q8_0 in the 128K fit probes, with about 2.00 GiB more
device headroom. A single repetition does not establish significance or promotion
quality at 128K.

## Scope and provenance

- Machine: RTX 3090, SM86, 24,576 MiB, GPU UUID
  `GPU-3307b546-fd93-b443-8e4d-27a437ad0082`.
- Environment: WSL2 kernel 5.15.167.4, NVIDIA driver 610.62, CUDA 12.6.85,
  GCC/G++ 11.5.0.
- Model SHA-256:
  `5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0`.
- Executable SHA-256:
  `a066bc87c9068978a70559f892c6f1a155ce9d1020994f60a4551185553e51ba`.
- Product digest:
  `c5ea9979d187f00ca111a025e7642aba29a1425e989f59b85d72292c05912efd`.
- Vendored GGML digest:
  `a923cb2d70641c971f3c0f6a5c47f976b79c751458519d208996d8ce742347c8`.
- Golden reference SHA-256:
  `aa3ada6f5a16dc233d43f64452b9a009cbd9a0441ddc3c7f5e1c987a14287427`.

The exact oracle checks the deterministic next token (`I`) for the campaign prompt.
It is sufficient for this baseline gate but is not a substitute for broad generation,
logit-distribution, or task-quality evaluation of a future optimization.

## Immutable evidence

Each normalized evidence record links to a separate content-addressed raw measurement
bundle. The raw bundles retain ordered samples, prompt-token counts, per-context
telemetry, correctness details, source identities, runtime environment, executable,
and shared-library hashes.

| Lane | Campaign ID | Evidence ID | Raw measurement ID |
|---|---|---|---|
| F16/F16 | `campaign-c9995470c49ba278f540dedad4ec2563a271e395f2dcf074b1ef698f72570108` | `evidence-c7ba377b9eca893f1b5385eebf232684cb23a2975c536fb89075dca6a94c99fa` | `measurement-f593924a1e18d8d17ecb8f449d321053f60666c78650427712150f2af43fdafb` |
| Q8_0/Q8_0 | `campaign-ae71870d2546368166a8adb7616f1e7deaae3af1912e477d16c7e927f792f9bf` | `evidence-55751ffb3680a914a4df502ef996aaedb9fb5633aad170877457ecf81576b564` | `measurement-4b3be1daf65cbd28c3748f62a3f34f8072ecd1ee2582198d63be8a9ca3a100b0` |
| Q4_0/Q4_0 | `campaign-cd4c8acbcf76edcd8df6416bc02ccfec9bcb7cb2e2c20c89c6a99818d59449be` | `evidence-be110b063519295ad77731784bf3bb6c03e7ad7408802227752e9a04e89f4f27` | `measurement-1fb6969fb9518bc1274f48fa8540ffccd94a6f842b2c720330a01de5f80cf7a8` |
| Q8_0/Q8_0 128K fit | `campaign-f7cb2f49a3cfa545868235369ffb2a3ad2661561f357920e74461c4bbe0a812b` | `evidence-407ee2528301cfad2ec109713d27d49c4ff9185e88b5c55607c8bcfc9b0129f8` | `measurement-11bf3bd7f3bedae8ab56f489fc20497068205d70da66aa409c70ea136f424ae4` |
| Q4_0/Q4_0 128K fit | `campaign-23207616945f549ae4f161347e9ed423dc35b0a038e5b4b9563debf5c9243146` | `evidence-8956cb30fe949efbf2eec7088c8be1a2e48d3b73c39c75b63005dc4438df78ed` | `measurement-d07222593df32a3c94ff93fbe5ac2bb001ef032f37e8007bdfa014a32bc6356d` |

Campaign contracts are under `campaigns/`; immutable normalized and raw records are
under `evidence/`. Comparison against another runtime can be attached later without
changing or discarding any of these measurements.

The 128K evidence records carry an explicit false `frontier_eligible` gate. They remain
immutable capacity observations but cannot be promoted or enter a Pareto frontier.
