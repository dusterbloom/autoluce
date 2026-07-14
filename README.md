# AutoLuce

Coordinate people, coding agents, and shared GPUs to improve
[Lucebox Hub](https://github.com/Luce-Org/lucebox-hub) and its vendored GGML with
reproducible evidence.

## TL;DR

```bash
curl -fsSL https://raw.githubusercontent.com/dusterbloom/autoluce/main/install.sh | bash
cd autoluce
uv run autoluce source status
uv run autoluce reproduce --simulate
# On an RTX 3090 CUDA development machine:
uv run autoluce nvfp4 test
```

AutoLuce currently provides:

- One pinned Lucebox product and vendor contract, with content-addressed run evidence.
- CUDA and HIP product builds.
- Safe machine inventory and research contracts.
- A shared queue for people and machines.
- Bounded challenges where agents can implement, review, and recombine ideas.
- Live `dflash_server` HTTP benchmarks with exact frozen-output quality gates.
- Full-vocabulary logit comparison with aggregate KL, top-k, margin, and finite-value gates.
- A tested SM86 NVFP4 W4A16 CUDA operator and microbenchmark for RTX 3090 work.
- Context-validated, prefill-only NVFP4 campaigns with recorded Lucebox/GGML tuning overrides.

The pinned Hub product does not expose token logits by default. AutoLuce carries a
validated, opt-in Lucebox patch for non-streaming first-token logits; product KL remains
fail-closed until that patch or an equivalent endpoint lands in Hub.

## Start Here

### Inspect without a GPU

```bash
uv sync
uv run autoluce source status
uv run autoluce source check --remote
uv run autoluce reproduce --simulate
uv run autoluce help
```

### Prepare a CUDA or HIP machine

```bash
uv run autoluce setup
```

Setup:

1. Checks out the pinned Lucebox Hub commit in `work/lucebox`.
2. Validates `server/deps/llama.cpp/VENDOR.md`.
3. Initializes Block-Sparse Attention when building CUDA.
4. Reuses existing GGUF files before downloading models.
5. Builds `dflash_server`, `test_dflash`, and `test_deepseek4_unit` with at most four jobs.

The product currently supports CUDA and HIP. Vulkan, Metal, and CPU are rejected at
the product boundary rather than silently using a different build. The vendored GGML
layer retains its separate CPU, CUDA, HIP, and Vulkan portability contract; AutoLuce
does not confuse vendor capability with a Lucebox product entry point.

Every real result bundle records the product revision, exact product and vendor tree
digests, runtime binary SHA-256, the resolved shared-library closure with content hashes,
dirty paths, and selected backend. AutoLuce snapshots that evidence immediately after
building and rejects the run if another person or agent changes source, executable, or a
loaded library before measurement finishes. Real local harness runs also hold a
nonblocking per-checkout source/build lease, so cooperating agents fail fast instead of
resetting or rebuilding underneath one another.

GPU clocks and thermal state can move an apparent result by several percent. Frontier
decisions therefore require interleaved clean/candidate measurements on the same
machine session; historical or sequential controls are diagnostic only.

## Current Status

| Workflow | Status |
|---|---|
| Source inspection and drift detection | Ready |
| CUDA/HIP product checkout and build | Ready |
| Doctor, consult, and machine contracts | Ready |
| Team coordinator and simulated workers | Ready |
| Agent challenges and isolated worktrees | Ready |
| HTTP `baseline`, `run`, exact `freeze`, live worker | Ready |
| Live `test-drive` | Ready after product/model setup |
| Product KL capture | Candidate endpoint validated; pending Hub integration |
| Interleaved remote `verify` | Not yet wired to the HTTP adapter |
| RTX 3090 NVFP4 operator test and microbenchmark | Ready |

Simulation is only a control-plane test. It never produces performance evidence or
updates the research frontier.

## Run A Research Campaign

`autoluce research` is the single campaign entry point. A campaign always names the
system under test, workload, objective direction, and constraints. A performance
reference is optional and is separate from correctness or quality oracles.

Start from the versioned campaign example:

```bash
mkdir -p .autoluce/research
cp examples/research-campaign.json .autoluce/research/campaign.json
uv run autoluce research
uv run autoluce research --json
```

The first command reports `observe` state with no reference. Record a harness bundle or
the compact measurement shape shown in `examples/research-measurement.json`:

```bash
uv run autoluce research --record examples/research-measurement.json --json
uv run autoluce research --advance discover
uv run autoluce research --advance explore
```

Comparison can be attached later without changing the campaign ID or any content-derived
evidence ID:

```bash
uv run autoluce research --goal 'prefill_tok_s >= 1500' --compare --json

mkdir -p .autoluce/references
cp examples/upstream-llama-reference.json .autoluce/references/upstream-llama.json
AUTOLUCE_REFERENCE_DIR=.autoluce/references \
  uv run autoluce research --against upstream-llama --compare --json
```

The bundled measurement is `1425 ± 8 tok/s`, so the `1500` SLO is intentionally
unmet; this demonstrates that goal interpretation is fail-closed rather than a
guaranteed happy-path result.

Named executable, branch/candidate, saved-bundle, accepted-baseline, published/manual
measurement, and absolute-goal references share the same campaign state. A named
reference may be attached for planning without measurements, but comparison fails closed
until it includes compatible machine, model, quantization, workload, backend, and
environment evidence. Runtime differences are allowed only when the reference explicitly
represents another runtime. Legacy result bundles remain recordable diagnostics; they are
not silently treated as comparable.

Plain output guides human decisions. `--json` emits one document for agents. Both use the
same campaign and immutable evidence archive. The lifecycle is:

```text
observe -> discover -> explore -> [compare] -> explain -> promote
```

Promotion closes one iteration, not the campaign: advancing from `promote` to
`discover` starts another cycle while retaining the accepted result and all evidence.

`compare` is optional. Promotion by evidence ID is explicit and limited to the
quality-constrained Pareto frontier:

```bash
uv run autoluce research --advance explain
uv run autoluce research --promote evidence-<sha256> --json
```

The older `consult` YAML remains the version-1 remote execution contract used by
`freeze`, `harness`, and `run`. AutoLuce can normalize it for campaign planning, but
unknown runtime, hardware, quantization, and environment identities remain explicit and
must be resolved in a fully observed v2 campaign before evidence can be recorded or
compared. Migrated headroom, accepted-baseline, power-mode, and KL policies become typed
gates rather than advisory fields. Unknown future schema versions are rejected.

### Validated RTX 3090 win

The current PR-ready candidate backports upstream llama.cpp's MMQ stream-k scheduler
into Hub's vendored GGML. On Qwen3.6-27B IQ4_XS it improves target-only prefill by
**6.02% at 1K** and **5.71% at 8K**.

Quality was checked independently of generated-text equality:

- 5/5 targeted IQ4_XS CUDA-vs-CPU operator cases passed `NMSE <= 5e-4`.
- 34 matched 248,320-value logit distributions measured mean KL `0.002729` and
  maximum KL `0.016672`, within the `0.01` mean / `0.1` maximum policy.
- Top-20 overlap remained at least 90%; clean and candidate repeats were bit-exact.
- Twenty diverse 128-token generation canaries showed no factual or structural corruption.

The patch, provenance, artifact hashes, per-prompt evidence, and quality decision are in
[`benchmarks/rtx3090-qwen36-27b-mmq-quality.md`](benchmarks/rtx3090-qwen36-27b-mmq-quality.md).

The subsequent Q4_K_M campaign stacks compact GDN Q/K broadcast and an automatic
Ampere grouped-column policy on that MMQ change. With the default 512-token ubatch and
no force overrides, its measured point estimates lead llama.cpp by **4.91% at 1K**,
**2.52% at 8K**, and **1.97% at 16K** in the final RTX 3090 A-B-B-A run. The 8K
and 16K points are the strongest; upstream phase imbalance makes the defensible
1K magnitude less precise (the first upstream phase alone implies about **3.3%**
before drift adjustment). The raw comparison is archived
under its canonical measurement ID in
[`benchmarks/q4km-prefill-upstream-20260714/`](benchmarks/q4km-prefill-upstream-20260714/README.md).
The runtimes report a two-token accounting difference, so this is retained as strong
performance evidence rather than mislabeled as bit-identical output promotion. The
accepted frontier remains draft Lucebox PR #524 stacked on MMQ PR #518; rejected
follow-on attention experiments are not included or stacked. Comparable 32K/64K
llama.cpp evidence remains the next campaign phase.

## Work As A Team

One team lead starts the coordinator on a private address:

```bash
export AUTOLUCE_COORDINATOR_TOKEN="$(openssl rand -hex 24)"
uv run autoluce coordinator --listen 127.0.0.1 --port 8765
```

Expose it through HTTPS or a private network such as Tailscale. Each contributor joins
once:

```bash
uv run autoluce join \
  --team "$TEAM_URL" --token "$TEAM_TOKEN" --name my-gpu
uv run autoluce status
```

The connection is stored in `~/.config/autoluce/team.json` with mode `0600`.

Submit a focused patch to compatible hardware:

```bash
uv run autoluce submit patches/my-candidate.patch \
  --title "Reduce expert gather overhead" \
  --backend hip --model deepseek-v4-flash
uv run autoluce status
```

A machine will process one assigned job at a time under its accelerator lease. Today,
`--simulate` can test that lifecycle with a disposable job:

```bash
uv run autoluce worker --once --simulate
```

Do not use a simulated result as research evidence or run simulation against a real
candidate you intend to measure.

Use `pause`, `resume`, and `leave` when availability changes. The coordinator sends
typed jobs and patch bytes, never arbitrary shell commands.

## Work As An Agent

Agents are researchers, not privileged hardware workers. They receive bounded task
packets, work at a pinned commit, and submit a patch plus structured evidence.
Run `uv run autoluce setup` once before starting agent worktrees.

Create a challenge with distinct approaches:

```bash
uv run autoluce agent challenge create \
  --title "Expert gather challenge" \
  --objective "Reduce batch-one routed expert overhead" \
  --why "A current product trace identifies expert gather as a bottleneck" \
  --evidence "current Lucebox product capture" \
  --model deepseek-v4-flash --backend hip --slots 2 \
  --approach "gather fusion" \
  --approach "persistent buffer reuse"
```

An implementation agent joins and claims work:

```bash
uv run autoluce agent join --name codex-kernel-1 --capability implement
uv run autoluce agent next --json
uv run autoluce agent start <task-id> --json
```

`start` creates an isolated worktree from `work/lucebox`. Submit the result:

```bash
uv run autoluce agent submit <task-id> --patch candidate.patch \
  --rationale "Focused product change" \
  --observation "What changed" \
  --risk "What might regress" --json
```

Parallel implementations remain blind until evaluation completes. Review agents compare
the evidence; recombination agents must credit at least two source artifacts. Negative
results and failures remain part of the challenge record instead of disappearing.

Agent patches may touch only:

- `server/src/`
- `server/include/`
- `server/deps/llama.cpp/ggml/`
- `server/CMakeLists.txt`

Benchmarks, models, goldens, contracts, and verifier code are protected.

## Use A Remote Lucebox

Copy `targets.example.toml` to the gitignored target configuration and edit the host,
root, and model paths:

```bash
mkdir -p ~/.config/autoluce
cp targets.example.toml ~/.config/autoluce/targets.toml
```

Then onboard the host:

```bash
uv run autoluce onboard --target strix-halo
uv run autoluce doctor --target strix-halo \
  --model deepseek-v4-flash --json
```

For a Strix Halo build:

```bash
uv run autoluce setup --target strix-halo --backend hip
```

Remote heavy commands use a nonblocking accelerator lock and cap compilation at four
jobs. `test-drive` remains safe for inventory and simulation; `test-drive --live`
builds and exercises the product-owned HTTP runtime.

## Develop NVFP4 On RTX 3090

The standalone CUDA target is a fast oracle and kernel laboratory before model-loader
work is added to Lucebox:

```bash
uv run autoluce nvfp4 test
uv run autoluce nvfp4 bench --rows 4096 --cols 4096 --iterations 200
```

It implements packed E2M1 weights, one E4M3 scale per 16 values, a global FP32 scale,
and fused W4A16 GEMV. AutoLuce prefers CUDA 12.6 under `/usr/local/cuda`, targets SM86,
and caps the build at four jobs. The test compares CUDA output with an independent CPU
oracle; the benchmark compares the packed operator with the same naive FP16 GEMV.

Vendored GGML includes the NVFP4 MMQ path. AutoLuce can convert the mixed Unsloth
checkpoint to GGUF, run it through Lucebox's product server, and benchmark actual
server-reported prompt depths. Select the local artifact explicitly; the compact
derivative leaves enough VRAM for the 128K campaign:

```bash
export AUTOLUCE_BENCHMARKS=qwen36-27b-nvfp4-prefill
export AUTOLUCE_QWEN36_NVFP4_MODEL="$HOME/models/Qwen3.6-27B-NVFP4-Q4_K_M.gguf"
export AUTOLUCE_BUILD_SUBDIR=build-cuda-sm86
export GGML_CUDA=ON

uv run autoluce setup
uv run autoluce freeze --benchmark qwen36-27b-nvfp4-prefill --overwrite
uv run autoluce baseline
```

Existing Lucebox environment controls work directly and are recorded with the result:

```bash
export DFLASH_PREFILL_UBATCH=1024
export DFLASH_CHUNKED_Q_BATCH=3072
export DFLASH_CHUNKED_CHUNK=8192
uv run autoluce harness --contexts 65536 --repetitions 1
```

AutoLuce discovers inherited uppercase `DFLASH*`, `GGML_*`, and `LUCE_*` controls, passes them
to the managed server, and stores their effective values in result provenance. For an
experiment-specific override or explicit unset, use `AUTOLUCE_RUNTIME_ENV_JSON`; its
values take precedence over the inherited environment. Diagnostic `--contexts` and
`--repetitions` overrides are written into result bundles; they do not mutate the
benchmark contract. The stable generic prefill names above are translated to the current
Lucebox `DFLASH27B_*` spellings, with an explicitly set product spelling taking precedence.

The shared 3090 performance targets and measurement rules live in
[`benchmarks/rtx3090-qwen36-27b-frontier.md`](benchmarks/rtx3090-qwen36-27b-frontier.md).

### Normal-KV prefill before TQ3

The Qwen3.6 Q4_K_M prefill campaign treats F16/F16, Q8_0/Q8_0, and Q4_0/Q4_0 KV
caches as separate compatible-evidence lanes. Both K and V are always selected
explicitly and recorded in source provenance; AutoLuce does not guess a runtime default.
The shared 1K/8K/16K/64K benchmark, promotion evidence contract, isolated hypothesis
ladder, GPU handoff commands, and optional Q8/Q4 128K fit probes are documented in
[`benchmarks/qwen36-normal-kv-prefill-campaign.md`](benchmarks/qwen36-normal-kv-prefill-campaign.md).
The initial RTX 3090 run, including content-addressed campaigns, normalized evidence,
ordered raw samples, and the frozen oracle, is archived in
[`benchmarks/normal-kv-prefill-20260713/`](benchmarks/normal-kv-prefill-20260713/README.md).
At 64K, Q4_0 retained 99.59% of F16 prefill throughput while reducing peak VRAM from
21.91 GiB to 19.26 GiB; its one-shot 128K fit probe also passed at 20.63 GiB.
TQ3 is intentionally excluded until these normal paths are correct and optimized.

## Keep Lucebox Current

[`sources/lucebox.toml`](sources/lucebox.toml) is the only source ownership manifest. It
contains the Hub commit, tracked branch, layout, build targets, backends, capabilities,
runtime, submodules, and expected GGML vendor provenance.

```bash
uv run autoluce source check --remote
uv run autoluce source status --json
```

A scheduled GitHub workflow checks for Hub movement every Monday. When it moves:

1. Review the new Hub commit.
2. Update the product pin and any changed `VENDOR.md` fields together.
3. Run `autoluce setup`, Ruff, and the test suite.
4. Add runtime or build changes to the manifest, not scattered path checks.

Never update the GGML vendor commit independently of the product commit. Hub's
`VENDOR.md` is the reconstruction contract.

## Research Rules

The intended live loop is simple: propose one change, build the pinned product, measure
baseline and candidate under the same machine lease, apply correctness and quality
gates, and retain only a statistically meaningful improvement. `dflash_server` supplies
authoritative per-request prefill/decode timings and exact output generation.

- Measure the pinned product, not a nearby standalone fork.
- Keep machine, model, backend, context, workload, and power profile in the evidence key.
- Reject exact correctness, memory, or workload-constraint regressions. Require KL only
  on runtimes that can capture token logits; never approximate it from generated text.
- Keep a candidate only when its gain clears the configured statistical threshold.
- Preserve negative results so another person or agent does not repeat them.
- Allow one active accelerator experiment per physical machine.

The archived `deepseek-v4-sinkhorn-77bccaa.patch` is reference material, not a ready
candidate. Current Hub owns DeepSeek V4 under `server/src/deepseek4` and already contains
custom fused HC CUDA/HIP paths. Re-profile the current product before porting it.

## Project Map

```text
sources/lucebox.toml       Product and vendor source contract
experiment.py              Applies one product or vendor experiment
autoluce/source_layout.py  Authoritative paths and capabilities
autoluce/bench/            Measurement, quality, telemetry, statistics
autoluce/runtime/          Product HTTP lifecycle and response adapters
autoluce/loop/             Experiment and verification lifecycle
autoluce/agent_*.py        Agent challenges, evidence, review, credit
autoluce/coordination.py   Shared machine queue
benchmarks/                Fixed workloads and constraints
python_tests/              Unit and end-to-end tests
native/nvfp4/              SM86 NVFP4 CUDA oracle, operator, and microbenchmark
program.md                 Detailed autonomous-agent rules
ROADMAP.md                 Research ideas and priorities
```

## Development

```bash
uv sync
uv run ruff check .
uv run pytest -q
uv run autoluce reproduce --simulate
```

CUDA builds must remain at `-j4` or lower. Do not run heavy containers and accelerator
builds at the same time on shared machines.

Use `uv run autoluce help` for the full command inventory and
`uv run autoluce <command> --help` for command-specific options.

## License

Apache-2.0.
