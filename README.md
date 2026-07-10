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

- One pinned Lucebox product and vendor contract.
- CUDA and HIP product builds.
- Safe machine inventory and research contracts.
- A shared queue for people and machines.
- Bounded challenges where agents can implement, review, and recombine ideas.
- Live `dflash_server` HTTP benchmarks with exact frozen-output quality gates.
- A tested SM86 NVFP4 W4A16 CUDA operator and microbenchmark for RTX 3090 work.

The product API does not expose token logits. Exact quality is operational; benchmarks
that explicitly require KL fail closed until Lucebox gains a logits endpoint.

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
the product boundary rather than silently using a different build.

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
| Product KL capture | Needs a Lucebox token-logits endpoint |
| Interleaved remote `verify` | Not yet wired to the HTTP adapter |
| RTX 3090 NVFP4 operator test and microbenchmark | Ready |

Simulation is only a control-plane test. It never produces performance evidence or
updates the research frontier.

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

This is the operator foundation, not complete Unsloth checkpoint support. Vendored GGML
already builds an NVFP4 MMQ template, which is now the first product comparison target.
Lucebox still needs the HF tensor loader/metadata contract and graph dispatch before an
Unsloth NVFP4 model can be served end to end.

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
