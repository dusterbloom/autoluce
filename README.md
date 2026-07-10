# autoggml

Coordinate people, coding agents, and shared GPUs to improve
[Lucebox Hub](https://github.com/Luce-Org/lucebox-hub) and its vendored GGML with
reproducible evidence.

## TL;DR

```bash
curl -fsSL https://raw.githubusercontent.com/dusterbloom/autoggml/main/install.sh | bash
cd autoggml
uv run autoggml source status
uv run autoggml reproduce --simulate
```

`autoggml` currently provides:

- One pinned Lucebox product and vendor contract.
- CUDA and HIP product builds.
- Safe machine inventory and research contracts.
- A shared queue for people and machines.
- Bounded challenges where agents can implement, review, and recombine ideas.
- Reproducible simulation and end-to-end coordination tests.

Live performance and quality measurements are temporarily blocked while the old
standalone `llama-*` adapter is replaced with a Lucebox `dflash_server` HTTP adapter.
Affected commands fail clearly instead of benchmarking a different engine.

## Start Here

### Inspect without a GPU

```bash
uv sync
uv run autoggml source status
uv run autoggml source check --remote
uv run autoggml reproduce --simulate
uv run autoggml help
```

### Prepare a CUDA or HIP machine

```bash
uv run autoggml setup
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
| `baseline`, `run`, `freeze`, `verify`, KL, live worker | Waiting for product HTTP adapter |
| Live `test-drive` | Waiting for product HTTP adapter |

Simulation is only a control-plane test. It never produces performance evidence or
updates the research frontier.

## Work As A Team

One team lead starts the coordinator on a private address:

```bash
export AUTOGGML_COORDINATOR_TOKEN="$(openssl rand -hex 24)"
uv run autoggml coordinator --listen 127.0.0.1 --port 8765
```

Expose it through HTTPS or a private network such as Tailscale. Each contributor joins
once:

```bash
uv run autoggml join \
  --team "$TEAM_URL" --token "$TEAM_TOKEN" --name my-gpu
uv run autoggml status
```

The connection is stored in `~/.config/autoggml/team.json` with mode `0600`.

Once the live product adapter is ready, submit a focused patch to compatible hardware:

```bash
uv run autoggml submit patches/my-candidate.patch \
  --title "Reduce expert gather overhead" \
  --backend hip --model deepseek-v4-flash
uv run autoggml status
```

A machine will process one assigned job at a time under its accelerator lease. Today,
`--simulate` can test that lifecycle with a disposable job:

```bash
uv run autoggml worker --once --simulate
```

Do not use a simulated result as research evidence or run simulation against a real
candidate you intend to measure.

Use `pause`, `resume`, and `leave` when availability changes. The coordinator sends
typed jobs and patch bytes, never arbitrary shell commands.

## Work As An Agent

Agents are researchers, not privileged hardware workers. They receive bounded task
packets, work at a pinned commit, and submit a patch plus structured evidence.
Run `uv run autoggml setup` once before starting agent worktrees.

Create a challenge with distinct approaches:

```bash
uv run autoggml agent challenge create \
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
uv run autoggml agent join --name codex-kernel-1 --capability implement
uv run autoggml agent next --json
uv run autoggml agent start <task-id> --json
```

`start` creates an isolated worktree from `work/lucebox`. Submit the result:

```bash
uv run autoggml agent submit <task-id> --patch candidate.patch \
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
mkdir -p ~/.config/autoggml
cp targets.example.toml ~/.config/autoggml/targets.toml
```

Then onboard the host:

```bash
uv run autoggml onboard --target strix-halo
uv run autoggml doctor --target strix-halo \
  --model deepseek-v4-flash --json
```

For a Strix Halo build:

```bash
uv run autoggml setup --target strix-halo --backend hip
```

Remote heavy commands use a nonblocking accelerator lock and cap compilation at four
jobs. `test-drive` remains safe for inventory and simulation; `test-drive --live` is
blocked until the product runtime adapter is ready.

## Keep Lucebox Current

[`sources/lucebox.toml`](sources/lucebox.toml) is the only source ownership manifest. It
contains the Hub commit, tracked branch, layout, build targets, backends, capabilities,
runtime, submodules, and expected GGML vendor provenance.

```bash
uv run autoggml source check --remote
uv run autoggml source status --json
```

A scheduled GitHub workflow checks for Hub movement every Monday. When it moves:

1. Review the new Hub commit.
2. Update the product pin and any changed `VENDOR.md` fields together.
3. Run `autoggml setup`, Ruff, and the test suite.
4. Add runtime or build changes to the manifest, not scattered path checks.

Never update the GGML vendor commit independently of the product commit. Hub's
`VENDOR.md` is the reconstruction contract.

## Research Rules

The intended live loop is simple: propose one change, build the pinned product, measure
baseline and candidate under the same machine lease, apply correctness and quality
gates, and retain only a statistically meaningful improvement. The product HTTP adapter
is the remaining piece needed to reactivate this loop.

- Measure the pinned product, not a nearby standalone fork.
- Keep machine, model, backend, context, workload, and power profile in the evidence key.
- Reject correctness, KL, memory, or workload-constraint regressions.
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
autoggml/source_layout.py  Authoritative paths and capabilities
autoggml/bench/            Measurement, quality, telemetry, statistics
autoggml/loop/             Experiment and verification lifecycle
autoggml/agent_*.py        Agent challenges, evidence, review, credit
autoggml/coordination.py   Shared machine queue
benchmarks/                Fixed workloads and constraints
python_tests/              Unit and end-to-end tests
program.md                 Detailed autonomous-agent rules
ROADMAP.md                 Research ideas and priorities
```

## Development

```bash
uv sync
uv run ruff check .
uv run pytest -q
uv run autoggml reproduce --simulate
```

CUDA builds must remain at `-j4` or lower. Do not run heavy containers and accelerator
builds at the same time on shared machines.

Use `uv run autoggml help` for the full command inventory and
`uv run autoggml <command> --help` for command-specific options.

## License

Apache-2.0.
