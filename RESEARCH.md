# Research: On-Demand Pre-Run GGML Graph Optimization

This document summarizes the prior art found on GitHub, Hugging Face, and arXiv relevant to building a GGML graph optimizer that performs on-demand pre-run optimization.

## 1. What GGML Is and Where Optimization Currently Happens

GGML is a low-level tensor library used by `llama.cpp`, `whisper.cpp`, and related projects. A model forward pass is represented as a `ggml_cgraph` — a directed acyclic graph of `ggml_tensor` nodes. Optimization in the current ecosystem happens at several layers, but **not** through a general, pluggable, on-demand graph optimizer.

### 1.1 Core Graph Infrastructure

- Repository: `ggml-org/ggml`
- Key files: `src/ggml.c`, `src/ggml-backend.cpp`, `src/ggml-alloc.c`, `include/ggml.h`
- Concepts:
  - `ggml_cgraph`: list of nodes in topological order + leaf tensors.
  - `ggml_build_forward_expand`: builds the forward graph from output tensors.
  - `ggml_backend_sched`: splits one graph into multiple backend-specific subgraphs, inserts copies for cross-backend transfers, and allocates memory via `ggml_gallocr`.
  - `ggml_gallocr`: pre-computes tensor memory offsets using a reference-counting lifetime analysis before the graph runs.

The scheduler does **backend placement** and **memory planning**, but it does not rewrite the graph to fuse, eliminate, or reorder operations beyond what is implicit in backend support checks.

### 1.2 Existing Graph-Level Optimizations in llama.cpp

From the source and discussions, llama.cpp applies architecture-specific graph optimizations in the model-building code (`src/llama.cpp` and per-architecture files):

- **Operator fusion**: attention blocks, FFN blocks, vision encoder blocks, MoE blocks are fused into single GGML graph dispatches where a backend kernel exists (e.g., `ggml_flash_attn_ext`, fused MLP kernels).
- **Backend partitioning**: `ggml_backend_sched_split_graph` assigns nodes to CPU/CUDA/Metal/Vulkan/etc. based on `supports_op` and buffer placement.
- **Cross-backend copies**: inserted automatically where tensors move between backends.
- **Quantization-aware execution**: quantized weights are used directly in `MUL_MAT` without dequantizing to FP32.
- **Memory pooling**: best-fit allocator, bounded pool retention, mmap-backed weights.
- **Warmup**: `llama.cpp` runs a tiny forward pass at startup to warm kernels and memory pools.

These optimizations are largely **static** (compiled into the architecture implementation) or **heuristic** (scheduler rules), not an on-demand pass that inspects the actual runtime graph and rewrites it.

## 2. Relevant Projects and Implementations

### 2.1 TensorSharp (.NET GGUF engine)

- URL: https://github.com/zhongkaifu/TensorSharp
- Relevance: Shows the performance gains of **fused whole-layer and whole-model GGML graphs**.
- Techniques:
  - Fused GPU decode/prefill for Gemma 4 (~2.6× speedup by reducing CPU↔GPU round-trips).
  - Fused Qwen attention + FFN blocks.
  - Batched/paged forward passes.
  - Kernel warmup at startup.
- Lesson: Fusing at graph-build time can yield large wins, but it is hand-written per architecture. A general optimizer could discover and apply similar fusions automatically.

### 2.2 VITRIOL (llama.cpp fork for host-memory MoE)

- URL: https://github.com/Randozart/VITRIOL
- Relevance: Demonstrates a **custom backend buffer type** and scheduler integration.
- Techniques:
  - Registers host memory as a CUDA DMA buffer (`cudaHostRegister`).
  - Uses `is_host=true` so the scheduler routes `MUL_MAT_ID` to CUDA while weights stay in system RAM.
  - Custom scheduler-aware memory tiering.
- Lesson: The scheduler can be influenced through buffer-type metadata; an optimizer could decide memory tiering per tensor on demand.

### 2.3 Paged KV Cache Prototype (llama.cpp discussion #21961)

- URL: https://github.com/ggml-org/llama.cpp/discussions/21961
- Relevance: A runtime system-level optimization that changes how KV tensors are allocated and scheduled.
- Lesson: Pre-run graph optimization could include KV-cache layout planning (block size, prefix sharing, quantization) based on the actual sequence mix.

### 2.4 TurboQuant / Ollama-TurboQuant-Integration

- URL: https://github.com/Lucien2468/Ollama-TurboQuant-Integration
- Relevance: New quantized types and kernel-level optimizations. The `turboquant_plus` investigation notes mention **graph optimizer interference** with rotation ops.
- Lesson: Graph optimizers must be correctness-preserving and aware of precision-sensitive operations.

### 2.5 ik_llama.cpp / Chimere

- URL: https://github.com/ikawrakow/ik_llama.cpp, https://github.com/AIdevsmartdata/chimere
- Relevance: Experimental forks with aggressive fusion, speculative decoding (DFlash), and recurrent-state handling.
- Lesson: There is active experimentation around graph structure; an optimizer harness could compare these forks or upstream against custom passes.

## 3. Academic Prior Art

### 3.1 Efficient LLM Inference Surveys

- "A Survey on Efficient LLM Inference" (arXiv:2401.08092, Xu et al.)
- Covers quantization, pruning, speculative decoding, FlashAttention/FlashDecoding, memory planning, and scheduling.
- Relevant to graph optimizer design: **kernel optimization**, **attention-specific fusion**, and **memory-efficient scheduling**.

### 3.2 FlashAttention / FlashDecoding

- FlashAttention-2 (Dao, 2023) and Flash-Decoding (Dao et al.) fuse the attention Q×K^T, mask, softmax, ×V sequence into a single memory-efficient kernel.
- FlashDecoding+ further optimizes decode-phase softmax and flat GEMM.
- Implication: A graph optimizer should map recognized attention sub-graphs to fused kernels when shapes and backend support allow it.

### 3.3 MoE On-Demand / Pre-gated MoE

- "Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference" (ISCA 2024, Hwang et al.)
- Code: https://github.com/ranggihwang/Pregated_MoE
- Compares:
  - **MoE-OnDemand**: fetch experts to GPU only when selected.
  - **MoE-Prefetch**: pre-transfer experts.
  - **Pre-gated MoE**: decouple gate selection from expert execution to hide migration latency.
- Relevance: The phrase "on-demand" appears here in the context of expert loading, not graph optimization, but the idea of **lazy, input-dependent optimization** is analogous.

### 3.4 DAOP — Data-Aware Offloading and Predictive Pre-Calculation

- "Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference" (DATE 2025)
- URL: https://arxiv.org/html/2501.10375v1
- Predicts which experts will be needed and pre-calculates them on the CPU one layer ahead, parallelizing CPU/GPU execution.
- Lesson: On-demand optimization can use prediction to overlap data movement and computation.

### 3.5 Agent.xpu — Pre-Compile and Warmup

- "Agent.xpu: Efficient Scheduling of Agentic LLM Workloads" (arXiv:2506.24045)
- Offline model compilation + warmup builds a heterogeneous execution graph (HEG) with performance/power annotations.
- Online scheduler dispatches kernels to NPU/iGPU.
- Lesson: A pre-run optimizer can annotate the graph with cost estimates and let a runtime scheduler use them.

### 3.6 On-Device Transformer Inference

- "Sometimes Painful but Promising: Feasibility and Trade-offs of On-Device Language Model Inference" (arXiv:2503.09114)
- Analyzes CPU/GPU governor and thread settings for edge inference.
- Lesson: For on-demand optimization, hardware state (frequency, thread count) is part of the optimization space.

## 4. Gaps and Opportunities

Current GGML/llama.cpp has **no centralized, on-demand pre-run graph optimizer**. The following are missing or done manually per architecture:

| Capability | Current State | Opportunity |
|---|---|---|
| Operator fusion | Hand-fused per model/arch | Automatic pattern matcher that discovers fusible sub-graphs at runtime |
| Backend placement | Scheduler heuristic | Cost-model-driven placement with measured latencies |
| Memory layout | Fixed by model/build | On-demand layout selection (NHWC/NCHW, quantized intermediate formats) |
| Precision selection | Mostly F32 intermediates | Dynamic F16/BF16 intermediate casting where correctness allows |
| KV-cache layout | Static per context | Optimize block size, sharing, and compression per batch |
| Speculative decoding graph | Hand-built | Automated draft/verify graph construction |
| Constant folding / dead-code elimination | Limited | General graph-level DCE and folding passes |
| Dynamic batching | Server-level | Graph-level batching/fusion based on actual batch shape |

The proposed autoresearch targets this gap: build a **modular graph optimizer** that runs after `ggml_build_forward_expand` and before `ggml_backend_sched_alloc_graph`, rewrites the `ggml_cgraph`, and is evaluated by latency, memory, and correctness against unoptimized baseline.

## 5. Key Design Questions for the Autoresearch

1. **Representation**: Operate directly on `ggml_cgraph` in C/C++, or on a serialized/Python mirror?
2. **Correctness**: How to verify that an optimized graph produces identical (or within tolerance) outputs?
3. **Cost model**: Use analytical cost, profiled micro-benchmarks, or runtime measurements?
4. **On-demand trigger**: Optimize once at model load, once per new input shape, or continuously?
5. **Backend awareness**: How does the optimizer query backend capabilities (`supports_op`, kernel perf)?
6. **Pass ordering**: Fusion → DCE → layout → placement → memory planning?
7. **Search strategy**: Fixed passes, greedy heuristics, genetic search, or learned policy?

## 6. References

- GGML: https://github.com/ggml-org/ggml
- llama.cpp: https://github.com/ggml-org/llama.cpp
- whisper.cpp: https://github.com/ggml-org/whisper.cpp
- TensorSharp: https://github.com/zhongkaifu/TensorSharp
- VITRIOL: https://github.com/Randozart/VITRIOL
- Paged KV prototype: https://github.com/ggml-org/llama.cpp/discussions/21961
- TurboQuant integration: https://github.com/Lucien2468/Ollama-TurboQuant-Integration
- Pre-gated MoE: https://github.com/ranggihwang/Pregated_MoE
- Efficient LLM Inference survey: https://arxiv.org/abs/2401.08092
- Pre-gated MoE paper: https://arxiv.org/abs/2308.12066
- DAOP paper: https://arxiv.org/abs/2501.10375
- Agent.xpu paper: https://arxiv.org/abs/2506.24045
- On-device LLM inference paper: https://arxiv.org/abs/2503.09114
