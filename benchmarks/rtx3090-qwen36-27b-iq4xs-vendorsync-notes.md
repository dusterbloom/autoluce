# Vendor Sync Experiment Notes

> **2026-07-11 correction:** the candidate was compared with a clean result from an
> earlier machine session. A later same-session control reduced the clean 8K gap from
> 9.1% to 3.36% without a code change. Treat the gains below as hypothesis evidence,
> not a retained delta, until the patch passes a paired clean/candidate ABBA rerun.

## STEP 0 - DONE
- work/lucebox stashed WIP, clean at 5e302cb (HEAD detached)
- stash@{0}: "skip-intermediate-output WIP (regression, saved as patch) — vendor sync"
- DO NOT pop/drop this stash
- patch already saved: patches/lucebox/qwen35-prefill-skip-intermediate-output.patch

## STEP 1 - IN PROGRESS
- Vendor manifest: work/lucebox/server/deps/llama.cpp/VENDOR.md
  - Source repo: https://github.com/Luce-Org/lucebox-ggml (branch luce-dflash)
  - base_commit: 6fbe72d67069136bbd370be703e1d4f441b5e942
  - included PRs (fork-local): #35 (0fe65d9354b7c5da52a7741d2e37ba85f0d0c925), #37 (0699be81480428f01b9b7ac49a09a2d51c77f8df)
  - Vendored paths: LICENSE, common/jinja, common/log.h, common/unicode.*, ggml, gguf-py
- sources/lucebox.toml has same info in [vendor] section, pin_subpath work/lucebox.pin
- work/lucebox-ggml checkout is SHALLOW (only 1 commit, tip 8c146a8 "DeepSeek V4 (#24162)")
  - Had to `git fetch origin 6fbe72d67069136bbd370be703e1d4f441b5e942` to get base_commit + full history (8876 commits, goes back to "Initial release" of llama.cpp - genuine full fork)
- Parent commits of base_commit (39dadf313, 1dfb82304) do NOT exist as hashes in ~/llama.cpp (00f5442cc, build 9970) - fork has diverged/rewritten history from upstream at some point, cannot use hash matching directly.
- NEXT: try matching by content/PR title/date instead of hash. Check ggml version markers in vendored tree (ggml/include/ggml.h GGML_VERSION or similar), check common/build-info.

## STEP 1 - DONE
- Vendor base is NOT ggml-org/llama.cpp directly; it's `Luce-Org/lucebox-ggml` fork (branch luce-dflash), itself
  a genuine full fork of ggml-org/llama.cpp history.
- work/lucebox-ggml checkout was SHALLOW (1 commit) — had to `git fetch --unshallow` from a local remote
  pointed at ~/llama.cpp (file:// transport, since arbitrary-SHA fetch over https is blocked by GitHub) to get
  full connected history.
- True shared ancestor between lucebox-ggml and upstream ggml-org/llama.cpp (~/llama.cpp @ 00f5442cc, build 9970):
  commit fae3a28070fe4026f87bd6a544aba1b2d1896566 "ggml : remove ggml-ext.h (#21869)", dated 2026-04-14, tag b8796.
  That's 1174 commits / ~3 months behind build 9970's HEAD.
- BUT the fork's actual pinned base_commit (VENDOR.md: 6fbe72d67069136bbd370be703e1d4f441b5e942, dated 2026-07-04)
  is only 80 commits past fae3a28070 on the FORK's own branch — i.e. the fork does NOT continuously rebase on
  upstream; it's "old base + sparse cherry-picks + heavy lucebox-local custom kernel work" (TQ3_0 quant,
  Block-Sparse-Attention integration, MoE fusion, turbo-WHT rotation, Gated Delta Net for qwen3.5/3.6).
- Reconstructed the actual vendored tree state = base_commit(6fbe72d) + cherry-pick PR#35 (rope mod-2pi fix,
  rope.cu) + PR#37 (Q4_K/Q5_K MMQ RDNA4 tuning, mmq.cuh) as branch `scratch-recon` in work/lucebox-ggml.
- VERDICT: vendored ggml-cuda is LOCALLY MODIFIED relative to that reconstruction. Diffed reconstruction vs the
  REAL committed vendor tree at work/lucebox/server/deps/llama.cpp/ggml/src/ggml-cuda: exactly 2 files differ
  (everything else pristine, byte-identical):
    - ggml-cuda/gated_delta_net.cu (343-line diff): in-place GDN state write (state_out_d param instead of
      writing into dst's tail region) + WRITE_INTER template bool replacing runtime skip_intermediate branch.
      This is the qwen35 dense decode recovery from pin commit 5e302cb.
    - ggml-cuda/ggml-cuda.cu (31-line diff): adds `ggml_cuda_set_skip_props_check()` extern C hook + a
      thread_local flag to let the CUDA-graph property-change check be skipped once warmup completes (perf
      hack around ggml_cuda_graph_update_required()).
  These are the ONLY local hunks. Saved at benchmarks/rtx3090-qwen36-27b-iq4xs-vendorsync-local-mod-gdn.diff and -local-mod-ggmlcuda.diff.
- CRITICAL FINDING: diffing recon-base -> 00f5442cc over ggml/src/ggml-cuda shows 11 Added / 24 Deleted / 67
  Modified files. The 24 "Deleted" files are FALSE deletions — they are lucebox-local inventions that never
  existed upstream at any point (verified against the true ancestor fae3a28070): fattn-chunked.cu/.cuh,
  fattn-sparse.cu/.cuh, moe-fused.cu/.cuh, tq3-quant.cuh, turbo-wht.cu/.cuh, and 14
  fattn-vec-instance-*-tq3_0*.cu template instances (all TQ3_0-quant-type / sparse-attention / MoE-fusion
  product code). MUST NOT be deleted by any sync patch.
- The 11 "Added" upstream files (allreduce, col2im-1d, fwht.cu(!, unrelated to lucebox's turbo-wht.cu),
  snake.cu, new fattn-tile/mmq template instances) implement NEW ggml ops (GGML_OP_FWHT, SNAKE, COL2IM_1D,
  ALLREDUCE) that Qwen3.6-27B doesn't exercise. Confirmed upstream's ggml-cuda.cu #includes all 4 new headers
  and dispatches the new ops in its switch statement — pulling ggml-cuda.cu wholesale would require these,
  cascading into ggml.h/ggml.c op-enum changes. DECISION: exclude ggml-cuda.cu (and these 4 new op files) from
  the sync entirely — see STEP 2 notes below.

## STEP 2 - sync patch construction
- Excluded from sync entirely (never touch): 24 lucebox-local-invention files (TQ3_0/sparse-attn/moe-fused/
  turbo-wht) that show as upstream "deletions" but are lucebox product code.
- Excluded: gated_delta_net.cu/.cuh — lucebox-owned custom GDN kernel (tree-mode spec-decode rollback,
  in-place state write) is materially different/richer than upstream's simpler GDN; syncing would be a
  functional regression, not a perf win. Confirmed via `git show --stat`: upstream's version is 518 lines
  SMALLER (net deletions) — a simplification incompatible with lucebox's qwen35 usage.
- Excluded: ggml-cuda.cu (the central dispatcher, 3466-line diff = near-total internal restructuring since
  fae3a28070). This file is where lucebox's custom-op dispatch cases (TQ3_0, sparse attn, moe-fused, GDN) AND
  the local skip_props_check hook all live, deeply interleaved with whatever upstream changed in scheduling /
  CUDA-graph handling / cuBLAS refactor entry points. A safe automated 2-way patch is impossible (base state
  doesn't match); a correct 3-way merge is possible in principle but the scale + entanglement with product-
  critical custom ops made it too risky to hand-verify correctness within session scope. Documented as an
  exclusion; this means the "(4) sched changes" and the ggml-cuda.cu portion of "(3) cuBLAS refactor
  74976e1ae" fallback-priority items are NOT captured by this experiment's patch.
- Excluded: CMakeLists.txt (ggml-cuda) — upstream's version drops the explicit
  `template-instances/fattn-vec-instance-tq3_0-tq3_0.cu` build entry (since TQ3_0 doesn't exist upstream);
  taking it wholesale would silently break TQ3_0 quant support compilation. New-file glob patterns
  (`file(GLOB SRCS "template-instances/mmf*.cu")` etc.) are unchanged between recon-base and vendor, so no
  edit is needed for files we do add.
- cuBLAS refactor 74976e1ae (`CUDA: remove -sm row, refactor cuBLAS #24216`) touches: docs/build.md (skip),
  ggml/include/ggml-cuda.h (removes split-buffer-type decl for multi-GPU row-split — lucebox is single-GPU
  target-only, skip), ggml-cuda/convert.cu (INCLUDED, +120/-x), ggml-cuda/ggml-cuda.cu (EXCLUDED, see above),
  ggml-cuda/mmvq.cu (INCLUDED, 3 lines), src/llama-model.cpp (not vendored, skip). So convert.cu + mmvq.cu
  changes ARE captured — partial cuBLAS refactor coverage.
- sched changes 3fc4e1052 / 86b94708f: confirmed 86b94708f is a REVERT of 3fc4e1052 ("sched: reintroduce less
  synchronizations during split compute #20793"), both touching only ggml-backend.cpp (10 lines) and
  ggml-cuda.cu (24 lines). Net effect at 00f5442cc HEAD is ~a no-op relative to pre-3fc4e1052 baseline (tried
  and reverted) for ggml-cuda.cu; ggml-backend.cpp's residual (if any) is captured separately below.
- ggml-backend.cpp: confirmed PRISTINE (byte-identical, no lucebox local mods) between recon-base and actual
  vendor tree. INCLUDED in a second smaller patch (not part of the ggml-cuda-only sync_v1.patch) since it's
  ggml/src/ggml-backend.cpp, not under ggml-cuda/ — 51-line diff, backend-agnostic scheduler, low risk.
  DECISION: apply separately, see below.
- FINAL SYNC SCOPE (sync_v1.patch): 63 files, all pristine (recon-base == actual vendor byte-for-byte before
  patching), 2-way diff recon-base->00f5442cc. Covers: fattn/attention (fattn.cu/.cuh, fattn-common.cuh,
  fattn-mma-f16.cuh + 10 template instances, fattn-tile.cu/.cuh, fattn-vec.cuh, fattn-wmma-f16.cu), mmq/mmvq
  (mmq.cu/.cuh, mmvq.cu/.cuh, mmf.cuh, mmvf.cu), partial cuBLAS refactor (convert.cu, mmvq.cu already listed),
  plus rope.cu, cpy.cu, set-rows.cu/.cuh, quantize.cu/.cuh, unary.cu/.cuh, common.cuh, mma.cuh, vecdotq.cuh,
  and misc ops (argsort, binbcast, concat, conv-transpose-1d, dequantize, getrows, im2col, mean, norm,
  out-prod, scale, softcap, ssm-conv/scan, sumrows, top-k, topk-moe, vendors/hip.h, vendors/musa.h).
  git diff --shortstat: 63 files, ~4664 insertions / ~5000 deletions (from the M-only 67-file stat, minus the
  3 excluded M-files' share).
- Applied cleanly: `git apply --directory=server/deps/llama.cpp --check` -> no errors (0 conflicts, only 2
  cosmetic "new blank line at EOF" whitespace warnings). Reverse-check also passes.
- Saved patch file: patches/llama.cpp/vendor-ggml-cuda-sync-9970.patch (to be finalized after build succeeds).

## STEP 2 - REVISED: ggml-backend.cpp reversal
- On inspection, ggml-backend.cpp's diff (recon->00f5442cc) turned out to contain a SECOND false-deletion trap:
  it removes `ggml_backend_tensor_check_raw_span()`, which is ALSO a lucebox-local addition (fork commit
  9b3b3b612 "ggml-backend: abort loudly when a raw tensor span crosses a stride gap"), confirmed absent at the
  true ancestor fae3a28070. Blindly taking this diff would delete a lucebox safety assert.
- The diff's other hunks (get_tensor_2d_async/get_tensor_2d bug fix; ggml_op_desc cosmetic debug change;
  `graph->uid = ggml_graph_next_uid()` calls for CUDA-graph key stability — directly relevant to the stashed
  qwen35-prefill-cuda-graph-keys.patch WIP!) require new core infra (`uid` field on `ggml_cgraph`,
  `ggml_graph_next_uid()`) that does NOT exist in the vendor's ggml.h/ggml.c (confirmed via grep — zero hits).
  Pulling that in cascades back into the ggml.h/ggml.c core-file sync this experiment already excluded.
- DECISION: exclude ggml-backend.cpp from the sync entirely. The one safe+valuable hunk (graph uid /
  CUDA-graph key stability) is out of reach without reopening the core-header exclusion (too large a scope
  increase for this experiment); the other hunks are low-value (2D batched tensor get/set bug fix, unused by
  our single-request greedy-decode workload) or purely cosmetic.
- FINAL sync scope unchanged from above: sync_v1.patch, 63 files, ggml-cuda/ subsystem only.

## STEP 2 - REVISED (final): switched from 2-way patch to proper 3-way merge
- The v1 2-way patch (recon-base -> 00f5442cc, blind `git diff`) was WRONG: it silently deleted lucebox-local
  content baked into files that ARE pristine-vs-reconstruction but carry lucebox-local additions from earlier
  in fork history (before base_commit). Discovered via BUILD FAILURE (16 undefined-symbol errors): common.cuh
  lost `stat_total/replay/capture/eager` CUDA-graph counters + `luce_q8_memo` struct/vector; mmvq.cuh lost
  `MMVQ_MAX_MOE_BATCH_SIZE`; set-rows.cu/.cuh lost `ggml_cuda_op_set_rows_dual` (dual SET_ROWS fusion,
  lucebox-local perf feature). These were all "false deletions" of the same kind as the TQ3/sparse-attn files
  in STEP1, just hiding inside otherwise-pristine files rather than being whole extra files.
- REVERTED v1 patch, redid the sync as a proper git 3-way merge in work/lucebox-ggml:
  `git checkout scratch-recon && git merge --no-commit --no-ff 00f5442cc` (merge-base = fae3a28070, confirmed).
  This correctly auto-merges non-overlapping regions and only flags TRUE overlaps as conflicts — the false-
  deletion problem vanishes because 3-way merge treats "ours added X, theirs never touched that region" as a
  clean keep, not a delete.
- Full-tree merge produced 24 conflicts (out of 632 changed files) across the whole lucebox-ggml fork (incl.
  files outside our vendor scope: src/llama-arch.cpp, gguf-py/gguf/constants.py, etc. — new-model-arch stuff,
  reverted to recon-base, out of scope for this experiment).
- Vendor/ggml-cuda-relevant conflicts (10 files) resolved by hand, one at a time:
  - common.cuh (1 block): union both sides' new `ggml_cuda_graph` fields (kept lucebox's stat_total/replay/
    capture/eager counters AND took upstream's uid/last_used_time). luce_q8_memo auto-merged cleanly already.
  - convert.cu (2 blocks): union both dequant dispatch cases (kept GGML_TYPE_TQ3_0 case, added upstream's new
    GGML_TYPE_Q1_0 case — confirmed GGML_TYPE_Q1_0 enum already exists in vendor's ggml.h).
  - cpy.cu (1 block): union — kept lucebox's 4 TQ3_0 cpy kernels, added upstream's new
    `ggml_cuda_cpy_as_memcpy_2d` 2D-strided-copy fast path (confirmed it's wired into ggml_cuda_cpy() below).
  - fattn.cu (2 blocks): took upstream's `ggml_cuda_fattn_kv_type_supported()` refactor (DRY'd duplicate
    switch statements) but ADDED lucebox's `GGML_TYPE_TQ3_0` case into the new helper so TQ3_0 KV-cache
    attention support is preserved; took upstream's `Q->ne[0] != 192` MMA-path fix for the vector-kernel
    eligibility check (strict correctness improvement, no lucebox conflict).
  - mmq.cuh (2 blocks): kept lucebox's superset RDNA tile-size override + amd_mfma_available gating (strict
    superset of upstream's condition; upstream's own file uses amd_mfma_available elsewhere too, just not in
    this specific gate — not a real removal, safe to keep ours).
  - ssm-conv.cu (2 blocks): union — kept lucebox's tree-mode (parent_ids, spec-decode rollback) SSM-conv
    dispatch AND added upstream's new fused-bias support for the non-tree path; added an explicit
    `GGML_ASSERT(!fuse_bias && "tree-mode ssm_conv does not support fused bias")` guard since the two features
    were never combined upstream or in lucebox and combining them silently would be unverified.
  - vendors/hip.h (1 block): kept lucebox's variadic `__shfl_*_sync` macro fix (a real correctness fix for
    ROCm compat, fork commit "fix(hip): make shfl_*_sync macros variadic + add cudaMallocAsync/FreeAsync
    aliases") — strict superset of upstream's non-variadic version. HIP-only file, doesn't affect the CUDA
    SM86 build either way.
  - mmvq.cu/.cuh (8 conflict blocks, EXCLUDED): the `mul_mat_vec_q` and `mul_mat_vec_q_moe` kernel templates
    were restructured incompatibly on both sides — lucebox added per-column `ids_tokenwise_samples` /
    `channel_xs[]` array handling for MoE multi-token dispatch AND fused-bias/gate support in the MoE kernel
    (`mul_mat_vec_q_moe` gained a `fusion` param + `has_fusion` template bool that upstream's signature does
    NOT have at all); upstream independently added a `ggml_cuda_pdl_sync()` prefetch call and a new
    `ggml_cuda_kernel_launch()` wrapper abstraction replacing raw `<<<>>>` launch syntax. Hand-merging two
    incompatible restructurings of the SAME hot kernel risked a silent correctness bug (wrong results at
    depth, not a crash) — unacceptable given the STEP4 output-correctness requirement. REVERTED to
    scratch-recon (unchanged, i.e. excluded from sync).
  - gated_delta_net.cu/.cuh (10 conflict blocks, EXCLUDED, confirms original STEP1/2 finding): upstream's GDN
    kernel is a materially simpler/different implementation lacking lucebox's TREE_MODE spec-decode rollback
    and in-place state write. REVERTED to scratch-recon (i.e., the actual vendored tree's local hunk from pin
    commit 5e302cb is preserved untouched, since this file's sync diff is now empty).
  - ggml-cuda.cu (6 conflict blocks, EXCLUDED, confirms original STEP1/2 finding): the largest single conflict
    block spans 374 lines of an op-fusion dispatch/planner loop (topk-moe fusion detection, rope+set_rows
    fusion) interleaved with 5 other conflicts across a 5364-line file. Confirmed genuinely too entangled for
    safe hand-merging within session scope. REVERTED to scratch-recon (i.e., the vendor tree's local hunk —
    `ggml_cuda_set_skip_props_check()` CUDA-graph property-check skip — is preserved untouched).
  - CMakeLists.txt: reverted to scratch-recon (unchanged), confirming original STEP2 decision (upstream drops
    the explicit tq3_0-tq3_0 fattn-vec-instance build entry).
  - Core files (ggml.c, ggml.h, ggml-quants.c, ggml-rpc.h): reverted to scratch-recon (unchanged) — out of
    declared scope (CUDA-kernel-layer only), confirms original STEP2 scope decision.
- Committed the resolved merge in work/lucebox-ggml as `96cfd6544` on branch `scratch-recon` (local scratch
  repo only, never pushed anywhere).
- FINAL PATCH (v2): `git diff ab50058c8 96cfd6544 -- <59 files>` = 59 files changed, 2729 insertions(+),
  1345 deletions(-). File count dropped from v1's 63 to 59 because mmq.cu's mmvq companion mmvq.cu/.cuh (2),
  gated_delta_net.cu/.cuh (2) are now correctly EXCLUDED (zero diff, matching scratch-recon exactly) —
  net -4 files vs v1's file *list* (v1 never actually included gated_delta_net anyway; the count difference
  vs v1 is mmvq.cu/.cuh dropping out, plus CMakeLists.txt/generate_cu_files.py bookkeeping).
- Applied cleanly to work/lucebox: `git apply --directory=server/deps/llama.cpp` — 0 errors, 1 cosmetic
  whitespace warning (blank line at EOF). Reverse-check passes. Saved as
  patches/llama.cpp/vendor-ggml-cuda-sync-9970.patch (7536 lines).

## STEP 2 - iterate-until-builds fixes
- First build attempt (v2 patch) FAILED to link: `undefined reference to
  ggml_cuda_flash_attn_ext_tile_case<320,256>` and `<192,128>`. Root cause: synced fattn-tile.cuh declares these
  new head-dim template instantiations but the corresponding .cu instance files (new upstream files, "A" list)
  weren't added. FIX: added the 2 self-contained instance files
  (template-instances/fattn-tile-instance-dkq192-dv128.cu, -dkq320-dv256.cu, ~5 lines each, just
  `#include "../fattn-tile.cuh"` + `DECL_FATTN_TILE_CASE(...)`) — picked up automatically by CMake's existing
  glob pattern, no CMakeLists.txt edit needed. Build succeeded (v3/v4 patch, 61 files).
- Build SUCCEEDED but a runtime crash surfaced on the very first inference request:
  `GGML_ASSERT(!fuse_bias || fuse_silu) failed` at ssm-conv.cu:273, aborting dflash_server mid-request.
  ROOT CAUSE (real bug I introduced during conflict resolution, not a pre-existing issue): my ssm-conv.cu
  3-way-merge resolution inserted upstream's new `bias_add_node` parameter into the MIDDLE of the function
  signature (`ctx, dst, bias_add_node, silu_dst`) rather than at the end. The vendor tree's UNSYNCED
  ggml-cuda.cu (excluded from sync, see above) has two existing POSITIONAL call sites,
  `ggml_cuda_op_ssm_conv(ctx, dst)` and `ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i+1])`. The
  second call's 3rd positional arg was ALWAYS meant to bind to `silu_dst` (fusing SSM-conv with a downstream
  SiLU op) — inserting the new param before it silently REBOUND that 3rd argument to `bias_add_node` instead,
  making `fuse_bias=true` with `fuse_silu=false` on every call that used to fuse SiLU, tripping the (pre-
  existing, correct) `!fuse_bias || fuse_silu` invariant. A pure compile-time typecheck could not catch this
  (both params are `ggml_tensor *`) — only found via runtime crash on first real inference request. FIX:
  reordered the new `bias_add_node` parameter to the END of the signature in both ssm-conv.cu and
  ssm-conv.cuh (`ctx, dst, silu_dst = nullptr, bias_add_node = nullptr`), restoring positional compatibility
  with the two unsynced call sites in ggml-cuda.cu. Verified via `grep -rn ggml_cuda_op_ssm_conv` across the
  whole work/lucebox tree that these are the ONLY 2 call sites — no other callers to check. This is a strong
  argument for "new parameters go at the end, never inserted mid-signature" as a general vendor-sync-safety
  rule when the calling file itself is excluded from the sync.
- Rebuilt clean from patch (v4, final): SUCCESS. `dflash_server`/`test_dflash`/`test_deepseek4_unit` all link.

## STEP 3 - build + smoke test results
- Build: cmake Release, `-DDFLASH27B_GPU_BACKEND=cuda`, build dir `build-cuda-sm86` (AUTOLUCE_BUILD_SUBDIR),
  targets `dflash_server test_dflash test_deepseek4_unit`. Build succeeded via `autoluce.prepare.build_lucebox()`.
- `ggml commit: 5e302cb-dirty` reported by cmake configure (expected — vendor patch applied on top of pin).
- Binary md5 (final, patch v4 applied):
    dflash_server:        5467429a5ed81481a9f029fc4026db24
    test_dflash:           fafaa19306ae41128af39ae0e03f6cf0
    test_deepseek4_unit:   f61bc247c2e0d00f31ab3695effb3db6
  NOTE: dflash_server/test_dflash/test_deepseek4_unit executable md5s are IDENTICAL between the pre-fix (v2,
  crashing) and post-fix (v4) builds — this is because ggml-cuda is a SHARED LIBRARY
  (libggml-cuda.so.0.9.11) the executables dynamically link against; the executables' own bytes don't encode
  the .so's content. The .so DOES differ: md5 e487fd24af3b598cae9452d5e8370061 (v4, verified via
  `strings | grep "tree-mode ssm_conv does not support fused bias"` — the new assert string is present,
  confirming the fix is compiled in).
- Smoke test: `test_deepseek4_unit` run to completion, exit code 0, all sub-tests report `ok`/`OK`, zero
  fail/assert hits in the log (includes CUDA-exercising tests: `test_hc_pre_kernel_gpu`, graph-reuse
  microbenches). `test_dflash` requires a draft-model + prompt-ids binary argument set not relevant to this
  target-only IQ4_XS benchmark; not run (out of scope, target-only workflow per task background).

## STEP 4 - measurement
- Reused `/tmp/parity_iq4xs/driver.py`'s exact `run_dflash_arm()` (same MODEL_PATH, SERVER_CTX=8448,
  f16/f16 KV, temp 0 / top_k 1 / seed 42 / n_predict 128, 1 warmup + 5 measured, salted distinct prompts,
  `prefix_cache: {scope: off}`, `--prefix-cache-slots 0 --prefill-cache-slots 0`) via a new driver script
  `benchmarks/rtx3090-qwen36-27b-iq4xs-vendorsync-measure.py`. GPU verified idle before start (683 MiB / 0-1% util, no stray
  servers). Results written to `/tmp/vendor_sync/results.json`, permanent copy at `benchmarks/rtx3090-qwen36-27b-iq4xs-vendorsync-raw.json`.
- prefix_len evidence: 24/24 requests logged `prefix_len=0` in the dflash server's own stdout — non-caching
  proven structurally (matches the parity contract).
- Determinism/correctness: the ORIGINAL clean-baseline `/tmp/parity_iq4xs/results.json` does NOT contain
  per-rep response text (its `driver.py` was extended to capture `"text"` AFTER that baseline run completed —
  confirmed by file mtimes: results.json older than driver.py). Byte-exact baseline text diff is therefore not
  possible; falling back per task instructions to self-consistency + logging actual text. Cross-rep text
  differs WITHIN a cell by design (each rep's prompt is independently salted, per the anti-caching contract —
  `build_prompt(depth, rep_label)` embeds `rep={rep_label}` in the prompt text itself, so different reps see
  genuinely different input and legitimately produce different output). Ran a dedicated SAME-PROMPT
  determinism side-check instead (2x identical salted-prompt calls, temp 0 / seed 42, outside the timed
  cells): byte-identical output both times (see `benchmarks/rtx3090-qwen36-27b-iq4xs-vendorsync-determinism.json`). All 4 depths'
  responses are coherent, on-topic, non-garbled prose (no signs of numeric/kernel corruption) — logged in
  `benchmarks/rtx3090-qwen36-27b-iq4xs-vendorsync-raw.json` `dflash_arm.cells.<depth>.raw_reps[*].text`.

## STEP 5 - verdict table (median prefill/decode tok/s, per cell)

| Depth | vendor-sync dflash | clean-dflash baseline | Δ vs clean-dflash | llama 9970 baseline | Δ vs llama 9970 | orig dflash-vs-llama gap | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 1K prefill | 1426.5 | 1357.0 | **+5.1%** | 1360.5 | +4.9% | -0.3% | improved (now ahead of llama) |
| 1K decode  | 46.3   | 45.8   | +1.1%  (noise) | 46.0   | +0.7% (noise) | -0.4% | unchanged (was already parity) |
| 2K prefill | 1447.4 | 1388.6 | **+4.2%** | 1450.6 | -0.2% (noise) | -4.3% | **gap closed** |
| 2K decode  | 45.6   | 45.5   | +0.2% (noise) | 47.3   | -3.6%  | -3.8% | unchanged (still trailing llama, within noise of orig gap) |
| 4K prefill | 1463.7 | 1379.7 | **+6.1%** | 1460.3 | +0.2% (noise) | -5.5% | **gap closed** |
| 4K decode  | 45.2   | 43.7   | **+3.4%** | 48.2   | -6.1%  | -9.3% | **narrowed** (gap -9.3% -> -6.1%) |
| 8K prefill | 1441.8 | 1337.7 | **+7.8%** | 1471.0 | -2.0%  | -9.1% | **narrowed sharply** (gap -9.1% -> -2.0%) |
| 8K decode  | 43.8   | 43.2   | +1.4%  (noise) | 47.4   | -7.7%  | -8.9% | unchanged (gap -8.9% -> -7.7%, within noise) |

Noise band: ±2.5% (per task-stated llama session-to-session sanity rerun: +2.5%/+2.6%).

Per-cell plain read:
- **1K**: prefill improved +5.1% (real, above noise) vs clean-dflash; dflash now *ahead* of llama by +4.9%
  (was -0.3%). Decode flat/noise both ways.
- **2K**: prefill **gap closed** — dflash went from -4.3% behind llama to -0.2% (noise-indistinguishable from
  parity). Decode gap unchanged (-3.8% -> -3.6%, no real movement).
- **4K**: prefill **gap closed** (-5.5% -> +0.2%). Decode gap **narrowed** materially (-9.3% -> -6.1%), a real
  +3.4% absolute decode improvement vs clean-dflash baseline, but still trailing llama by more than noise.
- **8K**: prefill gap **narrowed sharply** (-9.1% -> -2.0%, a +7.8% absolute prefill improvement vs clean-
  dflash — the largest single-cell win, right at the noise boundary vs llama). Decode gap narrowed slightly
  (-8.9% -> -7.7%) but the *absolute* decode-vs-clean-dflash improvement (+1.4%) is inside the noise band, so
  this narrowing is not conclusively attributable to the sync (llama's own baseline could itself be ±2.5%
  noisy at this cell — no fresh llama sanity rerun was done at 8K to confirm).
- **Nothing regressed beyond noise** on any cell, any axis.

### Overall shape
The sync closes essentially ALL of the depth-scaling **prefill** deficit (which is exactly what the excluded
subsystems predict it should NOT be able to do by itself if the deficit were sched/dispatcher-driven — the
fact that prefill closes almost completely while ggml-cuda.cu/gated_delta_net.cu/mmvq.cu were EXCLUDED points
to the fix living in the SYNCED fattn/mmq/convert/cpy/mma subsystems, not in scheduling). **Decode** only
narrows partially (best at 4K: -9.3% -> -6.1%) and the 1K/8K decode movements are inside the noise band —
decode's residual deficit likely lives partly in the excluded ggml-cuda.cu (op-fusion dispatch, CUDA-graph
property-check skip) and/or the excluded mmvq.cu (fused-bias MoE MMVQ kernel), consistent with decode being a
smaller-batch, more dispatch/kernel-launch-overhead-sensitive regime than prefill.

## Hypothesis verdict: **PARTIALLY CONFIRMED**
The "post-pin upstream CUDA work explains the depth-scaling deficit" hypothesis is confirmed for **prefill**
(deficit -4.3%/-5.5%/-9.1% at 2K/4K/8K collapses to -0.2%/+0.2%/-2.0%, all within or at the edge of the noise
band — i.e. prefill parity is restored) but only **partially** confirmed for **decode** (deficit narrows at
4K, -9.3% -> -6.1%, a real but incomplete recovery; 1K/8K decode movements are noise-level). Since the fix
required syncing fattn/mmq/mmvq-adjacent files (fattn.cu, fattn-common.cuh, fattn-mma-f16.cuh, mma.cuh,
convert.cu, cpy.cu, mmq.cu/cuh, mmf.cuh, mmvf.cu, rope.cu, set-rows.cu, quantize.cu, unary.cu, and friends) —
NOT ggml-cuda.cu's dispatcher/scheduler or the mmvq.cu MoE-batch kernel (both excluded as too entangled to
safely sync) — the evidence points to the residual DECODE gap living specifically in those excluded files:
either the op-fusion/CUDA-graph dispatch logic in ggml-cuda.cu (sched-adjacent, matches the original "sched
split-compute" hypothesis thread) or the mul_mat_vec_q/mul_mat_vec_q_moe kernel restructuring in mmvq.cu
(matches the "cuBLAS refactor" and general MMVQ-path hypothesis thread). A follow-up experiment isolating
JUST ggml-cuda.cu (via careful hand-port of only its dispatch/graph-handling diff, preserving all lucebox
custom-op cases) or JUST mmvq.cu (via careful hand-port preserving the fused-bias MoE kernel) would
disambiguate which excluded subsystem carries the remaining ~6% decode deficit at mid-depth.

## STEP 6 - final cleanup
- Reverse-applied `patches/llama.cpp/vendor-ggml-cuda-sync-9970.patch` via `git checkout -- server/deps/llama.cpp`
  (cleaner than `git apply --reverse` after a background-command interruption left the tree mid-revert once —
  `git checkout` is idempotent/atomic for tracked-file restoration, safer here). Removed the 2 new untracked
  template-instance files by hand (`git checkout` does not remove untracked additions).
- Final work/lucebox state: `git status` clean (HEAD detached at `5e302cbb483819cd21e72f5dd8becaa609eca8cf`,
  matches `work/lucebox.pin` exactly), only `build-cuda-sm86/` untracked (the build directory, expected/
  gitignored-equivalent artifact), stash `stash@{0}` ("skip-intermediate-output WIP...") intact and untouched
  (never popped/dropped per instructions).
- Patch artifact: `patches/llama.cpp/vendor-ggml-cuda-sync-9970.patch` (7558 lines, 61 files, +2739/-1345),
  untracked in the main repo (no commit made, per task instructions).
- work/lucebox-ggml scratch repo (the intermediate 3-way-merge workspace) still has its `scratch-recon` branch
  and the `fe95239de` merge commit — this is a local-only scratch checkout, not part of the pinned product
  checkout tracked by `sources/lucebox.toml`, and was not reset (harmless working state, purely an
  investigation/derivation artifact for this experiment, not touched by autoluce tooling).
