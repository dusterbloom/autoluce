#!/usr/bin/env python3
"""Measure the vendor-synced dflash arm only, reusing driver.py's exact
run_dflash_arm() contract from /tmp/parity_iq4xs/. Writes results to
/tmp/vendor_sync/results.json."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/tmp/parity_iq4xs")
import driver  # noqa: E402

RESULTS_PATH = Path("/tmp/vendor_sync/results.json")


def main() -> None:
    print("=" * 80)
    print("VENDOR-SYNC MEASUREMENT: dflash arm (full 4 cells), vendor-ggml-cuda-sync-9970.patch applied")
    print("=" * 80)

    nvidia_smi_before = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"],
        capture_output=True, text=True,
    ).stdout
    print(f"nvidia-smi before run:\n{nvidia_smi_before}")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    dflash_result = driver.run_dflash_arm()

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    nvidia_smi_after = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"],
        capture_output=True, text=True,
    ).stdout

    # ---- Extract prefix_len evidence from the dflash server log ----
    log_dir = driver.LOG_DIR
    dflash_log_files = sorted(log_dir.glob("dflash-*.log"), key=lambda p: p.stat().st_mtime)
    latest_dflash_log = dflash_log_files[-1]
    pattern = re.compile(r"in=(\d+) effective_in=(\d+) out=(\d+).*prefix_len=(\d+)")
    server_log_evidence = []
    for line in latest_dflash_log.read_text().splitlines():
        if "[server] chat DONE" in line:
            m = pattern.search(line)
            if m:
                server_log_evidence.append({
                    "in": int(m.group(1)), "effective_in": int(m.group(2)),
                    "out": int(m.group(3)), "prefix_len": int(m.group(4)),
                })
    all_zero = all(e["prefix_len"] == 0 for e in server_log_evidence)
    print(f"prefix_len evidence: {len(server_log_evidence)} requests, all prefix_len==0: {all_zero}")
    dflash_result["server_log_prefix_len_evidence"] = server_log_evidence
    dflash_result["server_log_prefix_len_all_zero"] = all_zero

    summary = {}
    for depth in driver.DEPTHS:
        summary[str(depth)] = driver.summarize(dflash_result["cells"][str(depth)]["raw_reps"])

    output = {
        "provenance": {
            "started_at": started_at,
            "finished_at": finished_at,
            "model_path": str(driver.MODEL_PATH),
            "model_sha256": driver.MODEL_SHA256_KNOWN,
            "server_ctx": driver.SERVER_CTX,
            "cache_type_k": driver.CACHE_TYPE_K,
            "cache_type_v": driver.CACHE_TYPE_V,
            "sampling": {
                "temperature": driver.TEMPERATURE, "top_k": driver.TOP_K,
                "seed": driver.SEED, "n_predict": driver.N_PREDICT,
            },
            "depths": driver.DEPTHS,
            "repetitions": {"warmup": driver.N_WARMUP, "measured": driver.N_MEASURED},
            "nvidia_smi_before": nvidia_smi_before,
            "nvidia_smi_after": nvidia_smi_after,
            "experiment": "vendor-ggml-cuda-sync-9970: ggml-cuda subsystem synced from lucebox-ggml fork "
                           "reconstruction (base_commit 6fbe72d + PR35 + PR37) to upstream ggml-org/llama.cpp "
                           "00f5442cc (build 9970) via 3-way merge. See patches/llama.cpp/"
                           "vendor-ggml-cuda-sync-9970.patch and /tmp/vendor_sync/notes.md for full scope.",
        },
        "dflash_arm": dflash_result,
        "summary": summary,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nwrote {RESULTS_PATH}")

    print("\n" + "=" * 80)
    print("Vendor-sync dflash arm summary (median per cell)")
    print("=" * 80)
    for depth in driver.DEPTHS:
        s = summary[str(depth)]
        print(f"depth={depth}: prefill_med={s['prefill_tok_s']['median']:.1f} tok/s  "
              f"decode_med={s['decode_tok_s']['median']:.1f} tok/s  wall_med={s['wall_s']['median']:.3f}s")


if __name__ == "__main__":
    main()
