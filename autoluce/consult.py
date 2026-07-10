"""Create a concrete research contract from target and model observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoluce.contracts import ResearchContract
from autoluce.doctor import build_profile
from autoluce.models import load_catalog
from autoluce.profiles import ModelProfile
from autoluce.remote import SSHWorker
from autoluce.targets import TargetConfig


HASH_MODEL = r'''
import hashlib, json, os, pathlib, sys
cache_path = pathlib.Path(sys.argv[1])
paths = [pathlib.Path(value) for value in sys.argv[2:]]
signature = [{"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in paths]
cache = {}
try:
    cache = json.loads(cache_path.read_text())
except Exception:
    pass
key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
if key not in cache:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    cache[key] = digest.hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    os.replace(tmp, cache_path)
print(cache[key])
'''.strip()


def _model_from_doctor(profile, model_id: str, sha256: str | None = None) -> ModelProfile:
    entry = load_catalog()[model_id]
    records = profile.observed.get("models", [])
    if not records or any(record.get("missing") for record in records):
        raise FileNotFoundError(f"model '{model_id}' is not present on target")
    size = sum(int(record["size_bytes"]) for record in records)
    return ModelProfile(model_id, entry.quant, [record["path"] for record in records], size, sha256, entry.metadata or {})


def create_contract(target: TargetConfig, model_id: str = "deepseek-v4-flash", hash_model: bool = False) -> ResearchContract:
    machine = build_profile(target, model_id)
    sha256 = None
    if hash_model:
        if target.transport != "ssh":
            raise ValueError("--hash-model currently requires an SSH target")
        paths = [record["path"] for record in machine.observed.get("models", [])]
        cache = f"{target.root.rstrip('/')}/work/model-digests.json"
        result = SSHWorker(target).run(
            ["python3", "-c", HASH_MODEL, cache, *paths], lease=True, timeout=7200,
        )
        sha256 = result.stdout.strip()
    model = _model_from_doctor(machine, model_id, sha256)
    return ResearchContract(
        target=target.name,
        machine_fingerprint=machine.fingerprint,
        model=model_id,
        model_fingerprint=model.fingerprint,
        build_jobs=target.build_jobs,
        confirmed_assumptions=[
            "hip_is_primary_backend",
            "interactive_batch_one_is_primary",
            "quality_tradeoff_is_not_allowed",
            "environment_upgrade_requires_new_baseline",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a machine-aware research contract")
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-model", action="store_true", help="Hash all model bytes under the fail-fast remote lease")
    args = parser.parse_args()
    contract = create_contract(TargetConfig.load(args.target), args.model, args.hash_model)
    contract.write(args.output)
    print(json.dumps(contract.to_dict(), indent=2))


if __name__ == "__main__":
    main()
