"""Install a user-local autoggml launcher on an SSH target."""

from __future__ import annotations

import argparse
import json

from autoggml.doctor import build_profile
from autoggml.remote import SSHWorker
from autoggml.targets import TargetConfig


INSTALL = r'''
import os, pathlib, re, sys
root, model_root, target_name, lock_path, jobs = sys.argv[1:6]
home = pathlib.Path.home()
config = home / ".config" / "autoggml" / "targets.toml"
config.parent.mkdir(parents=True, exist_ok=True)
text = config.read_text() if config.exists() else ""
section = (
    f"[targets.{target_name}]\n"
    f'transport = "local"\n'
    f'root = "{root}"\n'
    f'model_root = "{model_root}"\n'
    f'lock_path = "{lock_path}"\n'
    f"build_jobs = {jobs}\n"
)
pattern = re.compile(rf"(?ms)^\[targets\.{re.escape(target_name)}\]\n.*?(?=^\[|\Z)")
text = pattern.sub(section + "\n", text) if pattern.search(text) else text.rstrip() + "\n\n" + section
config.write_text(text.lstrip())

launcher = home / ".local" / "bin" / "autoggml"
launcher.parent.mkdir(parents=True, exist_ok=True)
launcher.write_text(
    "#!/bin/sh\n"
    f"export AUTOGGML_DEFAULT_TARGET={target_name}\n"
    f'exec "{root}/.tools/uv" run --no-dev --project "{root}" autoggml "$@"\n'
)
launcher.chmod(0o755)

profile = home / ".profile"
profile_text = profile.read_text() if profile.exists() else ""
path_line = 'export PATH="$HOME/.local/bin:$PATH"'
if path_line not in profile_text:
    profile.write_text(profile_text.rstrip() + "\n\n# autoggml user commands\n" + path_line + "\n")
print(str(launcher))
'''.strip()


def onboard(target: TargetConfig, local_name: str = "lucebox3") -> dict:
    if target.transport != "ssh":
        raise ValueError("onboard requires an SSH target")
    profile = build_profile(target)
    worker = SSHWorker(target)
    worker.sync_repo()
    worker.ensure_remote_uv()
    result = worker.run_python(INSTALL, [
        target.root.rstrip("/"),
        target.model_root or f"{target.root.rstrip('/')}/work/models",
        local_name,
        target.lock_path,
        str(target.build_jobs),
    ])
    return {
        "target": target.name,
        "local_target": local_name,
        "machine_fingerprint": profile.fingerprint,
        "busy": bool(profile.observed.get("busy_reasons")),
        "launcher": result.stdout.strip(),
        "next": [
            f"ssh {target.host}",
            "autoggml test-drive",
            "autoggml test-drive --live",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a simple user-local autoggml experience on an SSH target")
    parser.add_argument("--target", required=True)
    parser.add_argument("--local-name", default="lucebox3")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = onboard(TargetConfig.load(args.target), args.local_name)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("autoggml onboarding complete")
    print(f"  launcher: {result['launcher']}")
    print(f"  machine:  {result['machine_fingerprint'][:16]}")
    print("\nNext:")
    print(f"  {result['next'][0]}")
    print(f"  {result['next'][1]}")


if __name__ == "__main__":
    main()
