"""Low-overhead host/UMA telemetry sampled around benchmark subprocesses."""

from __future__ import annotations

import glob
import threading
import time
from pathlib import Path
from typing import Any


def _read_int(path: str | Path) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def _meminfo() -> dict[str, int]:
    result = {}
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        key, _, raw = line.partition(":")
        if raw:
            result[key] = int(raw.strip().split()[0]) * 1024
    return result


def _vmstat(name: str) -> int | None:
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            key, value = line.split()
            if key == name:
                return int(value)
    except (OSError, ValueError):
        pass
    return None


def _max_glob(patterns: list[str]) -> int | None:
    values = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            value = _read_int(path)
            if value is not None:
                values.append(value)
    return max(values) if values else None


def sample() -> dict[str, int | float | None]:
    mem = _meminfo()
    swap_used = None
    if "SwapTotal" in mem and "SwapFree" in mem:
        swap_used = mem["SwapTotal"] - mem["SwapFree"]
    return {
        "monotonic_s": time.monotonic(),
        "mem_available_bytes": mem.get("MemAvailable"),
        "swap_used_bytes": swap_used,
        "major_faults": _vmstat("pgmajfault"),
        "gtt_used_bytes": _max_glob(["/sys/class/drm/card*/device/mem_info_gtt_used"]),
        "vram_used_bytes": _max_glob([
            "/sys/class/drm/card*/device/mem_info_vram_used",
            "/sys/class/drm/card*/device/mem_info_vis_vram_used",
        ]),
        "temperature_millic": _max_glob(["/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input"]),
        "power_microw": _max_glob(["/sys/class/drm/card*/device/hwmon/hwmon*/power*_average"]),
        "gpu_clock_hz": _max_glob(["/sys/class/drm/card*/device/hwmon/hwmon*/freq*_input"]),
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, float]:
    if not samples:
        return {}

    def values(key: str) -> list[float]:
        return [float(item[key]) for item in samples if item.get(key) is not None]

    summary: dict[str, float] = {"telemetry_samples": float(len(samples))}
    available = values("mem_available_bytes")
    if available:
        summary["min_mem_available_GiB"] = min(available) / (1024 ** 3)
    for key, output in (
        ("gtt_used_bytes", "peak_gtt_used_GiB"),
        ("vram_used_bytes", "peak_vram_used_GiB"),
    ):
        current = values(key)
        if current:
            summary[output] = max(current) / (1024 ** 3)
    temperatures = values("temperature_millic")
    if temperatures:
        summary["peak_temperature_c"] = max(temperatures) / 1000.0
    powers = values("power_microw")
    if powers:
        summary["peak_power_w"] = max(powers) / 1_000_000.0
    clocks = values("gpu_clock_hz")
    if clocks:
        summary["min_gpu_clock_mhz"] = min(clocks) / 1_000_000.0
        summary["max_gpu_clock_mhz"] = max(clocks) / 1_000_000.0
    swaps = values("swap_used_bytes")
    if swaps:
        summary["swap_growth_GiB"] = max(0.0, swaps[-1] - swaps[0]) / (1024 ** 3)
    faults = values("major_faults")
    if faults:
        summary["major_faults_delta"] = max(0.0, faults[-1] - faults[0])
    return summary


class TelemetryCollector:
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.append(sample())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.samples.append(sample())

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)
        self.samples.append(sample())
        return summarize(self.samples)
