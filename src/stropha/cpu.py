"""CPU budget of the current process.

``os.cpu_count()`` reports every core of the host. Inside a container that is
usually far more than the process may use: the cgroup CPU quota (Kubernetes
``resources.limits.cpu``) and the CPU affinity mask both cap it. Thread pools
sized from the host count oversubscribe the quota and spend most of their time
throttled, so size them from :func:`available_cpus` instead.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_V1_CPU_DIRS = ("cpu,cpuacct", "cpu")


def _affinity_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _read_quota(cgroup_root: Path) -> float | None:
    """CPU quota in cores, or ``None`` when no quota is set or readable."""
    try:
        quota, period = (cgroup_root / "cpu.max").read_text().split()[:2]
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    for name in _V1_CPU_DIRS:
        cpu_dir = cgroup_root / name
        try:
            quota_us = int((cpu_dir / "cpu.cfs_quota_us").read_text())
            period_us = int((cpu_dir / "cpu.cfs_period_us").read_text())
        except (OSError, ValueError):
            continue
        return None if quota_us <= 0 or period_us <= 0 else quota_us / period_us
    return None


def available_cpus(
    *,
    cgroup_root: Path = _CGROUP_ROOT,
    affinity: int | None = None,
) -> int:
    """Number of CPUs this process can actually use (always >= 1).

    The smaller of the affinity mask and the cgroup quota, rounded up.
    ``cgroup_root`` and ``affinity`` are injection points for tests.
    """
    cpus = affinity if affinity is not None else _affinity_cpus()
    quota = _read_quota(cgroup_root)
    if quota is not None:
        cpus = min(cpus, math.ceil(quota))
    return max(1, cpus)
