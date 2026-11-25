"""Tests for the CPU budget a process may use (cgroup quota + affinity)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stropha.adapters.embedder.local import (
    CPU_EXECUTION,
    FastEmbedEngine,
    LocalEmbedder,
    LocalEmbedderConfig,
)
from stropha.cpu import available_cpus


def _cgroup_v2(root: Path, cpu_max: str) -> Path:
    (root / "cpu.max").write_text(cpu_max + "\n")
    return root


def _cgroup_v1(root: Path, quota: int, period: int) -> Path:
    cpu = root / "cpu,cpuacct"
    cpu.mkdir()
    (cpu / "cpu.cfs_quota_us").write_text(f"{quota}\n")
    (cpu / "cpu.cfs_period_us").write_text(f"{period}\n")
    return root


def test_cgroup_v2_quota_caps_visible_cpus(tmp_path: Path) -> None:
    root = _cgroup_v2(tmp_path, "400000 100000")
    assert available_cpus(cgroup_root=root, affinity=32) == 4


def test_cgroup_v2_without_quota_uses_affinity(tmp_path: Path) -> None:
    root = _cgroup_v2(tmp_path, "max 100000")
    assert available_cpus(cgroup_root=root, affinity=32) == 32


def test_cgroup_v1_quota_caps_visible_cpus(tmp_path: Path) -> None:
    root = _cgroup_v1(tmp_path, 400000, 100000)
    assert available_cpus(cgroup_root=root, affinity=32) == 4


def test_cgroup_v1_unlimited_quota_uses_affinity(tmp_path: Path) -> None:
    root = _cgroup_v1(tmp_path, -1, 100000)
    assert available_cpus(cgroup_root=root, affinity=32) == 32


def test_fractional_quota_rounds_up(tmp_path: Path) -> None:
    root = _cgroup_v2(tmp_path, "150000 100000")
    assert available_cpus(cgroup_root=root, affinity=32) == 2


def test_quota_never_reports_less_than_one_cpu(tmp_path: Path) -> None:
    root = _cgroup_v2(tmp_path, "10000 100000")
    assert available_cpus(cgroup_root=root, affinity=32) == 1


def test_affinity_below_quota_wins(tmp_path: Path) -> None:
    root = _cgroup_v2(tmp_path, "1600000 100000")
    assert available_cpus(cgroup_root=root, affinity=8) == 8


def test_no_cgroup_files_uses_affinity(tmp_path: Path) -> None:
    assert available_cpus(cgroup_root=tmp_path, affinity=6) == 6


def test_local_embedder_threads_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LocalEmbedderConfig(threads=0)


def _fake_text_embedding() -> MagicMock:
    model = MagicMock()
    model.embed.return_value = iter([[0.0, 1.0]])
    return model


def test_explicit_threads_reach_onnx_runtime() -> None:
    with patch("fastembed.TextEmbedding", return_value=_fake_text_embedding()) as ctor:
        LocalEmbedder(LocalEmbedderConfig(threads=3))
    assert ctor.call_args.kwargs["threads"] == 3


def test_default_threads_follow_cpu_budget() -> None:
    with (
        patch("fastembed.TextEmbedding", return_value=_fake_text_embedding()) as ctor,
        patch("stropha.adapters.embedder.local.available_cpus", return_value=5),
    ):
        LocalEmbedder(LocalEmbedderConfig())
    assert ctor.call_args.kwargs["threads"] == 5


def test_threads_do_not_change_adapter_identity() -> None:
    with patch("fastembed.TextEmbedding", side_effect=lambda **_: _fake_text_embedding()):
        two = LocalEmbedder(LocalEmbedderConfig(threads=2))
        eight = LocalEmbedder(LocalEmbedderConfig(threads=8))
    assert two.adapter_id == eight.adapter_id


def test_engine_reports_thread_count_in_health() -> None:
    with patch("fastembed.TextEmbedding", return_value=_fake_text_embedding()):
        engine = FastEmbedEngine("some/model", CPU_EXECUTION, threads=7)
    assert engine.health_detail["threads"] == "7"
