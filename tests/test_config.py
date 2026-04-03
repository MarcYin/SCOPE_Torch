"""Tests for scope.config.SimulationConfig validation."""

from __future__ import annotations

from datetime import datetime

import pytest
import torch

from scope.config import SimulationConfig


def test_valid_config():
    cfg = SimulationConfig(
        roi_bounds=(0.0, 0.0, 1.0, 1.0),
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
    )
    assert cfg.torch_device() == torch.device("cpu")
    assert cfg.torch_dtype() == torch.float32


def test_invalid_roi_bounds_min_exceeds_max():
    with pytest.raises(ValueError, match="roi_bounds min must not exceed max"):
        SimulationConfig(
            roi_bounds=(10.0, 0.0, 5.0, 1.0),
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
        )


def test_invalid_time_ordering():
    with pytest.raises(ValueError, match="start_time.*must not be after end_time"):
        SimulationConfig(
            roi_bounds=(0.0, 0.0, 1.0, 1.0),
            start_time=datetime(2024, 1, 2),
            end_time=datetime(2024, 1, 1),
        )


def test_negative_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SimulationConfig(
            roi_bounds=(0.0, 0.0, 1.0, 1.0),
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            chunk_size=-5,
        )


def test_zero_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        SimulationConfig(
            roi_bounds=(0.0, 0.0, 1.0, 1.0),
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            chunk_size=0,
        )


def test_none_chunk_size_valid():
    cfg = SimulationConfig(
        roi_bounds=(0.0, 0.0, 1.0, 1.0),
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        chunk_size=None,
    )
    assert cfg.chunks(100) == [slice(0, 100)]


def test_invalid_device_string():
    with pytest.raises(RuntimeError):
        SimulationConfig(
            roi_bounds=(0.0, 0.0, 1.0, 1.0),
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
            device="not_a_device",
        )


def test_chunks_splits_correctly():
    cfg = SimulationConfig(
        roi_bounds=(0.0, 0.0, 1.0, 1.0),
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        chunk_size=3,
    )
    slices = cfg.chunks(10)
    assert len(slices) == 4
    assert slices[0] == slice(0, 3)
    assert slices[-1] == slice(9, 10)


def test_equal_start_end_time_valid():
    cfg = SimulationConfig(
        roi_bounds=(0.0, 0.0, 1.0, 1.0),
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 1),
    )
    assert cfg.start_time == cfg.end_time
