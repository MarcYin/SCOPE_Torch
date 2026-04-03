"""Numerical edge-case tests for SCOPE-Torch models."""

from __future__ import annotations

import pytest
import torch

from scope.canopy.foursail import FourSAILModel, scope_lidf


@pytest.fixture
def model():
    lidf = scope_lidf(1.0, 0.0, dtype=torch.float64)
    return FourSAILModel(lidf=lidf)


def test_zero_lai(model):
    """Zero LAI should return initial zero-canopy outputs (tss=too=tsstoo=1)."""
    nwl = 10
    rho = torch.full((1, nwl), 0.1, dtype=torch.float64)
    tau = torch.full((1, nwl), 0.05, dtype=torch.float64)
    soil = torch.full((1, nwl), 0.3, dtype=torch.float64)
    result = model(
        rho,
        tau,
        soil,
        lai=torch.tensor([0.0]),
        hotspot=torch.tensor([0.05]),
        tts=torch.tensor([30.0]),
        tto=torch.tensor([0.0]),
        psi=torch.tensor([0.0]),
    )
    # With zero LAI, FourSAIL returns defaults: tss=too=tsstoo=1, rsot=0
    assert torch.isfinite(result.rsot).all()
    assert (result.tss == 1.0).all()
    assert (result.too == 1.0).all()


def test_very_high_lai(model):
    """Very high LAI should produce saturated reflectance without NaN/Inf."""
    nwl = 10
    rho = torch.full((1, nwl), 0.5, dtype=torch.float64)
    tau = torch.full((1, nwl), 0.3, dtype=torch.float64)
    soil = torch.full((1, nwl), 0.1, dtype=torch.float64)
    result = model(
        rho,
        tau,
        soil,
        lai=torch.tensor([15.0]),
        hotspot=torch.tensor([0.05]),
        tts=torch.tensor([30.0]),
        tto=torch.tensor([0.0]),
        psi=torch.tensor([0.0]),
    )
    assert torch.isfinite(result.rsot).all()
    assert (result.rsot >= 0).all()


def test_nadir_viewing(model):
    """Nadir viewing angle (tto=0) should work without division issues."""
    nwl = 10
    rho = torch.full((1, nwl), 0.1, dtype=torch.float64)
    tau = torch.full((1, nwl), 0.05, dtype=torch.float64)
    soil = torch.full((1, nwl), 0.2, dtype=torch.float64)
    result = model(
        rho,
        tau,
        soil,
        lai=torch.tensor([3.0]),
        hotspot=torch.tensor([0.05]),
        tts=torch.tensor([30.0]),
        tto=torch.tensor([0.0]),
        psi=torch.tensor([0.0]),
    )
    assert torch.isfinite(result.rsot).all()


def test_hotspot_direction(model):
    """Hotspot direction (tto=tts, psi=0) should produce enhanced reflectance."""
    nwl = 10
    rho = torch.full((1, nwl), 0.1, dtype=torch.float64)
    tau = torch.full((1, nwl), 0.05, dtype=torch.float64)
    soil = torch.full((1, nwl), 0.2, dtype=torch.float64)
    result_hotspot = model(
        rho,
        tau,
        soil,
        lai=torch.tensor([3.0]),
        hotspot=torch.tensor([0.05]),
        tts=torch.tensor([30.0]),
        tto=torch.tensor([30.0]),
        psi=torch.tensor([0.0]),
    )
    result_offspot = model(
        rho,
        tau,
        soil,
        lai=torch.tensor([3.0]),
        hotspot=torch.tensor([0.05]),
        tts=torch.tensor([30.0]),
        tto=torch.tensor([30.0]),
        psi=torch.tensor([180.0]),
    )
    # Hotspot direction should have higher or equal reflectance
    assert (result_hotspot.rsot >= result_offspot.rsot - 1e-9).all()


def test_extreme_solar_zenith(model):
    """Near-horizontal sun angle should work without NaN."""
    nwl = 10
    rho = torch.full((1, nwl), 0.1, dtype=torch.float64)
    tau = torch.full((1, nwl), 0.05, dtype=torch.float64)
    soil = torch.full((1, nwl), 0.2, dtype=torch.float64)
    result = model(
        rho,
        tau,
        soil,
        lai=torch.tensor([3.0]),
        hotspot=torch.tensor([0.05]),
        tts=torch.tensor([85.0]),
        tto=torch.tensor([0.0]),
        psi=torch.tensor([0.0]),
    )
    assert torch.isfinite(result.rsot).all()


def test_batch_consistency(model):
    """Batched and unbatched should produce identical results."""
    nwl = 10
    rho = torch.full((1, nwl), 0.1, dtype=torch.float64)
    tau = torch.full((1, nwl), 0.05, dtype=torch.float64)
    soil = torch.full((1, nwl), 0.2, dtype=torch.float64)
    single = model(
        rho,
        tau,
        soil,
        lai=torch.tensor([3.0]),
        hotspot=torch.tensor([0.05]),
        tts=torch.tensor([30.0]),
        tto=torch.tensor([10.0]),
        psi=torch.tensor([45.0]),
    )

    rho2 = rho.repeat(3, 1)
    tau2 = tau.repeat(3, 1)
    soil2 = soil.repeat(3, 1)
    batched = model(
        rho2,
        tau2,
        soil2,
        lai=torch.tensor([3.0, 3.0, 3.0]),
        hotspot=torch.tensor([0.05, 0.05, 0.05]),
        tts=torch.tensor([30.0, 30.0, 30.0]),
        tto=torch.tensor([10.0, 10.0, 10.0]),
        psi=torch.tensor([45.0, 45.0, 45.0]),
    )
    torch.testing.assert_close(single.rsot, batched.rsot[:1], rtol=1e-10, atol=1e-10)


def test_float32_numerical_stability(model):
    """Float32 should not produce NaN/Inf for typical inputs."""
    nwl = 10
    rho = torch.full((1, nwl), 0.1, dtype=torch.float32)
    tau = torch.full((1, nwl), 0.05, dtype=torch.float32)
    soil = torch.full((1, nwl), 0.2, dtype=torch.float32)
    result = model(
        rho,
        tau,
        soil,
        lai=torch.tensor([3.0], dtype=torch.float32),
        hotspot=torch.tensor([0.05], dtype=torch.float32),
        tts=torch.tensor([30.0], dtype=torch.float32),
        tto=torch.tensor([0.0], dtype=torch.float32),
        psi=torch.tensor([0.0], dtype=torch.float32),
    )
    assert torch.isfinite(result.rsot).all()
    assert (result.rsot >= 0).all()
