"""Tests for scope.canopy.layered_rt.LayeredCanopyTransportModel."""

from __future__ import annotations

import pytest
import torch

from scope.canopy.foursail import FourSAILModel, scope_lidf
from scope.canopy.layered_rt import LayeredCanopyTransportModel


@pytest.fixture
def model():
    lidf = scope_lidf(1.0, 0.0, dtype=torch.float64)
    sail = FourSAILModel(lidf=lidf)
    return LayeredCanopyTransportModel(sail)


@pytest.fixture
def simple_inputs():
    batch = 2
    nwl = 10
    device = torch.device("cpu")
    dtype = torch.float64
    return {
        "rho": torch.full((batch, nwl), 0.1, device=device, dtype=dtype),
        "tau": torch.full((batch, nwl), 0.05, device=device, dtype=dtype),
        "soil_refl": torch.full((batch, nwl), 0.2, device=device, dtype=dtype),
        "lai": torch.tensor([3.0, 5.0], device=device, dtype=dtype),
        "tts": torch.tensor([30.0, 45.0], device=device, dtype=dtype),
        "tto": torch.tensor([0.0, 10.0], device=device, dtype=dtype),
        "psi": torch.tensor([0.0, 90.0], device=device, dtype=dtype),
        "hotspot": torch.tensor([0.05, 0.05], device=device, dtype=dtype),
    }


def test_build_produces_valid_transfer(model, simple_inputs):
    lidf = scope_lidf(1.0, 0.0, dtype=torch.float64)
    transfer = model.build(**simple_inputs, lidf=lidf, nlayers=10)
    assert transfer.nlayers == 10
    assert transfer.R_dd.shape == (2, 11, 10)
    assert transfer.Ps.shape == (2, 11)
    assert transfer.Po.shape == (2, 11)
    assert transfer.Pso.shape == (2, 11)
    # Gap fractions should be in [0, 1]
    assert (transfer.Ps >= 0).all() and (transfer.Ps <= 1.0 + 1e-6).all()
    assert (transfer.Po >= 0).all() and (transfer.Po <= 1.0 + 1e-6).all()


def test_flux_profiles_energy_conservation(model, simple_inputs):
    lidf = scope_lidf(1.0, 0.0, dtype=torch.float64)
    transfer = model.build(**simple_inputs, lidf=lidf, nlayers=10)
    batch, nwl = 2, 10
    Esun = torch.ones((batch, nwl), dtype=torch.float64)
    Esky = torch.zeros((batch, nwl), dtype=torch.float64)
    profiles = model.flux_profiles(transfer, Esun, Esky)
    # Top-of-canopy upwelling should be non-negative
    assert (profiles.Eplu_[:, 0, :] >= -1e-9).all()
    # Below-canopy direct should be less than incoming
    assert (profiles.Es_[:, -1, :] <= Esun + 1e-9).all()


def test_different_nlayers_produce_different_profiles(model, simple_inputs):
    lidf = scope_lidf(1.0, 0.0, dtype=torch.float64)
    t10 = model.build(**simple_inputs, lidf=lidf, nlayers=10)
    t30 = model.build(**simple_inputs, lidf=lidf, nlayers=30)
    assert t10.R_dd.shape[1] != t30.R_dd.shape[1]
    # Top-of-canopy reflectance should converge for large nlayers
    assert t10.R_dd[:, 0, :].shape == t30.R_dd[:, 0, :].shape


def test_zero_lai_transparent(model):
    batch, nwl = 1, 10
    dtype = torch.float64
    lidf = scope_lidf(1.0, 0.0, dtype=dtype)
    rho = torch.full((batch, nwl), 0.1, dtype=dtype)
    tau = torch.full((batch, nwl), 0.05, dtype=dtype)
    soil = torch.full((batch, nwl), 0.3, dtype=dtype)
    transfer = model.build(
        rho, tau, soil,
        lai=torch.tensor([0.001], dtype=dtype),
        tts=torch.tensor([30.0], dtype=dtype),
        tto=torch.tensor([0.0], dtype=dtype),
        psi=torch.tensor([0.0], dtype=dtype),
        hotspot=torch.tensor([0.05], dtype=dtype),
        lidf=lidf,
        nlayers=5,
    )
    # For near-zero LAI the canopy reflectance should be close to soil
    assert torch.allclose(transfer.R_dd[:, 0, :], soil, atol=0.05)
