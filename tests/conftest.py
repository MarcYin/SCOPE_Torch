"""Shared test fixtures for SCOPE-Torch test suite."""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Device / dtype parametrization helpers
# ---------------------------------------------------------------------------

@pytest.fixture(params=[torch.float32, torch.float64], ids=["f32", "f64"])
def dtype(request):
    """Parametrize tests over common floating-point dtypes."""
    return request.param


@pytest.fixture
def device():
    """Default test device (CPU)."""
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Spectral grids
# ---------------------------------------------------------------------------

@pytest.fixture
def spectral_grids():
    """Compact spectral grids suitable for unit tests."""
    from scope.spectral.fluspect import SpectralGrids

    device = torch.device("cpu")
    dtype = torch.float64
    wlP = torch.linspace(400.0, 2500.0, 64, device=device, dtype=dtype)
    wlF = torch.linspace(640.0, 850.0, 16, device=device, dtype=dtype)
    wlE = torch.linspace(400.0, 750.0, 16, device=device, dtype=dtype)
    return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)


@pytest.fixture
def full_spectral_grids():
    """Full 1-nm resolution spectral grids matching upstream SCOPE."""
    from scope.spectral.fluspect import SpectralGrids

    return SpectralGrids.default(torch.device("cpu"), torch.float64)


# ---------------------------------------------------------------------------
# SCOPE upstream assets (skip if not available)
# ---------------------------------------------------------------------------

@pytest.fixture
def scope_root():
    """Path to the upstream SCOPE checkout, or skip if not available."""
    from scope.spectral.loaders import scope_root as _scope_root

    try:
        return _scope_root()
    except FileNotFoundError:
        pytest.skip("Upstream SCOPE assets not found")


@pytest.fixture
def fluspect_resources(scope_root):
    """Load FLUSPECT resources from upstream SCOPE."""
    from scope.spectral.loaders import load_fluspect_resources

    return load_fluspect_resources(scope_root_path=scope_root, dtype=torch.float64)


@pytest.fixture
def optipar(fluspect_resources):
    """OptiPar loaded from upstream SCOPE assets."""
    return fluspect_resources.optipar


@pytest.fixture
def full_spectral(fluspect_resources):
    """Full spectral grids from upstream SCOPE assets."""
    return fluspect_resources.spectral


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@pytest.fixture
def fluspect_model(fluspect_resources):
    """FluspectModel initialized from upstream SCOPE assets."""
    from scope.spectral.fluspect import FluspectModel

    return FluspectModel(
        fluspect_resources.spectral,
        fluspect_resources.optipar,
        ndub=15,
        doublings_step=5,
        dtype=torch.float64,
    )


@pytest.fixture
def default_lidf():
    """Default 13-element SCOPE LIDF for a spherical distribution."""
    from scope.canopy.foursail import scope_lidf

    return scope_lidf(1.0, 0.0, dtype=torch.float64)


@pytest.fixture
def sail_model(default_lidf):
    """FourSAILModel with default SCOPE litab/lidf."""
    from scope.canopy.foursail import FourSAILModel

    return FourSAILModel(lidf=default_lidf)


@pytest.fixture
def reflectance_model(fluspect_resources, default_lidf):
    """CanopyReflectanceModel initialized from upstream SCOPE assets."""
    from scope.canopy.reflectance import CanopyReflectanceModel

    return CanopyReflectanceModel.from_scope_assets(
        lidf=default_lidf,
        scope_root_path=None,
        path=None,
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# Leaf bio defaults
# ---------------------------------------------------------------------------

@pytest.fixture
def default_leafbio():
    """Default leaf biochemistry inputs for testing."""
    from scope.spectral.fluspect import LeafBioBatch

    return LeafBioBatch(
        Cab=torch.tensor([40.0], dtype=torch.float64),
        Cca=torch.tensor([10.0], dtype=torch.float64),
        Cw=torch.tensor([0.009], dtype=torch.float64),
        Cdm=torch.tensor([0.012], dtype=torch.float64),
        Cs=torch.tensor([0.0], dtype=torch.float64),
        N=torch.tensor([1.4], dtype=torch.float64),
        Cant=torch.tensor([0.0], dtype=torch.float64),
        fqe=torch.tensor([0.01], dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def assert_close():
    """Return a helper that wraps torch.testing.assert_close with defaults."""

    def _assert_close(actual, expected, *, rtol=1e-4, atol=1e-6, msg=""):
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, msg=msg,
        )

    return _assert_close


def _index_dataclass(dc, idx):
    """Index all tensor fields of a dataclass along dim 0."""
    from dataclasses import fields as dc_fields

    kwargs = {}
    for f in dc_fields(dc):
        value = getattr(dc, f.name)
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            kwargs[f.name] = value[idx : idx + 1]
        else:
            kwargs[f.name] = value
    return type(dc)(**kwargs)
