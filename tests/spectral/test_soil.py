import math

import numpy as np
import torch

from scope.spectral.loaders import load_fluspect_resources
from scope.spectral.soil import BSMSoilParameters, SoilBSMModel


def _np_fresnel_tav(alfa: float, nr: np.ndarray | float) -> np.ndarray:
    nr = np.asarray(nr, dtype=np.float64)
    if float(alfa) == 0.0:
        return 4.0 * nr / ((nr + 1.0) * (nr + 1.0))

    angle = np.deg2rad(alfa)
    sin_angle = np.sin(angle)
    n2 = nr**2
    np_ = n2 + 1.0
    nm = n2 - 1.0
    a = ((nr + 1.0) ** 2) / 2.0
    k = -((n2 - 1.0) ** 2) / 4.0

    b2 = sin_angle**2 - np_ / 2.0
    b1 = np.sqrt(np.clip(b2**2 + k, a_min=0.0, a_max=None))
    b = b1 - b2

    ts = ((k**2) / (6.0 * b**3) + k / b - b / 2.0) - ((k**2) / (6.0 * a**3) + k / a - a / 2.0)
    tp1 = -2.0 * n2 * (b - a) / (np_**2)
    tp2 = -2.0 * n2 * np_ * np.log(b / a) / (nm**2)
    tp3 = n2 * (1.0 / b - 1.0 / a) / 2.0
    tp4 = 16.0 * n2**2 * (n2**2 + 1.0) * np.log((2.0 * np_ * b - nm**2) / (2.0 * np_ * a - nm**2)) / (np_**3 * nm**2)
    tp5 = 16.0 * n2**3 * (1.0 / (2.0 * np_ * b - nm**2) - 1.0 / (2.0 * np_ * a - nm**2)) / (np_**3)
    return (ts + tp1 + tp2 + tp3 + tp4 + tp5) / (2.0 * sin_angle**2)


def _np_bsm_reference(
    brightness: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    smc: np.ndarray,
) -> np.ndarray:
    resources = load_fluspect_resources(dtype=torch.float64)
    GSV = resources.extras["GSV"].cpu().numpy()
    Kw = resources.optipar.Kw.cpu().numpy()
    nw = resources.extras["nw"].cpu().numpy()

    smc = np.asarray(smc, dtype=np.float64)
    if np.mean(smc) > 1.0:
        smc = smc / 100.0

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    coefficients = np.stack(
        [
            brightness * np.sin(lat_rad),
            brightness * np.cos(lat_rad) * np.sin(lon_rad),
            brightness * np.cos(lat_rad) * np.cos(lon_rad),
        ],
        axis=-1,
    )
    rdry = coefficients @ GSV.T

    smp = smc * 100.0
    mu = (smp - 5.0) / 25.0

    tav90_2 = _np_fresnel_tav(90.0, 2.0)
    tav90_2_over_nw = _np_fresnel_tav(90.0, 2.0 / nw)
    p = 1.0 - _np_fresnel_tav(90.0, nw) / (nw**2)
    Rw = 1.0 - _np_fresnel_tav(40.0, nw)
    rbac = 1.0 - (1.0 - rdry) * (rdry * tav90_2_over_nw[None, :] / tav90_2 + 1.0 - rdry)

    k = np.arange(7, dtype=np.float64)
    factorials = np.array([math.factorial(int(v)) for v in k], dtype=np.float64)
    mu_safe = np.clip(mu, a_min=1e-12, a_max=None)
    poisson = np.exp(-mu_safe)[:, None] * (mu_safe[:, None] ** k[None, :]) / factorials[None, :]
    tw = np.exp((-2.0 * Kw[:, None] * 0.015) * k[None, :])
    Rwet_k = Rw[None, :, None] + (
        (1.0 - Rw)[None, :, None]
        * (1.0 - p)[None, :, None]
        * tw[None, :, :]
        * rbac[:, :, None]
        / (1.0 - p[None, :, None] * tw[None, :, :] * rbac[:, :, None])
    )

    rwet = rdry * poisson[:, :1] + np.einsum("bnk,bk->bn", Rwet_k[:, :, 1:], poisson[:, 1:])
    return np.where(mu[:, None] > 0.0, rwet, rdry)


def test_soil_bsm_model_matches_reference_port():
    model = SoilBSMModel.from_scope_assets(dtype=torch.float64)
    params = BSMSoilParameters(
        BSMBrightness=torch.tensor([0.50, 0.85], dtype=torch.float64),
        BSMlat=torch.tensor([25.0, 35.0], dtype=torch.float64),
        BSMlon=torch.tensor([45.0, 60.0], dtype=torch.float64),
        SMC=torch.tensor([0.04, 0.25], dtype=torch.float64),
    )

    output = model(params).cpu().numpy()
    expected = _np_bsm_reference(
        brightness=np.array([0.50, 0.85], dtype=np.float64),
        lat=np.array([25.0, 35.0], dtype=np.float64),
        lon=np.array([45.0, 60.0], dtype=np.float64),
        smc=np.array([0.04, 0.25], dtype=np.float64),
    )

    assert np.allclose(output, expected, rtol=1e-10, atol=1e-10)


def test_soil_bsm_model_accepts_fraction_or_percentage_smc():
    model = SoilBSMModel.from_scope_assets(dtype=torch.float64)
    kwargs = {
        "BSMBrightness": torch.tensor([0.60], dtype=torch.float64),
        "BSMlat": torch.tensor([30.0], dtype=torch.float64),
        "BSMlon": torch.tensor([55.0], dtype=torch.float64),
    }

    fraction = model(BSMSoilParameters(SMC=torch.tensor([0.25], dtype=torch.float64), **kwargs))
    percent = model(BSMSoilParameters(SMC=torch.tensor([25.0], dtype=torch.float64), **kwargs))

    assert torch.allclose(fraction, percent)
