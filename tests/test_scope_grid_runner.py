import numpy as np
import pandas as pd
import torch
import xarray as xr

from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf
from scope_torch.config import SimulationConfig
from scope_torch.data import ScopeGridDataModule
from scope_torch.runners.grid import ScopeGridRunner
from scope_torch.spectral.fluspect import FluspectModel, LeafBioBatch, OptiPar, SpectralGrids


def _spectral(device, dtype):
    wlP = torch.linspace(400.0, 700.0, 32, device=device, dtype=dtype)
    wlF = torch.linspace(640.0, 740.0, 16, device=device, dtype=dtype)
    wlE = torch.linspace(400.0, 700.0, 16, device=device, dtype=dtype)
    return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)


def _optipar(spectral):
    wl = spectral.wlP
    base = torch.linspace(0, 1, wl.numel(), dtype=wl.dtype, device=wl.device)
    return OptiPar(
        nr=1.4 + 0.05 * torch.sin(base),
        Kab=0.01 + 0.005 * torch.cos(base),
        Kca=0.008 + 0.003 * torch.sin(base * 2),
        KcaV=0.008 + 0.003 * torch.sin(base * 2) * 0.95,
        KcaZ=0.008 + 0.003 * torch.sin(base * 2) * 1.05,
        Kdm=0.005 + 0.002 * torch.cos(base * 3),
        Kw=0.002 + 0.001 * torch.sin(base * 4),
        Ks=0.001 + 0.0005 * torch.cos(base * 5),
        Kant=0.0002 + 0.0001 * torch.sin(base * 6),
        phi=torch.full_like(wl, 0.5),
    )


def test_scope_grid_runner_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)

    times = pd.date_range("2020-07-01", periods=3, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    nwl = spectral.wlP.numel()
    rng = np.random.default_rng(0)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 3), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 3), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 3), 0.012)),
            "LAI": (("y", "x", "time"), np.linspace(2.0, 3.0, 3).reshape(1, 1, 3)),
            "tts": (("y", "x", "time"), np.full((1, 1, 3), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 3), 20.0)),
            "psi": (("y", "x", "time"), np.linspace(5.0, 15.0, 3).reshape(1, 1, 3)),
            "soil_refl": (("y", "x", "time", "wavelength"), rng.random((1, 1, 3, nwl)) * 0.2 + 0.1),
        },
        coords={"y": y, "x": x, "time": times, "wavelength": np.arange(nwl)},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["Cab", "Cw", "Cdm", "LAI", "tts", "tto", "psi", "soil_refl"])
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)
    outputs = runner.run(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_refl": "soil_refl",
        },
    )

    stacked = data.stack(batch=("y", "x", "time"))
    manual_leaf = []
    manual_rsot = []
    manual_rdd = []
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
        )
        leafopt = fluspect(leafbio)
        manual_leaf.append((leafopt.refl, leafopt.tran))
        soil_tensor = torch.tensor(stacked["soil_refl"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0)
        sail_out = sail(
            leafopt.refl,
            leafopt.tran,
            soil_tensor,
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([runner.default_hotspot], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            lidf=lidf,
        )
        manual_rsot.append(sail_out.rsot)
        manual_rdd.append(sail_out.rdd)

    manual_leaf_refl = torch.cat([r for r, _ in manual_leaf], dim=0)
    manual_leaf_tran = torch.cat([t for _, t in manual_leaf], dim=0)
    manual_rsot = torch.cat(manual_rsot, dim=0)
    manual_rdd = torch.cat(manual_rdd, dim=0)

    assert torch.allclose(outputs["leaf_refl"], manual_leaf_refl)
    assert torch.allclose(outputs["leaf_tran"], manual_leaf_tran)
    assert torch.allclose(outputs["rsot"], manual_rsot)
    assert torch.allclose(outputs["rdd"], manual_rdd)
