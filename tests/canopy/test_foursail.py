import numpy as np
import torch

from prosail.FourSAIL import foursail as foursail_np

from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf


def test_foursail_matches_prosail():
    device = torch.device("cpu")
    dtype = torch.float64
    nwl = 64
    torch.manual_seed(0)
    rho = torch.rand(nwl, device=device, dtype=dtype) * 0.2 + 0.05
    tau = torch.rand(nwl, device=device, dtype=dtype) * 0.2 + 0.03
    soil = torch.rand(nwl, device=device, dtype=dtype) * 0.3 + 0.1
    lai = torch.tensor(3.2, device=device, dtype=dtype)
    hotspot = torch.tensor(0.2, device=device, dtype=dtype)
    tts = torch.tensor(35.0, device=device, dtype=dtype)
    tto = torch.tensor(20.0, device=device, dtype=dtype)
    psi = torch.tensor(10.0, device=device, dtype=dtype)
    lidfa = 57.0

    lidf = campbell_lidf(lidfa, device=device, dtype=dtype)
    model = FourSAILModel(lidf=lidf)
    torch_out = model(rho, tau, soil, lai, hotspot, tts, tto, psi)

    np_result = foursail_np(
        rho.cpu().numpy(),
        tau.cpu().numpy(),
        lidfa,
        0.0,
        2,
        float(lai.item()),
        float(hotspot.item()),
        float(tts.item()),
        float(tto.item()),
        float(psi.item()),
        soil.cpu().numpy(),
    )
    # unpack numpy outputs
    keys = [
        "tss",
        "too",
        "tsstoo",
        "rdd",
        "tdd",
        "rsd",
        "tsd",
        "rdo",
        "tdo",
        "rso",
        "rsos",
        "rsod",
        "rddt",
        "rsdt",
        "rdot",
        "rsodt",
        "rsost",
        "rsot",
        "gammasdf",
        "gammasdb",
        "gammaso",
    ]
    numpy_out = dict(zip(keys, np_result))

    assert np.allclose(torch_out.rdd.cpu().numpy(), numpy_out["rdd"], atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.rsd.cpu().numpy(), numpy_out["rsd"], atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.rdo.cpu().numpy(), numpy_out["rdo"], atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.rso.cpu().numpy(), numpy_out["rso"], atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.rsot.cpu().numpy(), numpy_out["rsot"], atol=1e-8, rtol=1e-6)
