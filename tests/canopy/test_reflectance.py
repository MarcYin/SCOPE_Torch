import torch

from scope_torch.canopy.foursail import campbell_lidf
from scope_torch.canopy.reflectance import CanopyReflectanceModel
from scope_torch.spectral.fluspect import LeafBioBatch


def test_canopy_reflectance_outputs_remain_internally_consistent():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyReflectanceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
    )
    soil = model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))

    result = model(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
    )

    assert torch.allclose(result.rso, result.rsos + result.rsod, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rsot, result.rsost + result.rsodt, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rddt, result.rdd, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rsdt, result.rsd, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rdot, result.rdo, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rsot, result.rso, atol=1e-12, rtol=1e-10)
