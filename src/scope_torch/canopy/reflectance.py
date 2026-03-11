from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch

from .foursail import FourSAILModel, FourSAILResult
from ..spectral.fluspect import FluspectModel, LeafBioBatch, LeafOptics


@dataclass(slots=True)
class CanopyReflectanceResult:
    leaf_refl: torch.Tensor
    leaf_tran: torch.Tensor
    rdd: torch.Tensor
    tdd: torch.Tensor
    rsd: torch.Tensor
    tsd: torch.Tensor
    rdo: torch.Tensor
    tdo: torch.Tensor
    rso: torch.Tensor
    rsos: torch.Tensor
    rsod: torch.Tensor
    rddt: torch.Tensor
    rsdt: torch.Tensor
    rdot: torch.Tensor
    rsodt: torch.Tensor
    rsost: torch.Tensor
    rsot: torch.Tensor
    tss: torch.Tensor
    too: torch.Tensor
    tsstoo: torch.Tensor
    gammasdf: torch.Tensor
    gammasdb: torch.Tensor
    gammaso: torch.Tensor

    @classmethod
    def from_components(cls, leafopt: LeafOptics, sail: FourSAILResult) -> "CanopyReflectanceResult":
        return cls(
            leaf_refl=leafopt.refl,
            leaf_tran=leafopt.tran,
            **{field.name: getattr(sail, field.name) for field in fields(FourSAILResult)},
        )


class CanopyReflectanceModel:
    """Stable SCOPE-facing reflectance wrapper around leaf optics + 4SAIL."""

    def __init__(
        self,
        fluspect: FluspectModel,
        sail: FourSAILModel,
        *,
        lidf: torch.Tensor,
        default_hotspot: float = 0.2,
    ) -> None:
        self.fluspect = fluspect
        self.sail = sail
        self.lidf = lidf
        self.default_hotspot = default_hotspot

    @classmethod
    def from_scope_assets(
        cls,
        *,
        lidf: torch.Tensor,
        sail: Optional[FourSAILModel] = None,
        path: Optional[str] = None,
        scope_root_path: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        ndub: int = 15,
        doublings_step: int = 5,
        default_hotspot: float = 0.2,
    ) -> "CanopyReflectanceModel":
        fluspect = FluspectModel.from_scope_assets(
            path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
            ndub=ndub,
            doublings_step=doublings_step,
        )
        sail_model = sail if sail is not None else FourSAILModel(lidf=lidf)
        return cls(fluspect, sail_model, lidf=lidf, default_hotspot=default_hotspot)

    def __call__(
        self,
        leafbio: LeafBioBatch,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        *,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
    ) -> CanopyReflectanceResult:
        leafopt = self.fluspect(leafbio)
        hotspot_value = hotspot if hotspot is not None else torch.full_like(torch.as_tensor(lai), self.default_hotspot)
        sail_out = self.sail(
            leafopt.refl,
            leafopt.tran,
            soil_refl,
            lai,
            hotspot_value,
            tts,
            tto,
            psi,
            lidf=self.lidf if lidf is None else lidf,
        )
        return CanopyReflectanceResult.from_components(leafopt, sail_out)
