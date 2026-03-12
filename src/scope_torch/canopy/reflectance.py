from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch

from .foursail import FourSAILModel, FourSAILResult
from .layered_rt import LayeredCanopyTransportModel
from ..spectral.fluspect import FluspectModel, LeafBioBatch, LeafOptics
from ..spectral.loaders import SoilSpectraLibrary, load_fluspect_resources, load_soil_spectra
from ..spectral.soil import BSMSoilParameters, SoilBSMModel, SoilEmpiricalParams


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
        soil_spectra: Optional[SoilSpectraLibrary] = None,
        soil_bsm: Optional[SoilBSMModel] = None,
        soil_index_base: int = 1,
    ) -> None:
        self.fluspect = fluspect
        self.sail = sail
        self.layered_transport = LayeredCanopyTransportModel(sail)
        self.lidf = lidf
        self.default_hotspot = default_hotspot
        self.soil_spectra = soil_spectra
        self.soil_bsm = soil_bsm
        self.soil_index_base = soil_index_base

    @classmethod
    def from_scope_assets(
        cls,
        *,
        lidf: torch.Tensor,
        sail: Optional[FourSAILModel] = None,
        path: Optional[str] = None,
        soil_path: Optional[str] = None,
        scope_root_path: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        ndub: int = 15,
        doublings_step: int = 5,
        default_hotspot: float = 0.2,
        soil_index_base: int = 1,
        soil_empirical: SoilEmpiricalParams | None = None,
    ) -> "CanopyReflectanceModel":
        resources = load_fluspect_resources(
            path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
        )
        fluspect = FluspectModel(
            resources.spectral,
            resources.optipar,
            ndub=ndub,
            doublings_step=doublings_step,
            device=torch.device(device) if device is not None else resources.optipar.Kw.device,
            dtype=dtype,
        )
        soil_spectra = load_soil_spectra(
            soil_path,
            scope_root_path=scope_root_path,
            device=fluspect.device,
            dtype=fluspect.dtype,
        )
        soil_bsm = SoilBSMModel.from_resources(resources, empirical=soil_empirical, device=fluspect.device, dtype=fluspect.dtype)
        sail_model = sail if sail is not None else FourSAILModel(lidf=lidf)
        return cls(
            fluspect,
            sail_model,
            lidf=lidf,
            default_hotspot=default_hotspot,
            soil_spectra=soil_spectra,
            soil_bsm=soil_bsm,
            soil_index_base=soil_index_base,
        )

    def soil_reflectance(
        self,
        *,
        soil_refl: Optional[torch.Tensor] = None,
        soil_spectrum: Optional[torch.Tensor] = None,
        bsm: BSMSoilParameters | None = None,
        BSMBrightness: Optional[torch.Tensor] = None,
        BSMlat: Optional[torch.Tensor] = None,
        BSMlon: Optional[torch.Tensor] = None,
        SMC: Optional[torch.Tensor] = None,
        soil_index_base: Optional[int] = None,
    ) -> torch.Tensor:
        if soil_refl is not None:
            return torch.as_tensor(soil_refl, device=self.fluspect.device, dtype=self.fluspect.dtype)

        if soil_spectrum is not None:
            if self.soil_spectra is None:
                raise ValueError("soil_spectrum was provided but no soil spectra library is configured")
            return self.soil_spectra.batch(
                soil_spectrum,
                index_base=self.soil_index_base if soil_index_base is None else soil_index_base,
            )

        if bsm is None and all(value is not None for value in (BSMBrightness, BSMlat, BSMlon, SMC)):
            bsm = BSMSoilParameters(
                BSMBrightness=BSMBrightness,
                BSMlat=BSMlat,
                BSMlon=BSMlon,
                SMC=SMC,
            )
        if bsm is not None:
            if self.soil_bsm is None:
                raise ValueError("BSM soil inputs were provided but no BSM soil model is configured")
            return self.soil_bsm(bsm)

        raise KeyError("Provide either soil_refl, soil_spectrum, or BSM soil parameters")

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
        rtmo_core = self._rtmo_optical_reflectance(
            leafopt=leafopt,
            soil_refl=soil_refl,
            lai=lai,
            tts=tts,
            tto=tto,
            psi=psi,
            hotspot=hotspot_value,
            lidf=self.lidf if lidf is None else lidf,
        )
        sail_out.rdd = rtmo_core["rdd"]
        sail_out.rsd = rtmo_core["rsd"]
        sail_out.rdo = rtmo_core["rdo"]
        sail_out.rso = rtmo_core["rso"]
        sail_out.rsos = rtmo_core["rsos"]
        sail_out.rsod = rtmo_core["rsod"]
        # The layered RTMo terms already include the soil boundary condition, so
        # the 4SAIL "total" aliases should track those values directly rather
        # than retaining stale pre-override four-stream totals.
        sail_out.rddt = rtmo_core["rdd"]
        sail_out.rsdt = rtmo_core["rsd"]
        sail_out.rdot = rtmo_core["rdo"]
        sail_out.rsost = rtmo_core["rsos"]
        sail_out.rsodt = rtmo_core["rsod"]
        sail_out.rsot = rtmo_core["rso"]
        return CanopyReflectanceResult.from_components(leafopt, sail_out)

    def _rtmo_optical_reflectance(
        self,
        *,
        leafopt: LeafOptics,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        hotspot: torch.Tensor,
        lidf: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        soil = self.sail._ensure_2d(soil_refl, target_shape=leafopt.refl.shape)
        batch, nwl = leafopt.refl.shape
        outputs = {"rdd": [], "rsd": [], "rdo": [], "rso": [], "rsos": [], "rsod": []}
        for idx in range(batch):
            nlayers = max(2, int(torch.ceil(torch.as_tensor(lai[idx], device=leafopt.refl.device, dtype=leafopt.refl.dtype) * 10.0).item()))
            transfer = self.layered_transport.build(
                leafopt.refl[idx : idx + 1],
                leafopt.tran[idx : idx + 1],
                soil[idx : idx + 1],
                lai[idx : idx + 1],
                tts[idx : idx + 1],
                tto[idx : idx + 1],
                psi[idx : idx + 1],
                hotspot=hotspot[idx : idx + 1],
                lidf=lidf[idx : idx + 1] if lidf.ndim == 2 else lidf,
                nlayers=nlayers,
            )
            unit = torch.ones((1, nwl), device=leafopt.refl.device, dtype=leafopt.refl.dtype)
            zeros = torch.zeros_like(unit)
            direct = self.layered_transport.flux_profiles(transfer, unit, zeros)
            diffuse = self.layered_transport.flux_profiles(transfer, zeros, unit)

            Po = transfer.Po[:, : transfer.nlayers].unsqueeze(-1)
            Pso = transfer.Pso[:, : transfer.nlayers].unsqueeze(-1)
            iLAI = transfer.iLAI.view(1, 1, 1)

            piLocd = iLAI * torch.sum(
                (transfer.vb.unsqueeze(1) * diffuse.Emin_[:, : transfer.nlayers, :] + transfer.vf.unsqueeze(1) * diffuse.Eplu_[:, : transfer.nlayers, :]) * Po,
                dim=1,
            )
            piLosd = soil[idx : idx + 1] * diffuse.Emin_[:, -1, :] * transfer.Po[:, -1].unsqueeze(-1)
            piLocu_diffuse = iLAI * torch.sum(
                transfer.vb.unsqueeze(1) * direct.Emin_[:, : transfer.nlayers, :] * Po
                + transfer.vf.unsqueeze(1) * direct.Eplu_[:, : transfer.nlayers, :] * Po,
                dim=1,
            )
            piLocu_hotspot = iLAI * torch.sum(transfer.w.unsqueeze(1) * Pso, dim=1)
            piLosu_diffuse = soil[idx : idx + 1] * direct.Emin_[:, -1, :] * transfer.Po[:, -1].unsqueeze(-1)
            piLosu_hotspot = soil[idx : idx + 1] * transfer.Pso[:, -1].unsqueeze(-1)
            rsod = piLocu_diffuse + piLosu_diffuse
            rsos = piLocu_hotspot + piLosu_hotspot

            outputs["rdd"].append(transfer.R_dd[:, 0, :])
            outputs["rsd"].append(transfer.R_sd[:, 0, :])
            outputs["rdo"].append(piLocd + piLosd)
            outputs["rso"].append(rsod + rsos)
            outputs["rsos"].append(rsos)
            outputs["rsod"].append(rsod)

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}
