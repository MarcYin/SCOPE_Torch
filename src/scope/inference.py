from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import torch

from .biochem import BiochemicalOptions, LeafBiochemistryInputs
from .canopy.fluorescence import CanopyFluorescenceModel
from .canopy.foursail import FourSAILModel
from .canopy.reflectance import CanopyReflectanceModel
from .canopy.thermal import CanopyThermalRadianceModel
from .energy import (
    CanopyEnergyBalanceModel,
    EnergyBalanceCanopy,
    EnergyBalanceMeteo,
    EnergyBalanceOptions,
    EnergyBalanceSoil,
)
from .runners import ScopeGridRunner
from .spectral.fluspect import FluspectModel, LeafBioBatch
from .spectral.loaders import SoilSpectraLibrary
from .spectral.soil import BSMSoilParameters, SoilBSMModel, SoilEmpiricalParams


def _dataclass_field_map(value: object) -> dict[str, Any]:
    if not is_dataclass(value):
        return {}
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _resolve_output_path(result: object, path: str) -> Any:
    current = result
    traversed: list[str] = []
    for part in path.split("."):
        traversed.append(part)
        if not hasattr(current, part):
            available = ", ".join(sorted(_dataclass_field_map(current))) or "<none>"
            prefix = ".".join(traversed[:-1]) or type(current).__name__
            raise KeyError(f"Output '{path}' is not available at '{prefix}'. Available fields: {available}")
        current = getattr(current, part)
    return current


def select_inference_outputs(result: object, outputs: Sequence[str] | None) -> object:
    """Return a lightweight mapping for selected result fields.

    Field names may be direct dataclass fields such as ``rsot`` or dotted paths
    such as ``energy.Rntot`` / ``thermal.Lot_`` for coupled results.
    """

    if outputs is None:
        return result
    return {name: _resolve_output_path(result, name) for name in outputs}


class ScopeInferenceModel:
    """Lightweight tensor-facing inference API for production workflows.

    This surface avoids the xarray/grid orchestration layer and lets callers
    request only the output fields they need for repeated model inference.
    """

    def __init__(
        self,
        fluspect: FluspectModel,
        sail: FourSAILModel,
        *,
        lidf: torch.Tensor,
        default_hotspot: float = 0.2,
        soil_spectra: SoilSpectraLibrary | None = None,
        soil_bsm: SoilBSMModel | None = None,
        soil_index_base: int = 1,
    ) -> None:
        self.fluspect = fluspect
        self.sail = sail
        self.lidf = lidf
        self.default_hotspot = default_hotspot
        self.soil_spectra = soil_spectra
        self.soil_bsm = soil_bsm
        self.soil_index_base = soil_index_base
        self.reflectance_model = CanopyReflectanceModel(
            fluspect,
            sail,
            lidf=lidf,
            default_hotspot=default_hotspot,
            soil_spectra=soil_spectra,
            soil_bsm=soil_bsm,
            soil_index_base=soil_index_base,
        )
        self.fluorescence_model = CanopyFluorescenceModel(self.reflectance_model)
        self.thermal_model = CanopyThermalRadianceModel(self.reflectance_model)
        self.energy_balance_model = CanopyEnergyBalanceModel(self.reflectance_model)

    @classmethod
    def from_scope_assets(
        cls,
        *,
        lidf: torch.Tensor,
        sail: FourSAILModel | None = None,
        fluspect_path: str | None = None,
        soil_path: str | None = None,
        scope_root_path: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        ndub: int = 15,
        doublings_step: int = 5,
        default_hotspot: float = 0.2,
        soil_index_base: int = 1,
        soil_empirical: SoilEmpiricalParams | None = None,
    ) -> ScopeInferenceModel:
        runner = ScopeGridRunner.from_scope_assets(
            lidf=lidf,
            sail=sail,
            fluspect_path=fluspect_path,
            soil_path=soil_path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
            ndub=ndub,
            doublings_step=doublings_step,
            default_hotspot=default_hotspot,
            soil_index_base=soil_index_base,
            soil_empirical=soil_empirical,
        )
        return cls(
            runner.fluspect,
            runner.sail,
            lidf=runner.lidf,
            default_hotspot=runner.default_hotspot,
            soil_spectra=runner.soil_spectra,
            soil_bsm=runner.soil_bsm,
            soil_index_base=runner.soil_index_base,
        )

    def soil_reflectance(
        self,
        *,
        soil_refl: torch.Tensor | None = None,
        soil_spectrum: torch.Tensor | None = None,
        bsm: BSMSoilParameters | None = None,
        BSMBrightness: torch.Tensor | None = None,
        BSMlat: torch.Tensor | None = None,
        BSMlon: torch.Tensor | None = None,
        SMC: torch.Tensor | None = None,
        soil_index_base: int | None = None,
    ) -> torch.Tensor:
        return self.reflectance_model.soil_reflectance(
            soil_refl=soil_refl,
            soil_spectrum=soil_spectrum,
            bsm=bsm,
            BSMBrightness=BSMBrightness,
            BSMlat=BSMlat,
            BSMlon=BSMlon,
            SMC=SMC,
            soil_index_base=soil_index_base,
        )

    def reflectance(
        self,
        leafbio: LeafBioBatch,
        *,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        soil_refl: torch.Tensor | None = None,
        soil_spectrum: torch.Tensor | None = None,
        bsm: BSMSoilParameters | None = None,
        hotspot: torch.Tensor | None = None,
        lidf: torch.Tensor | None = None,
        nlayers: int | None = None,
        outputs: Sequence[str] | None = None,
        **soil_kwargs: torch.Tensor,
    ) -> object:
        soil = self.soil_reflectance(
            soil_refl=soil_refl,
            soil_spectrum=soil_spectrum,
            bsm=bsm,
            **soil_kwargs,
        )
        result = self.reflectance_model(
            leafbio,
            soil,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nlayers,
        )
        return select_inference_outputs(result, outputs)

    def fluorescence(
        self,
        leafbio: LeafBioBatch,
        *,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_: torch.Tensor,
        Esky_: torch.Tensor,
        soil_refl: torch.Tensor | None = None,
        soil_spectrum: torch.Tensor | None = None,
        bsm: BSMSoilParameters | None = None,
        etau: torch.Tensor | None = None,
        etah: torch.Tensor | None = None,
        hotspot: torch.Tensor | None = None,
        lidf: torch.Tensor | None = None,
        nlayers: int | None = None,
        outputs: Sequence[str] | None = None,
        **soil_kwargs: torch.Tensor,
    ) -> object:
        soil = self.soil_reflectance(
            soil_refl=soil_refl,
            soil_spectrum=soil_spectrum,
            bsm=bsm,
            **soil_kwargs,
        )
        result = self.fluorescence_model.layered(
            leafbio,
            soil,
            lai,
            tts,
            tto,
            psi,
            Esun_,
            Esky_,
            etau=etau,
            etah=etah,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nlayers,
        )
        return select_inference_outputs(result, outputs)

    def thermal(
        self,
        *,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Tcu: torch.Tensor,
        Tch: torch.Tensor,
        Tsu: torch.Tensor,
        Tsh: torch.Tensor,
        hotspot: torch.Tensor | None = None,
        lidf: torch.Tensor | None = None,
        nlayers: int | None = None,
        wlT: torch.Tensor | None = None,
        outputs: Sequence[str] | None = None,
    ) -> object:
        result = self.thermal_model(
            lai,
            tts,
            tto,
            psi,
            Tcu,
            Tch,
            Tsu,
            Tsh,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nlayers,
            wlT=wlT,
        )
        return select_inference_outputs(result, outputs)

    def energy_balance(
        self,
        leafbio: LeafBioBatch,
        biochemistry: LeafBiochemistryInputs,
        *,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_sw: torch.Tensor,
        Esky_sw: torch.Tensor,
        meteo: EnergyBalanceMeteo,
        canopy: EnergyBalanceCanopy,
        soil: EnergyBalanceSoil,
        soil_refl: torch.Tensor | None = None,
        soil_spectrum: torch.Tensor | None = None,
        bsm: BSMSoilParameters | None = None,
        Esun_lw: torch.Tensor | None = None,
        Esky_lw: torch.Tensor | None = None,
        wlT: torch.Tensor | None = None,
        options: EnergyBalanceOptions | None = None,
        biochem_options: BiochemicalOptions | None = None,
        hotspot: torch.Tensor | None = None,
        lidf: torch.Tensor | None = None,
        nlayers: int | None = None,
        Tcu0: torch.Tensor | None = None,
        Tch0: torch.Tensor | None = None,
        Tsu0: torch.Tensor | None = None,
        Tsh0: torch.Tensor | None = None,
        outputs: Sequence[str] | None = None,
        **soil_kwargs: torch.Tensor,
    ) -> object:
        optical_soil = self.soil_reflectance(
            soil_refl=soil_refl,
            soil_spectrum=soil_spectrum,
            bsm=bsm,
            **soil_kwargs,
        )
        result = self.energy_balance_model.solve(
            leafbio,
            biochemistry,
            optical_soil,
            lai,
            tts,
            tto,
            psi,
            Esun_sw,
            Esky_sw,
            Esun_lw=Esun_lw,
            Esky_lw=Esky_lw,
            wlT=wlT,
            meteo=meteo,
            canopy=canopy,
            soil=soil,
            options=options,
            biochem_options=biochem_options,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nlayers,
            Tcu0=Tcu0,
            Tch0=Tch0,
            Tsu0=Tsu0,
            Tsh0=Tsh0,
        )
        return select_inference_outputs(result, outputs)

    def energy_balance_fluorescence(
        self,
        leafbio: LeafBioBatch,
        biochemistry: LeafBiochemistryInputs,
        *,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_sw: torch.Tensor,
        Esky_sw: torch.Tensor,
        meteo: EnergyBalanceMeteo,
        canopy: EnergyBalanceCanopy,
        soil: EnergyBalanceSoil,
        soil_refl: torch.Tensor | None = None,
        soil_spectrum: torch.Tensor | None = None,
        bsm: BSMSoilParameters | None = None,
        Esun_lw: torch.Tensor | None = None,
        Esky_lw: torch.Tensor | None = None,
        wlT: torch.Tensor | None = None,
        options: EnergyBalanceOptions | None = None,
        biochem_options: BiochemicalOptions | None = None,
        hotspot: torch.Tensor | None = None,
        lidf: torch.Tensor | None = None,
        nlayers: int | None = None,
        outputs: Sequence[str] | None = None,
        **soil_kwargs: torch.Tensor,
    ) -> object:
        optical_soil = self.soil_reflectance(
            soil_refl=soil_refl,
            soil_spectrum=soil_spectrum,
            bsm=bsm,
            **soil_kwargs,
        )
        result = self.energy_balance_model.solve_fluorescence(
            leafbio,
            biochemistry,
            optical_soil,
            lai,
            tts,
            tto,
            psi,
            Esun_sw,
            Esky_sw,
            meteo=meteo,
            canopy=canopy,
            soil=soil,
            Esun_lw=Esun_lw,
            Esky_lw=Esky_lw,
            wlT=wlT,
            options=options,
            biochem_options=biochem_options,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nlayers,
        )
        return select_inference_outputs(result, outputs)

    def energy_balance_thermal(
        self,
        leafbio: LeafBioBatch,
        biochemistry: LeafBiochemistryInputs,
        *,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_sw: torch.Tensor,
        Esky_sw: torch.Tensor,
        meteo: EnergyBalanceMeteo,
        canopy: EnergyBalanceCanopy,
        soil: EnergyBalanceSoil,
        soil_refl: torch.Tensor | None = None,
        soil_spectrum: torch.Tensor | None = None,
        bsm: BSMSoilParameters | None = None,
        Esun_lw: torch.Tensor | None = None,
        Esky_lw: torch.Tensor | None = None,
        wlT_forcing: torch.Tensor | None = None,
        wlT: torch.Tensor | None = None,
        options: EnergyBalanceOptions | None = None,
        biochem_options: BiochemicalOptions | None = None,
        hotspot: torch.Tensor | None = None,
        lidf: torch.Tensor | None = None,
        nlayers: int | None = None,
        outputs: Sequence[str] | None = None,
        **soil_kwargs: torch.Tensor,
    ) -> object:
        optical_soil = self.soil_reflectance(
            soil_refl=soil_refl,
            soil_spectrum=soil_spectrum,
            bsm=bsm,
            **soil_kwargs,
        )
        result = self.energy_balance_model.solve_thermal(
            leafbio,
            biochemistry,
            optical_soil,
            lai,
            tts,
            tto,
            psi,
            Esun_sw,
            Esky_sw,
            meteo=meteo,
            canopy=canopy,
            soil=soil,
            Esun_lw=Esun_lw,
            Esky_lw=Esky_lw,
            wlT_forcing=wlT if wlT_forcing is None else wlT_forcing,
            wlT=wlT,
            options=options,
            biochem_options=biochem_options,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nlayers,
        )
        return select_inference_outputs(result, outputs)
