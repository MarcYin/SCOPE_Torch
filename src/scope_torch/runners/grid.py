from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from ..canopy.foursail import FourSAILModel
from ..spectral.fluspect import FluspectModel, LeafBioBatch
from ..spectral.loaders import SoilSpectraLibrary, load_soil_spectra
from ..data import ScopeGridDataModule


class ScopeGridRunner:
    """Dispatch batched SCOPE simulations across ROI/time grids."""

    def __init__(
        self,
        fluspect: FluspectModel,
        sail: FourSAILModel,
        *,
        lidf: torch.Tensor,
        default_hotspot: float = 0.2,
        soil_spectra: Optional[SoilSpectraLibrary] = None,
        soil_index_base: int = 1,
    ) -> None:
        self.fluspect = fluspect
        self.sail = sail
        self.lidf = lidf
        self.default_hotspot = default_hotspot
        self.soil_spectra = soil_spectra
        self.soil_index_base = soil_index_base

    @classmethod
    def from_scope_assets(
        cls,
        *,
        lidf: torch.Tensor,
        sail: Optional[FourSAILModel] = None,
        fluspect_path: Optional[str] = None,
        soil_path: Optional[str] = None,
        scope_root_path: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        ndub: int = 15,
        doublings_step: int = 5,
        default_hotspot: float = 0.2,
        soil_index_base: int = 1,
    ) -> "ScopeGridRunner":
        fluspect = FluspectModel.from_scope_assets(
            fluspect_path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
            ndub=ndub,
            doublings_step=doublings_step,
        )
        soil_spectra = load_soil_spectra(
            soil_path,
            scope_root_path=scope_root_path,
            device=fluspect.device,
            dtype=fluspect.dtype,
        )
        sail_model = sail if sail is not None else FourSAILModel(lidf=lidf)
        return cls(
            fluspect,
            sail_model,
            lidf=lidf,
            default_hotspot=default_hotspot,
            soil_spectra=soil_spectra,
            soil_index_base=soil_index_base,
        )

    def run(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        leaf_refl: list[torch.Tensor] = []
        leaf_tran: list[torch.Tensor] = []
        canopy_rsot: list[torch.Tensor] = []
        canopy_rdd: list[torch.Tensor] = []
        for batch in data_module.iter_batches():
            leaf_kwargs = self._leafbio_kwargs(batch, varmap)
            leafbio = LeafBioBatch(**leaf_kwargs)
            leafopt = self.fluspect(leafbio)
            leaf_refl.append(leafopt.refl)
            leaf_tran.append(leafopt.tran)

            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            tto = batch[varmap["tto"]]
            psi = batch[varmap["psi"]]
            soil = self._soil_refl(batch, varmap)
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            sail_out = self.sail(
                leafopt.refl,
                leafopt.tran,
                soil,
                lai,
                hotspot,
                tts,
                tto,
                psi,
                lidf=self.lidf,
            )
            canopy_rsot.append(sail_out.rsot)
            canopy_rdd.append(sail_out.rdd)

        return {
            "leaf_refl": torch.cat(leaf_refl, dim=0),
            "leaf_tran": torch.cat(leaf_tran, dim=0),
            "rsot": torch.cat(canopy_rsot, dim=0),
            "rdd": torch.cat(canopy_rdd, dim=0),
        }

    def _leafbio_kwargs(self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str]) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, torch.Tensor] = {}
        for field in LeafBioBatch.__dataclass_fields__:
            if field in varmap and varmap[field] in batch:
                kwargs[field] = batch[varmap[field]]
        return kwargs

    def _soil_refl(self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str]) -> torch.Tensor:
        if "soil_refl" in varmap and varmap["soil_refl"] in batch:
            return batch[varmap["soil_refl"]]
        if "soil_spectrum" in varmap and varmap["soil_spectrum"] in batch:
            if self.soil_spectra is None:
                raise ValueError("soil_spectrum was provided but no soil spectra library is configured")
            return self.soil_spectra.batch(batch[varmap["soil_spectrum"]], index_base=self.soil_index_base)
        raise KeyError("varmap must provide either 'soil_refl' or 'soil_spectrum'")
