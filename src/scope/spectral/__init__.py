"""Spectral models and input loaders for SCOPE."""

from .fluspect import FluspectModel, LeafBioBatch, LeafOptics, OptiPar, SpectralGrids
from .loaders import FluspectResources, SoilSpectraLibrary, load_fluspect_resources, load_scope_filenames, load_soil_spectra, scope_repo_root, scope_root
from .soil import BSMSoilParameters, SoilBSMModel, SoilEmpiricalParams

__all__ = [
    "FluspectModel",
    "LeafBioBatch",
    "LeafOptics",
    "OptiPar",
    "SpectralGrids",
    "FluspectResources",
    "SoilSpectraLibrary",
    "load_fluspect_resources",
    "load_scope_filenames",
    "load_soil_spectra",
    "scope_repo_root",
    "scope_root",
    "BSMSoilParameters",
    "SoilBSMModel",
    "SoilEmpiricalParams",
]
