from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from scipy.io import loadmat

from .fluspect import OptiPar, SpectralGrids


@dataclass(slots=True)
class FluspectResources:
    """Loaded FLUSPECT optical parameters together with their spectral grid."""

    spectral: SpectralGrids
    optipar: OptiPar
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    source: Path | None = None


@dataclass(slots=True)
class SoilSpectraLibrary:
    """Collection of soil reflectance spectra loaded from a SCOPE soil file."""

    wavelength: torch.Tensor
    spectra: torch.Tensor
    names: tuple[str, ...]
    source: Path | None = None

    def spectrum(self, key: int | str) -> torch.Tensor:
        if isinstance(key, str):
            try:
                index = self.names.index(key)
            except ValueError as exc:
                raise KeyError(f"Unknown soil spectrum '{key}'") from exc
            return self.spectra[:, index]
        return self.spectra[:, key]

    def batch(self, indices: torch.Tensor, *, index_base: int = 1) -> torch.Tensor:
        """Gather a batch of soil spectra using SCOPE-style spectrum indices."""

        indices = torch.as_tensor(indices, device=self.spectra.device)
        if indices.ndim != 1:
            raise ValueError(f"Expected 1D soil spectrum indices, got shape {tuple(indices.shape)}")
        zero_based = torch.round(indices).to(torch.int64) - index_base
        if zero_based.numel() == 0:
            return self.spectra.new_empty((0, self.spectra.shape[0]))
        if torch.any(zero_based < 0) or torch.any(zero_based >= self.spectra.shape[1]):
            valid = f"{index_base}..{self.spectra.shape[1] + index_base - 1}"
            raise IndexError(f"Soil spectrum indices must be in {valid}")
        return self.spectra.transpose(0, 1).index_select(0, zero_based)


def scope_repo_root() -> Path:
    """Return the repository root for the editable workspace checkout."""

    return Path(__file__).resolve().parents[3]


def scope_root(scope_root: str | Path | None = None) -> Path:
    """Resolve the vendored upstream SCOPE checkout used by this project."""

    root = Path(scope_root) if scope_root is not None else scope_repo_root() / "upstream" / "SCOPE"
    if not root.exists():
        raise FileNotFoundError(f"Could not find SCOPE root at {root}")
    return root


def load_scope_filenames(scope_root_path: str | Path | None = None) -> dict[str, str]:
    """Parse `input/filenames.csv` and return the first non-empty value per key."""

    filenames_path = scope_root(scope_root_path) / "input" / "filenames.csv"
    entries: dict[str, str] = {}
    with filenames_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip()
            value = row[1].strip()
            if not key:
                continue
            if key not in entries or (not entries[key] and value):
                entries[key] = value
    return entries


def load_fluspect_resources(
    path: str | Path | None = None,
    *,
    scope_root_path: str | Path | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    wlF: torch.Tensor | None = None,
    wlE: torch.Tensor | None = None,
) -> FluspectResources:
    """Load upstream FLUSPECT optical parameters from a MATLAB `.mat` file."""

    resolved_path = _resolve_scope_input_path(
        path,
        scope_root_path=scope_root_path,
        subdir="fluspect_parameters",
        filename_key="optipar_file",
    )
    data = loadmat(resolved_path, simplify_cells=True)
    raw = data.get("optipar")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected 'optipar' struct in {resolved_path}")

    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    wavelength = _to_tensor_1d(raw["wl"], device=device_obj, dtype=dtype)
    spectral = SpectralGrids(wlP=wavelength, wlF=_maybe_tensor_1d(wlF, device_obj, dtype), wlE=_maybe_tensor_1d(wlE, device_obj, dtype))
    optipar = OptiPar(
        nr=_to_tensor_1d(raw["nr"], device=device_obj, dtype=dtype),
        Kab=_to_tensor_1d(raw["Kab"], device=device_obj, dtype=dtype),
        Kca=_to_tensor_1d(raw["Kca"], device=device_obj, dtype=dtype),
        KcaV=_to_tensor_1d(raw["KcaV"], device=device_obj, dtype=dtype),
        KcaZ=_to_tensor_1d(raw["KcaZ"], device=device_obj, dtype=dtype),
        Kdm=_to_tensor_1d(raw["Kdm"], device=device_obj, dtype=dtype),
        Kw=_to_tensor_1d(raw["Kw"], device=device_obj, dtype=dtype),
        Ks=_to_tensor_1d(raw["Ks"], device=device_obj, dtype=dtype),
        Kant=_to_tensor_1d(raw["Kant"], device=device_obj, dtype=dtype),
        phi=_to_tensor_1d(raw["phi"], device=device_obj, dtype=dtype),
        Kp=_optional_tensor_1d(raw.get("Kp"), device=device_obj, dtype=dtype),
        Kcbc=_optional_tensor_1d(raw.get("Kcbc"), device=device_obj, dtype=dtype),
    )

    extras: dict[str, torch.Tensor] = {}
    for key in ("phiI", "phiII", "phiE", "GSV", "nw"):
        if key in raw:
            extras[key] = _to_tensor(raw[key], device=device_obj, dtype=dtype)

    return FluspectResources(spectral=spectral, optipar=optipar, extras=extras, source=resolved_path)


def load_soil_spectra(
    path: str | Path | None = None,
    *,
    scope_root_path: str | Path | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> SoilSpectraLibrary:
    """Load soil reflectance spectra from a SCOPE text file."""

    resolved_path = _resolve_scope_input_path(
        path,
        scope_root_path=scope_root_path,
        subdir="soil_spectra",
        filename_key="soil_file",
    )
    values = np.loadtxt(resolved_path)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Expected wavelength + one or more soil spectra in {resolved_path}")

    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    wavelength = torch.as_tensor(values[:, 0], device=device_obj, dtype=dtype)
    spectra = torch.as_tensor(values[:, 1:], device=device_obj, dtype=dtype)
    names = tuple(f"soil{i}" for i in range(1, spectra.shape[1] + 1))
    return SoilSpectraLibrary(wavelength=wavelength, spectra=spectra, names=names, source=resolved_path)


def _resolve_scope_input_path(
    path: str | Path | None,
    *,
    scope_root_path: str | Path | None,
    subdir: str,
    filename_key: str,
) -> Path:
    root = scope_root(scope_root_path)
    input_dir = root / "input"
    if path is None:
        default_name = load_scope_filenames(root).get(filename_key, "")
        if not default_name:
            raise FileNotFoundError(f"No default value for '{filename_key}' found in {input_dir / 'filenames.csv'}")
        candidate = Path(default_name)
    else:
        candidate = Path(path)

    search_paths = []
    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.extend(
            [
                input_dir / subdir / candidate,
                input_dir / candidate,
                root / candidate,
                scope_repo_root() / candidate,
            ]
        )

    for resolved in search_paths:
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve {candidate} under {input_dir}")


def _maybe_tensor_1d(value: torch.Tensor | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if value is None:
        return None
    return _to_tensor_1d(value, device=device, dtype=dtype)


def _optional_tensor_1d(value: object | None, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if value is None:
        return None
    return _to_tensor_1d(value, device=device, dtype=dtype)


def _to_tensor_1d(value: object, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    elif array.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape {array.shape}")
    return torch.as_tensor(array, device=device, dtype=dtype)


def _to_tensor(value: object, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)
