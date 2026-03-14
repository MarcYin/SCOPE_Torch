"""Backward-compatible aliases for the historical `scope_torch` package name."""

from __future__ import annotations

import importlib
import sys
import warnings

from scope import *  # noqa: F401,F403
from scope import __all__ as __all__
from scope import __version__


_ALIASED_MODULES = (
    "biochem",
    "biochem.leaf",
    "canopy",
    "canopy.fluorescence",
    "canopy.foursail",
    "canopy.layered_rt",
    "canopy.reflectance",
    "canopy.thermal",
    "cli",
    "cli.fetch_upstream",
    "cli.prepare_scope_input",
    "config",
    "data",
    "data.grid",
    "energy",
    "energy.balance",
    "energy.fluxes",
    "io",
    "io.export",
    "io.prepare",
    "runners",
    "runners.grid",
    "spectral",
    "spectral.fluspect",
    "spectral.loaders",
    "spectral.optics",
    "spectral.soil",
)


for module_name in _ALIASED_MODULES:
    legacy_name = f"{__name__}.{module_name}"
    target_name = f"scope.{module_name}"
    sys.modules.setdefault(legacy_name, importlib.import_module(target_name))


warnings.warn(
    "`scope_torch` is deprecated; import `scope` instead.",
    DeprecationWarning,
    stacklevel=2,
)
