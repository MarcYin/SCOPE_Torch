# SCOPE Torch

PyTorch-first port of the [SCOPE](https://github.com/Christiaanvandertol/SCOPE) canopy radiative transfer model with support for batched simulations, differentiable components, and GPU acceleration.

## Goals
- Maintain parity with the original MATLAB/Fortran implementations for all radiative, fluorescence, and energy balance components.
- Run many simulations (space *or* time grids) simultaneously so weather reanalyses, EO retrievals, and tower data can be fused efficiently.
- Enable gradient-based workflows (calibration, inversion, UQ) by keeping every module differentiable end-to-end.

## Repository Layout

```
src/
  scope_torch/
    config.py              # Simulation/IO dataclasses + device helpers
    data/grid.py           # ROI/time ingestion + batching (xarray -> torch)
    canopy/
      foursail.py          # Batched 4SAIL canopy reflectance core
    runners/
      grid.py              # Minimal ROI/time runner over fluspect + 4SAIL
    spectral/
      fluspect.py          # Leaf optics + fluorescence (PyTorch translation)
PLAN.md                    # Detailed implementation roadmap + physical equations
prepare_scope_input.py     # Legacy preprocessing prototype (ROI -> NetCDF)
scope_grid_netcdf_inmemory_refactored.m  # Legacy MATLAB grid runner reference
```

## Development Roadmap
See [PLAN.md](PLAN.md) for the physics summary, staged translation plan, and GPU-oriented design notes. Short version:
1. **Leaf optics + 4SAIL canopy core** → already implemented and unit-tested.
2. **Kernel hardening + real SCOPE inputs** → remove CPU-only paths, load upstream optical parameters, and add CI-backed verification.
3. **SCOPE canopy extensions** → add `RTMf`, `RTMt`, `RTMz`, soil optics, and stable SCOPE-facing output contracts.
4. **Biochemistry + energy balance** → port FvCB, stomatal conductance, and Newton energy closure.
5. **Grid runners + IO** → turn the prototype `ScopeGridRunner` and `prepare_scope_input.py` flow into a production ROI/time pipeline.

## Testing
After installing the project and dev dependencies, run the unit tests with

```bash
python3 -m pytest
```

Current coverage is strongest for the implemented kernels:

- `tests/spectral/test_fluspect.py` compares the leaf optics implementation to an analytically equivalent NumPy reference.
- `tests/canopy/test_foursail.py` checks the canopy solver against `prosail`'s 4SAIL implementation.
- `tests/test_scope_grid_runner.py` verifies that the ROI/time runner matches a manual single-scene execution path.
