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
      fluorescence.py      # One-pass, layered, and biochemical canopy fluorescence
      thermal.py           # Spectral and integrated canopy thermal RT
    biochem/
      leaf.py              # Leaf physiology and fluorescence-yield drivers
    energy/
      balance.py           # Coupled energy-balance closure + coupled RT handoff
    runners/
      grid.py              # ROI/time runner for reflectance, fluorescence, thermal, and energy balance
    spectral/
      fluspect.py          # Leaf optics + fluorescence (PyTorch translation)
      loaders.py           # Upstream SCOPE optical asset loaders
      soil.py              # Soil library and BSM soil optics
PLAN.md                    # Detailed implementation roadmap + physical equations
prepare_scope_input.py     # Legacy preprocessing prototype (ROI -> NetCDF)
scope_grid_netcdf_inmemory_refactored.m  # Legacy MATLAB grid runner reference
```

## Development Roadmap
See [PLAN.md](PLAN.md) for the physics summary, staged translation plan, and GPU-oriented design notes. Short version:
1. **Core physics stack** → leaf optics, 4SAIL reflectance, layered fluorescence, thermal RT, leaf biochemistry, and energy balance are now implemented.
2. **Current parity status** → the benchmark harness now scales to the full 100-case upstream Latin-hypercube suite, and there is now a separate 30-step upstream time-series parity sweep. Converged scene and time-series steps are locked for reflectance, fluorescence, thermal RT, and the coupled energy products, while non-converged upstream `ebal` cases such as scene `042` and time-series step `026` are tracked separately as stress diagnostics.
3. **Current highest priority** → make the grid path lazy, metadata-preserving, and able to write `xarray`/NetCDF outputs, then wire the benchmark policy into CI.
4. **Regression infrastructure next** → add CI plus CPU-vs-GPU and batched-vs-single regression coverage around the widened MATLAB benchmark suite.

## Testing
After installing the project and dev dependencies, run the unit tests with

```bash
PYTHONPATH=src python -m pytest -q
```

Current coverage is strongest for the implemented kernels:

- `tests/spectral/test_fluspect.py` compares the leaf optics implementation to an analytically equivalent NumPy reference.
- `tests/canopy/test_foursail.py` checks the canopy solver against `prosail`'s 4SAIL implementation.
- `tests/canopy/test_fluorescence.py`, `tests/canopy/test_thermal.py`, and `tests/energy/test_balance.py` cover layered fluorescence, thermal RT, and coupled energy balance.
- `tests/test_scope_grid_runner.py` verifies that the ROI/time runner matches manual single-scene execution paths for reflectance, fluorescence, thermal, and energy-balance workflows.

MATLAB parity tooling is also available:

- `tests/test_scope_benchmark_parity.py` is the opt-in single-scene MATLAB parity gate.
- `tests/test_scope_timeseries_benchmark_parity.py` is the opt-in MATLAB time-series parity gate.
- `scripts/compare_scope_benchmark.py` compares one exported MATLAB fixture against the Python implementation.
- `scripts/run_scope_benchmark_suite.py` now sweeps the full upstream Latin-hypercube case set by default and writes `tests/data/scope_benchmark_suite_summary.json`.
- `scripts/run_scope_timeseries_benchmark_suite.py` sweeps the upstream 30-step verification time series and writes `tests/data/scope_timeseries_benchmark_summary.json`.
