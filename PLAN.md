# SCOPE Torch Implementation Plan

## 1. Current Repository Status

### Implemented and already useful

| Area | Current implementation | Evidence |
| --- | --- | --- |
| Upstream asset loading | Real SCOPE optical assets, soil spectra, and SCOPE filename metadata can be loaded from the vendored upstream tree. | [src/scope_torch/spectral/loaders.py](src/scope_torch/spectral/loaders.py), [tests/spectral/test_loaders.py](tests/spectral/test_loaders.py) |
| Leaf optics | `FluspectModel` ports the leaf optics core, including fluorescence matrices `Mb`/`Mf`, and now stays tensor-native instead of detaching to SciPy. | [src/scope_torch/spectral/fluspect.py](src/scope_torch/spectral/fluspect.py), [tests/spectral/test_fluspect.py](tests/spectral/test_fluspect.py) |
| Canopy reflectance | `FourSAILModel` and `CanopyReflectanceModel` provide a SCOPE-facing reflectance stack with full reflectance outputs plus soil-library and BSM soil support. | [src/scope_torch/canopy/foursail.py](src/scope_torch/canopy/foursail.py), [src/scope_torch/canopy/reflectance.py](src/scope_torch/canopy/reflectance.py), [tests/canopy/test_foursail.py](tests/canopy/test_foursail.py), [tests/spectral/test_soil.py](tests/spectral/test_soil.py) |
| Canopy fluorescence | The repo includes one-pass fluorescence, layered fluorescence transport, and a physiology-coupled layered fluorescence path. | [src/scope_torch/canopy/fluorescence.py](src/scope_torch/canopy/fluorescence.py), [tests/canopy/test_fluorescence.py](tests/canopy/test_fluorescence.py) |
| Canopy thermal RT | The repo includes spectral thermal radiance and spectrally integrated thermal balance outputs with shared layered transport. | [src/scope_torch/canopy/thermal.py](src/scope_torch/canopy/thermal.py), [tests/canopy/test_thermal.py](tests/canopy/test_thermal.py) |
| Leaf biochemistry | `LeafBiochemistryModel` covers C3/C4 assimilation, Ball-Berry closure, and fluorescence-yield outputs used by the canopy models. | [src/scope_torch/biochem/leaf.py](src/scope_torch/biochem/leaf.py), [tests/biochem/test_leaf.py](tests/biochem/test_leaf.py) |
| Energy balance closure | `CanopyEnergyBalanceModel` iterates temperatures, boundary humidity/CO2, aerodynamic resistances, and coupled fluorescence/thermal outputs. | [src/scope_torch/energy/balance.py](src/scope_torch/energy/balance.py), [tests/energy/test_balance.py](tests/energy/test_balance.py) |
| Grid execution | `ScopeGridRunner` now exposes reflectance, fluorescence, biochemical fluorescence, thermal RT, and coupled energy-balance fluorescence/thermal paths over ROI/time batches. | [src/scope_torch/runners/grid.py](src/scope_torch/runners/grid.py), [tests/test_scope_grid_runner.py](tests/test_scope_grid_runner.py) |

### Present but still prototype-level

1. `prepare_scope_input.py` is still a host-specific script with hard-coded paths and no reusable library surface.
2. `ScopeGridDataModule` still stacks and materializes tensors before chunking instead of streaming lazily from the dataset.
3. `ScopeGridRunner` returns concatenated torch tensors, not metadata-preserving `xarray.Dataset` outputs.
4. The repository still lacks CI automation and institutionalized GPU/batched regression checks, even though local MATLAB parity fixtures now exist.

### Main accuracy and scope gaps

1. **The benchmark suite is widened, and the non-converged upstream case is now classified explicitly.**
   The MATLAB harness now scales to the full 100-case upstream Latin-hypercube set via `scripts/run_scope_benchmark_suite.py`, with per-case reports in `tests/data/benchmark_suite_reports/` and an aggregate summary in `tests/data/scope_benchmark_suite_summary.json`. The earlier `reflectance.refl` outlier was a benchmark-harness reconstruction bug and is now closed across the full 100-case sweep. The suite now treats upstream scenes that hit `ebal` max iterations as stress diagnostics instead of parity-gating cases. At the moment only case `042` falls into that bucket.
2. **Time-series parity is now covered separately from the scene suite.**
   `scripts/export_scope_timeseries_benchmarks.m` and `scripts/run_scope_timeseries_benchmark_suite.py` now exercise upstream SCOPE `simulation == 1` on the default 30-step verification series, writing per-step reports under `tests/data/timeseries_benchmark_reports/` and an aggregate summary in `tests/data/scope_timeseries_benchmark_summary.json`. The same non-converged-upstream policy applies there: step `026` is currently classified as a stress case because upstream MATLAB `ebal` hits `maxit`.
3. **The raw energy-balance iterate diagnostics are still easy to misread.**
   End-of-iteration same-state parity is now negligible, but the phase-lagged `energy_balance.sunlit_A` and `energy_balance.shaded_A` fields in the raw comparison reports still look worse than the true leaf-kernel parity because they compare the final leaf solve against post-update boundary states. The harness now exports like-for-like iteration inputs to separate those cases, but that distinction is not yet documented widely in the repo.
4. **GPU and batched consistency are not yet institutionalized.**
   The earlier CPU/detach hot spots are gone from the implemented kernels, but there is still no regression suite proving CPU-vs-GPU or batched-vs-single equivalence for the coupled products.
5. **Workflow parity is still narrow.**
   The model core is ahead of the workflow layer: options, metadata preservation, output assembly, and NetCDF/CSV parity are not finished.

## 2. Revised Target Architecture

### Core packages

1. `scope_torch.spectral`
   Real upstream optical assets, leaf optics, wavelength grids, and soil optics.
2. `scope_torch.canopy`
   Reflectance, fluorescence, and thermal canopy transport on a shared geometry backbone.
3. `scope_torch.biochem`
   Leaf physiology and fluorescence-yield drivers.
4. `scope_torch.energy`
   Aerodynamic resistances, heat fluxes, soil heat treatment, and coupled energy balance closure.
5. `scope_torch.runners`
   Scene and ROI/time entry points over the tensor model core.
6. `scope_torch.io`
   Output assembly back to `xarray` plus file exports.

### Data model principles

1. Keep spectral, layer, and orientation axes explicit in tensors.
2. Keep geometry, meteorology, canopy structure, soil, and solver options explicit in typed dataclasses.
3. Preserve `xarray` coordinates and metadata through the grid workflow instead of requiring manual reshaping after the run.

## 3. Updated Implementation Plan

### Phase 0: Kernel hardening and real asset loading

Status: substantially complete.

Completed:
1. Removed the earlier CPU/detach hotspots from the implemented leaf and canopy kernels.
2. Added real upstream optical asset loading for FLUSPECT and soil spectra.
3. Added soil-library and BSM soil support.

Remaining finish items:
1. Add CPU-vs-GPU and batched-vs-single regression tests for the hardened kernels.
2. Add a small set of mixed-dtype or mixed-device smoke tests if GPU support is expected in day-to-day use.

### Phase 1: SCOPE-facing reflectance core

Status: benchmark-locked for the current optical path.

Completed:
1. Wrapped the current leaf optics and canopy reflectance stack behind a SCOPE-facing API.
2. Exposed the full reflectance output set rather than only `rsot` and `rdd`.
3. Added real soil loading and BSM soil generation paths.
4. Locked reflectance outputs against the current MATLAB benchmark suite to negligible relative error.

Remaining finish items:
1. Keep the non-converged-scene classification explicit in the benchmark exports and summary reports.

### Phase 2: Canopy fluorescence and thermal radiative transfer

Status: implemented and benchmark-locked for the current product set.

Completed:
1. Added one-pass and layered fluorescence transport.
2. Added biochemical fluorescence coupling.
3. Added spectral thermal radiance and integrated thermal balance outputs.
4. Added coupled energy-balance fluorescence and thermal entry points.
5. Locked fluorescence and thermal transport outputs against the current MATLAB benchmark suite.

Remaining finish items:
1. Add any extra exported RT diagnostics needed to isolate future widened-suite outliers faster.

### Phase 3: Biochemistry and energy balance

Status: implemented and benchmark-locked for same-state parity.

Completed:
1. Added leaf biochemistry with Ball-Berry closure and fluorescence-yield outputs.
2. Added aerodynamic resistances and heat-flux kernels.
3. Added a tensor-native energy-balance closure loop.
4. Wired closure outputs into coupled fluorescence and thermal workflows.
5. Added MATLAB benchmark export of the actual final biochemical-call inputs and the exact `L` used by `resistances.m`, which closes the earlier comparison artifacts.
6. Locked current same-state energy-balance parity across the original curated 10-case benchmark suite to below `1e-3` relative error for the tracked parity metrics.

Remaining finish items:
1. Document the distinction between same-state parity and phase-lagged iterate diagnostics in the benchmark reports.
2. Keep explicit tolerances versioned in tests and CI rather than only in local reports.

### Phase 4: Production grid and IO workflow

Status: still the biggest functional gap outside parity.

Tasks:
1. Refactor `prepare_scope_input.py` into reusable library functions with configurable data sources and no machine-specific paths.
2. Make `ScopeGridDataModule` lazy and chunk-aware so it does not materialize the full dataset before batching.
3. Extend `ScopeGridRunner` to preserve metadata and assemble results back into `xarray.Dataset`s.
4. Add NetCDF writers and, if needed, tabular exports for existing downstream workflows.

Exit criteria:
1. A prepared ROI/time dataset can be simulated end-to-end and written back with coordinates and metadata intact.
2. Large ROI/time jobs can run in chunks without rewriting the model core.

### Phase 5: Lock parity and regression coverage

Status: partially complete.

Tasks:
1. Keep the widened 100-case MATLAB suite and the single-scene pytest parity gate in sync with the benchmark exports.
2. Turn the widened-suite summary into versioned tolerances rather than relying only on local JSON reports.
3. Add batched-vs-single and CPU-vs-GPU consistency tests.
4. Wire the full suite into CI, possibly with a fast sampled subset plus an opt-in full sweep.

Exit criteria:
1. Each implemented physics module has reference-backed regression tests in automated CI.
2. The widened MATLAB suite has explicit pass/fail policy for converged and non-converged upstream scenes.
3. End-to-end cases fail fast when parity drifts on CPU or GPU.

## 4. Suggested Next Step

The next step should be **workflow productionization plus automated regression coverage**.

Recommended sequence:

1. Refactor `prepare_scope_input.py` into reusable library code with configurable inputs and no machine-specific paths.
2. Make `ScopeGridDataModule` lazy so large ROI/time datasets do not materialize into one tensor map up front.
3. Add metadata-preserving `xarray.Dataset` assembly to `ScopeGridRunner`.
4. Add CI that runs the standard unit suite plus the MATLAB parity gates where available.
5. Add batched-vs-single and CPU-vs-GPU regression tests for the coupled products.

Why this should be next:

1. The widened suite showed the remaining parity work is now narrow and concrete, not broad missing physics.
2. With reflectance fixed and the non-converged-scene policy explicit, the main remaining work is workflow robustness, not unresolved parity semantics.
3. The benchmark harness now exists at the full-scene scale, so productization work can proceed safely as long as the parity checks stay attached to it.
