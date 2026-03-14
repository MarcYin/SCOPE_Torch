# SCOPE Implementation Plan

## 1. Current Repository Status

### Implemented and already useful

| Area | Current implementation | Evidence |
| --- | --- | --- |
| Upstream asset loading | Real SCOPE optical assets, soil spectra, and SCOPE filename metadata can be loaded from the vendored upstream tree. | [src/scope/spectral/loaders.py](src/scope/spectral/loaders.py), [tests/spectral/test_loaders.py](tests/spectral/test_loaders.py) |
| Leaf optics | `FluspectModel` ports the leaf optics core, including fluorescence matrices `Mb`/`Mf`, and now stays tensor-native instead of detaching to SciPy. | [src/scope/spectral/fluspect.py](src/scope/spectral/fluspect.py), [tests/spectral/test_fluspect.py](tests/spectral/test_fluspect.py) |
| Canopy reflectance | `FourSAILModel` and `CanopyReflectanceModel` provide a SCOPE-facing reflectance stack with full reflectance outputs plus soil-library and BSM soil support. | [src/scope/canopy/foursail.py](src/scope/canopy/foursail.py), [src/scope/canopy/reflectance.py](src/scope/canopy/reflectance.py), [tests/canopy/test_foursail.py](tests/canopy/test_foursail.py), [tests/spectral/test_soil.py](tests/spectral/test_soil.py) |
| Canopy fluorescence | The repo includes one-pass fluorescence, layered fluorescence transport, a physiology-coupled layered fluorescence path, and explicit directional/profile APIs on the current homogeneous canopy stack. | [src/scope/canopy/fluorescence.py](src/scope/canopy/fluorescence.py), [tests/canopy/test_fluorescence.py](tests/canopy/test_fluorescence.py) |
| Canopy thermal RT | The repo includes spectral thermal radiance, spectrally integrated thermal balance outputs with shared layered transport, and explicit directional/profile APIs on the current homogeneous canopy stack. | [src/scope/canopy/thermal.py](src/scope/canopy/thermal.py), [tests/canopy/test_thermal.py](tests/canopy/test_thermal.py) |
| Leaf biochemistry | `LeafBiochemistryModel` covers C3/C4 assimilation, Ball-Berry closure, and fluorescence-yield outputs used by the canopy models. | [src/scope/biochem/leaf.py](src/scope/biochem/leaf.py), [tests/biochem/test_leaf.py](tests/biochem/test_leaf.py) |
| Energy balance closure | `CanopyEnergyBalanceModel` iterates temperatures, boundary humidity/CO2, aerodynamic resistances, and coupled fluorescence/thermal outputs. | [src/scope/energy/balance.py](src/scope/energy/balance.py), [tests/energy/test_balance.py](tests/energy/test_balance.py) |
| Grid execution | `ScopeGridDataModule` now batches chunk-locally, and `ScopeGridRunner` exposes reflectance, reflectance profiles, directional reflectance, fluorescence, fluorescence profiles, directional fluorescence, biochemical fluorescence, thermal RT, thermal profiles, directional thermal, and coupled energy-balance workflows over ROI/time batches. | [src/scope/data/grid.py](src/scope/data/grid.py), [src/scope/runners/grid.py](src/scope/runners/grid.py), [tests/test_grid_data_module.py](tests/test_grid_data_module.py), [tests/test_scope_grid_runner.py](tests/test_scope_grid_runner.py) |
| IO preparation | Input preparation has a reusable library surface in `scope.io.prepare`, the old script is a thin CLI wrapper, and NetCDF export helpers now cover prepared and simulated `xarray` datasets. | [src/scope/io/prepare.py](src/scope/io/prepare.py), [src/scope/io/export.py](src/scope/io/export.py), [prepare_scope_input.py](prepare_scope_input.py), [tests/test_prepare_scope_input.py](tests/test_prepare_scope_input.py), [tests/test_netcdf_export.py](tests/test_netcdf_export.py) |
| Production usage surface | The package now has a real user-facing docs site, runnable examples, an installed `scope` CLI, a shell-level `scope run` path, and smoke-tested release workflows for `SCOPE-RTM`. | [README.md](README.md), [docs/index.md](docs/index.md), [docs/quickstart.md](docs/quickstart.md), [src/scope/cli/main.py](src/scope/cli/main.py), [src/scope/cli/run.py](src/scope/cli/run.py), [.github/workflows/release.yml](.github/workflows/release.yml) |
| Regression automation | Committed benchmark summaries, live-or-pregenerated MATLAB parity tests, and standard pytest automation are now present. | [tests/test_benchmark_summary_regression.py](tests/test_benchmark_summary_regression.py), [tests/test_scope_benchmark_parity.py](tests/test_scope_benchmark_parity.py), [tests/test_scope_timeseries_benchmark_parity.py](tests/test_scope_timeseries_benchmark_parity.py), [.github/workflows/tests.yml](.github/workflows/tests.yml) |

### Still incomplete or narrow

1. The default CI path now runs the MATLAB parity tests in live-or-pregenerated mode, but dedicated fresh-export MATLAB runs still require self-hosted licensed infrastructure.
2. Device coverage is substantially improved but still not exhaustive: standalone and coupled runner workflows now have batched-vs-single and dtype coverage, the lower-level kernel layer now has direct batch/dtype regression tests with optional CUDA mirrors, but mixed-dtype and broader backend coverage are still missing.
3. Workflow parity now includes high-level option-driven directional/profile dispatch through the shared runner surface, but it is still narrow around downstream export targets beyond the current shared NetCDF writer and any pipeline-specific workflow wrappers.
4. The pregenerated time-series MATLAB fixture set now makes hosted parity deterministic, but it adds meaningful repository weight and should be treated as an explicit maintenance decision rather than an accidental byproduct.

### Main accuracy and scope gaps

1. **The benchmark suite is widened, and the non-converged upstream case is now classified explicitly.**
   The MATLAB harness now scales to the full 100-case upstream Latin-hypercube set via `scripts/run_scope_benchmark_suite.py`, with per-case reports in `tests/data/benchmark_suite_reports/` and an aggregate summary in `tests/data/scope_benchmark_suite_summary.json`. The earlier `reflectance.refl` outlier was a benchmark-harness reconstruction bug and is now closed across the full 100-case sweep. The suite now treats upstream scenes that hit `ebal` max iterations as stress diagnostics instead of parity-gating cases. At the moment only case `042` falls into that bucket.
2. **Time-series parity is now covered separately from the scene suite.**
   `scripts/export_scope_timeseries_benchmarks.m` and `scripts/run_scope_timeseries_benchmark_suite.py` now exercise upstream SCOPE `simulation == 1` on the default 30-step verification series, writing per-step reports under `tests/data/timeseries_benchmark_reports/` and an aggregate summary in `tests/data/scope_timeseries_benchmark_summary.json`. The same non-converged-upstream policy applies there: step `026` is currently classified as a stress case because upstream MATLAB `ebal` hits `maxit`.
3. **The raw energy-balance iterate diagnostics are still easy to misread unless the policy is explicit.**
   End-of-iteration same-state parity is now negligible, but the phase-lagged `energy_balance.sunlit_A` and `energy_balance.shaded_A` fields in the raw comparison reports still look worse than the true leaf-kernel parity because they compare the final leaf solve against post-update boundary states. The real parity contract is the committed suite summaries plus the exported `leaf_iteration.*` and same-state energy metrics. That distinction is now reflected in the suite JSON `parity_policy` block and documented in [docs/benchmark-policy.md](docs/benchmark-policy.md).
4. **GPU and batched consistency are now covered at both runner and kernel layers.**
   The earlier CPU/detach hot spots are gone from the implemented kernels, there is now coupled energy/thermal batched-vs-single coverage, standalone and coupled runner batch-size/dtype coverage, direct lower-level kernel batch/dtype coverage, and selected workflows have optional CPU-vs-GPU checks. The remaining execution-mode gap is mixed-dtype or broader backend coverage rather than missing basic kernel/device regression tests.
5. **Workflow exports are still narrower than the model core.**
   Input preparation, chunk-local batching, metadata-preserving `xarray` assembly, and NetCDF export are now implemented, but downstream format parity beyond NetCDF is not finished.
6. **`mSCOPE` remains a future feature rather than an active implementation target.**
   The current stack now exposes directional and vertical-profile outputs on the homogeneous canopy path, which covers the immediate workflow need. True `mSCOPE` still requires layer-resolved leaf optics and canopy transport plumbing and is intentionally deferred until there is a real consumer for it.

## 2. Revised Target Architecture

### Core packages

1. `scope.spectral`
   Real upstream optical assets, leaf optics, wavelength grids, and soil optics.
2. `scope.canopy`
   Reflectance, fluorescence, and thermal canopy transport on a shared geometry backbone.
3. `scope.biochem`
   Leaf physiology and fluorescence-yield drivers.
4. `scope.energy`
   Aerodynamic resistances, heat fluxes, soil heat treatment, and coupled energy balance closure.
5. `scope.runners`
   Scene and ROI/time entry points over the tensor model core.
6. `scope.io`
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
1. Add a small set of mixed-dtype smoke tests if GPU support is expected in day-to-day use.
2. Add any broader backend checks only if non-CUDA accelerator support becomes a real requirement.

### Phase 1: SCOPE-facing reflectance core

Status: benchmark-locked for the current optical path.

Completed:
1. Wrapped the current leaf optics and canopy reflectance stack behind a SCOPE-facing API.
2. Exposed the full reflectance output set rather than only `rsot` and `rdd`.
3. Added real soil loading and BSM soil generation paths.
4. Locked reflectance outputs against the current MATLAB benchmark suite to negligible relative error.
5. Added explicit directional and radiative-profile APIs on the current homogeneous reflectance stack.

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
6. Added explicit directional and vertical-profile APIs for the current homogeneous fluorescence and thermal workflows.

Remaining finish items:
1. Add any extra exported RT diagnostics needed to isolate future widened-suite outliers faster.
2. Keep the profile-output contracts aligned with the runner surface as widened benchmark coverage grows.

### Phase 3: Biochemistry and energy balance

Status: implemented and benchmark-locked for same-state parity.

Completed:
1. Added leaf biochemistry with Ball-Berry closure and fluorescence-yield outputs.
2. Added aerodynamic resistances and heat-flux kernels.
3. Added a tensor-native energy-balance closure loop.
4. Wired closure outputs into coupled fluorescence and thermal workflows.
5. Added MATLAB benchmark export of the actual final biochemical-call inputs and the exact `L` used by `resistances.m`, which closes the earlier comparison artifacts.
6. Locked current same-state energy-balance parity across the committed scene and time-series benchmark policies, with explicit tolerances versioned in the test suite.

Remaining finish items:
1. Broaden the explicit parity policy only if more stress-scene diagnostics need to become gating metrics.
2. Revisit whether any additional benchmark metadata should be promoted into downstream reporting or dashboards.

### Phase 4: Production grid and IO workflow

Status: substantially complete and now shell-usable.

Completed:
1. Refactored `prepare_scope_input.py` into reusable library functions with configurable inputs and a thin CLI wrapper.
2. Made `ScopeGridDataModule` chunk-local so it no longer materializes one tensor map before batching.
3. Extended `ScopeGridRunner` to preserve metadata and assemble results back into `xarray.Dataset`s for the main workflows.
4. Added installed production-facing CLI entry points for asset fetch, input preparation, and prepared-dataset execution.
5. Added production docs, quickstart guides, runnable examples, and release/docs workflows.
6. Hardened the SciPy NetCDF fallback path so runner-style datasets round-trip safely without invalid unlimited-dimension usage.

Remaining finish items:
1. Add any remaining higher-level pipeline wrappers needed to call the now-implemented option-driven runner surface consistently.
2. Add any remaining tabular or downstream-specific exports needed beyond the shared NetCDF writer only if those are still required by real consumers.

Exit criteria:
1. A prepared ROI/time dataset can be simulated end-to-end and written back with coordinates and metadata intact.
2. Large ROI/time jobs can run in chunks without rewriting the model core.

### Phase 5: Lock parity and regression coverage

Status: substantially complete.

Completed:
1. Kept the widened 100-case MATLAB suite and the single-scene/time-series parity gates in sync with the benchmark exports, including pregenerated-fixture fallback when MATLAB is unavailable.
2. Turned the committed suite summaries into versioned tolerances in pytest.
3. Added coupled batched-vs-single regression checks and an optional CPU-vs-GPU check for the energy/thermal path.
4. Added batched-vs-single and dtype regression coverage for the standalone reflectance, fluorescence, biochemical fluorescence, thermal, coupled energy-balance fluorescence, and coupled energy-balance thermal runner paths, plus optional CUDA mirrors on selected workflows.
5. Wired the standard Python test suite into GitHub Actions CI.
6. Added opt-in self-hosted GPU and live-MATLAB parity jobs to the GitHub Actions workflow.

Remaining finish items:
1. Decide whether the current opt-in self-hosted live-MATLAB lane should eventually become a required branch-protection signal.
2. Add mixed-dtype or broader backend coverage if those execution modes are expected in production use.
3. Decide whether the checked-in time-series MAT fixtures should remain in-repo, move to Git LFS, or be regenerated from a separate artifact source.

Exit criteria:
1. Each implemented physics module has reference-backed regression tests in automated CI.
2. The widened MATLAB suite has explicit pass/fail policy for converged and non-converged upstream scenes.
3. End-to-end cases fail fast when parity drifts on CPU or GPU for the supported execution modes.

## 4. Suggested Next Step

The next step should be **decide how to operationalize the release and benchmark infrastructure now that the core model and user-facing shell workflow are in place**.

Recommended sequence:

1. Decide whether the checked-in `tests/data/timeseries_benchmark_fixtures/` set should stay in the repo, move to Git LFS, or be replaced by an external artifact/cache strategy. That is now the highest-cost maintenance choice.
2. Decide whether the self-hosted live-MATLAB lane should remain manual or become a required protected-branch signal once the infrastructure is reliable enough.
3. Decide whether docs deployment and package publishing should stay tag/manual-trigger based or move to stricter protected-release automation.
4. Add mixed-dtype coverage only if `float32`/`float64` interoperability is expected in production workflows; otherwise keep the current execution matrix stable.
5. Add any remaining downstream-specific wrappers around `run_scope_dataset(...)` only if a real consumer needs a thinner application-facing API.
6. Keep `mSCOPE` as a deferred phase until there is a concrete workflow that requires vertically heterogeneous leaf optics.

Why this should be next:

1. The physics stack, ROI/time workflow, benchmark policy, installed CLI, docs site, and release workflows are already in place, so the remaining risk is mostly operational rather than numerical.
2. The benchmark infrastructure now works on both hosted and self-hosted runners, which shifts the main open questions to storage cost, CI policy, release policy, and long-term maintenance.
3. Further feature work is lower leverage than making the release and parity infrastructure easier to maintain and cheaper to run.
