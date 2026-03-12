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
4. The repository still lacks official MATLAB/SCOPE verification cases, explicit product tolerances, and CI automation.

### Main accuracy and scope gaps

1. **Reference parity is still the main missing guardrail.**
   The core model now covers reflectance, fluorescence, thermal RT, biochemistry, and energy balance, but those modules are mostly validated by local consistency and curated unit references rather than official SCOPE benchmark scenes.
2. **The coupled energy-balance shortwave forcing still needs reference-backed locking.**
   The current `CanopyEnergyBalanceModel` derives layer shortwave absorption from the in-repo layered transport implementation. That is coherent and tested, but it is not yet proven against upstream `RTMo`/`RTMz` outputs across benchmark cases.
3. **GPU and batched consistency are not yet institutionalized.**
   The earlier CPU/detach hot spots are gone from the implemented kernels, but there is still no regression suite proving CPU-vs-GPU or batched-vs-single equivalence for the coupled products.
4. **Workflow parity is still narrow.**
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

Status: complete enough for downstream coupling.

Completed:
1. Wrapped the current leaf optics and canopy reflectance stack behind a SCOPE-facing API.
2. Exposed the full reflectance output set rather than only `rsot` and `rdd`.
3. Added real soil loading and BSM soil generation paths.

Remaining finish items:
1. Lock reflectance outputs against curated SCOPE or PROSAIL benchmark scenes under `tests/data/`.

### Phase 2: Canopy fluorescence and thermal radiative transfer

Status: implemented, but not yet reference-locked.

Completed:
1. Added one-pass and layered fluorescence transport.
2. Added biochemical fluorescence coupling.
3. Added spectral thermal radiance and integrated thermal balance outputs.
4. Added coupled energy-balance fluorescence and thermal entry points.

Remaining finish items:
1. Benchmark fluorescence and thermal RT outputs against upstream SCOPE cases.
2. Tighten any remaining post-processing or diagnostic-output differences discovered by those parity cases.

### Phase 3: Biochemistry and energy balance

Status: implemented prototype with good local regression coverage.

Completed:
1. Added leaf biochemistry with Ball-Berry closure and fluorescence-yield outputs.
2. Added aerodynamic resistances and heat-flux kernels.
3. Added a tensor-native energy-balance closure loop.
4. Wired closure outputs into coupled fluorescence and thermal workflows.

Remaining finish items:
1. Lock the shortwave forcing and convergence behavior against upstream `ebal.m` benchmark cases.
2. Define explicit tolerances for net radiation, sensible heat, latent heat, soil heat, leaf temperatures, soil temperatures, and iteration counts.

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

Status: now the top technical priority.

Tasks:
1. Vendor official or curated SCOPE verification cases into `tests/data/`.
2. Add single-scene MATLAB parity tests for reflectance, fluorescence, thermal radiance, and energy balance products.
3. Add batched-vs-single and CPU-vs-GPU consistency tests.
4. Define explicit tolerances by product class.
5. Wire the full suite into CI.

Exit criteria:
1. Each implemented physics module has reference-backed regression tests.
2. End-to-end cases fail fast when parity drifts.

## 4. Suggested Next Step

The next step should be **reference-backed parity locking, starting with one coupled benchmark scene**.

Recommended sequence:

1. Vendor one curated upstream SCOPE case into `tests/data/`.
2. Build a parity harness that compares at least these outputs:
   `rsot`, `LoF_`, `Lot_`, `Rnuc/Rnhc/Rnus/Rnhs`, `Tcu/Tch/Tsu/Tsh`, `H`, `lE`, and convergence count.
3. Run that case through the current `CanopyEnergyBalanceModel` and fix any discrepancy before expanding to more cases.
4. Only after the coupled-scene parity harness is stable, move on to grid metadata/output productionization.

Why this should be next:

1. The core physics stack is broad enough now that additional feature work will mostly create revalidation cost.
2. The highest remaining technical risk is not missing modules; it is unproven parity for the coupled energy-balance path.
3. Once the parity harness exists, the remaining grid/IO work becomes much safer because regressions will be visible immediately.
