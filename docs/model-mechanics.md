# Model Mechanics

This page explains how the current homogeneous-canopy `scope` stack is assembled.

## Physical Stack

At a high level, the model is organized as:

1. Leaf optics
2. Canopy radiative transfer
3. Fluorescence and thermal transport
4. Leaf biochemistry
5. Coupled energy balance

The current recommended user entry point is not the individual kernels but the runner layer, especially `ScopeGridRunner.run_scope_dataset(...)`.

## End-to-End Physics Flow

```mermaid
flowchart TD
    A["Leaf state<br/>Cab, Cw, Cdm, fqe, optional pigments"] --> B["FLUSPECT<br/>leaf reflectance, transmittance, fluorescence matrices"]
    B --> C["Canopy reflectance transport<br/>4SAIL + layered transfer"]
    C --> D["Directional / profile reflectance products"]
    B --> E["Layered fluorescence transport"]
    B --> F["Thermal radiance transport"]
    G["Leaf biochemistry<br/>Vcmax25, Ball-Berry, meteo state"] --> E
    G --> H["Energy balance closure<br/>temperatures, resistances, fluxes"]
    H --> E
    H --> F
    E --> I["Fluorescence outputs"]
    F --> J["Thermal outputs"]
    C --> K["Reflectance outputs"]
```

## ROI / Time Workflow

```mermaid
flowchart LR
    A["Weather / observation / bio inputs"] --> B["prepare_scope_input_dataset(...)"]
    B --> C["Runner-ready xarray.Dataset"]
    C --> D["ScopeGridDataModule"]
    D --> E["Batched torch tensors"]
    E --> F["ScopeGridRunner.run_scope_dataset(...)"]
    F --> G["Assembled xarray outputs"]
    G --> H["write_netcdf_dataset(...)"]
```

## Main Runtime Layers

### 1. Leaf optics

`FluspectModel` converts biochemical leaf parameters into:

- leaf reflectance
- leaf transmittance
- fluorescence source matrices

This is the lowest optical layer used by the canopy stack.

### 2. Canopy reflectance

`CanopyReflectanceModel` combines leaf optics, soil optics, canopy structure, and viewing geometry to produce:

- standard canopy reflectance terms
- directional reflectance products
- radiative profiles across layer interfaces

### 3. Fluorescence and thermal transport

`CanopyFluorescenceModel` and `CanopyThermalRadianceModel` reuse the canopy transport backbone to produce:

- canopy fluorescence radiance and flux products
- fluorescence directional / profile products
- thermal radiance and integrated thermal balance terms
- thermal directional / profile products

### 4. Biochemistry and energy balance

`LeafBiochemistryModel` provides assimilation, `Ci`, `rcw`, and fluorescence-yield drivers.

`CanopyEnergyBalanceModel` solves the coupled canopy state:

- sunlit / shaded leaf temperatures
- soil temperatures
- aerodynamic resistances
- sensible and latent heat fluxes
- coupled fluorescence and thermal outputs

## Same-State vs Phase-Lagged Diagnostics

For parity interpretation:

- use `leaf_iteration.*` metrics for true same-state leaf-physiology parity
- do not interpret raw `energy_balance.sunlit_*` and `energy_balance.shaded_*` diagnostics as the primary leaf-kernel parity signal

See [Benchmark Policy](benchmark-policy.md) for the exact reporting policy.
