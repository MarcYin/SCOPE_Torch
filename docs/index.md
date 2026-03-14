# SCOPE

`scope` is a PyTorch-first implementation of the SCOPE canopy radiative transfer model for:

- reflectance
- fluorescence
- thermal radiance
- coupled energy-balance workflows

It is designed for users who need:

- asset-backed SCOPE physics in Python
- batched ROI/time execution on `xarray` datasets
- differentiable model components in PyTorch
- reproducible MATLAB parity checks in local development and CI

## Start Here

If this is your first time using the project:

1. Read [Installation](installation.md)
2. Run the examples in [Quickstart](quickstart.md)
3. Use [Input / Output Reference](input-output-reference.md) when building real workflows

## Main User Entry Points

For most application code, prefer:

- `ScopeGridRunner.run_scope_dataset(...)`
- `prepare_scope_input_dataset(...)`
- `write_netcdf_dataset(...)`

These cover the common production path:

1. prepare a runner-ready `xarray.Dataset`
2. execute the requested SCOPE workflows
3. persist outputs to NetCDF when needed

## Current Feature Surface

The current implementation supports:

- FLUSPECT leaf optics
- 4SAIL-based canopy reflectance transport
- layered fluorescence and thermal radiative transfer
- leaf biochemistry and coupled energy balance
- directional and vertical-profile outputs on the homogeneous canopy path
- scene and time-series MATLAB parity checks

## What To Read Next

- [Quickstart](quickstart.md)
- [Model Mechanics](model-mechanics.md)
- [Examples](examples.md)
- [Production Notes](production-notes.md)
