# Production Notes

## Recommended Deployment Assumptions

For the current codebase, the safest production assumptions are:

- run from a source checkout or controlled internal package build
- keep a pinned upstream SCOPE checkout available
- prefer the runner surface over direct low-level kernel composition unless you need custom research code
- persist prepared and simulated datasets through the shared NetCDF writer

## Asset Strategy

Asset-backed constructors expect upstream SCOPE resources such as:

- FLUSPECT parameter MAT files
- soil spectra
- atmospheric files

The intended operational path is:

1. fetch the pinned upstream checkout with `scope-fetch-upstream` or `scope fetch-upstream`
2. keep that checkout version-controlled or provisioned by deployment automation
3. pass `scope_root_path=...` explicitly if the checkout is not under `./upstream/SCOPE`

For scientific or operational deployments, keep the attribution trail visible to users: this package is based on the original MATLAB SCOPE repository at [Christiaanvandertol/SCOPE](https://github.com/Christiaanvandertol/SCOPE), and the upstream manual lives at [scope-model.readthedocs.io](https://scope-model.readthedocs.io/en/master/).

## Benchmark and CI Strategy

The current repository uses two parity modes:

- live MATLAB export when MATLAB is available
- pregenerated MATLAB fixture fallback when it is not

That makes hosted CI deterministic without requiring MATLAB on every runner, while still keeping a self-hosted live-MATLAB lane for stronger operational checks.

The documentation surface is also treated as a build artifact:

- example scripts are executed in the test suite
- the docs site can be built locally with `mkdocs build --strict`
- CI should keep docs build failures separate from physics regressions

## Performance and Compilation

For kernel-level timing and eager-versus-compiled comparisons, use:

```bash
PYTHONPATH=src python scripts/benchmark_kernels.py --fixture scope-assets --mode compare
```

The current recommendation is:

- do not enable `torch.compile` by default in production workflows
- benchmark on the actual target hardware first
- only consider a compiled path for long-lived services with repeated same-shape calls

On the current reference CPU environment, `fluspect` and canopy `reflectance` benefit in steady state, `thermal` only pays off after a much larger number of calls, layered `fluorescence` currently fails under `torch.compile`, and leaf biochemistry currently regresses because scalar root-solving logic still causes graph breaks and recompilation churn.

## Release and Distribution

For maintainers, the repository now has separate operational paths for packaging and docs deployment:

- `.github/workflows/release.yml`
  Builds source and wheel distributions for `SCOPE-RTM`, runs `twine check`, smoke-installs the built wheel, auto-publishes to PyPI on version tags, and still supports manual TestPyPI or PyPI publishing through GitHub environments.
- `.github/workflows/docs.yml`
  Builds the MkDocs site and deploys it to GitHub Pages.

Local release verification uses:

```bash
python -m pip install -e ".[release]"
python -m build
python -m twine check dist/*
```

The packaged wheel is also smoke-installed in CI and must satisfy:

```bash
scope --help
scope-fetch-upstream --help
scope-prepare --help
scope-run --help
```

## Current Operational Tradeoffs

The main operational tradeoffs are now:

- upstream SCOPE asset provisioning
- checked-in benchmark fixture footprint
- whether the self-hosted live-MATLAB lane should eventually become required CI

These are operational decisions, not open physics gaps.
