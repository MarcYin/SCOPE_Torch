# Releasing

This page documents the current release path for the published package:

- PyPI package name: `SCOPE-RTM`
- Python import name: `scope`

## Release Trigger

The repository publishes automatically to PyPI from:

- `.github/workflows/release.yml`

The workflow behavior is:

- push a tag matching `v*` -> build, smoke-install, publish to PyPI
- manual dispatch with `publish_target=testpypi` -> publish to TestPyPI
- manual dispatch with `publish_target=pypi` -> publish to PyPI

## What the Release Workflow Verifies

Before publishing, the workflow:

1. builds `sdist` and wheel artifacts
2. runs `twine check`
3. installs the built wheel on a clean runner
4. verifies:
   - `import scope`
   - `scope.__version__ == importlib.metadata.version("SCOPE-RTM")`
   - `scope --help`
   - `scope-fetch-upstream --help`
   - `scope-prepare --help`
   - `scope-run --help`

## Required GitHub / PyPI Setup

Before the first real release, configure:

- GitHub environment `pypi`
- GitHub environment `testpypi`
- PyPI trusted publisher for project `SCOPE-RTM`
- TestPyPI trusted publisher for project `SCOPE-RTM`

The trusted publisher must point to:

- repository owner / repo
- workflow file: `.github/workflows/release.yml`
- environment: `pypi` or `testpypi`

## Maintainer Workflow

Recommended sequence:

1. Ensure `main` is green.
2. Optionally run a manual TestPyPI publish.
3. Bump version in `pyproject.toml` and any release notes you maintain.
4. Create and push a tag like `v0.1.0`.
5. Confirm the PyPI release and install path:

```bash
python -m pip install SCOPE-RTM
python -c "import scope; print(scope.__version__)"
```
