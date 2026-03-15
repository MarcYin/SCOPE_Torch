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

Release notes are prepared continuously by:

- `.github/workflows/release-drafter.yml`

The tagged release workflow also:

- creates or updates the GitHub release entry
- uploads built artifacts to that GitHub release
- emits GitHub artifact attestations for `dist/*`

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
3. Review the current GitHub draft release notes.
4. Bump version in `pyproject.toml`.
5. Create and push a tag like `v0.2.0`.
6. Confirm the GitHub release, artifact attestations, and PyPI install path:

```bash
python -m pip install SCOPE-RTM
python -c "import scope; print(scope.__version__)"
```
