# Publishing jstprove to PyPI

This document explains how to build, test, and publish a new release of the `jstprove` Python package.

## Prerequisites

- Python 3.10+
- A virtual environment (recommended)
- PyPI account with Maintainer/Owner permission for `jstprove`
- API token created at https://pypi.org/manage/account/token/
- Required tools:

```bash
python -m pip install --upgrade pip build twine
```

Optional: configure `~/.pypirc` for convenience:

```ini
# ~/.pypirc
[distutils]
index-servers = pypi testpypi

[pypi]
username = __token__
password = pypi-<your-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-testpypi-token>
```

You can also export credentials per session:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-token>
```

## Build the package

Builds sdist and wheel into `dist/`.

```bash
python -m build
```

Sanity check the artifacts:

```bash
python -m twine check dist/*
```

Optional: inspect file contents (sdist):

```bash
tar -tf dist/jstprove-<version>.tar.gz | head -n 50
```

## Test the package locally

Install the wheel into a clean environment and smoke-test the CLI.

```bash
python -m venv .venv-test
source .venv-test/bin/activate
pip install dist/jstprove-*.whl

# Verify CLI is wired
jstprove --help
```

## Publish to TestPyPI (recommended)

Upload to TestPyPI first to validate metadata and installability.

```bash
python -m twine upload -r testpypi dist/*
```

Test install from TestPyPI (note: dependencies resolve from PyPI):

```bash
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple jstprove==<version>
```

## Publish to PyPI

When satisfied with testing, upload to PyPI:

```bash
python -m twine upload dist/*
```

## Notes and tips

- Entry point: the CLI is `jstprove` which maps to `python.frontend.cli:main`.
- Data files: we include ONNX models under `python/models/models_onnx/*.onnx` via `MANIFEST.in` and `tool.setuptools.package-data`.
- Dependencies: runtime dependencies are pinned in `pyproject.toml`. Heavy packages (torch, onnxruntime, transformers) will make installs large. Consider reviewing pins as needed.
- If you add new data files or packages, update `pyproject.toml` and/or `MANIFEST.in` accordingly.
