# Releasing aimodelground

This document covers how to build, verify, and publish aimodelground and its plugin packages to PyPI.

---

## Packages

| Package | PyPI name | Directory |
|---------|-----------|-----------|
| Core | `aimodelground` | `.` (root) |
| Classical ML | `aimodelground-classical` | `packages/model_builder_classical/` |
| Deep learning | `aimodelground-dl` | `packages/model_builder_dl/` |
| LLM fine-tuning | `aimodelground-llm` | `packages/model_builder_llm/` |

---

## Pre-release checklist

- [ ] All tests pass (`pytest tests/` — 0 failures)
- [ ] Plugin package tests pass (classical, dl, llm)
- [ ] Version bumped in `model_builder/__init__.py`
- [ ] Plugin package versions bumped in their `pyproject.toml`
- [ ] `CHANGELOG.md` updated with release notes
- [ ] `README.md` accurate for current release
- [ ] `aimodelground --version` shows correct version
- [ ] `uv build` succeeds for all packages
- [ ] Wheel contents verified (no junk files)

---

## Version bump

Version is the single source of truth in `model_builder/__init__.py`:

```python
__version__ = "0.2.0"   # bump here
```

`pyproject.toml` reads it dynamically via:

```toml
[tool.hatch.version]
path = "model_builder/__init__.py"
```

Plugin packages have their own `version` in their `pyproject.toml`. Bump each manually to match.

---

## Running the test suite

```powershell
# Core tests
uv run --project "D:\Projects\model-builder" pytest tests/ -q

# Classical ML plugin tests
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_classical/tests/ -q

# Deep learning plugin tests
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_dl/tests/ -q

# LLM plugin tests
uv run --project "D:\Projects\model-builder" pytest packages/model_builder_llm/tests/ -q
```

All suites must show 0 failures before releasing.

---

## Building wheels

```powershell
# Install build tool
uv pip install build --project "D:\Projects\model-builder"

# Build core package
uv run --project "D:\Projects\model-builder" python -m build

# Build plugin packages
uv run --project "D:\Projects\model-builder" python -m build packages/model_builder_classical/
uv run --project "D:\Projects\model-builder" python -m build packages/model_builder_dl/
uv run --project "D:\Projects\model-builder" python -m build packages/model_builder_llm/
```

Output goes to `dist/` in each package directory.

---

## Verify wheel contents

```powershell
# List wheel contents
uv run --project "D:\Projects\model-builder" python -c "
import zipfile, sys
with zipfile.ZipFile('dist/model_builder-0.1.0-py3-none-any.whl') as z:
    for name in sorted(z.namelist()):
        print(name)
"
```

Check that:
- `model_builder/__init__.py` is present
- `model_builder/cli/main.py` is present
- `model_builder/web/templates/` is included
- No `.pyc`, `__pycache__`, `.venv`, or test files are included

---

## Dry-run check with twine

```powershell
uv pip install twine --project "D:\Projects\model-builder"
uv run --project "D:\Projects\model-builder" twine check dist/*
```

Expected: `PASSED` for all packages. Fix any warnings before uploading.

---

## Upload to PyPI

**Test PyPI first (recommended):**

```powershell
uv run --project "D:\Projects\model-builder" twine upload --repository testpypi dist/*
```

Verify installation from Test PyPI:

```powershell
# --extra-index-url is required: Test PyPI only hosts aimodelground itself.
# Dependencies (aiosqlite, duckdb, etc.) must come from main PyPI.
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aimodelground
aimodelground --version
```

**Production PyPI:**

```powershell
uv run --project "D:\Projects\model-builder" twine upload dist/*
```

Upload plugin packages in the same way from their `dist/` directories.

---

## Post-release

- [ ] Tag the release in git: `git tag v0.1.0 && git push origin v0.1.0`
- [ ] Create GitHub Release with changelog notes
- [ ] Verify `pip install aimodelground` works on a clean environment
- [ ] Update `CHANGELOG.md` with `[Unreleased]` section for next release

---

## Rollback

If a broken release is published:

```powershell
# Yank (not delete) the broken version — users can still pin it but won't get it by default
# Do this from the PyPI web UI: pypi.org/manage/project/aimodelground/releases/
```

Never delete a published release. Yank it and publish a patch version instead.

---

## Environment setup for releases

```powershell
uv sync --dev --project "D:\Projects\model-builder"
uv pip install build twine --project "D:\Projects\model-builder"
```

Set PyPI credentials in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE
```



