from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
DEFAULT_MATLAB = "/Applications/MATLAB_R2025b.app/bin/matlab"


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@functools.lru_cache(maxsize=1)
def _matlab_candidate() -> str | None:
    if _truthy(os.environ.get("SCOPE_DISABLE_MATLAB")):
        return None

    candidates: list[str] = []
    env_candidate = os.environ.get("MATLAB_BIN")
    if env_candidate:
        candidates.append(env_candidate)
    candidates.append(DEFAULT_MATLAB)
    which = shutil.which("matlab")
    if which:
        candidates.append(which)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
        if resolved and Path(resolved).exists():
            return resolved
    return None


@functools.lru_cache(maxsize=1)
def matlab_bin() -> str | None:
    candidate = _matlab_candidate()
    if candidate is None:
        if _truthy(os.environ.get("SCOPE_REQUIRE_MATLAB")):
            raise RuntimeError("MATLAB is required but no executable was found")
        return None

    try:
        subprocess.run(
            [candidate, "-batch", "disp(version)"],
            check=True,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        if _truthy(os.environ.get("SCOPE_REQUIRE_MATLAB")):
            raise RuntimeError(f"MATLAB is required but unavailable: {candidate}") from exc
        return None
    return candidate


def has_live_matlab() -> bool:
    return matlab_bin() is not None


def python_test_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not pythonpath else f"src{os.pathsep}{pythonpath}"
    return env


def compare_benchmark_fixture(benchmark: Path, report_path: Path, *, device: str = "cpu") -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "compare_scope_benchmark.py"),
            "--benchmark",
            str(benchmark),
            "--report-json",
            str(report_path),
            "--device",
            device,
        ],
        check=True,
        cwd=REPO_ROOT,
        env=python_test_env(),
        stdout=subprocess.DEVNULL,
    )
    return json.loads(report_path.read_text(encoding="utf-8"))
