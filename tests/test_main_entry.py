"""Tests for scope.__main__ module entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


def _run_scope(*args: str):
    """Run ``python -m scope`` with PYTHONPATH pointing at the source tree."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "scope", *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_python_m_scope_help():
    result = _run_scope("--help")
    assert result.returncode == 0
    assert "scope" in result.stdout.lower()


def test_python_m_scope_vars_help():
    result = _run_scope("vars", "--help")
    assert result.returncode == 0
