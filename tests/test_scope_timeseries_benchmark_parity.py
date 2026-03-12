from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_scope_timeseries_benchmark_locked_subsystems(tmp_path):
    if os.environ.get("SCOPE_TIMESERIES_BENCHMARK") != "1":
        pytest.skip("Set SCOPE_TIMESERIES_BENCHMARK=1 to run the MATLAB/Python time-series benchmark parity check.")

    repo_root = Path(__file__).resolve().parents[1]
    report_dir = tmp_path / "timeseries_reports"
    summary_path = tmp_path / "scope_timeseries_summary.json"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not pythonpath else f"src{os.pathsep}{pythonpath}"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_scope_timeseries_benchmark_suite.py"),
            "--reports-dir",
            str(report_dir),
            "--summary-json",
            str(summary_path),
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["nonconverged_energy_steps"] == ["026"]

    parity = summary["parity_worst_cases"]["max_rel"]
    assert parity["reflectance.refl"]["max_rel"] < 1e-8
    assert parity["fluorescence_transport.LoF_"]["max_rel"] < 1e-8
    assert parity["thermal_transport.Lot_"]["max_rel"] < 1e-8
    assert parity["energy_balance.Rntot"]["max_rel"] < 1e-3
    assert parity["energy_balance.lEtot"]["max_rel"] < 1e-3
    assert parity["energy_balance.Htot"]["max_rel"] < 1e-3
    assert parity["energy_balance.Tcu"]["max_rel"] < 1e-3
    assert parity["energy_balance.Tch"]["max_rel"] < 1e-3
