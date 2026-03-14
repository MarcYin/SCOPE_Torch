from __future__ import annotations

import json
import subprocess
import sys

from tests._matlab_benchmark_helpers import REPO_ROOT, TEST_DATA_DIR, has_live_matlab, matlab_bin, python_test_env


def test_scope_timeseries_benchmark_locked_subsystems(tmp_path):
    report_dir = tmp_path / "timeseries_reports"
    summary_path = tmp_path / "scope_timeseries_summary.json"
    fixture_dir = TEST_DATA_DIR / "timeseries_benchmark_fixtures"
    command = [
        str(REPO_ROOT / "scripts" / "run_scope_timeseries_benchmark_suite.py"),
        "--reports-dir",
        str(report_dir),
        "--summary-json",
        str(summary_path),
    ]
    if has_live_matlab():
        command = [
            str(REPO_ROOT / "scripts" / "run_scope_timeseries_benchmark_suite.py"),
            "--matlab",
            matlab_bin(),
            "--reports-dir",
            str(report_dir),
            "--summary-json",
            str(summary_path),
        ]
    else:
        assert fixture_dir.exists(), f"Pregenerated time-series benchmark fixtures not found: {fixture_dir}"
        command.extend(["--benchmark-dir", str(fixture_dir), "--skip-export"])

    subprocess.run(
        [
            sys.executable,
            *command,
        ],
        check=True,
        cwd=REPO_ROOT,
        env=python_test_env(),
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
