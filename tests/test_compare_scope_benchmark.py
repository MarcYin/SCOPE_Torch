from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_compare_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "compare_scope_benchmark.py"
    spec = importlib.util.spec_from_file_location("compare_scope_benchmark", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark comparison script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scope_refl_helper_uses_full_spectrum_esky_max_for_low_sky_fallback():
    module = _load_compare_module()

    rso = torch.tensor([0.15678350680496916], dtype=torch.float64)
    rdo = torch.tensor([0.1206171277543417], dtype=torch.float64)
    esun = torch.tensor([0.05995199702001321], dtype=torch.float64)
    esky = torch.tensor([0.00395839499202272], dtype=torch.float64)

    optical_only = module._scope_refl_from_rso_rdo(rso, rdo, esun, esky)
    full_spectrum = module._scope_refl_from_rso_rdo(rso, rdo, esun, esky, esky_max=20.873686530623086)

    assert not torch.allclose(optical_only, rso)
    assert torch.allclose(full_spectrum, rso)


def test_benchmark_status_marks_nonconverged_upstream_ebal_cases():
    module = _load_compare_module()

    status = module._benchmark_status(
        {
            "energy_counter": 101.0,
            "energy_maxit": 100.0,
            "energy_upstream_converged": False,
            "energy_hit_max_iterations": True,
            "energy_max_energy_error": 1.0,
            "energy_final_max_error_sunlit": 1.6,
            "energy_final_max_error_shaded": 0.77,
            "energy_final_max_error_soil": 5.85,
        }
    )

    assert status["energy_converged"] is False
    assert status["energy_hit_max_iterations"] is True
    assert status["energy_counter"] == 101.0
