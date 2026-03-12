from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_suite_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_scope_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_scope_benchmark_suite", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark suite script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric(max_rel: float) -> dict[str, float]:
    return {"max_abs": max_rel, "mean_abs": max_rel / 2.0, "max_rel": max_rel, "mean_rel": max_rel / 2.0}


def test_filtered_worst_cases_excludes_nonconverged_energy_metrics_only():
    module = _load_suite_module()

    per_case = {
        "001": {
            "energy_balance": {"Rnuct": _metric(1e-4)},
            "reflectance": {"refl": _metric(1e-6)},
        },
        "042": {
            "energy_balance": {"Rnuct": _metric(1e-2)},
            "reflectance": {"refl": _metric(2e-6)},
        },
    }

    summary = module._filtered_worst_cases(
        per_case,
        exclude=set(),
        nonconverged_energy_cases={"042"},
    )

    assert summary["max_rel"]["energy_balance.Rnuct"]["case"] == "001"
    assert summary["max_rel"]["reflectance.refl"]["case"] == "042"


def test_subset_worst_cases_keeps_nonconverged_energy_metrics_as_stress_diagnostics():
    module = _load_suite_module()

    per_case = {
        "001": {
            "energy_balance": {"Rnuct": _metric(1e-4)},
            "reflectance": {"refl": _metric(1e-6)},
        },
        "042": {
            "energy_balance": {"Rnuct": _metric(1e-2)},
            "energy_iteration_input": {"sunlit_Cs": _metric(2e-2)},
            "reflectance": {"refl": _metric(2e-6)},
        },
    }

    stress = module._subset_worst_cases(
        per_case,
        include_cases={"042"},
        include_prefixes=module.NONCONVERGED_ENERGY_PREFIXES,
    )

    assert stress["max_rel"]["energy_balance.Rnuct"]["case"] == "042"
    assert stress["max_rel"]["energy_iteration_input.sunlit_Cs"]["case"] == "042"
    assert "reflectance.refl" not in stress["max_rel"]
