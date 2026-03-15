from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script_path = REPO_ROOT / "scripts" / "benchmark_kernels.py"
    spec = importlib.util.spec_from_file_location("benchmark_kernels", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_kernels_eager_mode_writes_json(tmp_path):
    module = _load_module()
    output_path = tmp_path / "bench.json"
    report = module.main(
        [
            "--fixture",
            "synthetic",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--batch",
            "2",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--mode",
            "eager",
            "--kernels",
            "fluspect,reflectance",
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["fixture"] == "synthetic"
    assert loaded["device"] == "cpu"
    assert set(loaded["results"]) == {"fluspect", "reflectance"}
    assert loaded["results"]["fluspect"]["eager"]["median_seconds"] >= 0.0
    assert report["results"]["reflectance"]["eager"]["median_seconds"] >= 0.0
