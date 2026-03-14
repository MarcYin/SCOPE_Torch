from __future__ import annotations

import importlib
import warnings


def test_scope_is_the_public_package_name() -> None:
    scope = importlib.import_module("scope")
    assert scope.__name__ == "scope"
    assert hasattr(scope, "ScopeGridRunner")


def test_scope_torch_imports_remain_available_as_compatibility_aliases() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = importlib.import_module("scope_torch")

    assert legacy.ScopeGridRunner is importlib.import_module("scope").ScopeGridRunner
    assert importlib.import_module("scope_torch.runners.grid").ScopeGridRunner is legacy.ScopeGridRunner
    assert any(item.category is DeprecationWarning for item in caught)
