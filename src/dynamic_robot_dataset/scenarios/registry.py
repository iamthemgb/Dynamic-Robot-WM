"""Import canonical scenario modules from the authoritative corpus registry."""

from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
from typing import Iterable

from ..common.corpus_registry import CorpusRegistry, load_corpus_registry
from ..common.hashing import sha256_file
from .types import ScenarioDefinition, ScenarioModuleSpec, bind_module


def _import_spec(path: str) -> ScenarioModuleSpec:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"invalid scenario module reference {path!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, attribute, None)
    if not isinstance(value, ScenarioModuleSpec):
        raise TypeError(f"{path} does not export a ScenarioModuleSpec")
    return value


def load_scenario_definition(
    leaf_id: str,
    *,
    corpus: CorpusRegistry | None = None,
) -> ScenarioDefinition:
    registry = corpus or load_corpus_registry()
    leaf = registry.resolve(leaf_id)
    return bind_module(_import_spec(leaf.scenario_module), leaf)


def list_scenario_definitions(
    *, corpus: CorpusRegistry | None = None
) -> tuple[ScenarioDefinition, ...]:
    registry = corpus or load_corpus_registry()
    return tuple(
        load_scenario_definition(leaf.corpus_id, corpus=registry)
        for leaf in registry.leaves
    )


def implemented_review_variants() -> dict[str, tuple[str, ...]]:
    return {
        definition.leaf_id: definition.variants
        for definition in list_scenario_definitions()
        if definition.implemented
    }


def scenario_source_hashes(leaf_id: str) -> dict[str, str]:
    """Hash the selected leaf module and shared canonical scenario contracts."""

    definition = load_scenario_definition(leaf_id)
    module_name, _, _ = definition.module_path.partition(":")
    leaf_module = importlib.import_module(module_name)
    package_root = Path(__file__).resolve().parent
    leaf_path = Path(str(getattr(leaf_module, "__file__", ""))).resolve(strict=True)
    files = {
        "scenario_module_py": leaf_path,
        "scenario_shared_recipe_py": package_root / "_rigid_shared.py",
        "scenario_registry_py": package_root / "registry.py",
        "scenario_types_py": package_root / "types.py",
    }
    return {name: sha256_file(path) for name, path in files.items()}


__all__ = [
    "implemented_review_variants",
    "list_scenario_definitions",
    "load_scenario_definition",
    "scenario_source_hashes",
]
