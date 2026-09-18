"""The AI ↔ Physics contract, enforced (docs/architecture/ai-physics-contract.md).

Core rule: the AI may observe the world, but it can never read the answers.
Structurally: no AI-side module imports pymo.universes; the observation
channel carries measurements only; the Laboratory facade never hands out the
Universe.
"""

from __future__ import annotations

import ast
from pathlib import Path

from pymo.scientist import ObservationRecord

_SRC = Path(__file__).resolve().parents[2] / "src" / "pymo" / "scientist"

# Modules that ARE the AI (planning/cognition). experiment.py is the door —
# it must consume secrets to configure worlds but never re-export them.
AI_MODULES = [
    "agent.py", "designer.py", "state.py", "planner.py", "mission.py",
    "hypothesis.py", "knowledge.py", "information.py", "uncertainty.py",
    "budget.py", "experiment_value.py", "instrument.py",
]


def test_ai_modules_never_import_universes() -> None:
    for name in AI_MODULES:
        tree = ast.parse((_SRC / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("pymo.universes"), (name, alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("pymo.universes"), (name, node.module)


def test_observation_record_is_measurement_only() -> None:
    import dataclasses
    fields = {f.name for f in dataclasses.fields(ObservationRecord)}
    assert fields == {"experiment_id", "t", "z", "vx", "steps"}
    # no truth-shaped fields (values of hidden parameters, universe refs)
    assert not any("secret" in f or "truth" in f or "universe" in f
                   for f in fields)


def test_laboratory_hides_the_universe() -> None:
    import inspect

    from pymo.scientist.experiment import Laboratory, Universe
    public = [n for n, _ in inspect.getmembers(Laboratory, lambda m: not inspect.isfunction(m))
              if not n.startswith("_")]
    # A universe NAME may be public (dashboards show it); the Universe
    # object (secrets included) may not.
    assert "universe" not in public and "universe_ref" not in public, public
    assert not any(
        isinstance(getattr(Laboratory, n, None), property)
        and "universe" in n and n != "universe_name"
        for n in dir(Laboratory)
    )
    # and the type is not even importable from the AI-side namespace
    import pymo.scientist as s
    assert "Universe" not in s.__all__
    assert Universe.__module__ == "pymo.universes"


def test_experiments_do_not_read_secret_values_directly() -> None:
    """Experiment modules derive physics from RECORDS; the only sanctioned
    secret use is configuring a world inside ExperimentSession."""
    for path in (_SRC / "experiments").glob("*.py"):
        blob = path.read_text(encoding="utf-8")
        assert "secrets" not in blob, f"{path.name} touches universe secrets"
