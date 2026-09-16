"""Universe definitions — the physics-side truth of each world.

A universe directory declares a world's parameters. Values marked
``hidden: true`` are the ground truth of that universe: they are loaded ONLY
on the physics side (``pymo.scientist.experiment.Laboratory`` consumes them to
configure the WorldEngine). The AI scientist layer never imports this package
and never receives a :class:`Universe` object — it receives a ``Laboratory``
facade that can execute experiments and return observations, nothing more.

Core rule enforced by construction:
    physics engine = absolute truth · AI = learned approximation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_UNIVERSES_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class UniverseSecrets:
    """Hidden world parameters. Physics-side only — never handed to the AI."""

    gravity: float | None = None
    air_density: float | None = None


@dataclass(frozen=True)
class Universe:
    """A world definition: public name + physics-only secrets + world rules."""

    name: str
    secrets: UniverseSecrets
    rules: dict

    @property
    def id(self) -> str:
        return self.name


def _hidden_value(section: dict | None) -> float | None:
    """Extract the numeric truth from a ``{hidden: true, value/density: X}``
    section; returns None when the section is absent or not hidden."""
    if not isinstance(section, dict) or not section.get("hidden", False):
        return None
    value = section.get("value", section.get("density"))
    return None if value is None else float(value)


def load_universe(universe_id: str) -> Universe:
    """Load a universe definition from ``pymo/universes/<universe_id>/``."""
    base = _UNIVERSES_DIR / universe_id
    config_path = base / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"unknown universe: {universe_id} ({config_path})")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    rules: dict = {}
    rules_path = base / "rules.yaml"
    if rules_path.exists():
        with rules_path.open("r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh) or {}

    secrets = UniverseSecrets(
        gravity=_hidden_value(config.get("gravity")),
        air_density=_hidden_value(config.get("air")),
    )
    return Universe(
        name=str(config.get("name", universe_id)),
        secrets=secrets,
        rules=rules,
    )
