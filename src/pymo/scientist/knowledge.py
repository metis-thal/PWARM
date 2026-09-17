"""Knowledge base — the AI scientist's accumulated civilization knowledge.

Laws established in past missions persist as JSON on disk so future missions
do not re-discover what is already known. This file is the AI's possession:
it is written by the scientist layer and read by the planner; it never
contains universe secrets, only what the AI derived from observations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class LawRecord:
    """One established law in the knowledge base."""

    name: str
    formula: str
    value: float
    unit: str
    confidence: float
    r2: float
    experiments: list[str] = field(default_factory=list)
    derived_by: str = ""
    # Material laws carry named properties instead of a single value
    # (e.g. {"restitution": 0.72, "friction": 0.4}).
    properties: dict[str, float] = field(default_factory=dict)


class KnowledgeBase:
    """Persistent law store — one JSON file per universe."""

    def __init__(self, path: Path | str, universe: str):
        self.path = Path(path)
        self.universe = universe
        self.laws: dict[str, LawRecord] = {}
        self.load()

    @classmethod
    def load_or_create(cls, path: Path | str, universe: str) -> KnowledgeBase:
        """Open the store, creating an empty one if the file does not exist."""
        return cls(path, universe)

    # -- persistence ---------------------------------------------------------

    def load(self) -> None:
        if not self.path.exists():
            self.laws = {}
            return
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.laws = {
            rec["name"]: LawRecord(
                name=rec["name"],
                formula=rec.get("formula", ""),
                value=float(rec.get("value", 0.0)),
                unit=rec.get("unit", ""),
                confidence=float(rec.get("confidence", 0.0)),
                r2=float(rec.get("r2", 0.0)),
                experiments=list(rec.get("experiments", [])),
                derived_by=rec.get("derived_by", ""),
                properties=dict(rec.get("properties", {})),
            )
            for rec in data.get("laws", [])
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "universe": self.universe,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "laws": [asdict(rec) for rec in self.laws.values()],
        }
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    # -- queries -------------------------------------------------------------

    def knows(self, claim: str, min_confidence: float = 0.999) -> bool:
        """True when the claim is already established well enough that a new
        mission should not re-run experiments for it."""
        law = self.laws.get(claim)
        return law is not None and law.confidence >= min_confidence

    def get(self, claim: str) -> LawRecord | None:
        return self.laws.get(claim)

    # -- mutation ------------------------------------------------------------

    def record_law(self, name: str, formula: str, value: float, unit: str,
                   confidence: float, r2: float, experiments: list[str],
                   derived_by: str) -> LawRecord:
        rec = LawRecord(
            name=name, formula=formula, value=value, unit=unit,
            confidence=confidence, r2=r2,
            experiments=list(experiments), derived_by=derived_by,
        )
        self.laws[name] = rec
        return rec

    def record_material(self, material: str, properties: dict[str, float],
                        confidence: float, r2: float, experiments: list[str],
                        derived_by: str) -> LawRecord:
        """Publish a material's measured property set as one knowledge entry."""
        rec = LawRecord(
            name=material, formula="material properties", value=0.0, unit="",
            confidence=confidence, r2=r2,
            experiments=list(experiments), derived_by=derived_by,
            properties=dict(properties),
        )
        self.laws[material] = rec
        return rec

    def material(self, name: str) -> dict[str, float] | None:
        """A material's measured properties, or None if unknown."""
        rec = self.laws.get(name)
        if rec is None or not rec.properties:
            return None
        return dict(rec.properties)

    def summary(self) -> str:
        if not self.laws:
            return f"knowledge: empty ({self.universe})"
        lines = [f"knowledge of '{self.universe}': {len(self.laws)} law(s)"]
        for rec in self.laws.values():
            unit = f" {rec.unit}" if rec.unit else ""
            lines.append(f"  {rec.name} = {rec.value:.6f}{unit}"
                         f"  confidence {rec.confidence * 100:.2f}%"
                         f"  [{rec.derived_by}]")
        return "\n".join(lines)
