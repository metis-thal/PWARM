"""
AI Layer — Law Discovery, Closed-Loop Experimentation.

Core Rule: Physics engine is absolute truth (ground truth); AI is learned approximation.
Never conflate the two.
"""

from .closed_loop import ClosedLoopAI, PredictionResult
from .law_discovery import (
    DiscoveredLaw,
    GplearnRefineBackend,
    LawDiscovery,
    SymbolicBackend,
)

__all__ = [
    "ClosedLoopAI",
    "DiscoveredLaw",
    "GplearnRefineBackend",
    "LawDiscovery",
    "PredictionResult",
    "SymbolicBackend",
]