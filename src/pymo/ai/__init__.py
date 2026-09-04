"""
AI Layer — Observer, Law Discovery, Closed-Loop Experimentation, Autonomous Experiments.

Core Rule: Physics engine is absolute truth (ground truth); AI is learned approximation.
Never conflate the two.
"""

from .observer import WorldObserver, TimeSeriesDataset, Observation, collect_free_fall
from .law_discovery import (
    LawDiscovery, DiscoveredLaw, GplearnRefineBackend,
    SymbolicBackend,
)
from .closed_loop import ClosedLoopAI, PredictionResult
from .experiment import (
    ExperimentConfig, ExperimentResult, Hypothesis,
    ParameterSpace, ExperimentRunner, AutonomousExperimenter,
    create_free_fall_experiment, create_collision_experiment,
    create_parameter_sweep_experiment,
)

__all__ = [
    # Observer
    "WorldObserver", "TimeSeriesDataset", "Observation", "collect_free_fall",
    # Law Discovery
    "LawDiscovery", "DiscoveredLaw", "GplearnRefineBackend", "SymbolicBackend",
    # Closed Loop
    "ClosedLoopAI", "PredictionResult",
    # Experiments
    "ExperimentConfig", "ExperimentResult", "Hypothesis",
    "ParameterSpace", "ExperimentRunner", "AutonomousExperimenter",
    "create_free_fall_experiment", "create_collision_experiment", 
    "create_parameter_sweep_experiment",
]