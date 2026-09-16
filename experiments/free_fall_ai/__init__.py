"""P8 experiment package: closed-loop emergent-world experiments.

Each experiment follows the lifecycle setup() -> step() -> observe() ->
discover() -> evaluate(), with the physics engine as ground truth and the AI
as a learned approximation that only ever sees recorded observations.
"""

from .experiment import DiscoveryReport, FreeFallConfig, FreeFallExperiment

__all__ = ["DiscoveryReport", "FreeFallConfig", "FreeFallExperiment"]
