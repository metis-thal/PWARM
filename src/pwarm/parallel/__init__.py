"""Ray-based parallel simulation infrastructure for pwarm.

``ray`` is an optional dependency (the ``parallel`` extra).  Importing this
package never requires ray; the symbols below resolve lazily and raise an
actionable error when ray is absent.  Resolving a symbol can still fail on
the deleted ``pwarm.kernel`` (see DEFERRED_WORK.md, "Production Import
Integrity") — that defect is tracked separately and is not masked here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # editor support only; never executed at runtime
    from .ray_parallel import (
        CheckpointManager,
        ExperimentRunner,
        SimulationBatch,
        SimulationConfig,
        SimulationResult,
        SimulationWorker,
        run_parallel_sims,
        run_single_simulation,
    )

__all__ = [
    "CheckpointManager",
    "ExperimentRunner",
    "SimulationBatch",
    "SimulationConfig",
    "SimulationResult",
    "SimulationWorker",
    "run_parallel_sims",
    "run_single_simulation",
]

_RAY_INSTALL_HINT = (
    "pwarm.parallel needs the optional dependency 'ray' (extra 'parallel'). "
    "Install it with: pip install 'PWARM[parallel]'"
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from . import ray_parallel
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] == "ray":
            raise ModuleNotFoundError(_RAY_INSTALL_HINT) from exc
        raise  # e.g. the registered pwarm.kernel defect — never masked here
    return getattr(ray_parallel, name)


def __dir__() -> list[str]:
    return sorted(__all__)
