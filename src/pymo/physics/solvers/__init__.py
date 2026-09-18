"""
Base Solver Class — All physics solvers inherit from this.

Each solver owns disjoint DOFs in State. Solvers communicate ONLY via Coupler.
"""

from __future__ import annotations

# Concrete solvers import these from .base (kept as re-exports).
from .base import CouplingData, Solver

# Import stub solvers
# Import core solvers
from .rigid import RigidBodyBuilder, RigidOptions, RigidSolver
from .sph import SPHFluidBuilder, SPHOptions, SPHSolver
from .stubs import (
    ChemistryOptions,
    ChemistrySolver,
    FEMOptions,
    FEMSolver,
    GeologyOptions,
    GeologySolver,
    MPMOptions,
    MPMSolver,
    PBDOptions,
    PBDSolver,
    ThermalOptions,
    ThermalSolver,
)

__all__ = [
    "ChemistryOptions",
    "ChemistrySolver",
    "CouplingData",
    "FEMOptions",
    "FEMSolver",
    "GeologyOptions",
    "GeologySolver",
    "MPMOptions",
    "MPMSolver",
    "PBDOptions",
    "PBDSolver",
    "RigidBodyBuilder",
    "RigidOptions",
    "RigidSolver",
    "SPHFluidBuilder",
    "SPHOptions",
    "SPHSolver",
    "Solver",
    "ThermalOptions",
    "ThermalSolver",
]