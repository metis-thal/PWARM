"""
State — Immutable physics snapshot at time t.

Double-buffered: simulation writes to write-buffer, render reads from read-buffer.
All arrays are contiguous, flat, solver-owned. No references between solvers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .entity import EntityID, ComponentMask


@dataclass(slots=True)
class GlobalQuantities:
    """Conserved quantities for monitoring."""
    total_mass: float = 0.0
    total_linear_momentum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    total_angular_momentum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    total_energy: float = 0.0
    total_kinetic_energy: float = 0.0
    total_potential_energy: float = 0.0
    total_thermal_energy: float = 0.0
    total_chemical_energy: float = 0.0
    
    def compute_from_state(self, state: State) -> None:
        """Recompute all quantities from current state."""
        # Rigid bodies
        if state.rigid_mass is not None and len(state.rigid_mass) > 0:
            masses = state.rigid_mass
            linvel = state.rigid_linvel
            angvel = state.rigid_angvel
            
            self.total_mass = float(np.sum(masses))
            self.total_linear_momentum = np.sum(masses[:, None] * linvel, axis=0).astype(np.float64)
            
            # Angular momentum: sum(r x p) + I*ω
            if state.rigid_pos is not None:
                r = state.rigid_pos
                p = masses[:, None] * linvel
                self.total_angular_momentum = np.sum(np.cross(r, p), axis=0).astype(np.float64)
            
            # Kinetic energy
            ke_linear = 0.5 * np.sum(masses * np.sum(linvel**2, axis=1))
            self.total_kinetic_energy = float(ke_linear)
        
        # SPH kinetic energy
        if state.sph_mass is not None and len(state.sph_mass) > 0:
            if state.sph_vel is not None:
                ke_sph = 0.5 * np.sum(state.sph_mass * np.sum(state.sph_vel**2, axis=1))
                self.total_kinetic_energy += float(ke_sph)
        
        # Thermal energy
        if state.thermal_temp is not None and len(state.thermal_temp) > 0:
            # Simplified: E_thermal = sum(m * c * T)
            pass
    
    def relative_change(self, other: GlobalQuantities) -> dict[str, float]:
        """Compute relative change vs another state."""
        changes = {}
        for key in ["total_mass", "total_energy", "total_kinetic_energy"]:
            a = getattr(self, key)
            b = getattr(other, key)
            if abs(b) > 1e-12:
                changes[key] = abs(a - b) / abs(b)
            else:
                changes[key] = 0.0
        return changes


@dataclass(slots=True)
class State:
    """
    Immutable physics state at time t.
    
    All arrays are flat, contiguous, solver-owned.
    Solver i owns its arrays; other solvers read via coupling only.
    """
    # Time
    t: float = 0.0
    dt: float = 1.0 / 60.0
    
    # === RIGID BODIES (RigidSolver) ===
    rigid_pos: np.ndarray | None = None          # (N_rigid, 3)
    rigid_quat: np.ndarray | None = None         # (N_rigid, 4) w,x,y,z
    rigid_linvel: np.ndarray | None = None       # (N_rigid, 3)
    rigid_angvel: np.ndarray | None = None       # (N_rigid, 3)
    rigid_mass: np.ndarray | None = None         # (N_rigid,)
    rigid_inv_mass: np.ndarray | None = None     # (N_rigid,)
    rigid_inertia_local: np.ndarray | None = None      # (N_rigid, 3, 3)
    rigid_inv_inertia_local: np.ndarray | None = None  # (N_rigid, 3, 3)
    rigid_force_accum: np.ndarray | None = None       # (N_rigid, 3)
    rigid_torque_accum: np.ndarray | None = None      # (N_rigid, 3)
    
    # === SPH FLUID (SPHSolver) ===
    sph_pos: np.ndarray | None = None            # (N_sph, 3)
    sph_vel: np.ndarray | None = None            # (N_sph, 3)
    sph_density: np.ndarray | None = None        # (N_sph,)
    sph_pressure: np.ndarray | None = None       # (N_sph,)
    sph_mass: np.ndarray | None = None           # (N_sph,)
    sph_smoothing_h: float = 0.025
    sph_rest_density: float = 1000.0
    sph_viscosity: float = 0.1
    sph_surface_tension: float = 0.072
    sph_boundary_pos: np.ndarray | None = None   # (N_boundary, 3)
    sph_boundary_normal: np.ndarray | None = None # (N_boundary, 3)
    sph_neighbor_indices: list | None = None     # List of neighbor lists
    sph_rigid_contact_forces: np.ndarray | None = None  # (N_sph, 3)
    
    # === FEM DEFORMABLE (FEMSolver) ===
    fem_pos: np.ndarray | None = None            # (N_fem, 3)
    fem_vel: np.ndarray | None = None            # (N_fem, 3)
    fem_rest_pos: np.ndarray | None = None       # (N_fem, 3)
    fem_velocity: np.ndarray | None = None       # Alias
    fem_youngs_modulus: float = 1e5
    fem_poissons_ratio: float = 0.3
    fem_density: float = 1000.0
    fem_elements: np.ndarray | None = None       # (E, 4) tetrahedra
    fem_F: np.ndarray | None = None              # (E, 3, 3) deformation gradient
    fem_fixed_nodes: np.ndarray | None = None    # (K,) fixed node indices
    fem_contact_forces: np.ndarray | None = None # (N_fem, 3)
    
    # === MPM (MPMSolver) ===
    mpm_pos: np.ndarray | None = None            # (N_mpm, 3)
    mpm_vel: np.ndarray | None = None            # (N_mpm, 3)
    mpm_mass: np.ndarray | None = None           # (N_mpm,)
    mpm_volume: np.ndarray | None = None         # (N_mpm,)
    mpm_F: np.ndarray | None = None              # (N_mpm, 3, 3)
    mpm_material: np.ndarray | None = None       # (N_mpm,) int
    mpm_youngs_modulus: float = 1e5
    mpm_poissons_ratio: float = 0.3
    mpm_hardening: float = 10.0
    mpm_grid_size: tuple[int, int, int] = (64, 64, 64)
    mpm_grid_origin: np.ndarray | None = None
    mpm_cell_size: float = 0.05
    
    # === PBD (PBDSolver) ===
    pbd_pos: np.ndarray | None = None            # (N_pbd, 3)
    pbd_pred_pos: np.ndarray | None = None       # (N_pbd, 3)
    pbd_vel: np.ndarray | None = None            # (N_pbd, 3)
    pbd_inv_mass: np.ndarray | None = None       # (N_pbd,)
    pbd_distance_constraints: np.ndarray | None = None    # (C, 2)
    pbd_distance_rest: np.ndarray | None = None           # (C,)
    pbd_bending_constraints: np.ndarray | None = None     # (C, 3/4)
    pbd_bending_rest: np.ndarray | None = None            # (C,)
    pbd_collision_radius: float = 0.01
    pbd_external_forces: np.ndarray | None = None         # (N_pbd, 3)
    
    # === THERMAL (ThermalSolver) ===
    thermal_temp: np.ndarray | None = None       # (N_thermal,) or (Nx, Ny, Nz)
    thermal_conductivity: float = 1.0
    thermal_specific_heat: float = 1000.0
    thermal_density: float = 1000.0
    thermal_melting_temp: float = 273.15
    thermal_latent_heat: float = 3.34e5
    thermal_fixed_nodes: np.ndarray | None = None
    thermal_fixed_temps: np.ndarray | None = None
    thermal_heat_source: float = 0.0
    thermal_flux_from_chemistry: float = 0.0
    thermal_flux_from_rigid: float = 0.0
    
    # === CHEMISTRY (ChemistrySolver) ===
    chem_conc: np.ndarray | None = None          # (N_chem, num_species) or (grid, num_species)
    chem_num_species: int = 4
    chem_diffusion: np.ndarray | None = None     # (num_species,)
    chem_reaction_rates: dict = field(default_factory=dict)
    chem_sources: np.ndarray | None = None
    chem_advection_vel: np.ndarray | None = None # (N_chem, 3)
    
    # === GEOLOGY (GeologySolver) ===
    geo_rock_type: np.ndarray | None = None      # (Nx, Ny, Nz) int
    geo_porosity: np.ndarray | None = None       # (Nx, Ny, Nz)
    geo_permeability: np.ndarray | None = None   # (Nx, Ny, Nz)
    geo_layer_ids: np.ndarray | None = None      # (Nx, Ny, Nz)
    geo_layer_ages: np.ndarray | None = None     # (num_layers,)
    geo_elevation: np.ndarray | None = None      # (Nx, Ny)
    geo_sediment_flux: np.ndarray | None = None  # (Nx, Ny)
    geo_uplift_rate: np.ndarray | None = None    # (Nx, Ny)
    geo_stress: np.ndarray | None = None         # (Nx, Ny, Nz, 3, 3)
    geo_erosion_rate: float = 0.0
    geo_thermal_cond_map: dict = field(default_factory=dict)
    
    # === GLOBAL QUANTITIES ===
    global_quantities: GlobalQuantities = field(default_factory=GlobalQuantities)
    
    # === METADATA ===
    entity_to_rigid: dict = field(default_factory=dict)     # EntityID -> rigid_index
    entity_to_sph: dict = field(default_factory=dict)       # EntityID -> (start, count)
    entity_to_fem: dict = field(default_factory=dict)       # EntityID -> (start, count)
    entity_to_mpm: dict = field(default_factory=dict)       # EntityID -> (start, count)
    entity_to_pbd: dict = field(default_factory=dict)       # EntityID -> (start, count)
    entity_to_thermal: dict = field(default_factory=dict)   # EntityID -> thermal_index
    entity_to_chemistry: dict = field(default_factory=dict) # EntityID -> chemistry_index
    entity_to_geology: dict = field(default_factory=dict)   # EntityID -> geology_index
    
    def copy(self) -> State:
        """Deep copy for double-buffer swap."""
        # Shallow copy of arrays (they're immutable after write)
        new = State(
            t=self.t,
            dt=self.dt,
            # Rigid
            rigid_pos=self.rigid_pos.copy() if self.rigid_pos is not None else None,
            rigid_quat=self.rigid_quat.copy() if self.rigid_quat is not None else None,
            rigid_linvel=self.rigid_linvel.copy() if self.rigid_linvel is not None else None,
            rigid_angvel=self.rigid_angvel.copy() if self.rigid_angvel is not None else None,
            rigid_mass=self.rigid_mass.copy() if self.rigid_mass is not None else None,
            rigid_inv_mass=self.rigid_inv_mass.copy() if self.rigid_inv_mass is not None else None,
            rigid_inertia_local=self.rigid_inertia_local.copy() if self.rigid_inertia_local is not None else None,
            rigid_inv_inertia_local=self.rigid_inv_inertia_local.copy() if self.rigid_inv_inertia_local is not None else None,
            rigid_force_accum=self.rigid_force_accum.copy() if self.rigid_force_accum is not None else None,
            rigid_torque_accum=self.rigid_torque_accum.copy() if self.rigid_torque_accum is not None else None,
            # SPH
            sph_pos=self.sph_pos.copy() if self.sph_pos is not None else None,
            sph_vel=self.sph_vel.copy() if self.sph_vel is not None else None,
            sph_density=self.sph_density.copy() if self.sph_density is not None else None,
            sph_pressure=self.sph_pressure.copy() if self.sph_pressure is not None else None,
            sph_mass=self.sph_mass.copy() if self.sph_mass is not None else None,
            sph_smoothing_h=self.sph_smoothing_h,
            sph_rest_density=self.sph_rest_density,
            sph_viscosity=self.sph_viscosity,
            sph_surface_tension=self.sph_surface_tension,
            sph_boundary_pos=self.sph_boundary_pos.copy() if self.sph_boundary_pos is not None else None,
            sph_boundary_normal=self.sph_boundary_normal.copy() if self.sph_boundary_normal is not None else None,
            sph_neighbor_indices=self.sph_neighbor_indices,
            sph_rigid_contact_forces=self.sph_rigid_contact_forces.copy() if self.sph_rigid_contact_forces is not None else None,
            # FEM
            fem_pos=self.fem_pos.copy() if self.fem_pos is not None else None,
            fem_vel=self.fem_vel.copy() if self.fem_vel is not None else None,
            fem_rest_pos=self.fem_rest_pos.copy() if self.fem_rest_pos is not None else None,
            fem_youngs_modulus=self.fem_youngs_modulus,
            fem_poissons_ratio=self.fem_poissons_ratio,
            fem_density=self.fem_density,
            fem_elements=self.fem_elements.copy() if self.fem_elements is not None else None,
            fem_F=self.fem_F.copy() if self.fem_F is not None else None,
            fem_fixed_nodes=self.fem_fixed_nodes.copy() if self.fem_fixed_nodes is not None else None,
            fem_contact_forces=self.fem_contact_forces.copy() if self.fem_contact_forces is not None else None,
            # MPM
            mpm_pos=self.mpm_pos.copy() if self.mpm_pos is not None else None,
            mpm_vel=self.mpm_vel.copy() if self.mpm_vel is not None else None,
            mpm_mass=self.mpm_mass.copy() if self.mpm_mass is not None else None,
            mpm_volume=self.mpm_volume.copy() if self.mpm_volume is not None else None,
            mpm_F=self.mpm_F.copy() if self.mpm_F is not None else None,
            mpm_material=self.mpm_material.copy() if self.mpm_material is not None else None,
            mpm_youngs_modulus=self.mpm_youngs_modulus,
            mpm_poissons_ratio=self.mpm_poissons_ratio,
            mpm_hardening=self.mpm_hardening,
            mpm_grid_size=self.mpm_grid_size,
            mpm_grid_origin=self.mpm_grid_origin.copy() if self.mpm_grid_origin is not None else None,
            mpm_cell_size=self.mpm_cell_size,
            # PBD
            pbd_pos=self.pbd_pos.copy() if self.pbd_pos is not None else None,
            pbd_pred_pos=self.pbd_pred_pos.copy() if self.pbd_pred_pos is not None else None,
            pbd_vel=self.pbd_vel.copy() if self.pbd_vel is not None else None,
            pbd_inv_mass=self.pbd_inv_mass.copy() if self.pbd_inv_mass is not None else None,
            pbd_distance_constraints=self.pbd_distance_constraints.copy() if self.pbd_distance_constraints is not None else None,
            pbd_distance_rest=self.pbd_distance_rest.copy() if self.pbd_distance_rest is not None else None,
            pbd_bending_constraints=self.pbd_bending_constraints.copy() if self.pbd_bending_constraints is not None else None,
            pbd_bending_rest=self.pbd_bending_rest.copy() if self.pbd_bending_rest is not None else None,
            pbd_collision_radius=self.pbd_collision_radius,
            pbd_external_forces=self.pbd_external_forces.copy() if self.pbd_external_forces is not None else None,
            # Thermal
            thermal_temp=self.thermal_temp.copy() if self.thermal_temp is not None else None,
            thermal_conductivity=self.thermal_conductivity,
            thermal_specific_heat=self.thermal_specific_heat,
            thermal_density=self.thermal_density,
            thermal_melting_temp=self.thermal_melting_temp,
            thermal_latent_heat=self.thermal_latent_heat,
            thermal_fixed_nodes=self.thermal_fixed_nodes.copy() if self.thermal_fixed_nodes is not None else None,
            thermal_fixed_temps=self.thermal_fixed_temps.copy() if self.thermal_fixed_temps is not None else None,
            thermal_heat_source=self.thermal_heat_source,
            thermal_flux_from_chemistry=self.thermal_flux_from_chemistry,
            thermal_flux_from_rigid=self.thermal_flux_from_rigid,
            # Chemistry
            chem_conc=self.chem_conc.copy() if self.chem_conc is not None else None,
            chem_num_species=self.chem_num_species,
            chem_diffusion=self.chem_diffusion.copy() if self.chem_diffusion is not None else None,
            chem_reaction_rates=dict(self.chem_reaction_rates),
            chem_sources=self.chem_sources.copy() if self.chem_sources is not None else None,
            chem_advection_vel=self.chem_advection_vel.copy() if self.chem_advection_vel is not None else None,
            # Geology
            geo_rock_type=self.geo_rock_type.copy() if self.geo_rock_type is not None else None,
            geo_porosity=self.geo_porosity.copy() if self.geo_porosity is not None else None,
            geo_permeability=self.geo_permeability.copy() if self.geo_permeability is not None else None,
            geo_layer_ids=self.geo_layer_ids.copy() if self.geo_layer_ids is not None else None,
            geo_layer_ages=self.geo_layer_ages.copy() if self.geo_layer_ages is not None else None,
            geo_elevation=self.geo_elevation.copy() if self.geo_elevation is not None else None,
            geo_sediment_flux=self.geo_sediment_flux.copy() if self.geo_sediment_flux is not None else None,
            geo_uplift_rate=self.geo_uplift_rate.copy() if self.geo_uplift_rate is not None else None,
            geo_stress=self.geo_stress.copy() if self.geo_stress is not None else None,
            geo_erosion_rate=self.geo_erosion_rate,
            geo_thermal_cond_map=dict(self.geo_thermal_cond_map),
            # Global
            global_quantities=self.global_quantities,
            # Metadata
            entity_to_rigid=dict(self.entity_to_rigid),
            entity_to_sph=dict(self.entity_to_sph),
            entity_to_fem=dict(self.entity_to_fem),
            entity_to_mpm=dict(self.entity_to_mpm),
            entity_to_pbd=dict(self.entity_to_pbd),
            entity_to_thermal=dict(self.entity_to_thermal),
            entity_to_chemistry=dict(self.entity_to_chemistry),
            entity_to_geology=dict(self.entity_to_geology),
        )
        return new
    
    def get_rigid_slice(self, entity_id) -> slice | None:
        """Get array slice for a rigid entity."""
        idx = self.entity_to_rigid.get(entity_id)
        if idx is not None:
            return slice(idx, idx + 1)
        return None
    
    def get_sph_slice(self, entity_id) -> slice | None:
        """Get array slice for an SPH entity."""
        rng = self.entity_to_sph.get(entity_id)
        if rng is not None:
            return slice(rng[0], rng[0] + rng[1])
        return None
    
    def get_fem_slice(self, entity_id) -> slice | None:
        rng = self.entity_to_fem.get(entity_id)
        if rng is not None:
            return slice(rng[0], rng[0] + rng[1])
        return None
    
    def get_mpm_slice(self, entity_id) -> slice | None:
        rng = self.entity_to_mpm.get(entity_id)
        if rng is not None:
            return slice(rng[0], rng[0] + rng[1])
        return None
    
    def get_pbd_slice(self, entity_id) -> slice | None:
        rng = self.entity_to_pbd.get(entity_id)
        if rng is not None:
            return slice(rng[0], rng[0] + rng[1])
        return None
    
    def num_rigid(self) -> int:
        return len(self.rigid_pos) if self.rigid_pos is not None else 0
    
    def num_sph(self) -> int:
        return len(self.sph_pos) if self.sph_pos is not None else 0
    
    def num_fem(self) -> int:
        return len(self.fem_pos) if self.fem_pos is not None else 0
    
    def num_mpm(self) -> int:
        return len(self.mpm_pos) if self.mpm_pos is not None else 0
    
    def num_pbd(self) -> int:
        return len(self.pbd_pos) if self.pbd_pos is not None else 0
    
    def num_thermal(self) -> int:
        return len(self.thermal_temp) if self.thermal_temp is not None else 0
    
    def num_chemistry(self) -> int:
        return len(self.chem_conc) if self.chem_conc is not None else 0
    
    def __repr__(self) -> str:
        return (f"State(t={self.t:.3f}, rigid={self.num_rigid()}, sph={self.num_sph()}, "
                f"fem={self.num_fem()}, mpm={self.num_mpm()}, pbd={self.num_pbd()}, "
                f"thermal={self.num_thermal()}, chem={self.num_chemistry()})")