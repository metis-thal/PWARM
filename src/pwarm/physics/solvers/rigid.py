"""
Rigid Body Solver — Impulse-based rigid body dynamics with GJK/EPA collision.

Integrates with the new physics architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..core.entity import ComponentMask
from . import CouplingData
from .contact_kernels import solve_contacts_velocity

if TYPE_CHECKING:
    from ..core.state import State


@dataclass
class RigidOptions:
    solver_type: str = "impulse"  # "impulse" or "lcp"
    contact_tolerance: float = 1e-4
    max_iterations: int = 20
    ccd: bool = True
    gravity_scale: float = 1.0


class RigidSolver:
    """Rigid body solver with impulse-based contact resolution."""
    
    name = "rigid"
    required_components: ClassVar[list[ComponentMask]] = [ComponentMask.RIGID_BODY]
    
    def __init__(self, options: dict | None = None):
        self.options = RigidOptions(**(options or {}))
        self.scene = None
    
    def initialize(self, scene) -> None:
        self.scene = scene
    
    def step(self, state: State, dt: float, contacts: list) -> State:
        """Velocity-Verlet integration with impulse-based contacts."""
        new_state = state.copy()
        
        if state.rigid_pos is None or len(state.rigid_pos) == 0:
            return new_state
        
        len(state.rigid_pos)
        
        # 1. Apply gravity
        gravity = self.scene.gravity if self.scene else np.array([0, 0, -9.81], dtype=np.float32)
        accel = gravity * self.options.gravity_scale
        
        # Add accumulated forces (skip static bodies with mass=0)
        if state.rigid_force_accum is not None:
            safe_mass = np.where(state.rigid_mass > 0, state.rigid_mass, 1.0)[:, None]
            accel = accel + state.rigid_force_accum / safe_mass
            # Zero accel for static bodies
            static_mask = (state.rigid_mass <= 0)[:, None]
            accel = np.where(static_mask, 0.0, accel)
        
        # 2. Velocity-Verlet: v += a * dt
        new_state.rigid_linvel = state.rigid_linvel + accel * dt
        
        # 3. Position update: x += v * dt
        new_state.rigid_pos = state.rigid_pos + new_state.rigid_linvel * dt
        
        # 4. Angular velocity update (simplified)
        if state.rigid_torque_accum is not None and state.rigid_inv_inertia_local is not None:
            # Convert local inertia to world
            # For now, simple integration
            # (N,3,3) @ (N,3,1) -> (N,3,1) -> squeeze -> (N,3)
            torque_col = state.rigid_torque_accum[:, :, np.newaxis]
            ang_accel = (state.rigid_inv_inertia_local @ torque_col).squeeze(axis=2)
            new_state.rigid_angvel = state.rigid_angvel + ang_accel * dt
        
        # 4. Orientation update (quaternion)
        if state.rigid_angvel is not None:
            # q_new = q * exp(0.5 * ω * dt)
            omega = state.rigid_angvel
            omega_norm = np.linalg.norm(omega, axis=1, keepdims=True)
            mask = omega_norm > 1e-6
            if np.any(mask):
                half_angle = omega_norm * dt * 0.5
                axis = omega / (omega_norm + 1e-8)
                sin_half = np.sin(half_angle)
                cos_half = np.cos(half_angle)
                dq = np.concatenate([cos_half, axis * sin_half], axis=1)
                # Quaternion multiplication: q_new = q * dq
                q = state.rigid_quat
                new_state.rigid_quat = self._quat_multiply(q, dq)
                # Normalize
                new_state.rigid_quat = new_state.rigid_quat / np.linalg.norm(new_state.rigid_quat, axis=1, keepdims=True)
        
        # 5. Apply contact impulses (impulse-based)
        if contacts:
            new_state = self._resolve_contacts(new_state, contacts, dt)
        
        # Clear force accumulators for next step
        new_state.rigid_force_accum[:] = 0
        new_state.rigid_torque_accum[:] = 0
        
        return new_state
    
    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply quaternions q1 * q2. Shape: (N, 4)"""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.stack([w, x, y, z], axis=1)
    
    def _resolve_contacts(self, state: State, contacts: list, dt: float) -> State:
        """Resolve contact impulses using projected Gauss-Seidel."""
        # Filter rigid-rigid contacts
        rigid_contacts = [c for c in contacts 
                         if c.solver_a == "rigid" and c.solver_b == "rigid"]
        
        if not rigid_contacts:
            return state
        
        # Resolve entity ids to array indices once; unresolved pairs are skipped
        triples = []
        for contact in rigid_contacts:
            ia = state.entity_to_rigid.get(contact.entity_a)
            ib = state.entity_to_rigid.get(contact.entity_b)
            if ia is None or ib is None:
                continue
            triples.append((ia, ib, contact))
        if not triples:
            return state

        len(triples)
        idx_a_arr = np.array([t[0] for t in triples], dtype=np.int64)
        idx_b_arr = np.array([t[1] for t in triples], dtype=np.int64)
        normals = np.array([t[2].normal for t in triples], dtype=np.float64)
        fric_arr = np.array([t[2].friction for t in triples], dtype=np.float64)
        rest_arr = np.array([t[2].restitution for t in triples], dtype=np.float64)

        # Velocity pass: sequential impulses over flat arrays
        # (numba kernel when available, identical pure-Python fallback otherwise)
        linvel = solve_contacts_velocity(
            state.rigid_linvel.astype(np.float64),
            state.rigid_inv_mass.astype(np.float64),
            idx_a_arr, idx_b_arr, normals, fric_arr, rest_arr,
            self.options.max_iterations,
        )
        state.rigid_linvel[:] = linvel.astype(np.float32)

        # CCD rewind: contacts produced by conservative advancement carry the
        # mover's center position at the time of impact. End-of-step impulse
        # resolution would otherwise leave a very fast body far past the
        # impact point, so rewind it (A is always the CCD mover by
        # construction of CollisionSystem._conservative_ccd).
        rewound: set[int] = set()
        for contact in rigid_contacts:
            if contact.time_of_impact <= 0.0:
                continue
            idx_a = state.entity_to_rigid.get(contact.entity_a)
            if idx_a is None or idx_a in rewound:
                continue
            state.rigid_pos[idx_a] = np.asarray(contact.point, dtype=np.float32)
            rewound.add(idx_a)

        # Positional correction (single pass, once per contact): push bodies
        # out of penetration to prevent gradual sinking under gravity.
        # Partial correction with slop reduces jitter and energy injection.
        slop = 0.005
        beta = 0.8
        for contact in rigid_contacts:
            idx_a = state.entity_to_rigid.get(contact.entity_a)
            idx_b = state.entity_to_rigid.get(contact.entity_b)
            if idx_a is None or idx_b is None:
                continue
            inv_mass_a = state.rigid_inv_mass[idx_a]
            inv_mass_b = state.rigid_inv_mass[idx_b]
            inv_mass_sum = inv_mass_a + inv_mass_b
            if inv_mass_sum == 0:
                continue
            corr_mag = beta * max(contact.depth - slop, 0.0)
            if corr_mag > 0:
                corr = contact.normal * corr_mag
                state.rigid_pos[idx_a] -= corr * (inv_mass_a / inv_mass_sum)
                state.rigid_pos[idx_b] += corr * (inv_mass_b / inv_mass_sum)

        return state
    
    def get_coupling_data(self, state: State) -> CouplingData:
        data = CouplingData()
        if state.rigid_pos is not None:
            # Export positions/velocities for other solvers
            data.velocities["sph"] = state.rigid_linvel
            data.velocities["fem"] = state.rigid_linvel
            data.velocities["mpm"] = state.rigid_linvel
            data.velocities["pbd"] = state.rigid_linvel
            # AABBs for collision
            if state.rigid_pos is not None:
                # Simplified AABB
                pass
        return data
    
    def apply_coupling(self, state: State, coupling: CouplingData) -> State:
        new_state = state.copy()
        
        # Apply forces from other solvers
        if "sph" in coupling.forces:
            if new_state.rigid_force_accum is None:
                new_state.rigid_force_accum = coupling.forces["sph"]
            else:
                new_state.rigid_force_accum += coupling.forces["sph"]
        
        if "fem" in coupling.forces:
            if new_state.rigid_force_accum is None:
                new_state.rigid_force_accum = coupling.forces["fem"]
            else:
                new_state.rigid_force_accum += coupling.forces["fem"]
        
        if "mpm" in coupling.forces:
            if new_state.rigid_force_accum is None:
                new_state.rigid_force_accum = coupling.forces["mpm"]
            else:
                new_state.rigid_force_accum += coupling.forces["mpm"]
        
        if "pbd" in coupling.forces:
            if new_state.rigid_force_accum is None:
                new_state.rigid_force_accum = coupling.forces["pbd"]
            else:
                new_state.rigid_force_accum += coupling.forces["pbd"]
        
        return new_state


class RigidBodyBuilder:
    """Helper to create rigid body entities with proper components."""
    
    @staticmethod
    def create_sphere(position: np.ndarray, radius: float, mass: float = 1.0) -> dict:
        """Create components for a sphere rigid body."""
        from ..core.component import CollisionShapeComponent, RigidBodyComponent, TransformComponent
        from ..core.entity import ComponentMask
        
        return {
            "mask": ComponentMask.RIGID_DYNAMIC if mass > 0 else ComponentMask.RIGID_STATIC,
            "transform": TransformComponent(position=position),
            "rigid_body": RigidBodyComponent(
                mass=mass,
                inv_mass=1.0/mass if mass > 0 else 0.0,
                inertia_local=np.array([[2/5*mass*radius**2, 0, 0],
                                       [0, 2/5*mass*radius**2, 0],
                                       [0, 0, 2/5*mass*radius**2]], dtype=np.float32),
            ),
            "collision_shape": CollisionShapeComponent(
                shape_type=CollisionShapeComponent.ShapeType.SPHERE,
                radius=radius,
            ),
        }
    
    @staticmethod
    def create_box(position: np.ndarray, half_extents: np.ndarray, mass: float = 1.0) -> dict:
        """Create components for a box rigid body."""
        from ..core.component import CollisionShapeComponent, RigidBodyComponent, TransformComponent
        from ..core.entity import ComponentMask
        
        hx, hy, hz = half_extents
        inertia = np.diag([1/3*mass*(hy**2+hz**2),
                          1/3*mass*(hx**2+hz**2),
                          1/3*mass*(hx**2+hy**2)]).astype(np.float32)
        
        return {
            "mask": ComponentMask.RIGID_DYNAMIC if mass > 0 else ComponentMask.RIGID_STATIC,
            "transform": TransformComponent(position=position),
            "rigid_body": RigidBodyComponent(
                mass=mass,
                inv_mass=1.0/mass if mass > 0 else 0.0,
                inertia_local=inertia,
            ),
            "collision_shape": CollisionShapeComponent(
                shape_type=CollisionShapeComponent.ShapeType.BOX,
                half_extents=half_extents,
            ),
        }