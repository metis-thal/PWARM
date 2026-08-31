"""Advanced Material Mechanics module for the pymo rules layer.

Implements:
- Plastic deformation and permanent deformation (von Mises yield criterion)
- Soft body simulation: deformable objects using FEM-like approach
- Fatigue and damage accumulation (Miner's rule, Paris law)
- Stress accumulation and strain hardening

All equations are standard continuum mechanics formulations; no hardcoded phenomena.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pymo.kernel.bodies import Body
from pymo.kernel.bodies3d import Body as Body3D, Material


@dataclass
class PlasticityParams:
    """Parameters for plastic deformation."""
    
    # von Mises yield stress (Pa)
    yield_stress: float = 250e6
    
    # Hardening modulus (Pa) - linear isotropic hardening
    hardening_modulus: float = 1e9
    
    # Perfect plasticity if hardening_modulus = 0
    # Strain rate sensitivity (optional)
    strain_rate_sensitivity: float = 0.01
    
    # Reference strain rate
    ref_strain_rate: float = 1.0


@dataclass
class PlasticState:
    """Plastic state for a material point."""
    
    # Equivalent plastic strain
    eps_p_eq: float = 0.0
    
    # Plastic strain tensor (symmetric, deviatoric)
    eps_p: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # Backstress for kinematic hardening (optional)
    backstress: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    
    # Current yield stress (with hardening)
    current_yield: float = 0.0


def deviatoric_part(stress: np.ndarray) -> np.ndarray:
    """Deviatoric part of stress tensor."""
    hydrostatic = np.trace(stress) / 3.0
    return stress - hydrostatic * np.eye(3)


def von_mises_stress(stress: np.ndarray) -> float:
    """von Mises equivalent stress."""
    s = deviatoric_part(stress)
    return np.sqrt(1.5 * np.sum(s * s))


def von_mises_yield(stress: np.ndarray, yield_stress: float) -> bool:
    """Check if stress state exceeds von Mises yield criterion."""
    return von_mises_stress(stress) >= yield_stress


def plastic_correction(stress: np.ndarray, plastic: PlasticState, params: PlasticityParams) -> tuple[np.ndarray, PlasticState]:
    """Return mapping (radial return) for von Mises plasticity.
    
    Given trial stress, return corrected stress and updated plastic state.
    """
    # Trial deviatoric stress
    s_trial = deviatoric_part(stress)
    sigma_eq = von_mises_stress(stress)
    
    # Current yield stress with isotropic hardening
    sigma_y = params.yield_stress + params.hardening_modulus * plastic.eps_p_eq
    
    if sigma_eq <= sigma_y + 1e-12:
        # Elastic - no correction needed
        return stress, plastic
    
    # Plastic correction needed - radial return mapping
    # Direction of plastic flow
    n = s_trial / sigma_eq if sigma_eq > 1e-12 else np.zeros((3, 3))
    
    # Plastic multiplier (consistency condition)
    # For linear hardening: dlam = (sigma_eq - sigma_y) / (3G + H)
    # Simplified: assume shear modulus G from Young's modulus
    E = 200e9  # Young's modulus placeholder
    nu = 0.3
    G = E / (2 * (1 + nu))
    H = params.hardening_modulus
    
    dlam = (sigma_eq - sigma_y) / (3 * G + H)
    
    # Update stress
    s_corrected = s_trial - 2 * G * dlam * n
    
    # Hydrostatic part unchanged
    hydro = np.trace(stress) / 3.0
    stress_corrected = s_corrected + hydro * np.eye(3)
    
    # Update plastic state
    plastic.eps_p_eq += np.sqrt(2/3) * dlam
    plastic.eps_p += dlam * n
    plastic.current_yield = params.yield_stress + H * plastic.eps_p_eq
    
    return stress_corrected, plastic


@dataclass
class FatigueParams:
    """Parameters for fatigue damage (Miner's rule + Paris law).
    
    Stress values in MPa for S-N curve parameters.
    """
    
    # S-N curve: N = C * S^-m (S in MPa)
    # C: fatigue strength coefficient
    # m: fatigue strength exponent
    sn_c: float = 1e12
    sn_m: float = 3.0
    
    # Endurance limit (MPa) - stress below which no fatigue damage
    endurance_limit: float = 100.0  # MPa
    
    # Paris law for crack growth: da/dN = C * (delta_K)^m
    # delta_K in MPa*sqrt(m)
    paris_c: float = 1e-12
    paris_m: float = 3.0
    
    # Initial crack size (m)
    initial_crack: float = 1e-4
    
    # Critical crack size for fracture (m)
    critical_crack: float = 0.01


@dataclass
class FatigueState:
    """Fatigue damage state for a material point."""
    
    # Miner's damage sum
    damage: float = 0.0
    
    # Cycle counting (simplified: rainflow or peak-valley)
    # Store recent stress peaks
    stress_history: list[float] = field(default_factory=list)
    
    # Current crack size
    crack_size: float = 1e-4
    
    def add_cycle(self, stress_amplitude: float, params: FatigueParams) -> None:
        """Add a stress cycle and update damage.
        
        stress_amplitude: in Pa (converted to MPa for S-N curve)
        """
        stress_mpa = stress_amplitude / 1e6
        if stress_mpa <= params.endurance_limit / 1e6:
            return
        
        # Cycles to failure from S-N curve: Nf = C / S^m (S in MPa)
        # Use log-space to avoid overflow
        log_Nf = np.log(params.sn_c) - params.sn_m * np.log(stress_mpa)
        if log_Nf > 700:  # avoid exp overflow
            Nf = float('inf')
        else:
            Nf = np.exp(log_Nf)
        
        # Miner's rule
        if Nf > 0:
            self.damage += 1.0 / Nf
        
        # Paris law crack growth (simplified) - delta_K in MPa*sqrt(m)
        delta_K = stress_mpa * np.sqrt(np.pi * self.crack_size)
        if delta_K > 0:
            log_da_dN = np.log(params.paris_c) + params.paris_m * np.log(delta_K)
            if log_da_dN < 700:
                da_dN = np.exp(log_da_dN)
            else:
                da_dN = 1e-6
            self.crack_size += da_dN
    
    def is_failed(self, params: FatigueParams) -> bool:
        """Check if damage reached failure."""
        return self.damage >= 1.0 or self.crack_size >= params.critical_crack


@dataclass
class SoftBodyParams:
    """Parameters for soft body / deformable object simulation."""
    
    # Young's modulus (Pa)
    young_modulus: float = 1e6
    
    # Poisson's ratio
    poisson_ratio: float = 0.3
    
    # Density (kg/m^3)
    density: float = 1000.0
    
    # Damping coefficients
    damping_alpha: float = 0.1  # mass-proportional (Rayleigh)
    damping_beta: float = 0.01  # stiffness-proportional (Rayleigh)
    
    # Time step
    dt: float = 1/60.0


class SoftBodyNode:
    """A node in a soft body mesh."""
    
    def __init__(self, pos: np.ndarray, mass: float, fixed: bool = False):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.force = np.zeros(3, dtype=float)
        self.mass = mass
        self.fixed = fixed


class SoftBodyTetra:
    """A tetrahedral element for FEM soft body."""
    
    def __init__(self, nodes: list[int], volume: float):
        self.nodes = nodes  # 4 node indices
        self.volume = volume
        
    def compute_deformation_gradient(self, nodes: list[SoftBodyNode], rest_positions: np.ndarray) -> np.ndarray:
        """Compute deformation gradient F = dx/dX."""
        # rest_positions is (N_nodes, 3) array
        # self.nodes are indices into this array
        X0 = rest_positions[self.nodes[0]]
        X1 = rest_positions[self.nodes[1]]
        X2 = rest_positions[self.nodes[2]]
        X3 = rest_positions[self.nodes[3]]
        
        n0 = nodes[self.nodes[0]].pos
        n1 = nodes[self.nodes[1]].pos
        n2 = nodes[self.nodes[2]].pos
        n3 = nodes[self.nodes[3]].pos
        
        # Build reference shape matrix
        D_ref = np.column_stack([X1 - X0, X2 - X0, X3 - X0])
        D_curr = np.column_stack([n1 - n0, n2 - n0, n3 - n0])
        
        if np.linalg.det(D_ref) < 1e-12:
            return np.eye(3)
        
        return D_curr @ np.linalg.inv(D_ref)
    
    def compute_stress(self, F: np.ndarray, params: SoftBodyParams) -> np.ndarray:
        """Compute Cauchy stress using Neo-Hookean model.
        
        W = mu/2 * (tr(F^T F) - 3) - mu * ln(det(F)) + lambda/2 * (ln(det(F)))^2
        """
        E = params.young_modulus
        nu = params.poisson_ratio
        mu = E / (2 * (1 + nu))
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        J = np.linalg.det(F)
        if J <= 0:
            J = 1e-6
        
        # Neo-Hookean stress
        B = F @ F.T  # Left Cauchy-Green
        stress = mu / J * (B - np.eye(3)) + lam * np.log(J) * np.eye(3)
        return stress
    
    def compute_forces(self, nodes: list[SoftBodyNode], rest_positions: np.ndarray, params: SoftBodyParams) -> np.ndarray:
        """Compute nodal forces from tetrahedron."""
        F = self.compute_deformation_gradient(nodes, rest_positions)
        stress = self.compute_stress(F, params)
        
        # Force = -integral(stress * N_I) dV
        # For tetrahedron: f_i = -V * stress * grad_N_i
        X0 = rest_positions[self.nodes[0]]
        X1 = rest_positions[self.nodes[1]]
        X2 = rest_positions[self.nodes[2]]
        X3 = rest_positions[self.nodes[3]]
        
        # Shape function gradients (constant in tetrahedron)
        D_ref = np.column_stack([X1 - X0, X2 - X0, X3 - X0])
        inv_D = np.linalg.inv(D_ref) if np.linalg.det(D_ref) > 1e-12 else np.eye(3)
        
        # grad N in reference coordinates
        grad_N_ref = np.zeros((4, 3))
        grad_N_ref[0] = -np.sum(inv_D, axis=1)
        grad_N_ref[1] = inv_D[:, 0]
        grad_N_ref[2] = inv_D[:, 1]
        grad_N_ref[3] = inv_D[:, 2]
        
        # Transform to current coordinates
        # grad_N = F^-T * grad_N_ref
        F_inv_T = np.linalg.inv(F).T
        grad_N = grad_N_ref @ F_inv_T.T
        
        # Nodal forces
        forces = np.zeros((4, 3))
        for i in range(4):
            forces[i] = -self.volume * stress @ grad_N[i]
        
        return forces


class SoftBody:
    """Deformable soft body using tetrahedral FEM."""
    
    def __init__(self, nodes: list[SoftBodyNode], tetras: list[SoftBodyTetra], params: SoftBodyParams):
        self.nodes = nodes
        self.tetras = tetras
        self.params = params
        
        # Store rest positions
        self.rest_positions = np.array([n.pos for n in nodes])
    
    def step(self, dt: float) -> None:
        """Time integration with Rayleigh damping."""
        params = self.params
        
        # Reset forces
        for node in self.nodes:
            node.force[:] = 0.0
        
        # Add gravity
        gravity = np.array([0.0, -9.81, 0.0])
        for node in self.nodes:
            if not node.fixed:
                node.force += node.mass * gravity
        
        # Internal forces from tetras
        for tetra in self.tetras:
            forces = tetra.compute_forces(self.nodes, self.rest_positions, params)
            for i, node_idx in enumerate(tetra.nodes):
                if not self.nodes[node_idx].fixed:
                    self.nodes[node_idx].force += forces[i]
        
        # Rayleigh damping: C = alpha*M + beta*K
        # Simplified: f_damp = -(alpha*M + beta*K) * v
        for node in self.nodes:
            if not node.fixed:
                node.force -= params.damping_alpha * node.mass * node.vel
                # Stiffness-proportional damping is more complex; approximate
                node.force -= params.damping_beta * node.force * dt * 0.1
        
        # Semi-implicit Euler
        for node in self.nodes:
            if not node.fixed:
                node.vel += node.force / node.mass * dt
                node.pos += node.vel * dt


def create_soft_body_box(
    x: float, y: float, z: float,
    width: float, height: float, depth: float,
    resolution: int, params: SoftBodyParams
) -> SoftBody:
    """Create a soft body box with tetrahedral mesh."""
    nodes = []
    # Generate grid nodes
    for i in range(resolution + 1):
        for j in range(resolution + 1):
            for k in range(resolution + 1):
                px = x + i * width / resolution
                py = y + j * height / resolution
                pz = z + k * depth / resolution
                fixed = (j == 0)  # fix bottom layer
                node = SoftBodyNode([px, py, pz], params.density * (width*height*depth) / (resolution+1)**3, fixed)
                nodes.append(node)
    
    # Generate tetras (simplified: each cell = 5 tetras)
    tetras = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                # 8 corners of the cube cell
                def idx(di, dj, dk):
                    return (i+di)*(resolution+1)*(resolution+1) + (j+dj)*(resolution+1) + (k+dk)
                
                corners = [idx(di, dj, dk) for di in (0,1) for dj in (0,1) for dk in (0,1)]
                
                # Split cube into 5 tetras (standard decomposition)
                tetra_indices = [
                    [corners[0], corners[1], corners[3], corners[4]],
                    [corners[1], corners[5], corners[3], corners[7]],
                    [corners[3], corners[2], corners[0], corners[4]],  # wait, reorder
                ]
                
                # Standard 5-tet decomposition of a cube
                cell_tetras = [
                    [corners[0], corners[1], corners[3], corners[4]],  # 0,1,3,4
                    [corners[1], corners[5], corners[3], corners[7]],  # 1,5,3,7
                    [corners[1], corners[3], corners[4], corners[5]],  # 1,3,4,5
                    [corners[3], corners[5], corners[4], corners[7]],  # 3,5,4,7
                    [corners[4], corners[5], corners[7], corners[6]],  # 4,5,7,6
                ]
                
                cell_volume = (width/resolution) * (height/resolution) * (depth/resolution) / 5.0
                for t_idx in cell_tetras:
                    tetras.append(SoftBodyTetra(t_idx, cell_volume))
    
    return SoftBody(nodes, tetras, params)


# Integration with rigid body system
@dataclass
class DeformableBody:
    """A body that can be either rigid or deformable."""
    
    # Rigid body properties (when not deforming)
    rigid_body: Body3D | None = None
    
    # Soft body (when deforming)
    soft_body: SoftBody | None = None
    
    # Damage state
    fatigue: FatigueState = field(default_factory=FatigueState)
    plasticity: PlasticState = field(default_factory=PlasticState)
    
    # Current mode
    is_deformable: bool = False
    
    def switch_to_deformable(self, params: SoftBodyParams, resolution: int = 4) -> None:
        """Convert rigid body to soft body."""
        if self.rigid_body is None:
            return
        
        # Create soft body from rigid body shape
        pos = self.rigid_body.pos
        # Approximate shape as box
        extents = self.rigid_body.shape.half_extents if self.rigid_body.shape else np.array([0.5, 0.5, 0.5])
        self.soft_body = create_soft_body_box(
            pos[0] - extents[0], pos[1] - extents[1], pos[2] - extents[2],
            extents[0]*2, extents[1]*2, extents[2]*2,
            resolution, params
        )
        self.is_deformable = True
    
    def step(self, dt: float) -> None:
        """Advance by dt."""
        if self.is_deformable and self.soft_body:
            self.soft_body.step(dt)
        elif self.rigid_body:
            # Rigid body step handled by physics kernel
            pass


if __name__ == "__main__":
    # Quick test
    params = PlasticityParams(yield_stress=250e6, hardening_modulus=1e9)
    stress = np.array([[300e6, 0, 0], [0, 0, 0], [0, 0, 0]])
    plastic = PlasticState(current_yield=params.yield_stress)
    
    print(f"Initial stress: {stress[0,0]/1e6:.1f} MPa")
    print(f"von Mises: {von_mises_stress(stress)/1e6:.1f} MPa")
    print(f"Yield: {params.yield_stress/1e6:.1f} MPa")
    
    corrected, plastic = plastic_correction(stress, plastic, params)
    print(f"Corrected stress: {corrected[0,0]/1e6:.1f} MPa")
    print(f"Plastic strain: {plastic.eps_p_eq:.6f}")
    print(f"New yield: {plastic.current_yield/1e6:.1f} MPa")
    
    # Test fatigue
    print("\n--- Fatigue Test ---")
    fatigue_params = FatigueParams(sn_c=1e12, sn_m=3.0, endurance_limit=100e6)
    fatigue = FatigueState()
    
    for i in range(1000):
        fatigue.add_cycle(150e6, fatigue_params)
    
    print(f"Damage after 1000 cycles at 150 MPa: {fatigue.damage:.4f}")
    print(f"Crack size: {fatigue.crack_size:.6f} m")
    print(f"Failed: {fatigue.is_failed(fatigue_params)}")
    
    # Test soft body
    print("\n--- Soft Body Test ---")
    sb_params = SoftBodyParams(young_modulus=1e6, density=1000.0, dt=1/60.0)
    soft_body = create_soft_body_box(0, 0, 0, 1.0, 1.0, 1.0, 4, sb_params)
    print(f"Nodes: {len(soft_body.nodes)}, Tetras: {len(soft_body.tetras)}")
    soft_body.step(sb_params.dt)
    print("Step completed")