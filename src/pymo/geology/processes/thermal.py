"""Heat conduction PDE solver for geothermal evolution.

Solves: ρc ∂T/∂t = ∇·(k∇T) + H
where H = radiogenic heat production (W/m³)

Uses implicit (backward Euler) time stepping for unconditional stability.
3D 7-point stencil on structured grid.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from ..rock_materials import RockMaterial, get_rock_by_id, rock_count, get_thermal_conductivity_array, get_melting_point_array


@dataclass(slots=True)
class ThermalConfig:
    """Configuration for thermal solver."""
    dt: float                 # time step (years)
    surface_temp: float = 293.15     # K, Dirichlet BC at top
    mantle_heat_flux: float = 0.065  # W/m², Neumann BC at bottom
    radiogenic_heat: float = 1.0e-6  # W/m³ average crustal production
    max_iter: int = 1           # for nonlinear iteration if k=k(T)
    tolerance: float = 1e-4     # relative residual tolerance


def build_thermal_matrices(
    grid,
    rock_k: np.ndarray,         # (N,) thermal conductivity per cell
    rho_c: np.ndarray,          # (N,) ρc per cell
    dt: float,
    cell_size: float,
    radiogenic_heat: float = 1.0e-6,
) -> tuple[csr_matrix, np.ndarray]:
    """Build implicit system matrix A and RHS vector b for ∂T/∂t = ∇·(k∇T)/(ρc) + H/(ρc).

    A * T_new = b, where b = T_old + dt * (H/(ρc)) + boundary terms

    Returns (A, b_base) where b_base includes old temperature and source terms.
    Boundary conditions applied separately in solve step.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    n = nx * ny * nz
    h = cell_size
    h2 = h * h

    # Precompute coefficients
    # For each cell, k_face = harmonic mean of adjacent cell conductivities
    # Using simple arithmetic mean for now (adequate for moderate contrasts)
    alpha = dt / (rho_c * h2)   # (N,) factor for diffusion

    # Diagonal and off-diagonal entries
    # A = I - dt * L where L is discrete Laplacian with variable k
    # L[i,i] = sum(k_face/h²) over 6 neighbors
    # L[i,j] = -k_face/h² for neighbor j

    main_diag = np.ones(n, dtype=np.float32)
    off_diag_x = np.zeros(n, dtype=np.float32)  # i+1 neighbor
    off_diag_y = np.zeros(n, dtype=np.float32)  # i+nx neighbor
    off_diag_z = np.zeros(n, dtype=np.float32)  # i+nx*ny neighbor

    # Compute face conductivities (harmonic mean between adjacent cells)
    # X-faces
    k_x = np.zeros(n - nx * ny * (nz - 1), dtype=np.float32)  # all x-faces
    # Actually simpler: just compute on the fly per direction

    # We'll build using scipy.sparse.diags for structured grid
    # For variable coefficients, need to be careful with boundaries

    # X-direction connections
    k_xp = 0.5 * (rock_k[:-1] + rock_k[1:])  # between i and i+1 (where contiguous)
    # But need to handle nx boundaries...
    # Let's use a more direct approach with explicit indexing

    rows = []
    cols = []
    data = []

    def add_entry(i, j, val):
        rows.append(i)
        cols.append(j)
        data.append(val)

    # Loop over all cells
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                i = ix + nx * (iy + ny * iz)
                k_center = rock_k[i]
                rho_c_i = rho_c[i]

                diag_val = 1.0  # Identity
                source = dt * radiogenic_heat / rho_c_i  # will be added to b

                # X neighbors
                if ix > 0:
                    j = i - 1
                    k_face = 2.0 * k_center * rock_k[j] / (k_center + rock_k[j] + 1e-12)
                    coeff = alpha[i] * k_face
                    diag_val += coeff
                    add_entry(i, j, -coeff)
                if ix < nx - 1:
                    j = i + 1
                    k_face = 2.0 * k_center * rock_k[j] / (k_center + rock_k[j] + 1e-12)
                    coeff = alpha[i] * k_face
                    diag_val += coeff
                    add_entry(i, j, -coeff)

                # Y neighbors
                if iy > 0:
                    j = i - nx
                    k_face = 2.0 * k_center * rock_k[j] / (k_center + rock_k[j] + 1e-12)
                    coeff = alpha[i] * k_face
                    diag_val += coeff
                    add_entry(i, j, -coeff)
                if iy < ny - 1:
                    j = i + nx
                    k_face = 2.0 * k_center * rock_k[j] / (k_center + rock_k[j] + 1e-12)
                    coeff = alpha[i] * k_face
                    diag_val += coeff
                    add_entry(i, j, -coeff)

                # Z neighbors
                if iz > 0:
                    j = i - nx * ny
                    k_face = 2.0 * k_center * rock_k[j] / (k_center + rock_k[j] + 1e-12)
                    coeff = alpha[i] * k_face
                    diag_val += coeff
                    add_entry(i, j, -coeff)
                if iz < nz - 1:
                    j = i + nx * ny
                    k_face = 2.0 * k_center * rock_k[j] / (k_center + rock_k[j] + 1e-12)
                    coeff = alpha[i] * k_face
                    diag_val += coeff
                    add_entry(i, j, -coeff)

                add_entry(i, i, diag_val)

    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

    # Base RHS = T_old + dt * H/(ρc)
    b_base = np.zeros(n, dtype=np.float32)
    # Will be updated in solve() with T_old and boundary conditions

    return A, b_base


def solve_thermal_step(
    grid,
    config: ThermalConfig,
    rock_materials: dict[int, RockMaterial],
) -> None:
    """Advance temperature field by one time step.

    Modifies grid.temperature in place.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    n = nx * ny * nz

    # Get material properties per cell
    rock_ids = grid.rock_id
    rock_k = np.array([rock_materials[rid].thermal_conductivity for rid in rock_ids], dtype=np.float32)
    rho_c = np.array([rock_materials[rid].density * rock_materials[rid].specific_heat for rid in rock_ids], dtype=np.float32)

    # Build system matrix (can be cached if dt, grid, materials unchanged)
    A, _ = build_thermal_matrices(grid, rock_k, rho_c, config.dt, grid.config.cell_size, config.radiogenic_heat)

    # RHS: T_old + dt * H/(ρc)
    b = grid.temperature.copy()
    b += config.dt * config.radiogenic_heat / rho_c

    # Apply boundary conditions
    # Top surface (z = nz-1): Dirichlet T = surface_temp
    for iy in range(ny):
        for ix in range(nx):
            i = ix + nx * (iy + ny * (nz - 1))
            A[i, :] = 0.0
            A[i, i] = 1.0
            b[i] = config.surface_temp

    # Bottom (z = 0): Neumann ∂T/∂z = mantle_heat_flux / k
    # -k ∂T/∂z = q  =>  (T_ghost - T_0)/h = -q/k  =>  T_ghost = T_0 - h*q/k
    # Discretized: (T_1 - T_0)/h ≈ ∂T/∂z = -q/k  =>  -k/h T_0 + k/h T_1 = -q
    for iy in range(ny):
        for ix in range(nx):
            i0 = ix + nx * (iy + ny * 0)
            i1 = ix + nx * (iy + ny * 1)
            k0 = rock_k[i0]
            coeff = k0 / grid.config.cell_size
            A[i0, i0] += coeff
            A[i0, i1] -= coeff
            b[i0] += config.mantle_heat_flux

    # Solve
    T_new = spsolve(A, b)
    grid.temperature[:] = T_new.astype(np.float32)

    # Clamp to melting points (prevent unphysical superheating without phase change)
    melting_points = np.array([rock_materials[rid].melting_point for rid in rock_ids], dtype=np.float32)
    np.maximum(grid.temperature, melting_points, out=grid.temperature)


# ============================================================
# Steady-state solver (for initialization)
# ============================================================

def solve_steady_state(
    grid,
    surface_temp: float = 293.15,
    mantle_heat_flux: float = 0.065,
    rock_materials: dict[int, RockMaterial] | None = None,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
) -> None:
    """Solve steady-state heat equation: ∇·(k∇T) + H = 0.

    Uses Gauss-Seidel iteration (smoother for variable k).
    """
    if rock_materials is None:
        rock_materials = {r.rock_id: r for r in all_rocks()}

    nx, ny, nz = grid.nx, grid.ny, grid.nz
    h = grid.config.cell_size

    # Precompute face conductivities
    rock_ids = grid.rock_id
    k = np.array([rock_materials[rid].thermal_conductivity for rid in rock_ids], dtype=np.float32)

    # Harmonic mean face conductivities
    kx = np.zeros((nz, ny, nx - 1), dtype=np.float32)
    ky = np.zeros((nz, ny - 1, nx), dtype=np.float32)
    kz = np.zeros((nz - 1, ny, nx), dtype=np.float32)

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx - 1):
                i = ix + nx * (iy + ny * iz)
                j = i + 1
                kx[iz, iy, ix] = 2.0 * k[i] * k[j] / (k[i] + k[j] + 1e-12)

    for iz in range(nz):
        for iy in range(ny - 1):
            for ix in range(nx):
                i = ix + nx * (iy + ny * iz)
                j = i + nx
                ky[iz, iy, ix] = 2.0 * k[i] * k[j] / (k[i] + k[j] + 1e-12)

    for iz in range(nz - 1):
        for iy in range(ny):
            for ix in range(nx):
                i = ix + nx * (iy + ny * iz)
                j = i + nx * ny
                kz[iz, iy, ix] = 2.0 * k[i] * k[j] / (k[i] + k[j] + 1e-12)

    # Radiogenic heat per cell
    H = np.array([rock_materials[rid].density * rock_materials[rid].specific_heat
                  for rid in rock_ids], dtype=np.float32)
    H *= 1.0e-6  # W/m³ -> W per cell volume
    H = H.reshape(nz, ny, nx)

    T = grid.temperature.reshape(nz, ny, nx).copy()

    for iteration in range(max_iter):
        T_old = T.copy()

        # Interior cells
        for iz in range(1, nz - 1):
            for iy in range(1, ny - 1):
                for ix in range(1, nx - 1):
                    # Standard 7-point stencil with variable k
                    coeff_xp = kx[iz, iy, ix]
                    coeff_xm = kx[iz, iy, ix - 1]
                    coeff_yp = ky[iz, iy, ix]
                    coeff_ym = ky[iz, iy - 1, ix]
                    coeff_zp = kz[iz, iy, ix]
                    coeff_zm = kz[iz - 1, iy, ix]

                    denom = (coeff_xp + coeff_xm + coeff_yp + coeff_ym + coeff_zp + coeff_zm) / h**2

                    num = (coeff_xp * T[iz, iy, ix + 1] + coeff_xm * T[iz, iy, ix - 1] +
                           coeff_yp * T[iz, iy + 1, ix] + coeff_ym * T[iz, iy - 1, ix] +
                           coeff_zp * T[iz + 1, iy, ix] + coeff_zm * T[iz - 1, iy, ix]) / h**2

                    num += H[iz, iy, ix]  # source term

                    T[iz, iy, ix] = num / denom

        # Top boundary (Dirichlet)
        T[nz - 1, :, :] = surface_temp

        # Bottom boundary (Neumann)
        for iy in range(ny):
            for ix in range(nx):
                iz = 0
                i = ix + nx * (iy + ny * iz)
                j = ix + nx * (iy + ny * 1)
                k0 = k[i]
                # -k (T1 - T0)/h = q  =>  T0 = T1 + h*q/k
                T[0, iy, ix] = T[1, iy, ix] + h * mantle_heat_flux / (k0 + 1e-12)

        # Side boundaries: insulated (no flux)
        # Implemented by not updating boundary cells, or copying interior
        T[:, 0, :] = T[:, 1, :]
        T[:, -1, :] = T[:, -2, :]
        T[:, :, 0] = T[:, :, 1]
        T[:, :, -1] = T[:, :, -2]

        # Check convergence
        diff = np.abs(T - T_old).max()
        if diff < tolerance:
            print(f"Steady-state converged in {iteration+1} iterations, max diff={diff:.2e} K")
            break
    else:
        print(f"Steady-state did not converge after {max_iter} iterations, max diff={diff:.2e} K")

    grid.temperature[:] = T.ravel()


# ============================================================
# Utility: create rock_materials dict from library
# ============================================================

from ..rock_materials import all_rocks, get_rock_by_id

def get_material_dict() -> dict[int, RockMaterial]:
    return {r.rock_id: r for r in all_rocks()}


if __name__ == "__main__":
    # Test steady-state
    from ..geology_grid import GeologyGridConfig, create_stratified_grid
    from ..rock_materials import get_rock

    config = GeologyGridConfig(nx=16, ny=16, nz=16, cell_size=100.0)
    layers = [(500.0, get_rock("sandstone").rock_id),
              (500.0, get_rock("shale").rock_id),
              (600.0, get_rock("granite").rock_id)]
    grid = create_stratified_grid(config, layers, surface_temp=293.15)

    print("Solving steady-state...")
    solve_steady_state(grid)

    # Print temperature profile at center
    center_x, center_y = 8, 8
    for iz in range(config.nz):
        i = center_x + config.nx * (center_y + config.ny * iz)
        z = iz * config.cell_size
        print(f"  z={z/1000:.1f} km: T={grid.temperature[i]-273.15:.1f}°C, rock={grid.rock_id[i]}")