"""Numba-accelerated SPH kernels with pure-Python fallback.

The same implementation functions are wrapped with @njit when numba is
available and used as plain Python otherwise, so both environments execute
identical math (float64 scalar ops, matching the original per-op promoted
semantics of the float32 numpy code).

Neighbor lists are passed in CSR form (starts + flat indices) because numba
cannot handle ragged list-of-list structures efficiently.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised on py3.14 env
    HAVE_NUMBA = False


def neighbors_to_csr(neighbor_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert list-of-list neighbor structure to (starts, flat) CSR arrays."""
    n = len(neighbor_lists)
    starts = np.zeros(n + 1, dtype=np.int64)
    total = 0
    for i, lst in enumerate(neighbor_lists):
        total += len(lst)
        starts[i + 1] = total
    flat = np.empty(total, dtype=np.int64)
    k = 0
    for lst in neighbor_lists:
        for j in lst:
            flat[k] = j
            k += 1
    return starts, flat


def _make_sph_density():
    def sph_density(pos: np.ndarray, mass: np.ndarray,
                    nbr_start: np.ndarray, nbr_flat: np.ndarray,
                    h: float) -> np.ndarray:
        """SPH density from cubic spline kernel, self-contribution included."""
        n = pos.shape[0]
        density = np.zeros(n, dtype=np.float64)
        h2 = h * h
        norm = 8.0 / (np.pi * h ** 3)
        self_w = norm  # kernel(0, h): q2=0 -> norm * 1.0
        for i in range(n):
            acc = mass[i] * self_w
            xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
            for k in range(nbr_start[i], nbr_start[i + 1]):
                j = nbr_flat[k]
                if j == i:
                    continue
                rx = pos[j, 0] - xi
                ry = pos[j, 1] - yi
                rz = pos[j, 2] - zi
                r2 = rx * rx + ry * ry + rz * rz
                if r2 < h2:
                    q2 = r2 / h2
                    if q2 < 0.5:
                        w = norm * (1.0 - 6.0 * q2 + 6.0 * q2 * np.sqrt(q2))
                    else:
                        t = 1.0 - np.sqrt(q2)
                        w = norm * 2.0 * t * t * t
                    acc += mass[j] * w
            density[i] = acc
        return density

    return sph_density


def _make_sph_forces():
    def sph_forces(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray,
                   density: np.ndarray, pressure: np.ndarray,
                   nbr_start: np.ndarray, nbr_flat: np.ndarray,
                   h: float, viscosity: float,
                   surface_tension: float) -> np.ndarray:
        """Pair forces: pressure + artificial viscosity + surface tension in a
        single neighbor pass (gravity and coupling forces are added by the
        caller)."""
        n = pos.shape[0]
        forces = np.zeros((n, 3), dtype=np.float64)
        h2 = h * h
        grad_norm = 48.0 / (np.pi * h ** 3)
        for i in range(n):
            fx = 0.0
            fy = 0.0
            fz = 0.0
            xi, yi, zi = pos[i, 0], pos[i, 1], pos[i, 2]
            vxi, vyi, vzi = vel[i, 0], vel[i, 1], vel[i, 2]
            rho_i2 = density[i] * density[i] + 1e-6
            p_over_i = pressure[i] / rho_i2
            for k in range(nbr_start[i], nbr_start[i + 1]):
                j = nbr_flat[k]
                if j == i:
                    continue
                rx = pos[j, 0] - xi
                ry = pos[j, 1] - yi
                rz = pos[j, 2] - zi
                r2 = rx * rx + ry * ry + rz * rz
                if r2 >= h2:
                    continue
                r_norm = np.sqrt(r2)
                if r_norm < 1e-6:
                    continue

                # Gradient of cubic spline kernel (world direction r = pj - pi)
                q = r_norm / h
                if q < 0.5:
                    factor = grad_norm * (6.0 * q - 6.0 * np.sqrt(q)) / h
                else:
                    factor = grad_norm * (-2.0 * (1.0 - np.sqrt(q)) / (np.sqrt(q) * h))
                inv_r = factor / r_norm
                gx = rx * inv_r
                gy = ry * inv_r
                gz = rz * inv_r

                # Pressure force: -m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad W
                # grad_W is computed as grad W(r_j - r_i) pointing from i to j
                # For repulsion with positive pressure, force on i is opposite to grad_W
                p_over_j = pressure[j] / (density[j] * density[j] + 1e-6)
                p_term = p_over_i + p_over_j
                fx -= mass[j] * p_term * gx
                fy -= mass[j] * p_term * gy
                fz -= mass[j] * p_term * gz

                # Artificial viscosity (only for approaching pairs)
                vijx = vxi - vel[j, 0]
                vijy = vyi - vel[j, 1]
                vijz = vzi - vel[j, 2]
                r_hat_x = rx / r_norm
                r_hat_y = ry / r_norm
                r_hat_z = rz / r_norm
                v_dot_r = vijx * r_hat_x + vijy * r_hat_y + vijz * r_hat_z
                if v_dot_r < 0.0:
                    mu = h * v_dot_r / (r2 + 0.01 * h2)
                    pi_ij = (-viscosity * mu + 0.1 * mu * mu) / \
                            ((density[i] + density[j]) * 0.5)
                    fx -= mass[j] * pi_ij * gx
                    fy -= mass[j] * pi_ij * gy
                    fz -= mass[j] * pi_ij * gz

                # Surface tension (simplified central term)
                if surface_tension > 0.0:
                    st = surface_tension * mass[j]
                    fx += st * gx
                    fy += st * gy
                    fz += st * gz

            forces[i, 0] = fx
            forces[i, 1] = fy
            forces[i, 2] = fz
        return forces

    return sph_forces


sph_density_py = _make_sph_density()
sph_forces_py = _make_sph_forces()

if HAVE_NUMBA:
    sph_density = njit(cache=True, fastmath=True)(sph_density_py)
    sph_forces = njit(cache=True, fastmath=True)(sph_forces_py)
else:
    sph_density = sph_density_py
    sph_forces = sph_forces_py
