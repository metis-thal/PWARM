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
                    # Cubic-spline pieces split at q = 0.5, i.e. q2 = 0.25.
                    # Splitting at q2 = 0.5 (q = 0.707) made W discontinuous
                    # by ~2.4x exactly at r = h/sqrt(2) — the face-diagonal
                    # neighbor distance of a standard lattice — so float32
                    # jitter flipped branches and spiked the density.
                    if q2 < 0.25:
                        w = norm * (1.0 - 6.0 * q2 + 6.0 * q2 * np.sqrt(q2))
                    else:
                        t = 1.0 - np.sqrt(q2)
                        w = norm * 2.0 * t * t * t
                    # Coincidence guard: fade the pair contribution to zero
                    # as r -> 0 (full weight only beyond r = 0.1*h). An exact
                    # overlap — a tunneled particle clamped onto a pinned one
                    # — would otherwise double-count the self-weight (density
                    # 2x) and blast the neighborhood through the Tait EOS.
                    # Legitimate lattice distances (r >= 0.25*h) unaffected.
                    fade = r2 / (0.01 * h2)
                    if fade < 1.0:
                        w *= fade
                    acc += mass[j] * w
            density[i] = acc
        return density

    return sph_density


def _make_sph_forces():
    def sph_forces(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray,
                   density: np.ndarray, pressure: np.ndarray,
                   nbr_start: np.ndarray, nbr_flat: np.ndarray,
                   h: float, viscosity: float,
                   surface_tension: float,
                   contact_rc: float, contact_acc: float) -> np.ndarray:
        """Pair forces: pressure + artificial viscosity + near-contact
        repulsion + surface tension in a single neighbor pass (gravity and
        coupling forces are added by the caller)."""
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

                # Gradient of cubic spline kernel (world direction r = pj - pi).
                # W(q) = norm*(1-6q^2+6q^3) for q<=0.5, 2*norm*(1-q)^3 for q<1,
                # with norm = 8/(pi*h^3). Correct derivative:
                #   dW/dr = grad_norm*(3q^2-2q)/h      (q < 0.5)
                #   dW/dr = -grad_norm*(1-q)^2/h       (0.5 <= q < 1)
                # Both vanish linearly as q->0 (smooth kernel); the previous
                # formulas (6q-6*sqrt(q), -2(1-sqrt(q))/sqrt(q)) diverged at
                # q->0 and blew up near-coincident particles.
                q = r_norm / h
                if q < 0.5:
                    factor = grad_norm * (3.0 * q * q - 2.0 * q) / h
                else:
                    t = 1.0 - q
                    factor = -grad_norm * t * t / h
                inv_r = factor / r_norm
                gx = rx * inv_r
                gy = ry * inv_r
                gz = rz * inv_r

                # Pressure force (Monaghan): F_i = -m_j * (p_i/rho_i^2 +
                # p_j/rho_j^2) * grad_i W(r_i - r_j). This kernel's grad_W is
                # dW/dr * r_hat(i->j); since dW/dr < 0, grad_W points j -> i,
                # i.e. grad_W = -grad_i W. The repulsive force on i is
                # therefore +m_j * p_term * grad_W (the previous '-' made
                # pressure attractive and collapsed the fluid instantly).
                p_over_j = pressure[j] / (density[j] * density[j] + 1e-6)
                p_term = p_over_i + p_over_j
                fx += mass[j] * p_term * gx
                fy += mass[j] * p_term * gy
                fz += mass[j] * p_term * gz

                # Artificial viscosity (Monaghan 1992), approaching pairs only:
                #   mu_ij = h * (v_i - v_j) . (r_i - r_j) / (|r_ij|^2 + 0.01 h^2)
                # is NEGATIVE for approaching pairs, making
                #   Pi_ij = (-alpha*mu + beta*mu^2) / rho_bar  > 0,
                # and F_i = +m_j * Pi_ij * grad_W pushes the pair apart
                # (dissipative damping; grad_W points j -> i). mu vanishes
                # linearly as r -> 0 (self-regularizing). The previous code
                # dotted with (r_j - r_i), which INVERTED the activation
                # criterion — viscosity acted on separating pairs and
                # accelerated them (energy injection on every bounce) while
                # approaching pairs got no damping at all.
                vijx = vxi - vel[j, 0]
                vijy = vyi - vel[j, 1]
                vijz = vzi - vel[j, 2]
                v_dot_r = -(vijx * rx + vijy * ry + vijz * rz)
                if v_dot_r < 0.0:
                    mu = h * v_dot_r / (r2 + 0.01 * h2)
                    pi_ij = (-viscosity * mu + 0.1 * mu * mu) / \
                            ((density[i] + density[j]) * 0.5)
                    # Same sign relation as pressure: +pi_ij * grad_W pushes
                    # the approaching pair apart (dissipative damping).
                    fx += mass[j] * pi_ij * gx
                    fy += mass[j] * pi_ij * gy
                    fz += mass[j] * pi_ij * gz

                # Near-contact repulsion: the EOS is clamped at p = 0 for
                # rho < rest density, so particles in sparse neighborhoods
                # (free surface, piles at the wall) feel no pressure and can
                # interpenetrate into degenerate near-coincident stacks. A
                # linear contact force inside contact_rc pushes overlapping
                # pairs apart (reduced-mass weighted, momentum-conserving),
                # mirroring the rigid contact solver.
                if r_norm < contact_rc:
                    red_mass = (mass[i] * mass[j]) / (mass[i] + mass[j])
                    fc = red_mass * contact_acc * (1.0 - r_norm / contact_rc)
                    inv_rn = 1.0 / r_norm
                    fx -= fc * rx * inv_rn
                    fy -= fc * ry * inv_rn
                    fz -= fc * rz * inv_rn

                # Surface tension (simplified central cohesion term): pulls i
                # toward j, i.e. along -grad_W. NOTE: formulation is still
                # unphysical (no density normalization); default is 0.0.
                if surface_tension > 0.0:
                    st = surface_tension * mass[j]
                    fx -= st * gx
                    fy -= st * gy
                    fz -= st * gz

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
