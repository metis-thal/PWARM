"""Numba-accelerated contact velocity solver with pure-Python fallback.

Sequential-impulse (projected Gauss-Seidel) contact resolution over flat
arrays. The identical implementation is JIT-compiled when numba is available
and used as plain Python otherwise, so both environments execute the same
math in the same order (bit-for-bit parity per environment).

Mirrors the impulse math of RigidSolver._resolve_contacts:
- Normal impulse: j = -(1 + e) * vn / (inv_mass_a + inv_mass_b), j >= 0
- Coulomb friction: |jt| <= mu * j, opposing the slip of B relative to A
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised on py3.14 env
    HAVE_NUMBA = False


def _make_solve_contacts_velocity():
    def solve_contacts_velocity(linvel: np.ndarray,
                                inv_mass: np.ndarray,
                                idx_a: np.ndarray,
                                idx_b: np.ndarray,
                                normals: np.ndarray,
                                fric: np.ndarray,
                                rest: np.ndarray,
                                iterations: int) -> np.ndarray:
        """Apply iterative contact impulses to linvel (modified in place)."""
        m = len(idx_a)
        for _ in range(iterations):
            for c in range(m):
                ia = idx_a[c]
                ib = idx_b[c]
                ima = inv_mass[ia]
                imb = inv_mass[ib]
                isum = ima + imb
                if isum == 0.0:
                    continue

                rvx = linvel[ib, 0] - linvel[ia, 0]
                rvy = linvel[ib, 1] - linvel[ia, 1]
                rvz = linvel[ib, 2] - linvel[ia, 2]

                nx = normals[c, 0]
                ny = normals[c, 1]
                nz = normals[c, 2]
                vn = rvx * nx + rvy * ny + rvz * nz
                if vn > 0.0:
                    continue  # separating

                j = -(1.0 + rest[c]) * vn / isum
                if j < 0.0:
                    j = 0.0

                # Normal impulse
                linvel[ia, 0] -= nx * j * ima
                linvel[ia, 1] -= ny * j * ima
                linvel[ia, 2] -= nz * j * ima
                linvel[ib, 0] += nx * j * imb
                linvel[ib, 1] += ny * j * imb
                linvel[ib, 2] += nz * j * imb

                # Coulomb friction: tangential impulse clamped to mu * j_n
                if fric[c] > 0.0 and j > 0.0:
                    tx = rvx - vn * nx
                    ty = rvy - vn * ny
                    tz = rvz - vn * nz
                    tlen = np.sqrt(tx * tx + ty * ty + tz * tz)
                    if tlen > 1e-6:
                        jt = tlen / isum
                        jt_max = fric[c] * j
                        if jt > jt_max:
                            jt = jt_max
                        tx /= tlen
                        ty /= tlen
                        tz /= tlen
                        # fric_impulse = -t_hat * jt:
                        # v_a -= fric_impulse * ima  ->  v_a += t_hat * jt * ima
                        # v_b += fric_impulse * imb  ->  v_b -= t_hat * jt * imb
                        linvel[ia, 0] += tx * jt * ima
                        linvel[ia, 1] += ty * jt * ima
                        linvel[ia, 2] += tz * jt * ima
                        linvel[ib, 0] -= tx * jt * imb
                        linvel[ib, 1] -= ty * jt * imb
                        linvel[ib, 2] -= tz * jt * imb
        return linvel

    return solve_contacts_velocity


solve_contacts_velocity_py = _make_solve_contacts_velocity()

if HAVE_NUMBA:
    solve_contacts_velocity = njit(cache=True, fastmath=True)(solve_contacts_velocity_py)
else:
    solve_contacts_velocity = solve_contacts_velocity_py
