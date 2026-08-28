"""3D math utilities for the pymo physics kernel."""

from __future__ import annotations

import numpy as np
from numpy.linalg import det, inv


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3D cross product."""
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])


def dot3(a: np.ndarray, b: np.ndarray) -> float:
    """3D dot product."""
    return float(np.dot(a, b))


def quat_identity() -> np.ndarray:
    """Identity quaternion (w, x, y, z)."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (conjugate for unit quaternions)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create quaternion from axis-angle representation."""
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return quat_identity()
    axis = axis / axis_norm
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    return np.array([
        np.cos(half_angle),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s
    ])


def quat_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert quaternion to axis-angle representation."""
    q = q / np.linalg.norm(q)
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(1.0 - q[0]**2)
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = q[1:] / s
    return axis, angle


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    # v' = q * v * q^-1 (where v is treated as pure quaternion)
    qv = np.array([0.0, v[0], v[1], v[2]])
    q_conj = quat_conj(q)
    result = quat_mul(quat_mul(q, qv), q_conj)
    return result[1:]


def mat3_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def quat_from_mat3(m: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion."""
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from 3D vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q1 + t * (q2 - q1))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    return quat_normalize(w1 * q1 + w2 * q2)


def quat_log(q: np.ndarray) -> np.ndarray:
    """Quaternion logarithm (returns axis * angle/2)."""
    q = q / np.linalg.norm(q)
    if q[0] >= 1.0:
        return np.zeros(3)
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(1.0 - q[0]**2)
    if s < 1e-12:
        return np.zeros(3)
    return q[1:] / s * (angle * 0.5)


def quat_exp(v: np.ndarray) -> np.ndarray:
    """Quaternion exponential (inverse of log)."""
    angle = 2.0 * np.linalg.norm(v)
    if angle < 1e-12:
        return quat_identity()
    axis = v / (angle * 0.5)
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    return np.array([np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s])


# Interpolation and integration helpers

def integrate_angular_velocity(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate angular velocity to update orientation quaternion.
    
    q_new = q * exp(0.5 * omega * dt)
    """
    if np.linalg.norm(omega) < 1e-12:
        return q
    dq = quat_exp(omega * dt * 0.5)
    return quat_mul(q, dq)


def angular_velocity_to_quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Compute quaternion derivative from angular velocity.
    
    dq/dt = 0.5 * q * omega_quat
    """
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    return quat_mul(q, omega_quat) * 0.5


# Vector utilities

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def project(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Project v onto onto."""
    return onto * np.dot(v, onto) / np.dot(onto, onto)


def reject(v: np.ndarray, from_vec: np.ndarray) -> np.ndarray:
    """Reject v from from_vec (perpendicular component)."""
    return v - project(v, from_vec)


def clamp(v: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clamp vector components."""
    return np.clip(v, min_val, max_val)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation."""
    return a + t * (b - a)


# Matrix utilities

def mat3_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Create matrix from three column vectors."""
    return np.column_stack([x, y, z])


def mat3_transpose(m: np.ndarray) -> np.ndarray:
    return m.T


def mat3_det(m: np.ndarray) -> float:
    return det(m)


def mat3_inv(m: np.ndarray) -> np.ndarray:
    return inv(m)


def mat3_trace(m: np.ndarray) -> float:
    return np.trace(m)


def mat3_diag(x: float, y: float, z: float) -> np.ndarray:
    return np.diag([x, y, z])


def mat3_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.outer(a, b)


# Intersection tests

def ray_sphere_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray, 
                         sphere_center: np.ndarray, radius: float) -> tuple[bool, float]:
    """Ray-sphere intersection. Returns (hit, distance)."""
    oc = ray_origin - sphere_center
    a = dot3(ray_dir, ray_dir)
    b = 2.0 * dot3(oc, ray_dir)
    c = dot3(oc, oc) - radius**2
    disc = b*b - 4*a*c
    if disc < 0:
        return False, 0.0
    t = (-b - np.sqrt(disc)) / (2*a)
    if t < 0:
        t = (-b + np.sqrt(disc)) / (2*a)
    if t < 0:
        return False, 0.0
    return True, t


def ray_aabb_intersect(ray_origin: np.ndarray, ray_dir: np.ndarray,
                       aabb_min: np.ndarray, aabb_max: np.ndarray) -> tuple[bool, float, float]:
    """Ray-AABB intersection. Returns (hit, tmin, tmax)."""
    t1 = (aabb_min - ray_origin) / ray_dir
    t2 = (aabb_max - ray_origin) / ray_dir
    tmin = np.maximum(np.minimum(t1, t2))
    tmax = np.minimum(np.maximum(t1, t2))
    t_enter = np.max(tmin)
    t_exit = np.min(tmax)
    if t_enter > t_exit or t_exit < 0:
        return False, 0.0, 0.0
    return True, max(t_enter, 0.0), t_exit


def point_in_aabb(point: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> bool:
    return np.all(point >= aabb_min) and np.all(point <= aabb_max)


def triangle_area(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(cross3(v1 - v0, v2 - v0))


def barycentric_coords(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates of point p in triangle abc."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.array([1/3, 1/3, 1/3])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])