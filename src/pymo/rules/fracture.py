"""Fracture mechanics for the pymo rules layer.

Implements brittle fracture based on stress intensity and fracture toughness.
When stress at a contact exceeds the material's fracture toughness, the body
splits along the plane of maximum principal stress.

Uses a simplified brittle fracture model based on:
- Maximum principal stress criterion (Rankine criterion)
- Fracture toughness K_IC as critical stress intensity
- Crack propagation along maximum principal stress direction

This is a rules-layer extension; the physics kernel remains the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymo.kernel.bodies import Body
from pymo.kernel.collision import Contact


@dataclass
class FractureParams:
    """Parameters controlling fracture behavior."""

    min_fragment_mass: float = 0.01    # kg — fragments smaller than this are removed
    crack_speed: float = 1000.0        # m/s — not used yet, for future dynamic fracture
    max_fragments: int = 4             # max fragments per fracture event


def principal_stresses(stress_xx: float, stress_yy: float, stress_xy: float) -> tuple[float, float]:
    """Compute principal stresses from 2D stress tensor components.

    Returns (sigma_1, sigma_2) where sigma_1 >= sigma_2.
    """
    avg = (stress_xx + stress_yy) / 2.0
    diff = (stress_xx - stress_yy) / 2.0
    r = np.sqrt(diff**2 + stress_xy**2)
    return avg + r, avg - r


def principal_stress_angle(stress_xx: float, stress_yy: float, stress_xy: float) -> float:
    """Angle of maximum principal stress (from x-axis, in radians)."""
    return 0.5 * np.arctan2(2.0 * stress_xy, stress_xx - stress_yy)


def estimate_contact_stress(contact: Contact) -> tuple[float, float, float]:
    """Estimate stress tensor at a contact point from contact impulse.

    Returns (stress_xx, stress_yy, stress_xy) in Pa.

    This is a simplified estimate based on contact impulse and body properties.
    In a full FEM this would be a proper stress tensor field.
    """
    a, b = contact.a, contact.b
    n = contact.normal

    # Contact area proxy (unused but kept for documentation)
    ra = np.linalg.norm(contact.point - a.pos)
    rb = np.linalg.norm(contact.point - b.pos)
    _ = max(np.pi * (ra + rb) * max(contact.penetration, 1e-3), 1e-6)  # area_est

    # Normal stress from penetration (Hertzian contact approximation)
    # Compressive stress is NEGATIVE (convention: tension positive)
    E_eff = 2.0 * a.material.young_modulus * b.material.young_modulus / (
        a.material.young_modulus + b.material.young_modulus
    )
    R_eff = 1.0 / (1.0 / max(ra, 1e-3) + 1.0 / max(rb, 1e-3))
    sigma_n = -E_eff * np.sqrt(max(contact.penetration, 1e-6) / R_eff)  # NEGATIVE = compressive

    # Shear stress from friction
    tau = min(abs(sigma_n) * contact.friction, a.material.hardness)

    # Stress tensor in normal-tangent basis, rotated to world
    c, s = n[0], n[1]
    # Normal direction = n, tangent = (-n_y, n_x)
    _ = np.array([-n[1], n[0]])

    # Stress in normal-tangent coordinates (sigma_n is negative = compressive)
    sigma_nn = sigma_n
    sigma_tt = 0.0  # no tensile stress in tangent (only compressive contact)
    tau_nt = tau

    # Rotate to world coordinates
    c2, s2 = c*c, s*s
    cs = c * s

    stress_xx = sigma_nn * c2 + sigma_tt * s2 + 2 * tau_nt * cs
    stress_yy = sigma_nn * s2 + sigma_tt * c2 - 2 * tau_nt * cs
    stress_xy = (sigma_nn - sigma_tt) * cs + tau_nt * (c2 - s2)

    return stress_xx, stress_yy, stress_xy
    cs = c * s

    stress_xx = sigma_nn * c2 + sigma_tt * s2 + 2 * tau_nt * cs
    stress_yy = sigma_nn * s2 + sigma_tt * c2 - 2 * tau_nt * cs
    stress_xy = (sigma_nn - sigma_tt) * cs + tau_nt * (c2 - s2)

    return stress_xx, stress_yy, stress_xy


def check_fracture(contact: Contact) -> bool:
    """Check if a contact should cause fracture based on stress intensity.

    Returns True if fracture should occur.
    """
    a, b = contact.a, contact.b
    # Skip if either body is static (infinite mass) or already fractured
    if a.static or b.static:
        return False

    # Estimate stress at contact
    sx, sy, sxy = estimate_contact_stress(contact)
    sigma1, _ = principal_stresses(sx, sy, sxy)

    # Rankine criterion: fracture if max principal stress > fracture strength
    # Fracture strength from fracture toughness: sigma_c = K_IC / sqrt(pi * a)
    # where a is crack length (use contact penetration as crack length proxy)
    a_crack = max(contact.penetration, 1e-4)
    for body in (a, b):
        if body.material.fracture_toughness <= 0:
            continue
        K_IC = body.material.fracture_toughness
        sigma_c = K_IC / np.sqrt(np.pi * a_crack)
        # Apply brittleness factor (brittle materials fracture at lower stress)
        sigma_c *= (1.0 - 0.5 * body.material.brittleness)
        if sigma1 > sigma_c:
            return True
    return False


def split_body(body: Body, angle: float, params: FractureParams) -> list[Body]:
    """Split a body into two fragments along a line at given angle.

    Returns list of new bodies (2 fragments).
    """
    # Create two fragments by splitting the polygon along the line
    if body.is_circle():
        # Split circle by cutting through center at given angle
        # Create two half-circles approximated as polygons
        return _split_circle(body, angle)
    else:
        return _split_polygon(body, angle)


def _split_circle(body: Body, angle: float) -> list[Body]:
    """Split a circle into two half-circles."""
    # Create two half-circles (approximated as polygons with flat cut)
    n_segments = 16
    angles1 = np.linspace(angle - np.pi/2, angle + np.pi/2, n_segments + 1)
    angles2 = np.linspace(angle + np.pi/2, angle + 3*np.pi/2, n_segments + 1)

    verts1 = np.column_stack([
        body.radius * np.cos(angles1),
        body.radius * np.sin(angles1)
    ])
    # Add the cut edge (flat face)
    cut_vec = body.radius * np.array([np.cos(angle), np.sin(angle)])
    verts1 = np.vstack([verts1, cut_vec, -cut_vec, verts1[0]])

    verts2 = np.column_stack([
        body.radius * np.cos(angles2),
        body.radius * np.sin(angles2)
    ])
    verts2 = np.vstack([verts2, -cut_vec, cut_vec, verts2[0]])

    m = body.mass / 2.0
    mat = body.material
    b1 = Body(
        pos=body.pos + 0.5 * body.radius * np.array([np.cos(angle), np.sin(angle)]),
        mass=m,
        vertices=verts1,
        material=mat,
        angle=body.angle,
    )
    b2 = Body(
        pos=body.pos - 0.5 * body.radius * np.array([np.cos(angle), np.sin(angle)]),
        mass=m,
        vertices=verts2,
        material=mat,
        angle=body.angle,
    )
    return [b1, b2]


def _split_polygon(body: Body, angle: float) -> list[Body]:
    """Split a convex polygon by a line through its centroid at given angle.

    Returns two new polygon bodies.
    """
    if body.vertices is None:
        return []

    # Line through centroid: n·(x - c) = 0, where n = (cos(angle), sin(angle))
    n = np.array([np.cos(angle), np.sin(angle)])
    centroid = body.pos

    # Classify vertices by which side of the line they're on
    side_a, side_b = [], []
    for v in body.vertices:
        d = float(np.dot(v - centroid, n))
        if d >= 0:
            side_a.append(v)
        else:
            side_b.append(v)

    if len(side_a) < 3 or len(side_b) < 3:
        # Not enough vertices on one side yet, but intersections may add more
        pass

    # Find intersection points of the line with polygon edges
    intersections = []
    verts = body.vertices
    n_v = len(verts)
    for i in range(n_v):
        v1 = verts[i]
        v2 = verts[(i + 1) % n_v]
        d1 = float(np.dot(v1 - centroid, n))
        d2 = float(np.dot(v2 - centroid, n))
        if d1 * d2 < 0:  # edge crosses the line
            t = abs(d1) / (abs(d1) + abs(d2))
            intersect = v1 + t * (v2 - v1)
            intersections.append(intersect)

    if len(intersections) < 2:
        return [body]

    # Build the two polygon vertex lists
    # For each side, start with intersection, add vertices on that side, end with other intersection
    # This is a simplified version - proper polygon clipping is more complex
    # For now, just return the original body if split is complex
    if len(intersections) != 2:
        return [body]

    # Build fragment A (positive side)
    verts_a = [intersections[0]] + side_a + [intersections[1]]

    # Build fragment B (negative side)
    verts_b = [intersections[1]] + side_b + [intersections[0]]

    if len(verts_a) < 3 or len(verts_b) < 3:
        return [body]

    m = body.mass / 2.0
    mat = body.material
    b1 = Body(pos=body.pos, mass=m, vertices=np.array(verts_a), material=mat, angle=body.angle)
    b2 = Body(pos=body.pos, mass=m, vertices=np.array(verts_b), material=mat, angle=body.angle)
    return [b1, b2]


def process_fracture(contacts: list[Contact], bodies: list[Body], params: FractureParams | None = None) -> list[Body]:
    """Process all contacts for fracture, return updated body list with fractures applied.

    Modifies the bodies list in place (removes fractured bodies, adds fragments).
    """
    params = params or FractureParams()
    new_bodies = list(bodies)

    for contact in contacts:
        if check_fracture(contact):
            a, b = contact.a, contact.b
            angle = principal_stress_angle(*estimate_contact_stress(contact))

            for body in (a, b):
                # Use identity comparison to avoid numpy array comparison issues
                try:
                    idx = new_bodies.index(body)
                except ValueError:
                    continue
                if not new_bodies[idx].static:
                    new_bodies.pop(idx)
                    fragments = split_body(body, angle, params)
                    # Filter out too-small fragments
                    for frag in fragments:
                        if frag.mass >= params.min_fragment_mass:
                            new_bodies.append(frag)

    return new_bodies