"""2D collision detection for the pymo physics kernel.

Supports circle-circle, circle-polygon and convex polygon-polygon (SAT). Returns
a contact manifold: a pair of bodies plus contact points, normals and penetration
depths used by the impulse solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bodies import Body


@dataclass
class Contact:
    """A contact point between two bodies."""

    a: Body
    b: Body
    point: np.ndarray          # world-space contact point
    normal: np.ndarray         # unit normal pointing from a to b
    penetration: float
    restitution: float
    friction: float

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (f"Contact(p=({self.point[0]:.3f},{self.point[1]:.3f}), "
                f"pen={self.penetration:.4f})")


def _closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = float(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12))
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab


def circle_circle(a: Body, b: Body) -> list[Contact]:
    d = b.pos - a.pos
    dist = float(np.linalg.norm(d))
    r_sum = a.radius + b.radius
    if dist >= r_sum or dist < 1e-9:
        return []
    normal = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
    point = a.pos + normal * a.radius
    pen = r_sum - dist
    return [
        Contact(a, b, point, normal, pen, _restitution(a, b), _friction(a, b))
    ]


def circle_polygon(circle: Body, poly: Body) -> list[Contact]:
    """Circle vs convex polygon."""
    verts = poly.world_vertices()
    n = len(verts)
    # Find closest vertex/edge on polygon to circle center
    closest = verts[0]
    best_dist = float(np.linalg.norm(circle.pos - verts[0]))
    for i in range(n):
        p = _closest_point_on_segment(circle.pos, verts[i], verts[(i + 1) % n])
        d = float(np.linalg.norm(circle.pos - p))
        if d < best_dist:
            best_dist = d
            closest = p
    if best_dist > circle.radius + 1e-9:
        return []
    # Determine if circle center is inside polygon (then normal points outward)
    inside = _point_in_polygon(circle.pos, verts)
    if inside:
        normal = _polygon_outward_normal(poly, verts, circle.pos)
        pen = circle.radius + best_dist
    else:
        normal = (circle.pos - closest) / max(best_dist, 1e-9)
        pen = circle.radius - best_dist
    return [Contact(circle, poly, closest, normal, pen, _restitution(circle, poly), _friction(circle, poly))]


def polygon_circle(poly: Body, circle: Body) -> list[Contact]:
    """Mirror of circle_polygon with bodies swapped (normal flips)."""
    contacts = circle_polygon(circle, poly)
    for c in contacts:
        c.a, c.b = poly, circle
        c.normal = -c.normal
    return contacts


def polygon_polygon(a: Body, b: Body) -> list[Contact]:
    """Convex polygon-polygon collision via SAT + face clipping.

    Returns up to 2 contact points on the actual contact surface (Box2D-style
    reference/incident face clipping) with correct normal and penetration.
    """
    va = a.world_vertices()
    vb = b.world_vertices()
    axis, overlap = _sat_find_axis(va, vb)
    if axis is None:
        return []

    # Ensure normal points from a to b (toward b's center)
    center_dist = b.pos - a.pos
    if float(np.dot(axis, center_dist)) < 0:
        axis = -axis
        overlap = -overlap
    normal = axis / np.linalg.norm(axis)
    pen = abs(overlap)

    # Build contact manifold via face clipping
    face_a = _reference_face(va, normal)      # face of a facing toward b
    face_b = _reference_face(vb, -normal)     # face of b facing toward a
    # The face with the smaller overlap along normal is the reference face
    if abs(_face_overlap(face_a, normal)) <= abs(_face_overlap(face_b, normal)):
        ref_face = face_a
        inc_face = face_b
        flip = False
    else:
        ref_face = face_b
        inc_face = face_a
        flip = True

    points = _clip_incident(ref_face, inc_face)
    contacts = []
    restitution = _restitution(a, b)
    friction = _friction(a, b)
    for p in points:
        if flip:
            contacts.append(Contact(b, a, p, -normal, pen, restitution, friction))
        else:
            contacts.append(Contact(a, b, p, normal, pen, restitution, friction))
    return contacts


def detect_collision(a: Body, b: Body) -> list[Contact]:
    """Detect and return contacts between two bodies (empty if no overlap)."""
    if a.is_sensor or b.is_sensor:
        return []
    if a.is_circle() and b.is_circle():
        return circle_circle(a, b)
    if a.is_circle() and b.is_polygon():
        return circle_polygon(a, b)
    if a.is_polygon() and b.is_circle():
        return polygon_circle(a, b)
    if a.is_polygon() and b.is_polygon():
        return polygon_polygon(a, b)
    return []


# -- internal helpers -------------------------------------------------------

def _restitution(a: Body, b: Body) -> float:
    return max(a.material.restitution, b.material.restitution)


def _friction(a: Body, b: Body) -> float:
    return np.sqrt(a.material.friction * b.material.friction)


def _point_in_polygon(p: np.ndarray, verts: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test (convex ok)."""
    inside = False
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        if (y0 > p[1]) != (y1 > p[1]) and p[0] < (x1 - x0) * (p[1] - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def _polygon_outward_normal(poly: Body, verts: np.ndarray, interior_point: np.ndarray) -> np.ndarray:
    """Find the edge normal of the polygon pointing away from interior_point."""
    n = len(verts)
    best = None
    best_dist = -np.inf
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        edge = b - a
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        # normal should point outward: away from polygon centroid
        centroid = poly.pos
        if np.dot(normal, centroid - a) > 0:
            normal = -normal
        d = float(np.dot(normal, interior_point - a))
        if d > best_dist:
            best_dist = d
            best = normal
    return best if best is not None else np.array([1.0, 0.0])


def _sat_find_axis(va: np.ndarray, vb: np.ndarray):
    """SAT: return (axis, overlap) with minimal penetration, or (None, None)."""
    axes = _all_axes(va) + _all_axes(vb)
    best_axis = None
    best_overlap = np.inf
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            continue
        axis = axis / norm
        proj_a_min, proj_a_max = _project(va, axis)
        proj_b_min, proj_b_max = _project(vb, axis)
        overlap = min(proj_a_max, proj_b_max) - max(proj_a_min, proj_b_min)
        if overlap < 0:
            return None, None  # separating axis found -> no collision
        if overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis
    return best_axis, best_overlap


def _all_axes(verts: np.ndarray) -> list[np.ndarray]:
    n = len(verts)
    axes = []
    for i in range(n):
        edge = verts[(i + 1) % n] - verts[i]
        axes.append(np.array([edge[1], -edge[0]]))
    return axes


def _project(verts: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    projs = verts @ axis
    return float(projs.min()), float(projs.max())


def _contact_point_sat(va: np.ndarray, vb: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Estimate contact point as the deepest vertex of va along normal, projected."""
    # Find vertex of a deepest along -normal (most penetrating into b)
    depths = va @ (-normal)
    idx = int(np.argmax(depths))
    return va[idx].copy()


def _reference_face(verts: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the edge (p1, p2) of `verts` most anti-parallel to `normal`.

    This is the face facing the other polygon (the contact face).
    """
    n = len(verts)
    best = None
    best_score = -np.inf
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        edge_n = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(edge_n)
        if norm < 1e-12:
            continue
        edge_n /= norm
        score = float(np.dot(edge_n, normal))
        if score > best_score:
            best_score = score
            best = (p1, p2)
    return best


def _face_overlap(face: tuple[np.ndarray, np.ndarray], normal: np.ndarray) -> float:
    """Signed overlap of a face's edge normal with the collision normal."""
    p1, p2 = face
    edge = p2 - p1
    edge_n = np.array([edge[1], -edge[0]])
    norm = np.linalg.norm(edge_n)
    if norm < 1e-12:
        return 0.0
    edge_n /= norm
    return float(np.dot(edge_n, normal))


def _clip_incident(
    ref_face: tuple[np.ndarray, np.ndarray], inc_face: tuple[np.ndarray, np.ndarray]
) -> list[np.ndarray]:
    """Clip the incident face against the reference face's side planes.

    Returns 1-2 contact points lying on the reference face's support plane,
    clipped to the reference face's extent.
    """
    r1, r2 = ref_face
    i1, i2 = inc_face
    ref_edge = r2 - r1
    ref_len2 = float(np.dot(ref_edge, ref_edge))
    if ref_len2 < 1e-12:
        return [i1, i2]

    # Side plane normals (perpendicular to ref edge, pointing into ref face)
    tangent = ref_edge / np.sqrt(ref_len2)
    # Normal of the reference face
    ref_normal = np.array([-tangent[1], tangent[0]])
    # Ensure ref_normal points toward the incident face
    if np.dot(ref_normal, (i1 + i2) / 2 - r1) < 0:
        ref_normal = -ref_normal

    points = [i1.copy(), i2.copy()]

    # Clip each incident vertex by the two side planes
    for idx, (origin, direction) in enumerate([(r1, tangent), (r2, -tangent)]):
        clipped = []
        for k in range(len(points)):
            cur = points[k]
            nxt = points[(k + 1) % len(points)]
            d_cur = float(np.dot(cur - origin, direction))
            d_nxt = float(np.dot(nxt - origin, direction))
            if d_cur >= 0:
                clipped.append(cur)
            if (d_cur >= 0) != (d_nxt >= 0):
                # Segment crosses the plane -> clip
                t = d_cur / (d_cur - d_nxt)
                clipped.append(cur + t * (nxt - cur))
        points = clipped

    # Project clipped points onto the reference face's support plane (depth = 0)
    # Keep penetration as the distance along ref_normal
    result = []
    for p in points:
        depth = float(np.dot(p - r1, ref_normal))
        result.append(p - ref_normal * depth)
    return result if result else [r1.copy()]
