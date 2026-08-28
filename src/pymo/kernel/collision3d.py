"""3D collision detection using GJK (Gilbert-Johnson-Keerthi) and EPA (Expanding Polytope Algorithm).

Implements collision detection for convex shapes using support mappings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import norm

from pymo.kernel.bodies3d import Body
from pymo.kernel.math3d import cross3, dot3, mat3_from_quat, normalize


@dataclass
class Contact3D:
    """A contact point between two 3D bodies."""
    a: Body
    b: Body
    point: np.ndarray          # world-space contact point
    normal: np.ndarray         # unit normal pointing from a to b
    penetration: float         # penetration depth
    restitution: float
    friction: float


def _support_point(body: Body, direction: np.ndarray) -> np.ndarray:
    """Get the support point of a body in the given direction."""
    return body.world_support_point(direction)


def _gjk_support(body_a: Body, body_b: Body, direction: np.ndarray) -> np.ndarray:
    """Support function for the Minkowski difference A - B."""
    return _support_point(body_a, direction) - _support_point(body_b, -direction)


def _gjk(body_a: Body, body_b: Body, max_iterations: int = 32) -> tuple[bool, np.ndarray]:
    """GJK collision detection.
    
    Returns (intersecting, simplex) where simplex is the final simplex
    (list of points in Minkowski difference) or empty if not intersecting.
    """
    # Initial direction
    dir = np.array([1.0, 0.0, 0.0])
    
    # Initial point
    simplex = [_gjk_support(body_a, body_b, dir)]
    dir = -simplex[0]
    
    for _ in range(max_iterations):
        # Get new support point
        point = _gjk_support(body_a, body_b, dir)
        
        # Check if we're past the origin
        if dot3(point, dir) <= 0:
            return False, np.array(simplex)
        
        simplex.append(point)
        
        # Check if simplex contains origin
        if _handle_simplex(simplex, dir):
            return True, np.array(simplex)
    
    return False, np.array(simplex)


def _handle_simplex(simplex: list[np.ndarray], dir: np.ndarray) -> bool:
    """Process simplex and return True if it contains origin.
    Updates direction for next iteration.
    """
    n = len(simplex)
    
    if n == 2:
        # Line segment
        return _handle_line(simplex, dir)
    elif n == 3:
        # Triangle
        return _handle_triangle(simplex, dir)
    elif n == 4:
        # Tetrahedron
        return _handle_tetrahedron(simplex, dir)
    
    return False


def _handle_line(simplex: list[np.ndarray], dir: np.ndarray) -> bool:
    """Handle line segment case (2 points)."""
    a = simplex[1]  # newest point
    b = simplex[0]  # previous point
    ab = b - a
    ao = -a
    
    # Check if origin is on the line segment ab
    # Origin is on segment if it projects between a and b
    # Equivalently: dot(ao, ab) > 0 and dot(-b, a - b) > 0
    ab_ao = dot3(ab, ao)
    bo = -b
    ba = a - b
    bo_ba = dot3(bo, ba)
    
    if ab_ao > 0 and bo_ba > 0:
        # Origin is on the line segment - collision!
        return True
    
    if ab_ao > 0:
        # Origin is in the ab region
        dir[:] = cross3(cross3(ab, ao), ab)
    else:
        # Origin is in the a region
        simplex.pop(1)  # Remove b
        dir[:] = ao
    
    return False


def _handle_triangle(simplex: list[np.ndarray], dir: np.ndarray) -> bool:
    """Handle triangle case (3 points)."""
    a = simplex[2]
    b = simplex[1]
    c = simplex[0]
    
    ab = b - a
    ac = c - a
    ao = -a
    
    # Edge normals
    abc = cross3(ab, ac)
    
    # Check if origin is in abc region (inside triangle)
    if dot3(abc, ao) > 0:
        # Inside triangle region
        dir[:] = abc
        return True
    
    # Check edge regions
    if dot3(cross3(ab, abc), ao) > 0:
        # Origin is in ab edge region
        simplex.pop(2)  # Remove c
        dir[:] = cross3(cross3(ab, ao), ab)
    elif dot3(cross3(abc, ac), ao) > 0:
        # Origin is in ac edge region
        simplex.pop(1)  # Remove b
        dir[:] = cross3(cross3(ac, ao), ac)
    else:
        # Origin is in a region
        simplex.pop(2)
        simplex.pop(1)
        dir[:] = -simplex[0]
    
    return False


def _handle_tetrahedron(simplex: list[np.ndarray], dir: np.ndarray) -> bool:
    """Handle tetrahedron case (4 points)."""
    a = simplex[3]
    b = simplex[2]
    c = simplex[1]
    d = simplex[0]
    
    ao = -a
    
    # Face normals (pointing outward)
    abc = cross3(b - a, c - a)
    acd = cross3(c - a, d - a)
    adb = cross3(d - a, b - a)
    
    # Check if origin is outside any face
    if dot3(abc, ao) > 0:
        # Origin is outside face abc
        simplex.pop(0)  # Remove d
        return _handle_triangle(simplex, dir)
    
    if dot3(acd, ao) > 0:
        # Origin is outside face acd
        simplex[1] = simplex[3]  # Replace b with d
        return _handle_triangle(simplex, dir)
    
    if dot3(adb, ao) > 0:
        # Origin is outside face adb
        simplex[2] = simplex[3]  # Replace c with d
        return _handle_triangle(simplex, dir)
    
    # Origin is inside tetrahedron - collision!
    return True


def gjk_intersect(body_a: Body, body_b: Body) -> tuple[bool, np.ndarray]:
    """GJK collision test between two bodies.
    
    Returns (intersecting, simplex).
    """
    return _gjk(body_a, body_b)


@dataclass
class EPAResult:
    """Result of EPA penetration depth calculation."""
    normal: np.ndarray
    depth: float
    contact_point: np.ndarray


def epa(body_a: Body, body_b: Body, simplex: np.ndarray, 
        max_iterations: int = 32, tolerance: float = 1e-4) -> EPAResult | None:
    """EPA (Expanding Polytope Algorithm) for penetration depth.
    
    Takes the final GJK simplex and expands it to find the contact normal and depth.
    """
    if len(simplex) < 4:
        return None
    
    # Build initial polytope from GJK simplex (tetrahedron)
    faces = [
        (0, 1, 2),  # base
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3)
    ]
    
    vertices = list(simplex)
    
    for _ in range(max_iterations):
        # Find face closest to origin
        min_dist = float('inf')
        closest_face = -1
        
        for i, (i1, i2, i3) in enumerate(faces):
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            normal = normalize(cross3(v2 - v1, v3 - v1))
            dist = np.dot(v1, normal)
            if dist < min_dist:
                min_dist = dist
                closest_face = i
        
        if closest_face == -1:
            return None
        
        # Get closest face
        i1, i2, i3 = faces[closest_face]
        v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
        normal = normalize(cross3(vertices[i2] - vertices[i1], vertices[i3] - vertices[i1]))
        
        # Find support point in direction of normal
        support = _gjk_support(body_a, body_b, normal)
        
        # Check if we've converged
        dist = np.dot(support, normal)
        if dist - min_dist < tolerance:
            # Found penetration depth
            # Compute contact point on body A
            contact_point = _support_point(body_a, normal) - normal * min_dist * 0.5
            return EPAResult(normal=normal, depth=min_dist, contact_point=contact_point)
        
        # Add new vertex and update faces
        new_idx = len(vertices)
        vertices.append(support)
        
        # Find all faces visible from new vertex
        new_faces = []
        visible = [False] * len(faces)
        
        for i, (i1, i2, i3) in enumerate(faces):
            v1, v2, v3 = vertices[i1], vertices[i2], vertices[i3]
            normal = normalize(cross3(vertices[i2] - vertices[i1], vertices[i3] - vertices[i1]))
            if np.dot(vertices[new_idx] - vertices[i1], normal) > 0:
                visible[i] = True
        
# Build horizon edges and create new faces
        for i, (i1, i2, i3) in enumerate(faces):
            if visible[i]:
                # Edge is on horizon if adjacent face is not visible
                edges = [(i1, i2), (i2, i3), (i3, i1)]
                for e1, e2 in edges:
                    # Check if edge is on horizon
                    is_horizon = True
                    for j, (j1, j2, j3) in enumerate(faces):
                        if i != j and not visible[j] and e1 in (j1, j2, j3) and e2 in (j1, j2, j3):
                            is_horizon = False
                            break
                    
                    if is_horizon:
                        # Create new face with new vertex
                        # Ensure correct winding
                        if dot3(cross3(vertices[e2] - vertices[e1], vertices[new_idx] - vertices[e1]), 
                                vertices[0] - vertices[e1]) > 0:
                            new_faces.append((e1, e2, new_idx))
                        else:
                            new_faces.append((e2, e1, new_idx))
        
        # Remove visible faces and add new ones
        faces = [f for i, f in enumerate(faces) if not visible[i]]
        faces.extend(new_faces)
        
        if not faces:
            break
    
    # Fallback
    return None


def _sphere_sphere_penetration(body_a: Body, body_b: Body) -> Contact3D | None:
    """Direct penetration calculation for sphere-sphere collision."""
    if not (body_a.is_sphere() and body_b.is_sphere()):
        return None
    
    r1 = body_a.shape.radius
    r2 = body_b.shape.radius
    center_a = body_a.pos
    center_b = body_b.pos
    diff = center_b - center_a
    center_dist = norm(diff)
    
    if center_dist > r1 + r2:
        return None
    
    penetration = (r1 + r2) - center_dist
    if penetration <= 0:
        return None
    
    if center_dist > 1e-6:
        normal = diff / center_dist
    else:
        normal = np.array([1.0, 0.0, 0.0])
    
    contact_point = center_a + normal * r1 - normal * penetration * 0.5
    return Contact3D(
        a=body_a, b=body_b,
        point=contact_point, normal=normal,
        penetration=penetration,
        restitution=max(body_a.material.restitution, body_b.material.restitution),
        friction=np.sqrt(body_a.material.friction * body_b.material.friction)
    )


def _box_box_penetration(body_a: Body, body_b: Body) -> Contact3D | None:
    """Direct penetration calculation for box-box collision using SAT."""
    if not (body_a.is_box() and body_b.is_box()):
        return None
    
    # Use SAT (Separating Axis Theorem) for box-box
    # Transform boxes to world space
    a_verts = body_a.get_world_vertices()
    b_verts = body_b.get_world_vertices()
    
    # Test axes: face normals of both boxes + cross products of edges
    axes = []
    
    # Box A face normals (3)
    R_a = mat3_from_quat(body_a.orn)
    for i in range(3):
        axes.append(R_a[:, i])
    
    # Box B face normals (3)
    R_b = mat3_from_quat(body_b.orn)
    for i in range(3):
        axes.append(R_b[:, i])
    
    # Cross products of edges (9)
    for i in range(3):
        for j in range(3):
            cross = cross3(R_a[:, i], R_b[:, j])
            if norm(cross) > 1e-6:
                axes.append(normalize(cross))
    
    min_overlap = float('inf')
    best_axis = None
    
    for axis in axes:
        # Project both boxes onto axis
        proj_a = a_verts @ axis
        proj_b = b_verts @ axis
        
        a_min, a_max = np.min(proj_a), np.max(proj_a)
        b_min, b_max = np.min(proj_b), np.max(proj_b)
        
        if a_max < b_min or b_max < a_min:
            return None  # Separating axis found
        
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis
    
    if best_axis is None:
        return None
    
    # Ensure normal points from a to b
    center_diff = body_b.pos - body_a.pos
    if dot3(best_axis, center_diff) < 0:
        best_axis = -best_axis
    
    penetration = min_overlap
    contact_point = body_a.pos + best_axis * min_overlap * 0.5
    
    return Contact3D(
        a=body_a, b=body_b,
        point=contact_point, normal=best_axis,
        penetration=penetration,
        restitution=max(body_a.material.restitution, body_b.material.restitution),
        friction=np.sqrt(body_a.material.friction * body_b.material.friction)
    )


def _sphere_box_penetration(body_a: Body, body_b: Body) -> Contact3D | None:
    """Direct penetration calculation for sphere-box collision."""
    # Ensure a is sphere, b is box
    if body_a.is_sphere() and body_b.is_box():
        sphere, box = body_a, body_b
    elif body_b.is_sphere() and body_a.is_box():
        sphere, box = body_b, body_a
    else:
        return None
    
    # Transform sphere center to box local space
    R_box = mat3_from_quat(box.orn)
    local_sphere = R_box.T @ (sphere.pos - box.pos)
    
    # Find closest point on box to sphere center
    closest = np.clip(local_sphere, -box.shape.half_extents, box.shape.half_extents)
    dist_sq = np.sum((local_sphere - closest)**2)
    
    if dist_sq > sphere.shape.radius**2:
        return None
    
    dist = np.sqrt(dist_sq)
    penetration = sphere.shape.radius - dist
    if penetration <= 0:
        return None
    
    # Normal in world space (from box to sphere)
    normal_local = local_sphere - closest
    if norm(normal_local) > 1e-6:
        normal = R_box @ (normal_local / (dist + 1e-12))
    else:
        # Sphere center inside box - use face normal
        normal = np.array([0.0, 0.0, 1.0])  # fallback
    
    return Contact3D(
        a=body_a, b=body_b,
        point=sphere.pos - normal * sphere.shape.radius,
        normal=normal,
        penetration=sphere.shape.radius - dist,
        restitution=max(body_a.material.restitution, body_b.material.restitution),
        friction=np.sqrt(body_a.material.friction * body_b.material.friction)
    )


def detect_collision(body_a: Body, body_b: Body) -> list[Contact3D]:
    """Detect collisions between two 3D bodies.
    
    Returns list of contacts.
    """
    intersecting, simplex = gjk_intersect(body_a, body_b)
    if not intersecting:
        return []
    
    # Try EPA first for general convex shapes
    epa_result = epa(body_a, body_b, simplex)
    if epa_result is not None:
        restitution = max(body_a.material.restitution, body_b.material.restitution)
        friction = np.sqrt(body_a.material.friction * body_b.material.friction)
        
        contact = Contact3D(
            a=body_a,
            b=body_b,
            point=epa_result.contact_point,
            normal=epa_result.normal,
            penetration=epa_result.depth,
            restitution=restitution,
            friction=friction
        )
        return [contact]
    
    # Fallback: direct penetration calculation for simple shape pairs
    # Sphere-Sphere
    if body_a.is_sphere() and body_b.is_sphere():
        contact = _sphere_sphere_penetration(body_a, body_b)
        if contact:
            return [contact]
    
    # Box-Box
    if body_a.is_box() and body_b.is_box():
        contact = _box_box_penetration(body_a, body_b)
        if contact:
            return [contact]
    
    # Sphere-Box
    if body_a.is_sphere() and body_b.is_box():
        contact = _sphere_box_penetration(body_a, body_b)
        if contact:
            return [contact]
    if body_b.is_sphere() and body_a.is_box():
        contact = _sphere_box_penetration(body_b, body_a)
        if contact:
            return [contact]
    
    # Fallback for other shapes
    return []


def detect_all_collisions(bodies: list[Body]) -> list[Contact3D]:
    """Detect all pairwise collisions in a list of bodies."""
    contacts = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            contacts.extend(detect_collision(bodies[i], bodies[j]))
    return contacts