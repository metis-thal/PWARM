"""
Collision System — Unified broad/narrow phase for all solvers.

SAP (Sweep and Prune) broad phase + GJK/EPA narrow phase + CCD.
Shared by all solvers via CollisionShapeComponent.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations as _itertools_combinations
from typing import TYPE_CHECKING

import numpy as np


def _closest_point_in_affine_space(pts: np.ndarray):
    """Closest point to the origin within the convex hull of pts.

    Args:
        pts: (n, 3) array of simplex vertices.

    Returns:
        (point, valid): valid is False when the affine-hull closest point has
        a negative barycentric weight (the true closest point then lies on a
        lower-dimensional face, which the caller explores via subsets).
    """
    n = len(pts)
    if n == 1:
        return pts[0].copy(), True
    # KKT system for  min |sum(lam_i * p_i)|^2  s.t.  sum(lam_i) = 1
    G = pts @ pts.T
    KKT = np.zeros((n + 1, n + 1))
    KKT[:n, :n] = 2.0 * G
    KKT[:n, n] = 1.0
    KKT[n, :n] = 1.0
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    try:
        sol, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(3), False
    lam = sol[:n]
    if np.any(lam < -1e-9):
        return np.zeros(3), False
    return pts.T @ lam, True

if TYPE_CHECKING:
    from ..core.component import CollisionShapeComponent
    from ..core.entity import EntityID
    from ..core.state import State

# Runtime import needed for shape type dispatch (no circular dependency:
# core.component does not import collision)
from ..core.component import CollisionShapeComponent


@dataclass(slots=True)
class Contact:
    """Collision contact point."""
    entity_a: EntityID
    entity_b: EntityID
    point: np.ndarray          # (3,) world space contact point
    normal: np.ndarray         # (3,) unit normal (from A to B)
    depth: float               # Penetration depth
    friction: float = 0.5
    restitution: float = 0.0
    solver_a: str = ""         # "rigid", "sph", "fem", etc.
    solver_b: str = ""
    # For CCD
    time_of_impact: float = 0.0


@dataclass
class ContactList:
    """List of contacts for a simulation step."""
    contacts: list[Contact]
    
    def __len__(self) -> int:
        return len(self.contacts)
    
    def __iter__(self):
        return iter(self.contacts)
    
    def filter_by_solver(self, solver_a: str, solver_b: str | None = None) -> ContactList:
        """Filter contacts by solver types."""
        if solver_b is None:
            filtered = [c for c in self.contacts if c.solver_a == solver_a or c.solver_b == solver_a]
        else:
            filtered = [c for c in self.contacts 
                       if (c.solver_a == solver_a and c.solver_b == solver_b) or
                          (c.solver_a == solver_b and c.solver_b == solver_a)]
        return ContactList(filtered)
    
    def get_for_entity(self, entity_id: EntityID) -> list[Contact]:
        return [c for c in self.contacts if c.entity_a == entity_id or c.entity_b == entity_id]


class AABB:
    """Axis-Aligned Bounding Box."""
    __slots__ = ("entity_id", "max", "min", "solver_type")
    
    def __init__(self, min_bounds: np.ndarray, max_bounds: np.ndarray, entity_id: EntityID, solver_type: str):
        self.min = min_bounds.astype(np.float32)
        self.max = max_bounds.astype(np.float32)
        self.entity_id = entity_id
        self.solver_type = solver_type
    
    def overlaps(self, other: AABB) -> bool:
        return (self.min[0] <= other.max[0] and self.max[0] >= other.min[0] and
                self.min[1] <= other.max[1] and self.max[1] >= other.min[1] and
                self.min[2] <= other.max[2] and self.max[2] >= other.min[2])
    
    def surface_area(self) -> float:
        d = self.max - self.min
        return 2.0 * (d[0]*d[1] + d[1]*d[2] + d[2]*d[0])
    
    def center(self) -> np.ndarray:
        return (self.min + self.max) * 0.5


class SAPBroadPhase:
    """
    Sweep and Prune Broad Phase.
    
    Maintains sorted lists of AABB min/max on each axis.
    O(n log n) insertion, O(n + k) query where k = overlaps.
    """
    
    def __init__(self):
        self.axis_lists: list[list[tuple[float, int, bool]]] = [[], [], []]  # (value, aabb_index, is_min)
        self.aabbs: list[AABB] = []
        self.aabb_to_index: dict[int, int] = {}  # entity_id -> aabb index
        self._dirty = True
    
    def add(self, aabb: AABB) -> int:
        """Add AABB, return index."""
        index = len(self.aabbs)
        self.aabbs.append(aabb)
        self.aabb_to_index[id(aabb.entity_id)] = index
        self._dirty = True
        return index
    
    def remove(self, entity_id: EntityID) -> None:
        if id(entity_id) in self.aabb_to_index:
            index = self.aabb_to_index.pop(id(entity_id))
            # Mark as removed (lazy deletion)
            self.aabbs[index] = None
            self._dirty = True
    
    def update(self, entity_id: EntityID, new_aabb: AABB) -> None:
        if id(entity_id) in self.aabb_to_index:
            index = self.aabb_to_index[id(entity_id)]
            self.aabbs[index] = new_aabb
            self._dirty = True
        else:
            self.add(new_aabb)
    
    def _rebuild(self) -> None:
        """Rebuild sorted axis lists."""
        self.axis_lists = [[], [], []]
        for i, aabb in enumerate(self.aabbs):
            if aabb is None:
                continue
            for axis in range(3):
                self.axis_lists[axis].append((aabb.min[axis], i, True))
                self.axis_lists[axis].append((aabb.max[axis], i, False))
        
        for axis in range(3):
            self.axis_lists[axis].sort(key=lambda x: x[0])
        
        self._dirty = False
    
    def query(self) -> list[tuple[int, int]]:
        """Return list of potentially overlapping AABB index pairs.

        A pair overlaps only if its intervals overlap on ALL three axes;
        per-axis overlap sets are intersected.
        """
        if self._dirty:
            self._rebuild()

        axis_pairs: list[set[tuple[int, int]]] = []
        for axis in range(3):
            overlaps: set[tuple[int, int]] = set()
            active: set[int] = set()
            for value, index, is_min in self.axis_lists[axis]:
                if is_min:
                    for other in active:
                        if index != other:
                            overlaps.add(tuple(sorted((index, other))))
                    active.add(index)
                else:
                    active.discard(index)
            axis_pairs.append(overlaps)

        return list(axis_pairs[0] & axis_pairs[1] & axis_pairs[2])


class GJKNarrowPhase:
    """
    GJK (Gilbert-Johnson-Keerthi) + EPA (Expanding Polytope Algorithm).
    
    Computes exact contact point, normal, depth for convex shapes.
    """
    
    def __init__(self):
        self.max_iterations = 20
        self.tolerance = 1e-6
    
    def collide(self, shape_a: CollisionShapeComponent, transform_a: np.ndarray,
                shape_b: CollisionShapeComponent, transform_b: np.ndarray) -> Contact | None:
        """
        Test collision between two shapes.
        
        Args:
            shape_a, shape_b: CollisionShapeComponents
            transform_a, transform_b: 4x4 world transforms
            
        Returns:
            Contact if colliding, None otherwise
        """
        # Dispatch based on shape types
        type_a = shape_a.shape_type
        type_b = shape_b.shape_type
        
        # Sphere-Sphere (fast path)
        if type_a == CollisionShapeComponent.ShapeType.SPHERE and \
           type_b == CollisionShapeComponent.ShapeType.SPHERE:
            return self._sphere_sphere(shape_a, transform_a, shape_b, transform_b)
        
        # Sphere-Box
        if (type_a == CollisionShapeComponent.ShapeType.SPHERE and 
            type_b == CollisionShapeComponent.ShapeType.BOX):
            return self._sphere_box(shape_a, transform_a, shape_b, transform_b)
        if (type_a == CollisionShapeComponent.ShapeType.BOX and
            type_b == CollisionShapeComponent.ShapeType.SPHERE):
            contact = self._sphere_box(shape_b, transform_b, shape_a, transform_a)
            if contact:
                # Inner contact has A=sphere, B=box with normal sphere->box.
                # Outer convention wants A=box, B=sphere: swap identities and
                # flip the normal so it points from A (box) to B (sphere).
                contact.entity_a, contact.entity_b = contact.entity_b, contact.entity_a
                contact.normal = -contact.normal
            return contact
        
        # Box-Box (SAT)
        if type_a == CollisionShapeComponent.ShapeType.BOX and \
           type_b == CollisionShapeComponent.ShapeType.BOX:
            return self._box_box_sat(shape_a, transform_a, shape_b, transform_b)
        
        # General convex: GJK + EPA
        return self._gjk_epa(shape_a, transform_a, shape_b, transform_b)
    
    def _sphere_sphere(self, a, ta, b, tb) -> Contact | None:
        pa = ta[:3, 3]
        pb = tb[:3, 3]
        diff = pb - pa
        dist = np.linalg.norm(diff)
        radius_sum = a.radius + b.radius
        
        if dist < radius_sum:
            if dist > 1e-6:
                normal = diff / dist
            else:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            point = pa + normal * a.radius
            depth = radius_sum - dist
            return Contact(
                entity_a=a.entity_id, entity_b=b.entity_id,
                point=point, normal=normal, depth=depth,
                friction=min(a.friction, b.friction),
                restitution=min(a.restitution, b.restitution)
            )
        return None
    
    def _sphere_box(self, sphere, ts, box, tb) -> Contact | None:
        # Transform sphere center to box local space
        box_inv = np.linalg.inv(tb)
        center_local = box_inv[:3, :3] @ (ts[:3, 3] - tb[:3, 3])
        
        # Clamp to box extents
        closest = np.clip(center_local, -box.half_extents, box.half_extents)
        diff = center_local - closest
        dist_sq = np.sum(diff**2)
        
        if dist_sq < sphere.radius**2:
            dist = np.sqrt(dist_sq)
            if dist > 1e-6:
                normal_local = diff / dist
            else:
                normal_local = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # Contact convention: normal points from A (sphere) to B (box).
            # diff points from the box surface toward the sphere center (B->A),
            # so negate to get A->B.
            normal = -(tb[:3, :3] @ normal_local)
            point_world = tb[:3, 3] + tb[:3, :3] @ closest
            depth = sphere.radius - dist
            
            return Contact(
                entity_a=sphere.entity_id, entity_b=box.entity_id,
                point=point_world, normal=normal, depth=depth,
                friction=min(sphere.friction, box.friction),
                restitution=min(sphere.restitution, box.restitution)
            )
        return None
    
    def _box_box_sat(self, a, ta, b, tb) -> Contact | None:
        """Separating Axis Theorem for OBB-OBB."""
        # Test 15 axes: 3 face normals from A, 3 from B, 9 cross products
        Ra = ta[:3, :3]
        Rb = tb[:3, :3]
        Ta = ta[:3, 3]
        Tb = tb[:3, 3]
        
        # Axes in world space
        axes = []
        for i in range(3):
            axes.append(Ra[:, i])      # A face normals
            axes.append(Rb[:, i])      # B face normals
        for i in range(3):
            for j in range(3):
                cross = np.cross(Ra[:, i], Rb[:, j])
                if np.linalg.norm(cross) > 1e-6:
                    axes.append(cross / np.linalg.norm(cross))
        
        min_depth = float('inf')
        best_normal = None
        
        for axis in axes:
            # Project both boxes onto axis
            proj_a = self._project_box(Ra, a.half_extents, axis)
            proj_b = self._project_box(Rb, b.half_extents, axis)
            center_dist = abs(np.dot(Tb - Ta, axis))
            
            if center_dist > proj_a + proj_b:
                return None  # Separating axis found
            
            depth = proj_a + proj_b - center_dist
            if depth < min_depth:
                min_depth = depth
                best_normal = axis
        
        if best_normal is not None:
            # Ensure normal points from A to B
            if np.dot(best_normal, Tb - Ta) < 0:
                best_normal = -best_normal
            
            # Contact point approximation
            point = Ta + Ra @ a.half_extents  # Simplified
            
            return Contact(
                entity_a=a.entity_id, entity_b=b.entity_id,
                point=point, normal=best_normal, depth=min_depth,
                friction=min(a.friction, b.friction),
                restitution=min(a.restitution, b.restitution)
            )
        return None
    
    def _project_box(self, R: np.ndarray, half_extents: np.ndarray, axis: np.ndarray) -> float:
        """Project OBB onto axis, return half-extent."""
        return (abs(np.dot(R[:, 0], axis)) * half_extents[0] +
                abs(np.dot(R[:, 1], axis)) * half_extents[1] +
                abs(np.dot(R[:, 2], axis)) * half_extents[2])
    
    def _gjk_epa(self, a, ta, b, tb) -> Contact | None:
        """GJK (Gilbert-Johnson-Keerthi) + EPA for general convex shapes.

        Works in the Minkowski difference C = A ⊖ B, whose support in
        direction d is support_A(d) - support_B(-d). C contains the origin
        iff the shapes overlap. EPA then finds the closest boundary point of
        C to the origin; its direction is the contact normal (A->B) and its
        magnitude the penetration depth (verified against the sphere-box
        fast path convention).
        """
        simplex = self._gjk_simplex(a, ta, b, tb)
        if simplex is None:
            return None
        return self._epa(a, ta, b, tb, simplex)

    # -- support functions ---------------------------------------------------

    def _support(self, shape: CollisionShapeComponent, transform: np.ndarray,
                 d_world: np.ndarray) -> np.ndarray:
        """World-space support point: farthest vertex of shape along d_world."""
        R = transform[:3, :3]
        center = transform[:3, 3]
        d = R.T @ d_world
        n = float(np.linalg.norm(d))
        if n < 1e-12:
            d = np.array([0.0, 0.0, 1.0])
            n = 1.0
        d_hat = d / n

        st = shape.shape_type
        if st == CollisionShapeComponent.ShapeType.SPHERE:
            local = d_hat * shape.radius
        elif st == CollisionShapeComponent.ShapeType.BOX:
            local = np.sign(d_hat) * shape.half_extents
        elif st == CollisionShapeComponent.ShapeType.CAPSULE:
            # Segment along local Y: endpoints (0, ±half_height, 0), plus radius
            axis = np.array([0.0, 1.0, 0.0])
            endpoint = axis * shape.half_height if d_hat[1] >= 0 else -axis * shape.half_height
            local = endpoint + d_hat * shape.radius
        elif st == CollisionShapeComponent.ShapeType.CYLINDER:
            # Rim circle in local XZ plane, axis along local Y
            radial = np.array([d_hat[0], 0.0, d_hat[2]])
            rl = float(np.linalg.norm(radial))
            if rl > 1e-12:
                radial /= rl
            else:
                radial = np.array([1.0, 0.0, 0.0])
            local = radial * shape.radius + np.array([0.0, np.sign(d_hat[1]) * shape.half_height, 0.0])
        elif st == CollisionShapeComponent.ShapeType.CONVEX_HULL:
            verts = shape.vertices
            if verts is None or len(verts) == 0:
                local = np.zeros(3)
            else:
                local = verts[int(np.argmax(verts @ d_hat))]
        else:
            local = d_hat * 0.5  # conservative fallback

        return center + R @ local

    def _minkowski(self, a, ta, b, tb, d: np.ndarray) -> np.ndarray:
        """Support of C = A ⊖ B in direction d."""
        return self._support(a, ta, d) - self._support(b, tb, -d)

    # -- GJK ------------------------------------------------------------------

    def _closest_point_on_simplex(self, simplex: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
        """Brute-force closest point of the simplex to the origin.

        Evaluates every non-empty subset (<= 15 for a tetrahedron) and returns
        the point with the smallest norm plus the subset indices that span it.
        Robust and simple; only used for capsule/cylinder/convex pairs.
        """
        n = len(simplex)
        best_point = simplex[0].copy()
        best_subset = [0]
        best_norm = float(np.linalg.norm(best_point))
        for size in range(1, n + 1):
            for subset in _itertools_combinations(range(n), size):
                pts = np.array([simplex[i] for i in subset])
                point, valid = _closest_point_in_affine_space(pts)
                if valid is not None and valid:
                    norm = float(np.linalg.norm(point))
                    if norm < best_norm - 1e-12:
                        best_norm = norm
                        best_point = point
                        best_subset = list(subset)
        return best_point, best_subset

    def _gjk_simplex(self, a, ta, b, tb) -> list[np.ndarray] | None:
        """Run GJK. Returns a simplex (containing the origin) if the shapes
        overlap, else None."""
        d = ta[:3, 3] - tb[:3, 3]
        if float(np.linalg.norm(d)) < 1e-8:
            d = np.array([0.0, 0.0, 1.0])

        simplex: list[np.ndarray] = [self._minkowski(a, ta, b, tb, d)]
        d = -simplex[0]

        for _ in range(32):
            s = self._minkowski(a, ta, b, tb, d)
            if float(np.dot(s, d)) < 0.0:
                return None  # separating: support cannot reach the origin
            simplex.append(s)

            q, subset = self._closest_point_on_simplex(simplex)
            if float(np.linalg.norm(q)) < 1e-9:
                # Origin lies on/in the simplex: the shapes overlap. Build a
                # full tetrahedron for EPA by expanding along candidate
                # directions until fresh vertices are found (the support along
                # the current direction may duplicate an existing vertex when
                # the origin sits on a face/edge of the Minkowski difference).
                candidates = [
                    d,
                    np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]),
                    np.array([0.0, 1.0, 1.0]), np.array([1.0, 0.0, 1.0]),
                ]
                for n_dir in candidates:
                    if len(simplex) >= 4:
                        break
                    nl = float(np.linalg.norm(n_dir))
                    if nl < 1e-12:
                        continue
                    extra = self._minkowski(a, ta, b, tb, n_dir / nl)
                    if all(float(np.linalg.norm(extra - p)) > 1e-6 for p in simplex):
                        simplex.append(extra)
                if len(simplex) < 4:
                    return None  # degenerate (flat shapes) — treat as touching
                return simplex

            # Reduce to the closest feature and aim at the origin
            simplex[:] = [simplex[i] for i in subset]
            d = -q

        return None  # iteration budget exhausted: treat as separated

    # -- GJK distance ---------------------------------------------------------

    def distance(self, a, ta, b, tb) -> float:
        """Separation distance between two convex shapes (0 if overlapping).

        GJK distance mode: iterates the closest point of the Minkowski
        difference to the origin; its norm is the separation distance.
        """
        if self.collide(a, ta, b, tb) is not None:
            return 0.0

        d = ta[:3, 3] - tb[:3, 3]
        if float(np.linalg.norm(d)) < 1e-8:
            d = np.array([0.0, 0.0, 1.0])

        simplex = [self._minkowski(a, ta, b, tb, d)]
        q, _ = self._closest_point_on_simplex(simplex)
        prev_norm = float(np.linalg.norm(q))
        for _ in range(32):
            d = -q
            if float(np.linalg.norm(d)) < 1e-12:
                return 0.0
            s = self._minkowski(a, ta, b, tb, d)
            simplex.append(s)
            q, subset = self._closest_point_on_simplex(simplex)
            simplex[:] = [simplex[i] for i in subset]
            norm = float(np.linalg.norm(q))
            if prev_norm - norm < 1e-7:
                break  # support no longer improves: converged
            prev_norm = norm
        return prev_norm

    # -- EPA ------------------------------------------------------------------

    def _epa(self, a, ta, b, tb, simplex: list[np.ndarray]) -> Contact | None:
        """Expanding Polytope Algorithm: minimum penetration vector of
        C = A ⊖ B. Returns a Contact with normal (A->B) and depth."""
        pts = [p.copy() for p in simplex]
        if len(pts) < 4:
            return None

        # Faces as index triples; normals computed on the fly, oriented
        # outward via the polytope centroid.
        faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]

        normal = np.zeros(3)
        depth = 0.0
        for _ in range(48):
            centroid = np.mean(np.array(pts), axis=0)

            best_dist = np.inf
            best_face = None
            best_normal = None
            for f in faces:
                v0, v1, v2 = pts[f[0]], pts[f[1]], pts[f[2]]
                n = np.cross(v1 - v0, v2 - v0)
                nl = float(np.linalg.norm(n))
                if nl < 1e-12:
                    continue
                n_hat = n / nl
                fc = (v0 + v1 + v2) / 3.0
                if float(np.dot(n_hat, fc - centroid)) < 0:  # orient outward
                    n_hat = -n_hat
                dist = float(np.dot(n_hat, v0))  # origin inside => positive
                if dist < best_dist:
                    best_dist = dist
                    best_face = f
                    best_normal = n_hat

            if best_face is None or not np.isfinite(best_dist):
                return None

            s = self._minkowski(a, ta, b, tb, best_normal)
            expansion = float(np.dot(s, best_normal)) - best_dist
            if expansion < 1e-4:
                normal = best_normal
                depth = max(best_dist, 0.0)
                break

            # Insert s: remove faces visible from s, bridge horizon edges.
            # Each horizon edge (used by exactly one visible face) yields ONE
            # new face — this keeps the polytope closed and growth linear.
            pts.append(s)
            si = len(pts) - 1
            visible = []
            edge_use: dict[tuple[int, int], int] = {}
            for f in faces:
                v0, v1, v2 = pts[f[0]], pts[f[1]], pts[f[2]]
                n = np.cross(v1 - v0, v2 - v0)
                nl = float(np.linalg.norm(n))
                seen = False
                if nl >= 1e-12:
                    n_hat = n / nl
                    fc = (v0 + v1 + v2) / 3.0
                    if float(np.dot(n_hat, fc - centroid)) < 0:
                        n_hat = -n_hat
                    seen = float(np.dot(n_hat, s - v0)) > 1e-10
                if seen:
                    visible.append(f)
                    for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                        key = (min(u, v), max(u, v))
                        edge_use[key] = edge_use.get(key, 0) + 1
            if not visible:
                # Support point did not expand the polytope: converged
                normal = best_normal
                depth = max(best_dist, 0.0)
                break
            horizon = [e for e, c in edge_use.items() if c == 1]
            for f in visible:
                faces.remove(f)
            if not horizon or len(faces) + len(horizon) > 512:
                # Degenerate or overgrown polytope: return current best
                normal = best_normal
                depth = max(best_dist, 0.0)
                break
            for u, v in horizon:
                faces.append([u, v, si])
        else:
            # Budget exhausted: the best-so-far face is monotonic-improving
            # and accurate enough for impulse resolution.
            normal = best_normal
            depth = max(best_dist, 0.0)

        # Contact point: project A's center onto the contact plane (adequate
        # for impulse resolution; the solver only uses normal/depth today).
        point_a = ta[:3, 3]
        point = point_a - normal * depth
        return Contact(
            entity_a=a.entity_id, entity_b=b.entity_id,
            point=point, normal=normal, depth=depth,
            friction=min(a.friction, b.friction),
            restitution=min(a.restitution, b.restitution),
        )


class ConservativeCCD:
    """
    Conservative Continuous Collision Detection.

    Conservative advancement: repeatedly advance time by distance/speed (a
    lower bound on the time of impact, so the impact can never be skipped)
    until the shapes touch or the step ends. Prevents tunneling for
    fast-moving bodies whose per-frame displacement exceeds their size.
    """

    def __init__(self):
        self.ccd_threshold = 0.01   # min separation treated as contact
        self.max_iterations = 24

    def advance(self, shape_a: CollisionShapeComponent, ta: np.ndarray, va: np.ndarray,
                shape_b: CollisionShapeComponent, tb: np.ndarray, vb: np.ndarray,
                dt: float, narrow_phase: GJKNarrowPhase) -> Contact | None:
        """Time-of-impact query for A moving with va against B moving with vb
        within a step of length dt. Returns a Contact at the TOI or None."""
        v_rel = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
        speed = float(np.linalg.norm(v_rel))
        if speed < 1e-9:
            return None
        direction = v_rel / speed

        # Already overlapping: the discrete pipeline owns this pair
        if narrow_phase.collide(shape_a, ta, shape_b, tb) is not None:
            return None

        t = 0.0
        T = ta.copy()
        for _ in range(self.max_iterations):
            d = narrow_phase.distance(shape_a, T, shape_b, tb)
            if d <= self.ccd_threshold:
                break  # touching at time t
            t += d / speed  # conservative: cannot skip past the impact
            if t > dt:
                return None  # no impact within this step
            T = ta.copy()
            T[:3, 3] = T[:3, 3] + np.asarray(va, dtype=np.float64) * t

        if t <= 0.0 or t > dt:
            return None

        # Record A's center at the TOI. The solver rewinds the CCD mover to
        # this position: end-of-step impulse resolution would otherwise leave
        # the body far past the impact for very high speeds.
        T_toi = ta.copy()
        T_toi[:3, 3] = T_toi[:3, 3] + np.asarray(va, dtype=np.float64) * t

        # Nudge slightly past the TOI so the narrow phase produces a real
        # contact with the exact surface normal (TOI is a lower bound).
        T2 = ta.copy()
        T2[:3, 3] = T2[:3, 3] + np.asarray(va, dtype=np.float64) * min(t + 1e-3, dt)
        contact = narrow_phase.collide(shape_a, T2, shape_b, tb)
        if contact is None:
            # Fallback: head-on approximation along the relative motion
            contact = Contact(
                entity_a=shape_a.entity_id, entity_b=shape_b.entity_id,
                point=T2[:3, 3].astype(np.float32),
                normal=direction.astype(np.float32),
                depth=0.0,
                friction=min(shape_a.friction, shape_b.friction),
                restitution=min(shape_a.restitution, shape_b.restitution),
            )
        contact.time_of_impact = t
        contact.point = T_toi[:3, 3].astype(np.float32)
        return contact


class CollisionSystem:
    """
    Unified collision detection for all solvers.
    
    Broad phase (SAP) -> Narrow phase (GJK/EPA/SAT) -> CCD
    """
    
    def __init__(self):
        self.broad_phase = SAPBroadPhase()
        self.narrow_phase = GJKNarrowPhase()
        self.ccd = ConservativeCCD()
        self._aabb_cache: dict[EntityID, AABB] = {}
        # Scene reference for entity/shape lookups (set by WorldEngine)
        self.scene = None
    
    def detect(self, state: State) -> ContactList:
        """Detect all contacts for current state."""
        # 1. Build/update AABBs for all entities with collision shapes
        self._update_aabbs(state)
        
        # 2. Broad phase: get potentially overlapping pairs
        pairs = self.broad_phase.query()
        
        # 3. Narrow phase: exact collision test
        contacts = []
        for idx_a, idx_b in pairs:
            aabb_a = self.broad_phase.aabbs[idx_a]
            aabb_b = self.broad_phase.aabbs[idx_b]
            
            if aabb_a is None or aabb_b is None:
                continue
            
            # Get shapes and transforms from state
            shape_a = self._get_shape(state, aabb_a.entity_id)
            shape_b = self._get_shape(state, aabb_b.entity_id)
            transform_a = self._get_transform(state, aabb_a.entity_id)
            transform_b = self._get_transform(state, aabb_b.entity_id)
            
            if shape_a is None or shape_b is None:
                continue
            
            contact = self.narrow_phase.collide(shape_a, transform_a, shape_b, transform_b)
            if contact:
                contact.solver_a = aabb_a.solver_type
                contact.solver_b = aabb_b.solver_type
                contacts.append(contact)
        
        # 4. CCD for fast objects (opt-in via CollisionShapeComponent.use_ccd)
        ccd_contacts = self._conservative_ccd(state)
        contacts.extend(ccd_contacts)

        return ContactList(contacts)

    def _conservative_ccd(self, state: State) -> list[Contact]:
        """Run conservative advancement for CCD-enabled fast movers against
        all other rigid bodies."""
        out: list[Contact] = []
        if state.rigid_pos is None or len(state.rigid_pos) == 0:
            return out
        dt = state.dt if state.dt and state.dt > 0 else 1.0 / 60.0
        index_to_entity = {v: k for k, v in state.entity_to_rigid.items()}
        n = len(state.rigid_pos)

        movers = []
        for i in range(n):
            entity_id = index_to_entity.get(i)
            if entity_id is None:
                continue
            shape = self._get_shape(state, entity_id)
            if shape is None or not shape.use_ccd:
                continue
            vel = state.rigid_linvel[i] if state.rigid_linvel is not None else np.zeros(3)
            if float(np.linalg.norm(vel)) * dt < 1e-6:
                continue
            movers.append((i, shape, self._get_transform(state, entity_id), vel))
        if not movers:
            return out

        zeros = np.zeros(3, dtype=np.float32)
        for i, shape_a, ta, va in movers:
            index_to_entity[i]
            for j in range(n):
                if j == i:
                    continue
                entity_b = index_to_entity.get(j)
                if entity_b is None:
                    continue
                shape_b = self._get_shape(state, entity_b)
                if shape_b is None:
                    continue
                tb = self._get_transform(state, entity_b)
                vb = state.rigid_linvel[j] if state.rigid_linvel is not None else zeros
                contact = self.ccd.advance(shape_a, ta, va, shape_b, tb, vb, dt,
                                           self.narrow_phase)
                if contact is not None:
                    contact.solver_a = "rigid"
                    contact.solver_b = "rigid"
                    out.append(contact)
        return out
    
    def _update_aabbs(self, state: State) -> None:
        """Update AABBs for all collision entities."""
        # Rigid bodies
        if state.rigid_pos is not None:
            # Reverse lookup: rigid array index -> EntityID
            index_to_entity = {v: k for k, v in state.entity_to_rigid.items()}
            for i in range(len(state.rigid_pos)):
                entity_id = index_to_entity.get(i)
                if entity_id is None:
                    continue
                shape = self._get_shape(state, entity_id)
                if shape:
                    aabb = self._compute_aabb(shape, state.rigid_pos[i],
                                             state.rigid_quat[i] if state.rigid_quat is not None else None)
                    self.broad_phase.update(entity_id, aabb)
        
        # SPH particles (broad phase per particle is expensive; use spatial hash instead)
        # For now, skip SPH in broad phase; handle fluid-rigid in coupler
        
        # FEM nodes
        # ... similar
    
    def _compute_aabb(self, shape: CollisionShapeComponent, pos: np.ndarray, 
                      quat: np.ndarray | None) -> AABB:
        """Compute world-space AABB for shape at transform."""
        if shape.shape_type == CollisionShapeComponent.ShapeType.SPHERE:
            r = shape.radius
            return AABB(pos - r, pos + r, shape.entity_id, "rigid")
        
        elif shape.shape_type == CollisionShapeComponent.ShapeType.BOX:
            # Rotate half-extents
            if quat is not None:
                R = self._quat_to_rot(quat)
                half_extents_world = np.abs(R) @ shape.half_extents
            else:
                half_extents_world = shape.half_extents
            return AABB(pos - half_extents_world, pos + half_extents_world,
                       shape.entity_id, "rigid")

        elif shape.shape_type in (CollisionShapeComponent.ShapeType.CAPSULE,
                                  CollisionShapeComponent.ShapeType.CYLINDER):
            # Axis along local Y: extent is half_height + radius on Y,
            # radius on X/Z. Conservative world AABB via rotation matrix.
            r = shape.radius
            hh = shape.half_height + (r if shape.shape_type == CollisionShapeComponent.ShapeType.CAPSULE else 0.0)
            local_half = np.array([r, hh, r], dtype=np.float32)
            if quat is not None:
                R = self._quat_to_rot(quat)
                half = np.abs(R) @ local_half
            else:
                half = local_half
            return AABB(pos - half, pos + half, shape.entity_id, "rigid")

        elif shape.shape_type == CollisionShapeComponent.ShapeType.CONVEX_HULL:
            if shape.vertices is not None and len(shape.vertices) > 0:
                if quat is not None:
                    R = self._quat_to_rot(quat)
                    world_verts = shape.vertices @ R.T
                else:
                    world_verts = shape.vertices
                lo = (pos + world_verts).min(axis=0)
                hi = (pos + world_verts).max(axis=0)
                return AABB(lo, hi, shape.entity_id, "rigid")

        # Default: large box
        return AABB(pos - 10, pos + 10, shape.entity_id, "rigid")
    
    def _quat_to_rot(self, q: np.ndarray) -> np.ndarray:
        """Quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
    
    def _get_shape(self, state: State, entity_id: EntityID) -> CollisionShapeComponent | None:
        """Get collision shape from the entity's stored components."""
        if self.scene is None:
            return None
        entity = self.scene.entities.get(entity_id)
        if entity is None:
            return None
        return entity.user_data.get('collision_shape')
    
    def _get_transform(self, state: State, entity_id: EntityID) -> np.ndarray:
        """Get 4x4 world transform for entity."""
        # Get rigid index
        rigid_idx = state.entity_to_rigid.get(entity_id)
        if rigid_idx is not None and state.rigid_pos is not None:
            pos = state.rigid_pos[rigid_idx]
            quat = state.rigid_quat[rigid_idx] if state.rigid_quat is not None else np.array([1,0,0,0], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, 3] = pos
            T[:3, :3] = self._quat_to_rot(quat)
            return T
        return np.eye(4, dtype=np.float32)
    
    def _get_entity_id_from_rigid_index(self, index: int) -> EntityID:
        """Reverse lookup: rigid index -> entity_id."""
        # Would need inverse mapping
        from ..core.entity import EntityID
        return EntityID()