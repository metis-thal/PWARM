"""
Collision System — Unified broad/narrow phase for all solvers.

SAP (Sweep and Prune) broad phase + GJK/EPA narrow phase + CCD.
Shared by all solvers via CollisionShapeComponent.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.state import State
    from ..core.entity import EntityID, ComponentMask
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
    __slots__ = ("min", "max", "entity_id", "solver_type")
    
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
        """Return list of potentially overlapping AABB index pairs."""
        if self._dirty:
            self._rebuild()
        
        # Active intervals sweep
        active = [set(), set(), set()]
        pairs = set()
        
        for axis in range(3):
            for value, index, is_min in self.axis_lists[axis]:
                if is_min:
                    # Check overlap with active intervals
                    for other in active[axis]:
                        if index != other:
                            pair = tuple(sorted((index, other)))
                            pairs.add(pair)
                    active[axis].add(index)
                else:
                    active[axis].discard(index)
        
        return list(pairs)


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
            
            # Transform normal to world
            normal = tb[:3, :3] @ normal_local
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
        """General GJK + EPA for convex shapes. Placeholder."""
        # Full implementation requires support mapping for each shape type
        return None


class ConservativeCCD:
    """
    Conservative Continuous Collision Detection.
    
    Swept volumes + linear motion bounds. Catches tunneling for fast objects.
    """
    
    def __init__(self):
        self.ccd_threshold = 0.01  # Minimum velocity for CCD
    
    def sweep(self, state: State, narrow_phase: GJKNarrowPhase) -> list[Contact]:
        """Run CCD for fast-moving entities. Returns additional contacts."""
        ccd_contacts = []
        
        # For each entity with CCD enabled and high velocity
        # Compute swept AABB, test against other entities
        # If collision, compute time of impact and contact
        
        return ccd_contacts


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
    
    def detect(self, state: State) -> ContactList:
        """Detect all contacts for current state."""
        # 1. Build/update AABBs for all entities with collision shapes
        self._update_aabbs(state)
        
        # 2. Broad phase: get potentially overlapping pairs
        pairs = self.broad_phase.query()
        
        # 3. Narrow phase: exact collision test
        contacts = []
        for idx_a, idx_b in pairs:
            aabb_a = self.aabbs[idx_a]
            aabb_b = self.aabbs[idx_b]
            
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
        
        # 4. CCD for fast objects
        ccd_contacts = self.ccd.sweep(state, self.narrow_phase)
        contacts.extend(ccd_contacts)
        
        return ContactList(contacts)
    
    def _update_aabbs(self, state: State) -> None:
        """Update AABBs for all collision entities."""
        # Rigid bodies
        if state.rigid_pos is not None:
            for i in range(len(state.rigid_pos)):
                # Get entity_id from metadata
                entity_id = self._get_entity_id_from_rigid_index(i)
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
        """Get collision shape from state metadata."""
        # This would look up the shape from entity manager
        return None
    
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