"""Tests for the unified collision system (broad phase, narrow phase, detect).

These tests lock in the contact conventions that the rigid solver depends on:
- Contact.normal points from entity_a to entity_b (critical for impulse signs)
- Contact.depth >= 0 for overlapping shapes
- detect() produces contacts for rigid bodies added via WorldEngine
"""

import numpy as np
import pytest

from pymo.physics import (
    CollisionShapeComponent,
    CollisionSystem,
    EntityID,
    WorldEngine,
    WorldEngineConfig,
)
from pymo.physics.collision import AABB, SAPBroadPhase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sphere(x, y, z, radius, entity_id=None, restitution=0.0):
    shape = CollisionShapeComponent()
    shape.shape_type = CollisionShapeComponent.ShapeType.SPHERE
    shape.radius = radius
    shape.restitution = restitution
    shape.entity_id = entity_id if entity_id is not None else EntityID()
    return shape


def make_box(cx, cy, cz, hx, hy, hz, entity_id=None, restitution=0.0):
    shape = CollisionShapeComponent()
    shape.shape_type = CollisionShapeComponent.ShapeType.BOX
    shape.half_extents = np.array([hx, hy, hz], dtype=np.float32)
    shape.restitution = restitution
    shape.entity_id = entity_id if entity_id is not None else EntityID()
    return shape


def transform_at(x, y, z):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return T


def make_engine(*bodies, gravity=(0.0, 0.0, -9.81)):
    """Build a WorldEngine from (position, mass, shape, shape_params) tuples."""
    config = WorldEngineConfig(
        dt=1 / 60, substeps=1, gravity=gravity, rigid={"enabled": True}
    )
    engine = WorldEngine(config)
    for pos, mass, shape, params in bodies:
        engine.create_rigid_body(pos, mass=mass, shape=shape, shape_params=params)
    engine.finalize_setup()
    return engine


# ---------------------------------------------------------------------------
# Sphere-sphere narrow phase
# ---------------------------------------------------------------------------

class TestSphereSphere:
    def test_overlapping_produces_contact(self):
        a = make_sphere(0, 0, 0, 1.0)
        b = make_sphere(1.5, 0, 0, 1.0)
        contact = CollisionSystem().narrow_phase.collide(
            a, transform_at(0, 0, 0), b, transform_at(1.5, 0, 0)
        )
        assert contact is not None
        assert contact.depth == pytest.approx(0.5, abs=1e-5)

    def test_separated_returns_none(self):
        a = make_sphere(0, 0, 0, 1.0)
        b = make_sphere(5, 0, 0, 1.0)
        contact = CollisionSystem().narrow_phase.collide(
            a, transform_at(0, 0, 0), b, transform_at(5, 0, 0)
        )
        assert contact is None

    def test_normal_points_from_a_to_b(self):
        """Contact convention: normal (A->B). B is to the +X side of A."""
        a = make_sphere(0, 0, 0, 1.0)
        b = make_sphere(1.5, 0, 0, 1.0)
        contact = CollisionSystem().narrow_phase.collide(
            a, transform_at(0, 0, 0), b, transform_at(1.5, 0, 0)
        )
        assert np.dot(contact.normal, [1, 0, 0]) > 0.99

    def test_normal_toward_each_other_on_z(self):
        a = make_sphere(0, 0, 0, 0.5)
        b = make_sphere(0, 0, -0.5, 0.5)  # B below A
        contact = CollisionSystem().narrow_phase.collide(
            a, transform_at(0, 0, 0), b, transform_at(0, 0, -0.5)
        )
        assert np.dot(contact.normal, [0, 0, -1]) > 0.99

    def test_coincident_centers_has_fallback_normal(self):
        a = make_sphere(0, 0, 0, 0.5)
        b = make_sphere(0, 0, 0, 0.5)
        contact = CollisionSystem().narrow_phase.collide(
            a, transform_at(0, 0, 0), b, transform_at(0, 0, 0)
        )
        assert contact is not None
        assert np.isfinite(contact.normal).all()


# ---------------------------------------------------------------------------
# Sphere-box narrow phase (both orderings)
# ---------------------------------------------------------------------------

class TestSphereBox:
    def test_sphere_above_box_normal_points_down(self):
        """Sphere A resting on box B: normal must point from A (sphere) to B
        (box), i.e. downward. This is the convention the impulse solver uses."""
        sphere = make_sphere(0, 0, 0.4, 0.5)   # center 0.4 above box top (z=0)
        box = make_box(0, 0, -1.0, 10, 10, 1.0)  # top surface at z=0
        contact = CollisionSystem().narrow_phase.collide(
            sphere, transform_at(0, 0, 0.4), box, transform_at(0, 0, -1.0)
        )
        assert contact is not None
        assert np.dot(contact.normal, [0, 0, -1]) > 0.99, (
            "normal must point from sphere (A) into box (B): downward"
        )

    def test_order_swap_keeps_ab_convention(self):
        """Same geometry, but box passed as A and sphere as B. The contact
        must still obey 'normal from A to B' — now pointing up (box -> sphere)
        with entity ids swapped."""
        sphere = make_sphere(0, 0, 0.4, 0.5)
        box = make_box(0, 0, -1.0, 10, 10, 1.0)
        contact = CollisionSystem().narrow_phase.collide(
            box, transform_at(0, 0, -1.0), sphere, transform_at(0, 0, 0.4)
        )
        assert contact is not None
        assert contact.entity_a is box.entity_id or contact.entity_a == box.entity_id
        assert contact.entity_b is sphere.entity_id or contact.entity_b == sphere.entity_id
        assert np.dot(contact.normal, [0, 0, 1]) > 0.99, (
            "normal must point from box (A) toward sphere (B): upward"
        )

    def test_sphere_center_inside_box_pushes_out(self):
        """Sphere center deep inside the box: contact must exist and the
        normal (A->B) must point downward into the box."""
        sphere = make_sphere(0, 0, -0.5, 0.5)  # center inside box interior
        box = make_box(0, 0, -1.0, 10, 10, 1.0)
        contact = CollisionSystem().narrow_phase.collide(
            sphere, transform_at(0, 0, -0.5), box, transform_at(0, 0, -1.0)
        )
        assert contact is not None
        assert contact.depth > 0
        assert np.dot(contact.normal, [0, 0, -1]) > 0.99

    def test_far_apart_returns_none(self):
        sphere = make_sphere(0, 0, 50, 0.5)
        box = make_box(0, 0, -1.0, 10, 10, 1.0)
        contact = CollisionSystem().narrow_phase.collide(
            sphere, transform_at(0, 0, 50), box, transform_at(0, 0, -1.0)
        )
        assert contact is None


# ---------------------------------------------------------------------------
# Box-box SAT
# ---------------------------------------------------------------------------

class TestBoxBox:
    def test_overlapping_boxes_contact(self):
        a = make_box(0, 0, 0, 1, 1, 1)
        b = make_box(1.5, 0, 0, 1, 1, 1)  # overlap of 0.5 along x
        contact = CollisionSystem().narrow_phase.collide(
            a, transform_at(0, 0, 0), b, transform_at(1.5, 0, 0)
        )
        assert contact is not None
        assert contact.depth == pytest.approx(0.5, abs=1e-5)

    def test_separated_returns_none(self):
        a = make_box(0, 0, 0, 1, 1, 1)
        b = make_box(5, 0, 0, 1, 1, 1)
        contact = CollisionSystem().narrow_phase.collide(
            a, transform_at(0, 0, 0), b, transform_at(5, 0, 0)
        )
        assert contact is None

    def test_normal_from_a_to_b(self):
        a = make_box(0, 0, 0, 1, 1, 1)
        b = make_box(1.5, 0, 0, 1, 1, 1)
        contact = CollisionSystem().narrow_phase.collide(
            a, transform_at(0, 0, 0), b, transform_at(1.5, 0, 0)
        )
        assert np.dot(contact.normal, [1, 0, 0]) > 0.99


# ---------------------------------------------------------------------------
# Broad phase (SAP)
# ---------------------------------------------------------------------------

class TestSAPBroadPhase:
    def test_overlapping_pair_found(self):
        sap = SAPBroadPhase()
        id_a, id_b = EntityID(), EntityID()
        sap.add(AABB(np.array([-1, -1, -1.0]), np.array([1, 1, 1.0]), id_a, "rigid"))
        sap.add(AABB(np.array([0.5, -1, -1.0]), np.array([2, 1, 1.0]), id_b, "rigid"))
        pairs = sap.query()
        assert len(pairs) == 1

    def test_disjoint_no_pairs(self):
        sap = SAPBroadPhase()
        id_a, id_b = EntityID(), EntityID()
        sap.add(AABB(np.array([-1, -1, -1.0]), np.array([1, 1, 1.0]), id_a, "rigid"))
        sap.add(AABB(np.array([10, -1, -1.0]), np.array([12, 1, 1.0]), id_b, "rigid"))
        assert len(sap.query()) == 0


# ---------------------------------------------------------------------------
# GJK/EPA general convex path (capsule, cylinder, convex hull)
# ---------------------------------------------------------------------------

def make_capsule(y, z, radius, half_height, entity_id=None):
    shape = CollisionShapeComponent()
    shape.shape_type = CollisionShapeComponent.ShapeType.CAPSULE
    shape.radius = radius
    shape.half_height = half_height
    shape.entity_id = entity_id if entity_id is not None else EntityID()
    return shape


def make_cylinder(radius, half_height, entity_id=None):
    shape = CollisionShapeComponent()
    shape.shape_type = CollisionShapeComponent.ShapeType.CYLINDER
    shape.radius = radius
    shape.half_height = half_height
    shape.entity_id = entity_id if entity_id is not None else EntityID()
    return shape


def make_hull(vertices, entity_id=None):
    shape = CollisionShapeComponent()
    shape.shape_type = CollisionShapeComponent.ShapeType.CONVEX_HULL
    shape.vertices = np.array(vertices, dtype=np.float32)
    shape.entity_id = entity_id if entity_id is not None else EntityID()
    return shape


class TestGJKEPA:
    def setup_method(self):
        self.np_ = CollisionSystem().narrow_phase

    def test_capsule_capsule_separated(self):
        a = make_capsule(0, 0, 0.5, 1.0)
        b = make_capsule(0, 0, 0.5, 1.0)
        contact = self.np_.collide(a, transform_at(0, 0, 0), b, transform_at(5, 0, 0))
        assert contact is None

    def test_capsule_capsule_overlapping(self):
        a = make_capsule(0, 0, 0.5, 1.0)
        b = make_capsule(0, 0, 0.5, 1.0)
        contact = self.np_.collide(a, transform_at(0, 0, 0), b, transform_at(0.6, 0, 0))
        assert contact is not None
        assert contact.depth > 0
        # A->B convention: B is on +X side
        assert np.dot(contact.normal, [1, 0, 0]) > 0.9

    def test_capsule_on_ground_normal_down(self):
        """Capsule A resting on box ground B: normal must point from A into
        B (downward) — same convention as the sphere-box fast path."""
        cap = make_capsule(0, 0, 0.5, 0.5)
        box = make_box(0, 0, -1.0, 10, 10, 1.0)
        contact = self.np_.collide(
            cap, transform_at(0, 0, 0.4), box, transform_at(0, 0, -1.0)
        )
        assert contact is not None
        assert np.dot(contact.normal, [0, 0, -1]) > 0.9

    def test_cylinder_cylinder_overlap(self):
        """Cylinders offset 0.5 along X (axes parallel, both radius 0.5):
        minimum translation is 0.5 along X, so the A->B normal is +x."""
        a = make_cylinder(0.5, 1.0)
        b = make_cylinder(0.5, 1.0)
        contact = self.np_.collide(a, transform_at(0, 0, 0), b, transform_at(0.5, 0, 0))
        assert contact is not None
        assert contact.depth == pytest.approx(0.5, abs=0.05)
        assert np.dot(contact.normal, [1, 0, 0]) > 0.9
        # Normal must be horizontal: Y-offset was zero
        assert abs(np.dot(contact.normal, [0, 1, 0])) < 0.2

    def test_cylinder_separated(self):
        a = make_cylinder(0.5, 1.0)
        b = make_cylinder(0.5, 1.0)
        contact = self.np_.collide(a, transform_at(0, 0, 0), b, transform_at(0, 5, 0))
        assert contact is None

    def test_convex_hull_vs_sphere(self):
        """Tetrahedron hull vs sphere: overlap detected with sane normal."""
        hull = make_hull([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sphere = make_sphere(0.6, 0.6, 0.6, 0.8)
        contact = self.np_.collide(
            hull, transform_at(0, 0, 0), sphere, transform_at(0.6, 0.6, 0.6)
        )
        assert contact is not None
        assert contact.depth > 0
        # A->B: sphere is toward +xyz from hull centroid
        assert np.dot(contact.normal, [1, 1, 1]) > 0

    def test_convex_hull_separated(self):
        hull = make_hull([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sphere = make_sphere(0, 0, 0, 0.3)
        contact = self.np_.collide(
            hull, transform_at(10, 0, 0), sphere, transform_at(0, 0, 0)
        )
        assert contact is None

    def test_epa_depth_accuracy_sphere_pair(self):
        """Two unit spheres with centers 1.5 apart: depth must be 0.5."""
        a = make_sphere(0, 0, 0, 1.0)
        b = make_sphere(0, 0, 0, 1.0)
        contact = self.np_.collide(a, transform_at(0, 0, 0), b, transform_at(1.5, 0, 0))
        assert contact.depth == pytest.approx(0.5, abs=1e-3)

    def test_rotated_capsule_vs_ground(self):
        """Capsule rotated 90° about X (axis now along Z) resting near ground:
        contact normal still obeys A->B (downward)."""
        cap = make_capsule(0, 0, 0.3, 0.5)
        T = np.eye(4, dtype=np.float32)
        ang = np.pi / 2
        T[:3, :3] = np.array([
            [1, 0, 0],
            [0, np.cos(ang), -np.sin(ang)],
            [0, np.sin(ang), np.cos(ang)],
        ], dtype=np.float32)
        T[:3, 3] = [0, 0, 0.7]
        box = make_box(0, 0, -1.0, 10, 10, 1.0)
        contact = self.np_.collide(cap, T, box, transform_at(0, 0, -1.0))
        if contact is not None:  # geometry may or may not touch at z=0.7
            assert np.dot(contact.normal, [0, 0, -1]) > 0.9


# ---------------------------------------------------------------------------
# Conservative CCD
# ---------------------------------------------------------------------------

class TestConservativeCCD:
    def test_fast_ball_with_ccd_does_not_tunnel(self):
        """Ball at 1300 m/s travels 21.7 m/frame — far more than the 11.5 m
        clear window (slab 10 m + both radii). With use_ccd it must be
        rewound to the time of impact and stopped; without CCD it deterministically
        tunnels through the slab."""
        def run(use_ccd):
            engine = make_engine(
                ((0, 0, 50), 1.0, "sphere",
                 {"radius": 0.5, "use_ccd": use_ccd}),
                ((0, 0, -5.0), 0.0, "box", {"half_extents": [20, 20, 5.0]}),
                gravity=(0.0, 0.0, 0.0),
            )
            engine.scene.double_buffer_write.rigid_linvel[0][:] = [0.0, 0.0, -1300.0]
            for _ in range(60):  # 1 s — enough to cross the whole scene
                state = engine.tick()
            return float(state.rigid_pos[0][2])

        z_ccd = run(use_ccd=True)
        assert -1.0 < z_ccd < 1.5, (
            f"CCD ball ended at z={z_ccd}, expected to be stopped near surface"
        )
        z_no_ccd = run(use_ccd=False)
        assert z_no_ccd < -100.0, (
            f"control failed: ball without CCD should tunnel through (z={z_no_ccd})"
        )

    def test_ccd_ignores_slow_objects(self):
        """Slow movers must not produce CCD contacts (discrete path owns them)."""
        system = CollisionSystem()
        sphere = make_sphere(0, 0, 0, 0.5)
        box = make_box(0, 0, -1.0, 10, 10, 1.0)
        contact = system.ccd.advance(
            sphere, transform_at(0, 0, 5.0), np.array([0.0, 0.0, -0.1]),
            box, transform_at(0, 0, -1.0), np.zeros(3),
            dt=1 / 60, narrow_phase=system.narrow_phase,
        )
        assert contact is None  # TOI (50 s) far beyond the 16 ms step


# ---------------------------------------------------------------------------
# CollisionSystem.detect integration through WorldEngine
# ---------------------------------------------------------------------------

class TestDetectIntegration:
    def test_falling_ball_produces_contact_on_impact(self):
        """Ball dropped onto ground must generate contacts at impact. This
        locks the regression where _get_shape returned None and collision
        was silently inert."""
        engine = make_engine(
            ((0, 0, 1.0), 1.0, "sphere", {"radius": 0.5}),
            ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5]}),
        )
        saw_contact = False
        for _ in range(120):
            state = engine.tick()
            contacts = engine.scene.collision_system.detect(state)
            if len(contacts) > 0:
                saw_contact = True
                break
        assert saw_contact, "no contact detected during 2s drop onto ground"

    def test_entity_to_rigid_mapping_populated(self):
        engine = make_engine(
            ((0, 0, 5), 1.0, "sphere", {"radius": 0.5}),
            ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5]}),
        )
        mapping = engine.scene.double_buffer_read.entity_to_rigid
        assert len(mapping) == 2
        assert sorted(mapping.values()) == [0, 1]

    def test_contact_entity_ids_resolve_to_indices(self):
        """Every contact produced by detect() must map back through
        entity_to_rigid — the solver relies on this to apply impulses."""
        engine = make_engine(
            ((0, 0, 1.0), 1.0, "sphere", {"radius": 0.5}),
            ((0, 0, -0.5), 0.0, "box", {"half_extents": [20, 20, 0.5]}),
        )
        for _ in range(120):
            state = engine.tick()
            for c in engine.scene.collision_system.detect(state):
                assert c.entity_a in state.entity_to_rigid
                assert c.entity_b in state.entity_to_rigid
