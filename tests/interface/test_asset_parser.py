"""Tests for URDF/MJCF/glTF asset parsing (interface.asset_parser).

Scope: link/joint parsing rules, geometry keyword mapping, entity mask
creation, error paths, and the documented stub behaviour of the MJCF/glTF
loaders.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pwarm.physics  # noqa: F401  (breaks the interface<->physics cycle)
from pwarm.interface.asset_parser import (
    load_gltf,
    load_mjcf,
    load_urdf,
    parse_gltf,
    parse_mjcf,
    parse_urdf,
)
from pwarm.physics.core.component import CollisionShapeComponent

ShapeType = CollisionShapeComponent.ShapeType
from pwarm.physics.core.entity import ComponentMask

URDF = """<?xml version="1.0"?>
<robot name="two_link">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="2.5"/>
      <inertia ixx="0.1" ixy="0.01" ixz="0" iyy="0.2" iyz="0" izz="0.3"/>
    </inertial>
    <visual>
      <geometry><box size="1 2 3"/></geometry>
    </visual>
    <collision>
      <geometry><sphere radius="0.7"/></geometry>
    </collision>
  </link>
  <link name="child">
    <collision>
      <geometry><cylinder radius="0.2" length="1.4"/></geometry>
    </collision>
  </link>
  <link name="static_part">
    <collision>
      <geometry><capsule radius="0.3" length="2.0"/></geometry>
    </collision>
  </link>
  <link name="meshed">
    <collision>
      <geometry><mesh filename="body.obj" scale="2 2 2"/></geometry>
    </collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="child"/>
    <origin xyz="0 0 0.5" rpy="0 0.1 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="3"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="child"/>
    <child link="static_part"/>
  </joint>
</robot>
"""


@pytest.fixture
def urdf_path(tmp_path: Path) -> Path:
    p = tmp_path / "robot.urdf"
    p.write_text(URDF, encoding="utf-8")
    return p


def test_parse_urdf_links(urdf_path: Path) -> None:
    """Links parse with mass, inertia matrix, and inertial origin."""
    data = parse_urdf(urdf_path)
    assert set(data["links"]) == {"base", "child", "static_part", "meshed"}
    base = data["links"]["base"]
    assert base.mass == 2.5
    assert np.allclose(base.com, [0, 0, 0.1])
    assert np.allclose(np.diag(base.inertia), [0.1, 0.2, 0.3])
    assert base.inertia[0, 1] == 0.01 and base.inertia[1, 0] == 0.01


def test_parse_urdf_default_link_inertial(urdf_path: Path) -> None:
    """A link without <inertial> gets unit mass, identity inertia, zero com."""
    child = parse_urdf(urdf_path)["links"]["child"]
    assert child.mass == 1.0
    assert np.allclose(child.inertia, np.eye(3))
    assert np.allclose(child.com, 0.0)


def test_parse_urdf_joints(urdf_path: Path) -> None:
    """Joints parse type/parent/child/origin/axis/limit fields."""
    joints = parse_urdf(urdf_path)["joints"]
    assert [j.name for j in joints] == ["j1", "j2"]
    j1 = joints[0]
    assert j1.type == "revolute"
    assert j1.parent == "base" and j1.child == "child"
    assert np.allclose(j1.origin_xyz, [0, 0, 0.5])
    assert np.allclose(j1.origin_rpy, [0, 0.1, 0])
    assert np.allclose(j1.axis, [0, 1, 0])
    assert (j1.limit_lower, j1.limit_upper) == (-1.5, 1.5)
    assert (j1.effort, j1.velocity) == (10.0, 3.0)
    # Fixed joint without origin/axis/limit uses documented defaults.
    j2 = joints[1]
    assert j2.type == "fixed"
    assert j2.limit_lower is None and j2.velocity is None
    assert np.allclose(j2.axis, [1, 0, 0])


def test_parse_urdf_geometry_keywords(urdf_path: Path) -> None:
    """Every supported geometry keyword maps to its shape dict."""
    links = parse_urdf(urdf_path)["links"]
    assert links["base"].collision_geometry == {"type": "sphere", "radius": 0.7}
    assert links["child"].collision_geometry == {
        "type": "cylinder", "radius": 0.2, "length": 1.4}
    assert links["static_part"].collision_geometry == {
        "type": "capsule", "radius": 0.3, "length": 2.0}
    mesh = links["meshed"].collision_geometry
    assert mesh["type"] == "mesh" and mesh["filename"] == "body.obj"
    assert np.allclose(mesh["scale"], [2, 2, 2])
    # box lives on the visual side only
    box = links["base"].visual_geometry
    assert box["type"] == "box" and np.allclose(box["size"], [1, 2, 3])


def test_parse_urdf_minimal_link(tmp_path: Path) -> None:
    """A bare <link> produces empty geometry and default joint fields."""
    p = tmp_path / "min.urdf"
    p.write_text('<robot><link name="solo"/></robot>', encoding="utf-8")
    data = parse_urdf(p)
    link = data["links"]["solo"]
    assert link.visual_geometry is None
    assert link.collision_geometry is None
    assert data["joints"] == []


def test_parse_urdf_missing_geometry_child(tmp_path: Path) -> None:
    """A visual/collision element without <geometry> parses to None."""
    p = tmp_path / "nogeom.urdf"
    p.write_text('<robot><link name="a"><visual/></link></robot>', encoding="utf-8")
    assert parse_urdf(p)["links"]["a"].visual_geometry is None


def test_parse_urdf_missing_file(tmp_path: Path) -> None:
    """A nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_urdf(tmp_path / "nope.urdf")


def test_parse_urdf_malformed_xml(tmp_path: Path) -> None:
    """Malformed XML raises ParseError."""
    p = tmp_path / "bad.urdf"
    p.write_text("<robot><link name='a'></robot>", encoding="utf-8")
    from xml.etree import ElementTree

    with pytest.raises(ElementTree.ParseError):
        parse_urdf(p)


def test_load_urdf_entity_masks(urdf_path: Path) -> None:
    """Positive mass -> dynamic mask; zero mass -> static mask."""
    entities = {e.name: e for e in load_urdf(urdf_path)}
    assert set(entities) == {"base", "child", "static_part", "meshed"}
    assert entities["base"].mask == ComponentMask.RIGID_DYNAMIC
    assert entities["child"].mask == ComponentMask.RIGID_DYNAMIC
    # static_part has no <inertial> either... mass defaults to 1.0 -> dynamic
    assert entities["static_part"].mask == ComponentMask.RIGID_DYNAMIC
    assert entities["meshed"].mask == ComponentMask.RIGID_DYNAMIC


def test_load_urdf_zero_mass_is_static(tmp_path: Path) -> None:
    """mass=0 maps to the static mask and zero inverse mass."""
    p = tmp_path / "static.urdf"
    p.write_text(
        '<robot><link name="ground"><inertial><mass value="0"/></inertial>'
        '<collision><geometry><box size="10 10 1"/></geometry></collision></link></robot>',
        encoding="utf-8")
    entities = load_urdf(p)
    assert len(entities) == 1
    assert entities[0].mask == ComponentMask.RIGID_STATIC


def test_load_urdf_registers_with_scene(urdf_path: Path) -> None:
    """With a scene, every parsed entity is registered exactly once."""
    from pwarm.physics.core.scene import Scene

    scene = Scene()
    entities = load_urdf(urdf_path, scene=scene)
    assert len(entities) == 4
    assert len(scene.entities) == 4


def test_load_urdf_duplicate_link_names_last_wins(tmp_path: Path) -> None:
    """Duplicate link names collapse to one entity (last definition wins)."""
    p = tmp_path / "dup.urdf"
    p.write_text(
        '<robot><link name="a"><inertial><mass value="1"/></inertial></link>'
        '<link name="a"><inertial><mass value="9"/></inertial></link></robot>',
        encoding="utf-8")
    data = parse_urdf(p)
    assert list(data["links"]) == ["a"]
    assert data["links"]["a"].mass == 9.0
    assert len(load_urdf(p)) == 1


def test_collision_shape_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collision geometry drives CollisionShapeComponent shape types."""
    from pwarm.interface.asset_parser import _create_entity_from_link
    from pwarm.physics.core import component as component_mod

    created: list[object] = []

    class SpyComponent(component_mod.CollisionShapeComponent):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(component_mod, "CollisionShapeComponent", SpyComponent)

    cases = [
        ({"type": "box", "size": np.array([2.0, 4.0, 6.0])}, ShapeType.BOX),
        ({"type": "sphere", "radius": 1.0}, ShapeType.SPHERE),
        ({"type": "cylinder", "radius": 1.0, "length": 2.0}, ShapeType.CYLINDER),
        ({"type": "capsule", "radius": 1.0, "length": 2.0}, ShapeType.CAPSULE),
        ({"type": "mesh", "filename": "x.obj", "scale": np.array([1, 1, 1])},
         ShapeType.TRIANGLE_MESH),
    ]
    for geometry, expected in cases:
        link = _make_link("b", geometry)
        _create_entity_from_link(link)
        assert created[-1].shape_type == expected


def _make_link(name: str, geometry: dict):
    from pwarm.interface.asset_parser import URDFLink

    return URDFLink(name=name, mass=1.0, inertia=np.eye(3), com=np.zeros(3),
                    visual_geometry=None, collision_geometry=geometry)


def test_box_collision_half_extents(tmp_path: Path) -> None:
    """URDF box sizes are full extents; the component stores half extents."""
    p = tmp_path / "box.urdf"
    p.write_text(
        '<robot><link name="b"><collision><geometry><box size="2 4 6"/></geometry>'
        "</collision></link></robot>", encoding="utf-8")
    shapes = _capture_collision_shapes(p)
    assert len(shapes) == 1
    assert np.allclose(shapes[0].half_extents, [1.0, 2.0, 3.0])


def test_cylinder_collision_half_height(urdf_path: Path) -> None:
    """URDF cylinder length is full height; the component stores half height."""
    shapes = _capture_collision_shapes(urdf_path)
    cyl = next(s for s in shapes if s.shape_type == ShapeType.CYLINDER)
    assert cyl.radius == 0.2
    assert cyl.half_height == 0.7


def _capture_collision_shapes(path: Path) -> list:
    """Load a URDF while capturing constructed collision-shape components."""
    from pwarm.interface.asset_parser import load_urdf as _load
    from pwarm.physics.core import component as component_mod

    created: list[object] = []
    real_init = component_mod.CollisionShapeComponent.__init__

    def spy_init(self, *args, **kwargs) -> None:
        real_init(self, *args, **kwargs)
        created.append(self)

    orig = component_mod.CollisionShapeComponent
    component_mod.CollisionShapeComponent = type(
        "SpyCollisionShape", (orig,), {"__init__": spy_init})
    try:
        _load(path)
    finally:
        component_mod.CollisionShapeComponent = orig
    return created


def test_mjcf_and_gltf_are_stubs(tmp_path: Path) -> None:
    """MJCF/glTF loaders are documented stubs: empty results, no file access."""
    missing = tmp_path / "does_not_exist.any"
    assert load_mjcf(missing) == []
    assert parse_mjcf(missing) == {}
    assert load_gltf(missing) == []
    assert parse_gltf(missing) == {}
