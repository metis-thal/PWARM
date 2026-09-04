"""
Asset Parsers — URDF, MJCF, GLTF loaders.

Converts standard robotics/scene formats into PWARM entities.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    from ..physics.core.scene import Scene
    from ..physics.core.entity import Entity, ComponentMask
    from ..physics.core.component import (
        TransformComponent, RigidBodyComponent, CollisionShapeComponent,
        CollisionShapeComponent as Shape
    )


@dataclass
class URDFJoint:
    name: str
    type: str  # fixed, revolute, continuous, prismatic, floating
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    limit_lower: float | None
    limit_upper: float | None
    effort: float | None
    velocity: float | None


@dataclass
class URDFLink:
    name: str
    mass: float
    inertia: np.ndarray  # 3x3
    com: np.ndarray
    visual_geometry: dict | None
    collision_geometry: dict | None


def load_urdf(path: str | Path, scene: Scene | None = None) -> list[Entity]:
    """
    Load URDF file into physics entities.
    
    Args:
        path: Path to .urdf file
        scene: Optional scene to add entities to
        
    Returns:
        List of created entities (links as rigid bodies, joints as constraints)
    """
    tree = ET.parse(path)
    root = tree.getroot()
    
    # Parse links
    links = {}
    for link_elem in root.findall("link"):
        link = _parse_urdf_link(link_elem)
        links[link.name] = link
    
    # Parse joints
    joints = []
    for joint_elem in root.findall("joint"):
        joint = _parse_urdf_joint(joint_elem)
        joints.append(joint)
    
    # Create entities for each link
    entities = []
    for link in links.values():
        entity = _create_entity_from_link(link)
        entities.append(entity)
        if scene:
            scene.add_entity(entity)
    
    # TODO: Create joint constraints
    # for joint in joints:
    #     _create_joint_constraint(joint, entities)
    
    return entities


def _parse_urdf_link(elem: ET.Element) -> URDFLink:
    name = elem.get("name", "")
    
    # Inertial
    inertial = elem.find("inertial")
    mass = 1.0
    com = np.zeros(3)
    inertia = np.eye(3)
    
    if inertial is not None:
        mass_elem = inertial.find("mass")
        if mass_elem is not None:
            mass = float(mass_elem.get("value", "1.0"))
        
        origin = inertial.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            com = np.array([float(x) for x in xyz.split()], dtype=np.float32)
        
        inertia_elem = inertial.find("inertia")
        if inertia_elem is not None:
            inertia = np.array([
                [float(inertia_elem.get("ixx", "1")), float(inertia_elem.get("ixy", "0")), float(inertia_elem.get("ixz", "0"))],
                [float(inertia_elem.get("ixy", "0")), float(inertia_elem.get("iyy", "1")), float(inertia_elem.get("iyz", "0"))],
                [float(inertia_elem.get("ixz", "0")), float(inertia_elem.get("iyz", "0")), float(inertia_elem.get("izz", "1"))],
            ], dtype=np.float32)
    
    # Visual
    visual = None
    visual_elem = elem.find("visual")
    if visual_elem is not None:
        visual = _parse_urdf_geometry(visual_elem.find("geometry"))
    
    # Collision
    collision = None
    collision_elem = elem.find("collision")
    if collision_elem is not None:
        collision = _parse_urdf_geometry(collision_elem.find("geometry"))
    
    return URDFLink(
        name=name,
        mass=mass,
        inertia=inertia,
        com=com,
        visual_geometry=visual,
        collision_geometry=collision,
    )


def _parse_urdf_joint(elem: ET.Element) -> URDFJoint:
    name = elem.get("name", "")
    joint_type = elem.get("type", "fixed")
    
    parent = elem.find("parent").get("link", "") if elem.find("parent") is not None else ""
    child = elem.find("child").get("link", "") if elem.find("child") is not None else ""
    
    origin = elem.find("origin")
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if origin is not None:
        if origin.get("xyz"):
            xyz = np.array([float(x) for x in origin.get("xyz").split()], dtype=np.float32)
        if origin.get("rpy"):
            rpy = np.array([float(x) for x in origin.get("rpy").split()], dtype=np.float32)
    
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    axis_elem = elem.find("axis")
    if axis_elem is not None and axis_elem.get("xyz"):
        axis = np.array([float(x) for x in axis_elem.get("xyz").split()], dtype=np.float32)
    
    limit_lower = None
    limit_upper = None
    limit_elem = elem.find("limit")
    if limit_elem is not None:
        if limit_elem.get("lower"):
            limit_lower = float(limit_elem.get("lower"))
        if limit_elem.get("upper"):
            limit_upper = float(limit_elem.get("upper"))
    
    effort = None
    velocity = None
    if limit_elem is not None:
        if limit_elem.get("effort"):
            effort = float(limit_elem.get("effort"))
        if limit_elem.get("velocity"):
            velocity = float(limit_elem.get("velocity"))
    
    return URDFJoint(
        name=name, type=joint_type, parent=parent, child=child,
        origin_xyz=xyz, origin_rpy=rpy, axis=axis,
        limit_lower=limit_lower, limit_upper=limit_upper,
        effort=effort, velocity=velocity,
    )


def _parse_urdf_geometry(elem: ET.Element) -> dict | None:
    if elem is None:
        return None
    
    for child in elem:
        tag = child.tag
        if tag == "box":
            size = child.get("size", "1 1 1")
            return {"type": "box", "size": np.array([float(x) for x in size.split()], dtype=np.float32)}
        elif tag == "sphere":
            radius = float(child.get("radius", "1"))
            return {"type": "sphere", "radius": radius}
        elif tag == "cylinder":
            radius = float(child.get("radius", "1"))
            length = float(child.get("length", "1"))
            return {"type": "cylinder", "radius": radius, "length": length}
        elif tag == "capsule":
            radius = float(child.get("radius", "1"))
            length = float(child.get("length", "1"))
            return {"type": "capsule", "radius": radius, "length": length}
        elif tag == "mesh":
            filename = child.get("filename", "")
            scale = child.get("scale", "1 1 1")
            return {"type": "mesh", "filename": filename, 
                    "scale": np.array([float(x) for x in scale.split()], dtype=np.float32)}
    
    return None


def _create_entity_from_link(link: URDFLink) -> Entity:
    from ..physics.core.entity import Entity, ComponentMask
    from ..physics.core.component import (
        TransformComponent, RigidBodyComponent, CollisionShapeComponent
    )
    
    entity = Entity(name=link.name)
    entity.mask = ComponentMask.RIGID_DYNAMIC if link.mass > 0 else ComponentMask.RIGID_STATIC
    
    # Transform
    transform = TransformComponent()
    transform.position = link.com
    entity.add(ComponentMask.TRANSFORM)
    
    # Rigid body
    rb = RigidBodyComponent()
    rb.mass = link.mass
    rb.inv_mass = 1.0 / link.mass if link.mass > 0 else 0.0
    rb.inertia_local = link.inertia
    # Inverse inertia
    try:
        rb.inv_inertia_local = np.linalg.inv(link.inertia)
    except np.linalg.LinAlgError:
        rb.inv_inertia_local = np.eye(3, dtype=np.float32)
    rb.is_static = link.mass <= 0
    entity.add(ComponentMask.RIGID_BODY)
    
    # Collision shape
    if link.collision_geometry:
        shape = CollisionShapeComponent()
        geom = link.collision_geometry
        if geom["type"] == "box":
            shape.shape_type = Shape.ShapeType.BOX
            shape.half_extents = geom["size"] / 2.0
        elif geom["type"] == "sphere":
            shape.shape_type = Shape.ShapeType.SPHERE
            shape.radius = geom["radius"]
        elif geom["type"] == "cylinder":
            shape.shape_type = Shape.ShapeType.CYLINDER
            shape.radius = geom["radius"]
            shape.half_height = geom["length"] / 2.0
        elif geom["type"] == "capsule":
            shape.shape_type = Shape.ShapeType.CAPSULE
            shape.radius = geom["radius"]
            shape.half_height = geom["length"] / 2.0
        elif geom["type"] == "mesh":
            shape.shape_type = Shape.ShapeType.TRIANGLE_MESH
            # TODO: Load mesh
        entity.add(ComponentMask.COLLISION_SHAPE)
    
    return entity


def parse_urdf(path: str | Path) -> dict:
    """Parse URDF without creating entities. Returns parsed data."""
    tree = ET.parse(path)
    root = tree.getroot()
    
    links = {link.name: link for link in 
             [_parse_urdf_link(e) for e in root.findall("link")]}
    joints = [_parse_urdf_joint(e) for e in root.findall("joint")]
    
    return {"links": links, "joints": joints}


def load_mjcf(path: str | Path, scene: Scene | None = None) -> list[Entity]:
    """Load MuJoCo MJCF file."""
    # MJCF parsing similar to URDF but with MuJoCo-specific elements
    # TODO: Full implementation
    return []


def parse_mjcf(path: str | Path) -> dict:
    """Parse MJCF without creating entities."""
    return {}


def load_gltf(path: str | Path, scene: Scene | None = None) -> list[Entity]:
    """Load GLTF/GLB file."""
    # Requires pygltflib
    # TODO: Implementation
    return []


def parse_gltf(path: str | Path) -> dict:
    """Parse GLTF without creating entities."""
    return {}