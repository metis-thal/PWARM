"""
Sensors — Camera, force, and other sensors for AI/RL observation.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..physics.core.state import State
    from ..physics.core.entity import EntityID


class SensorType(IntEnum):
    CAMERA_RGB = 0
    CAMERA_DEPTH = 1
    CAMERA_SEGMENTATION = 2
    CAMERA_OPTICAL_FLOW = 3
    CAMERA_NORMALS = 4
    FORCE_TORQUE = 5
    CONTACT = 6
    IMU = 7
    JOINT_STATE = 8
    CUSTOM = 100


@dataclass
class SensorData:
    """Generic sensor data container."""
    sensor_type: SensorType
    timestamp: float
    data: np.ndarray | dict
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CameraSensor:
    """
    Virtual camera sensor.
    
    Renders from simulation state (reads double buffer).
    Supports: RGB, Depth, Segmentation, Normals, Optical Flow.
    """
    
    def __init__(self, 
                 sensor_type: SensorType = SensorType.CAMERA_RGB,
                 width: int = 640,
                 height: int = 480,
                 fov: float = 60.0,
                 near: float = 0.1,
                 far: float = 100.0):
        self.sensor_type = sensor_type
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        
        # Camera transform (world space)
        self.position = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # Previous frame for optical flow
        self._prev_rgb = None
        self._prev_depth = None
    
    def render(self, state: State) -> SensorData:
        """Render sensor data from simulation state."""
        # This would use the renderer (Nyx/Luisa/Pyrender)
        # For now, return placeholder
        
        if self.sensor_type == SensorType.CAMERA_RGB:
            data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elif self.sensor_type == SensorType.CAMERA_DEPTH:
            data = np.zeros((self.height, self.width), dtype=np.float32)
        elif self.sensor_type == SensorType.CAMERA_SEGMENTATION:
            data = np.zeros((self.height, self.width), dtype=np.uint32)
        elif self.sensor_type == SensorType.CAMERA_OPTICAL_FLOW:
            data = np.zeros((self.height, self.width, 2), dtype=np.float32)
        elif self.sensor_type == SensorType.CAMERA_NORMALS:
            data = np.zeros((self.height, self.width, 3), dtype=np.float32)
        else:
            data = np.array([])
        
        return SensorData(
            sensor_type=self.sensor_type,
            timestamp=state.t,
            data=data,
            metadata={
                "width": self.width,
                "height": self.height,
                "fov": self.fov,
                "position": self.position.copy(),
                "target": self.target.copy(),
            }
        )
    
    def set_pose(self, position: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> None:
        self.position = position.astype(np.float32)
        self.target = target.astype(np.float32)
        if up is not None:
            self.up = up.astype(np.float32)
    
    def look_at(self, position: np.ndarray, target: np.ndarray) -> None:
        self.position = position.astype(np.float32)
        self.target = target.astype(np.float32)


class ForceSensor:
    """
    Force/torque sensor at a specific frame.
    
    Measures contact forces, joint torques, etc.
    """
    
    def __init__(self, entity_id: EntityID, frame: str = "world"):
        self.entity_id = entity_id
        self.frame = frame  # "world" or "local"
        self._last_forces = np.zeros(3, dtype=np.float32)
        self._last_torques = np.zeros(3, dtype=np.float32)
    
    def read(self, state: State) -> SensorData:
        """Read forces from state."""
        # Get rigid index
        rigid_idx = state.entity_to_rigid.get(self.entity_id)
        if rigid_idx is not None and state.rigid_force_accum is not None:
            forces = state.rigid_force_accum[rigid_idx]
            torques = state.rigid_torque_accum[rigid_idx] if state.rigid_torque_accum is not None else np.zeros(3)
            
            self._last_forces = forces.copy()
            self._last_torques = torques.copy()
        else:
            forces = self._last_forces
            torques = self._last_torques
        
        return SensorData(
            sensor_type=SensorType.FORCE_TORQUE,
            timestamp=state.t,
            data={"force": forces, "torque": torques},
            metadata={"entity_id": str(self.entity_id), "frame": self.frame}
        )


class ContactSensor:
    """Contact sensor — reports all contacts for an entity."""
    
    def __init__(self, entity_id: EntityID):
        self.entity_id = entity_id
    
    def read(self, state: State, contacts: list) -> SensorData:
        """Filter contacts for this entity."""
        entity_contacts = []
        for c in contacts:
            if c.entity_a == self.entity_id or c.entity_b == self.entity_id:
                other = c.entity_b if c.entity_a == self.entity_id else c.entity_a
                entity_contacts.append({
                    "other_entity": str(other),
                    "point": c.point.copy(),
                    "normal": c.normal.copy(),
                    "depth": c.depth,
                    "friction": c.friction,
                })
        
        return SensorData(
            sensor_type=SensorType.CONTACT,
            timestamp=state.t,
            data={"contacts": entity_contacts},
            metadata={"entity_id": str(self.entity_id), "count": len(entity_contacts)}
        )


class IMUSensor:
    """IMU sensor — acceleration + angular velocity."""
    
    def __init__(self, entity_id: EntityID):
        self.entity_id = entity_id
        self._last_linvel = np.zeros(3)
        self._last_angvel = np.zeros(3)
    
    def read(self, state: State, dt: float) -> SensorData:
        rigid_idx = state.entity_to_rigid.get(self.entity_id)
        if rigid_idx is not None:
            linvel = state.rigid_linvel[rigid_idx]
            angvel = state.rigid_angvel[rigid_idx]
            
            # Approximate linear acceleration
            accel = (linvel - self._last_linvel) / dt if dt > 0 else np.zeros(3)
            
            self._last_linvel = linvel.copy()
            self._last_angvel = angvel.copy()
        else:
            accel = np.zeros(3)
            angvel = np.zeros(3)
        
        # Add gravity
        accel[2] += 9.81  # Assuming Z-up
        
        return SensorData(
            sensor_type=SensorType.IMU,
            timestamp=state.t,
            data={"acceleration": accel, "angular_velocity": angvel},
            metadata={"entity_id": str(self.entity_id)}
        )


class JointStateSensor:
    """Joint state sensor for articulated bodies."""
    
    def __init__(self, joint_names: list[str]):
        self.joint_names = joint_names
    
    def read(self, state: State) -> SensorData:
        # Would read joint positions/velocities from articulation
        return SensorData(
            sensor_type=SensorType.JOINT_STATE,
            timestamp=state.t,
            data={"positions": {}, "velocities": {}, "efforts": {}},
            metadata={"joints": self.joint_names}
        )