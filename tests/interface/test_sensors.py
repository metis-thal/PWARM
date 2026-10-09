"""Tests for the sensor suite (interface.sensors).

Scope: placeholder camera rendering per SensorType, force/contact/IMU sensor
reads against duck-typed state mocks (cache behaviour, gravity compensation),
and the joint-state stub. State objects are mocked with SimpleNamespace —
the sensor layer is duck-typed by design.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pwarm.interface.sensors import (
    CameraSensor,
    ContactSensor,
    ForceSensor,
    IMUSensor,
    JointStateSensor,
    SensorData,
    SensorType,
)
from pwarm.physics.core.entity import EntityID


def test_sensor_data_post_init_metadata() -> None:
    """A SensorData without metadata gets an empty dict."""
    sd = SensorData(sensor_type=SensorType.IMU, timestamp=1.0, data=np.zeros(3))
    assert sd.metadata == {}


def test_camera_render_rgb_shape() -> None:
    """RGB render returns an HxWx3 uint8 frame with camera metadata."""
    cam = CameraSensor(width=32, height=16, fov=90.0)
    sd = cam.render(SimpleNamespace(t=2.5))
    assert sd.sensor_type == SensorType.CAMERA_RGB
    assert sd.timestamp == 2.5
    assert sd.data.shape == (16, 32, 3)
    assert sd.data.dtype == np.uint8
    assert sd.metadata["width"] == 32
    assert sd.metadata["height"] == 16
    assert sd.metadata["fov"] == 90.0


@pytest.mark.parametrize(
    ("sensor_type", "shape", "dtype"),
    [
        (SensorType.CAMERA_DEPTH, (16, 32), np.float32),
        (SensorType.CAMERA_SEGMENTATION, (16, 32), np.uint32),
        (SensorType.CAMERA_OPTICAL_FLOW, (16, 32, 2), np.float32),
        (SensorType.CAMERA_NORMALS, (16, 32, 3), np.float32),
    ],
)
def test_camera_render_channel_types(
        sensor_type: SensorType, shape: tuple, dtype: np.dtype) -> None:
    """Each camera channel renders zeros with the documented shape/dtype."""
    cam = CameraSensor(sensor_type=sensor_type, width=32, height=16)
    sd = cam.render(SimpleNamespace(t=0.0))
    assert sd.data.shape == shape
    assert sd.data.dtype == dtype
    assert np.all(sd.data == 0)


def test_camera_render_non_camera_type_is_empty() -> None:
    """A non-camera SensorType on a CameraSensor yields an empty array."""
    cam = CameraSensor(sensor_type=SensorType.CUSTOM)
    sd = cam.render(SimpleNamespace(t=0.0))
    assert sd.data.size == 0


def test_camera_set_pose_and_look_at() -> None:
    """set_pose updates position/target (and optional up); look_at skips up."""
    cam = CameraSensor()
    cam.set_pose(np.array([1.0, 2, 3]), np.array([0.0, 0, 0]),
                 np.array([0.0, 1, 0]))
    assert np.allclose(cam.position, [1, 2, 3])
    assert np.allclose(cam.up, [0, 1, 0])
    cam.look_at(np.array([5.0, 0, 0]), np.array([1.0, 1, 0]))
    assert np.allclose(cam.position, [5, 0, 0])
    assert np.allclose(cam.target, [1, 1, 0])
    assert np.allclose(cam.up, [0, 1, 0])  # untouched


def test_force_sensor_reads_accumulators() -> None:
    """A known entity reads force/torque from the state accumulators."""
    eid = EntityID()
    sensor = ForceSensor(eid, frame="body")
    state = SimpleNamespace(
        t=1.0,
        entity_to_rigid={eid: 0},
        rigid_force_accum=np.array([[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]),
        rigid_torque_accum=np.array([[0.1, 0.2, 0.3], [9.0, 9.0, 9.0]]),
    )
    sd = sensor.read(state)
    assert sd.sensor_type == SensorType.FORCE_TORQUE
    assert np.allclose(sd.data["force"], [1, 2, 3])
    assert np.allclose(sd.data["torque"], [0.1, 0.2, 0.3])
    assert sd.metadata == {"entity_id": str(eid), "frame": "body"}


def test_force_sensor_caches_last_reading() -> None:
    """When the entity leaves the state, the sensor reports its last values."""
    eid = EntityID()
    sensor = ForceSensor(eid)
    state = SimpleNamespace(
        t=1.0,
        entity_to_rigid={eid: 0},
        rigid_force_accum=np.array([[4.0, 0.0, 0.0]]),
        rigid_torque_accum=None,
    )
    sd = sensor.read(state)
    assert np.allclose(sd.data["force"], [4, 0, 0])
    assert np.allclose(sd.data["torque"], [0, 0, 0])  # None accumulator -> zeros
    # Entity no longer mapped: cached values are returned.
    empty = SimpleNamespace(t=2.0, entity_to_rigid={}, rigid_force_accum=None)
    sd2 = sensor.read(empty)
    assert np.allclose(sd2.data["force"], [4, 0, 0])
    assert np.allclose(sd2.data["torque"], [0, 0, 0])


def test_force_sensor_unknown_entity_zero_start() -> None:
    """An entity never seen reads zeros until first contact with a state."""
    sensor = ForceSensor(EntityID())
    state = SimpleNamespace(t=0.0, entity_to_rigid={}, rigid_force_accum=None)
    sd = sensor.read(state)
    assert np.allclose(sd.data["force"], [0, 0, 0])


def _contact(entity_a: EntityID, entity_b: EntityID, point, normal,
             depth=0.1, friction=0.5) -> SimpleNamespace:
    return SimpleNamespace(entity_a=entity_a, entity_b=entity_b,
                           point=np.array(point, dtype=np.float32),
                           normal=np.array(normal, dtype=np.float32),
                           depth=depth, friction=friction)


def test_contact_sensor_filters_by_entity() -> None:
    """Only contacts involving the sensor's entity are reported."""
    mine = EntityID()
    other_a, other_b = EntityID(), EntityID()
    sensor = ContactSensor(mine)
    state = SimpleNamespace(t=3.0)
    contacts = [
        _contact(mine, other_a, [0, 0, 0], [0, 1, 0], depth=0.25, friction=0.8),
        _contact(other_b, other_a, [1, 1, 1], [1, 0, 0]),
    ]
    sd = sensor.read(state, contacts)
    assert sd.sensor_type == SensorType.CONTACT
    assert sd.metadata["count"] == 1
    entry = sd.data["contacts"][0]
    assert entry["other_entity"] == str(other_a)
    assert entry["depth"] == 0.25
    assert entry["friction"] == 0.8
    assert np.allclose(entry["normal"], [0, 1, 0])


def test_contact_sensor_matches_either_side() -> None:
    """A contact where the sensor's entity is body b is also reported."""
    mine = EntityID()
    other = EntityID()
    sensor = ContactSensor(mine)
    sd = sensor.read(SimpleNamespace(t=0.0),
                     [_contact(other, mine, [2, 0, 0], [0, 0, 1])])
    assert sd.metadata["count"] == 1
    assert sd.data["contacts"][0]["other_entity"] == str(other)


def test_contact_sensor_empty_contacts() -> None:
    """No contacts -> empty list and count 0."""
    sensor = ContactSensor(EntityID())
    sd = sensor.read(SimpleNamespace(t=0.0), [])
    assert sd.data["contacts"] == []
    assert sd.metadata["count"] == 0


def test_imu_gravity_compensation_at_rest() -> None:
    """A stationary body reads [0, 0, +9.81] (gravity compensated)."""
    eid = EntityID()
    imu = IMUSensor(eid)
    state = SimpleNamespace(
        t=0.0,
        entity_to_rigid={eid: 0},
        rigid_linvel=np.zeros((1, 3)),
        rigid_angvel=np.zeros((1, 3)),
    )
    sd = imu.read(state, dt=1 / 60)
    assert sd.sensor_type == SensorType.IMU
    assert np.allclose(sd.data["acceleration"], [0, 0, 9.81], atol=1e-12)
    assert np.allclose(sd.data["angular_velocity"], [0, 0, 0])


def test_imu_finite_difference_acceleration() -> None:
    """Consecutive reads differentiate the velocity trace."""
    eid = EntityID()
    imu = IMUSensor(eid)
    state = SimpleNamespace(
        t=0.0,
        entity_to_rigid={eid: 0},
        rigid_linvel=np.array([[0.0, 0.0, 0.0]]),
        rigid_angvel=np.array([[0.0, 0.0, 0.5]]),
    )
    imu.read(state, dt=0.1)
    state.rigid_linvel[0] = [1.0, 0.0, 0.0]
    sd = imu.read(state, dt=0.1)
    assert np.allclose(sd.data["acceleration"], [10.0, 0.0, 9.81], atol=1e-6)
    assert np.allclose(sd.data["angular_velocity"], [0, 0, 0.5])


def test_imu_zero_dt_gives_zero_acceleration() -> None:
    """dt <= 0 skips the finite difference but keeps gravity compensation."""
    eid = EntityID()
    imu = IMUSensor(eid)
    state = SimpleNamespace(
        t=0.0,
        entity_to_rigid={eid: 0},
        rigid_linvel=np.array([[3.0, 0.0, 0.0]]),
        rigid_angvel=np.zeros((1, 3)),
    )
    sd = imu.read(state, dt=0.0)
    assert np.allclose(sd.data["acceleration"], [0, 0, 9.81])


def test_imu_unknown_entity_still_reports_gravity() -> None:
    """An entity absent from the state reads pure gravity compensation."""
    imu = IMUSensor(EntityID())
    state = SimpleNamespace(t=0.0, entity_to_rigid={},
                            rigid_linvel=None, rigid_angvel=None)
    sd = imu.read(state, dt=1 / 60)
    assert np.allclose(sd.data["acceleration"], [0, 0, 9.81])
    assert np.allclose(sd.data["angular_velocity"], [0, 0, 0])


def test_joint_state_sensor_stub() -> None:
    """The joint-state sensor is a documented stub with empty channels."""
    sensor = JointStateSensor(["hip", "knee"])
    sd = sensor.read(SimpleNamespace(t=1.5))
    assert sd.sensor_type == SensorType.JOINT_STATE
    assert sd.timestamp == 1.5
    assert sd.data == {"positions": {}, "velocities": {}, "efforts": {}}
    assert sd.metadata == {"joints": ["hip", "knee"]}


def test_sensor_type_values() -> None:
    """SensorType is an IntEnum with the documented code points."""
    assert SensorType.CAMERA_RGB == 0
    assert SensorType.FORCE_TORQUE == 5
    assert SensorType.CUSTOM == 100
