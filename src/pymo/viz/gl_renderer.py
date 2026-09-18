"""GLRenderer: ModernGL + glfw renderer with GPU Instancing.

Renders SceneSnapshots at 60fps using:
- GPU Instancing: one DrawCall per mesh type (sphere, box, cylinder)
- GLSL shaders: vertex transform via instance matrix, PBR lighting
- Orbit camera with mouse control
- Frustum culling on CPU before submitting to GPU

Architecture:
  Simulation thread → SceneSnapshot → DoubleBuffer → GLRenderer (this module)
  GLRenderer reads snapshots lock-free and renders at display refresh rate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

try:
    import glfw
    import moderngl
except ModuleNotFoundError:  # pragma: no cover - exercised in CI without viz extras
    glfw = None
    moderngl = None

import numpy as np

from pymo.viz.snapshot import (
    CameraState,
    DoubleBuffer,
    InstanceData,
    MeshType,
    SceneSnapshot,
)
