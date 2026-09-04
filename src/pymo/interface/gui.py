"""
GUI — Built-in viewer with camera sensors, entity inspector, parameter tuning.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..physics.core.scene import Scene
    from ..physics.core.state import State


@dataclass
class CameraConfig:
    position: tuple[float, float, float] = (5.0, 5.0, 5.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 60.0
    near: float = 0.1
    far: float = 1000.0
    width: int = 1280
    height: int = 720


class GUI:
    """
    Built-in simulation GUI.
    
    Features:
    - Real-time 3D view with camera controls
    - Entity inspector (select entity, view/modify components)
    - Parameter tuning (solver options, coupling, gravity)
    - Play/pause/step controls
    - Console for commands
    - Profiling overlay
    """
    
    def __init__(self, scene: Scene, camera: CameraConfig | None = None):
        self.scene = scene
        self.camera = camera or CameraConfig()
        self.running = False
        self.paused = False
        self.show_ui = True
        self.selected_entity = None
        self._renderer = None
        self._window = None
    
    def run(self) -> None:
        """Main GUI loop."""
        self._init_window()
        self._init_renderer()
        
        self.running = True
        while self.running:
            self._handle_events()
            
            if not self.paused:
                self.scene.step()
            
            self._render()
        
        self._cleanup()
    
    def _init_window(self) -> None:
        """Initialize window (GLFW/SDL/pygame)."""
        # TODO: Use glfw or pygame
        pass
    
    def _init_renderer(self) -> None:
        """Initialize renderer (Nyx/Luisa/Pyrender)."""
        # TODO: Initialize Nyx renderer
        pass
    
    def _handle_events(self) -> None:
        """Handle input events."""
        # Keyboard/mouse handling
        # Camera controls: orbit, pan, zoom
        # Entity selection: click to select
        # UI: ImGui or similar
        pass
    
    def _render(self) -> None:
        """Render frame."""
        # Get render snapshot from double buffer
        snapshot = self.scene.get_render_snapshot()
        if snapshot is None:
            return
        
        # Render scene
        # self._renderer.render(snapshot, self.camera)
        
        # Render UI overlay
        if self.show_ui:
            self._render_ui()
    
    def _render_ui(self) -> None:
        """Render ImGui UI."""
        # Main menu bar
        # Entity hierarchy
        # Property inspector
        # Solver parameters
        # Profiler
        # Console
        pass
    
    def _cleanup(self) -> None:
        """Cleanup resources."""
        pass
    
    def set_camera(self, position: tuple = None, target: tuple = None, fov: float = None) -> None:
        if position: self.camera.position = position
        if target: self.camera.target = target
        if fov: self.camera.fov = fov
    
    def toggle_pause(self) -> None:
        self.paused = not self.paused
    
    def step_once(self) -> None:
        if self.paused:
            self.scene.step()
    
    def select_entity(self, entity_id) -> None:
        self.selected_entity = entity_id
    
    def screenshot(self, path: str) -> None:
        """Save screenshot."""
        pass


class CameraController:
    """Orbit camera controller."""
    
    def __init__(self, camera: CameraConfig):
        self.camera = camera
        self.distance = 10.0
        self.yaw = 45.0
        self.pitch = -30.0
    
    def update(self, dx: float, dy: float, dz: float, buttons: dict) -> None:
        """Update camera from mouse input."""
        if buttons.get("left"):
            self.yaw += dx * 0.5
            self.pitch = max(-89, min(89, self.pitch - dy * 0.5))
        if buttons.get("right"):
            # Pan
            pass
        if buttons.get("middle") or dz != 0:
            self.distance = max(0.1, self.distance - dz * 0.5)
        
        self._update_camera_position()
    
    def _update_camera_position(self) -> None:
        import math
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        self.camera.position = (
            self.camera.target[0] + self.distance * math.cos(pitch_rad) * math.cos(yaw_rad),
            self.camera.target[1] + self.distance * math.cos(pitch_rad) * math.sin(yaw_rad),
            self.camera.target[2] + self.distance * math.sin(pitch_rad),
        )