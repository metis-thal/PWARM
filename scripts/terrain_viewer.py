"""
Interactive 3D Terrain Evolution Viewer

Controls:
    Mouse drag     - Rotate view
    Mouse scroll   - Zoom in/out
    ← / → arrows  - Switch time snapshots (t0, t100, ..., t1000)
    ↑ / ↓ arrows  - Switch dataset (Everest/Grand Canyon/Fuji/Zhangjiajie)
    S              - Toggle cross-section view
    R              - Reset view
    Q / Esc        - Quit

Usage:
    python scripts/terrain_viewer.py
    python scripts/terrain_viewer.py --dataset everest
    python scripts/terrain_viewer.py --data-dir data/terrain/everest_simulation
"""
import os
import sys
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

try:
    import pyvista as pv
except ImportError:
    print("ERROR: PyVista not installed. Run: pip install pyvista")
    sys.exit(1)


# ============================================================
# Dataset definitions
# ============================================================

DATASETS = {
    "everest": {
        "name": "珠穆朗玛峰 (Mt. Everest)",
        "initial": "data/terrain/terrain_everest.npy",
        "simulation": "data/terrain/everest_simulation",
        "cell_size": 40.0,
        "vertical_exag": 2.0,
    },
    "grand_canyon": {
        "name": "大峡谷 (Grand Canyon)",
        "initial": "data/terrain/grand_canyon.npy",
        "simulation": None,  # no simulation yet
        "cell_size": 40.0,
        "vertical_exag": 5.0,  # canyon needs more exaggeration
    },
    "mt_fuji": {
        "name": "富士山 (Mt. Fuji)",
        "initial": "data/terrain/mt_fuji.npy",
        "simulation": None,
        "cell_size": 40.0,
        "vertical_exag": 2.0,
    },
    "zhangjiajie": {
        "name": "张家界 (Zhangjiajie)",
        "initial": "data/terrain/zhangjiajie.npy",
        "simulation": None,
        "cell_size": 40.0,
        "vertical_exag": 3.0,
    },
}


def load_snapshots(dataset_key):
    """Load all available snapshots for a dataset."""
    ds = DATASETS[dataset_key]
    snapshots = []

    # Load simulation snapshots (from simulation directory)
    sim_dir = os.path.join(PROJECT_ROOT, ds["simulation"]) if ds["simulation"] else None
    if sim_dir and os.path.exists(sim_dir):
        npy_files = [f for f in os.listdir(sim_dir)
                     if f.startswith("elevation_t") and f.endswith(".npy")]
        # Sort by year number, not alphabetically
        def _year(path):
            t = path.replace("elevation_", "").replace(".npy", "")
            return int(t.replace("t", ""))
        for f in sorted(npy_files, key=_year):
            t_str = f.replace("elevation_", "").replace(".npy", "")
            yr = _year(f)
            if yr >= 1_000_000:
                label = f"{yr/1_000_000:.0f}M年"
            elif yr >= 1_000:
                label = f"{yr/1_000:.0f}K年"
            else:
                label = f"{yr}年"
            elev = np.load(os.path.join(sim_dir, f))
            snapshots.append((f"{t_str} ({label})", elev))

    # Fallback: if no simulation dir, load raw heightmap
    if not snapshots:
        init_path = os.path.join(PROJECT_ROOT, ds["initial"])
        if os.path.exists(init_path):
            elev = np.load(init_path)
            snapshots.append(("t0 (现在)", elev))

    return snapshots


def create_terrain_mesh(elevation, cell_size, vertical_exag=1.0):
    """Create PyVista mesh from heightmap."""
    ny, nx = elevation.shape

    # Create coordinate grid
    x = np.arange(nx) * cell_size
    y = np.arange(ny) * cell_size
    xx, yy = np.meshgrid(x, y)

    # Apply vertical exaggeration
    zz = elevation * vertical_exag

    # Create structured grid
    grid = pv.StructuredGrid(xx, yy, zz)
    grid["Elevation"] = elevation.ravel(order="F")

    return grid


def create_cross_section(mesh, elevation, cell_size, vertical_exag=1.0,
                         slice_pos=None):
    """Create cross-section through the terrain."""
    ny, nx = elevation.shape

    if slice_pos is None:
        slice_pos = nx * cell_size / 2  # middle

    # Slice along X axis
    sliced = mesh.slice(normal="x", origin=(slice_pos, 0, 0))

    # Create a wall beneath the slice
    x = slice_pos
    y = np.arange(ny) * cell_size
    z_bottom = np.full_like(y, elevation.min() * vertical_exag - 50)

    # Get elevation along slice
    iy_slice = int(slice_pos / cell_size)
    iy_slice = max(0, min(ny - 1, iy_slice))
    elev_slice = elevation[:, iy_slice] if iy_slice < nx else elevation[:, nx // 2]

    # Create wall points
    points = []
    for i in range(ny):
        points.append([x, y[i], elev_slice[i] * vertical_exag])
    for i in range(ny - 1, -1, -1):
        points.append([x, y[i], z_bottom[i]])

    points = np.array(points)
    faces = []
    for i in range(ny - 1):
        faces.append([4, i, i + 1, 2 * ny - 2 - i, 2 * ny - 1 - i])
    faces = np.array(faces)

    wall = pv.PolyData(points, faces)
    wall["Elevation"] = np.tile(elev_slice, 2)[:len(points)]

    return wall


class TerrainViewer:
    """Interactive 3D terrain viewer."""

    def __init__(self, dataset_key="everest"):
        self.dataset_key = dataset_key
        self.dataset = DATASETS[dataset_key]
        self.snapshots = load_snapshots(dataset_key)
        self.current_idx = 0
        self.show_cross_section = False
        self.cross_section_pos = None

        # Create plotter
        self.plotter = pv.Plotter(
            window_size=[1400, 900],
            title=f"PWARM 地形演化查看器 - {self.dataset['name']}",
        )
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(
            xlabel="X (m)", ylabel="Y (m)", zlabel="Elevation (m)",
            line_width=2
        )

        # Add initial terrain
        self.terrain_actor = None
        self.wall_actor = None
        self.label_actor = None
        self.info_actor = None
        self.update_terrain()

        # Add controls info
        self.add_controls_text()

        # Bind keyboard
        self.plotter.add_key_event("Left", self.prev_snapshot)
        self.plotter.add_key_event("Right", self.next_snapshot)
        self.plotter.add_key_event("Up", self.prev_dataset)
        self.plotter.add_key_event("Down", self.next_dataset)
        self.plotter.add_key_event("s", self.toggle_cross_section)
        self.plotter.add_key_event("r", self.reset_view)
        self.plotter.add_key_event("q", lambda: self.plotter.close())
        self.plotter.add_key_event("Escape", lambda: self.plotter.close())

    def add_controls_text(self):
        """Add control instructions overlay."""
        controls = (
            "Controls:\n"
            "  ←/→ : Switch time\n"
            "  ↑/↓ : Switch dataset\n"
            "  S : Cross-section\n"
            "  R : Reset view\n"
            "  Q : Quit"
        )
        self.plotter.add_text(
            controls, position="upper_right",
            font_size=11, color="white",
            font_family="courier",
            background_color=(0.1, 0.1, 0.2, 0.8),
        )

    def update_terrain(self):
        """Update the displayed terrain."""
        label, elev = self.snapshots[self.current_idx]

        # Remove old actors
        if self.terrain_actor is not None:
            self.plotter.remove_actor(self.terrain_actor)
        if self.wall_actor is not None:
            self.plotter.remove_actor(self.wall_actor)
        if self.info_actor is not None:
            self.plotter.remove_actor(self.info_actor)

        # Create mesh
        mesh = create_terrain_mesh(
            elev, self.dataset["cell_size"], self.dataset["vertical_exag"]
        )

        # Add terrain
        self.terrain_actor = self.plotter.add_mesh(
            mesh,
            scalars="Elevation",
            cmap="terrain",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Elevation (m)"},
            lighting=True,
            ambient=0.3,
            diffuse=0.7,
            specular=0.2,
        )

        # Add cross-section if enabled
        if self.show_cross_section:
            self.update_cross_section(elev)

        # Update info text
        ny, nx = elev.shape
        info = (
            f"Dataset: {self.dataset['name']}\n"
            f"Time: {label}\n"
            f"Grid: {nx}×{nx} | Cell: {self.dataset['cell_size']}m\n"
            f"Max: {elev.max():.1f}m | Min: {elev.min():.1f}m | Mean: {elev.mean():.1f}m\n"
            f"Snapshot: {self.current_idx + 1}/{len(self.snapshots)}"
        )
        self.info_actor = self.plotter.add_text(
            info, position="upper_left",
            font_size=11, color="white",
            font_family="courier",
            background_color=(0.1, 0.1, 0.2, 0.8),
        )

        # Reset camera for new dataset
        self.plotter.reset_camera()

    def update_cross_section(self, elev):
        """Update cross-section view."""
        if self.wall_actor is not None:
            self.plotter.remove_actor(self.wall_actor)

        mesh = create_terrain_mesh(
            elev, self.dataset["cell_size"], self.dataset["vertical_exag"]
        )
        wall = create_cross_section(
            mesh, elev, self.dataset["cell_size"], self.dataset["vertical_exag"]
        )

        self.wall_actor = self.plotter.add_mesh(
            wall, scalars="Elevation", cmap="terrain",
            show_edges=True, edge_color="black",
        )

    def next_snapshot(self):
        """Go to next time snapshot."""
        if self.current_idx < len(self.snapshots) - 1:
            self.current_idx += 1
            self.update_terrain()

    def prev_snapshot(self):
        """Go to previous time snapshot."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_terrain()

    def next_dataset(self):
        """Switch to next dataset."""
        keys = list(DATASETS.keys())
        idx = keys.index(self.dataset_key)
        idx = (idx + 1) % len(keys)
        self.switch_dataset(keys[idx])

    def prev_dataset(self):
        """Switch to previous dataset."""
        keys = list(DATASETS.keys())
        idx = keys.index(self.dataset_key)
        idx = (idx - 1) % len(keys)
        self.switch_dataset(keys[idx])

    def switch_dataset(self, key):
        """Switch to a different dataset."""
        self.dataset_key = key
        self.dataset = DATASETS[key]
        self.snapshots = load_snapshots(key)
        self.current_idx = 0
        self.update_terrain()

    def toggle_cross_section(self):
        """Toggle cross-section view."""
        self.show_cross_section = not self.show_cross_section
        elev = self.snapshots[self.current_idx][1]
        if self.show_cross_section:
            self.update_cross_section(elev)
        else:
            if self.wall_actor is not None:
                self.plotter.remove_actor(self.wall_actor)
                self.wall_actor = None

    def reset_view(self):
        """Reset camera to default view."""
        self.plotter.reset_camera()

    def show(self):
        """Show the interactive viewer."""
        self.plotter.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D Terrain Viewer")
    parser.add_argument("--dataset", "-d", default="everest",
                        choices=list(DATASETS.keys()),
                        help="Initial dataset to load")
    args = parser.parse_args()

    viewer = TerrainViewer(args.dataset)
    viewer.show()


if __name__ == "__main__":
    main()
