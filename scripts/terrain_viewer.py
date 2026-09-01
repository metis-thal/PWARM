"""
PWARM Terrain Evolution Viewer — Dual-Mode (Professional + Layperson)

Controls:
    Mouse drag     — Rotate view
    Mouse scroll   — Zoom in/out
    ← / → arrows  — Switch time snapshots
    ↑ / ↓ arrows  — Switch dataset (Everest/Grand Canyon/Fuji/Zhangjiajie)
    S              — Toggle cross-section view
    M              — Toggle Professional / Layperson mode
    A              — Auto-play animation (t0 → t1M)
    R              — Reset view
    Q / Esc        — Quit

Usage:
    python scripts/terrain_viewer.py
    python scripts/terrain_viewer.py --dataset everest
"""
import os
import sys
import time
import argparse
import numpy as np
from scipy.ndimage import zoom as ndimage_zoom

# Support PyInstaller frozen exe
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = os.path.dirname(sys.executable)
else:
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
        "name": "珠穆朗玛峰",
        "name_en": "Mt. Everest",
        "initial": "data/terrain/terrain_everest.npy",
        "simulation": "data/terrain/everest_simulation",
        "cell_size": 40.0,
        "vertical_exag": 1.0,  # Natural proportions
    },
    "grand_canyon": {
        "name": "大峡谷",
        "name_en": "Grand Canyon",
        "initial": "data/terrain/grand_canyon.npy",
        "simulation": None,
        "cell_size": 40.0,
        "vertical_exag": 1.0,
    },
    "mt_fuji": {
        "name": "富士山",
        "name_en": "Mt. Fuji",
        "initial": "data/terrain/mt_fuji.npy",
        "simulation": None,
        "cell_size": 40.0,
        "vertical_exag": 1.0,
    },
    "zhangjiajie": {
        "name": "张家界",
        "name_en": "Zhangjiajie",
        "initial": "data/terrain/zhangjiajie.npy",
        "simulation": None,
        "cell_size": 40.0,
        "vertical_exag": 1.0,
    },
}


# ============================================================
# Realistic terrain coloring — based on ELEVATION + SLOPE
# Real mountains: vegetation on flat ground, bare rock on steep slopes,
# snow on high peaks. Not a simple elevation-only colormap.
# ============================================================

def compute_terrain_colors(elev_smooth, cell_size):
    """Compute per-vertex RGB colors based on elevation AND slope.
    
    This produces realistic mountain appearance:
    - Flat lowlands: dark green (forest/vegetation)
    - Gentle slopes: lighter green (grassland)
    - Steep slopes: brown/grey (exposed rock)
    - Very steep: dark grey (cliff faces)
    - High peaks: white (snow)
    - High slopes: light grey (alpine rock)
    """
    ny, nx = elev_smooth.shape

    # Compute slope (gradient magnitude) in radians
    spacing = cell_size / 4.0  # 4x upsample spacing
    dy, dx = np.gradient(elev_smooth, spacing)
    slope_rad = np.sqrt(dx**2 + dy**2)
    slope_deg = np.degrees(slope_rad)

    # Normalize elevation to [0, 1]
    e_min, e_max = elev_smooth.min(), elev_smooth.max()
    if e_max > e_min:
        e_norm = (elev_smooth - e_min) / (e_max - e_min)
    else:
        e_norm = np.zeros_like(elev_smooth)

    # Base color from elevation (vegetation zones)
    # Low: dark green → Mid: brown → High: grey → Peak: white
    r_base = np.where(e_norm < 0.3,
                       0.15 + 0.3 * (e_norm / 0.3),      # dark green → olive
              np.where(e_norm < 0.6,
                       0.45 + 0.2 * ((e_norm - 0.3) / 0.3),  # olive → brown
              np.where(e_norm < 0.85,
                       0.65 + 0.15 * ((e_norm - 0.6) / 0.25), # brown → grey
                       0.80 + 0.18 * ((e_norm - 0.85) / 0.15) # grey → white
                       )))
    g_base = np.where(e_norm < 0.3,
                       0.40 + 0.15 * (e_norm / 0.3),      # green → yellow-green
              np.where(e_norm < 0.6,
                       0.55 - 0.15 * ((e_norm - 0.3) / 0.3), # yellow-green → brown
              np.where(e_norm < 0.85,
                       0.40 - 0.1 * ((e_norm - 0.6) / 0.25),  # brown → grey
                       0.78 + 0.2 * ((e_norm - 0.85) / 0.15)  # grey → white
                       )))
    b_base = np.where(e_norm < 0.3,
                       0.15 + 0.05 * (e_norm / 0.3),      # low blue in shadows
              np.where(e_norm < 0.6,
                       0.20 + 0.05 * ((e_norm - 0.3) / 0.3),
              np.where(e_norm < 0.85,
                       0.25 + 0.15 * ((e_norm - 0.6) / 0.25),
                       0.40 + 0.57 * ((e_norm - 0.85) / 0.15)
                       )))

    # Slope effect: steep slopes → more grey/brown (exposed rock)
    # Flatten the base color toward rock colors on steep terrain
    rock_r, rock_g, rock_b = 0.50, 0.45, 0.40  # bare rock color
    slope_factor = np.clip((slope_deg - 15) / 30, 0, 1)  # 15°-45° transition

    # On steep slopes, override vegetation colors with rock
    r = r_base * (1 - slope_factor) + rock_r * slope_factor
    g = g_base * (1 - slope_factor) + rock_g * slope_factor
    b = b_base * (1 - slope_factor) + rock_b * slope_factor

    # Very steep (cliffs > 45°): dark rock
    cliff_factor = np.clip((slope_deg - 45) / 20, 0, 1)
    r = r * (1 - cliff_factor) + 0.30 * cliff_factor
    g = g * (1 - cliff_factor) + 0.28 * cliff_factor
    b = b * (1 - cliff_factor) + 0.25 * cliff_factor

    # High altitude + flat: snow cap
    snow_factor = np.clip((e_norm - 0.8) * 5, 0, 1) * (1 - slope_factor)
    r = r * (1 - snow_factor) + 0.95 * snow_factor
    g = g * (1 - snow_factor) + 0.95 * snow_factor
    b = b * (1 - snow_factor) + 0.98 * snow_factor

    # Add subtle noise for natural variation (not perfectly smooth)
    np.random.seed(42)
    noise = np.random.rand(ny, nx).astype(np.float64) * 0.04 - 0.02
    r = np.clip(r + noise, 0, 1)
    g = np.clip(g + noise, 0, 1)
    b = np.clip(b + noise, 0, 1)

    colors = np.stack([r.ravel(order="F"),
                       g.ravel(order="F"),
                       b.ravel(order="F")], axis=-1).astype(np.float32)
    return colors


# ============================================================
# Data loading
# ============================================================

def load_snapshots(dataset_key):
    """Load all available snapshots for a dataset, sorted by time."""
    ds = DATASETS[dataset_key]
    snapshots = []

    sim_dir = os.path.join(PROJECT_ROOT, ds["simulation"]) if ds["simulation"] else None
    if sim_dir and os.path.exists(sim_dir):
        npy_files = [f for f in os.listdir(sim_dir)
                     if f.startswith("elevation_t") and f.endswith(".npy")]
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
            snapshots.append({"key": t_str, "year": yr, "label": label,
                              "elevation": elev})

    if not snapshots:
        init_path = os.path.join(PROJECT_ROOT, ds["initial"])
        if os.path.exists(init_path):
            elev = np.load(init_path)
            snapshots.append({"key": "t0", "year": 0, "label": "现在",
                              "elevation": elev})

    return snapshots


# ============================================================
# Mesh creation — smooth, realistic like a 3D model
# ============================================================

def create_terrain_mesh(elevation, cell_size, vertical_exag=1.0):
    """Create a smooth PyVista StructuredGrid with realistic slope-based coloring.
       Uses 4x upsample for smooth surface (128->512 = 262K verts).
    """
    factor = 4
    elev_smooth = ndimage_zoom(elevation, factor, order=3).astype(np.float64)
    ny, nx = elev_smooth.shape
    smooth_cell = cell_size / factor

    x = np.arange(nx) * smooth_cell
    y = np.arange(ny) * smooth_cell
    xx, yy = np.meshgrid(x, y)
    zz = elev_smooth * vertical_exag

    grid = pv.StructuredGrid(xx, yy, zz)

    # Compute realistic colors based on elevation + slope
    colors = compute_terrain_colors(elev_smooth, cell_size)
    grid["RGB"] = colors

    return grid


def create_cross_section_wall(elevation, cell_size, vertical_exag=1.0,
                              slice_x=None):
    """Create a visible cross-section wall showing underground layers."""
    ny, nx = elevation.shape
    if slice_x is None:
        slice_x = nx // 2

    # Use 4x upsampled data for smooth wall
    factor = 4
    elev_smooth = ndimage_zoom(elevation, factor, order=3).astype(np.float64)
    sny, snx = elev_smooth.shape
    smooth_cell = cell_size / factor

    # Get elevation profile along the slice
    si_x = int(slice_x * factor)
    si_x = max(0, min(snx - 1, si_x))
    profile = elev_smooth[:, si_x]

    # Create wall points (terrain surface + deep base)
    base_depth = profile.min() - (profile.max() - profile.min()) * 0.5
    y_coords = np.arange(sny) * smooth_cell

    x_val = slice_x * cell_size
    points_top = np.column_stack([
        np.full(sny, x_val), y_coords, profile * vertical_exag,
    ])
    points_bot = np.column_stack([
        np.full(sny, x_val), y_coords,
        np.full(sny, base_depth * vertical_exag),
    ])

    # Build faces (quads)
    faces = []
    for i in range(sny - 1):
        faces.extend([4, i, i + 1, sny + i + 1, sny + i])
    faces = np.array(faces, dtype=np.int64)

    points = np.vstack([points_top, points_bot])
    wall = pv.PolyData(points, faces)

    # Color wall with geological layers
    n_pts = 2 * sny
    wall_colors = np.zeros((n_pts, 3), dtype=np.float32)
    for i in range(sny):
        t = i / max(sny - 1, 1)  # 0=base, 1=surface
        # Dark brown (deep) → tan (surface) with slight variation
        base_color = [0.30 + 0.25 * t, 0.22 + 0.18 * t, 0.12 + 0.12 * t]
        wall_colors[i] = base_color
        wall_colors[sny + i] = base_color
    wall["RGB"] = wall_colors

    return wall


# ============================================================
# Layperson text for each time period
# ============================================================

LAYPERSON_INFO = {
    0: (
        "【初始地形】\n"
        "高山区域：岩石和土壤，等待被雨水冲刷\n"
        "山谷洼地：将接收从高处冲下来的泥沙"
    ),
    100: (
        "【100 年】\n"
        "变化非常微弱，肉眼几乎看不出差异\n"
        "但雨水和风化已经悄悄开始工作"
    ),
    200: (
        "【200 年】\n"
        "少量泥土从陡峭山坡被冲走\n"
        "慢慢沉积到谷底"
    ),
    500: (
        "【500 年】\n"
        "山坡表层土壤在持续流失\n"
        "山谷底部逐渐有泥沙堆积"
    ),
    1000: (
        "【1000 年】\n"
        "短时间尺度变化微弱\n"
        "但侵蚀和搬运一直在持续进行"
    ),
    5000: (
        "【5000 年】\n"
        "山脊棱角开始被磨圆\n"
        "山谷河道逐渐加深"
    ),
    10000: (
        "【1万 年】\n"
        "侵蚀效果明显了\n"
        "山峰变矮，山谷变深，物质在重新分布"
    ),
    50000: (
        "【5万 年】\n"
        "大量高山岩石被风化剥蚀\n"
        "碎屑顺着山坡搬运到低处堆积"
    ),
    100000: (
        "【10万 年】\n"
        "地貌已经发生显著改变\n"
        "高处不断损失物质，低处不断接收"
    ),
    500000: (
        "【50万 年】\n"
        "大规模物质迁移：\n"
        "山体被大幅削低，谷底被大量填高"
    ),
    1000000: (
        "【100万 年】\n"
        "百万年的侵蚀搬运\n"
        "彻底改写了地貌：高处削平、低处填满"
    ),
}


def get_layperson_text(year):
    """Get layperson description for a given year."""
    best_key = 0
    for k in LAYPERSON_INFO:
        if k <= year:
            best_key = k
    return LAYPERSON_INFO[best_key]


# ============================================================
# Main Viewer Class
# ============================================================

class TerrainViewer:
    """Interactive 3D terrain viewer with dual-mode UI."""

    def __init__(self, dataset_key="everest"):
        self.dataset_key = dataset_key
        self.dataset = DATASETS[dataset_key]
        self.snapshots = load_snapshots(dataset_key)
        self.current_idx = 0
        self.show_cross_section = False
        self.professional_mode = True
        self.auto_play = False
        self.auto_play_timer = 0.0

        # PyVista plotter — OFF_SCREEN=False for interactive window
        self.plotter = pv.Plotter(
            window_size=[1400, 900],
            title="PWARM Terrain Evolution Viewer",
            off_screen=False,
        )
        self.plotter.set_background("#0d1117")
        self.plotter.add_axes(
            xlabel="X (m)", ylabel="Y (m)", zlabel="Elevation (m)",
            line_width=2, color="white"
        )

        # Enable lighting for realistic rendering
        self.plotter.enable_lightkit()

        # Actors
        self.terrain_actor = None
        self.wall_actor = None
        self.text_actor = None
        self.legend_actor = None
        self.controls_actor = None

        # Initial render
        self._update_terrain()
        self._update_ui()

        # Keyboard bindings
        self.plotter.add_key_event("Left", self._prev_snapshot)
        self.plotter.add_key_event("Right", self._next_snapshot)
        self.plotter.add_key_event("Up", self._prev_dataset)
        self.plotter.add_key_event("Down", self._next_dataset)
        self.plotter.add_key_event("s", self._toggle_cross_section)
        self.plotter.add_key_event("m", self._toggle_mode)
        self.plotter.add_key_event("a", self._toggle_auto_play)
        self.plotter.add_key_event("r", lambda: self.plotter.reset_camera())
        self.plotter.add_key_event("q", lambda: self.plotter.close())
        self.plotter.add_key_event("Escape", lambda: self.plotter.close())

    # ----------------------------------------------------------
    # Terrain rendering
    # ----------------------------------------------------------

    def _update_terrain(self):
        """Rebuild and display terrain mesh."""
        snap = self.snapshots[self.current_idx]
        elev = snap["elevation"]

        # Remove old actors
        if self.terrain_actor is not None:
            self.plotter.remove_actor(self.terrain_actor)
        if self.wall_actor is not None:
            self.plotter.remove_actor(self.wall_actor)

        # Create smooth mesh with realistic slope-based coloring
        mesh = create_terrain_mesh(
            elev, self.dataset["cell_size"], self.dataset["vertical_exag"]
        )

        # Render: RGB coloring with smooth shading for realistic appearance
        self.terrain_actor = self.plotter.add_mesh(
            mesh,
            scalars="RGB",
            rgb=True,
            smooth_shading=True,
            specular=0.15,
            specular_power=15,
            ambient=0.25,
            diffuse=0.75,
            show_scalar_bar=False,
            lighting=True,
        )

        # Cross-section
        if self.show_cross_section:
            self._update_cross_section(elev)

        self.plotter.reset_camera()

    def _update_cross_section(self, elev):
        """Add cross-section wall."""
        if self.wall_actor is not None:
            self.plotter.remove_actor(self.wall_actor)

        wall = create_cross_section_wall(
            elev, self.dataset["cell_size"], self.dataset["vertical_exag"]
        )
        self.wall_actor = self.plotter.add_mesh(
            wall,
            scalars="RGB",
            rgb=True,
            smooth_shading=True,
            show_edges=False,
            opacity=0.95,
        )

    # ----------------------------------------------------------
    # UI text — dual mode
    # ----------------------------------------------------------

    def _update_ui(self):
        """Update all overlay text based on current mode."""
        if self.text_actor is not None:
            self.plotter.remove_actor(self.text_actor)
        if self.legend_actor is not None:
            self.plotter.remove_actor(self.legend_actor)
        if self.controls_actor is not None:
            self.plotter.remove_actor(self.controls_actor)

        if self.professional_mode:
            self._draw_professional_ui()
        else:
            self._draw_layperson_ui()

    def _draw_professional_ui(self):
        """Professional mode: full technical info."""
        snap = self.snapshots[self.current_idx]
        elev = snap["elevation"]
        ny, nx = elev.shape
        ds_name = f"{self.dataset['name']} ({self.dataset['name_en']})"

        info = (
            f"Dataset: {ds_name}\n"
            f"Time: {snap['key']} ({snap['label']})\n"
            f"Grid: {nx}x{nx} | Cell: {self.dataset['cell_size']:.0f}m\n"
            f"Max: {elev.max():.1f}m | Min: {elev.min():.1f}m | Mean: {elev.mean():.1f}m\n"
            f"Snapshot: {self.current_idx + 1}/{len(self.snapshots)}"
        )
        self.text_actor = self.plotter.add_text(
            info, position="upper_left", font_size=12, color="white"
        )

        controls = (
            "Controls:\n"
            "  <-/-> : Switch time\n"
            "  Up/Down : Switch dataset\n"
            "  S : Cross-section\n"
            "  M : Toggle mode\n"
            "  A : Auto-play\n"
            "  R : Reset view\n"
            "  Q : Quit"
        )
        self.controls_actor = self.plotter.add_text(
            controls, position="upper_right", font_size=11, color="white"
        )

    def _draw_layperson_ui(self):
        """Layperson mode: plain language, big text."""
        snap = self.snapshots[self.current_idx]
        ds_name = self.dataset['name']

        info = f"\u3010{ds_name}\u3011\u65f6\u95f4\uff1a{snap['label']}\n"
        info += get_layperson_text(snap['year'])

        self.text_actor = self.plotter.add_text(
            info, position="upper_left", font_size=14, color="white"
        )

        # Bottom legend — emoji-free for font compat
        legend = (
            "[Low] Valleys: mud & sand settle here"
            "   [Mid] Slopes: soil washed away"
            "   [High] Peaks: bare rock weathering"
        )
        self.legend_actor = self.plotter.add_text(
            legend, position="lower_left", font_size=11, color="#cccccc"
        )

        controls = (
            "Left/Right  - Time forward/back\n"
            "Up/Down     - Switch landscape\n"
            "S           - Cut mountain, see layers\n"
            "M           - Pro / Simple mode\n"
            "A           - Auto-play evolution\n"
            "R           - Reset view\n"
            "Q           - Quit\n"
            "Mouse drag = rotate | Scroll = zoom"
        )
        self.controls_actor = self.plotter.add_text(
            controls, position="upper_right", font_size=11, color="#dddddd"
        )

    # ----------------------------------------------------------
    # Navigation
    # ----------------------------------------------------------

    def _next_snapshot(self):
        if self.current_idx < len(self.snapshots) - 1:
            self.current_idx += 1
            self._update_terrain()
            self._update_ui()

    def _prev_snapshot(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_terrain()
            self._update_ui()

    def _next_dataset(self):
        keys = list(DATASETS.keys())
        idx = (keys.index(self.dataset_key) + 1) % len(keys)
        self._switch_dataset(keys[idx])

    def _prev_dataset(self):
        keys = list(DATASETS.keys())
        idx = (keys.index(self.dataset_key) - 1) % len(keys)
        self._switch_dataset(keys[idx])

    def _switch_dataset(self, key):
        self.dataset_key = key
        self.dataset = DATASETS[key]
        self.snapshots = load_snapshots(key)
        self.current_idx = 0
        self._update_terrain()
        self._update_ui()

    def _toggle_cross_section(self):
        self.show_cross_section = not self.show_cross_section
        elev = self.snapshots[self.current_idx]["elevation"]
        if self.show_cross_section:
            self._update_cross_section(elev)
        else:
            if self.wall_actor is not None:
                self.plotter.remove_actor(self.wall_actor)
                self.wall_actor = None

    def _toggle_mode(self):
        self.professional_mode = not self.professional_mode
        self._update_ui()

    def _toggle_auto_play(self):
        self.auto_play = not self.auto_play
        self.auto_play_timer = time.time()

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------

    def _auto_play_tick(self):
        """Advance auto-play if enabled."""
        if self.auto_play:
            now = time.time()
            if now - self.auto_play_timer > 1.5:
                if self.current_idx < len(self.snapshots) - 1:
                    self.current_idx += 1
                    self._update_terrain()
                    self._update_ui()
                else:
                    self.auto_play = False
                self.auto_play_timer = now

    def show(self):
        """Run the interactive viewer."""
        self.plotter.show()


def main():
    parser = argparse.ArgumentParser(description="PWARM Terrain Viewer")
    parser.add_argument("--dataset", "-d", default="everest",
                        choices=list(DATASETS.keys()))
    args = parser.parse_args()

    viewer = TerrainViewer(args.dataset)
    viewer.show()


if __name__ == "__main__":
    main()
