"""
PWARM Terrain Evolution Viewer — Full Physics Visualization

Controls:
    Mouse drag     — Rotate view
    Mouse scroll   — Zoom in/out
    ← / → arrows  — Switch time snapshots
    ↑ / ↓ arrows  — Switch dataset (Everest/Grand Canyon/Fuji/Zhangjiajie)
    S              — Toggle cross-section view
    M              — Toggle coloring: realistic slope-based ↔ scientific heatmap
    T              — Toggle text: professional ↔ layperson
    R              — Toggle rain
    W              — Toggle snow
    V              — Toggle river lines
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
        "name": "珠穆朗玛峰", "name_en": "Mt. Everest",
        "initial": "data/terrain/terrain_everest.npy",
        "simulation": "data/terrain/everest_simulation",
        "cell_size": 40.0, "vertical_exag": 1.0,
    },
    "grand_canyon": {
        "name": "大峡谷", "name_en": "Grand Canyon",
        "initial": "data/terrain/grand_canyon.npy",
        "simulation": None,
        "cell_size": 40.0, "vertical_exag": 1.0,
    },
    "mt_fuji": {
        "name": "富士山", "name_en": "Mt. Fuji",
        "initial": "data/terrain/mt_fuji.npy",
        "simulation": None,
        "cell_size": 40.0, "vertical_exag": 1.0,
    },
    "zhangjiajie": {
        "name": "张家界", "name_en": "Zhangjiajie",
        "initial": "data/terrain/zhangjiajie.npy",
        "simulation": None,
        "cell_size": 40.0, "vertical_exag": 1.0,
    },
}


# ============================================================
# Terrain coloring — slope-based realistic
# ============================================================

def compute_terrain_colors(elev_smooth, cell_size):
    """Per-vertex RGB based on elevation + slope. Looks like real mountain."""
    ny, nx = elev_smooth.shape
    spacing = cell_size / 4.0
    dy, dx = np.gradient(elev_smooth, spacing)
    slope_deg = np.degrees(np.sqrt(dx**2 + dy**2))

    e_min, e_max = elev_smooth.min(), elev_smooth.max()
    e_norm = (elev_smooth - e_min) / (e_max - e_min) if e_max > e_min else np.zeros_like(elev_smooth)

    # Base color from elevation zones
    r = np.where(e_norm < 0.3, 0.15 + 0.3 * (e_norm / 0.3),
        np.where(e_norm < 0.6, 0.45 + 0.2 * ((e_norm - 0.3) / 0.3),
        np.where(e_norm < 0.85, 0.65 + 0.15 * ((e_norm - 0.6) / 0.25),
                 0.80 + 0.18 * ((e_norm - 0.85) / 0.15))))
    g = np.where(e_norm < 0.3, 0.40 + 0.15 * (e_norm / 0.3),
        np.where(e_norm < 0.6, 0.55 - 0.15 * ((e_norm - 0.3) / 0.3),
        np.where(e_norm < 0.85, 0.40 - 0.1 * ((e_norm - 0.6) / 0.25),
                 0.78 + 0.2 * ((e_norm - 0.85) / 0.15))))
    b = np.where(e_norm < 0.3, 0.15 + 0.05 * (e_norm / 0.3),
        np.where(e_norm < 0.6, 0.20 + 0.05 * ((e_norm - 0.3) / 0.3),
        np.where(e_norm < 0.85, 0.25 + 0.15 * ((e_norm - 0.6) / 0.25),
                 0.40 + 0.57 * ((e_norm - 0.85) / 0.15))))

    # Slope → bare rock override
    sf = np.clip((slope_deg - 15) / 30, 0, 1)
    r = r * (1 - sf) + 0.50 * sf
    g = g * (1 - sf) + 0.45 * sf
    b = b * (1 - sf) + 0.40 * sf

    # Cliff override
    cf = np.clip((slope_deg - 45) / 20, 0, 1)
    r = r * (1 - cf) + 0.30 * cf
    g = g * (1 - cf) + 0.28 * cf
    b = b * (1 - cf) + 0.25 * cf

    # Snow cap
    snow = np.clip((e_norm - 0.8) * 5, 0, 1) * (1 - sf)
    r = r * (1 - snow) + 0.95 * snow
    g = g * (1 - snow) + 0.95 * snow
    b = b * (1 - snow) + 0.98 * snow

    # Natural noise
    np.random.seed(42)
    noise = np.random.rand(ny, nx) * 0.04 - 0.02
    r = np.clip(r + noise, 0, 1)
    g = np.clip(g + noise, 0, 1)
    b = np.clip(b + noise, 0, 1)

    return np.stack([r.ravel(order="F"), g.ravel(order="F"),
                     b.ravel(order="F")], axis=-1).astype(np.float32)


def compute_heatmap_colors(elev_smooth):
    """Scientific rainbow heatmap — classic research visualization."""
    e_min, e_max = elev_smooth.min(), elev_smooth.max()
    t = (elev_smooth - e_min) / (e_max - e_min) if e_max > e_min else np.zeros_like(elev_smooth)
    t = np.clip(t, 0, 1)

    # Rainbow: blue → cyan → green → yellow → red
    r = np.where(t < 0.25, 0.0,
        np.where(t < 0.5, (t - 0.25) * 4,
        np.where(t < 0.75, 1.0,
                 1.0 - (t - 0.75) * 4)))
    g = np.where(t < 0.25, t * 4,
        np.where(t < 0.5, 1.0,
        np.where(t < 0.75, 1.0 - (t - 0.5) * 4, 0.0)))
    b = np.where(t < 0.25, 1.0 - t * 4,
        np.where(t < 0.5, 0.0,
        np.where(t < 0.75, 0.0, (t - 0.75) * 4)))

    return np.stack([r.ravel(order="F"), g.ravel(order="F"),
                     b.ravel(order="F")], axis=-1).astype(np.float32)


# ============================================================
# River extraction from D8 flow accumulation
# ============================================================

def compute_flow_accumulation(elev, cell_size):
    """D8 flow accumulation — returns drainage area per cell."""
    ny, nx = elev.shape
    acc = np.ones((ny, nx), dtype=np.float64)

    # D8 neighbor offsets and distances
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]
    dists = [cell_size * 1.414, cell_size, cell_size * 1.414, cell_size,
             cell_size, cell_size * 1.414, cell_size, cell_size * 1.414]

    # Process cells from highest to lowest elevation
    flat_elev = elev.ravel()
    order = np.argsort(-flat_elev)

    for idx in order:
        iy, ix = divmod(idx, nx)
        max_slope = 0
        best_ny, best_nx = iy, ix
        for (dy, dx), d in zip(offsets, dists):
            ny2, nx2 = iy + dy, ix + dx
            if 0 <= ny2 < ny and 0 <= nx2 < nx:
                drop = elev[iy, ix] - elev[ny2, nx2]
                slope = drop / d if d > 0 else 0
                if slope > max_slope:
                    max_slope = slope
                    best_ny, best_nx = ny2, nx2
        if max_slope > 0:
            acc[best_ny, best_nx] += acc[iy, ix]

    return acc


def extract_river_lines(elev, cell_size, threshold_ratio=0.02):
    """Extract river paths from flow accumulation. Returns list of polyline points."""
    acc = compute_flow_accumulation(elev, cell_size)
    threshold = np.percentile(acc, 100 - threshold_ratio * 100)
    river_mask = acc > threshold

    # Extract connected river segments as lines
    ny, nx = elev.shape
    lines = []
    visited = np.zeros_like(river_mask)

    for iy in range(ny):
        for ix in range(nx):
            if river_mask[iy, ix] and not visited[iy, ix]:
                # Trace downstream
                segment = [(ix * cell_size, iy * cell_size, elev[iy, ix])]
                visited[iy, ix] = True
                cy, cx = iy, ix
                while True:
                    # Find steepest downhill neighbor
                    max_drop = 0
                    best = None
                    for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                                   (0, 1), (1, -1), (1, 0), (1, 1)]:
                        ny2, nx2 = cy + dy, cx + dx
                        if 0 <= ny2 < ny and 0 <= nx2 < nx:
                            drop = elev[cy, cx] - elev[ny2, nx2]
                            if drop > max_drop and river_mask[ny2, nx2]:
                                max_drop = drop
                                best = (ny2, nx2)
                    if best is None:
                        break
                    cy, cx = best
                    if visited[cy, cx]:
                        break
                    visited[cy, cx] = True
                    segment.append((cx * cell_size, cy * cell_size, elev[cy, cx]))
                if len(segment) >= 2:
                    lines.append(segment)

    return lines, acc


def create_river_mesh(elev, cell_size, vertical_exag=1.0, threshold_ratio=0.02):
    """Create blue tube meshes for rivers."""
    lines, acc = compute_flow_accumulation(elev, cell_size)
    threshold = np.percentile(acc, 100 - threshold_ratio * 100)
    river_mask = acc > threshold

    # Create a single PolyData with all river points as tubes
    all_points = []
    ny, nx = elev.shape

    # Create tubes along river cells
    tubes = []
    for iy in range(ny):
        for ix in range(nx):
            if river_mask[iy, ix]:
                x = ix * cell_size
                y = iy * cell_size
                z = elev[iy, ix] * vertical_exag
                # Small sphere at each river cell
                sphere = pv.Sphere(radius=cell_size * 0.3,
                                    center=(x, y, z))
                tubes.append(sphere)

    if not tubes:
        return None

    combined = tubes[0]
    for t in tubes[1:]:
        combined = combined.merge(t)

    # Color blue
    n = combined.n_points
    colors = np.zeros((n, 3), dtype=np.float32)
    colors[:, 0] = 0.15  # R
    colors[:, 1] = 0.40  # G
    colors[:, 2] = 0.80  # B
    combined["RGB"] = colors
    return combined


# ============================================================
# Weather particles — rain and snow
# ============================================================

class WeatherSystem:
    """Animated rain and snow particles."""

    def __init__(self, plotter, x_range, y_range, z_range):
        self.plotter = plotter
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.rain_on = False
        self.snow_on = False
        self.rain_actor = None
        self.snow_actor = None
        self.rain_points = None
        self.snow_points = None
        self.rain_velocities = None
        self.snow_velocities = None

        self.n_rain = 2000
        self.n_snow = 1500

    def _init_rain(self):
        """Create rain drop points above terrain."""
        x = np.random.uniform(self.x_range[0], self.x_range[1], self.n_rain)
        y = np.random.uniform(self.y_range[0], self.y_range[1], self.n_rain)
        z = np.random.uniform(self.z_range[1] * 0.3, self.z_range[1], self.n_rain)
        self.rain_points = np.column_stack([x, y, z])
        self.rain_velocities = np.full(self.n_rain, -80.0)  # fall speed m/s

        mesh = pv.PolyData(self.rain_points.copy())
        self.rain_actor = self.plotter.add_mesh(
            mesh, color=[0.5, 0.6, 0.85], point_size=2,
            render_points_as_spheres=False, lighting=False,
            pickable=False,
        )

    def _init_snow(self):
        """Create snowflake points above terrain."""
        x = np.random.uniform(self.x_range[0], self.x_range[1], self.n_snow)
        y = np.random.uniform(self.y_range[0], self.y_range[1], self.n_snow)
        z = np.random.uniform(self.z_range[1] * 0.3, self.z_range[1], self.n_snow)
        self.snow_points = np.column_stack([x, y, z])
        # Snow: slow fall + lateral drift
        self.snow_velocities = np.column_stack([
            np.full(self.n_snow, -5.0),   # x drift
            np.full(self.n_snow, -3.0),   # y drift
            np.full(self.n_snow, -15.0),  # z fall (slow)
        ])

        mesh = pv.PolyData(self.snow_points.copy())
        self.snow_actor = self.plotter.add_mesh(
            mesh, color=[0.95, 0.95, 1.0], point_size=3,
            render_points_as_spheres=True, lighting=False,
            pickable=False,
        )

    def toggle_rain(self):
        if self.rain_on:
            # Turn off
            if self.rain_actor is not None:
                self.plotter.remove_actor(self.rain_actor)
                self.rain_actor = None
            self.rain_on = False
        else:
            self._init_rain()
            self.rain_on = True
        return self.rain_on

    def toggle_snow(self):
        if self.snow_on:
            if self.snow_actor is not None:
                self.plotter.remove_actor(self.snow_actor)
                self.snow_actor = None
            self.snow_on = False
        else:
            self._init_snow()
            self.snow_on = True
        return self.snow_on

    def update(self, dt):
        """Update particle positions each frame. dt in seconds."""
        if self.rain_on and self.rain_points is not None:
            self.rain_points[:, 2] += self.rain_velocities * dt
            # Respawn at top when below terrain
            below = self.rain_points[:, 2] < self.z_range[0]
            self.rain_points[below, 0] = np.random.uniform(
                self.x_range[0], self.x_range[1], below.sum())
            self.rain_points[below, 1] = np.random.uniform(
                self.y_range[0], self.y_range[1], below.sum())
            self.rain_points[below, 2] = self.z_range[1]
            # Update mesh
            if self.rain_actor is not None:
                mesh = pv.PolyData(self.rain_points.copy())
                self.plotter.remove_actor(self.rain_actor)
                self.rain_actor = self.plotter.add_mesh(
                    mesh, color=[0.5, 0.6, 0.85], point_size=2,
                    render_points_as_spheres=False, lighting=False,
                    pickable=False,
                )

        if self.snow_on and self.snow_points is not None:
            self.snow_points += self.snow_velocities * dt
            # Add turbulence
            self.snow_points[:, 0] += np.random.randn(self.n_snow) * 0.5
            self.snow_points[:, 1] += np.random.randn(self.n_snow) * 0.5
            # Respawn at top
            below = self.snow_points[:, 2] < self.z_range[0]
            self.snow_points[below, 0] = np.random.uniform(
                self.x_range[0], self.x_range[1], below.sum())
            self.snow_points[below, 1] = np.random.uniform(
                self.y_range[0], self.y_range[1], below.sum())
            self.snow_points[below, 2] = self.z_range[1]
            if self.snow_actor is not None:
                mesh = pv.PolyData(self.snow_points.copy())
                self.plotter.remove_actor(self.snow_actor)
                self.snow_actor = self.plotter.add_mesh(
                    mesh, color=[0.95, 0.95, 1.0], point_size=3,
                    render_points_as_spheres=True, lighting=False,
                    pickable=False,
                )


# ============================================================
# Mesh creation
# ============================================================

def create_terrain_mesh(elevation, cell_size, vertical_exag=1.0):
    """4x upsampled StructuredGrid with RGB colors."""
    factor = 4
    elev_smooth = ndimage_zoom(elevation, factor, order=3).astype(np.float64)
    ny, nx = elev_smooth.shape
    smooth_cell = cell_size / factor
    x = np.arange(nx) * smooth_cell
    y = np.arange(ny) * smooth_cell
    xx, yy = np.meshgrid(x, y)
    zz = elev_smooth * vertical_exag
    grid = pv.StructuredGrid(xx, yy, zz)
    return grid, elev_smooth


def create_cross_section_wall(elevation, cell_size, vertical_exag=1.0, slice_x=None):
    """Underground layer wall for cross-section view."""
    ny, nx = elevation.shape
    if slice_x is None:
        slice_x = nx // 2
    factor = 4
    elev_smooth = ndimage_zoom(elevation, factor, order=3).astype(np.float64)
    sny, snx = elev_smooth.shape
    smooth_cell = cell_size / factor
    si_x = min(int(slice_x * factor), snx - 1)
    profile = elev_smooth[:, si_x]
    base_depth = profile.min() - (profile.max() - profile.min()) * 0.5
    y_coords = np.arange(sny) * smooth_cell
    x_val = slice_x * cell_size
    pts_top = np.column_stack([np.full(sny, x_val), y_coords, profile * vertical_exag])
    pts_bot = np.column_stack([np.full(sny, x_val), y_coords, np.full(sny, base_depth * vertical_exag)])
    faces = []
    for i in range(sny - 1):
        faces.extend([4, i, i + 1, sny + i + 1, sny + i])
    points = np.vstack([pts_top, pts_bot])
    wall = pv.PolyData(points, np.array(faces, dtype=np.int64))
    n_pts = 2 * sny
    wall_colors = np.zeros((n_pts, 3), dtype=np.float32)
    for i in range(sny):
        t = i / max(sny - 1, 1)
        c = [0.30 + 0.25 * t, 0.22 + 0.18 * t, 0.12 + 0.12 * t]
        wall_colors[i] = c
        wall_colors[sny + i] = c
    wall["RGB"] = wall_colors
    return wall


# ============================================================
# Layperson text
# ============================================================

LAYPERSON_INFO = {
    0:     "【初始地形】\n高山区域：岩石和土壤，等待被雨水冲刷\n山谷洼地：将接收从高处冲下来的泥沙",
    100:   "【100 年】\n变化非常微弱，肉眼几乎看不出差异\n但雨水和风化已经悄悄开始工作",
    200:   "【200 年】\n少量泥土从陡峭山坡被冲走\n慢慢沉积到谷底",
    500:   "【500 年】\n山坡表层土壤在持续流失\n山谷底部逐渐有泥沙堆积",
    1000:  "【1000 年】\n短时间尺度变化微弱\n但侵蚀和搬运一直在持续进行",
    5000:  "【5000 年】\n山脊棱角开始被磨圆\n山谷河道逐渐加深",
    10000: "【1万 年】\n侵蚀效果明显了\n山峰变矮，山谷变深，物质在重新分布",
    50000: "【5万 年】\n大量高山岩石被风化剥蚀\n碎屑顺着山坡搬运到低处堆积",
    100000:"【10万 年】\n地貌已经发生显著改变\n高处不断损失物质，低处不断接收",
    500000:"【50万 年】\n大规模物质迁移：\n山体被大幅削低，谷底被大量填高",
    1000000:"【100万 年】\n百万年的侵蚀搬运\n彻底改写了地貌：高处削平、低处填满",
}

def get_layperson_text(year):
    best_key = 0
    for k in LAYPERSON_INFO:
        if k <= year:
            best_key = k
    return LAYPERSON_INFO[best_key]


# ============================================================
# Data loading
# ============================================================

def load_snapshots(dataset_key):
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
            snapshots.append({"key": t_str, "year": yr, "label": label, "elevation": elev})
    if not snapshots:
        init_path = os.path.join(PROJECT_ROOT, ds["initial"])
        if os.path.exists(init_path):
            elev = np.load(init_path)
            snapshots.append({"key": "t0", "year": 0, "label": "现在", "elevation": elev})
    return snapshots


# ============================================================
# Main Viewer
# ============================================================

class TerrainViewer:

    def __init__(self, dataset_key="everest"):
        self.dataset_key = dataset_key
        self.dataset = DATASETS[dataset_key]
        self.snapshots = load_snapshots(dataset_key)
        self.current_idx = 0

        # Toggle states
        self.realistic_mode = True    # M: True=slope-based, False=heatmap
        self.professional_mode = False  # T: True=pro text, False=layperson
        self.show_cross_section = False  # S
        self.auto_play = False          # A
        self.auto_play_timer = 0.0

        # PyVista plotter
        self.plotter = pv.Plotter(window_size=[1400, 900],
                                   title="PWARM Terrain Viewer", off_screen=False)
        self.plotter.set_background("#0d1117")
        self.plotter.add_axes(xlabel="X (m)", ylabel="Y (m)", zlabel="Elevation (m)",
                               line_width=2, color="white")
        self.plotter.enable_lightkit()

        # Actors
        self.terrain_actor = None
        self.wall_actor = None
        self.river_actor = None
        self.text_actor = None
        self.legend_actor = None
        self.controls_actor = None

        # Weather
        elev0 = self.snapshots[0]["elevation"]
        cs = self.dataset["cell_size"]
        ve = self.dataset["vertical_exag"]
        x_range = (0, elev0.shape[1] * cs)
        y_range = (0, elev0.shape[0] * cs)
        z_range = (elev0.min() * ve, elev0.max() * ve)
        self.weather = WeatherSystem(self.plotter, x_range, y_range, z_range)

        # Cached upscaled elevation
        self._elev_smooth = None

        # Initial render
        self._update_terrain()
        self._update_ui()

        # Key bindings
        self.plotter.add_key_event("Left", self._prev_snapshot)
        self.plotter.add_key_event("Right", self._next_snapshot)
        self.plotter.add_key_event("Up", self._next_dataset)
        self.plotter.add_key_event("Down", self._prev_dataset_down)
        self.plotter.add_key_event("s", self._toggle_cross_section)
        self.plotter.add_key_event("m", self._toggle_coloring)
        self.plotter.add_key_event("t", self._toggle_text_mode)
        self.plotter.add_key_event("r", self._toggle_rain)
        self.plotter.add_key_event("w", self._toggle_snow)
        self.plotter.add_key_event("v", self._toggle_river)
        self.plotter.add_key_event("a", self._toggle_auto_play)
        self.plotter.add_key_event("r", lambda: self.plotter.reset_camera())
        self.plotter.add_key_event("q", lambda: self.plotter.close())
        self.plotter.add_key_event("Escape", lambda: self.plotter.close())

        # Timer for animation (auto-play + weather particles)
        self._last_frame_time = time.time()
        self.plotter.add_timer_event(max_steps=100000, duration=50,
                                     callback=self._on_timer)

    # ---- terrain ----

    def _update_terrain(self):
        snap = self.snapshots[self.current_idx]
        elev = snap["elevation"]

        if self.terrain_actor is not None:
            self.plotter.remove_actor(self.terrain_actor)
        if self.wall_actor is not None:
            self.plotter.remove_actor(self.wall_actor)
        if self.river_actor is not None:
            self.plotter.remove_actor(self.river_actor)
            self.river_actor = None

        grid, elev_smooth = create_terrain_mesh(
            elev, self.dataset["cell_size"], self.dataset["vertical_exag"])
        self._elev_smooth = elev_smooth

        if self.realistic_mode:
            colors = compute_terrain_colors(elev_smooth, self.dataset["cell_size"])
        else:
            colors = compute_heatmap_colors(elev_smooth)
        grid["RGB"] = colors

        self.terrain_actor = self.plotter.add_mesh(
            grid, scalars="RGB", rgb=True, smooth_shading=True,
            specular=0.15, specular_power=15,
            ambient=0.25, diffuse=0.75,
            show_scalar_bar=False, lighting=True)

        if self.show_cross_section:
            self._update_cross_section(elev)

        # Rebuild river if visible
        if self.river_actor is not None:
            self._build_river()

        self.plotter.reset_camera()

    def _update_cross_section(self, elev):
        if self.wall_actor is not None:
            self.plotter.remove_actor(self.wall_actor)
        wall = create_cross_section_wall(elev, self.dataset["cell_size"],
                                          self.dataset["vertical_exag"])
        self.wall_actor = self.plotter.add_mesh(
            wall, scalars="RGB", rgb=True, smooth_shading=True,
            show_edges=False, opacity=0.95)

    # ---- river ----

    def _build_river(self):
        """Compute and display river tubes."""
        elev = self.snapshots[self.current_idx]["elevation"]
        mesh = create_river_mesh(elev, self.dataset["cell_size"],
                                  self.dataset["vertical_exag"])
        if mesh is not None:
            self.river_actor = self.plotter.add_mesh(
                mesh, scalars="RGB", rgb=True, smooth_shading=True,
                lighting=False, pickable=False)

    def _toggle_river(self):
        if self.river_actor is not None:
            self.plotter.remove_actor(self.river_actor)
            self.river_actor = None
        else:
            self._build_river()

    # ---- weather ----

    def _toggle_rain(self):
        self.weather.toggle_rain()

    def _toggle_snow(self):
        self.weather.toggle_snow()

    # ---- coloring / text mode ----

    def _toggle_coloring(self):
        self.realistic_mode = not self.realistic_mode
        self._update_terrain()

    def _toggle_text_mode(self):
        self.professional_mode = not self.professional_mode
        self._update_ui()

    # ---- cross section ----

    def _toggle_cross_section(self):
        self.show_cross_section = not self.show_cross_section
        elev = self.snapshots[self.current_idx]["elevation"]
        if self.show_cross_section:
            self._update_cross_section(elev)
        else:
            if self.wall_actor is not None:
                self.plotter.remove_actor(self.wall_actor)
                self.wall_actor = None

    # ---- auto play ----

    def _toggle_auto_play(self):
        self.auto_play = not self.auto_play
        self.auto_play_timer = time.time()

    # ---- timer callback ----

    def _on_timer(self):
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        # Auto-play advance
        if self.auto_play:
            if now - self.auto_play_timer > 1.5:
                if self.current_idx < len(self.snapshots) - 1:
                    self.current_idx += 1
                    self._update_terrain()
                    self._update_ui()
                else:
                    self.auto_play = False
                self.auto_play_timer = now

        # Weather particle animation
        self.weather.update(dt)

    # ---- navigation ----

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

    def _prev_dataset_down(self):
        keys = list(DATASETS.keys())
        idx = (keys.index(self.dataset_key) - 1) % len(keys)
        self._switch_dataset(keys[idx])

    def _switch_dataset(self, key):
        self.dataset_key = key
        self.dataset = DATASETS[key]
        self.snapshots = load_snapshots(key)
        self.current_idx = 0
        elev0 = self.snapshots[0]["elevation"]
        cs = self.dataset["cell_size"]
        ve = self.dataset["vertical_exag"]
        self.weather.x_range = (0, elev0.shape[1] * cs)
        self.weather.y_range = (0, elev0.shape[0] * cs)
        self.weather.z_range = (elev0.min() * ve, elev0.max() * ve)
        self._update_terrain()
        self._update_ui()

    # ---- UI text ----

    def _update_ui(self):
        for actor in [self.text_actor, self.legend_actor, self.controls_actor]:
            if actor is not None:
                self.plotter.remove_actor(actor)
        self.text_actor = self.legend_actor = self.controls_actor = None
        if self.professional_mode:
            self._draw_professional_ui()
        else:
            self._draw_layperson_ui()

    def _draw_professional_ui(self):
        snap = self.snapshots[self.current_idx]
        elev = snap["elevation"]
        ny, nx = elev.shape
        ds = f"{self.dataset['name']} ({self.dataset['name_en']})"
        mode = "Heatmap" if not self.realistic_mode else "Realistic"

        info = (
            f"Dataset: {ds}\n"
            f"Time: {snap['key']} ({snap['label']})\n"
            f"Grid: {nx}x{nx} | Cell: {self.dataset['cell_size']:.0f}m\n"
            f"Max: {elev.max():.1f}m | Min: {elev.min():.1f}m | Mean: {elev.mean():.1f}m\n"
            f"Coloring: {mode} | Snapshot: {self.current_idx + 1}/{len(self.snapshots)}"
        )
        self.text_actor = self.plotter.add_text(
            info, position="upper_left", font_size=12, color="white")

        controls = (
            "Controls:\n"
            "  <-/-> : Switch time\n"
            "  Up/Down : Switch dataset\n"
            "  S : Cross-section\n"
            "  M : Color mode\n"
            "  T : Text mode\n"
            "  R : Rain\n"
            "  W : Snow\n"
            "  V : Rivers\n"
            "  A : Auto-play\n"
            "  Q : Quit"
        )
        self.controls_actor = self.plotter.add_text(
            controls, position="upper_right", font_size=10, color="white")

    def _draw_layperson_ui(self):
        snap = self.snapshots[self.current_idx]
        ds = self.dataset['name']
        info = f"\u3010{ds}\u3011\u65f6\u95f4\uff1a{snap['label']}\n"
        info += get_layperson_text(snap['year'])
        self.text_actor = self.plotter.add_text(
            info, position="upper_left", font_size=14, color="white")

        legend = ("[Low] Valleys: mud settle   "
                  "[Mid] Slopes: soil washed   "
                  "[High] Peaks: rock weathering")
        self.legend_actor = self.plotter.add_text(
            legend, position="lower_left", font_size=11, color="#cccccc")

        controls = (
            "Left/Right  Time forward/back\n"
            "Up/Down     Switch landscape\n"
            "S           Cut mountain, see layers\n"
            "M           Color: real / heatmap\n"
            "R           Rain on/off\n"
            "W           Snow on/off\n"
            "V           Rivers on/off\n"
            "A           Auto-play\n"
            "Q           Quit\n"
            "Mouse drag rotate | Scroll zoom"
        )
        self.controls_actor = self.plotter.add_text(
            controls, position="upper_right", font_size=11, color="#dddddd")

    # ---- main ----

    def show(self):
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
