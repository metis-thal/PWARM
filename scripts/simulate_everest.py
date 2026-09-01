"""
Mt. Everest Future Evolution Simulation (1000 years)

Fast surface-process simulation:
- Tectonic uplift (India-Eurasia collision, ~5mm/yr)
- Stream Power Law erosion (river incision)
- Hillslope diffusion (soil creep)

Usage:
    python scripts/simulate_everest.py
"""
import os
import sys
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

from scipy.ndimage import zoom as ndimage_zoom


def compute_drainage_area(elevation_2d, cell_size):
    """D8 priority-flood drainage area."""
    ny, nx = elevation_2d.shape
    area = np.full((ny, nx), cell_size * cell_size, dtype=np.float64)
    dx = [1, 1, 0, -1, -1, -1, 0, 1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    dist = [1.0, 1.414, 1.0, 1.414, 1.0, 1.414, 1.0, 1.414]
    flat_idx = np.argsort(-elevation_2d.ravel())
    for flat in flat_idx:
        iy, ix = flat // nx, flat % nx
        e0 = elevation_2d[iy, ix]
        best_slope, best_ny, best_nx = 0.0, -1, -1
        for d in range(8):
            jy, jx = iy + dy[d], ix + dx[d]
            if 0 <= jy < ny and 0 <= jx < nx:
                slope = (e0 - elevation_2d[jy, jx]) / (cell_size * dist[d])
                if slope > best_slope:
                    best_slope, best_ny, best_nx = slope, jy, jx
        if best_ny >= 0:
            area[best_ny, best_nx] += area[iy, ix]
    return area


def stream_power_erosion(elev, cell_size, K=5e-6, dt=50.0):
    area = compute_drainage_area(elev, cell_size)
    sy, sx = np.gradient(elev, cell_size)
    slope = np.maximum(np.sqrt(sx**2 + sy**2), 1e-4)
    rate = np.minimum(K * np.power(area, 0.5) * slope, 0.01)
    return (rate * dt).astype(np.float32)


def hillslope_diffusion(elev, cell_size, D=0.005, dt=50.0):
    h2 = cell_size * cell_size
    lap = np.zeros_like(elev)
    lap[1:-1, 1:-1] = (elev[2:, 1:-1] + elev[:-2, 1:-1] +
                         elev[1:-1, 2:] + elev[1:-1, :-2] -
                         4 * elev[1:-1, 1:-1]) / h2
    return (D * lap * dt).astype(np.float32)


def generate_uplift_field(nx, ny, base_rate=0.005, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx, dtype=np.float32)
    y = np.linspace(0, 1, ny, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    uplift = np.exp(-0.5 * ((xx - 0.5) / 0.2) ** 2)
    noise = np.clip(rng.normal(1.0, 0.15, (ny, nx)), 0.6, 1.4)
    return (base_rate * uplift * noise).astype(np.float32)


def run_simulation(heightmap_path, output_dir, total_years=1000.0,
                   snapshot_interval=100.0, target_size=128):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Mt. Everest Evolution Simulation (Surface Processes)")
    print("=" * 60)

    print(f"\n[1/4] Loading heightmap...")
    elev_full = np.load(heightmap_path)
    print(f"  Original: {elev_full.shape} | {elev_full.min():.1f}m ~ {elev_full.max():.1f}m")

    scale = target_size / max(elev_full.shape)
    elev = ndimage_zoom(elev_full, scale, order=1).astype(np.float64)
    ny, nx = elev.shape
    cell_size = 5120.0 / nx
    print(f"  Downsampled: {nx}x{ny} | cell_size={cell_size:.1f}m")

    uplift_field = generate_uplift_field(nx, ny)
    os.makedirs(output_dir, exist_ok=True)

    snapshots = [("t0", elev.copy())]
    t0_elev = elev.copy()

    print(f"\n[2/4] Running {int(total_years)}-year simulation...")
    t, step_count = 0.0, 0
    start_time = time.time()
    time_hist, max_h, mean_h, min_h = [0.0], [elev.max()], [elev.mean()], [elev.min()]

    while t < total_years:
        dt = min(50.0, total_years - t)
        elev += uplift_field * dt
        elev -= stream_power_erosion(elev, cell_size, dt=dt)
        elev += hillslope_diffusion(elev, cell_size, dt=dt)
        t += dt
        step_count += 1
        time_hist.append(t); max_h.append(elev.max()); mean_h.append(elev.mean()); min_h.append(elev.min())

        if int(t / snapshot_interval) > int((t - dt) / snapshot_interval):
            snapshots.append((f"t{int(t)}", elev.copy()))
            print(f"  t={int(t):>5d}yr | max={elev.max():.1f}m mean={elev.mean():.1f}m min={elev.min():.1f}m | {time.time()-start_time:.1f}s")

    print(f"\n[3/4] Done! ({time.time()-start_time:.1f}s, {step_count} steps)")

    # --- Visualizations ---
    print(f"\n[4/4] Generating plots...")
    all_e = np.concatenate([s[1].ravel() for _, s in snapshots])
    vmin, vmax = all_e.min(), all_e.max()

    for name, e in snapshots:
        np.save(os.path.join(output_dir, f"elevation_{name}.npy"), e)

    # Before/After
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Mt. Everest: 1000-Year Evolution\n(Tectonic Uplift + Stream Power Erosion + Hillslope Diffusion)', fontsize=14, fontweight='bold')
    for ax, (name, e) in zip(axes, [snapshots[0], snapshots[-1]]):
        im = ax.imshow(e, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f'{name} ({int(name.replace("t",""))} years)', fontsize=13)
        ax.set_xlabel('X'); ax.set_ylabel('Y')
    plt.colorbar(im, ax=axes, label='Elevation (m)', shrink=0.6, pad=0.02)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches='tight'); plt.close()

    # Evolution grid
    n = len(snapshots); cols = min(n, 5); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    fig.suptitle('Mt. Everest Elevation Evolution (1000 years)', fontsize=14, fontweight='bold')
    af = axes.flat if hasattr(axes, 'flat') else [axes]
    for i, (name, e) in enumerate(snapshots):
        af[i].imshow(e, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
        af[i].set_title(f't={int(name.replace("t",""))}yr', fontsize=10)
        af[i].set_xticks([]); af[i].set_yticks([])
    for j in range(n, rows*cols): af[j].set_visible(False)
    plt.colorbar(im, ax=list(af)[:n], label='Elevation (m)', shrink=0.6, pad=0.02)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "evolution_grid.png"), dpi=150, bbox_inches='tight'); plt.close()

    # Stats
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_hist, max_h, 'r-o', label='Max', linewidth=2, markersize=4)
    ax.plot(time_hist, mean_h, 'g-s', label='Mean', linewidth=2, markersize=4)
    ax.plot(time_hist, min_h, 'b-^', label='Min', linewidth=2, markersize=4)
    ax.fill_between(time_hist, min_h, max_h, alpha=0.15, color='gray')
    ax.set_xlabel('Time (years)'); ax.set_ylabel('Elevation (m)')
    ax.set_title('Mt. Everest Elevation Over 1000 Years', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "elevation_stats.png"), dpi=150, bbox_inches='tight'); plt.close()

    # Change map
    delta = snapshots[-1][1] - snapshots[0][1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    vm = max(abs(delta.min()), abs(delta.max()))
    im = ax.imshow(delta, cmap='RdYlGn', origin='lower', vmin=-vm, vmax=vm)
    ax.set_title('Elevation Change (1000 yr)\nGreen=Uplift Red=Erosion', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='ΔElev (m)', shrink=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "elevation_change.png"), dpi=150, bbox_inches='tight'); plt.close()

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: t=0 → t=1000yr")
    print(f"  Initial: max={t0_elev.max():.1f}m mean={t0_elev.mean():.1f}m")
    print(f"  Final:   max={elev.max():.1f}m mean={elev.mean():.1f}m")
    print(f"  ΔMax: {elev.max()-t0_elev.max():+.1f}m  ΔMean: {elev.mean()-t0_elev.mean():+.1f}m")
    print(f"  Uplift area: {(delta>0).sum()/delta.size*100:.1f}% | Erosion area: {(delta<0).sum()/delta.size*100:.1f}%")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    hm = os.path.join(PROJECT_ROOT, "data", "terrain", "terrain_everest.npy")
    out = os.path.join(PROJECT_ROOT, "data", "terrain", "everest_simulation")
    if not os.path.exists(hm):
        print(f"ERROR: {hm} not found"); sys.exit(1)
    run_simulation(hm, out, total_years=1000.0, snapshot_interval=100.0, target_size=128)
