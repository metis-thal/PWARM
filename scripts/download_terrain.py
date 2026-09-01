"""
Download real-world terrain heightmap data from AWS Open Data (Mapzen Terrarium).

Usage:
    python scripts/download_terrain.py --lat 27.988 --lon 86.925 --zoom 12 --output terrain_everest.png

This downloads elevation tiles for the specified location and saves as a heightmap.
"""
import argparse
import math
import os
import struct
import urllib.request
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to tile coordinates at given zoom level."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def decode_terrarium(png_bytes: bytes) -> np.ndarray:
    """Decode Mapzen Terrarium encoding to elevation in meters.

    Terrarium encodes elevation as:
        red = ( elevation + 32768 ) / 256
        green = ( elevation + 32768 ) % 256
        No alpha channel needed.
    """
    import io
    img = Image.open(io.BytesIO(png_bytes))
    arr = np.array(img, dtype=np.float64)

    if arr.ndim == 3 and arr.shape[2] >= 2:
        red = arr[:, :, 0].astype(np.float64)
        green = arr[:, :, 1].astype(np.float64)
        elevation = (red * 256 + green) - 32768.0
    else:
        # Grayscale fallback
        elevation = arr.astype(np.float64)

    return elevation


def download_terrain_tiles(
    lat: float,
    lon: float,
    zoom: int = 12,
    grid_size: int = 4,
) -> np.ndarray:
    """Download a grid of terrain tiles centered on lat/lon.

    Args:
        lat: Center latitude
        lon: Center longitude
        zoom: Tile zoom level (higher = more detail, 10-16 typical)
        grid_size: Number of tiles in each direction (grid_size x grid_size)

    Returns:
        Combined elevation array in meters
    """
    cx, cy = lat_lon_to_tile(lat, lon, zoom)
    half = grid_size // 2

    rows = []
    for dy in range(-half, -half + grid_size):
        row_tiles = []
        for dx in range(-half, -half + grid_size):
            tx, ty = cx + dx, cy + dy
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{tx}/{ty}.png"
            print(f"  Downloading tile z={zoom} x={tx} y={ty} ...")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "pymo-terrain/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    png_data = resp.read()
                elev = decode_terrarium(png_data)
                row_tiles.append(elev)
            except Exception as e:
                print(f"  WARNING: Failed to download tile ({tx},{ty}): {e}")
                row_tiles.append(np.zeros((256, 256), dtype=np.float64))
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def save_heightmap_png(elevation: np.ndarray, output_path: str) -> None:
    """Save elevation as a 16-bit grayscale PNG heightmap."""
    if not HAS_PIL:
        print("PIL not available, saving as .npy only")
        return

    # Normalize to 0-65535 (16-bit)
    elev_min = elevation.min()
    elev_max = elevation.max()
    if elev_max > elev_min:
        normalized = (elevation - elev_min) / (elev_max - elev_min) * 65535.0
    else:
        normalized = np.zeros_like(elevation)

    img = Image.fromarray(normalized.astype(np.uint16), mode="I;16")
    img.save(output_path)
    print(f"  Saved heightmap: {output_path} ({elevation.shape[1]}x{elevation.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description="Download real-world terrain heightmap")
    parser.add_argument("--lat", type=float, default=27.988, help="Latitude (default: Mt Everest)")
    parser.add_argument("--lon", type=float, default=86.925, help="Longitude")
    parser.add_argument("--zoom", type=int, default=12, help="Zoom level (10-16)")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (NxN tiles)")
    parser.add_argument("--output", type=str, default="terrain_heightmap", help="Output path (no ext)")
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "data", "terrain")
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.join(output_dir, args.output)

    print(f"Downloading terrain for lat={args.lat}, lon={args.lon}, zoom={args.zoom}")
    print(f"Grid: {args.grid}x{args.grid} tiles ({args.grid * 256}x{args.grid * 256} pixels)")

    elevation = download_terrain_tiles(args.lat, args.lon, args.zoom, args.grid)

    print(f"\nElevation range: {elevation.min():.1f}m to {elevation.max():.1f}m")

    # Save as numpy array (primary format for pymo)
    npy_path = base + ".npy"
    np.save(npy_path, elevation)
    print(f"  Saved: {npy_path}")

    # Also save as PNG if PIL available
    png_path = base + ".png"
    save_heightmap_png(elevation, png_path)

    # Save metadata
    meta_path = base + "_meta.txt"
    with open(meta_path, "w") as f:
        f.write(f"lat: {args.lat}\n")
        f.write(f"lon: {args.lon}\n")
        f.write(f"zoom: {args.zoom}\n")
        f.write(f"grid: {args.grid}\n")
        f.write(f"pixels_x: {elevation.shape[1]}\n")
        f.write(f"pixels_y: {elevation.shape[0]}\n")
        f.write(f"elevation_min: {elevation.min():.1f}\n")
        f.write(f"elevation_max: {elevation.max():.1f}\n")
        f.write(f"source: AWS Open Data / Mapzen Terrarium\n")
    print(f"  Saved: {meta_path}")

    print(f"\nDone! Files saved to {output_dir}")


if __name__ == "__main__":
    main()
