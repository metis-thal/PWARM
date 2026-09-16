"""P8 — Free Fall → AI Discovers Gravity: the first closed-loop demo.

Left:  the real world (ground-truth physics, OpenGL rendered).
Right: the AI observer's live hypothesis, fit and verdict.
Console: the observation log + the final discovery announcement.

The AI sees ONLY recorded (t, z) samples — never the engine's gravity. It
discovers g from data alone; the demo verifies it against ground truth.

Run:
    python experiments/free_fall_ai/demo.py                # interactive
    python experiments/free_fall_ai/demo.py --frames 140   # auto-quit (CI)

Keys: Space pause · Q / Esc quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from experiments.free_fall_ai.experiment import (
    DiscoveryReport,
    FreeFallConfig,
    FreeFallExperiment,
)
from pymo.viz.gl_renderer import GLRenderer, RendererConfig
from pymo.viz.snapshot import (
    CameraState,
    DoubleBuffer,
    build_snapshot_from_physics_engine,
)
from pymo.viz.text_overlay import TextPanel


def _panel_lines(rep: DiscoveryReport, exp: FreeFallExperiment) -> list:
    """Build the right-hand AI panel content from report + live ground truth."""
    st = exp.sphere_state()
    t, pos, vel = st if st is not None else (0.0, np.zeros(3), np.zeros(3))
    lines: list = [
        ("PWARM · P8 EMERGENT WORLD", "#7fd7ff"),
        ("physics = truth · AI = approximation", "#8a94a8"),
        "",
        ("GROUND TRUTH (engine)", "#7fd7ff"),
        f"  t   = {t:7.2f} s",
        f"  z   = {float(pos[2]):7.3f} m",
        f"  v_z = {float(vel[2]):7.3f} m/s",
        "  g   = 9.810 m/s^2   (hidden from AI)",
        "",
        (f"AI OBSERVER — {rep.phase}", "#7fd7ff"),
        f"  samples: {rep.n_samples}   t = {rep.t_span[0]:.2f} .. {rep.t_span[1]:.2f} s",
        "",
        ("HYPOTHESIS", "#ffd479"),
        "  z(t) = a*t^2 + b*t + c",
        "",
    ]
    if rep.expression:
        lines += [
            ("AI DISCOVERY", "#ffd479"),
            f"  z(t) = {rep.expression}",
            f"  a = {rep.a:12.6f}",
            f"  b = {rep.b:12.6f}",
            f"  c = {rep.c:12.6f}",
            "",
            f"  g_ai  = {rep.g_ai:9.4f} m/s^2",
            f"  R^2   = {rep.r2:9.6f}",
            f"  error = {rep.error_pct:8.5f} %",
        ]
        if rep.verified:
            lines += ["", ("  VERIFIED — gravity discovered", "#7dffa8")]
    else:
        lines.append("  collecting samples...")
    if exp.landing_t is not None:
        lines += ["", (f"  collision detected at t = {exp.landing_t:.2f} s", "#ffd479")]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="P8 free-fall AI discovery demo")
    parser.add_argument("--frames", type=int, default=0,
                        help="auto-quit after N rendered frames (0 = run until quit)")
    parser.add_argument("--height", type=float, default=10.0,
                        help="drop height in meters")
    args = parser.parse_args()

    exp = FreeFallExperiment(FreeFallConfig(height=args.height))
    engine = exp.setup()

    buffer = DoubleBuffer()
    renderer = GLRenderer(buffer, RendererConfig(
        window_size=(1280, 720), title="PWARM · P8 — AI Physics Discovery"))
    renderer.init()
    if renderer.ctx is None:
        print("ERROR: OpenGL context unavailable")
        return 1
    panel = TextPanel(renderer.ctx, width_px=430)
    renderer.post_draw_hook = lambda r: panel.draw(r.framebuffer_size())

    camera = CameraState()
    camera.target = np.array([0.0, 0.0, 4.5], dtype=np.float32)
    camera.distance = 22.0
    camera.elevation = 0.42
    camera.azimuth = 0.55

    print("Physics World\n")
    print("Object:\n  sphere_001")
    print(f"Initial:\n  height = {exp.config.height}m   mass = {exp.config.mass}kg\n")
    print("Simulation running...\n")

    paused = False
    rep = exp.report
    frame = 0
    collision_printed = False
    announced = False
    fps_t0 = time.time()
    fps_n = 0

    try:
        while renderer.running:
            key = renderer.poll_key()
            if key == ord("Q") or key == 27:
                break
            if key == ord(" "):
                paused = not paused
                print(f"    {'PAUSED' if paused else 'RUNNING'}")

            if not paused:
                rep = exp.step()
                if frame % exp.config.discover_every == 0:
                    exp.discover()
                    exp.evaluate()

                if exp.landed and not collision_printed:
                    print(f"  time: {exp.landing_t:5.2f}   collision detected\n")
                    collision_printed = True
                elif not exp.landed and frame % 15 == 0:
                    st = exp.sphere_state()
                    if st is not None:
                        print(f"  time: {st[0]:5.2f}   position: {float(st[1][2]):7.2f}")

                if rep.verified and not announced:
                    announced = True
                    print("Hypothesis: z(t) = a*t^2 + b*t + c")
                    print("Searching... done\n")
                    print(f"  a = {rep.a:.6f}   b = {rep.b:.6f}   c = {rep.c:.6f}")
                    print(f"  R^2 = {rep.r2:.6f}\n")
                    print(f"Ground Truth:  g = {exp.config.gravity:.4f} m/s^2")
                    print(f"AI Model:      g = {rep.g_ai:.4f} m/s^2")
                    print(f"error:         {rep.error_pct:.4f} %   VERIFIED\n")

            snapshot = build_snapshot_from_physics_engine(engine, camera)
            buffer.write(snapshot)
            panel.set_lines(_panel_lines(rep, exp))
            renderer.render_frame()

            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                st = exp.sphere_state()
                sim_t = st[0] if st is not None else 0.0
                renderer.set_title(
                    f"PWARM P8 | t={sim_t:.2f}s | {fps_n / (now - fps_t0):.0f} FPS")
                fps_t0, fps_n = now, 0

            frame += 1
            if args.frames and frame >= args.frames:
                break
    finally:
        renderer.close()
        panel.release()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
