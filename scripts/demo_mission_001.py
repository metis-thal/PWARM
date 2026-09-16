"""AI Scientist Mission 001 — Discover Gravity (live dashboard).

Left:  the unknown universe (ground-truth physics; parameters hidden from the AI).
Right: the AI scientist at work — plan, live observations, hypothesis,
       cross-experiment verification, knowledge publication.

The scientific method, step by step:
    plan (3 independent drop heights) -> execute -> observe -> hypothesize
    -> cross-verify (same g from different heights = a law) -> publish knowledge

Run:
    python scripts/demo_mission_001.py               # interactive
    python scripts/demo_mission_001.py --frames 500  # auto-quit (CI)

If the knowledge base already contains gravity, the mission concludes from
civilization knowledge without re-running experiments (delete
knowledge/universe_001.json to start a fresh scientist).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from pymo.scientist import (
    ExperimentSpec,
    Hypothesis,
    KnowledgeBase,
    Laboratory,
    Mission,
    ScientistAgent,
    Verification,
    fit_free_fall,
)
from pymo.universes import load_universe
from pymo.viz.gl_renderer import GLRenderer, RendererConfig
from pymo.viz.snapshot import (
    CameraState,
    DoubleBuffer,
    build_snapshot_from_physics_engine,
)
from pymo.viz.text_overlay import TextPanel

KNOWLEDGE_PATH = Path("knowledge/universe_001.json")


def _panel_lines(mission: Mission, lab: Laboratory, specs: list[ExperimentSpec],
                 current: int, session, n_samples: int,
                 hypotheses: list[Hypothesis], verification: Verification | None,
                 mission_done: bool, knowledge: KnowledgeBase) -> list:
    lines: list = [
        ("PWARM · MISSION 001", "#7fd7ff"),
        ("AI SCIENTIST — DISCOVER GRAVITY", "#7fd7ff"),
        "",
        ("UNIVERSE", "#7fd7ff"),
        f"  {lab.universe_name}",
        "  (parameters hidden from the AI)",
        "",
    ]

    if current < len(specs):
        spec = specs[current]
        lines += [
            (f"EXPERIMENT {current + 1}/{len(specs)}", "#ffd479"),
            f"  {spec.id}  (m = {spec.mass:g} kg)",
        ]
        if session is not None:
            t, z, vz = session.state
            phase = "FALLING" if session.running and not session.t else "RECORDING"
            lines.append(f"  t = {t:6.2f} s   z = {z:8.3f} m   v_z = {vz:7.2f}")
            lines.append(f"  samples: {n_samples}   [{phase}]")
        else:
            lines.append("  pending...")
        lines.append("")

    if hypotheses:
        lines.append(("AI HYPOTHESIS (fitted per drop)", "#ffd479"))
        for hyp in hypotheses:
            lines.append(f"  {hyp.experiment_id}: g = {hyp.value:.5f} {hyp.unit}")
        last = hypotheses[-1]
        lines += [f"  z(t) = {last.formula}", ""]
    else:
        lines += [("AI HYPOTHESIS", "#ffd479"), "  collecting observations...", ""]

    if verification is not None:
        est = " / ".join(f"{v:.4f}" for v in verification.estimates)
        lines += [
            ("VERIFICATION", "#7fd7ff"),
            f"  estimates: {est}",
            f"  spread: {verification.rel_spread * 100:.4f}%",
            f"  confidence: {verification.confidence * 100:.2f}%",
        ]
        if verification.stable and mission_done:
            lines += ["",
                      ("  ✓ STABLE — GRAVITY IS A LAW", "#7dffa8"),
                      (f"  knowledge saved: {len(knowledge.laws)} law(s)", "#7dffa8")]
        else:
            lines += ["", "  testing..."]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Mission 001 scientist dashboard")
    parser.add_argument("--frames", type=int, default=0,
                        help="auto-quit after N rendered frames (0 = until quit)")
    args = parser.parse_args()

    lab = Laboratory(load_universe("universe_001"))
    knowledge = KnowledgeBase.load_or_create(KNOWLEDGE_PATH, universe="universe_001")
    agent = ScientistAgent(lab, knowledge)
    mission = Mission(
        id="001", title="Discover Gravity",
        objective="Determine the gravitational acceleration of Unknown Planet",
        target="gravity", unit="m/s^2",
    )

    specs = agent.create_experiment(mission)
    print("=" * 64)
    print("  PWARM · AI SCIENTIST — MISSION 001: Discover Gravity")
    print(f"  Universe: {lab.universe_name} (parameters hidden from the AI)")
    print("=" * 64)

    if not specs:
        print("\n[KNOWLEDGE] mission already concluded from civilization knowledge:")
        print(knowledge.summary())
        print("\n(delete knowledge/universe_001.json to start a fresh scientist)")
        return 0

    print(f"\n[PLAN] {len(specs)} experiments proposed (different drop heights):")
    for i, spec in enumerate(specs, 1):
        print(f"  {i}. {spec.id}")

    # -- visualization setup ------------------------------------------------

    buffer = DoubleBuffer()
    renderer = GLRenderer(buffer, RendererConfig(
        window_size=(1280, 720), title="PWARM · Mission 001 — AI Scientist"))
    renderer.init()
    if renderer.ctx is None:
        print("ERROR: OpenGL context unavailable")
        return 1
    panel = TextPanel(renderer.ctx, width_px=440)
    renderer.post_draw_hook = lambda r: panel.draw(r.framebuffer_size())

    camera = CameraState()
    camera.target = np.array([0.0, 0.0, 7.0], dtype=np.float32)
    camera.distance = 26.0
    camera.elevation = 0.35
    camera.azimuth = 0.55

    hypotheses: list[Hypothesis] = []
    verification: Verification | None = None
    mission_done = False
    paused = False
    frame = 0
    fps_t0 = time.time()
    fps_n = 0
    session = None
    current = 0
    n_samples = 0
    collision_printed = False

    def _finish_current() -> None:
        nonlocal session, hypotheses, n_samples
        if session is None:
            return
        record = session.finish()
        _, hyp = fit_free_fall(record)
        hypotheses.append(hyp)
        agent.observe(record)
        print(f"  -> fitted: g = {hyp.value:.5f} {hyp.unit}  (R^2 = {hyp.r2:.6f})")
        session = None

    print()
    try:
        while renderer.running:
            key = renderer.poll_key()
            if key == ord("Q") or key == 27:
                break
            if key == ord(" "):
                paused = not paused
                print(f"    {'PAUSED' if paused else 'RUNNING'}")

            if not paused and not mission_done:
                if session is None:
                    if current >= len(specs):
                        # all experiments done -> cross-verify + publish
                        verification = agent.verify(hypotheses)
                        assert verification is not None
                        concluded = (verification.stable
                                     and verification.confidence >= mission.min_confidence)
                        if concluded:
                            knowledge.record_law(
                                name=mission.target,
                                formula=hypotheses[0].formula,
                                value=verification.mean,
                                unit=mission.unit,
                                confidence=verification.confidence,
                                r2=verification.mean_r2,
                                experiments=[h.experiment_id for h in hypotheses],
                                derived_by=hypotheses[0].source,
                            )
                            knowledge.save()
                        mission_done = True
                        print(f"\n[VERIFY] {len(hypotheses)} independent experiments:")
                        for hyp in hypotheses:
                            print(f"  {hyp.experiment_id}: g = {hyp.value:.5f} m/s^2"
                                  f"  (R^2 = {hyp.r2:.6f})")
                        print(f"  spread: {verification.rel_spread * 100:.4f}%   "
                              f"confidence: {verification.confidence * 100:.2f}%")
                        if concluded:
                            print(f"\n[KNOWLEDGE] gravity = {verification.mean:.4f} m/s^2 "
                                  f"saved to {KNOWLEDGE_PATH}")
                            print(f"\nMission {mission.id}: DISCOVERED")
                        else:
                            print(f"\nMission {mission.id}: INCOMPLETE "
                                  "(evidence not conclusive yet)")
                    else:
                        spec = specs[current]
                        session = lab.start_experiment(spec)
                        n_samples = 0
                        collision_printed = False
                        print(f"\n--- Experiment {current + 1}/{len(specs)}: "
                              f"{spec.id} ---")
                else:
                    session.step()
                    if session.recording_done and not collision_printed:
                        print("  collision detected (recording stopped)")
                        collision_printed = True
                    elif not session.recording_done:
                        n_samples = len(session.t)
                        if frame % 15 == 0:
                            t, z, _vz = session.state
                            print(f"  time: {t:5.2f}   position: {z:8.2f}")
                    if not session.running:
                        _finish_current()
                        current += 1

            # -- render -------------------------------------------------------
            engine = session.engine if session is not None else None
            if engine is not None:
                snapshot = build_snapshot_from_physics_engine(engine, camera)
                buffer.write(snapshot)
            panel.set_lines(_panel_lines(
                mission, lab, specs, current, session, n_samples,
                hypotheses, verification, mission_done, knowledge))
            renderer.render_frame()

            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                renderer.set_title(
                    f"PWARM Mission 001 | {fps_n / (now - fps_t0):.0f} FPS"
                    + (" | MISSION COMPLETE" if mission_done else ""))
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
