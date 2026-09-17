"""AI Scientist Mission 002 — Material Discovery (adaptive, live dashboard).

The Mission 002 loop, visualized:

    Unknowns -> Uncertainty Model -> Experiment Designer -> Experiment
    -> Observation -> Theory -> Knowledge

Left:  the unknown universe executing the AI's chosen experiments
       (bouncing drops, sliding boxes).
Right: the AI scientist's live cognition — the question, the unknowns,
       the chosen design WITH its information-gain reasoning, the running
       measurement, the discoveries so far, and the honest list of what it
       CANNOT know with the current apparatus (density).

Run:
    python scripts/demo_mission_002.py               # interactive
    python scripts/demo_mission_002.py --frames 1500 # auto-quit (CI)

A second run concludes from civilization knowledge without re-running
experiments (delete knowledge/universe_002.json for a fresh scientist).
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
    REGISTRY,
    ExperimentDesigner,
    ExperimentSpec,
    KnowledgeBase,
    Laboratory,
    Mission,
    ScientistState,
)
from pymo.universes import load_universe
from pymo.viz.gl_renderer import GLRenderer, RendererConfig
from pymo.viz.snapshot import (
    CameraState,
    DoubleBuffer,
    build_snapshot_from_physics_engine,
)
from pymo.viz.text_overlay import TextPanel

KNOWLEDGE_PATH = Path("knowledge/universe_002.json")


def _panel_lines(state: ScientistState, lab: Laboratory, proposal,
                 session, n_samples: int, discoveries: list[str],
                 unidentifiable: list[str], mission_done: bool,
                 knowledge: KnowledgeBase) -> list:
    lines: list = [
        ("PWARM · MISSION 002", "#7fd7ff"),
        ("AI SCIENTIST — MATERIAL DISCOVERY", "#7fd7ff"),
        "",
        ("UNIVERSE", "#7fd7ff"),
        f"  {lab.universe_name}",
        f"  unknown parameters: {len(state.unknown_parameters())}",
        "",
        ("QUESTION", "#ffd479"),
        "  What is material_A?  (and material_B)",
        "",
    ]

    if proposal is not None and not mission_done:
        lines += [
            ("PLANNING — information gain", "#ffd479"),
            f"  chose: {proposal.kind}",
            f"  {ExperimentDesigner._describe(proposal.design)}",
            (f"  expected gain: {proposal.expected_gain:.2f}"
             f"   resolution: {proposal.rel_resolution * 100:.2f}%"),
            f"  reason: {proposal.reason[:52]}",
            "",
        ]

    if session is not None and not mission_done:
        t, z, vz = session.state
        phase = "SLIDING" if session.spec.kind == "slide_test" else (
            "FALLING" if not session.recording_done else "SETTLING")
        lines += [
            (f"RUNNING — {session.spec.id}", "#ffd479"),
            f"  t = {t:6.2f} s   z = {z:8.3f} m   v_z = {vz:7.2f}",
            f"  samples: {n_samples}   [{phase}]",
            "",
        ]

    if discoveries:
        lines += [("DISCOVERY SO FAR", "#7dffa8")]
        lines += [f"  {d}" for d in discoveries]
        lines.append("")

    if unidentifiable:
        lines += [("HONEST UNKNOWN (cannot know)", "#ff9d9d")]
        lines += [f"  {u}" for u in unidentifiable]
        lines.append("")

    if mission_done:
        lines += [("VERDICT", "#7dffa8")]
        for mat in sorted(knowledge.laws):
            rec = knowledge.laws[mat]
            if rec.properties:
                props = ", ".join(f"{k}={v:.3f}" for k, v in rec.properties.items())
                lines.append(f"  {mat}: {props}")
            elif rec.name == "gravity":
                lines.append(f"  gravity = {rec.value:.4f} m/s^2")
        lines.append(f"  knowledge saved: {len(knowledge.laws)} entries"
                     f"  ✓")
    return lines


def _publish(state: ScientistState, knowledge: KnowledgeBase,
             hypotheses: list) -> int:
    """Publish established material properties + gravity (mirrors the agent)."""
    published = 0
    materials = sorted({n.split(".", 1)[0] for n in state.beliefs if "." in n})
    for material in materials:
        props: dict[str, float] = {}
        widths: list[float] = []
        for name, belief in state.beliefs.items():
            if not name.startswith(material + "."):
                continue
            if belief.status == "known":
                props[name.split(".", 1)[1]] = round(belief.midpoint, 6)
                widths.append((belief.hi - belief.lo)
                              / max(abs(belief.midpoint), 1e-9))
        if props:
            confidence = max(0.0, 1.0 - sum(widths) / len(widths))
            knowledge.record_material(
                material, props, confidence, 1.0,
                [h.experiment_id for h in hypotheses],
                derived_by="adaptive-designer",
            )
            published += 1
    gravity_belief = state.belief("gravity")
    if (gravity_belief is not None and gravity_belief.status == "known"
            and knowledge.get("gravity") is None):
        rel_width = ((gravity_belief.hi - gravity_belief.lo)
                     / max(abs(gravity_belief.midpoint), 1e-9))
        knowledge.record_law(
            name="gravity",
            formula=f"g = {gravity_belief.midpoint:.4f} m/s^2 (free-fall fit)",
            value=gravity_belief.midpoint, unit="m/s^2",
            confidence=max(0.0, 1.0 - rel_width), r2=1.0,
            experiments=[h.experiment_id for h in hypotheses[:1]],
            derived_by="adaptive-designer",
        )
        published += 1
    if published:
        knowledge.save()
    return published


def main() -> int:
    parser = argparse.ArgumentParser(description="Mission 002 scientist dashboard")
    parser.add_argument("--frames", type=int, default=0,
                        help="auto-quit after N rendered frames (0 = until quit)")
    args = parser.parse_args()

    lab = Laboratory(load_universe("universe_002"))
    knowledge = KnowledgeBase.load_or_create(KNOWLEDGE_PATH, universe="universe_002")
    universe = load_universe("universe_002")
    state = ScientistState(universe.manifest, knowledge)
    designer = ExperimentDesigner()
    mission = Mission(
        id="002", title="Material Discovery",
        objective="Characterize the materials of Material World",
        target="material_A", unit="", min_confidence=0.95,
    )

    print("=" * 64)
    print("  PWARM · AI SCIENTIST — MISSION 002: Material Discovery")
    print(f"  Universe: {lab.universe_name} (parameters hidden from the AI)")
    print("=" * 64)
    print(f"\n[QUESTION] {mission.objective}")
    print(state.report())

    if knowledge.material("material_A") is not None:
        print("\n[KNOWLEDGE] mission already concluded from civilization knowledge:")
        print(knowledge.summary())
        print(f"\n(delete {KNOWLEDGE_PATH} to start a fresh scientist)")
        return 0


    buffer = DoubleBuffer()
    renderer = GLRenderer(buffer, RendererConfig(
        window_size=(1280, 720), title="PWARM · Mission 002 — Material Discovery"))
    renderer.init()
    if renderer.ctx is None:
        print("ERROR: OpenGL context unavailable")
        return 1
    panel = TextPanel(renderer.ctx, width_px=460)
    renderer.post_draw_hook = lambda r: panel.draw(r.framebuffer_size())

    camera = CameraState()
    camera.target = np.array([0.0, 0.0, 7.0], dtype=np.float32)
    camera.distance = 26.0
    camera.elevation = 0.35
    camera.azimuth = 0.55

    proposal = None
    session = None
    n_samples = 0
    discoveries: list[str] = []
    unidentifiable: list[str] = []
    hypotheses: list = []
    experiments_run = 0
    mission_done = False
    paused = False
    frame = 0
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

            if not paused and not mission_done:
                if session is None:
                    proposal = designer.choose(state)
                    if proposal is None:
                        # Nothing informative left: conclude the mission.
                        for claim in designer.unreachable_claims(state):
                            state.mark_unidentifiable(
                                claim, "no available experiment informs this "
                                       "parameter (identifiability limit of "
                                       "the current apparatus)")
                            unidentifiable.append(claim)
                        published = _publish(state, knowledge, hypotheses)
                        mission_done = True
                        print(f"\n[CONCLUDED] {experiments_run} experiments; "
                              f"{published} knowledge entries published")
                        print(state.report())
                    else:
                        session = lab.start_experiment(ExperimentSpec(
                            kind=proposal.kind,
                            drop_height=float(proposal.design.get("height", 10.0)),
                            v0=proposal.design.get("v0"),
                            material=proposal.design.get("material"),
                        ))
                        n_samples = 0
                        print(f"\n[PLAN] {proposal.kind} "
                              f"({ExperimentDesigner._describe(proposal.design)})")
                        print(f"  reason: {proposal.reason}")
                        # Frame the apparatus for this experiment.
                        if proposal.kind == "drop_test":
                            h = float(proposal.design.get("height", 10.0))
                            camera.target = np.array(
                                [0.0, 0.0, h * 0.4 + 1.0], dtype=np.float32)
                            camera.distance = h * 0.8 + 10.0
                        else:
                            camera.target = np.array(
                                [5.0, 0.0, 0.5], dtype=np.float32)
                            camera.distance = 18.0
                else:
                    alive = session.step()
                    n_samples = len(session.t)
                    if frame % 20 == 0:
                        t, z, vz = session.state
                        print(f"  t = {t:6.2f}   z = {z:8.3f}   v_z = {vz:8.2f}")
                    if not alive or (session.recording_done
                                     and session.spec.kind == "slide_test"):
                        record = session.finish()
                        g_known = None
                        gb = state.belief("gravity")
                        if gb is not None and gb.status == "known":
                            g_known = gb.midpoint
                        module = REGISTRY[proposal.kind]
                        hypothesis, fitted_g = module.derive(
                            record, g_known, proposal.design.get("material", "material"))
                        hypotheses.append(hypothesis)
                        experiments_run += 1
                        state.update(hypothesis.claim, hypothesis.value,
                                     proposal.rel_resolution)
                        if (fitted_g is not None
                                and state.belief("gravity").status != "known"):
                            state.update("gravity", fitted_g,
                                         proposal.rel_resolution)
                        discoveries.append(
                            f"{hypothesis.claim} = {hypothesis.value:.4f}"
                            f"  (R^2 = {hypothesis.r2:.4f})")
                        print(f"  -> discovered: {hypothesis.claim} = "
                              f"{hypothesis.value:.4f}  (R^2 = {hypothesis.r2:.4f})")
                        session = None

            # -- render -------------------------------------------------------
            engine = session.engine if session is not None else None
            if engine is not None:
                if (session.spec.kind == "slide_test"
                        and engine.scene.double_buffer_read is not None):
                    box_x = float(engine.scene.double_buffer_read.rigid_pos[0][0])
                    camera.target = np.array([box_x, 0.0, 0.5], dtype=np.float32)
                snapshot = build_snapshot_from_physics_engine(engine, camera)
                buffer.write(snapshot)
            panel.set_lines(_panel_lines(
                state, lab, proposal, session, n_samples, discoveries,
                unidentifiable, mission_done, knowledge))
            renderer.render_frame()

            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                renderer.set_title(
                    f"PWARM Mission 002 | {fps_n / (now - fps_t0):.0f} FPS"
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
