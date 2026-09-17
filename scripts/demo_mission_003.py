"""AI Scientist Mission 003 — Science Under Constraints (live dashboard).

The Mission 003 loop, visualized:

    Knowledge -> Unknowns -> Uncertainty -> Candidate Experiments
    -> Cost / Information -> Experiment Selection -> Experiment
    -> Observation -> Theory -> Knowledge -> New Unknowns -> (repeat)

Left:  the Constrained World executing the AI's chosen experiments
       (drops, slides, and — after the instrument grant — the Fluid Tank).
Right: the AI scientist's live cognition:

  * KNOWLEDGE — every parameter as known / interval / unidentifiable
  * EXPERIMENT BUDGET — three currencies being spent, live
  * AVAILABLE EXPERIMENTS — the candidate menu ranked by VALUE
    (expected information per cost unit), not raw information
  * SCIENTIFIC REASONING — including the instrument arc: density proves
    UNIDENTIFIABLE with drop/slide apparatus (equivalence principle),
    the scientist analyzes WHY, files an instrument request, the Fluid
    Tank is granted with a budget extension, and buoyancy finally
    identifies the density the old apparatus could never see.

Run:
    python scripts/demo_mission_003.py               # interactive
    python scripts/demo_mission_003.py --frames 2500 # auto-quit (CI)

A second run concludes from civilization knowledge without re-running
experiments (delete knowledge/universe_003.json for a fresh scientist).
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
    ExperimentBudget,
    ExperimentDesigner,
    ExperimentSpec,
    InstrumentCatalog,
    KnowledgeBase,
    Laboratory,
    Mission,
    ScientistAgent,
    ScientistState,
    analyze_gap,
)
from pymo.scientist.uncertainty import knowledge_lines
from pymo.universes import load_universe
from pymo.viz.gl_renderer import GLRenderer, RendererConfig
from pymo.viz.snapshot import (
    CameraState,
    DoubleBuffer,
    build_snapshot_from_physics_engine,
)
from pymo.viz.text_overlay import TextPanel

KNOWLEDGE_PATH = Path("knowledge/universe_003.json")


def _budget_from(universe) -> ExperimentBudget:
    cfg = universe.budget or {}
    return ExperimentBudget(
        experiments=int(cfg.get("experiments", 8)),
        simulation_steps=int(cfg.get("simulation_steps", 5000)),
        compute_cost=float(cfg.get("compute_cost", 50.0)))


def _frame_camera(camera: CameraState, kind: str, design: dict,
                  engine=None) -> None:
    """Aim the camera at the apparatus of the running experiment."""
    if kind == "drop_test":
        h = float(design.get("height", 10.0))
        camera.target = np.array([0.0, 0.0, h * 0.4 + 1.0], dtype=np.float32)
        camera.distance = h * 0.8 + 10.0
    elif kind == "slide_test":
        camera.target = np.array([5.0, 0.0, 0.5], dtype=np.float32)
        camera.distance = 18.0
        if engine is not None and engine.scene.double_buffer_read is not None:
            box_x = float(engine.scene.double_buffer_read.rigid_pos[0][0])
            camera.target = np.array([box_x, 0.0, 0.5], dtype=np.float32)
    else:  # buoyancy_test — frame the Fluid Tank
        camera.target = np.array([0.0, 0.0, 4.5], dtype=np.float32)
        camera.distance = 15.0


def _question_line(state: ScientistState) -> str:
    """The current scientific question, in priority order."""
    unidentifiable = [b for b in state.beliefs.values()
                      if b.status == "unidentifiable"]
    if unidentifiable:
        return f"What is {unidentifiable[0].name}?  (apparatus cannot tell)"
    measurable = [b for b in state.beliefs.values()
                  if b.status in ("unknown", "constrained")]
    if measurable:
        widest = max(measurable,
                     key=lambda b: (b.hi - b.lo) / max(abs(b.midpoint), 1e-9))
        return f"What is {widest.name}?"
    return "All questions answered."


def _panel_lines(state, lab, budget, menu, proposal, session, n_samples,
                 discoveries, instrument_log, mission_done, knowledge,
                 budget_note) -> list:
    lines: list = [
        ("PWARM · MISSION 003", "#7fd7ff"),
        ("AI SCIENTIST — SCIENCE UNDER CONSTRAINTS", "#7fd7ff"),
        "",
        ("UNIVERSE", "#7fd7ff"),
        f"  {lab.universe_name}",
        f"  unknown parameters: {len(state.unknown_parameters())}",
        "",
        ("KNOWLEDGE", "#7fd7ff"),
    ]
    lines += knowledge_lines(state)
    lines += ["", ("EXPERIMENT BUDGET", "#7fd7ff")]
    lines += budget.report_lines()

    lines += ["", ("CURRENT QUESTION", "#ffd479"),
              f"  {_question_line(state)}"]

    if menu and not mission_done:
        lines += ["", ("AVAILABLE EXPERIMENTS  (value = info / cost)",
                       "#ffd479")]
        for i, p in enumerate(menu[:3]):
            design = ExperimentDesigner._describe(p.design)
            selected = (proposal is not None and p.kind == proposal.kind
                        and p.design == proposal.design)
            marker = "  <== SELECTED" if selected else ""
            afford = "" if budget.can_afford(p.cost) else "  (unaffordable)"
            lines.append(f"  {i + 1}. {p.kind} {design}")
            lines.append(f"     cost {p.cost:.1f}  info {p.expected_gain:.2f}"
                         f"  value {p.expected_value:.2f}{marker}{afford}")

    if not mission_done and (instrument_log or budget_note):
        lines += ["", ("SCIENTIFIC REASONING", "#ffd479")]
        for entry in instrument_log[-5:]:
            lines.append(f"  {entry}")
        if budget_note:
            lines.append((f"  {budget_note}", "#ff9d9d"))

    if session is not None and not mission_done:
        t, z, vz = session.state
        phase = ("SINKING" if session.spec.kind == "buoyancy_test"
                 else "SLIDING" if session.spec.kind == "slide_test"
                 else "FALLING" if not session.recording_done else "SETTLING")
        lines += ["", (f"RUNNING — {session.spec.id}", "#ffd479"),
                  f"  t = {t:6.2f} s   z = {z:8.3f} m   v_z = {vz:7.2f}",
                  f"  samples: {n_samples}   [{phase}]"]

    if discoveries:
        lines += ["", ("DISCOVERY SO FAR", "#7dffa8")]
        lines += [f"  {d}" for d in discoveries[-6:]]

    if mission_done:
        lines += ["", ("VERDICT", "#7dffa8")]
        for mat in sorted(knowledge.laws):
            rec = knowledge.laws[mat]
            if rec.properties:
                props = ", ".join(f"{k}={v:.3g}"
                                  for k, v in rec.properties.items())
                lines.append(f"  {mat}: {props}")
            elif rec.name == "gravity":
                lines.append(f"  gravity = {rec.value:.4f} m/s^2")
        lines.append(f"  knowledge saved: {len(knowledge.laws)} entries  ✓")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Mission 003 dashboard")
    parser.add_argument("--frames", type=int, default=0,
                        help="auto-quit after N rendered frames (0 = until quit)")
    args = parser.parse_args()

    universe = load_universe("universe_003")
    lab = Laboratory(universe)
    knowledge = KnowledgeBase.load_or_create(KNOWLEDGE_PATH,
                                             universe="universe_003")
    state = ScientistState(universe.manifest, knowledge)
    budget = _budget_from(universe)
    catalog = InstrumentCatalog(universe.instruments)
    designer = ExperimentDesigner()
    agent = ScientistAgent(lab, knowledge)
    mission = Mission(
        id="003", title="Science Under Constraints",
        objective=("Characterize Constrained World within the experiment "
                   "budget; escape unidentifiability by requesting "
                   "instruments"),
        target="material_A", unit="", min_confidence=0.95,
    )

    print("=" * 64)
    print("  PWARM · AI SCIENTIST — MISSION 003: Science Under Constraints")
    print(f"  Universe: {lab.universe_name} (parameters hidden from the AI)")
    print("=" * 64)
    print(f"\n[QUESTION] {mission.objective}")
    print(state.report())
    print("\n[BUDGET]")
    for line in budget.report_lines():
        print(line)

    if knowledge.material("material_A") is not None:
        print("\n[KNOWLEDGE] mission already concluded from civilization knowledge:")
        print(knowledge.summary())
        print(f"\n(delete {KNOWLEDGE_PATH} to start a fresh scientist)")
        return 0

    buffer = DoubleBuffer()
    renderer = GLRenderer(buffer, RendererConfig(
        window_size=(1280, 720),
        title="PWARM · Mission 003 — Science Under Constraints"))
    renderer.init()
    if renderer.ctx is None:
        print("ERROR: OpenGL context unavailable")
        return 1
    panel = TextPanel(renderer.ctx, width_px=460)
    renderer.post_draw_hook = lambda r: panel.draw(r.framebuffer_size())

    camera = CameraState()
    camera.target = np.array([0.0, 0.0, 4.0], dtype=np.float32)
    camera.distance = 22.0
    camera.elevation = 0.35
    camera.azimuth = 0.55

    proposal = None
    session = None
    menu: list = []
    n_samples = 0
    discoveries: list[str] = []
    hypotheses: list = []
    instrument_log: list[str] = []
    budget_note = ""
    experiments_run = 0
    mission_done = False
    paused = False
    frame = 0
    fps_t0 = time.time()
    fps_n = 0

    def conclude(note: str = "") -> None:
        nonlocal mission_done
        published, _ = agent.publish_established(state, hypotheses)
        mission_done = True
        print(f"\n[CONCLUDED] {experiments_run} experiments; "
              f"{published} knowledge entries published{(' — ' + note) if note else ''}")
        print(state.report())

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
                    menu = designer.available_experiments(state)[:4]
                    proposal = designer.choose(state, budget)
                    if proposal is None:
                        if designer.available_experiments(state):
                            budget_note = ("informative candidates remain but "
                                           "the budget cannot afford them")
                            conclude("BUDGET EXHAUSTED")
                        else:
                            gaps = designer.unreachable_claims(state)
                            for claim in gaps:
                                state.mark_unidentifiable(
                                    claim, "no available experiment informs "
                                           "this parameter (identifiability "
                                           "limit of the current apparatus)")
                                print(f"\n[UNIDENTIFIABLE] {claim} — no "
                                      "observable of the current apparatus "
                                      "depends on it")
                            request = analyze_gap(state)
                            if request is None:
                                conclude()
                            else:
                                print(f"\n[INSTRUMENT REQUEST] "
                                      f"{request.instrument} "
                                      f"({request.capability})")
                                print(f"  why: {request.reason[:100]}")
                                grant = catalog.grant(request)
                                if grant is None:
                                    budget_note = ("instrument refused — the "
                                                   "claim stays honestly "
                                                   "unidentifiable")
                                    conclude("INSTRUMENT REFUSED")
                                else:
                                    add = grant.budget
                                    budget.extend(
                                        experiments=int(add.get("experiments", 0)),
                                        simulation_steps=int(add.get("simulation_steps", 0)),
                                        compute_cost=float(add.get("compute_cost", 0.0)))
                                    designer.enable_kind(grant.enables)
                                    for claim in request.target_claims:
                                        state.revive(claim)
                                    instrument_log.append(
                                        f"{request.target_claims[0]} UNIDENTIFIABLE: "
                                        "drop/slide dynamics are density-invariant")
                                    instrument_log.append(
                                        f"REQUEST filed: {request.instrument} "
                                        f"({request.capability}) — measure "
                                        "buoyant force")
                                    instrument_log.append(
                                        f"GRANTED: {grant.enables} unlocked; budget "
                                        f"+{add.get('experiments', 0)} exp, "
                                        f"+{add.get('simulation_steps', 0)} steps, "
                                        f"+{add.get('compute_cost', 0.0):.1f} cost")
                                    print(f"[GRANTED] {grant.instrument}: "
                                          f"{grant.enables} unlocked, budget "
                                          "extended")
                    else:
                        session = lab.start_experiment(ExperimentSpec(
                            kind=proposal.kind,
                            drop_height=float(proposal.design.get(
                                "height", proposal.design.get("depth", 10.0))),
                            v0=proposal.design.get("v0"),
                            material=proposal.design.get("material"),
                        ))
                        n_samples = 0
                        print(f"\n[PLAN] {proposal.kind} "
                              f"({ExperimentDesigner._describe(proposal.design)})")
                        print(f"  reason: {proposal.reason}")
                        _frame_camera(camera, proposal.kind, proposal.design)
                else:
                    alive = session.step()
                    n_samples = len(session.t)
                    if frame % 20 == 0:
                        t, z, vz = session.state
                        print(f"  t = {t:6.2f}   z = {z:8.3f}   v_z = {vz:8.2f}")
                    if (not alive or (session.recording_done
                                      and session.spec.kind == "slide_test")):
                        record = session.finish()
                        budget.spend(proposal.cost, record.steps)
                        experiments_run += 1
                        g_known = None
                        gb = state.belief("gravity")
                        if gb is not None and gb.status == "known":
                            g_known = gb.midpoint
                        module = REGISTRY[proposal.kind]
                        hypothesis, fitted_g = module.derive(
                            record, g_known,
                            proposal.design.get("material", "material"))
                        hypotheses.append(hypothesis)
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
                              f"{hypothesis.value:.4f}  "
                              f"(R^2 = {hypothesis.r2:.4f})  "
                              f"[cost {proposal.cost:.1f}, {record.steps} steps]")
                        session = None

            # -- render -------------------------------------------------------
            engine = session.engine if session is not None else None
            if engine is not None:
                if session.spec.kind == "slide_test":
                    _frame_camera(camera, "slide_test", session.spec.__dict__,
                                  engine)
                snapshot = build_snapshot_from_physics_engine(engine, camera)
                buffer.write(snapshot)
            panel.set_lines(_panel_lines(
                state, lab, budget, menu, proposal, session, n_samples,
                discoveries, instrument_log, mission_done, knowledge,
                budget_note))
            renderer.render_frame()

            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                renderer.set_title(
                    f"PWARM Mission 003 | {fps_n / (now - fps_t0):.0f} FPS"
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
