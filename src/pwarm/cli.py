"""PWARM command-line interface — one-command credibility.

A stranger should be able to run:

    git clone https://github.com/metis-thal/PWARM.git
    cd PWARM && pip install -e .
    pwarm demo

and watch the AI scientist discover gravity in a plain terminal — no GPU,
no OpenGL, no window. The CLI is the HUMAN-FACING narrator: it may read the
universe's hidden truth for the epilogue ("ground truth / error"), but the
AI scientist itself never does (see docs/architecture/ai-physics-contract.md).

Commands:
    pwarm demo [001|002|003|pilot]        headless discovery run (default 001)
    pwarm mission run <id> [--json DIR]   same run + AI-side result artifacts
    pwarm dashboard <id>                  how to launch the GL dashboard
    pwarm --version
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from pwarm.scientist import (
    ExperimentBudget,
    ExperimentDesigner,
    InstrumentCatalog,
    KnowledgeBase,
    Laboratory,
    Mission,
    ScientistAgent,
    ScientistState,
)
from pwarm.universes import load_universe

_BOX = "=" * 70

MISSIONS = {
    "001": {"universe": "universe_001", "title": "DISCOVER GRAVITY"},
    "002": {"universe": "universe_002", "title": "MATERIAL PROPERTIES"},
    "003": {"universe": "universe_003", "title": "SCIENCE UNDER CONSTRAINTS"},
}


def _mission_def(mission_id: str) -> Mission:
    if mission_id == "001":
        return Mission(id="001", title="Discover Gravity",
                       objective="Determine the gravitational acceleration of "
                                 "Unknown Planet",
                       target="gravity", unit="m/s^2")
    if mission_id == "002":
        return Mission(id="002", title="Material Discovery",
                       objective="Characterize the materials of Material World",
                       target="material_A", unit="", min_confidence=0.95)
    return Mission(id="003", title="Science Under Constraints",
                   objective="Characterize Constrained World within budget",
                   target="material_A", unit="", min_confidence=0.95)


def _run(mission_id: str, knowledge_path: Path) -> dict:
    """Execute one mission headlessly. Returns the narrator's view:
    AI-side report + (narrator-only) ground truth for the epilogue."""
    universe = load_universe(MISSIONS[mission_id]["universe"])
    lab = Laboratory(universe)
    knowledge = KnowledgeBase(knowledge_path, universe=MISSIONS[mission_id]["universe"])
    agent = ScientistAgent(lab, knowledge)
    mission = _mission_def(mission_id)
    np.random.seed(0)   # belt-and-braces: the physics path is deterministic

    if mission_id == "001":
        report = agent.run_mission(mission)
    elif mission_id == "002":
        state = ScientistState(universe.manifest, knowledge)
        report = agent.run_adaptive_mission(mission, state, ExperimentDesigner())
    else:
        state = ScientistState(universe.manifest, knowledge)
        cfg = universe.budget or {}
        budget = ExperimentBudget(
            experiments=int(cfg.get("experiments", 8)),
            simulation_steps=int(cfg.get("simulation_steps", 5000)),
            compute_cost=float(cfg.get("compute_cost", 50.0)))
        catalog = InstrumentCatalog(universe.instruments)
        report = agent.run_constrained_mission(mission, state, budget,
                                               ExperimentDesigner(), catalog)

    truth: dict = {}
    if universe.secrets.gravity is not None:
        truth["gravity"] = universe.secrets.gravity
    for name, props in universe.secrets.materials.items():
        for prop, value in props.items():
            truth[f"{name}.{prop}"] = value
    return {"report": report, "truth": truth, "universe_name": lab.universe_name}


def _print_001(report, truth: dict) -> None:
    print(f"""
{_BOX}
   PWARM MISSION 001 — DISCOVER GRAVITY
{_BOX}

Universe: Unknown Planet
Hidden gravity:  ????            (the AI is not told)

AI Scientist
    ↓
Experiment #1   drop from 10 m   → trajectory samples (t, z)
Experiment #2   drop from 20 m   → trajectory samples (t, z)
Experiment #3   drop from  5 m   → trajectory samples (t, z)
    ↓
Hypothesis (per experiment, free-fall parabola fit):
""")
    for h in report.hypotheses:
        print(f"    {h['experiment_id']}:  g ≈ {h['value']:.5f} m/s^2"
              f"   (R^2 = {h['r2']:.4f})")
    print(f"""
Verification (cross-experiment):
    {report.summary}

{_BOX}
   DISCOVERY {'CONFIRMED' if report.status == 'DISCOVERED' else report.status}
{_BOX}""")

    if report.status == "DISCOVERED" and "gravity" in truth:
        g_true = truth["gravity"]
        err = abs(report.value - g_true) / g_true * 100
        print(f"""
Ground truth (narrator only — the AI never saw this):
    g = {g_true:.2f} m/s^2
    measured = {report.value:.5f} m/s^2
    error    = {err:.4f}%""")
    if report.knowledge_saved:
        print("    knowledge published: civilization accumulates.")


def _print_002(report, truth: dict) -> None:
    print(f"""
{_BOX}
   PWARM MISSION 002 — MATERIAL PROPERTIES
{_BOX}

Universe: Material World
Hidden per material: density, restitution, friction  (the AI is not told)

AI Scientist (information-gain designer)
    ↓
{report.summary}

{_BOX}
   {'DISCOVERY CONFIRMED' if report.status == 'DISCOVERED' else report.status}
{_BOX}""")
    for h in report.hypotheses:
        print(f"    {h['experiment_id']}:  {h['claim']} = {h['value']:.4f}"
              f"   (R^2 = {h['r2']:.4f})")
    unidentifiable = [line for line in report.summary.splitlines()
                      if "unidentifiable" in line.lower()]
    if unidentifiable:
        print("""
Honest metacognition (the scientist does not guess):""")
        for line in unidentifiable:
            print(f"    {line.strip()}")


def _print_003(report, truth: dict) -> None:
    print(f"""
{_BOX}
   PWARM MISSION 003 — SCIENCE UNDER CONSTRAINTS
{_BOX}

Universe: Constrained World
Budget: 4 experiments / 1400 steps / 12 cost units   (a constraint, not a secret)

AI Scientist (value = information / cost designer)
    ↓
{report.summary}

{_BOX}
   {'DISCOVERY CONFIRMED' if report.status == 'DISCOVERED' else report.status}
{_BOX}""")
    for h in report.hypotheses:
        print(f"    {h['experiment_id']}:  {h['claim']} = {h['value']:.4f}"
              f"   (R^2 = {h['r2']:.4f})")


_PRINTERS = {"001": _print_001, "002": _print_002, "003": _print_003}


def _write_artifacts(mission_id: str, report, out_dir: Path) -> None:
    """Write AI-side reproducibility artifacts (measurement-only data)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "observations.json").write_text(json.dumps(
        {"mission": mission_id, "records": report.observations},
        indent=2), encoding="utf-8")
    (out_dir / "hypotheses.json").write_text(json.dumps(
        report.hypotheses, indent=2), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(
        {"mission": mission_id, "status": report.status,
             "experiments_run": report.experiments_run,
             "value": report.value, "unit": report.unit,
             "confidence": report.confidence,
             "knowledge_saved": report.knowledge_saved,
             "summary": report.summary},
        indent=2), encoding="utf-8")
    print(f"\n[artifacts] AI-side results written to {out_dir}"
          "  (observations / hypotheses / report — no universe secrets)")


def _run_pilot_demo() -> int:
    """`pwarm demo pilot` — the Open Genesis Pilot, stranger-facing.

    Cold start by default (fresh temp store, the mission-demo convention):
    the scientist begins from NOTHING and must derive everything from
    measurements through the measurement-only facade."""
    from pwarm.scientist.pilot import run_open_genesis_pilot

    knowledge_path = Path(tempfile.gettempdir()) / "pwarm_cli_pilot_kb.json"
    knowledge_path.unlink(missing_ok=True)      # cold start: begin from nothing
    print("=" * 70)
    print("   PWARM OPEN GENESIS PILOT — an autonomous scientist cold start")
    print("=" * 70)
    print("Hidden universe constants: ?????          (the AI is not told)")
    print("Knowledge base: EMPTY          (no models, no relations, no laws)")
    print()
    report = run_open_genesis_pilot(knowledge_path)
    print(f"cold start (G-1): fitted family "
          f"{report.bootstrap.family!r} (r^2 = {report.bootstrap.r2:.4f}), "
          f"declared tolerance {report.bootstrap.tolerance:.4g}")
    print(f"research cycles: {[o.action for o in report.cycle_outcomes]}")
    for discovery in report.discoveries:
        pair = discovery.representative_heldout_confirmed
        print(f"first discovery: relation {discovery.relation_id} "
              f"({discovery.subject} {discovery.formula}) held out at "
              f"{pair.experiment_id} -> {pair.status.upper()}")
    print()
    for criterion in report.criterion_results:
        print(f"  {criterion.criterion:<42} "
              f"{'PASS' if criterion.passed else 'FAIL'}")
    print()
    print("=" * 70)
    print("   PILOT ACCEPTED — independent physics verified every claim"
          if report.accepted else
          "   PILOT REJECTED — see the failing criteria above")
    print("=" * 70)
    return 0 if report.accepted else 1


def cmd_run(args: argparse.Namespace) -> int:
    mission_id = args.mission
    if mission_id == "pilot":
        return _run_pilot_demo()
    if mission_id not in MISSIONS:
        print(f"unknown mission {mission_id!r}: choose from "
              f"{sorted(MISSIONS) + ['pilot']}")
        return 2
    if args.knowledge:
        knowledge_path = Path(args.knowledge)
    else:
        knowledge_path = Path(tempfile.gettempdir()) / f"pwarm_cli_{mission_id}.json"
        knowledge_path.unlink(missing_ok=True)   # fresh scientist by default
    result = _run(mission_id, knowledge_path)
    _PRINTERS[mission_id](result["report"], result["truth"])
    if args.json:
        _write_artifacts(mission_id, result["report"], Path(args.json))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    script = {"001": "demo_mission_001.py", "002": "demo_mission_002.py",
              "003": "demo_mission_003.py"}.get(args.mission)
    if script is None:
        print(f"unknown mission {args.mission!r}")
        return 2
    print(f"The GL dashboard needs a window (OpenGL 3.3+). Run:\n"
          f"    python scripts/{script}\n"
          f"headless alternative:\n"
          f"    python scripts/{script} --frames 500\n"
          f"or stay in the terminal:\n"
          f"    pwarm demo {args.mission}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # The discovery output uses box-drawing glyphs (═ ↓); Windows consoles
    # default to a legacy codepage (cp1252 on GitHub runners) that cannot
    # encode them — force UTF-8 before anything prints.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    import pwarm
    parser = argparse.ArgumentParser(
        prog="pwarm",
        description="PWARM — Physical World AI Reasoning Model (v0.1 Research Preview)")
    parser.add_argument("--version", action="version",
                        version=f"pwarm {getattr(pwarm, '__version__', '0.1.0a0')}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="headless discovery run (stranger test)")
    demo.add_argument("mission", nargs="?", default="001",
                      choices=sorted(MISSIONS) + ["pilot"])
    demo.set_defaults(func=cmd_run, json=None, knowledge=None)

    run = sub.add_parser("mission", help="mission operations")
    run_sub = run.add_subparsers(dest="mission_command", required=True)
    run_cmd = run_sub.add_parser("run", help="run one mission headlessly")
    run_cmd.add_argument("mission", choices=sorted(MISSIONS))
    run_cmd.add_argument("--json", default=None, metavar="DIR",
                         help="write AI-side result artifacts to DIR")
    run_cmd.add_argument("--knowledge", default=None, metavar="PATH",
                         help="reuse a knowledge file (civilization persists)")
    run_cmd.set_defaults(func=cmd_run)

    dash = sub.add_parser("dashboard", help="how to launch the GL dashboard")
    dash.add_argument("mission", choices=sorted(MISSIONS))
    dash.set_defaults(func=cmd_dashboard)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
