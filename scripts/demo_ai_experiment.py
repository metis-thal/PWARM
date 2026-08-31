"""AI Autonomous Experiment Demo.

Demonstrates the AI layer's ability to:
1. Design experiments with parameter variations
2. Run batch experiments
3. Evaluate hypotheses against results
4. Discover the best parameters

Usage:
    python scripts/demo_ai_experiment.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pymo.ai.experiment import (
    AutonomousExperimenter,
    ExperimentRunner,
    Hypothesis,
    ParameterSpace,
    create_free_fall_experiment,
    create_collision_experiment,
)
from pymo.ai.observer import WorldObserver
from pymo.kernel.bodies import circle_body
from pymo.kernel.world import World
from pymo.ai.closed_loop import ClosedLoopAI


def main():
    print("=" * 60)
    print("PWARM AI Autonomous Experiment Demo")
    print("=" * 60)

    # --- Part 1: Single Experiment + Law Discovery ---
    print("\n[Part 1] Single Free-Fall Experiment + Law Discovery")
    print("-" * 60)

    config = create_free_fall_experiment(height=10.0, n_steps=300)
    runner = ExperimentRunner()
    result = runner.run_experiment(config)

    print(f"  Experiment: {config.description}")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration_seconds:.3f}s")
    print(f"  Metrics:")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    # Run closed-loop AI on the data
    print("\n  Running AI law discovery on free-fall data...")
    ai = ClosedLoopAI()
    results = ai.run(result.dataset, quantities=["body0.pos.y"])
    print(ai.summary())

    # --- Part 2: Parameter Sweep ---
    print("\n[Part 2] Parameter Sweep: Gravity Variation")
    print("-" * 60)

    space = ParameterSpace()
    space.add_parameter("gravity_y", [-9.81, -4.9, -1.0])
    space.add_parameter("body0_y", [5.0, 10.0])

    experimenter = AutonomousExperimenter(runner)

    # Define hypothesis: body should fall
    def body_falls(dataset):
        y = dataset.get("body0.pos.y")
        if y is None or len(y) < 2:
            return False
        return y[-1] < y[0]

    hypothesis = Hypothesis(
        name="free_fall",
        description="Body falls under gravity",
        expected_behavior=body_falls
    )
    experimenter.add_hypothesis(hypothesis)

    # Run campaign
    results = experimenter.run_experiment_campaign(space, strategy="grid")
    print(f"  Ran {len(results)} experiments")

    # Find best parameters
    best = experimenter.get_best_parameters(metric="body0.pos.y_final", maximize=False)
    if best:
        print(f"  Best parameters for lowest final height:")
        for k, v in best.items():
            print(f"    {k}: {v}")

    # Show summary
    print(f"\n{experimenter.summary()}")

    # --- Part 3: Collision Experiment ---
    print("\n[Part 3] Two-Body Collision Experiment")
    print("-" * 60)

    config = create_collision_experiment(v1=5.0, v2=-5.0, separation=4.0, n_steps=200)
    result = runner.run_experiment(config)

    print(f"  Experiment: {config.description}")
    print(f"  Success: {result.success}")

    # Check if bodies bounced
    y1 = result.dataset.get("body0.pos.x")
    y2 = result.dataset.get("body1.pos.x")
    if y1 is not None and y2 is not None:
        print(f"  Body 0 final x: {y1[-1]:.3f} (started at -2.0)")
        print(f"  Body 1 final x: {y2[-1]:.3f} (started at 2.0)")

        # Verify collision occurred (bodies reversed direction)
        if y1[-1] > y1[0] and y2[-1] < y2[0]:
            print("  [PASS] Bodies bounced off each other")
        else:
            print("  [INFO] Collision behavior may need tuning")

    # --- Part 4: Hypothesis Testing ---
    print("\n[Part 4] Hypothesis Testing Campaign")
    print("-" * 60)

    space = ParameterSpace()
    space.add_parameter("body0_y", [5.0, 10.0, 15.0])
    space.add_parameter("body0_mass", [0.5, 1.0, 2.0])

    def energy_conserved(dataset):
        ke = dataset.get("total.ke")
        if ke is None or len(ke) < 10:
            return True  # not enough data
        # In free fall, KE should increase as PE decreases
        return ke[-1] > ke[0]

    def reasonable_fall_time(dataset):
        t = dataset.time()
        y = dataset.get("body0.pos.y")
        if y is None or len(y) < 2:
            return True
        # Should take reasonable time to fall
        return t[-1] > 0.5

    h1 = Hypothesis("energy_increases", "KE increases during fall", energy_conserved)
    h2 = Hypothesis("reasonable_time", "Fall takes reasonable time", reasonable_fall_time)

    experimenter = AutonomousExperimenter(runner)
    experimenter.add_hypothesis(h1)
    experimenter.add_hypothesis(h2)

    results = experimenter.run_experiment_campaign(space, n_experiments=9, strategy="grid")
    print(f"  Ran {len(results)} experiments")
    print(f"\n{experimenter.summary()}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
