"""Phase 1.3 demo: AI closed loop discovers the free-fall law and predicts.

Observes a falling body -> symbolic regression discovers y(t) -> AI predicts
held-out future positions -> error measured vs physics-kernel ground truth.

Usage:
    .\\.venv\\Scripts\\python.exe scripts/demo_ai_closed_loop.py
"""

from __future__ import annotations

from pymo.ai.closed_loop import ClosedLoopAI
from pymo.ai.observer import collect_free_fall


def main() -> None:
    print("=" * 70)
    print("Phase 1.3 — AI closed loop: observe -> discover -> predict")
    print("=" * 70)

    # 1. Observe free-fall from the physics kernel (ground truth)
    print("\n[1] Observing a falling body (g=9.81)...")
    data = collect_free_fall(n_steps=300, dt=0.01, g=9.81)
    y = data.get("body0.pos.y")
    t = data.time()
    print(f"    collected {len(t)} samples, y from {y[0]:.2f} to {y[-1]:.2f}")

    # 2. Closed loop: discover law on training window, predict test window
    print("\n[2] Discovering law + predicting held-out positions...")
    ai = ClosedLoopAI(train_fraction=0.6)
    results = ai.run(data, quantities=["body0.pos.y", "body0.vel.y"])
    print(ai.summary())

    # 3. Detail on position
    pos = next(r for r in results if r.quantity == "body0.pos.y")
    print(f"\n[3] Discovered law:  y(t) = {pos.law.expression}")
    print(f"    fit R2 on train = {pos.law.r2:.4f}")
    print(f"    test rel error  = {pos.relative_error*100:.2f}% "
          f"(target < 5%) -> {'PASS' if pos.passed() else 'FAIL'}")
    # Compare a few predictions vs truth
    print("\n    t      truth      predicted   err")
    for i in range(0, len(pos.t_test), max(1, len(pos.t_test) // 5)):
        print(f"    {pos.t_test[i]:5.2f}  {pos.y_true[i]:9.3f}  "
              f"{pos.y_pred[i]:9.3f}   {abs(pos.y_pred[i]-pos.y_true[i]):.3f}")


if __name__ == "__main__":
    main()
