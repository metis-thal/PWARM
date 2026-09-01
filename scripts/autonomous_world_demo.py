"""Autonomous World Demo: emergent phenomena from bottom-up physics.

Sets up a scene with rock, iron, wood, ice + SPH water + ecology (solar
radiation, convection). All macro phenomena (melting, oxidation, thermal
equilibration, fluid flow) emerge from the differential equations — no
hardcoded triggers.

Usage:
    python scripts/autonomous_world_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pymo.kernel.bodies3d import sphere_body, box_body, Material
from pymo.kernel.world_engine import WorldEngine
from pymo.rules.chemistry import ChemicalSystem
from pymo.rules.fluid import create_water_column, SPHParams


def main():
    print("=" * 64)
    print("PWARM Autonomous World Demo")
    print("All phenomena emerge from bottom-up differential equations")
    print("=" * 64)

    # ── 1. Create engine ───────────────────────────────────────────────
    engine = WorldEngine()
    engine.world.gravity = np.array([0.0, 0.0, -9.81])
    engine.world.dt = 1 / 60.0
    engine.world.solver_iterations = 10

    # ── 2. Ground plane ────────────────────────────────────────────────
    ground = box_body(
        [0, 0, -0.5],
        half_extents=np.array([20.0, 20.0, 0.5]),
        static=True,
        material=Material(
            restitution=0.3, friction=0.8,
            specific_heat=800.0, thermal_conductivity=2.0,
            density=2500.0,
        ),
    )
    engine.add_body(ground, albedo=0.2)

    # ── 3. Hot iron ball ──────────────────────────────────────────────
    iron = sphere_body(
        [0, 0, 4],
        radius=0.4, mass=2.5,
        material=Material(
            restitution=0.3, friction=0.5,
            specific_heat=449.0, thermal_conductivity=80.0,
            density=7874.0,
        ),
    )
    iron.temperature = 600.0  # hot — should conduct heat into ground
    engine.add_body(iron, albedo=0.3)

    # ── 4. Cold ice sphere ────────────────────────────────────────────
    ice = sphere_body(
        [-2, 0, 3],
        radius=0.35, mass=0.3,
        material=Material(
            restitution=0.1, friction=0.2,
            specific_heat=2100.0, thermal_conductivity=2.2,
            density=917.0,
        ),
    )
    ice.temperature = 250.0  # below freezing
    engine.add_body(ice, albedo=0.5)

    # ── 5. Wood block ─────────────────────────────────────────────────
    wood = box_body(
        [2, 0, 2.5],
        half_extents=np.array([0.3, 0.3, 0.3]),
        mass=0.16,
        material=Material(
            restitution=0.4, friction=0.6,
            specific_heat=1700.0, thermal_conductivity=0.15,
            density=600.0,
        ),
    )
    wood.temperature = 293.0
    engine.add_body(wood, albedo=0.25)

    # ── 6. Rock sphere ────────────────────────────────────────────────
    rock = sphere_body(
        [3, 1, 3],
        radius=0.3, mass=1.0,
        material=Material(
            restitution=0.2, friction=0.7,
            specific_heat=790.0, thermal_conductivity=3.0,
            density=2600.0,
        ),
    )
    rock.temperature = 293.0
    engine.add_body(rock, albedo=0.15)

    # ── 7. SPH fluid (water column) ───────────────────────────────────
    sph_params = SPHParams(
        h=0.15, rest_density=1000.0, stiffness=5000.0,
        viscosity=1.0, particle_mass=0.5, gamma=1.0,
    )
    sph = create_water_column(x=-3, y=-1, width=4, height=4, spacing=0.15,
                              params=sph_params)
    engine.add_fluid(sph)

    # ── 8. Chemistry: wood + oxygen at high T ─────────────────────────
    chem = ChemicalSystem(
        masses={"wood": 0.16, "oxygen": 0.5},
        temperature=293.0,
    )
    engine.add_chemistry(chem)

    # ── 9. Ecology: solar radiation + convection ──────────────────────
    engine.solar_flux = 800.0          # W/m^2 (strong sunlight)
    engine.environment_temp = 300.0    # K ambient

    n_bodies = len(engine.world.bodies)
    print(f"\n  Bodies: {n_bodies}  |  Fluid particles: {len(sph.particles)}")
    print(f"  Solar flux: {engine.solar_flux} W/m^2  |  Ambient: {engine.environment_temp} K")
    print(f"  Chemistry: wood={chem.masses.get('wood', 0):.3f} kg, "
          f"O2={chem.masses.get('oxygen', 0):.3f} kg")

    # ── 10. Run simulation ────────────────────────────────────────────
    n_steps = 300  # 5 seconds at 60 fps
    print(f"\nRunning {n_steps} steps ({n_steps / 60:.1f} s simulated)...\n")

    for step in range(n_steps):
        result = engine.tick(dt=1 / 60.0)

        if step % 60 == 0:  # print every second
            iron_T = engine.world.bodies[1].temperature
            ice_T = engine.world.bodies[2].temperature
            wood_T = engine.world.bodies[3].temperature
            rock_T = engine.world.bodies[4].temperature
            print(
                f"  t={result['t']:5.1f}s  "
                f"KE={result['kinetic_energy']:8.2f} J  "
                f"ThermE={result['thermal_energy']:10.1f} J  "
                f"| T_iron={iron_T:.0f}  T_ice={ice_T:.0f}  "
                f"T_wood={wood_T:.0f}  T_rock={rock_T:.0f}"
            )

    # ── 11. Final state ───────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Final state:")
    for i, body in enumerate(engine.world.bodies):
        label = ["ground", "iron", "ice", "wood", "rock"][i] if i < 5 else f"body{i}"
        print(f"  {label:8s}: T={body.temperature:7.1f} K  "
              f"pos=({body.pos[0]:+.2f}, {body.pos[1]:+.2f}, {body.pos[2]:+.2f})")

    # Chemistry state
    print(f"\n  Chemistry: {chem.masses}")
    print(f"  Element masses: {chem.element_masses()}")

    # Conservation
    cons = engine._snapshot_conservation()
    print(f"\n  Total mass:    {cons['total_mass']:.4f} kg")
    print(f"  Total energy:  {cons['total_energy']:.2f} J")
    print(f"  Fluid energy:  {cons['fluid_energy']:.2f} J")

    print("\n  All phenomena emerged from differential equations — no hardcoding.")
    print("=" * 64)


if __name__ == "__main__":
    main()
