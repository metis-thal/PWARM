"""Full demonstration: 3D physics + thermodynamics + chemistry + ecology.

Simulates a multi-physics world with:
- Rigid body dynamics (World3D with collision detection)
- Heat conduction between bodies
- Chemical reactions (combustion)
- Ecological coupling (solar radiation, evaporation, precipitation)

Usage:
    python scripts/demo_full.py
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pymo.kernel.bodies3d import sphere_body, box_body, Material
from pymo.kernel.world3d import World3D
from pymo.rules.chemistry import ChemicalSystem, step_chemistry
from pymo.rules.ecology import create_ecology_system, couple_body_environment
from pymo.rules.thermal import BodyThermalSystem


def main():
    print("=" * 60)
    print("PWARM Full Demonstration: Multi-Physics Simulation")
    print("=" * 60)
    
    # --- Setup 3D World ---
    print("\n[1/4] Setting up 3D physics world...")
    world = World3D(
        gravity=np.array([0.0, 0.0, -9.81]),
        dt=1/60.0,
        solver_iterations=10
    )
    
    # Ground plane
    ground = box_body(
        [0, 0, -0.5],
        half_extents=np.array([10.0, 10.0, 0.5]),
        static=True,
        material=Material(restitution=0.5, friction=0.8, specific_heat=800.0)
    )
    world.add(ground)
    
    # Hot iron ball (will fall and heat surroundings)
    iron_ball = sphere_body(
        [0, 0, 5],
        radius=0.5,
        mass=7.8,  # dense iron
        material=Material(
            restitution=0.3,
            friction=0.6,
            specific_heat=450.0,
            thermal_conductivity=80.0,
            density=7800.0
        )
    )
    iron_ball.temperature = 500.0  # hot!
    world.add(iron_ball)
    
    # Cold water sphere
    water_ball = sphere_body(
        [2, 0, 4],
        radius=0.4,
        mass=2.5,
        material=Material(
            restitution=0.2,
            friction=0.3,
            specific_heat=4186.0,
            thermal_conductivity=0.6,
            density=1000.0
        )
    )
    water_ball.temperature = 273.0  # cold
    world.add(water_ball)
    
    # Wooden block
    wood_block = box_body(
        [-2, 0, 3],
        half_extents=np.array([0.3, 0.3, 0.3]),
        mass=0.5,
        material=Material(
            restitution=0.4,
            friction=0.7,
            specific_heat=1700.0,
            thermal_conductivity=0.15,
            density=600.0
        )
    )
    wood_block.temperature = 293.0  # room temp
    world.add(wood_block)
    
    print(f"  Created {len(world.bodies)} bodies: ground, iron ball, water, wood")
    
    # --- Setup Thermal System ---
    print("\n[2/4] Setting up thermal coupling...")
    thermal_system = BodyThermalSystem(contact_area=0.1, contact_distance=0.01)
    
    print(f"  Iron ball temperature: {iron_ball.temperature:.1f} K")
    print(f"  Water ball temperature: {water_ball.temperature:.1f} K")
    
    # --- Setup Chemistry ---
    print("\n[3/4] Setting up chemical system...")
    chem_system = ChemicalSystem(
        masses={"methane": 0.01, "oxygen": 1.0},
        temperature=300.0
    )
    print(f"  Methane mass: {chem_system.masses.get('methane', 0):.3f} kg")
    print(f"  Oxygen mass: {chem_system.masses.get('oxygen', 0):.3f} kg")
    
    # --- Setup Ecology ---
    print("\n[4/4] Setting up ecological system...")
    ecology = create_ecology_system(latitude_deg=45.0, longitude_deg=0.0)
    
    # Add bodies to ecology (use 2D-compatible circle_body for ecology coupling)
    from pymo.kernel.bodies import circle_body, Material as Mat2D
    iron_2d = circle_body([0, 0], 0.5, mass=7.8, material=Mat2D(specific_heat=450.0, thermal_conductivity=80.0))
    iron_2d.temperature = iron_ball.temperature
    iron_eco = ecology.add_body(iron_2d, albedo=0.3, water_mass=0.0)
    
    water_2d = circle_body([2, 0], 0.4, mass=2.5, material=Mat2D(specific_heat=4186.0, thermal_conductivity=0.6))
    water_2d.temperature = water_ball.temperature
    water_eco = ecology.add_body(water_2d, albedo=0.1, water_mass=0.5)
    
    wood_2d = circle_body([-2, 0], 0.3, mass=0.5, material=Mat2D(specific_heat=1700.0, thermal_conductivity=0.15))
    wood_2d.temperature = wood_block.temperature
    wood_eco = ecology.add_body(wood_2d, albedo=0.6, water_mass=0.0)
    
    print(f"  Ecology system initialized with {len(ecology.ecological_bodies)} bodies")
    
    # --- Simulation Loop ---
    print("\n" + "=" * 60)
    print("Running simulation: 5 seconds (300 steps at 60 FPS)")
    print("=" * 60)
    
    dt = 1/60.0
    total_time = 5.0
    steps = int(total_time / dt)
    
    # Record initial state
    initial_fe = 0.5 * iron_ball.mass * np.linalg.norm(iron_ball.vel)**2 + iron_ball.mass * 9.81 * iron_ball.pos[2]
    print(f"\nInitial iron ball energy: {initial_fe:.2f} J")
    print(f"Initial iron ball position: [{iron_ball.pos[0]:.2f}, {iron_ball.pos[1]:.2f}, {iron_ball.pos[2]:.2f}]")
    
    for step in range(steps):
        t = step * dt
        
        # 1. Step 3D physics
        world.step(1)
        
        # 2. Step thermal coupling (every 10 frames for performance)
        if step % 10 == 0:
            # Heat transfer between iron and water (simplified)
            thermal_system.conduct_between(iron_ball, water_ball, dt * 10)
        
        # 3. Step chemistry (exothermic reaction releases heat)
        if step % 60 == 0:  # once per second
            heat = step_chemistry(chem_system, 1.0)
            if heat > 0:
                # Add reaction heat to iron ball (combustion near it)
                iron_ball.temperature += heat / (iron_ball.mass * 450.0)
        
        # 4. Step ecology (solar, evaporation)
        if step % 60 == 0:  # once per second
            ecology.step(1.0)
            couple_body_environment(iron_eco, ecology.environment, 1.0)
            couple_body_environment(water_eco, ecology.environment, 1.0)
        
        # Print progress every second
        if step % 60 == 0:
            ke = 0.5 * iron_ball.mass * np.linalg.norm(iron_ball.vel)**2
            pe = iron_ball.mass * 9.81 * iron_ball.pos[2]
            print(f"  t={t:.1f}s: iron pos=[{iron_ball.pos[0]:.2f}, {iron_ball.pos[1]:.2f}, {iron_ball.pos[2]:.2f}], "
                  f"T={iron_ball.temperature:.1f}K, KE={ke:.1f}J, PE={pe:.1f}J")
    
    # --- Final Report ---
    print("\n" + "=" * 60)
    print("Simulation Complete - Final State")
    print("=" * 60)
    
    final_fe = 0.5 * iron_ball.mass * np.linalg.norm(iron_ball.vel)**2 + iron_ball.mass * 9.81 * iron_ball.pos[2]
    print(f"\nIron ball:")
    print(f"  Position: [{iron_ball.pos[0]:.2f}, {iron_ball.pos[1]:.2f}, {iron_ball.pos[2]:.2f}]")
    print(f"  Temperature: {iron_ball.temperature:.1f} K (started at 500 K)")
    print(f"  Final energy: {final_fe:.2f} J (started at {initial_fe:.2f} J)")
    
    print(f"\nWater ball:")
    print(f"  Position: [{water_ball.pos[0]:.2f}, {water_ball.pos[1]:.2f}, {water_ball.pos[2]:.2f}]")
    print(f"  Temperature: {water_ball.temperature:.1f} K (started at 273 K)")
    
    print(f"\nWood block:")
    print(f"  Position: [{wood_block.pos[0]:.2f}, {wood_block.pos[1]:.2f}, {wood_block.pos[2]:.2f}]")
    print(f"  Temperature: {wood_block.temperature:.1f} K")
    
    print(f"\nChemistry:")
    print(f"  Methane remaining: {chem_system.masses.get('methane', 0):.4f} kg")
    print(f"  System temperature: {chem_system.temperature:.1f} K")
    
    print(f"\nEcology:")
    print(f"  Environment temperature: {ecology.environment.temperature:.1f} K")
    print(f"  Time elapsed: {ecology.environment.time:.0f} s")
    
    # Conservation check
    energy_diff = abs(final_fe - initial_fe)
    print(f"\nEnergy conservation: dE = {energy_diff:.2f} J")
    if energy_diff < 100:
        print("[OK] Energy reasonably conserved (within numerical tolerance)")
    else:
        print("[WARN] Significant energy change - check collision/thermal parameters")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
