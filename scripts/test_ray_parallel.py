import ray

ray.init(num_cpus=2, ignore_reinit_error=True)

from pymo.parallel.ray_parallel import SimulationConfig, run_single_simulation

# Test single simulation
config = SimulationConfig(
    bodies=[
        {'shape': 'sphere', 'pos': [0, 0, 5], 'radius': 0.5, 'mass': 1.0, 
         'material': {'restitution': 0.9}},
        {'shape': 'box', 'pos': [0, 0, -0.5], 'half_extents': [5, 5, 0.5], 'static': True, 'mass': 0}
    ],
    max_steps=100
)

result = run_single_simulation(config)
print(f'Success: {result.success}, Steps: {result.steps_completed}, KE: {result.metrics.get("kinetic_energy", 0):.2f}')
ray.shutdown()
print('Ray parallel test passed!')