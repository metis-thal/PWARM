# pymo — Physical World AI Reasoning Model / 物理世界AI推理模型

A self-evolving virtual physics world. All phenomena (rigid body, fluid, thermal, collision, deformation, reaction) emerge from bottom-up differential equations — no hardcoded animations, no preset events. An AI layer observes the simulated world, discovers physical laws, and continuously refines its world model.

一个自演化的虚拟物理世界。所有现象（刚体、流体、热力学、碰撞、形变、反应）都从底层微分方程中涌现——没有硬编码的动画，没有预设事件。AI层观察模拟世界，发现物理定律，并持续优化其世界模型。

## Architecture (Four Immutable Layers) / 四层不可变架构

```
src/pymo/
├── kernel/   # Math/physics core: ODE/PDE solvers, Verlet integration, conservation (GROUND TRUTH)
│             # 数学/物理核心：ODE/PDE求解器、Verlet积分、守恒定律（真值层）
├── rules/    # Multi-discipline: mechanics, thermodynamics, fluids, materials, chemistry
│             # 多学科规则：力学、热力学、流体、材料、化学
├── ai/       # AI reasoning/evolution: observer, hypothesis, verification, experimentation
│             # AI推理/演化：观察器、假设、验证、实验
└── viz/      # Visualization/observation: 3D render, data panels, state recording, replay
              # 可视化/观察：3D渲染、数据面板、状态录制、回放
```

## Core Rules / 核心规则

1. **Zero hardcoded phenomena** — combustion, fracture, flow MUST emerge from equations.
   **零硬编码现象**——燃烧、断裂、流动必须从方程中涌现。
2. **Ground truth / AI separation** — physics engine is absolute truth; AI is a learned approximation; never conflate.
   **真值/AI分离**——物理引擎是绝对真值；AI是学习到的近似；绝不混淆。
3. **Visual acceptance every phase** — no headless development; every phase demoable.
   **每阶段可视化验收**——禁止无头开发；每阶段必须可演示。

## Setup / 安装

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[core,ai,viz,parallel,dev]"
```

### Dependencies / 依赖

| Package | Purpose / 用途 |
|---------|---------------|
| numpy, scipy, numba | Core math & JIT compilation / 核心数学与JIT编译 |
| gplearn | Symbolic regression (fallback when Julia unavailable) / 符号回归 |
| pyvista | 3D visualization / 3D可视化 |
| ray | Distributed parallel simulation / 分布式并行模拟 |
| pytest, ruff | Testing & linting / 测试与代码检查 |

## Quick Start / 快速开始

### 2D Free Fall / 2D自由落体

```python
from pymo.kernel.bodies import circle_body
from pymo.kernel.world import World
import numpy as np

world = World(gravity=np.array([0.0, -9.81]), dt=1/60.0)
world.add(circle_body([0.0, 10.0], radius=0.5, mass=1.0))

world.step(300)  # simulate 5 seconds / 模拟5秒
b = world.bodies[0]
print(f"Height: {b.pos[1]:.2f} m")  # ≈ 10 - 0.5*9.81*25 ≈ -112.6
```

### 3D Falling Spheres / 3D落球

```python
from pymo.kernel.bodies3d import sphere_body, box_body, Material
from pymo.kernel.world3d import World3D
import numpy as np

world = World3D(gravity=np.array([0.0, 0.0, -9.81]))
world.add(box_body([0, 0, -0.5], half_extents=[10, 10, 0.5], static=True))  # ground
world.add(sphere_body([0, 0, 5], radius=0.5, mass=1.0))

world.step(600)
print(f"Height: {world.bodies[1].pos[2]:.2f} m")
```

### AI Law Discovery / AI定律发现

```python
from pymo.ai.observer import WorldObserver
from pymo.ai.closed_loop import ClosedLoopAI

# Create world and observe / 创建世界并观察
from pymo.kernel.bodies import circle_body
from pymo.kernel.world import World
import numpy as np

w = World(gravity=np.array([0.0, -9.81]), dt=0.01)
w.add(circle_body([0.0, 10.0], 0.5, mass=1.0))
dataset = WorldObserver(w, sample_every=1).observe(300)

# AI discovers y(t) = f(t) / AI发现y(t) = f(t)
ai = ClosedLoopAI()
results = ai.run(dataset, quantities=["body0.pos.y"])
print(results[0].law.expression)  # e.g. "10.0 - 4.905*t^2"
print(f"Error: {results[0].relative_error:.4f}")  # ≈ 0.0000
```

### SPH Fluid / SPH流体

```python
from pymo.rules.fluid import SPHSystem, SPHParams, create_water_column

system = create_water_column(x=0, y=0, width=5, height=10, spacing=0.1)
system.step(dt=0.01)  # one SPH step / 一个SPH步长
print(f"Density range: {system.min_density_ratio():.2f} - {system.max_density_ratio():.2f}")
```

## Module Reference / 模块参考

| Module / 模块 | Description / 说明 |
|--------------|-------------------|
| `kernel.bodies` | 2D rigid bodies: circle, polygon / 2D刚体：圆、多边形 |
| `kernel.bodies3d` | 3D rigid bodies: sphere, box, cylinder, convex hull / 3D刚体 |
| `kernel.collision` | 2D SAT collision detection / 2D SAT碰撞检测 |
| `kernel.collision3d` | 3D GJK/EPA collision detection / 3D GJK/EPA碰撞检测 |
| `kernel.solver` | 2D impulse-based contact solver / 2D基于冲量的接触求解器 |
| `kernel.world` | 2D physics world / 2D物理世界 |
| `kernel.world3d` | 3D physics world / 3D物理世界 |
| `kernel.math3d` | Quaternion math, GJK support / 四元数数学、GJK支撑 |
| `kernel.integrators` | Velocity-Verlet integrator (Numba JIT) / 速度Verlet积分器 |
| `rules.thermal` | Fourier heat conduction / 傅里叶热传导 |
| `rules.fluid` | SPH fluid solver / SPH流体求解器 |
| `rules.fracture` | Brittle fracture mechanics / 脆性断裂力学 |
| `ai.observer` | World state observer / 世界状态观察器 |
| `ai.law_discovery` | Symbolic regression / 符号回归 |
| `ai.closed_loop` | Closed-loop AI reasoning / 闭环AI推理 |
| `viz.viewer` | 2D PyVista renderer / 2D PyVista渲染器 |
| `viz.viewer3d` | 3D PyVista renderer / 3D PyVista渲染器 |
| `parallel.ray_parallel` | Ray distributed simulation / Ray分布式模拟 |

## Running Tests / 运行测试

```bash
pytest -q                    # run all tests / 运行所有测试
pytest tests/kernel/         # kernel tests only / 仅内核测试
pytest tests/rules/          # rules tests only / 仅规则测试
pytest tests/ai/             # AI tests only / 仅AI测试
```

## Project Status / 项目状态

| Phase / 阶段 | Status / 状态 | Tests / 测试 |
|-------------|--------------|-------------|
| P0: Technical research / 技术调研 | Done / 完成 | 5 reports |
| P1: 2D MVP / 2D最小可行产品 | Done / 完成 | 16 pass |
| P2.2: Multi-physics rules / 多物理规则 | Done / 完成 | 22 pass |
| P2.1: 3D physics kernel / 3D物理内核 | Done / 完成 | 10 pass |
| P4.1: Ray parallel / Ray并行 | Done / 完成 | 4 pass |
| **Total** | | **52 pass** |

## License / 许可证

MIT
