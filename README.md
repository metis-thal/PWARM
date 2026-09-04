# pymo — Physical World AI Reasoning Model / 物理世界AI推理模型

A self-evolving virtual physics world. All phenomena (rigid body, fluid, thermal, collision, deformation, reaction) emerge from bottom-up differential equations — no hardcoded animations, no preset events. An AI layer observes the simulated world, discovers physical laws, and continuously refines its world model.

一个自演化的虚拟物理世界。所有现象（刚体、流体、热力学、碰撞、形变、反应）都从底层微分方程中涌现——没有硬编码的动画，没有预设事件。AI层观察模拟世界，发现物理定律，并持续优化其世界模型。

## Architecture (Five Immutable Layers) / 五层不可变架构

```
src/pymo/
├── kernel/     # Legacy 2D/3D physics kernels (deprecated, kept for compatibility)
│               # 旧版 2D/3D 物理内核（已弃用，保留兼容性）
├── geology/    # Geology system: stratigraphy, thermal conduction, erosion, tectonics
│               # 地质系统：地层学、热传导、侵蚀、构造运动
├── rules/      # Multi-discipline: mechanics, thermodynamics, fluids, materials, chemistry
│               # 多学科规则：力学、热力学、流体、材料、化学
├── ai/         # AI reasoning/evolution: observer, hypothesis, verification, experimentation
│               # AI推理/演化：观察器、假设、验证、实验
├── physics/    # NEW: Unified multi-physics engine (Genesis-inspired)
│   ├── core/   # Scene, State, Entity, Component (single source of truth)
│   │           # 场景、状态、实体、组件（唯一真值源）
│   ├── solvers/ # Rigid, SPH, FEM, MPM, PBD, Thermal, Chemistry, Geology
│   │           # 刚体、SPH、FEM、MPM、PBD、热力学、化学、地质
│   ├── coupling/ # Explicit multi-physics coupler (rigid↔SPH, thermal↔all, etc.)
│   │           # 显式多物理耦合器
│   ├── collision/ # Unified collision (SAP + GJK/EPA + CCD)
│   │           # 统一碰撞检测
│   └── integrator/ # TimeStepper (sub-steps + coupling iterations)
│               # 时间步进器
├── interface/  # Asset parsers (URDF/MJCF/GLTF), GUI, Sensors, Parallel envs
│               # 资产解析、GUI、传感器、并行环境
└── viz/        # Visualization: OpenGL GPU instancing, PBR, ray-tracing
              # 可视化：OpenGL GPU实例化、PBR、光线追踪
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

### AI Law Discovery (NEW: Works with pymo.physics) / AI定律发现

```python
from pymo.physics import WorldEngine, WorldEngineConfig
from pymo.ai import LawDiscovery, ClosedLoopAI
from pymo.physics import create_free_fall_experiment

# Create physics engine and observe
config = WorldEngineConfig(dt=1/60, substeps=1, rigid={'enabled': True})
engine = WorldEngine(config)

dataset = create_free_fall_experiment(engine, height=10.0, mass=1.0, n_steps=100)

# AI discovers y(t) = f(t) / AI发现y(t) = f(t)
ld = LawDiscovery()
law = ld.discover_from_observation(dataset.time(), dataset.get('rigid0.pos.z'))
print(f"Discovered: {law.expression}")  # e.g. "-4.905*t^2"
print(f"R²: {law.r2:.6f}")  # ≈ 1.0

# Closed-loop verification
ai = ClosedLoopAI()
results = ai.run(dataset, quantities=['rigid0.pos.z'])
print(results[0].law.expression)
print(f"Test error: {results[0].relative_error:.6f}")  # ≈ 0.0000
```

### SPH Fluid / SPH流体

```python
from pymo.rules.fluid import SPHSystem, SPHParams, create_water_column

system = create_water_column(x=0, y=0, width=5, height=10, spacing=0.1)
system.step(dt=0.01)  # one SPH step / 一个SPH步长
print(f"Density range: {system.min_density_ratio():.2f} - {system.max_density_ratio():.2f}")
```

### WorldEngine (NEW: Unified Multi-Physics Engine) / WorldEngine 统一多物理引擎

```python
from pymo.physics import WorldEngine, WorldEngineConfig

# Configure all subsystems
config = WorldEngineConfig(
    dt=1/60,
    substeps=1,
    gravity=(0.0, 0.0, -9.81),
    rigid={'enabled': True},
    sph={'enabled': False},
    fem={'enabled': False},
    mpm={'enabled': False},
    pbd={'enabled': False},
    thermal={'enabled': False},
    chemistry={'enabled': False},
    geology={'enabled': False},
    enable_ai=True,
)
engine = WorldEngine(config)

# Create entities
engine.create_rigid_body((0, 0, 5), mass=1.0, shape='sphere', shape_params={'radius': 0.5})
engine.finalize_setup()

# Run simulation — all solvers coupled via explicit coupler
for _ in range(300):
    state = engine.tick()
    print(f"t={state.t:.3f}, pos={state.rigid_pos}")

# Checkpoint for reproducibility
engine.checkpoint("checkpoint.pt")
engine.restore("checkpoint.pt")
```

### Legacy WorldEngine (deprecated)
```python
from pymo.kernel.world_engine import WorldEngine as LegacyWorldEngine
# ... old API still works but deprecated
```

### OpenGL GPU-Instanced Rendering / OpenGL GPU实例化渲染

```python
from pymo.viz.gl_renderer import GLRenderer, RendererConfig
from pymo.viz.snapshot import build_snapshot_from_world
from pymo.viz.double_buffer import DoubleBuffer

buffer = DoubleBuffer()
renderer = GLRenderer(buffer, RendererConfig(window_size=(1280, 720)))

# Render loop — reads immutable snapshots, drives nothing
while renderer.running:
    snapshot = build_snapshot_from_world(engine.world, engine.geology_solver)
    buffer.swap(snapshot)
    renderer.render_frame()
```

### Geology Module / 地质模块

```python
from pymo.geology import GeologySolver, GeologySolverConfig, create_stratified_grid
from pymo.geology.rock_materials import get_material_properties_for_gpu

# Create a 32×32×16 geology grid with sedimentary layers
grid = create_stratified_grid(nx=32, ny=32, nz=16, cell_size=10.0)
solver = GeologySolver(grid, GeologySolverConfig())

# Step geological processes (thermal conduction + sedimentation)
solver.step(dt_years=1000.0)

# Query rock properties
props = get_material_properties_for_gpu()  # → (rock_ids, colors, emissivities)
```

### Module Reference / 模块参考

| Module / 模块 | Description / 说明 |
|--------------|-------------------|
| `kernel.bodies` | 2D rigid bodies: circle, polygon / 2D刚体：圆、多边形 (legacy) |
| `kernel.bodies3d` | 3D rigid bodies: sphere, box, cylinder, convex hull / 3D刚体 (legacy) |
| `kernel.collision` | 2D SAT collision detection / 2D SAT碰撞检测 (legacy) |
| `kernel.collision3d` | 3D GJK/EPA collision detection / 3D GJK/EPA碰撞检测 (legacy) |
| `kernel.solver` | 2D impulse-based contact solver / 2D基于冲量的接触求解器 (legacy) |
| `kernel.world` | 2D physics world / 2D物理世界 (legacy) |
| `kernel.world3d` | 3D physics world / 3D物理世界 (legacy) |
| `kernel.world_engine` | Legacy unified orchestration (deprecated) / 旧版统一编排 |
| `kernel.math3d` | Quaternion math, GJK support / 四元数数学、GJK支撑 |
| `kernel.integrators` | Velocity-Verlet integrator (Numba JIT) / 速度Verlet积分器 |
| `physics.core` | **NEW**: Scene, State, Entity, Component / 场景、状态、实体、组件 |
| `physics.solvers` | **NEW**: Rigid, SPH, FEM, MPM, PBD, Thermal, Chemistry, Geology |
| `physics.coupling` | **NEW**: Explicit multi-physics coupler / 显式多物理耦合器 |
| `physics.collision` | **NEW**: Unified SAP + GJK/EPA + CCD / 统一碰撞检测 |
| `physics.integrator` | **NEW**: TimeStepper with sub-steps / 时间步进器 |
| `physics.interface` | **NEW**: URDF/MJCF/GLTF parsers, GUI, Sensors, Parallel envs |
| `physics.ai` | **NEW**: Observer, LawDiscovery, ClosedLoopAI / 观察器、定律发现、闭环AI |
| `geology.rock_materials` | Rock material definitions (granite, basalt, sandstone, etc.) / 岩石材料库 |
| `geology.geology_grid` | 3D voxel grid with flat-array storage / 3D体素网格 |
| `geology.geology_solver` | Geology process orchestrator / 地质过程编排器 |
| `geology.processes.thermal` | Implicit heat conduction PDE (scipy.sparse) / 隐式热传导PDE |
| `geology.processes.sedimentation` | Stratigraphic layering / 地层层序 |
| `rules.thermal` | Fourier heat conduction / 傅里叶热传导 |
| `rules.fluid` | SPH fluid solver / SPH流体求解器 |
| `rules.fracture` | Brittle fracture mechanics / 脆性断裂力学 |
| `ai.observer` | World state observer (legacy) / 世界状态观察器 |
| `ai.law_discovery` | Symbolic regression / 符号回归 |
| `ai.closed_loop` | Closed-loop AI reasoning / 闭环AI推理 |
| `ai.experiment` | Autonomous experimentation / 自主实验 |
| `viz.gl_renderer` | OpenGL GPU-instanced renderer (PBR + frustum cull) / OpenGL GPU实例化渲染器 |
| `viz.snapshot` | Double-buffered immutable scene snapshots / 双缓冲不可变场景快照 |
| `viz.viewer` | 2D PyVista renderer / 2D PyVista渲染器 |
| `viz.viewer3d` | 3D PyVista renderer / 3D PyVista渲染器 |
| `parallel.ray_parallel` | Ray distributed simulation / Ray分布式模拟 |

## Running Tests / 运行测试

```bash
pytest -q                    # run all tests / 运行所有测试
pytest tests/kernel/         # kernel tests only / 仅内核测试
pytest tests/rules/          # rules tests only / 仅规则测试
pytest tests/ai/             # AI tests only / 仅AI测试
pytest tests/geology/        # geology tests only / 仅地质测试
pytest tests/viz/            # viz tests only / 仅渲染测试
```

## Project Status / 项目状态

| Phase / 阶段 | Status / 状态 | Tests / 测试 |
|-------------|--------------|-------------|
| P0: Technical research / 技术调研 | Done / 完成 | 5 reports |
| P1: 2D MVP / 2D最小可行产品 | Done / 完成 | 16 pass |
| P2.2: Multi-physics rules / 多物理规则 | Done / 完成 | 22 pass |
| P2.1: 3D physics kernel / 3D物理内核 | Done / 完成 | 10 pass |
| P3: OpenGL rendering / OpenGL渲染 | Done / 完成 | 190 pass |
| P4.1: Ray parallel / Ray并行 | Done / 完成 | 4 pass |
| P5: Geology module / 地质模块 | Phase 1 / 第一阶段 | 7 pass (4+3 skip) |
| P6: Terrain viewer / 地形可视化器 | Done / 完成 | 4 pass (1 real data + 1 simulation + 1 viewer + 1 exe) |
| **P7: Genesis-inspired Multi-Physics Engine** | **Done / 完成** | **4 pass (free-fall, collision, conservation, architecture)** |
| **Total** | | **198+ pass** |

### Terrain Evolution Viewer / 地形演化可视化器

Interactive 3D terrain viewer with real-world heightmaps and geological simulation.

```bash
# Run from source / 从源码运行
python scripts/terrain_viewer.py --dataset everest

# Or use prebuilt EXE / 或使用预编译可执行文件
dist/PWARM_TerrainViewer.exe --dataset everest
```

| Key / 按键 | Action / 动作 |
|-----------|--------------|
| ← → | Navigate snapshots / 浏览时间步快照 |
| ↑ ↓ | Switch dataset (Everest, Grand Canyon, Mt. Fuji, Zhangjiajie) |
| M | Toggle realistic ↔ heatmap coloring / 切换着色模式 |
| T | Toggle professional ↔ layperson UI / 切换专业/科普界面 |
| S | Toggle cross-section / 切换剖面视图 |
| R | Toggle rain / 开关降雨 |
| W | Toggle snow / 开关降雪 |
| V | Toggle rivers / 开关河流 |
| A | Auto-play simulation / 自动播放 |
| Q | Quit / 退出 |

Features:
- **4 real-world datasets**: SRTM heightmaps (Everest, Grand Canyon, Mt. Fuji, Zhangjiajie) at 512×512
- **1M-year simulation**: Stream Power Law erosion + tectonic uplift on Everest, generating 17 time-step snapshots (t0 → t1M)
- **Slope-based terrain coloring**: Vegetation zones, bare rock on steep slopes, snow caps, natural noise
- **River extraction**: D8 flow accumulation algorithm computes drainage areas and renders river lines
- **Weather effects**: Animated rain (2000 particles) and snow (1500 particles) via PyVista timer events
- **Dual-mode UI**: Professional mode shows quantitative data (elevation, temperature, slope); layperson mode shows natural language descriptions

## Physics Engine Demo / 物理引擎演示

```bash
# Run from source / 从源码运行
python scripts/demo_physics_engine.py

# Or use prebuilt EXE / 或使用预编译可执行文件
dist/PWARM_PhysicsEngine.exe
```

| Demo | Description / 说明 |
|------|-------------------|
| Free-Fall + AI | Drop ball → AI discovers z(t) = -4.905*t² (R²≈1.0) |
| Two-Body Collision | Opposite velocities → detect bounce |
| Energy Conservation | Monitor KE/momentum during 3-body drop |
| Multi-Physics Architecture | Verify all subsystems (rigid, SPH, FEM, MPM, PBD, thermal, chemistry, geology) |

## Algorithms & Data Structures / 算法与数据结构

### Collision Detection / 碰撞检测

| Algorithm | Complexity | Description / 说明 |
|-----------|-----------|-------------------|
| **SAP (Sweep and Prune)** | O(n log n) insert, O(n+k) query | Broad phase: maintain sorted AABB min/max on each axis |
| **GJK (Gilbert-Johnson-Keerthi)** | O(n) iterations | Narrow phase: exact contact for convex shapes via Minkowski difference |
| **EPA (Expanding Polytope Algorithm)** | O(n²) worst case | Extends GJK to compute penetration depth and contact normal |
| **SAT (Separating Axis Theorem)** | O(15) for box-box | Fast OBB-OBB: test 15 axes (3+3 face normals + 9 cross products) |
| **Conservative CCD** | O(n) sweep | Continuous collision detection for fast-moving objects (prevent tunneling) |

### Physics Solvers / 物理求解器

| Solver | Algorithm | Data Structure | Description / 说明 |
|--------|-----------|----------------|-------------------|
| **RigidSolver** | Velocity-Verlet + impulse contacts | `(N,3)` pos/vel arrays, `(N,4)` quaternions | Rigid body dynamics with quaternion integration |
| **SPHSolver** | WCSPH (Weakly Compressible SPH) | KD-tree neighbors, `(N,3)` particle arrays | Fluid simulation with cubic kernel, boundary reflection |
| **FEMSolver** | Implicit Euler (placeholder) | `(E,4)` tetrahedra, `(E,3,3)` deformation gradient | Deformable solids |
| **MPMSolver** | Material Point Method (placeholder) | `(N,3)` particles + background grid | Sand, snow, clay simulation |
| **PBDSolver** | Position-Based Dynamics (placeholder) | Distance/bending constraints | Cloth, hair simulation |
| **ThermalSolver** | Implicit heat equation | `(N,)` temperature array | Fourier heat conduction |
| **ChemistrySolver** | Reaction-diffusion (placeholder) | `(N,num_species)` concentration | Chemical reactions |
| **GeologySolver** | Stratigraphy + erosion (placeholder) | `(Nx,Ny,Nz)` voxel grid | Geological processes |

### State Management / 状态管理

| Structure | Description / 说明 |
|-----------|-------------------|
| **Double Buffer** | Simulation writes to `write` buffer, render reads from `read` buffer. Swap after each frame. |
| **Immutable State** | `State` dataclass with flat contiguous arrays. Copy-on-swap for render thread safety. |
| **Entity-Component** | `ComponentMask` (IntFlag) enables/disables subsystems per entity. `EntityManager` for archetype queries. |
| **Global Quantities** | `GlobalQuantities` tracks mass, energy, momentum for conservation monitoring. |

### AI Layer / AI层

| Algorithm | Description / 说明 |
|-----------|-------------------|
| **Symbolic Regression** | PolynomialBackend (default) or GplearnRefineBackend. Discovers expressions like z(t) = -4.905*t² |
| **Closed-Loop Verification** | Train/test split → discover law on train → predict on test → measure relative error |
| **WorldObserver** | Samples ground-truth state at intervals. Produces TimeSeriesDataset for AI. |

### Multi-Physics Coupling / 多物理耦合

| Pair | Interaction | Description / 说明 |
|------|-------------|-------------------|
| rigid↔sph | Boundary particles, buoyancy, drag | SPH particles interact with rigid surfaces |
| rigid↔fem/mpm | Contact constraints, friction | Deformable bodies collide with rigid objects |
| sph↔fem/mpm | Fluid pressure on deformable, porosity | Fluid-structure interaction |
| thermal↔all | Heat flux, latent heat, reaction heat | Temperature coupling across all materials |
| chemistry↔sph | Species diffusion, reaction sources | Chemical transport in fluid |
| geology↔thermal | Crustal heat flow, radiogenic heating | Geological heat sources |

## EXE Build Results / EXE构建结果

| EXE | Size | Description / 说明 |
|-----|------|-------------------|
| `PWARM_PhysicsEngine.exe` | **53 MB** | Console demo: WorldEngine + AI law discovery + collision + conservation |
| `PWARM_TerrainViewer.exe` | **209 MB** | 3D terrain viewer with PyVista/VTK |

## Recent Changes / 近期变更

### v0.6 — Genesis-inspired Multi-Physics Engine / 类经典物理世界引擎

- **NEW: `pymo.physics`** — Unified multi-physics engine (Genesis-inspired architecture)
  - Core: Scene, State, Entity, ComponentMask, EntityManager (single source of truth)
  - RigidSolver: impulse-based contacts, Velocity-Verlet, quaternion integration
  - SPHSolver: WCSPH with KD-tree neighbor search, pressure/viscosity/surface tension
  - Stub solvers: FEM, MPM, PBD, Thermal, Chemistry, Geology (architecture ready)
  - Explicit Coupler: rigid↔SPH, rigid↔FEM, thermal↔all, chemistry↔SPH, geology↔thermal
  - CollisionSystem: SAP broad phase + GJK/EPA narrow phase + SAT (box-box) + CCD
  - TimeStepper: sub-steps + Gauss-Seidel coupling iterations + conservation monitoring
  - WorldEngine: unified entry point with double-buffered immutable snapshots
- **AI Layer** (`pymo.physics.ai`):
  - WorldObserver: samples ground-truth state from WorldEngine
  - LawDiscovery: polynomial backend (no external deps) — R²≈1.0 on free-fall
  - ClosedLoopAI: observe→discover→predict→compare — test error 4.35e-7 (PASS)
  - AutonomousExperimenter: ParameterSpace, ExperimentRunner, hypothesis evaluation
- **Interface** (`pymo.interface`): URDF/MJCF/GLTF parsers, GUI, Sensors, Parallel environments
- **Bugfixes**: rigid.py angular velocity broadcast error, ai.py collision experiment API
- **EXEs**: `PWARM_PhysicsEngine.exe` (53MB), `PWARM_TerrainViewer.exe` (209MB rebuilt)

### v0.5 — Terrain Evolution Viewer / 地形演化可视化器

- **Real-world terrain data** (`data/terrain/`): 4 SRTM heightmaps downloaded and processed
- **Everest simulation** (`scripts/simulate_everest.py`): 1M-year geological simulation (tectonics + erosion)
- **3D viewer** (`scripts/terrain_viewer.py`): Interactive PyVista viewer with weather, rivers, dual-mode
- **Standalone EXE** (`dist/PWARM_TerrainViewer.exe`): PyInstaller-built, no dependencies required

### v0.4 — Geology Module + OpenGL Rendering / 地质模块 + OpenGL渲染

- **Geology system** (`src/pymo/geology/`): Stratigraphy, thermal conduction (implicit Euler PDE), sedimentation processes, 9 built-in rock types
- **OpenGL renderer** (`src/pymo/viz/gl_renderer.py`): GPU-instanced rendering, PBR shaders with temperature emission, frustum culling, double-buffered snapshots
- **WorldEngine** (`src/pymo/kernel/world_engine.py`): Unified orchestration layer coupling physics, SPH, chemistry, ecology, and geology
- **Bug fixes**: Model matrix column-major transpose, scale propagation, view/projection matrix transpose for moderngl compatibility

## License / 许可证

MIT
