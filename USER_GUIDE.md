# pymo User Guide / pymo用户手册

Complete bilingual (中英双语) guide to using the pymo physics simulation framework.

---

## Table of Contents / 目录

1. [Installation / 安装](#1-installation--安装)
2. [Architecture Overview / 架构概览](#2-architecture-overview--架构概览)
3. [2D Physics / 2D物理](#3-2d-physics--2d物理)
4. [3D Physics / 3D物理](#4-3d-physics--3d物理)
5. [Collision Detection / 碰撞检测](#5-collision-detection--碰撞检测)
6. [Multi-Physics Rules / 多物理规则](#6-multi-physics-rules--多物理规则)
7. [AI Layer / AI层](#7-ai-layer--ai层)
8. [Visualization / 可视化](#8-visualization--可视化)
9. [Parallel Simulation / 并行模拟](#9-parallel-simulation--并行模拟)
10. [API Reference / API参考](#10-api-reference--api参考)

---

## 1. Installation / 安装

### Prerequisites / 前提条件

- Python 3.10-3.11 (Python 3.14 not supported due to missing Numba/SciPy wheels)
- 64-bit OS (Windows/Linux/macOS)
- 4GB+ RAM recommended for SPH fluid simulations

Python 3.10-3.11（Python 3.14因缺少Numba/SciPy轮子暂不支持）
64位操作系统（Windows/Linux/macOS）
SPH流体模拟建议4GB+内存

### Setup / 安装步骤

```bash
# Clone the repository / 克隆仓库
git clone <repo-url>
cd pymo

# Create virtual environment / 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate          # Windows
source .venv/bin/activate         # Linux/macOS

# Install with all optional dependencies / 安装所有可选依赖
pip install -e ".[core,ai,viz,parallel,dev]"

# Verify installation / 验证安装
pytest -q   # Should show 52 passed / 应显示52个通过
```

### China Network Setup / 中国网络配置

If you are in China and pip install is slow, configure the Aliyun mirror:

如果在中国且pip安装速度慢，配置阿里云镜像：

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

---

## 2. Architecture Overview / 架构概览

pymo follows a strict four-layer architecture with one rule: **the physics kernel is ground truth; everything else is a learned or extended approximation.**

pymo采用严格的四层架构，核心规则是：**物理内核是真值；其他一切都是学习或扩展的近似。**

```
┌─────────────────────────────────────────────────┐
│  Viz Layer (viz/)                                │
│  可视化层                                          │
│  PyVista 2D/3D rendering, data panels            │
│  PyVista 2D/3D渲染、数据面板                        │
├─────────────────────────────────────────────────┤
│  AI Layer (ai/)                                  │
│  AI层                                             │
│  Observer → Symbolic Regression → Prediction      │
│  观察器 → 符号回归 → 预测                           │
├─────────────────────────────────────────────────┤
│  Rules Layer (rules/)                            │
│  规则层                                           │
│  Thermal, Fluid (SPH), Fracture, Materials       │
│  热力学、流体(SPH)、断裂、材料                      │
├─────────────────────────────────────────────────┤
│  Kernel Layer (kernel/) — GROUND TRUTH            │
│  内核层 — 真值                                     │
│  Bodies, Collision, Solver, World, Integrators    │
│  刚体、碰撞、求解器、世界、积分器                     │
└─────────────────────────────────────────────────┘
```

### Data Flow / 数据流

```
World.step(n)
    │
    ├─ 1. Integration (gravity + forces → velocity → position)
    │     积分（重力+力→速度→位置）
    │
    ├─ 2. Collision Detection (SAT/GJK/EPA)
    │     碰撞检测（SAT/GJK/EPA）
    │
    ├─ 3. Contact Resolution (impulse-based, momentum-conserving)
    │     接触求解（基于冲量，动量守恒）
    │
    ├─ 4. Rules Extension (thermal conduction, fracture check)
    │     规则扩展（热传导、断裂检查）
    │
    └─ 5. Record State (for AI observer / visualization)
          记录状态（供AI观察器/可视化使用）
```

---

## 3. 2D Physics / 2D物理

### Creating Bodies / 创建刚体

```python
from pymo.kernel.bodies import circle_body, box_body, Body, Material
import numpy as np

# Circle body / 圆形刚体
ball = circle_body(
    pos=[0.0, 10.0],       # position (x, y) / 位置
    radius=0.5,            # radius in meters / 半径（米）
    mass=1.0,              # mass in kg / 质量（千克）
    material=Material(restitution=0.8, friction=0.3),
    static=False           # dynamic body / 动态刚体
)

# Box body / 盒形刚体
platform = box_body(
    pos=[0.0, -0.5],       # center position / 中心位置
    half_w=10.0,           # half-width / 半宽
    half_h=0.5,            # half-height / 半高
    mass=0,                # mass=0 + static=True = immovable / 质量0+静态=不可移动
    static=True
)

# Custom polygon / 自定义多边形
hexagon = Body(
    pos=np.array([5.0, 0.0]),
    mass=2.0,
    vertices=np.array([      # local coordinates, CCW / 局部坐标，逆时针
        [1.0, 0.0],
        [0.5, 0.866],
        [-0.5, 0.866],
        [-1.0, 0.0],
        [-0.5, -0.866],
        [0.5, -0.866],
    ]),
    material=Material(restitution=0.5)
)
```

### Material Properties / 材料属性

```python
@dataclass
class Material:
    restitution: float = 0.3       # bounciness 0..1 / 弹性系数
    friction: float = 0.4          # Coulomb friction / 库仑摩擦系数
    density: float = 1.0           # mass per unit area / 面密度

    # Thermal (used by rules/thermal.py) / 热力学（用于热传导规则）
    specific_heat: float = 1000.0   # J/(kg*K) / 比热容
    thermal_conductivity: float = 0.0  # W/(m*K), 0=insulator / 热导率，0=绝缘体

    # Mechanical (used by rules/fracture.py) / 力学（用于断裂规则）
    young_modulus: float = 1e9     # Pa, stiffness / 杨氏模量
    hardness: float = 1e6          # Pa / 硬度
    fracture_toughness: float = 1e6   # Pa*sqrt(m) / 断裂韧性
    brittleness: float = 0.5       # 0=ductile, 1=brittle / 脆性
```

### World Simulation / 世界模拟

```python
from pymo.kernel.world import World
import numpy as np

# Create world / 创建世界
world = World(
    gravity=np.array([0.0, -9.81]),  # gravity vector / 重力向量
    dt=1/60.0,                       # timestep (seconds) / 时间步长（秒）
    solver_iterations=10,            # contact solver iterations / 接触求解迭代次数
    substeps=1                       # substeps per frame / 每帧子步数
)

# Add bodies / 添加刚体
world.add(ball, platform)

# Step simulation / 推进模拟
world.step(300)  # 300 steps = 5 seconds at dt=1/60

# Query state / 查询状态
print(f"Ball position: {ball.pos}")
print(f"Ball velocity: {ball.vel}")
print(f"Kinetic energy: {world.total_kinetic_energy():.4f} J")
print(f"Total momentum: {world.total_momentum()}")
```

### Forces and Torques / 力和力矩

```python
# Apply force at center of mass (no torque) / 在质心施加力（无力矩）
ball.apply_force(np.array([10.0, 0.0]))

# Apply force at a point (produces torque) / 在某点施加力（产生力矩）
ball.apply_force_at(
    f=np.array([0.0, 50.0]),      # force vector / 力向量
    point=np.array([0.5, 0.0])    # world point / 世界坐标点
)

# Apply pure torque / 施加纯力矩
ball.apply_torque(np.array([0.0, 0.0, 5.0]))  # 3D only / 仅3D

# Forces are cleared each step / 力在每步结束时自动清除
```

---

## 4. 3D Physics / 3D物理

### 3D Shapes / 3D形状

```python
from pymo.kernel.bodies3d import (
    sphere_body, box_body, cylinder_body, convex_hull_body,
    Material, Body, ShapeType
)
from pymo.kernel.math3d import quat_from_axis_angle
import numpy as np

# Sphere / 球体
ball = sphere_body(
    pos=[0, 0, 5],
    radius=0.5,
    mass=1.0,
    material=Material(restitution=0.9)
)

# Box / 盒体
platform = box_body(
    pos=[0, 0, -0.5],
    half_extents=np.array([10.0, 10.0, 0.5]),
    static=True
)

# Cylinder / 圆柱体
cylinder = cylinder_body(
    pos=[3, 0, 2],
    radius=0.3,
    half_height=1.0,
    mass=2.0,
    angle=0.5  # initial rotation angle / 初始旋转角度
)

# Convex hull / 凸包
import scipy.spatial
pts = scipy.spatial.ConvexHull(np.random.randn(20, 3)).vertices
hull = convex_hull_body(
    pos=[-3, 0, 3],
    vertices=np.random.randn(20, 3),
    faces=scipy.spatial.ConvexHull(np.random.randn(20, 3)).simplices,
    mass=1.5
)
```

### 3D World / 3D世界

```python
from pymo.kernel.world3d import World3D
import numpy as np

world = World3D(
    gravity=np.array([0.0, 0.0, -9.81]),
    dt=1/60.0,
    solver_iterations=10,
    substeps=1
)

world.add(platform, ball, cylinder)
world.step(600)  # 10 seconds / 10秒

# Query / 查询
print(f"Ball height: {ball.pos[2]:.2f} m")
print(f"Angular momentum: {world.total_angular_momentum()}")
```

### Quaternion Operations / 四元数运算

```python
from pymo.kernel.math3d import (
    quat_identity, quat_mul, quat_conj, quat_normalize,
    quat_from_axis_angle, quat_to_axis_angle,
    quat_rotate, mat3_from_quat, quat_slerp
)

# Create rotation / 创建旋转
q = quat_from_axis_angle(axis=np.array([0, 0, 1]), angle=np.pi/4)  # 45° around Z

# Rotate a vector / 旋转一个向量
v = np.array([1.0, 0.0, 0.0])
v_rotated = quat_rotate(q, v)

# Convert to rotation matrix / 转换为旋转矩阵
R = mat3_from_quat(q)

# Interpolate rotations / 插值旋转
q2 = quat_from_axis_angle(np.array([0, 0, 1]), np.pi/2)
q_mid = quat_slerp(q, q2, 0.5)  # halfway / 中间值
```

---

## 5. Collision Detection / 碰撞检测

### 2D Collision (SAT) / 2D碰撞（SAT）

The 2D kernel uses the **Separating Axis Theorem (SAT)** for convex polygon collision, with face-clipping for multi-point contact generation.

2D内核使用**分离轴定理（SAT）**进行凸多边形碰撞检测，使用面裁剪生成多接触点。

```python
from pymo.kernel.collision import detect_collision, Contact

# Automatic detection between any shape pair / 自动检测任意形状对
contacts = detect_collision(body_a, body_b)

for c in contacts:
    print(f"Contact at {c.point}")
    print(f"Normal: {c.normal}, Penetration: {c.penetration:.4f}")
```

**Supported pairs / 支持的碰撞对:**
- Circle-Circle / 圆-圆
- Circle-Polygon / 圆-多边形
- Polygon-Polygon / 多边形-多边形

### 3D Collision (GJK + EPA) / 3D碰撞（GJK + EPA）

The 3D kernel uses **GJK (Gilbert-Johnson-Keerthi)** for intersection testing and **EPA (Expanding Polytope Algorithm)** for penetration depth.

3D内核使用**GJK算法**进行相交测试，使用**EPA算法**计算穿透深度。

```python
from pymo.kernel.collision3d import detect_collision, gjk_intersect, epa

# Full detection (GJK + EPA) / 完整检测
contacts = detect_collision(body_a, body_b)

# Manual GJK test / 手动GJK测试
intersecting, simplex = gjk_intersect(body_a, body_b)
if intersecting:
    result = epa(body_a, body_b, simplex)
    print(f"Penetration: {result.depth:.4f}")
    print(f"Normal: {result.normal}")
```

**Algorithm details / 算法细节:**

| Step / 步骤 | Algorithm / 算法 | Complexity / 复杂度 |
|-------------|-----------------|-------------------|
| Broad phase / 粗筛 | AABB overlap / AABB重叠 | O(1) |
| Narrow phase / 精确检测 | GJK simplex iteration / GJK单纯形迭代 | O(k), k≤32 |
| Penetration / 穿透深度 | EPA polytope expansion / EPA多面体扩展 | O(k), k≤32 |
| Fallback / 回退 | Direct sphere-sphere, box-box, sphere-box | O(1) |

### 2D Contact Solver / 2D接触求解器

The impulse-based solver conserves momentum by applying equal-and-opposite impulses at contact points.

基于冲量的求解器通过在接触点施加等大反向冲量来守恒动量。

```python
from pymo.kernel.solver import solve_contacts

# Solve contacts (mutates body velocities in place) / 求解接触（就地修改速度）
solve_contacts(contacts, iterations=10, baumgarte=0.2)
```

**Solver algorithm / 求解器算法:**

```
for each iteration:
    for each contact:
        1. Compute relative velocity at contact point / 计算接触点相对速度
        2. Compute normal impulse: j = -(1+e)*vn / inv_mass_sum / 计算法向冲量
        3. Apply equal-and-opposite impulse to both bodies / 对两个刚体施加等大反向冲量
        4. Compute friction impulse (clamped Coulomb) / 计算摩擦冲量（库仑截断）

Baumgarte positional correction:
    Push overlapping bodies apart / 推开重叠刚体
    correction = max(penetration - slop, 0) / total_inv * beta
```

---

## 6. Multi-Physics Rules / 多物理规则

### 6.1 Heat Conduction / 热传导

Fourier's law between contacting bodies: `dQ/dt = k_eff * A * ΔT / d`

接触刚体间的傅里叶定律：`dQ/dt = k_eff * A * ΔT / d`

```python
from pymo.rules.thermal import BodyThermalSystem, TemperatureField, diffuse_field
from pymo.kernel.bodies import circle_body, Material
import numpy as np

# Create bodies with different temperatures / 创建不同温度的刚体
hot_body = circle_body([0, 0], 0.5, mass=1.0,
    material=Material(thermal_conductivity=100.0, specific_heat=1000.0))
hot_body.temperature = 373.15  # 100°C

cold_body = circle_body([0.9, 0], 0.5, mass=1.0,
    material=Material(thermal_conductivity=100.0, specific_heat=1000.0))
cold_body.temperature = 273.15  # 0°C

# Create thermal system / 创建热系统
thermal = BodyThermalSystem(contact_area=1.0, contact_distance=0.1)

# Conduct heat / 传导热量
q = thermal.conduct_between(hot_body, cold_body, dt=0.01)
print(f"Heat transferred: {q:.2f} J")
print(f"Hot: {hot_body.temperature:.1f} K, Cold: {cold_body.temperature:.1f} K")

# Temperature field diffusion / 温度场扩散
field = TemperatureField(nx=50, ny=50, dx=0.1, dy=0.1)
field.set(0.25, 0.25, 373.15)  # hot spot / 热点
diffuse_field(field, alpha=0.01, dt=0.01)
```

**Conservation guarantee / 守恒保证:** Heat leaving body A is exactly added to body B. Total field energy conserved by Neumann boundary conditions.

从刚体A离开的热量精确加到刚体B。诺伊曼边界条件保证总场能量守恒。

### 6.2 SPH Fluid / SPH流体

Smoothed Particle Hydrodynamics with Poly6/Spiky/Viscosity kernels and Tait equation of state.

使用Poly6/Spiky/粘性核函数和Tait状态方程的光滑粒子流体动力学。

```python
from pymo.rules.fluid import SPHSystem, SPHParams, create_water_column

# Create fluid system / 创建流体系统
params = SPHParams(
    h=0.1,              # smoothing length / 平滑长度
    rest_density=1000.0, # kg/m^3 / 静止密度
    stiffness=5000.0,    # Pa / 刚度
    viscosity=1.0,       # Pa*s / 粘度
    particle_mass=8.66   # kg / 粒子质量
)

system = SPHSystem(params)

# Add particles / 添加粒子
for i in range(10):
    for j in range(20):
        system.add_particle([i * 0.1, j * 0.1])

# Simulate / 模拟
for step in range(100):
    system.step(dt=0.005)

# Query / 查询
print(f"Total energy: {system.total_energy():.2f} J")
print(f"Density ratio: {system.min_density_ratio():.2f} - {system.max_density_ratio():.2f}")
```

**SPH Kernels / SPH核函数:**

| Kernel / 核函数 | Formula / 公式 | Use / 用途 |
|----------------|---------------|-----------|
| Poly6 | `W(r) = 4/(πh⁸)(h²-r²)³` | Density estimation / 密度估计 |
| Spiky | `∇W = -30/(πh⁵)(h-r)²(r̂)` | Pressure gradient / 压力梯度 |
| Viscosity | `∇²W = 20/(πh⁵)(h-r)` | Viscous diffusion / 粘性扩散 |

### 6.3 Fracture / 断裂

Rankine criterion: fracture when maximum principal stress exceeds fracture toughness.

Rankine准则：当最大主应力超过断裂韧性时发生断裂。

```python
from pymo.rules.fracture import (
    check_fracture, split_body, process_fracture,
    principal_stresses, FractureParams
)

# Check if a contact causes fracture / 检查接触是否导致断裂
if check_fracture(contact):
    angle = principal_stress_angle(*estimate_contact_stress(contact))
    fragments = split_body(body, angle, FractureParams())
    print(f"Body split into {len(fragments)} fragments")

# Process all contacts for fracture / 处理所有接触的断裂
new_bodies = process_fracture(contacts, bodies, FractureParams(
    min_fragment_mass=0.01,  # remove tiny fragments / 移除微小碎片
    max_fragments=4          # max per event / 每事件最多碎片数
))
```

---

## 7. AI Layer / AI层

### Observer / 观察器

Records time-series data from world simulation (simulates real sensors).

从世界模拟中录制时间序列数据（模拟真实传感器）。

```python
from pymo.ai.observer import WorldObserver, TimeSeriesDataset

# Create world / 创建世界
from pymo.kernel.bodies import circle_body
from pymo.kernel.world import World
import numpy as np

w = World(gravity=np.array([0.0, -9.81]), dt=0.01)
w.add(circle_body([0.0, 10.0], 0.5, mass=1.0))

# Observe / 观察
observer = WorldObserver(w, sample_every=1)
dataset = observer.observe(n_steps=300)

# Access data / 访问数据
t = dataset.time()
y = dataset.get("body0.pos.y")
ke = dataset.get("total.ke")
```

### Law Discovery / 定律发现

Symbolic regression discovers analytic expressions from data.

符号回归从数据中发现解析表达式。

```python
from pymo.ai.law_discovery import LawDiscovery, GplearnRefineBackend

# Discover y = f(t) / 发现 y = f(t)
discovery = LawDiscovery(backend=GplearnRefineBackend())
law = discovery.discover_from_observation(time=t, values=y)

print(f"Expression: {law.expression}")
print(f"R² score: {law.r2:.4f}")

# Predict / 预测
y_pred = law.predict(t)
```

### Closed-Loop AI / 闭环AI

The full observe → discover → predict → compare loop.

完整的观察→发现→预测→比较循环。

```python
from pymo.ai.closed_loop import ClosedLoopAI

ai = ClosedLoopAI(train_fraction=0.6)
results = ai.run(dataset, quantities=["body0.pos.y"])

for r in results:
    print(f"{r.quantity}: error={r.relative_error:.4f} {'PASS' if r.passed() else 'FAIL'}")
    print(f"  Law: {r.law.expression}")

# Summary / 摘要
print(ai.summary())
```

---

## 8. Visualization / 可视化

### 2D Viewer / 2D查看器

```python
from pymo.viz.viewer import PhysicsViewer, ViewerConfig

# Interactive mode / 交互模式
viewer = PhysicsViewer(world, config=ViewerConfig(
    window_size=(900, 700),
    bg_color="black",
    body_color="lightblue"
))
viewer.run()  # opens window with keyboard controls / 打开窗口，支持键盘控制

# Headless recording / 无头录制
paths = viewer.record(n_frames=300, out_dir="output/", steps_per_frame=1)
```

**Keyboard controls / 键盘控制:**

| Key / 按键 | Action / 操作 |
|-----------|--------------|
| Space / 空格 | Pause / 暂停 |
| + / = | Speed up / 加速 |
| - | Slow down / 减速 |
| R | Reset / 重置 |
| Esc | Quit / 退出 |

### 3D Viewer / 3D查看器

```python
from pymo.viz.viewer3d import PhysicsViewer3D, ViewerConfig3D

viewer = PhysicsViewer3D(world3d, config=ViewerConfig3D(
    window_size=(1200, 800),
    bg_color="white"
))
viewer.run()

# Record to PNGs / 录制为PNG
paths = viewer.record(n_frames=600, out_dir="output_3d/")
```

---

## 9. Parallel Simulation / 并行模拟

Ray-based distributed simulation with fault tolerance and checkpointing.

基于Ray的分布式模拟，支持容错和检查点。

```python
from pymo.parallel.ray_parallel import (
    SimulationConfig, SimulationBatch, ExperimentRunner
)

# Define simulation configs / 定义模拟配置
config = SimulationConfig(
    gravity=(0, 0, -9.81),
    dt=1/60.0,
    max_steps=1000,
    bodies=[
        {"shape": "sphere", "pos": [0, 0, 5], "radius": 0.5, "mass": 1.0},
        {"shape": "box", "pos": [0, 0, -0.5], "half_extents": [10, 10, 0.5], "static": True}
    ]
)

# Run parameter sweep / 运行参数扫描
runner = ExperimentRunner(num_workers=4, output_dir="experiments/")
results = runner.run_sweep(
    param_grid={"gravity": [(0,0,-9.81), (0,0,-1.62)]},  # Earth vs Moon
    base_config=config,
    max_steps=500
)

for r in results:
    print(f"{r.config_id}: success={r.success}, steps={r.steps_completed}")
```

---

## 10. API Reference / API参考

### kernel.bodies (2D)

| Class/Function / 类/函数 | Description / 说明 |
|-------------------------|-------------------|
| `Body` | 2D rigid body with position, velocity, angle, shape / 2D刚体 |
| `Material` | Material properties (restitution, friction, thermal, etc.) / 材料属性 |
| `circle_body(pos, radius, mass, ...)` | Create a circle body / 创建圆形刚体 |
| `box_body(pos, half_w, half_h, mass, ...)` | Create a box body / 创建盒形刚体 |
| `polygon_inertia(vertices, mass)` | Compute polygon moment of inertia / 计算多边形转动惯量 |

### kernel.bodies3d (3D)

| Class/Function / 类/函数 | Description / 说明 |
|-------------------------|-------------------|
| `Body` | 3D rigid body with quaternion orientation / 3D刚体（四元数朝向） |
| `SphereShape`, `BoxShape`, `CylinderShape`, `ConvexHullShape` | Collision shapes / 碰撞形状 |
| `sphere_body(pos, radius, mass, ...)` | Create a sphere / 创建球体 |
| `box_body(pos, half_extents, mass, ...)` | Create a box / 创建盒体 |
| `cylinder_body(pos, radius, half_height, ...)` | Create a cylinder / 创建圆柱体 |
| `convex_hull_body(pos, vertices, faces, ...)` | Create a convex hull / 创建凸包 |

### kernel.collision / kernel.collision3d

| Function / 函数 | Description / 说明 |
|----------------|-------------------|
| `detect_collision(a, b)` | Detect contacts between two bodies / 检测两刚体间的接触 |
| `gjk_intersect(a, b)` | GJK intersection test / GJK相交测试 |
| `epa(a, b, simplex)` | EPA penetration depth / EPA穿透深度 |

### kernel.world / kernel.world3d

| Method / 方法 | Description / 说明 |
|--------------|-------------------|
| `World.add(*bodies)` | Add bodies to world / 向世界添加刚体 |
| `World.step(n)` | Advance n timesteps / 推进n个时间步 |
| `World.total_kinetic_energy()` | Total KE / 总动能 |
| `World.total_momentum()` | Total momentum vector / 总动量向量 |
| `World.total_angular_momentum()` | Total angular momentum (3D only) / 总角动量（仅3D） |
| `World.record()` | Snapshot state to history / 快照状态到历史记录 |

### rules.thermal

| Class/Function / 类/函数 | Description / 说明 |
|-------------------------|-------------------|
| `BodyThermalSystem` | Heat conduction between bodies / 刚体间热传导 |
| `TemperatureField` | 2D temperature grid / 2D温度场 |
| `diffuse_field(field, alpha, dt)` | Explicit diffusion step / 显式扩散步 |

### rules.fluid

| Class/Function / 类/函数 | Description / 说明 |
|-------------------------|-------------------|
| `SPHSystem` | SPH fluid particle system / SPH流体粒子系统 |
| `SPHParams` | Simulation parameters / 模拟参数 |
| `SPHParticle` | Single fluid particle / 单个流体粒子 |
| `create_water_column(x, y, w, h, spacing)` | Create rectangular fluid block / 创建矩形流体块 |

### rules.fracture

| Function / 函数 | Description / 说明 |
|----------------|-------------------|
| `check_fracture(contact)` | Check if contact causes fracture / 检查接触是否导致断裂 |
| `split_body(body, angle, params)` | Split body along plane / 沿平面分裂刚体 |
| `process_fracture(contacts, bodies, params)` | Process all fracture events / 处理所有断裂事件 |
| `principal_stresses(sxx, syy, sxy)` | Compute principal stresses / 计算主应力 |

### ai modules

| Class/Function / 类/函数 | Description / 说明 |
|-------------------------|-------------------|
| `WorldObserver` | Collect time-series data from world / 从世界收集时间序列数据 |
| `LawDiscovery` | Discover symbolic laws from data / 从数据中发现符号定律 |
| `ClosedLoopAI` | Observe → discover → predict → compare / 观察→发现→预测→比较 |
| `GplearnRefineBackend` | gplearn + coefficient refinement backend / gplearn+系数优化后端 |

---

## License / 许可证

MIT
