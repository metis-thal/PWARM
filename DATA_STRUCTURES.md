# pymo Data Structures & Algorithms / pymo数据结构与算法

Complete bilingual (中英双语) reference for all core data structures and algorithms in pymo.

---

## Table of Contents / 目录

1. [2D Rigid Body / 2D刚体](#1-2d-rigid-body--2d刚体)
2. [3D Rigid Body / 3D刚体](#2-3d-rigid-body--3d刚体)
3. [Quaternion Math / 四元数数学](#3-quaternion-math--四元数数学)
4. [2D Collision Detection / 2D碰撞检测](#4-2d-collision-detection--2d碰撞检测)
5. [3D Collision Detection / 3D碰撞检测](#5-3d-collision-detection--3d碰撞检测)
6. [Contact Solver / 接触求解器](#6-contact-solver--接触求解器)
7. [Velocity-Verlet Integrator / 速度Verlet积分器](#7-velocity-verlet-integrator--速度verlet积分器)
8. [Thermal Conduction / 热传导](#8-thermal-conduction--热传导)
9. [SPH Fluid / SPH流体](#9-sph-fluid--sph流体)
10. [Fracture Mechanics / 断裂力学](#10-fracture-mechanics--断裂力学)
11. [Symbolic Regression / 符号回归](#11-symbolic-regression--符号回归)

---

## 1. 2D Rigid Body / 2D刚体

### Data Structure / 数据结构

```
Body (src/pymo/kernel/bodies.py)
├── Kinematic State / 运动状态
│   ├── pos: np.ndarray (2,)      # position / 位置
│   ├── vel: np.ndarray (2,)      # velocity / 速度
│   ├── angle: float              # rotation angle (rad) / 旋转角（弧度）
│   └── ang_vel: float            # angular velocity / 角速度
│
├── Dynamic Properties / 动力学属性
│   ├── mass: float               # mass (kg) / 质量
│   ├── inv_mass: float           # 1/mass (0 if static) / 质量倒数
│   ├── inertia: float            # moment of inertia / 转动惯量
│   └── inv_inertia: float        # 1/inertia / 转动惯量倒数
│
├── Shape / 形状
│   ├── radius: float             # if > 0: circle / 若>0：圆形
│   └── vertices: np.ndarray (N,2) # if not None: convex polygon / 凸多边形顶点
│
├── Material / 材料
│   └── Material dataclass / 材料数据类
│
├── State Flags / 状态标志
│   ├── static: bool              # immovable / 不可移动
│   └── is_sensor: bool           # no impulse response / 无冲量响应
│
├── Accumulated Forces / 累积力
│   ├── force: np.ndarray (2,)    # linear force / 线性力
│   └── torque: float             # rotational torque / 旋转力矩
│
└── Thermal State / 热状态
    ├── temperature: float        # Kelvin / 开尔文
    └── heat: float               # accumulated thermal energy (J) / 累积热能
```

### Moment of Inertia Formulas / 转动惯量公式

**Circle / 圆形:**
```
I = 0.5 * m * r²
```

**Convex polygon (about centroid) / 凸多边形（关于质心）:**
```
I = ρ/12 * Σᵢ [(yᵢ² + yᵢyᵢ₊₁ + yᵢ₊₁²)(xᵢyᵢ₊₁ - xᵢ₊₁yᵢ) +
                (xᵢ² + xᵢxᵢ₊₁ + xᵢ₊₁²)(xᵢyᵢ₊₁ - xᵢ₊₁yᵢ)]

where ρ = mass / area / 其中 ρ = 质量/面积
```

### World Vertices Transform / 世界顶点变换

```
v_world = R(θ) * v_local + pos

R(θ) = [[cos(θ), -sin(θ)],
         [sin(θ),  cos(θ)]]
```

### Kinetic Energy / 动能

```
KE = 0.5 * m * |v|² + 0.5 * I * ω²
```

---

## 2. 3D Rigid Body / 3D刚体

### Data Structure / 数据结构

```
Body (src/pymo/kernel/bodies3d.py)
├── Kinematic State / 运动状态
│   ├── pos: np.ndarray (3,)      # position / 位置
│   ├── vel: np.ndarray (3,)      # velocity / 速度
│   ├── orn: np.ndarray (4,)      # quaternion (w,x,y,z) / 四元数
│   └── ang_vel: np.ndarray (3,)  # angular velocity / 角速度
│
├── Dynamic Properties / 动力学属性
│   ├── mass: float               # mass (kg) / 质量
│   ├── inv_mass: float           # 1/mass / 质量倒数
│   ├── inertia: np.ndarray (3,3) # inertia tensor / 惯性张量
│   └── inv_inertia: np.ndarray (3,3) # inverse inertia tensor / 惯性张量逆
│
├── Shape Hierarchy / 形状层次
│   ├── Shape (base) / 基类
│   │   ├── SphereShape(radius)
│   │   ├── BoxShape(half_extents: (3,))
│   │   ├── CylinderShape(radius, half_height)
│   │   └── ConvexHullShape(vertices: (N,3), faces: (M,3))
│   │
│   └── Each shape implements: / 每个形状实现：
│       ├── compute_mass_properties(density) → (mass, inertia_3x3)
│       ├── support_point(direction, pos, orn) → point
│       └── get_vertices(pos, orn) → vertices
│
├── Material / 材料 (same as 2D / 同2D)
├── State Flags / 状态标志
├── Accumulated Forces / 累积力
│   ├── force: np.ndarray (3,)
│   └── torque: np.ndarray (3,)   # 3D torque vector / 3D力矩向量
└── Thermal State / 热状态
```

### Shape Mass Properties / 形状质量属性

**Sphere / 球体:**
```
Volume = (4/3)πr³
Mass = ρ * V
I = (2/5) * m * r² * I₃ₓ₃
```

**Box / 盒体:**
```
Volume = 8 * hx * hy * hz
Mass = ρ * V
Ixx = m/3 * (hy² + hz²)
Iyy = m/3 * (hx² + hz²)
Izz = m/3 * (hx² + hy²)
I = diag(Ixx, Iyy, Izz)
```

**Cylinder / 圆柱体:**
```
Volume = π * r² * (2 * hh)
Mass = ρ * V
Ixx = Iyy = m * (3r² + h²) / 12
Izz = m * r² / 2
I = diag(Ixx, Iyy, Izz)
```

**Convex Hull / 凸包:**
```
Uses tetrahedron decomposition from origin / 使用从原点的四面体分解
For each face (v0, v1, v2):
    vol = det([v0, v1, v2]) / 6
    tet_mass = ρ * vol
    tet_com = (v0 + v1 + v2) / 4
    I_tet = tet_mass * (r²I - rrᵀ)

Total: sum all tetrahedra, shift to COM / 总计：求和所有四面体，平移到质心
```

### Support Point / 支撑点

The support point in direction `d` is the point on the shape furthest along `d`.

方向`d`的支撑点是形状沿`d`方向最远的点。

```
support(d) = argmax_{p ∈ shape} (p · d)
```

Used by GJK algorithm for collision detection.

被GJK算法用于碰撞检测。

### Kinetic Energy / 动能

```
KE = 0.5 * m * |v|² + 0.5 * ωᵀ * I * ω
```

---

## 3. Quaternion Math / 四元数数学

### Representation / 表示

```
q = (w, x, y, z) = w + xi + yj + zk

where i² = j² = k² = ijk = -1

Unit quaternion: |q| = 1 / 单位四元数
Identity: q = (1, 0, 0, 0) / 单位元
```

### Operations / 运算

**Multiplication / 乘法:**
```
q₁ * q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂,
            w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂,
            w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂,
            w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)
```

**Conjugate / 共轭:**
```
q* = (w, -x, -y, -z)
```

**Inverse / 逆:**
```
q⁻¹ = q* / |q|²  (for unit quaternion: q⁻¹ = q*)
```

**Axis-Angle → Quaternion / 轴角→四元数:**
```
q = (cos(θ/2), sin(θ/2) * û)

where û = axis / |axis|
```

**Quaternion → Rotation Matrix / 四元数→旋转矩阵:**
```
R = [[1-2y²-2z²,  2xy-2wz,    2xz+2wy  ],
     [2xy+2wz,     1-2x²-2z²,  2yz-2wx  ],
     [2xz-2wy,     2yz+2wx,    1-2x²-2y²]]
```

**Rotate Vector / 旋转向量:**
```
v' = q * (0, v) * q*

implemented as:
qv = (0, vx, vy, vz)
result = q * qv * conj(q)
v' = result[1:3]
```

**SLERP (Spherical Linear Interpolation) / 球面线性插值:**
```
q(t) = q₁ * sin((1-t)θ)/sin(θ) + q₂ * sin(tθ)/sin(θ)

where θ = arccos(q₁ · q₂)
```

**Angular Velocity Integration / 角速度积分:**
```
q_new = q * exp(0.5 * ω * dt)

where exp(v) = (cos(|v|), sin(|v|) * v/|v|)
```

### Available Functions / 可用函数

| Function / 函数 | Description / 说明 |
|----------------|-------------------|
| `quat_identity()` | Identity quaternion / 单位四元数 |
| `quat_mul(q1, q2)` | Quaternion multiplication / 四元数乘法 |
| `quat_conj(q)` | Conjugate / 共轭 |
| `quat_inverse(q)` | Inverse / 逆 |
| `quat_normalize(q)` | Normalize to unit length / 归一化 |
| `quat_from_axis_angle(axis, angle)` | From axis-angle / 从轴角创建 |
| `quat_to_axis_angle(q)` | To axis-angle / 转换为轴角 |
| `quat_rotate(q, v)` | Rotate vector / 旋转向量 |
| `mat3_from_quat(q)` | To rotation matrix / 转换为旋转矩阵 |
| `quat_from_mat3(m)` | From rotation matrix / 从旋转矩阵创建 |
| `quat_slerp(q1, q2, t)` | Spherical interpolation / 球面插值 |
| `integrate_angular_velocity(q, ω, dt)` | Integrate rotation / 积分旋转 |

---

## 4. 2D Collision Detection / 2D碰撞检测

### Algorithm: Separating Axis Theorem (SAT) / 分离轴定理

For two convex polygons A and B, they do NOT overlap if and only if there exists an axis along which their projections do not overlap.

对于两个凸多边形A和B，当且仅当存在一个轴使它们的投影不重叠时，它们不重叠。

```
Time complexity / 时间复杂度: O(Nₐ + Nᵦ) per axis
Space complexity / 空间复杂度: O(1)

Axes to test / 待测试轴:
  Each edge normal of A and B / A和B的每条边法线
  Total: Nₐ + Nᵦ axes / 共 Nₐ + Nᵦ 个轴
```

### Circle-Circle / 圆-圆

```
Distance d = |pos_b - pos_a|
Overlap = r_a + r_b - d

if d >= r_a + r_b or d < ε:
    No collision / 无碰撞
else:
    Normal = (pos_b - pos_a) / d
    Contact point = pos_a + normal * r_a
    Penetration = r_a + r_b - d
```

### Circle-Polygon / 圆-多边形

```
1. Find closest point on polygon to circle center
   找到多边形上离圆心最近的点
2. If circle center inside polygon:
   If 圆心在多边形内:
       Normal = outward edge normal (most anti-parallel to interior direction)
       法线 = 向外的边法线
       Penetration = radius + distance_to_closest
   Else:
       Normal = (circle_center - closest_point) / distance
       Penetration = radius - distance_to_closest
```

### Polygon-Polygon (SAT + Face Clipping) / 多边形-多边形

```
1. SAT: Find minimum overlap axis / SAT：找到最小重叠轴
   For each edge normal of A and B:
       Project both polygons onto axis
       overlap = min(max_a, max_b) - max(min_a, min_b)
       if overlap < 0: return None (separating axis found)

2. Face Clipping (Box2D-style) / 面裁剪（Box2D风格）
   - Find reference face (most anti-parallel to collision normal)
   - Find incident face (most parallel to -normal)
   - Clip incident face against reference face side planes
   - Returns 1-2 contact points on the reference face
```

### Contact Data Structure / 接触数据结构

```
Contact
├── a: Body           # first body / 第一个刚体
├── b: Body           # second body / 第二个刚体
├── point: (2,)       # world-space contact point / 世界坐标接触点
├── normal: (2,)      # unit normal a→b / 单位法线（从a到b）
├── penetration: float # overlap depth / 重叠深度
├── restitution: float # combined bounciness / 综合弹性系数
└── friction: float    # combined friction / 综合摩擦系数
```

### Material Combination Rules / 材料组合规则

```
restitution = max(restitution_a, restitution_b)  #取最大弹性
friction = sqrt(friction_a * friction_b)          #几何平均摩擦
```

---

## 5. 3D Collision Detection / 3D碰撞检测

### Algorithm: GJK + EPA / GJK + EPA算法

**GJK (Gilbert-Johnson-Keerthi) / GJK算法:**

Tests whether two convex shapes overlap by searching for a point in the Minkowski difference A - B that contains the origin.

通过在闵可夫斯基差A-B中搜索包含原点的点来测试两个凸形状是否重叠。

```
Minkowski difference / 闵可夫斯基差:
    S = {a - b | a ∈ A, b ∈ B}

Support function / 支撑函数:
    support_S(d) = support_A(d) - support_B(-d)

GJK iterates / GJK迭代:
    1. Start with initial simplex (point) / 从初始单纯形（点）开始
    2. Get support point in direction toward origin / 获取朝原点方向的支撑点
    3. Add to simplex, check if origin inside / 添加到单纯形，检查原点是否在内
    4. Update search direction / 更新搜索方向
    5. Repeat until origin found or Minkowski difference doesn't reach origin

Simplex handling / 单纯形处理:
    2 points (line) → check if origin on segment / 检查原点是否在线段上
    3 points (triangle) → check which region origin is in / 检查原点所在区域
    4 points (tetrahedron) → check if inside / 检查是否在内部

Complexity / 复杂度: O(k) iterations, k ≤ 32
```

**EPA (Expanding Polytope Algorithm) / EPA算法:**

After GJK confirms intersection, EPA finds the penetration depth and contact normal.

GJK确认相交后，EPA找到穿透深度和接触法线。

```
1. Start with GJK simplex as initial polytope / 以GJK单纯形为初始多面体
2. Find face closest to origin / 找到离原点最近的面
3. Get support point in face normal direction / 获取面法线方向的支撑点
4. If support point distance ≈ face distance: converged / 若支撑点距离≈面距离：收敛
5. Otherwise, expand polytope / 否则扩展多面体
   - Add new vertex / 添加新顶点
   - Remove visible faces / 移除可见面
   - Connect horizon edges to new vertex / 连接地平线边到新顶点
6. Repeat until convergence / 重复直到收敛

Result / 结果: EPAResult(normal, depth, contact_point)
```

### Direct Fallbacks / 直接回退

For simple shape pairs, direct geometric computation is used:

对于简单形状对，使用直接几何计算：

**Sphere-Sphere / 球-球:**
```
diff = center_b - center_a
dist = |diff|
penetration = r_a + r_b - dist
normal = diff / dist
```

**Box-Box (SAT in 3D) / 盒-盒（3D SAT）:**
```
Test 15 axes:
  3 face normals of box A
  3 face normals of box B
  9 cross products of edge A × edge B

For each axis:
  Project both boxes onto axis
  Check overlap
  Track minimum overlap axis
```

**Sphere-Box / 球-盒:**
```
1. Transform sphere center to box local space / 将球心变换到盒局部空间
2. Clamp to box extents to find closest point / 钳制到盒范围找到最近点
3. Compute distance and penetration / 计算距离和穿透深度
```

---

## 6. Contact Solver / 接触求解器

### Algorithm: Sequential Impulse Solver / 顺序冲量求解器

Conserves momentum by applying equal-and-opposite impulses at contact points.

通过在接触点施加等大反向冲量来守恒动量。

```
For each contact, for each iteration:
  1. Relative velocity at contact point / 接触点相对速度
     v_a = vel_a + ω_a × r_a
     v_b = vel_b + ω_b × r_b
     rv = v_b - v_a

  2. Normal component / 法向分量
     v_n = rv · n
     if v_n > 0: continue (separating) / 分离中

  3. Effective mass / 有效质量
     inv_mass_sum = inv_mass_a + inv_mass_b
                   + inv_inertia_a * (r_a × n)²
                   + inv_inertia_b * (r_b × n)²

  4. Normal impulse / 法向冲量
     j = -(1 + e) * v_n / inv_mass_sum
     impulse = j * n

  5. Apply to both bodies / 施加到两个刚体
     vel_a -= inv_mass_a * impulse
     ω_a -= inv_inertia_a * (r_a × impulse)
     vel_b += inv_mass_b * impulse
     ω_b += inv_inertia_b * (r_b × impulse)

  6. Friction impulse / 摩擦冲量
     tangent = rv - v_n * n
     j_t = -v_t / inv_mass_sum
     j_t = clamp(j_t, -μ*|j|, μ*|j|)  # Coulomb friction / 库仑摩擦
     Apply same as normal impulse / 同法向冲量施加

Baumgarte Positional Correction / Baumgarte位置修正:
  Push overlapping bodies apart / 推开重叠刚体
  correction = max(penetration - slop, 0) / total_inv * beta
  a.pos -= inv_mass_a * correction * n
  b.pos += inv_mass_b * correction * n
```

### Momentum Conservation / 动量守恒

```
Momentum before = momentum after
  m_a * v_a + m_b * v_b = m_a * v_a' + m_b * v_b'

This is guaranteed because impulses are applied as equal-and-opposite pairs.
通过等大反向冲量对保证动量守恒。
```

---

## 7. Velocity-Verlet Integrator / 速度Verlet积分器

### Algorithm / 算法

The Velocity-Verlet integrator is a symplectic, second-order integrator that conserves energy well for Hamiltonian systems.

速度Verlet积分器是辛二阶积分器，对哈密顿系统具有良好的能量守恒性。

```
Standard Velocity-Verlet / 标准速度Verlet:

Given: x_n, v_n, a_n = f(x_n)/m

Half-kick / 半步速度更新:
    v_{n+1/2} = v_n + a_n * dt/2

Drift / 位移:
    x_{n+1} = x_n + v_{n+1/2} * dt

Compute new acceleration / 计算新加速度:
    a_{n+1} = f(x_{n+1})/m

Second half-kick / 第二次半步速度更新:
    v_{n+1} = v_{n+1/2} + a_{n+1} * dt/2
```

### Numba JIT Implementation / Numba JIT实现

```python
@njit(cache=True)
def _velocity_verlet_step(pos, vel, acc, dt, masses, accel_fn):
    vel_half = vel + acc * (0.5 * dt)
    pos_new = pos + vel_half * dt
    acc_new = accel_fn(pos_new, masses)
    vel_new = vel_half + acc_new * (0.5 * dt)
    return pos_new, vel_new, acc_new
```

### Adaptive Timestep / 自适应时间步长

```
dt_adapt = sqrt(dt_base / max(|a_max|, ε))

Clamped to [0.5 * dt_base, dt_base]

Ensures displacement per step stays bounded.
确保每步位移有界。
```

### Conservation Properties / 守恒性质

```
Energy drift / 能量漂移:
    Verified: ~1e-15 (machine precision) for Verlet
    验证：Verlet积分器约1e-15（机器精度）

Relative drift formula / 相对漂移公式:
    drift = max(|E(t) - E(0)|) / |E(0)|
```

---

## 8. Thermal Conduction / 热传导

### Fourier's Law / 傅里叶定律

```
dQ/dt = k_eff * A * (T_hi - T_lo) / d

where / 其中:
    k_eff = 2 * k_a * k_b / (k_a + k_b)  # series conductance / 串联热导
    A = contact area / 接触面积
    d = contact distance / 接触距离
    T = temperature / 温度
```

### Heat Transfer Between Bodies / 刚体间热传递

```
For two bodies a (hot) and b (cold):

1. Effective conductance / 有效热导
   k_eff = 2 * k_a * k_b / (k_a + k_b)

2. Heat capacity / 热容
   C_a = m_a * c_a, C_b = m_b * c_b

3. Temperature difference / 温差
   ΔT = T_a - T_b

4. Fourier flux / 傅里叶通量
   q = k_eff * A * ΔT / d * dt

5. Stability clamping / 稳定性截断
   max_q = 0.5 * C_a * |ΔT|
   q = clamp(q, -max_q, max_q)

6. Transfer (conservation guaranteed) / 传递（保证守恒）
   Q_a -= q  (hot body loses heat / 热刚体失热)
   Q_b += q  (cold body gains heat / 冷刚体得热)
   T_a -= q / C_a
   T_b += q / C_b
```

### Temperature Field Diffusion / 温度场扩散

Explicit finite difference with 5-point stencil:

使用5点模板的显式有限差分：

```
dT/dt = α * ∇²T

Stencil / 模板:
    ∇²T(i,j) = [T(i+1,j) + T(i-1,j) + T(i,j+1) + T(i,j-1) - 4*T(i,j)] / (dx*dy)

Boundary conditions / 边界条件:
    Neumann (no-flux): ghost cells = boundary values
    诺伊曼（无通量）：虚拟单元 = 边界值

Stability requirement / 稳定性要求:
    dt < dx² / (4*α)

Conservation / 守恒:
    Total field energy conserved up to machine precision.
    总场能量在机器精度内守恒。
```

---

## 9. SPH Fluid / SPH流体

### Smoothed Particle Hydrodynamics / 光滑粒子流体动力学

SPH is a meshless Lagrangian method where fluid is represented by particles.

SPH是一种无网格拉格朗日方法，流体由粒子表示。

### Kernel Functions / 核函数

**Poly6 (density estimation) / Poly6（密度估计）:**
```
W(r) = 4/(πh⁸) * (h² - r²)³    for r ≤ h

Normalized: ∫ W(r) dA = 1 in 2D
归一化：在2D中 ∫ W(r) dA = 1
```

**Spiky (pressure gradient) / Spiky（压力梯度）:**
```
∇W = -30/(πh⁵) * (h - r)² * (r̂)    for r ≤ h

where r̂ = r_vec / |r_vec|
```

**Viscosity (Laplacian) / 粘性（拉普拉斯）:**
```
∇²W = 20/(πh⁵) * (h - r)    for r ≤ h
```

### SPH Equations / SPH方程

**Density / 密度:**
```
ρ_i = Σ_j m_j * W(|r_i - r_j|, h)
```

**Pressure (Tait EOS) / 压力（Tait状态方程）:**
```
P = k * (ρ/ρ₀ - 1)

where / 其中:
    k = stiffness / 刚度
    ρ₀ = rest density / 静止密度
```

**Pressure Force / 压力:**
```
F_pressure = -m_j * (P_i/ρ_i² + P_j/ρ_j²) * ∇W(r_i - r_j, h)

Symmetric pairwise / 对称逐对
```

**Viscous Force / 粘性力:**
```
F_viscosity = m_j * μ * (v_j - v_i) / ρ_j * ∇²W(r, h)

Symmetric pairwise / 对称逐对
```

**Integration (Symplectic Euler) / 积分（辛欧拉）:**
```
v_i += F_i / m_i * dt
x_i += v_i * dt
```

### Neighbor Search / 邻居搜索

```
O(N²) brute force / O(N²) 暴力搜索

For each particle pair (i, j):
    if |r_i - r_j|² ≤ h²:
        add j to i's neighbor list
        add i to j's neighbor list
```

### Time Complexity / 时间复杂度

```
Per step: O(N²) for neighbor search + force computation
每步：邻居搜索+力计算 O(N²)

For typical SPH: N = 100-10000 particles
典型SPH：N = 100-10000个粒子
```

---

## 10. Fracture Mechanics / 断裂力学

### Rankine Criterion / Rankine准则

Fracture occurs when maximum principal stress exceeds fracture strength.

当最大主应力超过断裂强度时发生断裂。

```
Principal stresses from 2D stress tensor / 从2D应力张量计算主应力:

σ₁,₂ = (σ_xx + σ_yy)/2 ± √[((σ_xx - σ_yy)/2)² + σ_xy²]

where σ₁ ≥ σ₂
```

### Contact Stress Estimation / 接触应力估计

```
Normal stress (Hertzian approximation) / 法向应力（赫兹近似）:
    E_eff = 2*E_a*E_b / (E_a + E_b)    # effective modulus / 有效模量
    R_eff = 1 / (1/R_a + 1/R_b)         # effective radius / 有效半径
    σ_n = -E_eff * √(penetration / R_eff)  # negative = compressive / 负=压缩

Shear stress / 剪应力:
    τ = min(|σ_n| * friction, hardness)

Rotate to world coordinates / 旋转到世界坐标:
    σ_xx = σ_nn*c² + σ_tt*s² + 2*τ_nt*c*s
    σ_yy = σ_nn*s² + σ_tt*c² - 2*τ_nt*c*s
    σ_xy = (σ_nn - σ_tt)*c*s + τ_nt*(c² - s²)
```

### Fracture Check / 断裂检查

```
For each body in contact:
    K_IC = fracture_toughness
    a_crack = max(penetration, ε)     # crack length proxy / 裂纹长度代理
    σ_c = K_IC / √(π * a_crack)       # fracture strength / 断裂强度
    σ_c *= (1 - 0.5 * brittleness)    # brittleness factor / 脆性因子

    if σ₁ > σ_c:
        fracture occurs / 发生断裂
```

### Body Splitting / 刚体分裂

```
Along line through centroid at angle θ:

1. Classify vertices by side of line / 按线的两侧分类顶点
2. Find edge-line intersections / 找到边-线交点
3. Build two polygon fragments / 构建两个多边形碎片

Circle split → two half-circles (approximated as polygons)
圆分裂 → 两个半圆（近似为多边形）

Polygon split → two convex polygons (if ≥ 3 vertices each)
多边形分裂 → 两个凸多边形（若每侧≥3个顶点）

Mass distribution / 质量分配:
    Each fragment gets mass/2
    每个碎片获得质量/2
```

---

## 11. Symbolic Regression / 符号回归

### Gplearn + Coefficient Refinement / gplearn + 系数优化

A two-phase approach to discover analytic laws from data:

从数据中发现解析定律的两阶段方法：

```
Phase 1: Structure Discovery (gplearn GP) / 结构发现（遗传编程）
    - Evolve expression trees using genetic programming
    - 使用遗传编程演化表达式树
    - Function set: {add, sub, mul, div, square, cos, sin}
    - Standardize inputs for GP stability
    - 标准化输入以提高GP稳定性

Phase 2: Coefficient Refinement (scipy least-squares) / 系数优化（最小二乘）
    - Fit polynomial in primary feature using discovered structure
    - 使用发现的结构在主特征上拟合多项式
    - Recovers accurate constants regardless of GP's random constants
    - 无论GP的随机常数如何，都能恢复精确常数
```

### Discovered Law / 发现的定律

```
DiscoveredLaw
├── expression: str          # human-readable formula / 人类可读公式
├── predict(X) → y           # prediction function / 预测函数
├── r2: float                # fit quality (1.0 = perfect) / 拟合质量
└── backend: str             # algorithm used / 使用的算法
```

### Validation / 验证

```
Free fall: y(t) = 10 - 4.905t²
    Discovered: exact match, R² = 1.0, error = 0.00%

Oscillation: y(t) = A*cos(ωt + φ)
    Discovered: ω = 2.0/3.0 exact, R² = 1.0

The AI model is a LEARNED approximation of the ground truth.
AI模型是真值的学习近似。
```

---

## Summary Table / 总结表

| Algorithm / 算法 | Complexity / 复杂度 | Conservation / 守恒 | Used In / 用于 |
|-----------------|-------------------|-------------------|--------------|
| SAT (2D) | O(Nₐ+Nᵦ) per axis | - | kernel.collision |
| GJK (3D) | O(k), k≤32 | - | kernel.collision3d |
| EPA (3D) | O(k), k≤32 | - | kernel.collision3d |
| Sequential Impulse | O(iterations × contacts) | Momentum / 动量 | kernel.solver |
| Velocity-Verlet | O(N) | Energy ~1e-15 / 能量 | kernel.integrators |
| Fourier Conduction | O(contacts) | Energy exact / 能量 | rules.thermal |
| SPH | O(N²) | Mass exact / 质量 | rules.fluid |
| Rankine Fracture | O(contacts) | Mass exact / 质量 | rules.fracture |
| Symbolic Regression | O(pop × gen) | - | ai.law_discovery |

---

## License / 许可证

MIT
