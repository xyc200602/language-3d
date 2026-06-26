# Husky A200 底盘结构 — 官方 URDF 权威数据

> 来源：[bulletphysics/bullet3 — data/husky/husky.urdf](https://github.com/bulletphysics/bullet3/blob/master/data/husky/husky.urdf)
> （由 Clearpath 官方 `husky_description` xacro 自动生成，是 Husky A200 的权威结构定义。）
> 摘录于 2026-06-24。**所有数值直接取自 URDF XML，非估算。**

这份文档是 `mobile_base_gen.py` 重写的**第一事实源**。之前 `real_ugv_chassis_engineering.md`
用的是手册规格页（外形尺寸），这里用的是 URDF 的**精确连杆/关节定义**——
生成器需要的是后者（关节 origin xyz、碰撞体 box size），因为生成器输出的是
assembly JSON（连杆+关节），不是外形渲染。

---

## 1. 连接树（URDF 原文）

```
base_footprint (虚拟根, Z=0 地面)
   └─ [fixed, xyz="0 0 0.14493"] base_link          ← 车体本体
         ├─ [fixed] top_plate_link                   ← 顶板（机械臂安装面）
         ├─ [fixed, xyz="0.272 0 0.245"] user_rail
         ├─ [fixed, xyz="0.48 0 0.091"] front_bumper
         ├─ [fixed, xyz="-0.48 0 0.091"] rear_bumper
         ├─ [continuous, xyz="0.256 0.2854 0.03282", axis="0 1 0"] front_left_wheel_link
         ├─ [continuous, xyz="0.256 -0.2854 0.03282", axis="0 1 0"] front_right_wheel_link
         ├─ [continuous, xyz="-0.256 0.2854 0.03282", axis="0 1 0"] rear_left_wheel_link
         └─ [continuous, xyz="-0.256 -0.2854 0.03282", axis="0 1 0"] rear_right_wheel_link
```

**关键事实（颠覆之前的错误假设）：**

1. **轮子直接挂在 base_link 上**（continuous joint），不经过电机 link、不经过悬架 link。
   电机在车体内部（URDF 不建为独立 link），通过 belt 传动。
2. **没有悬架**。Husky 是刚性底盘，靠大轮径 + 橡胶形变吸震。
3. **base_footprint 是虚拟根**（Z=0 地面投影），base_link 通过 fixed joint 抬高到
   z=0.14493m（145mm 离地间隙）。
4. **轮子关节轴 = Y 轴**（axis="0 1 0"），轮子碰撞体 rpy="1.570795 0 0"
   （绕 X 转 90°，使圆柱轴线沿 Y）。这确认了 axis=y 是正确的轮子旋转轴。

---

## 2. 精确尺寸（URDF XML 原值，单位 mm）

### base_link 碰撞体（车体本体）

```xml
<collision>
  <origin rpy="0 0 0" xyz="0 0 0.12498"/>
  <box size="1.0074 0.5709 0.2675"/>   <!-- L × W × H, 单位 m -->
</collision>
```

| 维度 | 值 (m) | 值 (mm) | URDF 语义 |
|---|---|---|---|
| box size X | 1.0074 | **1007.4** | 车体**长度**（行驶方向，前后） |
| box size Y | 0.5709 | **570.9** | 车体**宽度**（左右，轮轴方向） |
| box size Z | 0.2675 | **267.5** | 车体**高度** |
| collision origin z | 0.12498 | 125.0 | 碰撞体中心在 base_link 上方 125mm |

### base_link 位置

```xml
<joint name="chassis_joint" type="fixed">
  <origin rpy="0 0 0" xyz="0 0 0.14493"/>   <!-- base_link 抬高 145mm -->
  <parent link="base_footprint"/>
  <child link="base_link"/>
</joint>
```

base_link 中心 Z = 0.14493m（145mm，这是离地间隙）。

### 轮子（4 个，尺寸相同）

```xml
<collision>
  <origin rpy="1.570795 0 0" xyz="0 0 0"/>
  <cylinder length="0.1143" radius="0.17775"/>
</collision>
```

| 维度 | 值 (m) | 值 (mm) |
|---|---|---|
| cylinder length | 0.1143 | **114.3** （轮宽） |
| cylinder radius | 0.17775 | **177.75** （轮半径） |
| 轮径（直径） | 0.3555 | **355.5** |

### 轮子位置（关节 origin xyz，相对 base_link）

| 轮子 | X (前后) | Y (左右) | Z |
|---|---|---|---|
| front_left  | +0.256 | +0.2854 | 0.03282 |
| front_right | +0.256 | -0.2854 | 0.03282 |
| rear_left   | -0.256 | +0.2854 | 0.03282 |
| rear_right  | -0.256 | -0.2854 | 0.03282 |

派生：
- **轮距 track** = 0.2854 × 2 = **0.5708 m (570.8 mm)**（Y 方向，左右）
- **轴距 wheelbase** = 0.256 × 2 = **0.512 m (512.0 mm)**（X 方向，前后）
- 轮子 Z = 0.03282m：轮中心在 base_link 坐标系（base_link 中心在地面以上 145mm）
  下方 33mm，即轮中心绝对高度 = 145 - 33 = 112mm。轮半径 178mm，所以轮底 = 112 - 178 = -66mm？
  不对——base_footprint 的 Z=0 才是地面。轮底应在 Z≈0。

  **重新核算**：URDF 里 base_footprint 到地面有 wheel_radius - |wheel_z| 的关系。
  base_link z=0.14493，轮 z=0.03282（相对 base_link），轮中心绝对 z = 0.14493 + 0.03282 = 0.17775m = 轮半径。
  所以**轮底正好在 Z=0（地面）**。✓ 这确认了"轮中心高度 = 轮半径"的 Z-stack 约定。

---

## 3. 关键比例（用于校验生成器）

| 比例 | Husky 实测 | 含义 | 生成器该用 |
|---|---|---|---|
| **车体宽 / 轮距** | 570.9 / 570.8 = **1.00** | 车体宽 = 轮距，轮子**正好在车体侧边缘**（半嵌入） | body_width ≈ track（不是 track×1.2） |
| **车体长 / 轴距** | 1007.4 / 512.0 = **1.97** | 车体长度 ≈ 2× 轴距，轮子**内缩**于车体前后端 | body_length ≈ wheelbase × 2.0（不是 wheelbase×1.2） |
| **车体长 / 车体宽** | 1007.4 / 570.9 = **1.76** | 车体是**长方形**（长 > 宽），不是正方形 | length > width（不是相等） |
| 轮径 / 轮距 | 355.5 / 570.8 = **0.62** | 大轮子 | 0.4–0.6 |
| 轴距 / 轮距 | 512.0 / 570.8 = **0.90** | 轴距略小于轮距 | ≈ 0.9 |
| 离地间隙 / 轮径 | 144.9 / 355.5 = **0.41** | 离地间隙 ≈ 0.4×轮径 | 0.4 |

---

## 4. 核心教训：为什么"轮子完全暴露在外面"

**Language-3D 之前的错误（2026-06-24 审计）：**

```
生成器输出（payload=5kg）:
  body: 240mm × 240mm（正方形！）
  track = 200mm, wheelbase = 200mm
  body_width = track × 1.2 = 240mm
  body_length = wheelbase × 1.2 = 240mm
  wheel_diameter = 80mm
```

求解后实际位置：
```
  body X 范围: ±120mm
  wheel X 中心: ±163mm  ← 超出 body 边缘 43mm！
```

**问题根源**：
1. `body_width = track × 1.2` 把轮子往里塞 20%，但 Husky 是 `body_width = track`
   （轮子在边缘）。我们的轮子反而被推出去了——因为 anchor 偏移（left/right = ±半宽）
   叠加到 track/2 上，把轮心推到 ±163 > ±120。
2. `body_length = wheelbase × 1.2 = 240` 让车体几乎是正方形。Husky 是 1.97× 轴距
   的长方形。正方形车体让轮子看起来在四角"悬空"，而不是内缩于长边。
3. wheel_diameter=80mm 太小（track 的 0.4×），Husky 是 0.62×。小轮子在正方形小
   车体上像贴片，不像真车轮。

**正确做法（基于 Husky URDF）**：
- `body_width ≈ track`（轮子半嵌入侧面，中心在车体边缘）
- `body_length ≈ wheelbase × 1.9`（车体长方形，轮子内缩于前后端）
- wheel_diameter 取 track 的 0.55–0.62（大轮子，像 Husky）

---

## 5. 轮子-车体关系的几何模型（生成器实现指南）

从上方俯视（X 右 = 行驶方向，Y 上 = 左）：

```
        ←────── body_length (1007) ──────→
   ┌──────────────────────────────────────┐  ▲
   │                                      │  │ body_width
   │   ●─────────────●          ●─────●   │  │ (571)
   │   │ fl     fr  │          │顶板  │   │  │
   │   ●─────────────●          ●─────●   │  │
   │                                      │  │
   │   ●─────────────●                    │  │
   │   │ rl     rr  │                     │  │
   │   ●─────────────●                    │  │
   └──────────────────────────────────────┘  ▼
        ↑轮距 track=571=body_width          ↑轴距 wheelbase=512
        （轮子在车体侧边边缘，半嵌入）        （轮子内缩于车体前后端）
```

注意：
- 轮子的**圆面**朝 ±Y（轮轴沿 Y），所以轮子在俯视图里是**沿 X 方向延伸的矩形**
  （轮径×轮宽投影）。
- 4 个轮子在**长方形的四个角内侧**，不是悬在外面的四个突起。
- 车体长边（X）远大于轮距（Y），所以车体看起来像"长盒子下面装了四个轮子"，
  不是"小方块两侧挂了四个轮子"。

---

## 6. Language-3D 坐标系映射

Husky URDF 的坐标系与 Language-3D 的 solver 坐标系映射：

| Husky URDF | Language-3D solver (ANCHOR_DIRECTIONS) | 语义 |
|---|---|---|
| X (+前 -后) | X (+right -left) via `left`/`right` anchor | **不同！** Husky X 是前后，我们 X 是左右 |
| Y (+左 -右) | Y (+back -front) via `front`/`back` anchor | Husky Y 是左右，我们 Y 是前后 |
| Z (+上) | Z (+上) | 相同 |

**关键差异**：Husky 的行驶方向是 X，轮轴是 Y。但在 Language-3D 里：
- 轮子用 `axis="y"`（绕 Y 转）→ 轮轴沿 Y → 轮子在**左右**（X 方向）滚动，行驶方向是 X。
- 这和 Husky 一致！轮轴 Y，行驶 X。
- 但我们的 `base_length`（dimensions.length）映射到哪个轴？需要查 solver。

**确认（assembly_solver.py）**：part dimensions `{length, width, height}`：
- `length` → 沿 X 轴（行驶方向，前后）✓ 和 Husky box size X 一致
- `width` → 沿 Y 轴（左右，轮轴方向）
- `height` → 沿 Z 轴

所以：
- `base_plate.dimensions.length` = 车体前后长度（应 ≈ 2× wheelbase）
- `base_plate.dimensions.width` = 车体左右宽度（应 ≈ track）

**等等**——需要验证 solver 到底把 length 映射到哪个轴。如果 length→Y，
那 body 的长边就在左右方向，与轮轴平行——这正是用户要的"轮子跟长边平行"。
**必须实际渲染确认，不能假设。**（见 `test_solver_dimension_mapping` 验证）

---

## 来源

- [bulletphysics/bullet3 — husky.urdf](https://github.com/bulletphysics/bullet3/blob/master/data/husky/husky.urdf)（完整 URDF，直接可读）
- [husky/husky — husky_description/urdf](https://github.com/husky/husky/blob/noetic-devel/husky_description/urdf/husky.urdf.xacro)（xacro 源，需展开）
- [Clearpath Husky A200 用户手册](https://docs.clearpathrobotics.com/docs_robots/outdoor_robots/husky/a200/user_manual_husky/)
- [Dual UR5 Husky MuJoCo 模型](https://github.com/wangcongrobot/dual_ur5_husky_mujoco)（基于 Husky 底盘 + 双 UR5 臂的真实参考项目）
