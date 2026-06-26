# 轮式机器人悬架建模（真实参考 + 落地方案）

> 研究于 2026-06-24。结论来自真实机器人规格 + MuJoCo/URDF 文档 + 代码审计。

## 核心认知纠正

**大多数轮式机器人根本没有弹簧悬架——刚性挂载是主流。**
- TurtleBot/Husky/Fetch/Ridgeback/Jackal：**全刚性**，轮直接拧在电机/底盘上
- Husky A300 实测：**刚性车桥**，靠大直径低压轮胎 + 橡胶形变吸震，不是独立弹簧悬架
- articulate-anything 全部轮式数据集：刚性建模
- 真正的弹簧悬架是少数派：NASA 火星车(rocker-bogie 连杆式)、汽车级 AGV(空气悬架)

**所以：加悬架是增强，不是修 bug。** 现有 `base_plate→motor→wheel` 刚性同轴结构在真实机器人里完全合理。

## URDF/MuJoCo 悬架能力对比

| 能力 | URDF | MuJoCo(MJCF) |
|---|---|---|
| prismatic joint | ✅ | ✅ |
| damping | ✅ `<dynamics damping>` | ✅ `dof_damping` |
| stiffness(弹簧) | ❌ **URDF 规范没有** | ✅ `dof_stiffness` |
| 程序注入 | — | ✅ 加载后写 `model.dof_stiffness` |

**关键：URDF 原生不支持弹簧**，想做真回弹悬架必须 MuJoCo 侧程序注入 stiffness，或用 ros2_control 力元。

## 落地方案：B（suspension_link 中间体）

结构：`motor →(prismatic,axis=z,垂向) suspension_link →(revolute,axis=y) wheel`

为什么 B 不是 A：URDF 是树不是图，一个 child 只能挂一个 parent joint。垂向跳动(prismatic)和滚动(revolute)必须串两个 link。

### 现有基础设施支持度（审计结论）

| 环节 | 支持 | 需改动 |
|---|---|---|
| solver 求解 prismatic | ✅ 现成 `assembly_solver.py:1018-1040` | 0 |
| URDF 导出 prismatic | ✅ mm→m 已对 `urdf_export.py:558-561` | 0 |
| URDF dynamics | ⚠️ 硬编码 0.1 `urdf_export.py:790` | 小(读字段) |
| MuJoCo 阻尼悬架 | ✅ | 小(`sim_mujoco.py` 解锁 suspension slide) |
| MuJoCo 弹簧悬架 | ⚠️ URDF 无字段 | 中(程序注入 dof_stiffness) |

### 必须改的（按优先级）

1. **`mobile_base_gen.py:288-306`**：每个 corner 加 suspension_link + prismatic joint，revolute 的 parent 从 motor 改成 suspension_link
2. **`sim_mujoco.py:539-563`**：`_run_physics_hold` 把所有 SLIDE joint 钉死（模拟夹爪锁），必须区分"夹爪 slide(锁)"和"悬架 slide(不锁)"——按名字含 suspension 跳过
3. **`mechanics.py:298`**(可选)：Joint 加 damping/stiffness 字段
4. **`urdf_export.py:790`**(可选)：damping 读字段非硬编码
5. **`sim_mujoco.py:335`**(可选)：对悬架 slide 注入 dof_stiffness 实现真回弹

## 引用
- [MuJoCo Modeling](https://mujoco.readthedocs.io/en/stable/modeling.html)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Husky 模型也是刚性）
- [dual_ur5_husky_mujoco](https://github.com/wangcongrobot/dual_ur5_husky_mujoco)
- [Clearpath Husky A300](https://clearpathrobotics.com/husky-a300-unmanned-ground-vehicle-robot/)
