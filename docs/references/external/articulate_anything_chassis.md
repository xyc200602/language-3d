# articulate-anything 的轮式底盘连接结构（真实参考）

> 来源项目：`C:\Users\xyc\ZCodeProject\references\articulate-anything`
> 摘录于 2026-06-24。这是真实开源项目的代码结构，不是我的笔记。

## 核心范式（一句话）

轮式底盘被建模成一棵 URDF 连接树：
`base →(fixed) chassis/leg/frame →(fixed) caster/axle →(revolute, axis=轮自转轴) wheel`

**轮子是一个 link，通过一个绕自转轴的 revolute joint 挂在父连杆（caster/axle）下。**
"轮子被驱动"= 那个 revolute joint 本身（施加扭矩/速度即驱动），不显式建电机 body。
**悬架完全不存在** —— 刚体运动学建模，不是车辆动力学。

## 正确的连接树（来自办公椅 partnet_36280）

```
base
 └─(fixed) chair_leg          # 底盘/中心柱
     ├─(fixed, "above") seat
     ├─(fixed, "below") caster        # 万向轮叉 = 轮子的直接父连杆
     │   ├─(revolute) wheel           # 轮 1
     │   └─(revolute) wheel_2         # 轮 2
     ├─(fixed, "below") caster_2
     │   ├─ wheel_3
     │   └─ wheel_4
     └─ ...
```

关键：
- **轮子绝不直接挂底盘**，中间一定有过渡连杆（caster/axle）
- 系统提示明确禁止 "关节直接 parent 是 base"，代码用 `_ensure_base_helper()` 强制插入中间 link

## 轮子关节 API

`Robot.make_revolute_joint(child, parent, global_axis, lower_deg, upper_deg)`
- 定义：`articulate_anything/api/odio_urdf.py:1144-1223`
- `global_axis` 世界系旋转轴，转父 link 局部系（`odio_urdf.py:1179`）
- 坐标约定：x=前后，y=左右，z=上下（`joint_actor.py:124-128`）
- 轮子自转轴 = 穿过轮心的水平轴（绕 y 轴，左右方向）

## 关键判断：哪些 articulate-anything 不做

| 它不做的 | 为什么 | Language-3D 该不该照搬 |
|---|---|---|
| 电机 body | PartNet 是家具数据集，目标"能动的家具" | ❌ 不照搬。项目期望要"生产级机器人"，电机是真实 COTS 功能件（AGENTS.md），保留 |
| 悬架 | 刚体运动学，非车辆动力学 | ⚠️ 至少体现"轮子能转"。悬架可作为后续增强 |
| transmission | URDF schema 占位，从不调用 | 后续 ros2_control 时再加 |

## Language-3D 底盘的当前错误（对照）

我的结构：`base_plate →(fixed) motor(body) →(revolute, axis=y) wheel`

对照 articulate-anything 的正确范式，问题在：
1. **轮子的 revolute joint 的 axis 是否真的是轮自转轴？** 需验证
2. **parent-child 层级是否对？** motor 作为 wheel 的 parent 是合理的（如果保留电机 body），但要确保 fixed joint(motor→base) 和 revolute joint(wheel→motor) 的 origin/axis 正确
3. **轮子位置（错位）** —— origin.xyz 是否正确反映轮子在底盘四角

## 关键文件（绝对路径）
- URDF DSL + 关节 API：`...\articulate_anything\api\odio_urdf.py`（make_revolute_joint @1144, place_relative_to @735, _ensure_base_helper @1037）
- 关节预测系统提示：`...\articulate_anything\agent\actor\joint_prediction\joint_actor.py:36-139`
- 轮式完整例子：`...\articulate_anything\examples\joint_examples_all.py:1137-1243`
- 轮子语义：`...\partnet_mobility_embeddings.csv`（"A wheel attached to cart body by a hinge"）
- 类别表（Cart）：`...\obj_types.json:1045-1107`
