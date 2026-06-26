# 夹爪"实心块"问题：根因与解法

> 用途：Language-3D 的夹爪在 VLM 验证里始终被报为"实心块"的诊断与解法。
> 研究于 2026-06-24。**关键发现颠覆了之前的假设：几何是对的，问题在 VLM 模型降级。**

## 颠覆性发现

用独立视觉模型验证已渲染的 `gripper_closeup.png`，判定是：

> "TWO separate finger prongs with a gap/opening between them (not one solid block)... two distinct, parallel prongs."

**夹爪几何和 VTK 渲染在 closeup 视角下是视觉正确的。"实心块"是 VLM 解读/模型降级问题，不是几何问题。**

代码本身已定位到同一根因（`assembly_generator.py:3054`）：
```
# critical: GLM-4.6V-Flash 和 GLM-4V-Plus 都会把夹爪误判为 "solid block" (false negative)
```
只是没被严格执行。

## 比例规范（证明几何不是病态值）

| 夹爪 | 指宽 W | gap/W 比 | 说明 |
|---|---|---|---|
| Robotiq 2F-85 | ~22mm | ~1.9(全开)/~1.0(工作位) | 行业基准 |
| Franka Panda Hand | ~20mm | ~1.0 | 极简 |
| ROBOTIS RH-P12-RN | ~25mm | ~0.9-1.0 | 窄行程 |
| **Language-3D** | **14mm** | **1.4** | **在区间内** |

**结论：1.4 比例落在工业典型区间 [1.0, 1.9] 内。不要为"显得更开"放大 gap。**

## 业界共同模式

MuJoCo Menagerie / PyBullet / ROS / Gazebo 的夹爪都遵循：
1. 指是独立 link + 独立 mesh + 对比色（已做：finger 强制黄色）
2. **总是有针对夹爪开口的专门相机**，绝不依赖全局 iso（已有 gripper_closeup）
3. MuJoCo/PyBullet demo 默认拉近相机到指尖

## 解法（按 ROI 排序）

### 🥇 修复 1（最高 ROI，几乎零成本）：强制完整版 GLM-4.6V + 确认视角路由
1. `_vlm_check_assembly` 入口断言 `vision_model == "GLM-4.6V"`（完整版），禁止 Flash/Plus 降级进入夹爪判定
2. 确认 GRIPPER_INVISIBLE/FINGER_OVERLAP 判定**仅**采信 gripper_closeup，整体视图的"solid block"不参与
3. 这两项改完，大概率问题消失

### 🥈 修复 2（中 ROI，~30 行）：增加正面直视开口视角
在 `VIEW_PRESETS` 加 `gripper_front_open`，沿 forward 轴看进 gap（direction=(0,-1,0)）。
两指会呈现为两个并排矩形块 + 中间贯通 gap，任何 VLM 都不会误判。
与 iso closeup 双视角判定。这是工业界（Robotiq/Franka 仿真）标准做法。

### 🥉 修复 3（低 ROI，仅在 1+2 仍失败时）：对比图 prompt
预渲染黄金夹爪，横向拼接当前 closeup，in-context 对比。仅重试路径启用。

### ❌ 不推荐：改 CAD 指几何
- 指几何 + 1.4 比例已在工业区间内
- closeup 渲染已被独立 VLM 确认视觉正确
- 改 CAD（倒角/L 尖端/ribs）对 iso 投影"两指可见"信号增益有限，却增加非流形风险（已踩过坑）
- 只有 1+2 仍失败且诊断为 VTK 着色问题时才动"指腹倒角/开浅槽"

## CAD 几何（如必须改，按此顺序）
- P0：不要布尔 cut 出 shell，用单实体 makeBox + makeFillet（避免非流形）
- P1：指腹内侧开浅槽（深 1mm）形成阴影线，仿 Robotiq 指腹纹路
- P2：尖端只做 45° 倒角，不做 L 形/钩状（避免 iso 投影自遮挡）

## 学术引用
- [CADCodeVerify (ICLR 2025)](https://arxiv.org/abs/2501.03182) — render→VLM→fix→loop，每轮视角与判定责任匹配
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（robotiq_2f85, franka_fr3）
- [PyBullet assets / panda_gripper.urdf](https://github.com/bulletphysics/bullet3)
- [Robotiq 2F-85 URDF (ros-industrial)](https://github.com/ros-industrial/robotiq)
- RT-2 / PaLM-E (Google)：robotics VLM 对细小机械部件识别高度依赖 crop/zoom，全图显著退化

## 一句话
> 夹爪不是方块——是验证器在用降级模型或错误视角看它。先锁死"完整版 GLM-4.6V + 仅 gripper_closeup 裁判"，再加正面直视开口视角；几何和比例都不要动。
