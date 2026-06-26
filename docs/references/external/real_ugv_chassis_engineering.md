# 真实 UGV/AGV 底盘工程结构参考（权威数据）

> 用途：重写 Language-3D 底盘生成器的权威依据。所有数据来自官方手册/URDF 源文件。
> 摘录于 2026-06-24。**这是机械工程数据，不是家具运动学。**

## 为什么需要这份文档

之前被 articulate-anything（PartNet 家具数据集）带偏：它不建电机 body、不建悬架、轮子直接挂 caster。
真实 UGV（Husky、Leo Rover）是完全不同的机械结构。这份文档纠正方向。

---

## 1. 正确的 URDF 连接树（来自官方 Husky URDF）

来源：[husky/husky — husky_description/urdf/husky.urdf.xacro](https://github.com/husky/husky/blob/noetic-devel/husky_description/urdf/husky.urdf.xacro)

```
base_footprint                                    ← 根，Z=0 地面投影
   └─ (fixed joint) base_link                     ← 车体本体（含 visual/collision/inertial）
         ├─ (continuous joint, axis=y) front_left_wheel_link
         ├─ (continuous joint, axis=y) front_right_wheel_link
         ├─ (continuous joint, axis=y) rear_left_wheel_link
         └─ (continuous joint, axis=y) rear_right_wheel_link
```

### 关键结构事实（颠覆我之前的错误假设）

1. **轮子的 parent 是 base_link，不是电机、不是转向节、不是悬架。**
   - Husky 用 4×4 belt drivetrain（皮带传动），电机在车体内部，**不显式建为独立 link**。
   - 轮子通过 `continuous joint`（无限旋转）直接挂在 base_link 上，axis=y（水平左右轴）。

2. **没有悬架 link。** Husky 是刚性底盘，靠大直径低压轮胎（330mm）+ 橡胶形变吸震。
   - 官方 URDF 里没有 prismatic joint、没有 spring、没有 damper link。

3. **base_footprint 是虚拟根**（Z=0 地面），base_link 通过 fixed joint 抬高到离地间隙。

4. **轮子位置在 base_link 的四角**，origin xyz 直接写在 wheel joint 里（如 front_left 在 +x+y 角，距 base_link 中心 = wheelbase/2, track/2）。

### Leo Rover 结构（类似，更小）
来源：[Fictionlab Leo Rover 文档](https://docs.fictionlab.pl/leo-rover/documentation/specification)
- 同样 base_link → 4 wheel_link (continuous joint)，无悬架 link。
- Leo Rover 有用户自行加装的悬架扩展，但**默认出厂是刚性**。

---

## 2. 真实尺寸数据（校验生成器用）

### Husky A200（中型室外 UGV，50kg，75kg 载荷）
来源：[Clearpath 用户手册](https://docs.clearpathrobotics.com/docs_robots/outdoor_robots/husky/a200/user_manual_husky/)、[规格对比页](https://clearpathrobotics.com/husky-spec-comparison/)

| 参数 | 值 |
|---|---|
| 外形 (L×W×H) | 990 × 670 × 390 mm |
| 轮距 track | **555 mm** |
| 轴距 wheelbase | **512 mm** |
| 轮径 | **330 mm** |
| 离地间隙 | **130 mm** |
| 重量 | 50 kg |
| 载荷 | 75 kg |
| 驱动 | 4×4 皮带传动 |

### Leo Rover（小型教育 UGV，6.5kg）
来源：[Fictionlab 规格](https://docs.fictionlab.pl/leo-rover/documentation/specification)

| 参数 | 值 |
|---|---|
| 外形 (L×W×H) | 433 × 447 × 249 mm |
| 轮距 track | **354 mm** |
| 轴距 wheelbase | **295 mm** |
| 轮径 | **130 mm** |
| 离地间隙 | **108 mm** |
| 重量 | 6.5 kg |
| 载荷 | 5 kg |

### 关键比例（用于校验）

| 比例 | Husky | Leo Rover | 我该用 |
|---|---|---|---|
| 轮径 / 轮距 | 330/555 = **0.59** | 130/354 = **0.37** | 0.4–0.6 |
| 离地间隙 / 轮径 | 130/330 = **0.39** | 108/130 = **0.83** | 0.4–0.8 |
| 车体宽 / 轮距 | 670/555 = **1.21** | 447/354 = **1.26** | **车体比轮距宽 ~20%**（轮子不在最外侧！） |

**重要纠正**：车体宽度 **大于** 轮距（Husky 670 > 555，Leo 447 > 354）。
这意味着**轮子中心在车体宽度范围内**（不是像我之前那样把轮子推到车体外侧）。
轮子半嵌在车体侧面，不是完全在外面。我之前的"轮子必须在外壳外"是错的。

---

## 3. 我的底盘生成器该怎么改（基于真实结构）

### 正确的连接树（推荐）
```
base_footprint (Z=0 地面根)
   └─ (fixed) base_link (车体本体, 抬高到离地间隙)
         ├─ (continuous, axis=y) wheel_fl   ← 轮子直接挂车体
         ├─ (continuous, axis=y) wheel_fr
         ├─ (continuous, axis=y) wheel_rl
         └─ (continuous, axis=y) wheel_rr
```

### 与我当前结构的差异（要改的）

| 我当前（错误） | Husky 真实结构 | 改法 |
|---|---|---|
| `base_plate → motor(body) → suspension → wheel` | `base_link → wheel`（电机在内部不显式建） | **简化**：电机作为 base_link 的 metadata/BOM，不建独立 body link；或建但放在 base_link 内部（不挂轮子） |
| chassis_body 外壳包住一切 | base_link **就是**车体（visual+collision 合一） | base_link 兼任外壳，不额外加 chassis_body |
| 轮子推到 X=±121（外壳外） | 轮子在 track/2=±277（Husky），在车体宽度(±335)内 | 轮子位置 = track/2，车体宽 = track×1.2（轮子半嵌车体） |
| 悬架 prismatic joint | **无悬架**（刚性） | 删掉 suspension_link，轮子直接 continuous joint 挂 base_link |

### 轮子位置的正确计算
- track_width 决定轮子 Y 位置（左右）：wheel_Y = ±track/2
- wheelbase 决定轮子 X 位置（前后）：wheel_X = ±wheelbase/2
- 车体宽 = track × 1.2（轮子在车体内，半嵌侧面）
- 轮径 = track × 0.4~0.6

---

## 4. 关于悬架的最终结论

**Husky 和 Leo Rover 默认都是刚性底盘，没有悬架。** 离地间隙靠大轮径 + 轮胎形变。
- Husky: 330mm 轮径 + 130mm 离地间隙（轮子半径 165mm > 离地间隙 130mm，base_link 底部在轮轴上方）
- 如果项目要悬架（作为增强），参考 rocker-bogie（NASA 火星车）或独立 A 臂，但**不是 prismatic joint 简单竖向滑动**——那是简化模型，不是真实悬架结构。

**推荐**：默认刚性（跟 Husky 一致），悬架作为可选 profile，默认不建。

---

## 来源
- [Husky 官方 URDF (GitHub husky/husky)](https://github.com/husky/husky/blob/noetic-devel/husky_description/urdf/husky.urdf.xacro)
- [Husky A200 用户手册 (Clearpath)](https://docs.clearpathrobotics.com/docs_robots/outdoor_robots/husky/a200/user_manual_husky/)
- [Husky 规格对比 (Clearpath)](https://clearpathrobotics.com/husky-spec-comparison/)
- [Leo Rover 技术规格 (Fictionlab)](https://docs.fictionlab.pl/leo-rover/documentation/specification)
- [ROS Wiki husky_description](http://wiki.ros.org/husky_description)
