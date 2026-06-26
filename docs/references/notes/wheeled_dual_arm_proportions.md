# 真实轮式双臂机器人的尺寸比例参考

> 用途：约束 Language-3D 生成器，避免"机械臂比机器人底盘还大""臂伸到 5 倍底盘外"这类比例失真。
> 所有数据来自真实产品规格页 / 论文，非编造。更新于 2026-06-24。

## 为什么需要这份文档

代码审计发现 `arm_topology.py` 生成的臂是**独立桌面机械臂**尺度（base 200×150mm、连杆 120/100/80/60mm、总臂展 ~445mm），
而 `assembly_compose` 把它原样塞到一个 120×120mm 的小型轮式底盘上，导致：
- 臂底座（200×150）比整个机器人底盘（108×108）还大
- 臂前伸 600mm，远超底盘 ±60mm 的范围
- 双臂底座互相重叠（两个 200mm 底座只隔 ±70mm = 140mm 间距）

真实机器人不是这样设计的。下面是真实数据。

---

## 1. 真实轮式（双臂/单臂）移动操作机器人

### Toyota HSR（单臂，家用辅助）
来源：[Toyota 官方](https://global.toyota/en/detail/8709541)、[robotsguide](https://robotsguide.com/robots/hsr)

| 参数 | 值 |
|---|---|
| 底盘直径 | **430 mm**（圆柱形全向底盘） |
| 机体高度 | 1005–1350 mm（可调） |
| 臂长 | **~600 mm** |
| 臂前伸距离 | **~450 mm** |
| 臂垂直可达 | 地面至 1350 mm |
| 重量 | ~37 kg |

**关键比例**：臂前伸(450) / 底盘直径(430) ≈ **1.05**。臂水平前伸基本等于底盘宽度，而不是 5 倍。

### PAL TIAGo / TIAGo++（单臂/双臂，研究服务）
来源：[PAL Robotics](https://pal-robotics.com/robot/tiago/)、[The Robot Report](https://www.therobotreport.com/tiago-robot-pal-robotics-ready-two-armed-tasks/)、[arXiv 2510.10273](https://arxiv.org/html/2510.10273v1)

| 参数 | 值 |
|---|---|
| 底盘（omni/differential） | 约 500mm 级（官方 datasheet） |
| TIAGo++ 臂 | **双 7-DOF 臂** |
| 臂可达 | 地面至 1750 mm（垂直） |
| 驱动 | 全向麦克纳姆轮 / 差速 |
| 单臂负载 | ~6 kg |

**关键比例**：双 7-DOF 臂的安装底座（肩部）紧凑地排在 ~500mm 底盘上方，臂向上/向前伸展但水平前伸受底盘尺度约束。

### Fetch Freight（单臂，仓储）
- 底盘约 500mm，单臂，垂直可达高货架；臂水平前伸约 0.5–0.6m，与底盘尺度相当。

---

## 2. 关键设计原则（用于校验生成器）

| 原则 | 真实机器人 | Language-3D 当前（错误） |
|---|---|---|
| **臂水平前伸 / 底盘宽度** | ≈ 1.0–1.5 | 600/120 = **5.0** ❌ |
| **臂安装底座 vs 底盘甲板** | 臂底座**小于**甲板（约 30–50%） | 臂底座(200×150)**大于**甲板(108×108) ❌ |
| **双臂底座间距** | 两肩间距 < 底盘宽度 | 200mm 底座隔 140mm → 重叠 ❌ |
| **舵机/电机** | 真实 COTS（Dynamixel XM430: 28×46.5×34mm；MG996R: 40.7×19.7×42.9mm） | 发明的 Ø40/36/30/28（无型号）❌ |

**底线**：一个移动操作机器人的臂，其前伸距离应当与底盘尺度同量级（1–1.5 倍），
绝不能达到底盘尺寸的数倍。臂底座必须**小于**承载它的甲板。

---

## 3. 移动臂 vs 固定臂的尺度差异

| 类型 | 底座 | 臂展 | 比例 | 来源 |
|---|---|---|---|---|
| **固定桌面臂**（如 BCN3D MOVEO） | 大底座 300+mm | ~400mm | 臂/底 ≈ 1.3 | — |
| **移动单臂**（HSR） | 底盘 430mm | 前伸 450mm | ≈ 1.05 | Toyota |
| **移动双臂**（TIAGo++） | 底盘 ~500mm | 双臂紧凑排列 | < 1.5 | PAL Robotics |

固定臂底座可以大（螺栓固定在工作台，靠大底座稳）。移动臂底座必须**小**（受底盘甲板约束，且要给轮子/电池留空间）。
所以 `arm_topology` 的 `_BASE_PLATE = {200,150,8}`（固定臂尺度）**不能**原样用于移动底盘。

---

## 4. 真实 COTS 舵机/电机规格（功能件，不可缩放）

这些在 `parts_catalog.py` 里有正确记录，生成器**应当引用**而非发明：

| 型号 | 主体尺寸 L×W×H (mm) | 用于 | 文件 |
|---|---|---|---|
| TowerPro SG90 | 22.2 × 11.8 × 31.0 | 夹爪舵机 | `parts_catalog.py:3403` |
| TowerPro MG996R | 40.7 × 19.7 × 42.9 | 中型臂关节 | `parts_catalog.py:3427` |
| JX DS3218 | 40.0 × 20.0 × 38.5 | 大扭矩臂关节 | `parts_catalog.py:4098` |
| ROBOTIS Dynamixel XM430-W350-T | 28.0 × 46.5 × 34.0 | 工业级臂关节 | `parts_catalog.py:5119` |

**规则（AGENTS.md §1.2）**：电机/舵机是功能件，有真实规格，生成器必须从 catalog 取值，禁止发明或缩放。
`arm_topology._SERVO_DIMS`（发明的 Ø40/36/30/28）应替换为对 catalog 的引用。

---

## 5. 代码审计发现（2026-06-24）

- `arm_topology.py:140-154`：`_SERVO_DIMS`/`_LINK_LENGTHS`/`_BASE_PLATE` 全是**发明的启发式常数**，未引用 `parts_catalog` 的真实舵机。
- `arm_topology.py` import 只有 json/math/typing，**不引用**任何 catalog。
- `assembly_compose.py`：把固定臂尺度的臂原样塞到小底盘，无比例耦合。
- 单臂独立比例**可接受**（臂展 360mm / 底座 200mm ≈ 1.8），双臂**失配**因为同一个大底座臂被装到小底盘上。
- 三处代码可非法缩放功能件（无 `part_class`/`scalable` 守卫）：
  - `agent/modifier.py:201` `_scale_part`
  - `tools/iteration.py:341` `_apply_reach`
  - `agent/assembly_visual_verifier.py:731` `scale_part`

## 修复方向（正确做法，非缩放捷径）

1. **移动臂用独立的、更小的尺度档**：移动底盘上的臂，连杆应取 60–100mm（不是 120），底座应 ≤ 甲板的 50%。
2. **舵机从 catalog 取真实值**：`arm_topology` 引用 MG996R/DS3218/XM430，而非发明 _SERVO_DIMS。
3. **`assembly_compose` 加比例校验**：组装时检查臂底座 ≤ 甲板面积、臂前伸 ≤ 底盘对角线的 1.5 倍，否则报错让上层重选尺度档（不是缩放）。
4. **三处缩放代码加功能件守卫**：`part_class == "functional"` 或 `scalable is False` 时拒绝缩放。

---

## 来源

- [Toyota HSR 官方规格](https://global.toyota/en/detail/8709541)
- [robotsguide HSR](https://robotsguide.com/robots/hsr)
- [PAL Robotics TIAGo](https://pal-robotics.com/robot/tiago/)
- [The Robot Report: TIAGo++ 双臂](https://www.therobotreport.com/tiago-robot-pal-robotics-ready-two-armed-tasks/)
- [arXiv 2510.10273 TIAGo++ Omni](https://arxiv.org/html/2510.10273v1)
- [NSF HSR 评估论文](https://par.nsf.gov/servlets/purl/10466728)
