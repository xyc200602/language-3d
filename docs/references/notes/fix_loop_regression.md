# 验证-修复循环退化：根因与解法

> 用途：Language-3D 的 NL→装配 生成循环里，"修一个问题引入另一个"的退化问题。
> 研究于 2026-06-24，含论文/项目引用。

## 现象

3 轮验证-修复循环，退化明显：
- Round 1（确定性生成）：底盘几何完全正确。验证器报 2 类问题：
  - 夹爪手指间距 12.4mm（外观问题）
  - `motor_fl` 和 `standoff_fl` overlap by **6x6x3mm**（角部极轻微相切）
- Round 2/3：fixer 对 6mm 相切做位移修正 → 把对的底盘带歪 → `base_plate and wheel_fl` overlap 11mm(r2)→22mm(r3)，**越修越糟**。

## 根因（两个工程缺口）

### 缺口 1：没有"净改善闸门"（Net-Improvement Gate）
`assembly_generator.py` Round 2/3 路径直接覆盖 assembly/positions，没有"修复后问题数 > 修复前就回滚"的逻辑。
- `fix_strategy.check_convergence` 只用字符串相似度检测"卡在同一失败"，检测不到"修 A 破坏 B"。
- `apply_targeted_fix_from_vlm` 有单修复局部 sanity guard（拒绝尺寸>2×），但不是全局回归检测。

### 缺口 2：6mm 相切本不该报
`check_assembly_collisions(skip_adjacent=True)` 只跳过 joint graph 的直接 parent-child 对。
但 `motor_fl` 和 `standoff_fl` 都挂在 `base_plate` 上（**siblings**，非 parent-child），skip_adjacent 没覆盖它们 → 6mm 预期接触被当 collision → 喂给 CollisionResolver → 位移修正 → 带歪底盘。

### 缺口 3：输出最后一轮而非历史最优
循环结束直接用当前 assembly 输出，而最后一轮恰恰是最差的（11→22mm 退化）。应输出 `best_assembly = min(history, key=score)`。

## 业界术语与解法

| 领域 | 术语 | 解法 |
|---|---|---|
| 自动程序修复(APR) | overfitting patch / regression patch | 修复后必须重验全部，只接受净改善 |
| 几何约束求解(CAD) | over-constrained / conflicting constraints | FreeCAD Assembly3: 冲突时 undo 最后加的约束 |
| LLM 自精炼 | regression error | Detect-Repair-Verify: 引入 functional regression 就停/回滚 |

三者共识：**修复后必须对全部约束重新验证，只接受"净改善"的修复**。

## 落地方案

### 第一优先：Net-Improvement Gate（堵住退化，改动最小）
在 `apply_targeted_fix_from_vlm` 之后、`_vlm_check_assembly` 之前：
1. `deepcopy` 前一个 assembly+positions 作快照
2. 用 `MeshCollisionChecker.generate_interference_report`（已有，返回带 severity）算 baseline 和 post-fix score
3. score 变差就回滚

score 用加权和（不是简单计数），让"消除严重穿透"压住"引入轻微相切"：
- severe（穿透≥2mm/plate_overlap/missing/floating）：权重 10
- moderate（0.5-2mm）：权重 3
- light/clearance（<0.5mm）：权重 1
- 预期接触：权重 0

`mesh_collision.py:234-236` 已有 none/clearance/light/moderate/severe 五级，直接用 severity 字段。

### 第二优先：预期接触白名单（消除假阳性源头）
扩展 `skip_adjacent` 从"joint 直接 parent-child"到更宽的拓扑关系：
- **结构件邻接**：standoff 和 motor 都固定在同一 base_plate 上 → 预期接触
- **运动链邻接**：齿轮/轴承/轴的配合面
- **紧固件穿透**：螺钉穿过孔

落地点：`mesh_collision._check_pair` 加 `_is_expected_contact(a, b)` 判定，category 组合在白名单（motor+standoff, bearing+shaft, screw+plate）→ 只报 severe 以上，light/moderate 降级为 warning 不进 problems。

### 第三优先：Best-so-far 输出 + 优先级修复
- 循环结束输出历史 best-score 轮，不是最后一轮
- 只对 severe 做修复，light/clearance/外观不进修复循环
- 任何 fix 被 gate 拒超过 2 次 → 该问题类型降级为"已知限制"，不再尝试

## 收敛保证
带闸门的接受策略保证 score 严格单调不增，max_rounds 内停在"最优已接受"状态（即使没收敛到 0 问题）。

## 引用
- [One Step Forward, Two Steps Back: Regression Errors in LLM Refinement (OpenReview)](https://openreview.net/pdf/d15ab24d7157dec4a8663870b541b4f8d100179c.pdf)
- [Detect-Repair-Verify (arXiv 2603.00897)](https://arxiv.org/html/2603.00897v1)
- [When Automated Program Repair Meets Regression Testing (ACM TOSEM 2024)](https://dl.acm.org/doi/10.1145/3672450)
- [Hoffmann et al., Making constraint solvers more usable: overconstraint (CAD Journal 2004)](https://www.sciencedirect.com/science/article/abs/pii/S001044850300099X)
- [FreeCAD Assembly3 Workbench wiki](https://wiki.freecad.org/Assembly3_Workbench)
- [Continuous Penetration Depth (CAD 2013)](https://www.sciencedirect.com/science/article/abs/pii/S001044851300153X)
