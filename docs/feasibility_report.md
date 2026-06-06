# Language-3D 复杂机器人设计能力升级 — 可行性报告

> 编制日期：2026-06-05
> 研究范围：2024-2026 年全球 AI CAD / 机器人设计 / 工程自动化全领域
> 覆盖资料：60+ 论文 / 产品 / 工具（截至 2026.6.5 最新）

---

## 一、项目定位

### 1.1 我们在做什么

Language-3D Agent 是一个自主 AI 系统，从自然语言描述出发，自动完成 3D 机械零件的建模、装配、仿真、代码生成、文档生成、3D 打印准备的全流程。

**目标升级**：从当前单臂 3-DOF 机器人（8 零件）扩展到复杂移动机器人（40-50 零件），例如：

```
        ┌─ 左臂 (3-DOF) ─┐  ┌─ 右臂 (3-DOF) ─┐
        └──────────────────┘  └──────────────────┘
              ┌──────────────────────┐
              │    工控机 (IPC)       │
              │  + 传感器 + 电池     │
              └──────────────────────┘
        ┌──────────────────────────────────┐
        │        底盘 (Chassis)             │
        │  ┌──┐              ┌──┐          │
        │  │轮│ 电机        │轮│ 电机      │
        │  └──┘              └──┘          │
        │  ┌──┐              ┌──┐          │
        │  │轮│ 电机        │轮│ 电机      │
        │  └──┘              └──┘          │
        └──────────────────────────────────┘
```

### 1.2 为什么这件事值得做

**全球没有第二个系统能做到这件事。**

经过 3 轮深度调研，覆盖 60+ 论文/产品/工具，结论明确：

> **自然语言 → 带完整工程验证的多零件机器人设计，全球没有已实现的系统。**

这是一片空白赛道。

---

## 二、全球竞争格局（2024-2026）

### 2.1 最接近的竞争对手

| 系统 | 类型 | 时间 | 能力 | 关键缺口 |
|------|------|------|------|----------|
| **Leo AI** (YC) | 商用 | 2026 | 文本→多零件装配体，对接 PLM/供应商目录 | 非机器人专用，无运动学/物理验证 |
| **RobotDesignGPT** | 学术 | arXiv 2026.01 | 文本/图片→URDF 机器人描述文件 | 输出 URDF 不是工程 CAD，无结构验证 |
| **ArtiCAD** | 学术 | arXiv 2026.04 | 多 Agent 协作生成可活动装配体（家具） | 非机器人，无物理约束，无工程验证 |
| **Text2Robot** | 学术 | arXiv 2024-25 | 文本→四足机器人→物理制造 | 网格表示非参数化 CAD，无工程细节 |
| **SOLIDWORKS 2026** | 商用 | 2026 | AI 装配结构设计、图纸自动化 (AURA/LEO) | 闭源商业，通用机械非机器人专用 |
| **STEP-LLM** | 学术 | DATE 2026 | 文本→STEP 文件（首个直接产出 STEP） | 仅单零件 |
| **Physics-in-the-Loop** | 学术 | arXiv 2026 | 物理-验证闭环的 CAD 生成架构 | 通用 CAD，非机器人 |
| **Self-Improving CAD w/FEA** | 学术 | arXiv 2026 | FEA 反馈驱动的多零件 STEP 生成 | 面向板件，非机器人 |
| **NURBGen** | 学术 | AAAI 2026 | 文本→NURBS 参数化 CAD（STEP 输出） | 仅单零件 |
| **Blox-Net** | 学术 | UC Berkeley | VLM 监督的生成式积木装配 | 积木雕塑，非功能机器人 |

### 2.2 竞争对手能力矩阵

| 能力 | Leo AI | RobotDesignGPT | ArtiCAD | Text2Robot | SOLIDWORKS | **Language-3D (目标)** |
|------|:------:|:--------------:|:-------:|:----------:|:----------:|:---------------------:|
| 自然语言输入 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 多零件装配 | ✅ | ❌ | ✅ | 部分 | ✅ | ✅ |
| 参数化 CAD 输出 | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| 机器人专用 | ❌ | ✅ | ❌ | ✅ | ❌ | **✅** |
| 运动学/动力学 | ❌ | 部分 | ❌ | 部分 | ❌ | **✅** |
| 物理验证 (FEA) | ❌ | ❌ | ❌ | ❌ | 部分 | **✅** |
| 稳定性分析 | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| 电机/执行器选型 | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| 固件代码生成 | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| URDF 导出 | ❌ | ✅ | ❌ | ❌ | ❌ | **✅** |
| BOM + 文档 | 部分 | ❌ | ❌ | ❌ | 部分 | **✅** |
| 3D 打印切片 | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| 开源 | ❌ | ✅ | ✅ | ✅ | ❌ | **✅** |

**Language-3D 的独特组合（全球唯一）：** 自然语言 + 参数化 CAD + 多 Agent + VLM 验证 + 机器人专用知识 + 物理验证 + 固件生成 + URDF 导出 + 工程文档。

### 2.3 关键学术参考

| 论文 | 关键贡献 | 对我们的启示 |
|------|---------|-------------|
| ArtiCAD (arXiv 2604.10992) | 首个免训练多 Agent 装配生成 | 验证多 Agent 装配架构可行 |
| Physics-in-the-Loop (arXiv 2605.19717) | 物理验证闭环的 CAD 生成 | 物理在环是我们应该采用的架构模式 |
| RobotDesignGPT (arXiv 2601.11801) | VLM 从文本生成 URDF | 验证 VLM 视觉反馈迭代设计可行 |
| Seek-CAD (arXiv 2505.17702) | DeepSeek 驱动的自精化 CAD | 自精化循环模式值得借鉴 |
| CADMorph (NeurIPS 2025) | Plan-Generate-Verify 循环编辑 | 验证我们已有的验证循环是正确方向 |
| Text2Robot (arXiv 2406.19963) | 文本→物理机器人端到端 | 最接近的端到端系统，但用网格不用 CAD |
| MorphoGen (ICLR 2026 sub) | LLM 驱动机器人形态进化 | 形态优化代码生成方法可参考 |
| RoboMoRe (arXiv 2506.00276) | 机身+控制协同优化 | 机-电-控协同设计值得借鉴 |
| Force-Angle (Papadopoulos 1996) | 稳定性指标经典算法 | 稳定性分析的标准方法 |

---

## 三、现有架构能力评估

### 3.1 已完成能力（Task 1-42）

| 类别 | 已有工具 | 数量 |
|------|---------|------|
| 文件操作 | read/write/edit/search/glob/list_dir | 6 |
| 命令执行 | bash, python_exec | 2 |
| 截图 | screen/window/list_windows | 3 |
| VLM 分析 | vlm_analyze/screen_analyze/window_analyze/cad_verify/vlm_locate | 5 |
| FreeCAD 建模 | fc_batch + 22 个专用工具 | 23 |
| GUI 自动化 | click/type/hotkey/drag/scroll 等 | 8 |
| 仿真 | fea_run/fea_vlm_analyze/interference_check/motion_sim 等 | 6 |
| CFD | cfd_run/cfd_vlm_analyze | 2 |
| 零件库 | part_search/get/generate/list/import/save/assemble | 8 |
| 装配/IK | assembly_solve, ik_solve | 2 |
| 切片 | slice_model/analyze/preview/vlm_analyze | 4 |
| 执行器 | actuator_select/analyze/power_budget | 3 |
| 代码生成 | gen_firmware/wiring_diagram/test_sequence | 3 |
| BOM/文档/质量 | gen_bom, gen_assembly_guide, quality_check | 3 |
| 生产 | production_check, iteration_design, scheme_compare, print_optimize | 4 |
| **合计** | | **~86 个工具注册** |

测试覆盖：600+ 测试全通过。

### 3.2 核心架构瓶颈（40-50 零件场景）

| 瓶颈 | 当前状态 | 影响 | 改动规模 |
|------|---------|------|----------|
| **扁平规划** | Planner 生成 `List[PlanStep]`，无子系统概念 | 50 零件 → 50+ 步骤，Agent 迷失 | 大改 planner.py, state.py |
| **开环装配** | assembly_solver 只支持树状 BFS/DFS | 四轮底盘闭环链无法求解 | 大改 assembly_solver.py |
| **无质量属性** | Part 无 mass/CoM/inertia | 无法做稳定性分析、力矩计算 | 中改 mechanics.py, freecad.py |
| **单臂 IK** | 只支持单臂 3-DOF + CCD | 双臂 12+ DOF 无法求解 | 大改 ik_solver.py |
| **无稳定性分析** | 完全缺失 | 工控机+双臂重心偏高无法验证 | 新建 stability.py |
| **无差速运动学** | 完全缺失 | 底盘运动学无法计算 | 新建 mobile_base.py |
| **仅舵机固件** | 只有 PWM 舵机控制 | 底盘直流电机无法驱动 | 中改 code_gen.py |
| **基本体素建模** | box/cylinder/sphere/cone + boolean | 复杂支架/外壳无法高效建模 | 中改 freecad.py |

---

## 四、技术可行性分析

### 4.1 核心技术成熟度

| 技术领域 | 成熟度 | 难度 | 理论依据 |
|----------|--------|------|----------|
| **分层规划** | 工程实现级 | 高（工程量大） | ArtiCAD 已验证多 Agent 装配生成可行；PLanner 本身有 LLM few-shot 基础 |
| **差速运动学** | 教科书级 | 低 | `v = (v_r+v_l)/2, ω = (v_r-v_l)/L`，公式完全确定 |
| **闭环装配求解** | 成熟 | 中高 | 迭代约束求解 (Newton-Raphson) 是标准方法 |
| **质心/稳定性** | 成熟 | 中 | 递归质心（加权平均）+ Force-Angle（Papadopoulos 1996）是标准算法 |
| **双臂 IK** | 学术前沿 | 高 | Jacobian + nullspace 需矩阵运算，但有成熟数学基础 |
| **碰撞检测** | 成熟 | 中 | 胶囊体近似 + GJK 算法是游戏/机器人标准方法 |
| **直流电机 PID** | 工程级 | 中 | PID 控制是嵌入式标准实现 |
| **URDF 导出** | 成熟 | 低 | FreeCAD RobotCAD v10 已有参考实现 |
| **电缆走线** | 研究级 | 中高 | JPS 路径搜索 + B-spline 平滑有论文方案（Oxford JPS 2024） |
| **草图建模** | 成熟 | 中 | FreeCAD Sketcher API 完整支持 |

### 4.2 风险评估

| 风险 | 等级 | 概率 | 影响 | 缓解措施 |
|------|------|------|------|---------|
| LLM 规划 40+ 零件时丢失上下文 | **高** | 40% | 严重 | 分层规划 + 子系统隔离，LLM 每次只看一个子系统 |
| FreeCAD 复杂零件子进程超时 | 中 | 30% | 中等 | 超时增至 300s，子装配分批处理 |
| VLM 验证复杂装配准确率不足 | 中 | 35% | 中等 | 三级验证策略（零件级→子系统级→整机级） |
| 双臂 Jacobian IK 不收敛 | 中 | 25% | 中等 | CCD + Jacobian 双求解器 + 随机重启 |
| 工程量超出预期 | **中高** | 45% | 严重 | 严格按 Phase 推进，每 Phase 有独立可交付成果 |
| 闭环装配求解器精度不足 | 中 | 20% | 中等 | 迭代精度阈值可调，收敛判据宽松化 |

### 4.3 可行性结论

**可行性评级：高。**

- 所有核心算法均有成熟方案，不存在未解决的科学问题
- 主要挑战在工程集成和系统复杂度管理，而非算法创新
- Language-3D 已有的 LLM + VLM + FreeCAD + 多 Agent 架构是正确的技术选型
- 与全球竞争对手相比，我们已有的功能覆盖面最广
- 参考文献中多个系统（ArtiCAD、Physics-in-the-Loop、RobotDesignGPT）已分别验证了我们需要的各个子架构模式

---

## 五、实施路线图

### Phase E：架构升级（基础，必须先做）

```
Task 43  分层规划架构 — HierarchicalPlan + 子系统分解
Task 44  分阶段编排器 — 子系统并行设计→集成验证
Task 45  零件质量属性 — 密度/质心/惯性张量
```

**依赖关系**：43 → 44 → 45
**这是后续所有 Phase 的基础。没有分层规划，40+ 零件的设计无法运作。**

### Phase F：移动底盘能力

```
Task 46  移动底盘知识库 — 差速运动学/电机力矩/电池计算
Task 47  闭环装配求解器 — 四轮约束/差速器
Task 48  稳定性分析 — Force-Angle/ZMP/翻倒风险
```

**依赖关系**：45 → 48（质量属性先于稳定性）、46 → 47（底盘知识先于闭环求解）

### Phase G：双臂协调 + 高级建模

```
Task 49  双臂协调 IK — Jacobian + 碰撞检测 + 工作空间
Task 50  直流电机固件 — PID + 编码器 + 差速驱动
Task 52  FreeCAD 高级建模 — 扫掠/放样/阵列/镜像/薄壳
Task 53  零件库扩展 — 轮子/支架/铜柱/底盘零件
Task 56  草图建模 — Sketch + Extrude/Revolve
```

**依赖关系**：46 → 50；49/52/53/56 可并行

### Phase H：系统集成与输出

```
Task 51  功率预算与电池选型
Task 54  ROS URDF 自动导出
Task 55  电缆走线规划
Task 57  复杂机器人端到端验证（最终验收）
Task 58  Web 面板复杂机器人支持
```

**依赖关系**：45 → 54；57 依赖所有其他 Task

### 完整依赖图

```
Phase E (基础)
  43 ──→ 44 ──→ 45
                     │
                     ├──────────────────────────┐
                     ▼                          ▼
Phase F (底盘)                     Phase G (双臂+建模)
  46 ──→ 47                        49 (独立)
  45 ──→ 48                        52 (独立)
                                   53 (独立)
                                   46 ──→ 50
                                   56 (独立)
                     │                          │
                     └──────────┬───────────────┘
                                ▼
Phase H (集成输出)
  51 (独立)     54 (依赖45)     55 (独立)
                     │
                     ▼
  57 ── 最终验收（依赖所有 Task）
  58 ── Web 面板（可与 57 并行）
```

---

## 六、Language-3D 的独特价值主张

### 6.1 全球唯一的能力组合

| 维度 | Language-3D | 最接近竞争者 |
|------|------------|-------------|
| 输入方式 | 自然语言 | Leo AI 也是自然语言 |
| 输出格式 | 参数化 CAD (FCStd/STEP/STL) | Leo AI 有 CAD，RobotDesignGPT 只有 URDF |
| 机器人专用知识 | 差速运动学/双臂IK/稳定性/电机选型 | **全球唯一** |
| 视觉验证 | VLM 多角度验证 (4-level) | CADCodeVerify 有，但不支持装配体 |
| 物理验证 | FEA + 稳定性 + 干涉 | Physics-in-the-Loop 有 FEA，但不做机器人 |
| 固件生成 | 舵机 PWM + 直流电机 PID + 差速 | **全球唯一** |
| URDF 导出 | 自动从装配体生成 ROS2 包 | RobotDesignGPT 有，但无工程 CAD |
| 工程文档 | BOM + 装配指导 + 质量控制 + 打印 | **全球唯一** |
| 开源 | 是 | ArtiCAD/Text2Robot 是，Leo AI/SOLIDWORKS 否 |

### 6.2 做成后的定位

> **全球第一个从自然语言到完整工程级机器人设计的自主 AI 系统。**

涵盖：需求解析 → 子系统分解 → 零件参数化建模 → 装配约束求解 → 运动学分析 → 稳定性验证 → 执行器选型 → 固件代码生成 → ROS URDF 导出 → BOM → 装配指导 → 3D 打印切片 → 生产检查。

没有任何现有系统（学术或商业）覆盖这个完整链路。

---

## 七、参考资料

### 7.1 核心竞争系统

| 系统 | 链接 |
|------|------|
| Leo AI | https://www.getleo.ai/ |
| RobotDesignGPT | arXiv:2601.11801 |
| ArtiCAD | arXiv:2604.10992 |
| Text2Robot | arXiv:2406.19963 |
| SOLIDWORKS 2026 AI | https://www.solidworks.com/product/solidworks-design/ai-overview |
| STEP-LLM | arXiv:2601.12641 |
| Physics-in-the-Loop | arXiv:2605.19717 |
| Self-Improving CAD w/FEA | arXiv:2605.17448 |
| NURBGen (AAAI 2026) | arXiv:2511.06194 |
| Text2CAD (NeurIPS 2024) | arXiv:2409.17106 |
| CAD-Llama (CVPR 2025) | arXiv:2505.04481 |
| Seek-CAD | arXiv:2505.17702 |
| MorphoGen | OpenReview (ICLR 2026 sub) |
| RoboMoRe | arXiv:2506.00276 |

### 7.2 技术基础参考

| 领域 | 参考 |
|------|------|
| Force-Angle 稳定性 | Papadopoulos & Rey, "The Force-Angle Measure of Tipover Stability Margin" |
| 差速运动学 | CMU Mobile Robot Kinematics, Caltech DiffDrive Dynamics |
| 双臂协调 | MDPI Machines 12(6):387, EPFL Sparse Collision Avoidance |
| 电缆走线 | Oxford JPS Cable Routing (JCDE 2024) |
| FreeCAD RobotCAD | https://github.com/drfenixion/freecad.robotcad (v10.8.1) |
| UrdfArchitect | https://discourse.openrobotics.org (AI URDF Editor) |
| 拓扑优化开源 | topoptlab (JOSS 2025), FEniCSx, OpenPISCO |
| NVIDIA PhysicsNeMo | https://developer.nvidia.com/physicsnemo |
| GLM-5 | arXiv:2602.15763 (MIT License, 744B params) |
| GLM-5V-Turbo | Multimodal coding foundation model |

### 7.3 综述文献

| 综述 | 链接 |
|------|------|
| LLMs for CAD Survey | arXiv:2505.08137 |
| AI-driven 3D CAD Generation Survey | IEEE Xplore (Apr 2026) |
| Generative AI for 3D CAD in Engineering | IEEE Computer Graphics (Feb 2026) |
| Foundation Models in Robotics | arXiv:2604.15395 |

---

## 附录：调研方法

1. **第一轮**：全局 SOTA 调研 — AI CAD 系统、AI 机器人设计、VLM 工程验证、生成式设计、数字孪生（覆盖 5 大领域）
2. **第二轮**：深度技术调研 — 差速运动学、双臂协调、稳定性分析、电子-机械集成、开源机器人工具、装配层级管理（覆盖 6 大子领域）
3. **第三轮**：2026 最新专项 — 14 个专项搜索覆盖 2026.01-06 所有新发布（产品/论文/工具），确认无遗漏竞争对手

总计覆盖 **60+ 篇论文、30+ 产品/工具、10+ 开源项目**，时间跨度 2024-2026.06.05。
