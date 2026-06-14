# 相关研究全景 — Language-3D Agent

> 编制日期：2026-06-14
> 覆盖范围：2024-2026.6 全球 AI CAD / 机器人设计 / 工程自动化全领域
> 关联文档：`feasibility_report.md`、`solver_research_2025.md`、`freecad_assembly3_feasibility.md`

---

## 0. 文档目的

本文档汇总与 **Language-3D Agent**（自然语言 → 生产级 3D 机器人装配体多智能体系统）相关的全部公开研究、商业产品、开源项目与基准评测，按相似度分级，用于：

1. **定位差异化**：明确 Language-3D 在全球格局中的独占位
2. **对标架构**：找出可借鉴的多智能体 / VLM 闭环 / CAD 验证设计
3. **评测对齐**：选取合适的外部 benchmark 衡量本项目能力
4. **规避重复工作**：避免重新发明已成熟方案

---

## 1. Language-3D 项目自画像（用于对比基线）

| 维度 | 实现 |
|---|---|
| **目标** | 自然语言 → 多零件机器人装配体（机械臂/夹爪/轮式/足式）+ 完整工程包 |
| **架构** | Orchestrator + Planner + SubAgent×N + AssemblyVerifier + Reflector（5 智能体） |
| **LLM/VLM** | GLM-5.1（推理）+ GLM-4V（4 层路由视觉验证） |
| **CAD 后端** | FreeCAD 1.1 子进程 + Python 脚本（23 个建模工具） |
| **求解器** | BFS 树遍历 + 6 面 anchor + Rodrigues 旋转（1066 行） |
| **双通道验证** | 代码侧（mesh_stats、碰撞、水密性）+ 视觉侧（VLM cad_verify） |
| **输出包** | STL + URDF + BOM + 装配指南 + 固件代码 |
| **知识库** | 7 个机器人模板（BCN3D MOVEO / Thor / PAROL6 / ANYmal / Solo12…）、ISO/DIN 紧固件、ISO 286 公差 |
| **规模** | 57 工具模块、16 知识模块、3880 单元测试 + 4 E2E（91.4% 评分） |
| **License** | MIT 开源 |

---

## 2. 最直接的相似系统（按相似度降序）

### 2.1 ★★★★★ 商业直接对手

#### **Leo AI**（YC，2026）

- **官网**：[getleo.ai](https://www.getleo.ai/blog/best-ai-cad-tools-assembly-design-2026)
- **报道**：[Engineering.com](https://www.engineering.com/leo-ai-can-now-generate-full-cad-assemblies/)、[Machine Design](https://www.machinedesign.com/automation-iiot/article/55330891/leo-ai-leo-ai-how-cad-aware-ai-is-changing-mechanical-design-and-engineering-workflows)
- **能力**：文本 → 多零件参数化装配体，兼容 Solidworks / Onshape / Catia / Inventor；在 100 万+ 工程文献上训练，声称 96% 准确率、每任务节省 5+ 小时
- **与 Language-3D 重合**：自然语言输入、多零件装配、参数化 CAD 输出
- **Language-3D 独占**：机器人专用、运动学/动力学、URDF 导出、固件代码、3D 打印切片、开源

### 2.2 ★★★★★ 学术直接对手

#### **RobotDesignGPT**（arXiv 2026.01）

- **链接**：[arxiv.org/abs/2601.11801](https://arxiv.org/html/2601.11801v1)
- **能力**：文本 + 图像 → URDF 机器人描述文件，基于 VLM
- **与 Language-3D 重合**：自然语言、机器人专用、URDF
- **Language-3D 独占**：多零件参数化 CAD、结构验证、BOM、装配指南、固件

#### **CADSmith**（arXiv 2026.03，CMU Farimani Lab）

- **链接**：[arxiv.org/abs/2603.26512](https://arxiv.org/html/2603.26512v1)
- **作者**：J. Barkley, R. Loghmani, A. B. Farimani
- **能力**：Planner / Coder / Executor / Validator / Refiner 多智能体 → CadQuery 代码 → 程序化几何验证 + 迭代精修
- **与 Language-3D 重合**：**架构最像**（5-Agent ↔ Language-3D 的 Orchestrator+Planner+SubAgent+Verifier+Reflector 几乎一一对应）
- **Language-3D 独占**：装配体级（CADSmith 仅单零件）、机器人专用、VLM 视觉通道

### 2.3 ★★★★☆ 高度相似

#### **Text2Robot**（ICRA 2025，Duke）

- **链接**：[arxiv.org/abs/2406.19963](https://arxiv.org/html/2406.19963v1)
- **能力**：文本 → 四足机器人 → 实体制造（进化算法 + 真实电子件约束）
- **与 Language-3D 重合**：自然语言、机器人专用、考虑可制造性
- **Language-3D 独占**：参数化 CAD（非网格）、装配体级工程包、双通道验证

#### **Blox-Net**（UC Berkeley，ICRA 2025）

- **项目主页**：[bloxnet.org](https://bloxnet.org/)
- **链接**：[arxiv.org/abs/2409.17126](https://arxiv.org/abs/2409.17126)、[GitHub](https://github.com/Apgoldberg1/blox-net-coderelease)
- **能力**：VLM + 物理仿真 + 真实机器人 → GDfRA（生成式装配设计）；Top-1 识别率 63.5%
- **与 Language-3D 重合**：VLM 监督、物理仿真、装配生成
- **Language-3D 独占**：功能机器人（Blox-Net 是积木雕塑）、参数化 CAD、工程文档

#### **VLMGINEER**（UPenn 2026）

- **链接**：[PDF](https://www.seas.upenn.edu/~dineshj/publication/gao-2026-vlmgineer/gao-2026-vlmgineer.pdf)、[OpenReview](https://openreview.net/pdf?id=nESyz4PvJL)
- **能力**：VLM + 进化搜索 → 自主协同设计机器人工具/夹具
- **与 Language-3D 重合**：VLM 驱动、迭代设计、机器人夹具
- **Language-3D 独占**：装配体级、固件/URDF/工程包

#### **Text to Robotic Assembly of Multi-Component Objects**（arXiv 2025.11）

- **链接**：[NASA ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv251102162H/abstract)、[OpenReview PDF](https://openreview.net/pdf/2b24ec0d69dda37f95c11cb60c248f487099136c.pdf)
- **能力**：3D 生成 AI + VLM → 多元件装配
- **与 Language-3D 第 7 阶段（装配）高度重合**

#### **Self-Improving CAD Generation Agents with FEA**（arXiv 2026.05）

- **链接**：[arxiv.org/abs/2605.17448](https://arxiv.org/html/2605.17448v1)
- **能力**：FEA 反馈驱动的迭代 STEP 生成
- **与 Language-3D 重合**：闭环验证、多零件 STEP
- **Language-3D 独占**：机器人专用（FEA 论文面向板件）

### 2.4 ★★★☆☆ 部分相似

#### **ArtiCAD**（arXiv 2026.04）

- **能力**：多 Agent 协作生成**可活动装配体（家具）**
- **Language-3D 独占**：机器人、物理约束、工程验证

#### **STEP-LLM**（DATE 2026）

- **能力**：首个文本 → STEP 文件
- **Language-3D 独占**：多零件（STEP-LLM 仅单零件）

#### **NURBGen**（AAAI 2026）

- **能力**：文本 → NURBS 参数化 CAD（STEP 输出）
- **Language-3D 独占**：多零件、装配体

#### **Physics-in-the-Loop**（arXiv 2026）

- **能力**：物理验证闭环 CAD 生成架构
- **Language-3D 独占**：机器人专用

---

## 3. 基准与评测（Language-3D 当前未对接）

| 基准 | 链接 | 关键维度 | 推荐用途 |
|---|---|---|---|
| **MUSE** | [arXiv:2605.28579](https://www.semanticscholar.org/paper/a1d0494deacb7e65862080a14d39bdb4b5d55088) | "Manufacturable, Functional, **Assemblable**" text-to-CAD | **最对口**，建议作为 Language-3D 主评测 |
| **Text2CAD-Bench** | [arXiv:2605.18430](https://arxiv.org/html/2605.18430v1) | 首个系统化文本→参数化 CAD benchmark | 几何复杂度 + 应用多样性评测 |
| **P3D-Bench** | [arXiv:2606.11152](https://arxiv.org/html/2606.11152v1) | MLLM 参数化 3D 生成评测 | VLM 路由质量评测 |
| **HistCAD** | [arXiv:2602.19171](https://arxiv.org/html/2602.19171v3) | 约束感知 + 参数化历史 CAD | 求解器/约束系统评测 |
| **UniCAD** | [arXiv:2606.05058](https://arxiv.org/html/2606.05058v1) | 统一 benchmark（text/image/point-cloud→CAD + QA） | 多模态对齐评测 |

---

## 4. 单零件/单技术点基础研究

| 工作 | 链接 | 与 Language-3D 关系 |
|---|---|---|
| **Generative AI for CAD Automation** | [arXiv:2508.00843](https://arxiv.org/html/2508.00843v1) | **最接近的开源 FreeCAD+LLM 论文**（GPT-4+LangChain → FreeCAD Python 脚本）；单零件、无 VLM 验证 |
| **LLM4CAD** | [ASME J. Computing & Info. Sci. in Eng.](https://asmedigitalcollection.asme.org/computingengineering/article/25/2/021005/1208543) | 多模态 LLM 生成 3D CAD 的实验评估 |
| **NVIDIA GTC 2026 — NL→3D CAD Multi-Agent** | [GTC session](https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-p81308/) | 分层式多 Agent NL→参数化 CAD |
| **CADDesigner** | [ResearchGate](https://www.researchgate.net/publication/405189573_CADDesigner_Conceptual_CAD_model_generation_with_a_general-purpose_agent) | 通用 Agent 概念 CAD 生成 |
| **AI-driven 3D CAD Survey** | [SciOpen](https://www.sciopen.com/article/10.26599/CVM.2025.9450521) | 综述，含视觉反馈方法分类 |
| **Awesome Neural CAD** | [bunnysocrazy.com](https://bunnysocrazy.com/) | 神经 CAD 论文集，含 CAD-Coder 等 |

---

## 5. 开源/工程实现

| 项目 | 链接 | 说明 |
|---|---|---|
| **freecad-ai** | [github.com/ghbalf/freecad-ai](https://github.com/ghbalf/freecad-ai) | FreeCAD AI 助手 Workbench，聊天式生成 Python 代码 → 3D 模型；**最接近的开源工程实现** |
| **Zoo Keeper** | [zoo.dev/zookeeper](https://zoo.dev/zookeeper) | 开源 prompt-to-CAD（Zoo.dev/KittyCAD 内核） |
| **BlenderLLM** | [github.com/FreedomIntelligence/BlenderLLM](https://github.com/FreedomIntelligence/BlenderLLM) | NL → Blender 脚本；非参数化工程 CAD |
| **gNucleus** | [FreeCAD 论坛](https://forum.freecad.org/viewtopic.php?t=86927) | 商业化 FreeCAD GenAI（已知问题：FreeCAD API 变动 + LLM 训练数据滞后） |

---

## 6. 工业闭源产品

| 系统 | 状态 | 与 Language-3D 关系 |
|---|---|---|
| **SOLIDWORKS 2026**（AURA/LEO 集成） | 闭源 | AI 装配结构 + 图纸自动化；通用机械 |
| **Onshape → URDF 直出** | [报道](https://www.instagram.com/p/DTs76BwgdFm/) | CAD → URDF 桥接的商业方案 |
| **Siemens D-Cubed 3D DCM** | [siemens.com](https://www.siemens.com/en-us/products/plm-components/d-cubed/3d-dcm/) | 工业约束求解引擎（SolidWorks/NX/Inventor/Fusion 360 共同底层） |
| **FreeCAD Assembly3 (SolveSpace)** | [FreeCAD Wiki](https://wiki.freecad.org/Assembly3_Workbench) | 40+ 约束类型 vs Language-3D 的 6 面 anchor |

---

## 7. 能力矩阵对比

| 能力 | Leo AI | RobotDesignGPT | CADSmith | Blox-Net | Text2Robot | VLMGINEER | ArtiCAD | **Language-3D** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 自然语言输入 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 多零件装配 | ✅ | ❌ | ❌ | ✅ | 部分 | ❌ | ✅ | ✅ |
| 参数化 CAD (STEP/FCStd) | ✅ | ❌ | ✅ | ❌ | ❌ | 部分 | ✅ | ✅ |
| 机器人专用 | ❌ | ✅ | ❌ | ❌ | ✅ | 部分 | ❌ | ✅ |
| 运动学/动力学 | ❌ | 部分 | ❌ | ❌ | 部分 | ❌ | ❌ | ✅ |
| 物理验证 (FEA) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🚧 路线图 |
| VLM 视觉闭环 | ❌ | ❌ | 部分 | ✅ | ❌ | ✅ | ❌ | ✅ |
| 双通道验证 (code + visual) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| URDF 导出 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| BOM + 装配指南 | 部分 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 固件代码生成 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 3D 打印切片 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| 开源 | ❌ | ✅ | ✅ | ✅ | ✅ | 部分 | ✅ | ✅ |

**Language-3D 核心独占位**：**「自然语言 → 多零件机器人 + 双通道验证 + 完整工程包（URDF + BOM + 装配指南 + 固件）+ 开源」** 的组合，在已检索范围内（截至 2026.6.14）无公开系统具备。

> 与 `feasibility_report.md` 第 2.2 节的"空白赛道"判断一致。

---

## 8. 可借鉴的架构与方法

### 8.1 多智能体架构对标 CADSmith

CADSmith 的 5-Agent 流水线（Planner → Coder → Executor → Validator → Refiner）与 Language-3D 的 `Orchestrator + Planner + SubAgent + AssemblyVerifier + Reflector` **几乎一一对应**。可借鉴点：

- CADSmith 的 **程序化几何验证**（programmatic geometric validation）通过 CadQuery AST 检查，比 Language-3D 当前 mesh_stats 更结构化
- CADSmith 的 **迭代精修** 频率与策略可直接对比 Language-3D 的 `fix_strategy.py`

### 8.2 VLM 闭环对标 Blox-Net / VLMGINEER

- Blox-Net 的 **VLM-as-judge + 物理仿真双重验证** 是 Language-3D 双通道验证的同一思路
- VLMGINEER 的 **进化搜索 + VLM 评分** 可启发 Language-3D 的 Reflector 在多候选 fix 之间打分排序

### 8.3 FEA 反馈对标 Self-Improving CAD

`arXiv:2605.17448` 把 FEA 结果嵌入主循环作为反馈信号。Language-3D 当前 Phase 3 路线图（`README.md:264`）的 "FEA 结构分析 (CalculiX)" 应直接参考其反馈嵌入方式。

### 8.4 评测对齐 MUSE

**MUSE** 是当前唯一明确评测 "**assemblable**"（可装配）维度的公开 benchmark。建议：

1. 把 Language-3D 的 E2E 输出（4 个案例：4dof_arm 等）转换为 MUSE 输入格式
2. 对比 RobotDesignGPT / Text2Robot 在同基准下的得分
3. 用 MUSE 量化 Language-3D 当前 91.4% 评分在国际坐标下的位置

---

## 9. 对 Language-3D 后续研究的建议

基于项目自陈的 5 个限制（`solver_research_2025.md:14-20`）和外部研究：

| 优先级 | 方向 | 对标 | 预期影响 |
|:---:|---|---|---|
| **P0** | 碰撞检测（trimesh + python-fcl） | `solver_research_2025.md` 自行规划 | 立即消除零件穿透 |
| **P0** | 评测对齐 MUSE benchmark | MUSE (arXiv:2605.28579) | 国际可比性 |
| **P1** | 约束求解升级（6 面 anchor → coincident/concentric/distance） | D-Cubed / SolveSpace | 复杂装配几何正确性 |
| **P1** | 多 Agent 架构对标 CADSmith | CADSmith (arXiv:2603.26512) | Validator 结构化 |
| **P2** | FEA 反馈嵌入主循环 | Self-Improving CAD (arXiv:2605.17448) | Phase 3 落地 |
| **P2** | 进化搜索多候选 fix 排序 | VLMGINEER | Reflector 决策质量 |
| **P3** | 闭链运动学求解器 | ANYmal/Solo12 profile | 足式机器人完整性 |

---

## 10. 检索方法说明

- **检索时间**：2026-06-14
- **覆盖来源**：arXiv、ASME、IEEE、OpenReview、Semantic Scholar、ResearchGate、PyPI、GitHub、Engineering.com、Machine Design、官方厂商博客
- **检索关键词组合**（示例）：
  - `LLM agent natural language 3D CAD mechanical assembly generation FreeCAD`
  - `VLM visual verification 3D modeling agent robotic assembly URDF`
  - `RobotDesignGPT / Text2Robot / ArtiCAD / Blox-Net / VLMGINEER / CADSmith / MUSE`
  - `Leo AI YC mechanical engineering generative CAD assembly`
  - `text-to-CAD benchmark assembly multi-part parametric STEP`
- **项目内部交叉验证**：`feasibility_report.md` 第 2.1 节列出 10 个最接近系统，本文档新增了 **CADSmith、Self-Improving CAD w/ FEA、MUSE、Text2CAD-Bench、P3D-Bench、UniCAD、HistCAD、VLMGINEER、Text to Robotic Assembly (NASA ADS 2025.11)** 等 9 个项目方原报告未覆盖的工作。

---

## 附录 A：完整引用链接

### 学术论文
- RobotDesignGPT: https://arxiv.org/html/2601.11801v1
- CADSmith: https://arxiv.org/html/2603.26512v1
- Text2Robot (ICRA 2025): https://arxiv.org/html/2406.19963v1
- Blox-Net: https://arxiv.org/abs/2409.17126
- Blox-Net Code: https://github.com/Apgoldberg1/blox-net-coderelease
- VLMGINEER: https://www.seas.upenn.edu/~dineshj/publication/gao-2026-vlmgineer/gao-2026-vlmgineer.pdf
- Text to Robotic Assembly: https://ui.adsabs.harvard.edu/abs/2025arXiv251102162H/abstract
- Self-Improving CAD w/ FEA: https://arxiv.org/html/2605.17448v1
- Generative AI for CAD Automation: https://arxiv.org/html/2508.00843v1
- Text2CAD-Bench: https://arxiv.org/html/2605.18430v1
- P3D-Bench: https://arxiv.org/html/2606.11152v1
- UniCAD: https://arxiv.org/html/2606.05058v1
- HistCAD: https://arxiv.org/html/2602.19171v3
- MUSE: https://www.semanticscholar.org/paper/a1d0494deacb7e65862080a14d39bdb4b5d55088
- LLM4CAD (ASME): https://asmedigitalcollection.asme.org/computingengineering/article/25/2/021005/1208543
- CADDesigner: https://www.researchgate.net/publication/405189573_CADDesigner_Conceptual_CAD_model_generation_with_a_general-purpose_agent
- VLM-driven Skill Selection: https://arxiv.org/html/2511.05680v1
- AI-driven 3D CAD Survey: https://www.sciopen.com/article/10.26599/CVM.2025.9450521

### 商业/工业
- Leo AI: https://www.getleo.ai/blog/best-ai-cad-tools-assembly-design-2026
- Engineering.com Leo AI: https://www.engineering.com/leo-ai-can-now-generate-full-cad-assemblies/
- Machine Design Leo AI: https://www.machinedesign.com/automation-iiot/article/55330891/leo-ai-leo-ai-how-cad-aware-ai-is-changing-mechanical-design-and-engineering-workflows
- NVIDIA GTC 2026 NL→3D CAD: https://www.nvidia.com/gtc/session-catalog/sessions/gtc26-p81308/
- Siemens D-Cubed 3D DCM: https://www.siemens.com/en-us/products/plm-components/d-cubed/3d-dcm/
- FreeCAD Assembly3: https://wiki.freecad.org/Assembly3_Workbench
- SolveSpace: https://solvespace.github.io/solvespace-web/tech.html
- Onshape → URDF: https://www.instagram.com/p/DTs76BwgdFm/

### 开源项目
- freecad-ai: https://github.com/ghbalf/freecad-ai
- Zoo Keeper: https://zoo.dev/zookeeper
- BlenderLLM: https://github.com/FreedomIntelligence/BlenderLLM
- gNucleus 讨论: https://forum.freecad.org/viewtopic.php?t=86927
- SurveyBrainBody (Embodied Co-Design 集合): https://github.com/Yuxing-Wang-THU/SurveyBrainBody
- Awesome Neural CAD: https://bunnysocrazy.com/

### 碰撞/求解（关联 `solver_research_2025.md`）
- trimesh.collision: https://trimesh.org/trimesh.collision.html
- python-fcl: https://github.com/BerkeleyAutomation/python-fcl
- distance3d: https://pypi.org/project/distance3d/
- MeshLib: https://meshlib.io/feature/collision-detection/
- Geometric Constraint Solving Survey (Purdue): https://www.cs.purdue.edu/cgvlab/www/resources/papers/Bettig-Comp_and_Info_Sci_in_Eng-2011-Geometric_Constraint_Solving_In_Parametric_CAD.pdf
