<div align="center">

# Language-3D Agent

**Autonomous Multi-Expert Agent System for Production-Level 3D Modeling**
**自主多专家智能体系统 — 生产级 3D 建模**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FreeCAD 1.1](https://img.shields.io/badge/FreeCAD-1.1-green.svg)](https://freecad.org)
[![GLM-4.6](https://img.shields.io/badge/LLM-GLM--4.6-orange.svg)](https://open.bigmodel.cn)
[![E2E Score](https://img.shields.io/badge/E2E-92.1%25-brightgreen.svg)](#test-results)

*LLM Reasoning + VLM Visual Perception + CAD Automation + Geometric Arbitration*
*LLM 推理 + VLM 视觉感知 + CAD 自动化 + 几何仲裁*

</div>

---

## Overview / 概述

**English**

Language-3D Agent is an autonomous AI system that understands natural language descriptions of mechanical assemblies, generates production-level 3D models (STL/STEP), exports complete engineering packages (URDF, BOM, assembly guide, firmware), and verifies results through **dual-channel inspection** — code-side geometric checks and vision-side AI analysis, with **geometric arbitration** that overrules VLM false-negatives when the geometry is provably correct.

The system uses a **multi-expert agent pipeline** (Architect → Solver → CAD Engineer → Verifier → Fixer) where each stage is an independent specialist with its own system prompt and tool whitelist. Failures are routed by the Fixer to the specific stage that needs re-running, not a full regeneration.

**中文**

Language-3D Agent 是一个自主 AI 系统，能够理解自然语言描述的机械装配体，生成生产级 3D 模型（STL/STEP），导出完整工程包（URDF、BOM、装配指南、固件），并通过**双通道验证**（代码侧几何检查 + 视觉侧 AI 分析）确保质量。当几何验证确认正确时，**几何仲裁**会推翻 VLM 的误判。

系统使用**多专家智能体流水线**（架构师 → 求解器 → CAD 工程师 → 验证器 → 修复器），每个阶段是独立的专家角色，有自己的系统提示和工具白名单。失败时修复器精准路由到出问题的阶段，而非整体重生成。

```
"设计一个 4 自由度机械臂，带夹爪"     ← Natural Language / 自然语言
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Architect Agent       Verifier Agent
   (GLM-4.6)             (GLM-4.6V)
         │                     │
    Solver → CAD          Geometric Arbitration
         │              + VLM Split Verification
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Engineering Package │
         │  工程包输出           │
         │  ├ STL parts (11)    │
         │  ├ URDF + MuJoCo     │
         │  ├ BOM + Guide       │
         │  ├ FreeCAD scripts   │
         │  └ Firmware          │
         └─────────────────────┘
```

---

## Quick Start / 快速开始

### Prerequisites / 前置条件

- Python 3.12+
- [FreeCAD 1.1+](https://freecad.org) — for 3D modeling / 用于 3D 建模
- GLM API key from [Zhipu AI](https://open.bigmodel.cn) / 智谱 AI API 密钥

### Installation / 安装

```bash
pip install -e .

# Configure / 配置
cp .env.example .env
# Edit .env: add GLM_API_KEY / 编辑 .env 添加密钥
```

### Usage / 使用

```bash
# Interactive CLI / 交互式命令行
lang3d

# E2E production pipeline (legacy loop) / 端到端生产流水线（传统循环）
python tests/test_e2e_production.py --case 4dof_arm

# E2E with multi-expert pipeline / 使用多专家流水线
python tests/test_e2e_production.py --case 4dof_arm --pipeline
```

```python
# Programmatic / 编程接口
from lang3d.tools.assembly_generator import generate_assembly_from_nl

assembly = generate_assembly_from_nl("4自由度机械臂，带夹爪")
# → Assembly with parts, joints, connections, default_angles
```

---

## Architecture / 架构

### Multi-Expert Agent Pipeline / 多专家智能体流水线

```
┌──────────────────────────────────────────────────────────────┐
│                  AssemblyPipeline / 流水线编排                 │
│                                                              │
│   "Design a 4-DOF arm with gripper"                         │
│   "设计一个带夹爪的 4 自由度机械臂"                              │
│                          │                                   │
│              ┌───────────┼───────────┐                       │
│              ▼           ▼           ▼                       │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐                 │
│  │ Architect  │→│  Solver    │→│   CAD    │                 │
│  │ 架构师      │ │  求解器     │ │ Engineer │                 │
│  │ GLM-4.6    │ │ Positions  │ │ STL 生成  │                 │
│  │ NL→JSON    │ │ +Sanitize  │ │ FreeCAD  │                 │
│  └────────────┘ └────────────┘ └────┬─────┘                 │
│                                     ▼                        │
│                              ┌──────────┐                    │
│                              │ Verifier │                    │
│                              │ 验证器    │                    │
│                              │ GLM-4.6V │                    │
│                              │ Geo+VLM  │                    │
│                              └────┬─────┘                    │
│                          ┌────────┴────────┐                 │
│                          ▼                 ▼                 │
│                     [ PASS ]         [ FAIL ]               │
│                     → Export       → Fixer 路由              │
│                                    ┌────────────┐            │
│                                    │  Fixer     │            │
│                                    │ 修复路由    │            │
│                                    │ 精准回退    │            │
│                                    └────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

### Expert Agent Roles / 专家角色

| Role / 角色 | Model / 模型 | Responsibility / 职责 | Tools / 工具 |
|---|---|---|---|
| **Architect / 架构师** | GLM-4.6 | NL → assembly JSON, joint chain, part selection | assembly_template, part_recommend |
| **Solver / 求解器** | Deterministic | assembly.json → 3D positions, collision resolution | assembly_solver, mesh_collision |
| **CAD Engineer / CAD 工程师** | GLM-4.6 | positions → water-tight STLs | fc_batch, fc_script, cad_verify |
| **Verifier / 验证器** | GLM-4.6V | Dual-channel: geometric checks + VLM visual | vlm, _geometric_prevalidation |
| **Fixer / 修复器** | Deterministic | Classify failure → route to correct upstream stage | _classify_problems |

### Assembly Generation Pipeline / 装配体生成流水线

| Stage / 阶段 | Module / 模块 | Description / 描述 |
|---|---|---|
| 1. Architect / 架构 | `assembly_generator.py` + `pipeline.py` | GLM-4.6 generates assembly JSON from prompt |
| 2. Sanitize / 清洗 | `assembly_generator.py` | Fix anchors, grippers, proportions, angles, base sizing |
| 3. Solve / 求解 | `assembly_solver.py` | Compute 3D positions from joint constraints |
| 4. CAD Generate / CAD 生成 | `part_feature_engine.py` | FreeCAD scripts → STL meshes (axis-correct bolt holes) |
| 5. Render / 渲染 | `vtk_renderer.py` | VTK offscreen multi-view + crop-to-content + clipping fix |
| 6. VLM Verify / VLM 验证 | `vlm.py` + `_vlm_check_assembly` | Split: structural (panoramic) + gripper (close-up) |
| 7. Export / 导出 | `export_package.py` | STL + URDF + BOM + Guide + Firmware + ROS2 |

### Verification Architecture / 验证架构

| Channel / 通道 | Method / 方法 | Authority / 权威性 |
|---|---|---|
| **Geometric / 几何仲裁** | Check 1-7: collision, connectivity (BFS), COM, finger gap, arm pose | **Ground truth** — overrules VLM when geometry is correct |
| **VLM Structural / VLM 结构** | Panoramic views (iso/front/top/right) — structural integrity only | Advisory (geometry arbitrates) |
| **VLM Gripper / VLM 夹爪** | Dedicated close-up view — two finger prongs + gap | Sole authority for gripper question |
| **MuJoCo Physics / 物理仿真** | PD-hold stability test (prismatic joint clamping) | Must pass for URDF export |

---

## Tool System / 工具系统

57 tool modules organized into 9 categories. Each expert agent sees only its whitelisted tools (`ROLE_TOOL_CATEGORIES`):

| Category / 类别 | Tools | Key Modules / 关键模块 |
|---|---|---|
| FreeCAD Modeling / 建模 | 23 | `freecad.py` — box, cylinder, boolean, fillet, export |
| File Operations / 文件 | 6 | `file_ops.py` — read, write, edit, search, glob |
| GUI Automation / GUI 自动化 | 8 | `gui_action.py` — click, type, hotkey, drag, scroll |
| VLM Vision / 视觉 | 4 | `vlm.py` — analyze, verify, screen capture |
| Assembly / 装配 | 10+ | `assembly_generator.py`, `assembly_solver.py`, `pipeline.py` |
| CAD Features / CAD 特征 | 3+ | `part_feature_engine.py`, `connection_features.py` |
| Export / 导出 | 2+ | `export_package.py`, `urdf_export.py`, `bom_gen.py` |
| Rendering / 渲染 | 1 | `vtk_renderer.py` — offscreen VTK + crop + clipping |
| Utilities / 工具 | 2+ | `ik_solver.py`, `mesh_collision.py`, `sim_mujoco.py` |

---

## Knowledge Base / 知识库

| Module / 模块 | Content / 内容 |
|---|---|
| `mechanics.py` | Parts (66 types), Joints, Assemblies, ConnectionMethod |
| `assembly_patterns.py` | Robot profiles: BCN3D MOVEO, Thor, PAROL6, ANYmal, Solo12 |
| `fastener_catalog.py` | ISO/DIN fasteners, bolt length, torque specs |
| `actuators.py` | Servo/stepper database (NEMA17, Dynamixel, etc.) |
| `assembly_templates.py` | Few-shot examples for LLM prompt |
| `materials.py` | PLA, ABS, PETG, aluminum properties |
| `tolerance.py` | ISO 286 IT grades, fits (H7/g6, H7/js6) |

---

## Project Structure / 项目结构

```
language-3d/
├── src/lang3d/
│   ├── agent/                     # Multi-Agent / 多智能体
│   │   ├── pipeline.py            # AssemblyPipeline (5-stage expert flow) / 流水线
│   │   ├── assembly_generator_helpers.py  # Stage function re-exports
│   │   ├── sub_agent.py           # Expert roles (Architect/Solver/CAD/Verifier/Fixer)
│   │   ├── orchestrator.py        # DAG-based wave-parallel orchestration
│   │   ├── core.py                # Agent main entry + dispatch
│   │   ├── planner.py             # Task decomposition (flat/DAG/hierarchical)
│   │   ├── executor.py            # Step execution + auto-fix loop
│   │   ├── verifier.py            # Dual-channel verification
│   │   ├── modifier.py            # Targeted modification engine
│   │   ├── fix_strategy.py        # Failure classification + convergence detection
│   │   ├── dag.py                 # Task DAG with cycle detection
│   │   ├── message_bus.py         # Inter-agent pub/sub messaging
│   │   └── ...
│   ├── models/                    # LLM/VLM Backends
│   │   ├── glm.py                 # GLM-4.6 (text) + GLM-4.6V (vision)
│   │   └── router.py              # 4-level vision routing (fast/standard/detailed/maximum)
│   ├── tools/                     # 57 tool modules
│   │   ├── assembly_generator.py  # NL → assembly JSON + VLM loop + geometric arbitration
│   │   ├── assembly_solver.py     # Position & constraint solving
│   │   ├── part_feature_engine.py # Per-part CAD features (axis-correct bolt holes)
│   │   ├── connection_features.py # Bolt holes (anchor-rotated), bearings, snaps
│   │   ├── vtk_renderer.py        # VTK rendering (crop-to-content, clipping fix)
│   │   ├── export_package.py      # Engineering package export
│   │   ├── urdf_export.py         # URDF (inertial frame-correct)
│   │   ├── sim_mujoco.py          # MuJoCo physics (prismatic joint clamping)
│   │   └── ...
│   ├── knowledge/                 # Domain Knowledge
│   │   └── ...
│   ├── web/app.py                 # FastAPI dashboard
│   └── config.py                  # Configuration
├── tests/                         # 3,600+ tests
│   ├── test_e2e_production.py     # E2E pipeline (--pipeline flag for multi-agent)
│   ├── test_expert_roles.py       # Expert agent role + tool whitelist tests
│   └── ...
├── data/runs/                     # E2E outputs (canonical layout)
├── examples/                      # Usage examples
└── pyproject.toml
```

---

## Test Results / 测试结果

| Suite / 测试套件 | Tests | Status / 状态 |
|---|---|---|
| Unit + Integration / 单元 + 集成 | 3,608 | 3,605 PASS, 3 known failures (gripper/coordinate) |
| E2E Production (legacy) / 端到端 | 1 | **92.1% score, 0 critical fails** |
| E2E Production (pipeline) / 流水线 | 1 | **92.1% score, 0 critical fails** |
| Expert Roles / 专家角色 | 21 | 21 PASS |

```bash
# Run tests / 运行测试
python -m pytest tests/ -m "not e2e" -q    # Unit tests / 单元测试
python tests/test_e2e_production.py --case 4dof_arm           # Legacy e2e
python tests/test_e2e_production.py --case 4dof_arm --pipeline  # Multi-agent pipeline
```

### E2E 4dof_arm Score Breakdown / 端到端评分明细

| Phase / 阶段 | Checks | Status |
|---|---|---|
| Phase 1: NL → Assembly | 5 | ✅ 11 parts, 10 joints, connected tree |
| Phase 2: Position Solving | 4 | ✅ 11/11 positioned, 0 NaN |
| Phase 3: Render Quality | 2 | ✅ 4 views, 873KB avg |
| Phase 4: Engineering Package | 10 | ✅ All files exported |
| Phase 5: Content Validation | 9 | ✅ **VLM verification: PASSED** |
| Phase 6: Physical Sanity | 5 | ✅ COM stable, 0 collisions, 733mm workspace |
| Phase 7: MuJoCo Simulation | 3 | ✅ Physics stable, 6 joints actuated |

---

## Roadmap / 路线图

### Phase 1 — Foundation / 基础 (Completed)
- [x] FreeCAD subprocess bridge (23 tools)
- [x] VLM visual verification (4-level routing)
- [x] GUI automation (PyAutoGUI, 8 tools)
- [x] Dual verification pipeline

### Phase 2 — Assembly Generation / 装配体生成 (Completed)
- [x] NL → assembly JSON (GLM-4.6 + VLM feedback loop)
- [x] Assembly solver (anchor constraints)
- [x] Connection engine (bolted, press_fit, snap_fit, adhesive, welded, magnetic)
- [x] Part feature engine (axis-correct bolt holes, water-tight fingers)
- [x] VTK offscreen rendering (crop-to-content, clipping fix)
- [x] Engineering package (STL, URDF, BOM, assembly guide, firmware)

### Phase 3 — Multi-Agent Architecture / 多智能体架构 (Completed)
- [x] Expert agent roles (Architect/Solver/CAD/Verifier/Fixer)
- [x] Tool whitelisting per role (ROLE_TOOL_CATEGORIES)
- [x] AssemblyPipeline (5-stage flow with Fixer routing)
- [x] Geometric arbitration (Check 7 connectivity + false-alarm filtering)
- [x] Split VLM verification (structural panoramic + gripper close-up)
- [x] MuJoCo physics validation (prismatic joint clamping)

### Phase 4 — Simulation & Verification / 仿真与验证 (In Progress)
- [x] URDF export (joint limits, mimic joints, inertial frame-correct)
- [x] MuJoCo PD-hold physics stability test
- [ ] URDF `<transmission>` + `ros2_control` for Gazebo actuation
- [ ] Closed-chain kinematic solver
- [ ] FEA structural analysis (CalculiX)
- [ ] Grasp simulation (sim_grasp three-phase)
- [ ] Dual-arm collision avoidance

### Phase 5 — Production System / 生产系统
- [ ] Web dashboard (real-time monitoring)
- [ ] Part library expansion (66 → 200+ types)
- [ ] Manufacturing process planning (G-code)
- [ ] Quality control (statistical verification)

---

## License

MIT License

---

## Acknowledgments / 致谢

- [FreeCAD](https://freecad.org) — Open-source parametric 3D CAD
- [Zhipu AI / GLM](https://open.bigmodel.cn) — GLM-4.6 reasoning + GLM-4.6V vision
- [MuJoCo](https://mujoco.org) — Physics simulation
- [VTK](https://vtk.org) — Offscreen 3D rendering
- [trimesh](https://trimesh.org) — Mesh processing + python-fcl collision
- [PyAutoGUI](https://pyautogui.readthedocs.io) — GUI automation
