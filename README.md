<div align="center">

# Language-3D Agent

**Autonomous Multi-Agent System for Production-Level 3D Modeling**
**自主多智能体系统 — 生产级 3D 建模**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FreeCAD 1.1](https://img.shields.io/badge/FreeCAD-1.1-green.svg)](https://freecad.org)
[![GLM-5.1](https://img.shields.io/badge/LLM-GLM--5.1-orange.svg)](https://open.bigmodel.cn)
[![Tests](https://img.shields.io/badge/Tests-3880-brightgreen.svg)](#test-results)

*LLM Reasoning + VLM Visual Perception + CAD Automation + Dual Verification*
*LLM 推理 + VLM 视觉感知 + CAD 自动化 + 双通道验证*

</div>

---

## Overview / 概述

**English**

Language-3D Agent is an autonomous AI system that understands natural language descriptions of mechanical assemblies, generates production-level 3D models (STL/STEP), exports complete engineering packages (URDF, BOM, assembly guide, firmware), and verifies results through **dual-channel inspection** — code-side numerical checks and vision-side AI analysis.

The system can generate robotic arms, grippers, wheeled robots, and legged robots from a single natural language prompt, using a VLM-guided feedback loop to iteratively improve structural quality.

**中文**

Language-3D Agent 是一个自主 AI 系统，能够理解自然语言描述的机械装配体，生成生产级 3D 模型（STL/STEP），导出完整工程包（URDF、BOM、装配指南、固件），并通过**双通道验证**（代码侧数值检查 + 视觉侧 AI 分析）确保质量。

系统可从单条自然语言提示生成机械臂、夹爪、轮式机器人、足式机器人，使用 VLM 引导的反馈循环迭代改进结构质量。

```
"设计一个 4 自由度机械臂，带夹爪"     ← Natural Language / 自然语言
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   LLM Assembly Gen      VLM Visual Check
   (GLM-5.1)             (GLM-4V)
         │                     │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Engineering Package │
         │  工程包输出           │
         │  ├ STL parts (11)    │
         │  ├ URDF              │
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

# E2E production pipeline / 端到端生产流水线
python tests/test_e2e_production.py --case 4dof_arm
```

```python
# Programmatic / 编程接口
from lang3d.tools.assembly_generator import generate_assembly_from_nl

assembly = generate_assembly_from_nl("4自由度机械臂，带夹爪")
# → Assembly with parts, joints, connections, default_angles
```

---

## Architecture / 架构

### Multi-Agent Loop / 多智能体循环

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator / 调度器                  │
│             (Task Decomposition / 任务分解)               │
│                                                          │
│   "Design a 4-DOF arm with gripper"                     │
│   "设计一个带夹爪的 4 自由度机械臂"                        │
│                          │                               │
│              ┌───────────┼───────────┐                   │
│              ▼           ▼           ▼                   │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│     │ Modeling │  │  Vision  │  │   GUI    │            │
│     │ 建模     │  │  视觉    │  │  自动化  │            │
│     │ FreeCAD  │  │ GLM-4V   │  │PyAutoGUI │            │
│     └────┬─────┘  └────┬─────┘  └────┬─────┘            │
│          └──────────┬──┘──────────────┘                  │
│                     ▼                                    │
│            ┌─────────────────┐                           │
│            │  Verification   │                           │
│            │  验证            │                           │
│            │  Code + Visual  │                           │
│            └────────┬────────┘                           │
│                     │                                    │
│          [ PASS ]   [ FAIL → Fix → Retry ]              │
│          [ 通过 ]   [ 失败 → 修复 → 重试 ]                │
└─────────────────────────────────────────────────────────┘
```

### Assembly Generation Pipeline / 装配体生成流水线

| Stage / 阶段 | Module / 模块 | Description / 描述 |
|---|---|---|
| 1. NL Parse / 自然语言解析 | `assembly_generator.py` | LLM generates assembly JSON from prompt / LLM 从提示生成装配 JSON |
| 2. Sanitize / 清洗 | `assembly_generator.py` | Fix anchors, grippers, proportions, angles / 修复锚点、夹爪、比例、角度 |
| 3. Solve / 求解 | `assembly_solver.py` | Compute 3D positions from joint constraints / 从关节约束计算 3D 位置 |
| 4. CAD Generate / CAD 生成 | `part_feature_engine.py` | FreeCAD scripts → STL meshes / FreeCAD 脚本 → STL 网格 |
| 5. Render / 渲染 | `vtk_renderer.py` | VTK offscreen multi-view renders / VTK 离屏多视图渲染 |
| 6. VLM Verify / VLM 验证 | `vlm.py` | Visual analysis → pass/fail + feedback / 视觉分析 → 通过/失败 + 反馈 |
| 7. Export / 导出 | `export_package.py` | STL + URDF + BOM + Guide + Firmware / STL + URDF + BOM + 指南 + 固件 |

### Verification Channels / 验证通道

| Channel / 通道 | Method / 方法 | Checks / 检查项 |
|---|---|---|
| **Code-Side / 代码侧** | `mesh_stats`, collision detection | Volume, watertightness, collisions / 体积、水密性、碰撞 |
| **Visual-Side / 视觉侧** | `cad_verify`, `vlm_analyze` | Shape match, gripper recognition, structure / 形状匹配、夹爪识别、结构 |

---

## Tool System / 工具系统

57 tool modules organized into 9 categories:

| Category / 类别 | Tools | Key Modules / 关键模块 |
|---|---|---|
| FreeCAD Modeling / 建模 | 23 | `freecad.py` — box, cylinder, boolean, fillet, export |
| File Operations / 文件 | 6 | `file_ops.py` — read, write, edit, search, glob |
| GUI Automation / GUI 自动化 | 8 | `gui_action.py` — click, type, hotkey, drag, scroll |
| VLM Vision / 视觉 | 4 | `vlm.py` — analyze, verify, screen capture |
| Assembly / 装配 | 10+ | `assembly_generator.py`, `assembly_solver.py`, `assembly_doc.py` |
| CAD Features / CAD 特征 | 3+ | `part_feature_engine.py`, `connection_features.py` |
| Export / 导出 | 2+ | `export_package.py`, `urdf_export.py`, `bom_gen.py` |
| Rendering / 渲染 | 1 | `vtk_renderer.py` — offscreen VTK with feature edges |
| Utilities / 工具 | 2+ | `ik_solver.py`, `mesh_collision.py`, `code_gen.py` |

---

## Knowledge Base / 知识库

| Module / 模块 | Content / 内容 |
|---|---|
| `mechanics.py` | Parts (66 types), Joints, Assemblies, ConnectionMethod / 零件（66 种）、关节、装配、连接方式 |
| `assembly_patterns.py` | Robot profiles: BCN3D MOVEO, Thor, PAROL6, ANYmal, Solo12 / 机器人模板 |
| `fastener_catalog.py` | ISO/DIN fasteners, bolt length, torque specs / ISO/DIN 紧固件、螺栓长度、扭矩 |
| `actuators.py` | Servo/stepper database (NEMA17, Dynamixel, etc.) / 舵机/步进电机数据库 |
| `assembly_templates.py` | Few-shot examples for LLM prompt / LLM 提示的 few-shot 示例 |
| `materials.py` | PLA, ABS, PETG, aluminum properties / 材料属性 |
| `tolerance.py` | ISO 286 IT grades, fits (H7/g6, H7/js6) / ISO 286 公差等级、配合 |

---

## Project Structure / 项目结构

```
language-3d/
├── src/lang3d/
│   ├── agent/                     # Multi-Agent / 多智能体
│   │   ├── core.py                # Orchestrator + main loop / 调度器 + 主循环
│   │   ├── planner.py             # Task decomposition / 任务分解
│   │   ├── executor.py            # Step execution / 步骤执行
│   │   ├── verifier.py            # Dual-channel verification / 双通道验证
│   │   └── reflector.py           # VLM failure analysis / VLM 失败分析
│   ├── models/                    # LLM/VLM Backends / 模型后端
│   │   ├── glm.py                 # GLM backend (Coding Plan API)
│   │   └── router.py              # Task-based routing / 任务路由
│   ├── tools/                     # 57 tool modules / 57 个工具模块
│   │   ├── assembly_generator.py  # NL → assembly JSON pipeline
│   │   ├── assembly_solver.py     # Position & constraint solving / 位置约束求解
│   │   ├── part_feature_engine.py # Per-part CAD features / 逐零件 CAD 特征
│   │   ├── connection_features.py # Bolt holes, bearings, snaps / 螺栓孔、轴承、卡扣
│   │   ├── vtk_renderer.py        # VTK offscreen rendering / VTK 离屏渲染
│   │   ├── export_package.py      # Engineering package export / 工程包导出
│   │   ├── urdf_export.py         # URDF robot description / URDF 机器人描述
│   │   ├── bom_gen.py             # Bill of materials / 物料清单
│   │   ├── assembly_doc.py        # Assembly guide / 装配指南
│   │   ├── ik_solver.py           # IK (Jacobian + CCD) / 逆运动学
│   │   └── ...                    # 47 more modules / 另外 47 个模块
│   ├── knowledge/                 # Domain Knowledge / 领域知识
│   │   ├── mechanics.py           # Parts, joints, connections / 零件、关节、连接
│   │   ├── assembly_patterns.py   # 7 robot profiles / 7 个机器人模板
│   │   ├── fastener_catalog.py    # ISO/DIN specs / ISO/DIN 规格
│   │   └── ...                    # 11 more modules / 另外 11 个模块
│   ├── web/app.py                 # FastAPI dashboard / FastAPI 仪表板
│   └── config.py                  # Configuration / 配置管理
├── tests/                         # 3,880 tests / 3,880 个测试
│   ├── test_e2e_production.py     # E2E pipeline / 端到端流水线
│   ├── test_arm_pipeline_fix.py   # Arm geometry / 臂几何修复
│   ├── test_audit_fixes.py        # P0 regression tests / P0 回归测试
│   └── ...                        # 130 more files / 另外 130 个文件
├── data/e2e_results/              # E2E outputs / 端到端输出
├── examples/                      # Usage examples / 使用示例
└── pyproject.toml
```

---

## Test Results / 测试结果

| Suite / 测试套件 | Tests | Status / 状态 |
|---|---|---|
| Unit + Integration / 单元 + 集成 | 3,880 | 3,876 PASS |
| E2E Production / 端到端生产 | 4 | 4 PASS (91.4% score) |
| Known Failures / 已知失败 | 4 | IK solver convergence (pre-existing) / IK 求解器收敛（遗留） |

```bash
# Run tests / 运行测试
python -m pytest tests/ -m "not e2e" -q    # Unit tests / 单元测试
python tests/test_e2e_production.py         # E2E test / 端到端测试
```

---

## Roadmap / 路线图

### Phase 1 — Foundation / 基础 (Completed / 已完成)
- [x] FreeCAD subprocess bridge / FreeCAD 子进程桥 (23 tools)
- [x] VLM visual verification / VLM 视觉验证 (4-level routing)
- [x] GUI automation / GUI 自动化 (PyAutoGUI, 8 tools)
- [x] Dual verification pipeline / 双通道验证流水线
- [x] Agent loop: planning → execution → verification → reflection / 智能体循环

### Phase 2 — Assembly Generation / 装配体生成 (Completed / 已完成)
- [x] NL → assembly JSON (LLM + VLM feedback loop) / 自然语言 → 装配 JSON
- [x] Assembly solver / 装配求解器 (anchor constraints / 锚点约束)
- [x] Connection engine / 连接引擎 (bolted, press_fit, snap_fit, adhesive, welded, magnetic)
- [x] Part feature engine / 零件特征引擎 (C-channel links, servo housings, gripper fingers)
- [x] VTK offscreen rendering / VTK 离屏渲染 (feature edge extraction / 特征边提取)
- [x] Engineering package / 工程包 (STL, URDF, BOM, assembly guide, firmware)
- [x] Knowledge base / 知识库 (7 robot profiles, ISO/DIN fasteners / 7 个机器人模板)
- [x] IK solver / 逆运动学求解器 (Jacobian + CCD)

### Phase 3 — Simulation & Verification / 仿真与验证 (In Progress / 进行中)
- [x] URDF export / URDF 导出 (joint limits, mimic joints / 关节限位、 mimic 关节)
- [ ] URDF `<transmission>` + `ros2_control` for Gazebo actuation / URDF 传动标签
- [ ] Closed-chain kinematic solver / 闭链运动学求解器
- [ ] FEA structural analysis / 有限元结构分析 (CalculiX)
- [ ] Motion simulation / 运动仿真 + interference checking / 干涉检查
- [ ] Dual-arm collision avoidance / 双臂避障

### Phase 4 — Production System / 生产系统
- [ ] Web dashboard / Web 仪表板 (real-time monitoring / 实时监控)
- [ ] Part library expansion / 零件库扩展 (66 → 200+ types / 66 → 200+ 种)
- [ ] Manufacturing process planning / 制造工艺规划 (G-code)
- [ ] Quality control / 质量控制 (statistical verification / 统计验证)

---

## License

MIT License

---

## Acknowledgments / 致谢

- [FreeCAD](https://freecad.org) — Open-source parametric 3D CAD / 开源参数化 3D CAD
- [Zhipu AI / GLM](https://open.bigmodel.cn) — GLM-5.1 reasoning + GLM-4V vision / 智谱 AI 推理 + 视觉模型
- [PyAutoGUI](https://pyautogui.readthedocs.io) — GUI automation / GUI 自动化
- [VTK](https://vtk.org) — Offscreen 3D rendering / 离屏 3D 渲染
