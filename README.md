<div align="center">

# Language-3D Agent

**Multi-Expert Agent System for Functional-Prototype 3D Modeling**
**多专家智能体系统 — 功能原型级 3D 建模**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FreeCAD 1.1](https://img.shields.io/badge/FreeCAD-1.1-green.svg)](https://freecad.org)
[![GLM-5.2](https://img.shields.io/badge/LLM-GLM--5.2-orange.svg)](https://open.bigmodel.cn)
[![E2E Score](https://img.shields.io/badge/E2E-~95%25-brightgreen.svg)](#test-results)

*LLM Reasoning + VLM Visual Perception + CAD Automation + Geometric Arbitration*
*LLM 推理 + VLM 视觉感知 + CAD 自动化 + 几何仲裁*

</div>

---

## Overview / 概述

**English**

Language-3D Agent is an AI system that understands natural language descriptions of mechanical assemblies, generates functional-prototype 3D models (STL/STEP via boolean geometry), exports complete engineering packages (URDF, BOM, assembly guide, firmware), and verifies results through **dual-channel inspection** — code-side geometric checks and vision-side AI analysis, with **geometric arbitration** that overrules VLM false-negatives when the geometry is provably correct.

The system uses a **multi-expert agent pipeline** (Architect → Solver → CAD Engineer → Verifier → Fixer) where each stage is an independent specialist with its own system prompt and tool whitelist. The Fixer classifies VLM failures by type (structural defects like collisions vs. framing nitpicks like "arm too horizontal") and routes each failure back to the specific upstream stage that needs re-running — a solver-classified collision re-runs only solver→cad→verifier, skipping an unnecessary Architect regeneration, while a soft VLM framing complaint on a deterministic compose output is retained rather than triggering a corrupting LLM regeneration. The pipeline is the **production path** for the CLI (`lang3d`) and the web dashboard; pipeline errors propagate (fail-loud) rather than silently falling back to the legacy loop.

**中文**

Language-3D Agent 是一个 AI 系统，能够理解自然语言描述的机械装配体，生成功能原型级 3D 模型（STL/STEP，基于布尔几何），导出完整工程包（URDF、BOM、装配指南、固件），并通过**双通道验证**（代码侧几何检查 + 视觉侧 AI 分析）确保质量。当几何验证确认正确时，**几何仲裁**会推翻 VLM 的误判。

系统使用**多专家智能体流水线**（架构师 → 求解器 → CAD 工程师 → 验证器 → 修复器），每个阶段是独立的专家角色，有自己的系统提示和工具白名单。修复器把失败精准路由到需要重跑的上游阶段（例如求解器判定的碰撞只重跑 求解器→CAD→验证器，跳过不必要的架构师重生成），而非每次都从头重生。该流水线是 CLI（`lang3d`）和网页面板的**生产路径**——运行时生成的每个装配体都流经它；流水线抛错时会**显式报错**（不再静默回退旧循环，旧循环仅供 `LANG3D_LEGACY_FALLBACK=1` 诊断用）。

```
"设计一个 4 自由度机械臂，带夹爪"     ← Natural Language / 自然语言
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Architect Agent       Verifier Agent
   (GLM-5.2)             (GLM-4.6V)
         │                     │
    Solver → CAD          Geometric Arbitration
         │              + VLM Split Verification
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Engineering Package │
         │  工程包输出           │
         │  ├ STL parts (13)    │
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
# Core install (LLM + CAD + web dashboard) / 核心安装
pip install -e .

# With collision detection + physics simulation / 含碰撞检测 + 物理仿真
pip install -e ".[collision,sim]"

# With all dev dependencies / 含开发依赖
pip install -e ".[collision,sim,dev]"

# Configure / 配置
cp .env.example .env
# Edit .env: add GLM_API_KEY / 编辑 .env 添加密钥
```

### Usage / 使用

```bash
# Interactive CLI / 交互式命令行（生产路径，走 AssemblyPipeline）
lang3d

# E2E test harness / 端到端测试
python tests/test_e2e_production.py --case 4dof_arm           # 默认：传统循环
python tests/test_e2e_production.py --case 4dof_arm --pipeline  # 多专家流水线

# Local web dashboard + 3D viewer (auto-opens browser) / 本地网页面板 + 3D 查看器（自动开浏览器）
lang3d web                       # http://127.0.0.1:8765/simulate
lang3d web 0.0.0.0 9000          # custom host/port / 自定义主机与端口

# MuJoCo physics GUI for a run's URDF / 打开某次运行的 MuJoCo 物理仿真窗口
lang3d sim data/runs/4dof_arm/20260624_172515
```

> **Web viewer / 网页查看器**: `lang3d web` serves a Three.js viewer at
> `/simulate` that loads any run's STLs at their solved positions and plays
> back a MuJoCo physics animation (each joint articulates via a PD-tracked
> sinusoidal sweep). Use the **Run Simulation** button to record a clip and
> the timeline slider to scrub frames.
> `lang3d web` 启动的 `/simulate` 页面会按求解位置加载任意运行的 STL，并可播放
> MuJoCo 物理动画（各关节经 PD 跟踪的正弦扫描协调运动）。点 **Run Simulation**
> 录制片段，用时间轴滑块拖动定位任意帧。

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
│  │ GLM-5.2    │ │ Positions  │ │ STL 生成  │                 │
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
| **Architect / 架构师** | GLM-5.2 | NL → assembly JSON, joint chain, part selection | assembly_template, part_recommend |
| **Solver / 求解器** | Deterministic | assembly.json → 3D positions, collision resolution | assembly_solver, mesh_collision |
| **CAD Engineer / CAD 工程师** | GLM-5.2 | positions → water-tight STLs | fc_batch, fc_script, cad_verify |
| **Verifier / 验证器** | GLM-4.6V | Dual-channel: geometric checks + VLM visual | vlm, _geometric_prevalidation |
| **Fixer / 修复器** | Deterministic | Classify failure → route to correct upstream stage | _classify_problems |

### Assembly Generation Pipeline / 装配体生成流水线

| Stage / 阶段 | Module / 模块 | Description / 描述 |
|---|---|---|
| 1. Architect / 架构 | `assembly_generator.py` + `pipeline.py` | GLM-5.2 generates assembly JSON from prompt |
| 2. Sanitize / 清洗 | `assembly_gen/sanitizers.py` | Fix anchors, grippers, proportions, angles, base sizing, **numeric pitch/yaw caps** (topology-aware: humanoid shoulder 60°, fixed-base 90°, wrist-roll ±120°) + FCL endpoint validation |
| 3. Solve / 求解 | `assembly_solver.py` + `assembly_compose.py` | Compute 3D positions, then **collision-aware range clamp** (FCL sweep narrows range_deg to collision-free subset for all arms) |
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
| **MuJoCo Physics / 物理仿真** | PD-hold stability + joint actuation + ground-plane driving (floating-base differential drive) + grasp | Must pass for URDF export |

### Self-Evolving Experience Store / 自演化经验库

A retrieve-before / store-after memory of verified-good assemblies (`experience/store.py`). Before generation, the Architect retrieves similar past cases (lexical scoring: keyword 3×, robot-category 5×, DOF proximity 2×); after verification passes, the case is stored for future retrieval. **Successes only** (failed assemblies never stored); per-category cap with popularity-based pruning. The store starts empty on a fresh checkout (generation falls back to parametric examples — no behaviour change until cases accumulate).

---

## Tool System / 工具系统

55 tool modules organized into 9 categories. Each expert agent sees only its whitelisted tools (`ROLE_TOOL_CATEGORIES`):

> **Note:** Tool whitelist enforcement is active in the `OrchestratorAgent` path. The production `AssemblyPipeline` path calls domain functions directly (not through the tool registry), so the whitelist is advisory there.

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
| `mechanics.py` | Parts (94 types), Joints, Assemblies, ConnectionMethod |
| `assembly_patterns.py` | Robot profiles: BCN3D MOVEO, Thor, PAROL6, ANYmal, Solo12 |
| `fastener_catalog.py` | ISO/DIN fasteners, bolt length, torque specs |
| `actuators.py` | Servo/stepper database (NEMA17, Dynamixel, etc.) |
| `assembly_templates.py` | Few-shot examples for LLM prompt |
| `materials.py` | PLA, ABS, PETG, aluminum properties |
| `tolerance.py` | ISO 286 IT grades, fits (H7/g6, H7/js6) |
| `mobile_base_gen.py` | Parametric wheeled chassis (Husky-style Z-stack, ground-contact wheels) |
| `arm_topology.py` | Parametric arm topology generator (2-7 DOF, zig-zag pose) |
| `assembly_compose.py` | Deterministic dual-arm SE(3) composition (ArtiCAD Derive Mechanism) |
| `docs/references/` | Real-robot proportion data (HSR, TIAGo++) + COTS specs for validation |

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
│   │   ├── planner.py             # Task decomposition (flat/hierarchical)
│   │   ├── executor.py            # Step execution + auto-fix loop
│   │   ├── verifier.py            # Dual-channel verification
│   │   ├── modifier.py            # Targeted modification engine
│   │   ├── fix_strategy.py        # Failure classification + convergence detection
│   │   ├── dag.py                 # Task DAG with cycle detection
│   │   ├── message_bus.py         # Inter-agent pub/sub messaging
│   │   └── ...
│   ├── models/                    # LLM/VLM Backends
│   │   ├── glm.py                 # GLM-5.2 (text) + GLM-4.6V (vision)
│   │   └── router.py              # 4-level vision model tiers (fast/standard/detailed/maximum; default: detailed)
│   ├── tools/                     # 55 tool modules
│   │   ├── assembly_generator.py  # NL → assembly JSON + VLM loop (main loop)
│   │   ├── assembly_gen/          # Extracted sub-modules (P1-1 split)
│   │   │   ├── sanitizers.py      # Post-generation correction (anchors, angles, limits)
│   │   │   └── vlm_verify.py      # VLM verification + false-alarm filters + geometric pre-validation
│   │   ├── freecad_script_builder.py  # FreeCAD operation → Python script generation
│   │   ├── sim_grasp.py           # Three-phase grasp simulation (zero-G → gravity → lift)
│   │   ├── assembly_solver.py     # Position & constraint solving
│   │   ├── part_feature_engine.py # Per-part CAD features (axis-correct bolt holes)
│   │   ├── connection_features.py # Bolt holes (anchor-rotated), bearings, snaps
│   │   ├── vtk_renderer.py        # VTK rendering (crop-to-content, clipping fix)
│   │   ├── export_package.py      # Engineering package export
│   │   ├── urdf_export.py         # URDF (inertial frame-correct)
│   │   ├── sim_mujoco.py          # MuJoCo physics (prismatic joint clamping)
│   │   └── ...
│   ├── experience/                # Self-evolving experience store (audit H2)
│   │   └── store.py               # Retrieve-before / store-after (lexical, successes-only)
│   ├── knowledge/                 # Domain Knowledge
│   │   ├── parts_catalog.py       # COTS part templates + lookup functions
│   │   ├── _catalog_entries.py    # PART_CATALOG dict entries (94 templates)
│   │   └── ...
│   ├── web/
│   │   ├── app.py                 # FastAPI dashboard (core routes + WebSocket)
│   │   └── routes/                # API route modules (P1-1 split)
│   │       ├── convert.py         # STEP/FCStd conversion
│   │       ├── parts.py           # Part catalog, generate, analyze
│   │       ├── slicing.py         # G-code slicing
│   │       └── design.py          # Design hierarchy, stability, power
│   └── config.py                  # Configuration
├── tests/                         # 2,300+ tests
│   ├── test_e2e_production.py     # E2E pipeline (4dof_arm + 4wheel_dual_arm)
│   ├── test_expert_roles.py       # Expert agent role + tool whitelist tests
│   └── ...
├── data/runs/                     # E2E outputs (canonical layout)
├── docs/references/               # Real-robot reference data (proportions, COTS specs)
├── examples/                      # Usage examples
├── .github/workflows/ci.yml       # CI: unit + integration tests on push/PR
└── pyproject.toml
```

---

## Test Results / 测试结果

| Suite / 测试套件 | Tests | Status / 状态 |
|---|---|---|
| Unit + Integration / 单元 + 集成 | 2,360+ | 5 pre-existing failures (wheel-rotation/gripper/message-bus/geometric-arbitration, registered in AGENTS.md §6.3) |
| E2E: 4dof_arm (pipeline) / 机械臂 | 1 | **95.1%** (0 SKIP; MuJoCo + grasp + motion-collision PASS, range clamp active) |
| E2E: 4wheel_dual_arm (pipeline) / 轮式双臂 | 1 | **95.3%** (0 SKIP; drives + turns, deterministic compose, motion-collision-free) |
| Expert Roles / 专家角色 | 27 | 27 PASS |

> **Note / 注意**: E2E tests require `GLM_API_KEY` + FreeCAD — without them they are skipped. Scores are **per-case** results (not a multi-case average). Scoring = PASS/(PASS+FAIL+WARN); SKIP (missing optional dep) is excluded from the denominator, and critical checks (collision, COM stability, MuJoCo physics, grasp) FAIL rather than downgrade to warning. The 4dof_arm score varies slightly run-to-run due to LLM non-determinism; the wheeled dual-arm is deterministic (compose path).
> E2E 测试需要 `GLM_API_KEY` + FreeCAD——没有时跳过。分数是**各 case 单独**的结果（非多 case 平均）。评分公式 PASS/(PASS+FAIL+WARN)，SKIP 不计分母，critical 检查失败即 FAIL 不降级。4dof_arm 因 LLM 非确定性分数会小幅波动；轮式双臂走确定性 compose 路径，稳定。

```bash
# Run tests / 运行测试
python -m pytest tests/ -m "not e2e" -q    # Unit + Integration / 单元 + 集成
python tests/test_e2e_production.py --case 4dof_arm           # 机械臂 e2e
python tests/test_e2e_production.py --case 4wheel_dual_arm    # 轮式双臂 e2e
```

### E2E 4dof_arm Score Breakdown / 端到端评分明细

| Phase / 阶段 | Checks | Status |
|---|---|---|
| Phase 1: NL → Assembly | 5 | ✅ 13 parts, 12 joints, connected tree, range clamp active |
| Phase 2: Position Solving | 4 | ✅ 13/13 positioned, 0 NaN |
| Phase 3: Render Quality | 2 | ✅ 4 views, 560KB avg |
| Phase 4: Engineering Package | 10 | ✅ All files exported |
| Phase 5: Content Validation | 9 | ✅ **VLM verification: PASSED** |
| Phase 6: Physical Sanity | 5 | ✅ COM stable, **0 collisions (motion sweep collision-free)**, 535mm workspace |
| Phase 7: MuJoCo Simulation | 4 | ✅ Physics stable, 6 joints actuated, grasp PASS |

---

## Roadmap / 路线图

### Phase 1 — Foundation / 基础 (Completed)
- [x] FreeCAD subprocess bridge (23 tools)
- [x] VLM visual verification (4-level model tiers, default: detailed)
- [x] GUI automation (PyAutoGUI, 8 tools)
- [x] Dual verification pipeline

### Phase 2 — Assembly Generation / 装配体生成 (Completed)
- [x] NL → assembly JSON (GLM-5.2 + VLM feedback loop)
- [x] Assembly solver (anchor constraints)
- [x] Connection engine (bolted, press_fit, snap_fit, adhesive, welded, magnetic)
- [x] Part feature engine (axis-correct bolt holes, water-tight fingers)
- [x] VTK offscreen rendering (crop-to-content, clipping fix)
- [x] Engineering package (STL, URDF, BOM, assembly guide, firmware)

### Phase 3 — Multi-Agent Architecture / 多智能体架构 (Completed)
- [x] Expert agent roles (Architect/Solver/CAD/Verifier/Fixer/Chassis)
- [x] Tool whitelisting per role (ROLE_TOOL_CATEGORIES)
- [x] AssemblyPipeline is the production path (pipeline errors propagate; legacy loop is opt-in via LANG3D_LEGACY_FALLBACK)
- [x] Architect persona injected on round 1 (not only on repair rounds)
- [x] Selective Fixer routing by failure type (structural defects trigger fix; framing complaints on deterministic output are retained, not regenerated)
- [x] Deterministic dual-arm generation (chassis expert tools, no LLM for topology)
- [x] Geometric arbitration (Check 7 connectivity + false-alarm filtering)
- [x] Split VLM verification (structural panoramic + gripper close-up)
- [x] MuJoCo physics validation (prismatic joint clamping)

### Phase 4 — Simulation & Verification / 仿真与验证 (In Progress)
- [x] URDF export (joint limits, mimic joints, inertial frame-correct)
- [x] MuJoCo PD-hold physics stability test
- [x] Husky-style chassis Z-stack (base_footprint ground reference, wheels rest on ground)
- [x] Wheeled base drives in MuJoCo (floating-base + differential wheel torque; **ground-plane injection** so wheel-ground contact activates — MuJoCo URDF loader creates no floor by default)
- [x] Assembly STL with fastener bodies (trimesh fallback bypasses FreeCAD stack limit)
- [x] Collision-aware bolt-hole alignment across mating parts (shared ConnectionInterface)
- [x] Wheeled dual-arm proportion coupling (mobile profile + chassis-matched arm scale)
- [x] Real COTS servos wired into arm generator (MG996R/DS3218/SG90 sourced from parts_catalog)
- [x] **Kinematic safety**: numeric pitch/yaw caps (topology-aware: humanoid shoulder 60°, fixed-base 90°, wrist-roll ±120°) + FCL endpoint validation — arm cannot sweep through its own base during motion
- [x] **Collision-aware range clamp for all arms** (generalised from dual-arm-only; FCL sweep narrows range_deg in run_solver before URDF export)
- [x] **Contact-setup deduplication** (centralised `_setup_wheel_contacts` replaces 3 inline copies; dynamic z-drop from real wheel-bottom height)
- [x] URDF `<ros2_control>` tag for Gazebo actuation (GazeboSystem hardware plugin + ros2_control.yaml; validated: check_urdf + colcon build + robot_state_publisher + gzserver spawn)
- [ ] Closed-chain kinematic solver
- [ ] FEA structural analysis (CalculiX)
- [x] Grasp simulation (sim_grasp three-phase, dual-gripper support)
- [x] Dual-arm collision avoidance (static collision-aware pose configurator + workspace-safe joint limits; NOT real-time motion planning)

### Phase 5 — Production System / 生产系统
- [ ] Web dashboard (real-time monitoring)
- [ ] Part library expansion (94 → 200+ types)
- [ ] Manufacturing process planning (G-code)
- [ ] Quality control (statistical verification)

---

## License

MIT License

---

## Acknowledgments / 致谢

- [FreeCAD](https://freecad.org) — Open-source parametric 3D CAD
- [Zhipu AI / GLM](https://open.bigmodel.cn) — GLM-5.2 reasoning + GLM-4.6V vision
- [MuJoCo](https://mujoco.org) — Physics simulation
- [VTK](https://vtk.org) — Offscreen 3D rendering
- [trimesh](https://trimesh.org) — Mesh processing + python-fcl collision
- [PyAutoGUI](https://pyautogui.readthedocs.io) — GUI automation
