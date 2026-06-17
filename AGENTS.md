# AGENTS.md — Language-3D 全局协作规则

> 本文件是**所有 AI 编码 agent（ZCode / Claude Code / 其他）在本仓库工作时的强制规则**。
> Claude Code 请将 `CLAUDE.md` 软链或拷贝指向本文件，保持单一事实源。
> 人类开发者同样适用。规则有冲突时，**以本文件为准**。

---

## 0. 项目是什么

Language-3D Agent：自然语言 → 生产级机器人装配体（STL/STEP/URDF/BOM/固件），通过 LLM 推理 + VLM 视觉 + CAD 自动化 + 双通道验证交付。**产物必须能动、能抓、能在仿真里跑**。

- 代码语言：Python 3.12
- CAD 后端：FreeCAD 1.1（子进程桥）
- LLM/VLM：GLM 系列（智谱）
- 单位：**mm（毫米）**，全局统一，不得混用 m/cm
- 工作目录：`src/lang3d/`（源码）、`tests/`（测试）、`data/`（产物）、`docs/`（文档）、`logs/`（运行日志）

开始任何任务前，先读：`README.md`、`项目期望.txt`、`docs/`、`tests/conftest.py`。

---

## 1. 做正确的事，而不是做简单的事

**核心：宁可多花时间做对，也不要用看似省事的捷径制造技术债。**

### 1.1 禁止的"简单做法"
- ❌ **用 `try/except: pass` 吞掉错误**。异常必须要么处理（记日志 + 降级），要么向上抛。静默吞异常 = 埋雷。
- ❌ **为了让测试过而改测试的断言**。测试失败先查被测代码，确认是测试本身错了才能改测试，并在 commit message 里说明理由。
- ❌ **注释掉失败的测试或加 `@pytest.mark.skip` 绕过**。要 skip 必须写明原因（`reason="..."`），且登记到本规则的"已知失败"清单（见 §6）。
- ❌ **用 mock/stub 让端到端测试"看起来通过"**。`e2e` 标记的测试必须是真跑：真 FreeCAD、真 VLM、真 MuJoCo。`progress.txt` 里已经记录过"task 标 passes:true 但 steps 是 [缺测]"的教训。
- ❌ **复制粘贴重复代码**。超过 3 处相同逻辑必须抽函数。
- ❌ **在根目录写一次性脚本**（`check_xxx.py`、`debug_xxx.py`、`true_e2e.py`）。脚本进 `scripts/`。

### 1.2 鼓励的"正确做法"
- ✅ 修 bug 先**复现**（写失败测试），再修，再确认测试转绿。
- ✅ 重构前先**保证有测试覆盖**，没有测试先补测试。
- ✅ 拿不准时**问人**（用户/规则），不要凭直觉改关键逻辑。
- ✅ 涉及物理/运动学/公差的数值，查 `src/lang3d/knowledge/`（`fastener_catalog.py`、`tolerance.py`、`materials.py`）的真实规格，不要编。

---

## 2. 代码干净、结构规范、测试及时、验证可靠

### 2.1 代码风格
- 每个模块顶部有 docstring 说明职责（看现有模块，如 `core.py`、`assembly_solver.py` 的风格）。
- 函数有类型注解（`from __future__ import annotations` 已是项目惯例）。
- 公共 API 用 dataclass / TypedDict / pydantic 建模，不要散装 dict 传来传去。
- 单文件控制在 **500 行以内**，超了就拆。
- 命名：`snake_case` 函数/变量，`PascalCase` 类，`UPPER_SNAKE` 常量。工具类以 `Tool` 结尾。

### 2.2 目录职责（严格）
| 目录 | 放什么 | **不放什么** |
|---|---|---|
| `src/lang3d/agent/` | 多智能体调度、规划、执行、验证、反思 | 具体工具实现 |
| `src/lang3d/tools/` | 57+ 个 `*Tool(Tool)` 类，每个一个文件 | 业务编排逻辑 |
| `src/lang3d/knowledge/` | 领域知识（零件、紧固件、材料、模板）纯数据/规则 | 副作用、IO |
| `src/lang3d/models/` | LLM/VLM 后端 + 路由 + 重试 + 缓存 | 业务逻辑 |
| `tests/` | 测试，且仅测试 | 临时脚本、demo |
| `scripts/` | 一次性运维/数据脚本 | 被代码 import 的模块 |
| `docs/` | 设计文档、研究笔记、历史日志 | 运行产物 |
| `data/` | 运行产物（输出工件） | 手写源码 |
| `logs/` | 运行日志（`.log`） | 工件、源码 |

### 2.3 测试规范
- **测试 marker 必须准确**。`tests/conftest.py` 会按文件内容自动分类（unit/integration/e2e/api/gui），但写测试时要符合分类语义：
  - `unit`：纯 Python，无外部依赖，秒级
  - `integration`：会起 FreeCAD / trimesh+fcl / MuJoCo
  - `e2e`：NL → assembly → export 全链路，慢，通常要 GLM_API_KEY
  - `api`：调远程 LLM，无 key 自动 skip
  - `gui`：要桌面/显示，无头 CI 跳过
- **命名**：`test_<被测模块>.py`，一个模块对应一个测试文件。
- **工整**：用 fixture 复用（`tmp_workspace`、`mock_router` 已在 conftest），不要每个测试自己造数据。
- **及时**：新功能/bugfix 同一个 PR 带测试。**没有测试的代码不算完成**。
- **验证可靠**：断言要具体（`assert result.volume == pytest.approx(125000, rel=0.05)`），不要 `assert result is not None`。

### 2.4 运行测试
```bash
pytest -m unit -q                    # 快速本地验证（秒级）
pytest -m "unit or integration" -q   # 本机能跑的全集
pytest -m "not e2e and not api" -q    # CI 默认（不烧 API、不跑慢 E2E）
pytest -m e2e                         # 完整端到端（需要 FreeCAD + key）
```

---

## 3. 先思考、研究、摸清关联，再动手

**这是最重要的一条。本项目的关联很深：装配 → 求解 → CAD → 渲染 → VLM → 导出，一环动错全链崩。**

### 3.1 改任何代码前的强制流程
1. **定位**：用 Grep/Read 找到所有调用点。改一个函数前，先搜谁在调它。
   ```bash
   # 例：要改 assembly_solver 的输出格式
   grep -rn "assembly_solve\|AssemblySolveTool" src/ tests/
   ```
2. **理解数据流**：装配 JSON 的 schema 被多少工具消费？（generator → solver → cad → urdf → export，至少 5 处）。改 schema 必须全链同步。
3. **看测试**：被改模块对应的 `tests/test_xxx.py` 是否覆盖了你要动的行为？没覆盖先补。
4. **看文档**：`README.md` 和 `docs/` 里有没有相关约定。
5. **再动手**：小步改，每步可验证。

### 3.2 谨慎对待的部分（动之前必须读源码）
- **装配求解器** `assembly_solver.py`（闭环约束、Newton-Raphson）—— 改了可能导致所有装配位置错乱。
- **URDF 导出** `urdf_export.py` + `sim_mujoco.py` —— 旋转矩阵、关节轴、mesh 路径，历史上有过"相对路径→绝对路径"的修复，别回退。
- **VLM 循环** `assembly_vlm.py` —— 最多 3 轮 fix，别改成死循环。
- **夹爪/抓取** `sim_grasp` 三阶段（零重力 → 重力 → 抬升）—— 这是"能抓东西"的硬指标，别简化。
- **公差/紧固件** `knowledge/fastener_catalog.py`、`tolerance.py` —— ISO/DIN 真实规格，查表不编。

### 3.3 改动规模决策
- **小修**（< 20 行，单文件，无 API 变更）：直接改 + 跑相关单测。
- **中改**（多文件 / 改接口）：先在对话里说方案，列受影响文件，用户点头再改。
- **大改**（架构级 / 改数据流 / 改 schema）：**必须先写设计说明**（哪怕几行），用 plan mode 确认。

---

## 4. README 及时更新

**触发 README 更新的"实质性变更"（必须同步改 README）：**
- ✅ 新增 / 删除 / 重命名工具模块（影响"Tool System"表的分类与数量）
- ✅ 流水线阶段变化（"Assembly Generation Pipeline"表）
- ✅ 验证通道变化（"Verification Channels"表）
- ✅ 架构图变化（多智能体循环、目录结构）
- ✅ roadmap 里程碑状态翻转（`[ ]` ↔ `[x]`）
- ✅ 安装/使用方式变化（CLI 命令、依赖）

**不需要每次同步的（避免 README 抖动）：**
- 测试数量（3880 这种数字）—— 每月或里程碑时刷新一次即可
- 模块精确数量（"57 tool modules"）—— 同上
- 内部实现细节 —— 属于代码注释和 `docs/`，不属于 README

README 是**中英双语**，更新时**中英都要改**，保持对齐。改完检查两侧表格行数一致。

---

## 5. 该做视觉验证时，必须做视觉验证

**项目核心卖点就是"双通道验证"——代码侧 + 视觉侧。绕过视觉验证 = 自欺欺人。**

### 5.1 强制分级验证矩阵

| 产物类型 | 代码侧验证 | 视觉/物理验证 | 不做不许标完成 |
|---|---|---|---|
| Level 1-2 简单零件（方块/圆柱/平板） | `fc_batch` 的 `volume_check`（尺寸/体积） | 可省 | ✅ 必须 |
| Level 3+ 复杂零件（带孔/倒角/多特征） | `mesh_stats`（水密性/体积） | `cad_verify`（多角度 VLM，MATCH=True） | ✅ 必须 VLM |
| 装配体 | `mesh_collision_check`（trimesh+FCL 干涉） | `cad_verify` 或 `assembly_vlm_solve`（结构合理性） | ✅ 必须 VLM |
| 带夹爪的装配体 | 同上 | + `sim_grasp`（三阶段抓取） | ✅ 必须 sim_grasp |
| 导出 URDF 的装配体 | — | + `sim_mujoco`（加载 + 关节能动 + 物理稳定） | ✅ 必须 sim_mujoco |

### 5.2 视觉验证的执行规则
- `cad_verify` 默认 `angles=isometric,front,top`，复杂件用 `detail="detailed"`。
- VLM 返回必须检查 `MATCH` 字段：`true` 才通过，`false` 必须按 `FIX_COMMANDS` 修正后重验。
- **VLM 三轮仍未通过**：停下来报告，不要无限重试。记录失败 case 到 `docs/` 供后续分析。
- `verify_result=UNKNOWN` **不得视为成功**（历史教训：`sys.exit(0)` 把全失败也当成功）。
- **不得用 mock 的 VLM 返回值冒充通过**。`api`/`e2e` 标记的测试必须有真 GLM key 才跑，没 key 就 skip，不要造假。

### 5.3 什么时候"该做"
- 任何**改变了零件几何外观**的操作（建模、布尔、倒角、抽壳、阵列）
- 任何**改变了装配体相对位置/朝向**的操作（求解、镜像、阵列）
- 任何**导出供下游消费**的产物（STL/URDF 给仿真用前）
- 不确定时就做——做视觉验证的成本远低于"以为对了其实错了"的返工成本。

---

## 6. 产物与日志归置（保持仓库干净）

**历史教训：根目录曾堆了 12 个临时日志 + `MUJOCO_LOG.TXT` + `nul` + `temp_doc.xml`，全是漏网产物。**

### 6.1 强制路径
| 类型 | 去向 |
|---|---|
| 运行日志（`.log`、`MUJOCO_LOG.TXT`） | `logs/<feature>_<timestamp>.log` |
| E2E 输出工件（STL/URDF/报告） | `data/runs/<case>_<timestamp>/` |
| 截图（VLM/render） | `data/screenshots/`（已 gitignore） |
| 临时调试脚本 | `scripts/` 或直接删，**不得放根目录** |
| 开发日志/历史 | `docs/history/`（`progress.txt` 已迁此） |

### 6.2 根目录只允许存在
`README.md`、`AGENTS.md`、`CLAUDE.md`、`pyproject.toml`、`.env(.example)`、`.gitignore`、`项目期望.txt`、以及标准目录（`src/ tests/ docs/ scripts/ examples/ data/ logs/`）。

**任何其他文件出现在根目录 = 违规。** 发现了顺手清掉或归档。

### 6.3 已知失败清单（skip 必须登记于此）
- IK solver 收敛（遗留，README 已记录）

新增 skip 的测试，在此补一行：`- <test_name>: <原因> (since YYYY-MM-DD)`

---

## 7. Git 与提交

- **不主动 commit/push**，除非用户明确要求。
- commit message 用项目历史风格：`feat:` / `fix:` / `refactor:` / `test:` / `docs:` / `checkpoint:` 开头，简述做什么。
- 一个 commit 一件事，别把无关改动塞一起。
- **永远不要 `git push --force`** 到主干。

---

## 8. 工作风格

- 改动前先汇报"我打算动这几个文件、为什么"，别闷头改一堆再回来解释。
- 遇到不确定的物理/工程常识（公差配合、电机扭矩、材料强度），查 `knowledge/` 或问用户，**不要编**。
- 报告结果要诚实：测试没跑就说没跑，验证跳过了就说跳过了，不要用"应该没问题"糊弄。
- 用户说"之前用 Claude Code 做的，有一堆问题"——意味着对现状要**批判性看待**，发现遗留问题主动指出，不要假设老代码都对。
