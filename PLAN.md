# 计划：重写视觉验证 — 从截屏改为 VTK 离屏渲染

## 问题诊断

### 当前截图方案的 8 个致命缺陷

| # | 缺陷 | 严重性 | 影响 |
|---|------|--------|------|
| 1 | 最小化窗口时 GetWindowRect 返回 (-32000,-32000)，截图区域为零 | **致命** | 直接失败 |
| 2 | 窗口被遮挡时 ImageGrab.grab 截到其他窗口内容 | **致命** | 静默错误结果 |
| 3 | 多显示器不支持，GetSystemMetrics 只返回主显示器尺寸 | **致命** | 截图区域被裁剪 |
| 4 | DPI 缩放导致 GetWindowRect 坐标与 ImageGrab 像素不对齐 | **高** | 截图偏移 |
| 5 | SetForegroundWindow 静默失败（无前台权限） | **高** | 截到旧画面 |
| 6 | 多角度切换用 headless subprocess 试图控制 running GUI，架构矛盾 | **致命** | 多角度根本不工作 |
| 7 | 固定 time.sleep(0.5/1.0/5.0) 等待，不适应系统负载 | **高** | 截到渲染中间帧 |
| 8 | taskkill /F /IM FreeCAD.exe 杀死所有 FreeCAD 进程 | **高** | 影响用户其他工作 |

### 装配体 matplotlib 渲染的质量问题

- 零件是 12 面圆柱和方块的近似体
- 没有光照、没有阴影、没有纹理
- matplotlib 3D 有 z-ordering bug（远处的面会挡住近处）
- 固定坐标范围 [-200,200]x[-150,150]x[-80,280]，大零件被裁剪

## 解决方案：VTK 离屏渲染

### 为什么选 VTK

| 对比项 | 截屏 | matplotlib | **VTK** |
|--------|------|-----------|---------|
| 已安装 | N/A | ✅ | ✅ (9.3.1) |
| 无需显示器 | ❌ | ✅ | ✅ (SetOffScreenRendering) |
| 无需窗口管理 | ❌ | ✅ | ✅ |
| 无 timing 问题 | ❌ | ✅ | ✅ |
| 加载真实 STL | ❌ | ❌ | ✅ (vtkSTLReader) |
| Phong 光照 | N/A | ❌ | ✅ |
| 抗锯齿 | N/A | ❌ | ✅ (SetMultiSamples) |
| 多角度批量渲染 | ❌ (每次重开GUI) | ✅ | ✅ |
| 已在本机验证 | ❌ | ✅ | ✅ (已测试通过) |

### 架构设计

```
                         ┌──────────────────┐
                         │  VLM Verification │
                         │  (发送 PNG 给 VLM) │
                         └────────┬─────────┘
                                  │ PNG files
                         ┌────────▼─────────┐
                         │  VTK Offscreen    │
                         │  Renderer         │
                         │  (新建模块)        │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │                           │
           ┌────────▼────────┐        ┌────────▼────────┐
           │ 单零件验证       │        │ 装配体验证       │
           │ CADVerifyTool   │        │ AssemblyVisual  │
           │                 │        │ Verifier        │
           │ 加载单个 STL    │        │ 加载多个 STL     │
           │ 4 角度渲染      │        │ + 位置+着色      │
           └─────────────────┘        │ 4 角度渲染      │
                                       └────────────────┘
```

## 实现步骤

### Step 1: 新建 VTK 渲染模块

**文件**: `src/lang3d/tools/vtk_renderer.py` (新建)

核心类 `VTKOffscreenRenderer`:

```python
class VTKOffscreenRenderer:
    """VTK 离屏渲染器 — 无需显示器、无需窗口、无 timing 问题"""

    # 4 个标准视角预设
    VIEW_PRESETS = {
        "isometric": {"position": (130, 130, 100), "focal": (0, 0, 50), "up": (0, 0, 1)},
        "front":     {"position": (0, -180, 50),   "focal": (0, 0, 50), "up": (0, 0, 1)},
        "top":       {"position": (0, 1, 200),     "focal": (0, 0, 0),  "up": (0, 1, 0)},
        "right":     {"position": (180, 0, 50),    "focal": (0, 0, 50), "up": (0, 0, 1)},
    }

    def __init__(self, width=1200, height=900):
        self.width = width
        self.height = height
        self.actors: list[vtk.vtkActor] = []

    def load_stl(self, stl_path: str, color=(0.7, 0.7, 0.7), opacity=1.0):
        """加载 STL 文件并添加到场景"""

    def add_box(self, dimensions, position, color, opacity=1.0):
        """从尺寸数据添加近似方块"""

    def add_cylinder(self, dimensions, position, color, opacity=1.0):
        """从尺寸数据添加近似圆柱"""

    def add_axes(self, length=30):
        """添加 XYZ 坐标轴指示器"""

    def add_floor_grid(self, size=400, spacing=50):
        """添加地面网格"""

    def render_to_file(self, view_preset: str, output_path: str):
        """渲染指定视角到 PNG 文件"""

    def render_all_views(self, output_dir: str, views=None) -> list[str]:
        """批量渲染多个视角，返回 PNG 文件路径列表"""
```

关键特性:
- `SetOffScreenRendering(1)` — 不需要显示器
- `vtkPolyDataNormals` — 平滑着色
- `vtkSTLReader` — 加载真实 STL 几何
- 批量渲染 4 个视角，每次只换相机位置，不重新加载 mesh
- 自动计算 viewAll（所有零件完整可见）

### Step 2: 重写 CADVerifyTool（单零件验证）

**文件**: `src/lang3d/tools/vlm.py` — 重写 `CADVerifyTool`

当前流程（不可靠）:
```
fc_open_gui → SetForegroundWindow → sleep(0.5) → ImageGrab.grab → VLM
```

新流程（可靠）:
```
fc_batch 导出 STL → VTK 离屏渲染 4 角度 PNG → VLM 分析 PNG
```

具体改动:
1. 删除所有 `ctypes.windll.user32` 调用
2. 删除 `SetForegroundWindow` + `time.sleep` + `ImageGrab.grab`
3. 删除 `_find_windows_by_title` 依赖
4. 新增: 在 execute() 中先确保 STL 存在（调用 fc_batch export_stl 如果需要）
5. 新增: 用 `VTKOffscreenRenderer.load_stl()` 加载 STL
6. 新增: `render_all_views()` 生成 4 个角度的 PNG
7. 新增: 将 4 个 PNG 发送给 VLM，用置信度投票聚合
8. 保留: VLM prompt 结构和 JSON 解析逻辑
9. 保留: 多角度投票聚合逻辑

不再需要:
- `fc_open_gui` / `fc_close_gui`（验证不再需要 GUI）
- `FCSetCameraTool`（VTK 直接设相机）
- 所有 Windows 特定 API

### Step 3: 重写装配体渲染

**文件**: `src/lang3d/agent/assembly_visual_verifier.py` — 重写 `_render_to_dir()`

两种模式:
- **STL 模式**（首选）: 如果 STL 文件已导出，加载真实 mesh
- **Dimension 模式**（降级）: 如果没有 STL，用 `add_box()`/`add_cylinder()` 从尺寸近似

子系统着色（保留现有颜色方案）:
```python
SUBSYSTEM_COLORS = {
    "chassis":       (0.20, 0.40, 0.80),  # 蓝色
    "arm_left":      (0.85, 0.30, 0.20),  # 红色
    "arm_right":     (0.20, 0.75, 0.30),  # 绿色
    "ipc":           (0.75, 0.55, 0.10),  # 金色
    "sensor_tower":  (0.60, 0.20, 0.75),  # 紫色
}
```

降级策略:
```python
def _render_to_dir(assembly, positions, output_dir):
    try:
        import vtk
        return _render_vtk(assembly, positions, output_dir)
    except ImportError:
        return _render_matplotlib(assembly, positions, output_dir)  # 保留旧代码
```

### Step 4: 更新 fc_batch — 新增 render 操作

**文件**: `src/lang3d/tools/freecad.py` — `_build_script()` 和 `_build_batch_script()`

新增操作类型 `render`:
```python
elif op_type == "render":
    # 在 FreeCAD 脚本中用 Coin3D SoOffscreenRenderer 渲染
    # 这是备选方案，主要方案是 VTK
    width = op.get("width", 1200)
    height = op.get("height", 900)
    views = op.get("views", ["isometric", "front", "top", "right"])
    output_dir = op["output_dir"]
    # 生成 Coin3D 渲染代码...
```

这样 fc_batch 可以在建模后立即渲染，无需单独开 GUI。

### Step 5: 重写测试

**删除** (或标记 deprecated):
- 所有依赖 `SetForegroundWindow` / `ImageGrab.grab` 的测试
- `test_vlm_locate_e2e.py` 中截屏相关的部分

**新建** `tests/test_vtk_renderer.py` (~30 个测试):
- STL 加载和渲染（不需要 FreeCAD）
- 多角度批量渲染
- 尺寸模式渲染（box/cylinder 近似）
- 子系统着色
- 视角预设正确性
- 自动 viewAll（所有零件可见）
- 降级到 matplotlib（VTK 不可用时）
- 输出文件格式和尺寸验证

**重写** `tests/test_vlm_accuracy.py`:
- 断言改为 `assert result.match == True`（对于应该匹配的案例）
- 不再将 UNKNOWN 视为成功
- 使用 VTK 渲染而非截屏

**新建** `tests/test_assembly_vlm_e2e.py` (Task 64 缺失文件):
- 真实 VLM + VTK 渲染 + 41 零件机器人
- 断言视觉问题减少 >= 50%

### Step 6: 清理 GUI 依赖

以下工具可以标记为 deprecated（不再用于验证）:
- `FCOpenGUITool` — 不再需要开 GUI 来验证
- `FCCloseGUITool` — 不再需要关 GUI
- `FCSetCameraTool` — VTK 直接设相机
- `WindowAnalyzeTool` — 不再需要窗口分析
- `VLMLocateTool` — 不再需要通过 VLM 定位 GUI 元素

保留（有其他用途）:
- `gui_click`, `gui_type`, `gui_drag` — 可能用于 SolidWorks 自动化
- `screen_capture`, `list_windows` — 通用工具

## 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `src/lang3d/tools/vtk_renderer.py` | **新建** | VTK 离屏渲染器核心模块 |
| `src/lang3d/tools/vlm.py` | **重写** | CADVerifyTool 改用 VTK 渲染 |
| `src/lang3d/agent/assembly_visual_verifier.py` | **重写** | _render_to_dir() 改用 VTK |
| `src/lang3d/tools/freecad.py` | **修改** | fc_batch 新增 render 操作 |
| `src/lang3d/tools/base.py` | **修改** | TOOL_CATEGORIES 新增 vtk 类别 |
| `src/lang3d/agent/core.py` | **修改** | 注册 VTK 渲染工具 |
| `tests/test_vtk_renderer.py` | **新建** | VTK 渲染器测试 |
| `tests/test_vlm_accuracy.py` | **重写** | 加强断言，使用 VTK |
| `tests/test_assembly_vlm_e2e.py` | **新建** | Task 64 缺失的真实 VLM 测试 |

## 不改动的部分

- `assembly_vlm.py` (工具接口) — 只调用 verifier，不改
- `core.py` 的 auto-fix 循环 — 只检测 `MATCH: False` 文本，与渲染方式无关
- `reflector.py` — 只解析 VLM 文本输出，与渲染方式无关
- `fix_strategy.py` — 纯文本分析，不改
- 所有非验证相关的工具和模块

## 依赖

**零新依赖** — VTK 9.3.1 已安装，trimesh 4.12.2 已安装（用于 STL 加载的备选）。
