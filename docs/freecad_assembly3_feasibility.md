# FreeCAD Assembly3 Python API 可行性研究报告

> 研究日期：2026-06-05
> 结论：FreeCAD Assembly3 **不能** headless 使用，但 SolveSpace 求解器可通过 py_slvs 独立使用

---

## 核心结论

### FreeCAD Assembly3 不能 headless 使用
- 约束求解流程深度绑定 `FreeCADGui`
- headless 模式下 `import FreeCADGui` 崩溃或不可用
- 社区确认：FreeCAD 的 Python API 是 "UI first" 设计

### 推荐：py_slvs / python-solvespace
- `pip install py-slvs` 或 `pip install python-solvespace`
- 与 Assembly3 底层使用的同一个 SolveSpace 引擎
- 完全无 GUI 依赖，34 种约束类型
- FreeCAD Assembly3 和 Blender CAD Sketcher 都用这个库

---

## 方案对比

| 方案 | Headless? | 约束类型 | 推荐度 |
|------|-----------|---------|--------|
| FreeCAD Assembly3 | 不行 | 34 种（SolveSpace） | 放弃 |
| py_slvs (py-slvs) | 可以 | 34 种（SolveSpace） | 推荐 |
| python-solvespace | 可以 | 34 种（SolveSpace） | 推荐（更高级 API） |
| FreeCAD Assembly4 | 部分 | 无求解器 | 不适合 |
| FreeCAD 新 Assembly WB (1.0) | 不行 | 高级关节（OndselSolver） | 不适合 headless |
| OndselSolver 独立 | C++ only | 高级关节 + 动力学 | 工作量太大 |

---

## SolveSpace 约束类型（34 种）

```
POINTS_COINCIDENT       - 两点重合
PT_PT_DISTANCE          - 两点距离
PT_PLANE_DISTANCE       - 点到面距离
PT_LINE_DISTANCE        - 点到线距离
PT_FACE_DISTANCE        - 点到面距离
PT_IN_PLANE             - 点在平面上
PT_ON_LINE              - 点在线上
PT_ON_FACE              - 点在面上
EQUAL_LENGTH_LINES      - 等长线段
LENGTH_RATIO            - 线段长度比
EQ_LEN_PT_LINE_D        - 等长与点线距离
EQ_PT_LN_DISTANCES      - 等点线距离
EQUAL_ANGLE             - 等角
EQUAL_LINE_ARC_LEN      - 线长等弧长
SYMMETRIC               - 关于面对称
SYMMETRIC_HORIZ         - 水平对称
SYMMETRIC_VERT          - 垂直对称
SYMMETRIC_LINE          - 关于线对称
AT_MIDPOINT             - 在中点
HORIZONTAL              - 水平
VERTICAL                - 垂直
DIAMETER                - 直径
PT_ON_CIRCLE            - 点在圆上
SAME_ORIENTATION        - 法向同向
ANGLE                   - 角度
PARALLEL                - 平行
PERPENDICULAR           - 垂直
ARC_LINE_TANGENT        - 弧线相切
CUBIC_LINE_TANGENT      - 三次曲线相切
EQUAL_RADIUS            - 等半径
PROJ_PT_DISTANCE        - 投影点距
WHERE_DRAGGED           - 固定位置
CURVE_CURVE_TANGENT     - 曲线相切
LENGTH_DIFFERENCE       - 长度差
```

## 实体类型

```
POINT_IN_3D    - 3D 点
POINT_IN_2D    - 2D 点（在工作平面上）
NORMAL_IN_3D   - 3D 法向（四元数）
NORMAL_IN_2D   - 2D 法向
DISTANCE       - 距离值
WORKPLANE      - 工作平面（原点 + 法向）
LINE_SEGMENT   - 线段
CUBIC          - 三次贝塞尔曲线
CIRCLE         - 圆
ARC_OF_CIRCLE  - 圆弧
TRANSFORM      - 变换
```

---

## 关节到约束的映射

| 关节类型 | DOF 消除 | SolveSpace 约束组合 |
|---------|----------|---------------------|
| **Fixed** | 6 | 3 点重合 + 2 法向同向 |
| **Revolute** | 5（留 1 旋转） | 1 点重合（轴点）+ 1 法向同向（轴向） |
| **Prismatic** | 5（留 1 平移） | 2 法向同向（防旋转）+ 1 点在线上 |
| **Cylindrical** | 4（留 1 旋转 + 1 平移） | 1 点重合 + 1 点在线上 |
| **Spherical/Ball** | 3（留 3 旋转） | 1 点重合（球心） |
| **Planar** | 3（留 2 平移 + 1 旋转） | 1 点在平面上 + 1 法向同向 |

---

## 代码示例

### python-solvespace 基本用法

```python
from python_solvespace import SolverSystem

sys = SolverSystem()

# 固定零件（支架）
bracket_point = sys.add_point_3d(0, 0, 0)
bracket_normal = sys.add_normal_3d(1, 0, 0, 0)  # Z-up 四元数
sys.dragged(bracket_point)  # 固定不动

# 可动零件（旋转臂）
arm_point = sys.add_point_3d(0, 0, 0)
arm_end = sys.add_point_3d(20, 0, 0)
arm_normal = sys.add_normal_3d(1, 0, 0, 0)

# Revolute 关节：轴点重合 + 轴向同向
sys.coincident(bracket_point, arm_point)
sys.same_orientation(bracket_normal, arm_normal)
sys.distance(arm_point, arm_end, 20.0)

result = sys.solve()
print(f"DOF remaining: {sys.dof()}")  # 应为 1（绕轴旋转自由）
```

### py_slvs 低级 API 用法

```python
from py_slvs import slvs

solvesys = slvs.System()
FIXED_GROUP = 1
group = 3

# 固定原点
base_point = add_point((0, 0, 0), fixed=True)
base_normal = solvesys.addNormal3dV(1, 0, 0, 0, group=FIXED_GROUP)

# 自由点
point = add_point((5, 0, 5))

# 距离约束：原点到自由点 = 10
solvesys.addPointsDistance(10, base_point, point, group=group)

result = solvesys.solve(group=group)
# result: 0 = 成功
```

---

## 求解结果码

```
0 = 成功求解
1 = 约束不一致（矛盾）
2 = 未收敛
3 = 未知数过多
4 = 求解器初始化失败
5 = 检测到冗余约束
6 = 未知失败
```

---

## AADvark 的做法

AADvark (ACM CAIS 2026) 是最接近我们系统的论文：

1. LLM Agent 生成 JSON 零件+装配定义
2. JSON 通过编译器传给 OndselSolver（非 SolveSpace）
3. FreeCAD 渲染结果
4. 视觉反馈 + 错误信息回传 Agent

AADvark 对 FreeCAD/OndselSolver 做了 5 项修改：
- 四元数替代欧拉角（避免 180° 歧义）
- 错误时也更新位置（让 Agent 看到问题）
- 增强错误消息
- 确定性 Newton's Method
- 每个面/边唯一颜色+纹理标识

**但他们不是 headless 的** — FreeCAD 需要运行 GUI 来渲染。

---

## 风险和注意事项

1. **约束到关节的映射**：SolveSpace 只有几何约束，没有"关节"概念。需要自建映射层。
2. **初始猜测敏感**：Newton's Method 需要好的初始值，否则不收敛或收敛到错误解。
3. **无 B-rep 支持**：求解器操作抽象几何实体（点/线/面/法向），不是 CAD 实体。需要从零件提取参考几何。
4. **四元数数学**：SolveSpace 用四元数表示方向，需要转换工具。
5. **组管理**：Group 1 是固定组，后续组包含可求解实体。错误管理会导致求解失败。
6. **Windows 支持**：py_slvs 有预编译 wheel，需验证与 Python 版本兼容。

---

## 参考文献

1. [realthunder/slvs_py (GitHub)](https://github.com/realthunder/slvs_py) — py_slvs 源码
2. [py-slvs (PyPI)](https://pypi.org/project/py-slvs/) — 安装包
3. [python-solvespace API 文档](https://pyslvs-ui.readthedocs.io/en/stable/python-solvespace-api/) — 完整 API 参考
4. [py_slvs 代码示例 (Andrej730 Gist)](https://gist.github.com/Andrej730/5b99ed5dfcb69734bb53005c71f18813) — 2D+3D 约束求解示例
5. [AADvark 论文 (arXiv)](https://arxiv.org/html/2604.15184v2) — Agent-aided design with FreeCAD
6. [FreeCAD Assembly3 Wiki](https://wiki.freecad.org/Assembly3_Workbench) — Assembly3 文档
7. [FreeCAD Assembly3 Build Instructions](https://github.com/realthunder/FreeCAD_assembly3/wiki/Build-Instruction) — 多求解器架构
8. [FreeCAD Headless Wiki](https://wiki.freecad.org/Headless_FreeCAD) — headless 模式文档
9. [OndselSolver (GitHub)](https://github.com/Ondsel-Development/OndselSolver) — C++ Multibody Dynamics Solver
10. [FreeCAD Forum: Assembly Python API](https://forum.freecad.org/viewtopic.php?t=92402) — Python 约束定义讨论
11. [FreeCAD Forum: Joint Variables](https://forum.freecad.org/viewtopic.php?t=92562) — Python 驱动关节动画
12. [FreeCAD Forum: Headless Mode](https://forum.freecad.org/viewtopic.php?t=39470) — headless 限制确认
13. [GitHub Issue #16407](https://github.com/FreeCAD/FreeCAD/issues/16407) — headless 导入 GUI 模块崩溃
14. [GitHub Issue #20377](https://github.com/FreeCAD/FreeCAD/issues/20377) — 求解器稳定性问题
15. [Reddit: FreeCAD API Stability](https://www.reddit.com/r/FreeCAD/comments/1p9gpia/) — 社区评估
