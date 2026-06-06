# Assembly Solver 深度研究报告

> 研究日期：2026-06-05
> 研究范围：2024-2026 年装配求解器相关论文、工具、工业方案
> 目标：诊断 Language-3D 当前 assembly_solver.py 的根本问题，找到改进方向

---

## 0. 当前 Solver 诊断

### 现状
- `assembly_solver.py`：1066 行，BFS 树遍历 + 6 面 anchor 定位 + Rodrigues 旋转
- 3 轮修补后 123 个测试全通过，但 41 零件机器人视觉结果仍有碰撞/漂浮/不合理

### 5 个根本限制
1. **无碰撞检测** — 零件可以互相穿透
2. **无空间推理** — 不知道零件占据了多少空间
3. **Anchor-face 模型过于简化** — 6 个面无法表达 L 型、U 型等复杂零件
4. **无视觉反馈闭环** — 没有 VLM 验证装配正确性
5. **Ad-hoc 修补** — 每次修复只针对一个具体机器人，不可泛化

---

## 1. 约束求解器：CAD 系统怎么做装配

### 工业标准：Siemens D-Cubed 3D DCM
- SolidWorks/NX/Inventor/Fusion 360 都用这个引擎
- 算法：约束图（constraint graph）→ 非线性方程组 → Newton-Raphson 迭代
- 约束类型：coincident, concentric, parallel, distance, angle, tangent 等
- 两阶段：分析阶段分解为子问题 → 构造阶段数值求解

### FreeCAD Assembly3：SolveSpace
- SolveSpace 引擎，符号代数系统 + Newton-Raphson
- 40+ 约束类型（vs 我们的 6 面 anchor）
- 无几何/拓扑理解，纯数值消歧
- 已知问题：约束不一致时会发散；不允许冗余约束

### 对我们的意义
- 我们的 6 面 anchor 模型 vs 工业的 40+ 约束类型，差距巨大
- 但不需要达到 D-Cubed 的复杂度
- 关键改进：从 "命名锚点面" 升级为 "约束关系"（如 coincident, concentric, distance）

### 参考文献
- [Siemens D-Cubed 3D DCM](https://www.siemens.com/en-us/products/plm-components/d-cubed/3d-dcm/)
- [Engineering.com: Parasolid, D-Cubed and Siemens](https://www.engineering.com/parasolid-d-cubed-and-siemens-the-heart-of-your-cad-software-belongs-to-another/)
- [Geometric Constraint Solving Survey (Purdue)](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Bettig-Comp_and_Info_Sci_in_Eng-2011-Geometric_Constraint_Solving_In_Parametric_CAD.pdf)
- [Siemens PLM Blog: Geometric Constraint Solving](https://blogs.sw.siemens.com/plm-components/geometric-constraint-solving-1-introduction/)
- [FreeCAD Assembly3 Wiki](https://wiki.freecad.org/Assembly3_Workbench)
- [SolveSpace Technology Page](https://solvespace.github.io/solvespace-web/tech.html)

---

## 2. 碰撞检测（最高优先级改进）

### 标准两阶段管线
1. **Broadphase**（快速过滤）：AABB 树 / BVH，O(n log n)
2. **Narrowphase**（精确检测）：
   - **GJK**（Gilbert-Johnson-Keerthi）：凸体相交检测
   - **SAT**（Separating Axis Theorem）：分离轴测试
   - **EPA**（Expanding Polytope Algorithm）：穿透深度计算

### 推荐 Python 方案：trimesh + python-fcl
- `trimesh.collision.CollisionManager` 封装了 `python-fcl`（Flexible Collision Library）
- API：`add_object()`, `in_collision_internal()`, `min_distance_single()`
- FCL 内部用 AABB broadphase + GJK narrowphase
- 安装：`pip install trimesh python-fcl`
- 能获取穿透深度和分离向量（不仅检测还能修复）

### 替代方案
- `distance3d`（PyPI）：MPR 算法，直接提供分离向量
- `MeshLib`：C++/Python SDK，实时网格碰撞
- `PyBullet`：完整物理引擎，包含碰撞检测

### 修复策略
检测到碰撞后：计算穿透深度和法向量 → 沿法向量平移零件（穿透深度 + 小余量）

### 对我们的意义
**这是单个最高影响的改进。** 当前 solver 零碰撞感知。50-100 行代码即可集成 trimesh + python-fcl 的 post-solve 碰撞检测，立即捕获最严重的零件重叠问题。

### 参考文献
- [trimesh.collision 文档](https://trimesh.org/trimesh.collision.html)
- [python-fcl (GitHub)](https://github.com/BerkeleyAutomation/python-fcl)
- [distance3d (PyPI)](https://pypi.org/project/distance3d/)
- [MeshLib 碰撞检测](https://meshlib.io/feature/collision-detection/)

---

## 3. 空间推理

### 可选方案（按复杂度排序）
1. **BVH**（trimesh 内置）— 最简单，O(log n) 空间查询
2. **Voxel 占据网格** — 离散化空间，适合"放这个零件到最大空隙"
3. **Octree** — 层次空间索引，适合大规模装配
4. **SDF（Signed Distance Field）** — 最优雅但最复杂

### 对我们的意义
5-30 个零件的机器人场景，**BVH（通过 trimesh）就够了**。Voxel grid 是下一步。SDF 是 overkill。

### 参考文献
- [Representing Robot Geometry as Distance Fields (ICRA 2024)](https://publications.idiap.ch/attachments/papers/2024/Li_ICRA-2_2024.pdf)
- [BayesFusion-SDF (arXiv 2025)](https://arxiv.org/html/2602.19697v1)

---

## 4. 物理信息定位

### 核心原则
装配体在重力下稳定 ⟺ 质心投影落在支撑多边形内

### 工具
- **PyBullet**：免费 Python 物理引擎，最实用
- **MuJoCo**：更高保真度，更复杂
- **NVIDIA Isaac Sim**：GPU 加速，overkill

### Blox-Net（Berkeley, CoRL 2024）— 最相关论文
架构：GPT-4o 提议设计 → 物理模拟验证 → VLM 评估 → 迭代
关键方法：扰动分析（小随机位移测试鲁棒性）

### 实用稳定性检查
1. **质心投影**：加权平均 → 投影到地面接触凸包
2. **倾倒测试**：施加侧向力，检查是否倾倒
3. **支撑链**：每个零件必须有到地面的完整支撑链

### 参考文献
- [Blox-Net (arXiv)](https://arxiv.org/abs/2409.17126)
- [Blox-Net 项目页](https://bloxnet.org/)
- [Blox-Net 代码 (GitHub)](https://github.com/Apgoldberg1/blox-net-coderelease)

---

## 5. AI/ML 装配方法（2024-2026 关键论文）

### 核心论文

| 论文 | 会议 | 关键贡献 | 与我们的关系 |
|------|------|---------|-------------|
| **CADCodeVerify** | ICLR 2025 | 渲染→VLM验证→修正→循环 | **最重要的模式：视觉反馈闭环** |
| **AADvark** | arXiv 2025 | JSON定义→FreeCAD约束求解 | **架构最接近我们，验证了方向** |
| **ArtiCAD** | arXiv 2025 | 5种关节类型（Fixed/Revolute/Slider/Cylindrical/Ball） | **关节类型升级参考** |
| **CADFusion** | ICML 2025 | LLM + 视觉反馈交替生成 | 视觉反馈模式 |
| **Text2CAD** | NeurIPS 2024 | 文本→参数化 CAD | 文本到建模 |
| **CAD-Llama** | CVPR 2025 | LLM 生成 CAD 建模序列 | LLM 建模 |
| **Blox-Net** | CoRL 2024 | VLM + 物理 + 机器人 | 全栈参考架构 |
| **AssemLM** | arXiv 2025 | 多模态 LLM 装配推理，6D 位姿预测 | 空间推理 |
| **Assembler** | arXiv 2025 | Diffusion-based 锚点预测 | 扩散模型方法 |

### AADvark 的关键洞察
- JSON 定义零件和关节 → FreeCAD 约束求解 → LLM agent 代码综合
- 修改求解器工具产生"强验证信号"
- **验证了我们的 JSON 方向，但需要更丰富的约束语义**

### ArtiCAD 的关键洞察
- 5 种关节类型比 6 面 anchor 更灵活
- 每个关节在共享坐标系上连接零件，具有特定自由度
- **自然的升级路径：从 anchor-face → 共享坐标系 + DOF**

### CADCodeVerify 的关键洞察
- 渲染→VLM验证→修正循环显著提升生成质量
- 视觉反馈改善了 3D 物体结构和编译成功率
- **这是我们需要的关键架构模式**

### 参考文献
- [CADCodeVerify (ICLR 2025)](https://openreview.net/forum?id=BLWaTeucYX) | [GitHub](https://github.com/Kamel773/CAD_Code_Generation) | [arXiv](https://arxiv.org/abs/2410.05340)
- [AADvark (arXiv 2025)](https://arxiv.org/html/2604.15184v2)
- [ArtiCAD (arXiv 2025)](https://arxiv.org/html/2604.10992v1)
- [CADFusion (ICML 2025)](https://icml.cc/virtual/2025/poster/46007)
- [Text2CAD (NeurIPS 2024)](https://github.com/SadilKhan/Text2CAD)
- [CAD-Llama (CVPR 2025)](http://openaccess.thecvf.com/content/CVPR2025/html/Li_CAD-Llama_Leveraging_Large_Language_Models_for_Computer-Aided_Design_Parametric_3D_CVPR_2025_paper.html)
- [AssemLM (arXiv 2025)](https://arxiv.org/html/2604.08983v1) | [项目页](https://assemlmhome.github.io/)
- [Blox-Net (arXiv)](https://arxiv.org/abs/2409.17126) | [项目页](https://bloxnet.org/)
- [Assembler: Anchor Point Diffusion (arXiv 2025)](https://arxiv.org/html/2506.17074v1)
- [Multi-part kinematic constraint prediction (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0010448524001325)

---

## 6. FreeCAD Assembly3 内部机制

### 求解引擎：SolveSpace
- 约束→非线性方程组→Newton-Raphson
- 无几何/拓扑理解，纯数值消歧
- 不允许冗余约束

### 40+ 约束类型（vs 我们的 6 面）
point-on-line, point-on-plane, point-on-point, axis-parallel, axis-coincident, plane-coincident, etc.

### 已知问题
- 约束不一致时求解器会发散
- 初始猜测不好时收敛到错误解
- 最佳实践：先粗定位零件再应用约束

### 对我们的意义
可以直接利用 FreeCAD 的 Assembly3 求解器（AADvark 就这么做），而不是自己重新实现约束求解。这给我们 40+ 约束类型 + 成熟求解器。

### 参考文献
- [FreeCAD Assembly3 Wiki](https://wiki.freecad.org/Assembly3_Workbench)
- [SolveSpace Technology Page](https://solvespace.github.io/solvespace-web/tech.html)
- [FreeCAD Forum: SolveSpace 冗余约束](https://forum.freecad.org/viewtopic.php?t=69052)
- [FreeCAD Forum: 求解器速度调试](https://forum.freecad.org/viewtopic.php?style=8&t=72904)
- [FreeCAD GitHub: 求解器稳定性问题](https://github.com/FreeCAD/FreeCAD/issues/20377)

---

## 7. 工业方案（ROS URDF, MoveIt）

### URDF 模型
- 链接树 + 关节描述
- 每个链接有：visual geometry（外观）+ **collision geometry**（碰撞）+ 惯性属性
- **关键洞察**：分离视觉几何和碰撞几何是最佳实践

### MoveIt 碰撞检测
- 用 FCL（同 python-fcl）
- 维护 collision robot（简化碰撞几何）+ collision world（环境物体）
- AABB broadphase + GJK narrowphase

### 对我们的意义
- 每个 Part 应有两套几何：详细 mesh（渲染）+ 简化凸包（碰撞）
- MoveIt 用 FCL 验证了 python-fcl 的选择

### 参考文献
- [MoveIt URDF/SRDF 教程](http://docs.ros.org/en/melodic/api/moveit_tutorials/html/doc/urdf_srdf/urdf_srdf_tutorial.html)
- [Articulated Robotics: URDF 教程](https://articulatedrobotics.xyz/tutorials/ready-for-ros/urdf/)

---

## 8. 最低可行改进

### 分析：当前代码的问题
- `_half_extent()` 从字典猜测尺寸，fallback 到 0.0，非常脆弱
- 无实际几何反馈到定位决策
- `ClosedChainSolver` 的 Newton-Raphson 用朴素扰动策略
- `assembly_verifier.py` 检查公差和文件存在，**不检查碰撞**
- `_compute_distribution_offset()` 做分布但无空间感知，零件可能分布到已占据空间

### 最高影响、最低成本的改进
**Post-solve 碰撞检测 via trimesh + python-fcl**

实现步骤：
1. Solver 计算完位置后，加载每个零件的 mesh（STL/OBJ）
2. 对每个 mesh 应用求解出的位置/旋转变换
3. 创建 `trimesh.collision.CollisionManager`，添加所有 mesh
4. 运行 `in_collision_internal()` 检查所有配对
5. 对碰撞配对，通过 `min_distance_single()` 计算穿透深度
6. 沿穿透法向量推开碰撞零件
7. 报告碰撞为验证失败

预估工作量：50-100 行新代码。预估影响：捕获 80%+ 的严重碰撞问题。

---

## 改进优先级排序

| 优先级 | 改进 | 工作量 | 影响 | 关键依赖 |
|--------|------|--------|------|----------|
| **P1** | Post-solve 碰撞检测 | 低（50-100 行） | **非常高** | `trimesh` + `python-fcl` |
| **P2** | 更丰富的关节/约束模型 | 中 | 高 | 数据模型变更 |
| **P3** | VLM 视觉反馈闭环 | 中 | 高 | VLM API |
| **P4** | 物理稳定性检查 | 中 | 中 | PyBullet |
| **P5** | 空间占据推理 | 中高 | 中 | trimesh BVH/voxel |
| **P6** | 完整约束方程求解器 | 高 | 低中 | SolveSpace 或自研 |

---

## 目标架构

综合 AADvark + Blox-Net + CADCodeVerify 三篇最相关论文：

```
LLM 提议装配 (JSON)
  → Solver 计算位置（当前 BFS + 更丰富的约束）
  → 碰撞检测器验证（trimesh + python-fcl）
  → 物理模拟检查稳定性（PyBullet，可选）
  → 渲染器生成多角度视图
  → VLM 验证视觉正确性
  → 如有问题：反馈 → LLM 修正 → 循环
```

此架构解决所有 5 个根本限制：
1. python-fcl 碰撞检测 → 解决"无碰撞检测"
2. trimesh BVH 空间推理 → 解决"无空间推理"
3. 任意附着点替代 6 面 anchor → 解决"anchor 模型过简"
4. VLM-in-the-loop 验证 → 解决"无视觉反馈"
5. 系统化约束模型 → 解决"ad-hoc 修补"

---

## 完整参考文献列表

### 约束求解器
- [Siemens D-Cubed 3D DCM](https://www.siemens.com/en-us/products/plm-components/d-cubed/3d-dcm/)
- [Engineering.com: Parasolid, D-Cubed and Siemens](https://www.engineering.com/parasolid-d-cubed-and-siemens-the-heart-of-your-cad-software-belongs-to-another/)
- [Hacker News: Geometric constraint solvers](https://news.ycombinator.com/item?id=30626043)
- [Geometric Constraint Solving Survey (Purdue)](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Bettig-Comp_and_Info_Sci_in_Eng-2011-Geometric_Constraint_Solving_In_Parametric_CAD.pdf)
- [Siemens PLM Blog: Constraint Solving Introduction](https://blogs.sw.siemens.com/plm-components/geometric-constraint-solving-1-introduction/)
- [Wikipedia: Geometric Constraint Solving](https://en.wikipedia.org/wiki/Geometric_constraint_solving)
- [Spatial Corp: Constraint Design Solver](https://www.spatial.com/solutions/3d-modeling/constraint-design-solver)
- [SolidWorks Best Practices for Mates](https://help.solidworks.com/2024/english/solidworks/sldworks/c_Best_Practices_for_Mates_SWassy.htm)
- [MDPI: CSP-based CAD Configurators](https://www.mdpi.com/1999-4893/15/9/318)

### FreeCAD Assembly3
- [FreeCAD Assembly3 Wiki](https://wiki.freecad.org/Assembly3_Workbench)
- [SolveSpace Technology Page](https://solvespace.github.io/solvespace-web/tech.html)
- [FreeCAD Forum: SolveSpace Redundant Constraints](https://forum.freecad.org/viewtopic.php?t=69052)
- [FreeCAD Forum: Solver Speed Debug](https://forum.freecad.org/viewtopic.php?style=8&t=72904)
- [FreeCAD GitHub: Solver Instability Issue](https://github.com/FreeCAD/FreeCAD/issues/20377)

### 碰撞检测
- [trimesh.collision 文档](https://trimesh.org/trimesh.collision.html)
- [python-fcl (GitHub)](https://github.com/BerkeleyAutomation/python-fcl)
- [distance3d (PyPI)](https://pypi.org/project/distance3d/)
- [MeshLib Collision Detection](https://meshlib.io/feature/collision-detection/)
- [Newcastle University: Collision Detection Tutorial (PDF)](https://research.ncl.ac.uk/game/mastersdegree/gametechnologies/previousinformation/physics4collisiondetection/2017%20Tutorial%204%20-%20Collision%20Detection.pdf)
- [GameDev StackExchange: Resolving Penetration](https://gamedev.stackexchange.com/questions/22310/how-to-resolve-penetration-of-two-colliding-bodies)
- [MLR: Active Learning of Neural Collision Handler](https://proceedings.mlr.press/v162/tan22b/tan22b.pdf)

### 空间推理
- [Representing Robot Geometry as Distance Fields (ICRA 2024)](https://publications.idiap.ch/attachments/papers/2024/Li_ICRA-2_2024.pdf)
- [BayesFusion-SDF (arXiv 2025)](https://arxiv.org/html/2602.19697v1)
- [Efficient Octree-Based Volumetric SLAM with SDF](https://www.doc.ic.ac.uk/~sleutene/publications/EVespaRAL_final.pdf)
- [iSDF: Real-Time Neural SDF (RSS 2022)](https://www.roboticsproceedings.org/rss18/p012.pdf)

### 物理与稳定性
- [Blox-Net (arXiv)](https://arxiv.org/abs/2409.17126)
- [Blox-Net 项目页](https://bloxnet.org/)
- [Blox-Net 代码 (GitHub)](https://github.com/Apgoldberg1/blox-net-coderelease)
- [Blox-Net (IEEE Xplore)](https://ieeexplore.ieee.org/document/11127489/)
- [Medium: Correll Lab Review of Blox-Net](https://medium.com/correll-lab/paper-review-blox-net-generative-design-for-robot-assembly-using-vlm-supervision-physics-90f892168f3f)

### AI/ML 装配方法
- [CADCodeVerify (ICLR 2025)](https://openreview.net/forum?id=BLWaTeucYX) | [GitHub](https://github.com/Kamel773/CAD_Code_Generation) | [arXiv](https://arxiv.org/abs/2410.05340)
- [AADvark (arXiv 2025)](https://arxiv.org/html/2604.15184v2)
- [ArtiCAD (arXiv 2025)](https://arxiv.org/html/2604.10992v1)
- [CADFusion (ICML 2025)](https://icml.cc/virtual/2025/poster/46007)
- [Text2CAD (NeurIPS 2024)](https://github.com/SadilKhan/Text2CAD)
- [CAD-Llama (CVPR 2025)](http://openaccess.thecvf.com/content/CVPR2025/html/Li_CAD-Llama_Leveraging_Large_Language_Models_for_Computer-Aided_Design_Parametric_3D_CVPR_2025_paper.html)
- [AssemLM (arXiv 2025)](https://arxiv.org/html/2604.08983v1) | [项目页](https://assemlmhome.github.io/)
- [Blox-Net (arXiv)](https://arxiv.org/abs/2409.17126)
- [Assembler: Anchor Point Diffusion (arXiv 2025)](https://arxiv.org/html/2506.17074v1)
- [3D-GPT (OpenReview)](https://openreview.net/forum?id=ttMwEuEPeB)
- [LL3M (ResearchGate)](https://www.researchgate.net/publication/394439477_LL3M_Large_Language_3D_Modelers)
- [SceneTeller](https://sceneteller.github.io/)
- [Multi-part kinematic constraint prediction (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0010448524001325)
- [Quantum Annealing for CAD Disassembly (Springer 2025)](https://link.springer.com/article/10.1007/s00170-025-17182-3)

### 工业（ROS/URDF/MoveIt）
- [MoveIt URDF/SRDF 教程](http://docs.ros.org/en/melodic/api/moveit_tutorials/html/doc/urdf_srdf/urdf_srdf_tutorial.html)
- [Articulated Robotics: URDF 教程](https://articulatedrobotics.xyz/tutorials/ready-for-ros/urdf/)
- [VLM-driven Skill Selection (ResearchGate)](https://www.researchgate.net/publication/397479550_VLM-driven_Skill_Selection_for_Robotic_Assembly_Tasks)
- [ExploreVLM (arXiv)](https://arxiv.org/html/2508.11918v1)
- [VLM Self-Feedback (arXiv)](https://arxiv.org/html/2404.06510v2)
