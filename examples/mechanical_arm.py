"""
Example: 3-DOF Mechanical Arm Design Task

This example demonstrates how to use the Language-3D Agent
to design a 3-DOF robotic arm for 3D printing.

Usage:
    python -m lang3d
    > /run 设计一个3自由度机械臂，所有部件可用于3D打印
"""

# Task description that can be fed to the agent
TASK = """
设计一个3自由度桌面机械臂，要求：
1. 所有部件可以用 FDM 3D 打印机制造（PLA 材料）
2. 使用 SG90 舵机驱动
3. 工作半径 ≥ 250mm
4. 底座使用法兰安装
5. 末端执行器接口标准化

请完成以下步骤：
1. 设计底座和旋转关节
2. 设计肩部和肘部连杆
3. 设计腕部和末端执行器安装座
4. 设计舵机安装座
5. 导出所有零件为 STL 文件
6. 验证零件可打印性
"""

# Expected agent plan (for reference)
EXPECTED_PLAN = [
    "分析机械臂结构需求，确定各部件尺寸",
    "创建项目目录结构",
    "设计底座板（base_plate）",
    "设计底座旋转关节（base_joint_housing）",
    "设计肩部连杆（shoulder_link）",
    "设计肘部关节（elbow_joint）",
    "设计前臂连杆（forearm_link）",
    "设计腕部关节（wrist_joint）",
    "设计末端执行器安装座（end_effector_mount）",
    "设计舵机安装座（servo_holder）",
    "验证所有零件尺寸配合",
    "导出 STL 文件",
]

if __name__ == "__main__":
    print("This is a reference example. Run with:")
    print("  python -m lang3d")
    print("Then enter the task in the CLI.")
