"""FreeCAD mechanical arm parts modeling test.

Models all 8 parts of the 3-DOF robotic arm from the knowledge base:
1. base_plate - 底座板 (120mm diameter, 8mm thick, 4x M6 holes)
2. base_joint_housing - 底座关节外壳 (R40, H40, wall 5mm)
3. shoulder_link - 肩部连杆 (150x40x30mm)
4. elbow_joint - 肘关节 (R30, H35, shaft R6)
5. forearm_link - 前臂连杆 (120x35x25mm, hollow)
6. wrist_joint - 腕关节 (R20, H25, shaft R4)
7. end_effector_mount - 末端安装座 (R17.5, H15mm)
8. servo_holder - SG90 舵机座 (30x24x12mm)
"""

from __future__ import annotations

import os
import sys

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lang3d.tools.freecad import _execute_operations, _build_script, _find_freecad_python

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "projects", "robotic_arm")


def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def model_base_plate() -> str:
    """底座板: 圆盘 R60, 厚 8mm, 4x M6 安装孔 (通孔 R3.3)"""
    ops = [
        {"type": "new_doc", "name": "BasePlate"},
        # 主圆盘
        {"type": "make_cylinder", "radius": 60, "height": 8, "name": "Plate"},
        # 4 个 M6 安装孔 (clearance hole R3.3), 分布在 R45 的圆周上
        {"type": "make_cylinder", "radius": 3.3, "height": 10, "name": "Hole1"},
        {"type": "move", "object": "Hole1", "dx": 45, "dy": 0, "dz": -1},
        {"type": "make_cylinder", "radius": 3.3, "height": 10, "name": "Hole2"},
        {"type": "move", "object": "Hole2", "dx": 0, "dy": 45, "dz": -1},
        {"type": "make_cylinder", "radius": 3.3, "height": 10, "name": "Hole3"},
        {"type": "move", "object": "Hole3", "dx": -45, "dy": 0, "dz": -1},
        {"type": "make_cylinder", "radius": 3.3, "height": 10, "name": "Hole4"},
        {"type": "move", "object": "Hole4", "dx": 0, "dy": -45, "dz": -1},
        # 逐个做差集
        {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "Hole1", "result_name": "Plate"},
        {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "Hole2", "result_name": "Plate"},
        {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "Hole3", "result_name": "Plate"},
        {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "Hole4", "result_name": "Plate"},
        # 中心孔 (轴承座预留)
        {"type": "make_cylinder", "radius": 15, "height": 10, "name": "CenterHole"},
        {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "CenterHole", "result_name": "BasePlate"},
        {"type": "object_info", "object": "BasePlate"},
        {"type": "export_stl", "path": os.path.join(OUTPUT_DIR, "base_plate.stl")},
        {"type": "save", "path": os.path.join(OUTPUT_DIR, "base_plate.FCStd")},
    ]
    return _execute_operations(ops)


def model_base_joint_housing() -> str:
    """底座关节外壳: 外 R40, H40, 壁厚 5mm"""
    ops = [
        {"type": "new_doc", "name": "BaseJointHousing"},
        {"type": "cylinder_with_hole", "outer_radius": 40, "inner_radius": 35, "height": 40, "name": "Housing"},
        # 底部法兰 (3 个 M4 安装孔)
        {"type": "make_cylinder", "radius": 45, "height": 5, "name": "Flange"},
        {"type": "move", "object": "Flange", "dx": 0, "dy": 0, "dz": 0},
        {"type": "boolean", "operation": "union", "object1": "Housing", "object2": "Flange", "result_name": "WithFlange"},
        # 法兰安装孔 (3 个, R=2.25, M4 tap)
        {"type": "make_cylinder", "radius": 2.25, "height": 8, "name": "FH1"},
        {"type": "move", "object": "FH1", "dx": 38, "dy": 0, "dz": -1},
        {"type": "boolean", "operation": "cut", "object1": "WithFlange", "object2": "FH1", "result_name": "WithFlange"},
        {"type": "make_cylinder", "radius": 2.25, "height": 8, "name": "FH2"},
        # 120 degrees apart
        {"type": "move", "object": "FH2", "dx": -19, "dy": 32.9, "dz": -1},
        {"type": "boolean", "operation": "cut", "object1": "WithFlange", "object2": "FH2", "result_name": "WithFlange"},
        {"type": "make_cylinder", "radius": 2.25, "height": 8, "name": "FH3"},
        {"type": "move", "object": "FH3", "dx": -19, "dy": -32.9, "dz": -1},
        {"type": "boolean", "operation": "cut", "object1": "WithFlange", "object2": "FH3", "result_name": "BaseJointHousing"},
        {"type": "object_info", "object": "BaseJointHousing"},
        {"type": "export_stl", "path": os.path.join(OUTPUT_DIR, "base_joint_housing.stl")},
        {"type": "save", "path": os.path.join(OUTPUT_DIR, "base_joint_housing.FCStd")},
    ]
    return _execute_operations(ops)


def model_shoulder_link() -> str:
    """肩部连杆: 150x40x30mm, 两端有连接孔"""
    ops = [
        {"type": "new_doc", "name": "ShoulderLink"},
        # 主体
        {"type": "make_box", "length": 150, "width": 40, "height": 30, "name": "Body"},
        # 减重孔 (中间挖 4 个大孔)
        {"type": "make_cylinder", "radius": 10, "height": 35, "name": "Lighten1"},
        {"type": "move", "object": "Lighten1", "dx": 40, "dy": 20, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "Lighten1", "result_name": "Body"},
        {"type": "make_cylinder", "radius": 10, "height": 35, "name": "Lighten2"},
        {"type": "move", "object": "Lighten2", "dx": 75, "dy": 20, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "Lighten2", "result_name": "Body"},
        {"type": "make_cylinder", "radius": 10, "height": 35, "name": "Lighten3"},
        {"type": "move", "object": "Lighten3", "dx": 110, "dy": 20, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "Lighten3", "result_name": "Body"},
        # 端部连接孔 (两端各 2x M5)
        {"type": "make_cylinder", "radius": 2.75, "height": 35, "name": "EndHole1"},
        {"type": "move", "object": "EndHole1", "dx": 12, "dy": 12, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "EndHole1", "result_name": "Body"},
        {"type": "make_cylinder", "radius": 2.75, "height": 35, "name": "EndHole2"},
        {"type": "move", "object": "EndHole2", "dx": 12, "dy": 28, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "EndHole2", "result_name": "Body"},
        {"type": "make_cylinder", "radius": 2.75, "height": 35, "name": "EndHole3"},
        {"type": "move", "object": "EndHole3", "dx": 138, "dy": 12, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "EndHole3", "result_name": "Body"},
        {"type": "make_cylinder", "radius": 2.75, "height": 35, "name": "EndHole4"},
        {"type": "move", "object": "EndHole4", "dx": 138, "dy": 28, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "EndHole4", "result_name": "ShoulderLink"},
        # 圆角
        {"type": "fillet", "object": "ShoulderLink", "radius": 2},
        {"type": "object_info", "object": "ShoulderLink"},
        {"type": "export_stl", "path": os.path.join(OUTPUT_DIR, "shoulder_link.stl")},
        {"type": "save", "path": os.path.join(OUTPUT_DIR, "shoulder_link.FCStd")},
    ]
    return _execute_operations(ops)


def model_elbow_joint() -> str:
    """肘关节: 外 R30, H35, 轴 R6"""
    ops = [
        {"type": "new_doc", "name": "ElbowJoint"},
        {"type": "cylinder_with_hole", "outer_radius": 30, "inner_radius": 6, "height": 35, "name": "Joint"},
        # 紧定螺钉孔 (M3, 侧面 1 个)
        {"type": "make_cylinder", "radius": 1.65, "height": 40, "name": "SetScrew"},
        {"type": "move", "object": "SetScrew", "dx": 0, "dy": -20, "dz": 17},
        {"type": "boolean", "operation": "cut", "object1": "Joint", "object2": "SetScrew", "result_name": "ElbowJoint"},
        {"type": "object_info", "object": "ElbowJoint"},
        {"type": "export_stl", "path": os.path.join(OUTPUT_DIR, "elbow_joint.stl")},
        {"type": "save", "path": os.path.join(OUTPUT_DIR, "elbow_joint.FCStd")},
    ]
    return _execute_operations(ops)


def model_forearm_link() -> str:
    """前臂连杆: 120x35x25mm, 中空轻量化"""
    ops = [
        {"type": "new_doc", "name": "ForearmLink"},
        # 外壳
        {"type": "make_box", "length": 120, "width": 35, "height": 25, "name": "Shell"},
        # 内部中空 (壁厚 4mm)
        {"type": "make_box", "length": 108, "width": 27, "height": 17, "name": "Void"},
        {"type": "move", "object": "Void", "dx": 6, "dy": 4, "dz": 4},
        {"type": "boolean", "operation": "cut", "object1": "Shell", "object2": "Void", "result_name": "ForearmLink"},
        # 连接孔
        {"type": "make_cylinder", "radius": 2.75, "height": 30, "name": "CH1"},
        {"type": "move", "object": "CH1", "dx": 10, "dy": 10, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "ForearmLink", "object2": "CH1", "result_name": "ForearmLink"},
        {"type": "make_cylinder", "radius": 2.75, "height": 30, "name": "CH2"},
        {"type": "move", "object": "CH2", "dx": 10, "dy": 25, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "ForearmLink", "object2": "CH2", "result_name": "ForearmLink"},
        {"type": "make_cylinder", "radius": 2.75, "height": 30, "name": "CH3"},
        {"type": "move", "object": "CH3", "dx": 110, "dy": 10, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "ForearmLink", "object2": "CH3", "result_name": "ForearmLink"},
        {"type": "make_cylinder", "radius": 2.75, "height": 30, "name": "CH4"},
        {"type": "move", "object": "CH4", "dx": 110, "dy": 25, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "ForearmLink", "object2": "CH4", "result_name": "ForearmLink"},
        # 圆角
        {"type": "fillet", "object": "ForearmLink", "radius": 1.5},
        {"type": "object_info", "object": "ForearmLink"},
        {"type": "export_stl", "path": os.path.join(OUTPUT_DIR, "forearm_link.stl")},
        {"type": "save", "path": os.path.join(OUTPUT_DIR, "forearm_link.FCStd")},
    ]
    return _execute_operations(ops)


def model_wrist_joint() -> str:
    """腕关节: 外 R20, H25, 轴 R4"""
    ops = [
        {"type": "new_doc", "name": "WristJoint"},
        {"type": "cylinder_with_hole", "outer_radius": 20, "inner_radius": 4, "height": 25, "name": "Joint"},
        # 紧定螺钉孔
        {"type": "make_cylinder", "radius": 1.25, "height": 25, "name": "SS"},
        {"type": "move", "object": "SS", "dx": 0, "dy": -14, "dz": 12},
        {"type": "boolean", "operation": "cut", "object1": "Joint", "object2": "SS", "result_name": "WristJoint"},
        {"type": "object_info", "object": "WristJoint"},
        {"type": "export_stl", "path": os.path.join(OUTPUT_DIR, "wrist_joint.stl")},
        {"type": "save", "path": os.path.join(OUTPUT_DIR, "wrist_joint.FCStd")},
    ]
    return _execute_operations(ops)


def model_end_effector_mount() -> str:
    """末端执行器安装座: R17.5, H15, 标准法兰"""
    ops = [
        {"type": "new_doc", "name": "EndEffectorMount"},
        # 法兰盘
        {"type": "make_cylinder", "radius": 17.5, "height": 4, "name": "Flange"},
        # 连接柱
        {"type": "make_cylinder", "radius": 12, "height": 15, "name": "Pillar"},
        {"type": "move", "object": "Pillar", "dx": 0, "dy": 0, "dz": 4},
        {"type": "boolean", "operation": "union", "object1": "Flange", "object2": "Pillar", "result_name": "Mount"},
        # 中心孔
        {"type": "make_cylinder", "radius": 4, "height": 20, "name": "CenterHole"},
        {"type": "boolean", "operation": "cut", "object1": "Mount", "object2": "CenterHole", "result_name": "Mount"},
        # 法兰安装孔 (4x M3)
        {"type": "make_cylinder", "radius": 1.7, "height": 8, "name": "FH1"},
        {"type": "move", "object": "FH1", "dx": 12, "dy": 0, "dz": -1},
        {"type": "boolean", "operation": "cut", "object1": "Mount", "object2": "FH1", "result_name": "Mount"},
        {"type": "make_cylinder", "radius": 1.7, "height": 8, "name": "FH2"},
        {"type": "move", "object": "FH2", "dx": -12, "dy": 0, "dz": -1},
        {"type": "boolean", "operation": "cut", "object1": "Mount", "object2": "FH2", "result_name": "Mount"},
        {"type": "make_cylinder", "radius": 1.7, "height": 8, "name": "FH3"},
        {"type": "move", "object": "FH3", "dx": 0, "dy": 12, "dz": -1},
        {"type": "boolean", "operation": "cut", "object1": "Mount", "object2": "FH3", "result_name": "Mount"},
        {"type": "make_cylinder", "radius": 1.7, "height": 8, "name": "FH4"},
        {"type": "move", "object": "FH4", "dx": 0, "dy": -12, "dz": -1},
        {"type": "boolean", "operation": "cut", "object1": "Mount", "object2": "FH4", "result_name": "EndEffectorMount"},
        {"type": "object_info", "object": "EndEffectorMount"},
        {"type": "export_stl", "path": os.path.join(OUTPUT_DIR, "end_effector_mount.stl")},
        {"type": "save", "path": os.path.join(OUTPUT_DIR, "end_effector_mount.FCStd")},
    ]
    return _execute_operations(ops)


def model_servo_holder() -> str:
    """SG90 舵机座: 30x24x12mm"""
    ops = [
        {"type": "new_doc", "name": "ServoHolder"},
        # 底板
        {"type": "make_box", "length": 30, "width": 24, "height": 3, "name": "Base"},
        # 左侧板
        {"type": "make_box", "length": 30, "width": 3, "height": 12, "name": "LeftWall"},
        {"type": "boolean", "operation": "union", "object1": "Base", "object2": "LeftWall", "result_name": "Part"},
        # 右侧板
        {"type": "make_box", "length": 30, "width": 3, "height": 12, "name": "RightWall"},
        {"type": "move", "object": "RightWall", "dx": 0, "dy": 21, "dz": 0},
        {"type": "boolean", "operation": "union", "object1": "Part", "object2": "RightWall", "result_name": "Part"},
        # 舵机腔体 (标记用, 不实际挖空)
        # 安装孔 (底板 4x M2)
        {"type": "make_cylinder", "radius": 1.1, "height": 8, "name": "SH1"},
        {"type": "move", "object": "SH1", "dx": 4, "dy": 4, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Part", "object2": "SH1", "result_name": "Part"},
        {"type": "make_cylinder", "radius": 1.1, "height": 8, "name": "SH2"},
        {"type": "move", "object": "SH2", "dx": 26, "dy": 4, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Part", "object2": "SH2", "result_name": "Part"},
        {"type": "make_cylinder", "radius": 1.1, "height": 8, "name": "SH3"},
        {"type": "move", "object": "SH3", "dx": 4, "dy": 20, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Part", "object2": "SH3", "result_name": "Part"},
        {"type": "make_cylinder", "radius": 1.1, "height": 8, "name": "SH4"},
        {"type": "move", "object": "SH4", "dx": 26, "dy": 20, "dz": -2},
        {"type": "boolean", "operation": "cut", "object1": "Part", "object2": "SH4", "result_name": "ServoHolder"},
        {"type": "object_info", "object": "ServoHolder"},
        {"type": "export_stl", "path": os.path.join(OUTPUT_DIR, "servo_holder.stl")},
        {"type": "save", "path": os.path.join(OUTPUT_DIR, "servo_holder.FCStd")},
    ]
    return _execute_operations(ops)


def main():
    ensure_dir()

    if not _find_freecad_python():
        print("ERROR: FreeCAD not found!")
        sys.exit(1)

    print("=" * 60)
    print("  3-DOF Robotic Arm - FreeCAD Modeling Test")
    print("=" * 60)

    parts = [
        ("base_plate",           "底座板",            model_base_plate),
        ("base_joint_housing",   "底座关节外壳",       model_base_joint_housing),
        ("shoulder_link",        "肩部连杆",           model_shoulder_link),
        ("elbow_joint",          "肘关节",             model_elbow_joint),
        ("forearm_link",         "前臂连杆",           model_forearm_link),
        ("wrist_joint",          "腕关节",             model_wrist_joint),
        ("end_effector_mount",   "末端执行器安装座",    model_end_effector_mount),
        ("servo_holder",         "SG90 舵机座",        model_servo_holder),
    ]

    results = []
    for name, desc, model_fn in parts:
        print(f"\n--- {desc} ({name}) ---")
        try:
            result = model_fn()
            print(result)
            # Check output file
            stl_path = os.path.join(OUTPUT_DIR, f"{name}.stl")
            fc_path = os.path.join(OUTPUT_DIR, f"{name}.FCStd")
            stl_size = os.path.getsize(stl_path) if os.path.exists(stl_path) else 0
            fc_size = os.path.getsize(fc_path) if os.path.exists(fc_path) else 0
            if stl_size > 0:
                print(f"  STL: {stl_size:,} bytes | FCStd: {fc_size:,} bytes")
                results.append((name, desc, "OK", stl_size, fc_size))
            else:
                print(f"  STL export failed (0 bytes)")
                results.append((name, desc, "FAILED", 0, 0))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((name, desc, "FAILED", 0, 0))

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    ok_count = sum(1 for r in results if r[2] == "OK")
    total_stl = sum(r[3] for r in results)
    print(f"  Parts: {ok_count}/{len(results)} successful")
    print(f"  Total STL size: {total_stl:,} bytes ({total_stl/1024:.1f} KB)")
    print()
    for name, desc, status, stl_size, fc_size in results:
        mark = "OK" if status == "OK" else "FAIL"
        print(f"  [{mark}] {desc:20s} STL={stl_size:>8,} bytes  FCStd={fc_size:>8,} bytes")

    if ok_count == len(parts):
        print("\n  ALL PARTS MODELED SUCCESSFULLY")
    else:
        print(f"\n  {len(parts) - ok_count} PART(S) FAILED")

    return ok_count == len(parts)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
