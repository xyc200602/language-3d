"""True End-to-End Test: 四轮双机械臂机器人

测试描述: 设计一个四轮双臂机器人，底盘四个麦克纳姆轮，底盘上安装两个3自由度机械臂

目的: 验证旋转修复后整个 pipeline 的正确性
  - FreeCAD assembly script 包含 rotate() 调用
  - URDF joint origin 有非零 rpy
  - 轮子方向正确（竖直圆柱而非水平）
  - 完整工程包输出

Usage:
    python e2e_4wheel_dual_arm.py
"""

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"

results = []

def check(step_name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append({"step": step_name, "status": status, "detail": detail})
    icon = "PASS" if status == PASS else "FAIL"
    print(f"  [{icon}] {step_name}: {detail}")
    return condition

def skip(step_name, reason):
    results.append({"step": step_name, "status": SKIP, "detail": reason})
    print(f"  [SKIP] {step_name}: SKIP - {reason}")

def warn(step_name, detail):
    results.append({"step": step_name, "status": WARN, "detail": detail})
    print(f"  [WARN] {step_name}: {detail}")


def main():
    # ================================================================
    # 用户输入 — 四轮双机械臂机器人
    # ================================================================
    description = "设计一个四轮底盘机器人，四个麦克纳姆轮分布在底盘四角，底盘上方安装两个3自由度机械臂，左边一个右边一个"

    print("=" * 70)
    print(f"  E2E Test: 四轮双臂机器人")
    print(f"  输入: {description}")
    print("=" * 70)

    output_dir = "data/e2e_4wheel_dual_arm"
    os.makedirs(output_dir, exist_ok=True)

    # ================================================================
    # PHASE 1: NL → Assembly（LLM从零生成）
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: 自然语言 → Assembly (LLM生成 + VLM闭环)")
    print(f"{'='*70}")

    from lang3d.tools.assembly_generator import generate_assembly_with_vlm_loop

    t0 = time.time()
    result = generate_assembly_with_vlm_loop(
        description=description,
        output_dir=output_dir,
        max_rounds=3,
    )
    dt_phase1 = time.time() - t0

    assembly = result["assembly"]
    passed = result["passed"]
    rounds = result["rounds"]

    check("vlm_loop_completed", assembly is not None,
          f"VLM闭环完成 ({dt_phase1:.1f}s), {rounds}轮, passed={passed}")

    check("vlm_passed", passed,
          f"VLM验证: {'PASSED' if passed else 'FAILED'} after {rounds} rounds")

    if not assembly:
        print("\n装配体生成失败，终止。")
        _print_summary(description, None, output_dir)
        return

    check("part_count", len(assembly.parts) >= 10,
          f"零件数: {len(assembly.parts)} (期望 >=10)")
    check("joint_count", len(assembly.joints) >= len(assembly.parts) - 2,
          f"关节数: {len(assembly.joints)} (期望 >={len(assembly.parts)-2})")

    # 检查期望零件
    part_names = {p.name for p in assembly.parts}
    wheel_parts = [n for n in part_names if "wheel" in n.lower()]
    arm_parts = [n for n in part_names if "arm" in n.lower()]
    check("has_wheels", len(wheel_parts) >= 4,
          f"轮子零件: {wheel_parts} (期望 >=4)")
    check("has_arms", len(arm_parts) >= 2,
          f"机械臂零件: {arm_parts} (期望 >=2)")

    # 打印零件清单
    print(f"\n  --- 零件清单 ({len(assembly.parts)} parts) ---")
    for p in assembly.parts:
        dims_str = ", ".join(f"{k}={v}" for k, v in p.dimensions.items()) if p.dimensions else "N/A"
        print(f"    {p.name:30s} [{p.category:12s}] {dims_str}")

    # ================================================================
    # PHASE 2: 位置求解 + 旋转数据检查
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: 装配求解器 — 位置 + 旋转数据验证")
    print(f"{'='*70}")

    positions = result.get("positions", {})

    check("all_parts_positioned",
          len(positions) == len(assembly.parts),
          f"位置数: {len(positions)}/{len(assembly.parts)}")

    # 检查旋转数据存在性
    parts_with_rotation = 0
    parts_with_nonzero_rotation = 0
    for pname, pdata in positions.items():
        rot = pdata.get("rotation")
        if rot is not None:
            parts_with_rotation += 1
            if len(rot) >= 4 and abs(rot[3]) > 1e-6:
                parts_with_nonzero_rotation += 1

    check("rotation_data_present",
          parts_with_rotation == len(positions),
          f"含旋转数据的零件: {parts_with_rotation}/{len(positions)}")

    check("nonzero_rotations",
          parts_with_nonzero_rotation > 0,
          f"含非零旋转的零件: {parts_with_nonzero_rotation} (期望 >0, 轮子应有90°旋转)")

    # 打印所有零件位置+旋转
    print(f"\n  --- 位置 + 旋转 ---")
    for pname, pdata in sorted(positions.items()):
        pos = pdata.get("position", [0, 0, 0])
        rot = pdata.get("rotation", [0, 0, 1, 0])
        is_wheel = "wheel" in pname.lower()
        marker = " <-- WHEEL" if is_wheel else ""
        print(f"    {pname:30s} pos=({pos[0]:7.1f},{pos[1]:7.1f},{pos[2]:7.1f}) "
              f"rot=[{rot[0]:.2f},{rot[1]:.2f},{rot[2]:.2f},{rot[3]:.2f}]{marker}")

    # 检查轮子旋转合理性（轮子应有90°绕某轴旋转使其竖立）
    wheel_rotation_ok = True
    for pname, pdata in positions.items():
        if "wheel" in pname.lower():
            rot = pdata.get("rotation", [0, 0, 1, 0])
            if len(rot) >= 4 and abs(rot[3]) > 1e-3:
                wheel_rotation_ok = True
                break
    else:
        wheel_rotation_ok = False

    check("wheel_has_rotation",
          wheel_rotation_ok,
          f"轮子零件旋转检查: {'有旋转' if wheel_rotation_ok else '无旋转（可能方向错误）'}")

    # NaN 检查
    nan_count = 0
    for pname, pdata in positions.items():
        pos = pdata.get("position", [0, 0, 0])
        if any(math.isnan(v) or math.isinf(v) for v in pos):
            nan_count += 1
    check("no_nan_positions", nan_count == 0, f"异常位置: {nan_count}")

    # ================================================================
    # PHASE 3: FreeCAD Script 旋转检查
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: FreeCAD Assembly Script 旋转验证")
    print(f"{'='*70}")

    # 检查 assembly_render_script.py
    script_path = Path(output_dir) / "engineering_package" / "assembly_render_script.py"
    if script_path.exists():
        script_content = script_path.read_text(encoding="utf-8")
        has_rotate = "_shape.rotate" in script_content
        rotate_count = script_content.count("_shape.rotate")

        check("freecad_has_rotate", has_rotate,
              f"FreeCAD脚本包含 rotate() 调用: {has_rotate} (共{rotate_count}处)")

        # 检查轮子的 rotate
        wheel_rotate_found = False
        lines = script_content.split("\n")
        for i, line in enumerate(lines):
            if "_shape.rotate" in line:
                # Check surrounding lines for context
                context = "\n".join(lines[max(0,i-5):i+2])
                print(f"    rotate调用: {line.strip()}")

        # 检查 rotate 在 translate 之前（按零件逐个检查）
        if has_rotate:
            rotate_before_translate_count = 0
            parts_with_rotate = 0
            # Split script by parts: each part creates _shape, then optionally rotates, then translates
            part_blocks = script_content.split("_shape = ")
            for block in part_blocks[1:]:  # skip first (before any shape creation)
                has_r = "_shape.rotate" in block
                has_t = "_shape.translate" in block
                if has_r and has_t:
                    parts_with_rotate += 1
                    r_idx = block.index("_shape.rotate")
                    t_idx = block.index("_shape.translate")
                    if r_idx < t_idx:
                        rotate_before_translate_count += 1

            check("rotate_before_translate",
                  rotate_before_translate_count == parts_with_rotate and parts_with_rotate > 0,
                  f"rotate在translate之前: {rotate_before_translate_count}/{parts_with_rotate} 零件")
    else:
        skip("freecad_script_check", f"脚本不存在: {script_path}")

    # 检查 exploded view script
    exploded_script_path = Path(output_dir) / "engineering_package" / "assembly_exploded_script.py"
    if exploded_script_path.exists():
        exploded_content = exploded_script_path.read_text(encoding="utf-8")
        check("exploded_has_rotate",
              "_shape.rotate" in exploded_content,
              f"Exploded view脚本包含 rotate() 调用: {'_shape.rotate' in exploded_content}")
    else:
        skip("exploded_script_check", "Exploded view脚本不存在")

    # ================================================================
    # PHASE 4: URDF 旋转检查
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 4: URDF 关节 Origin 旋转验证")
    print(f"{'='*70}")

    urdf_path = Path(output_dir) / "engineering_package" / "urdf.xml"
    if urdf_path.exists():
        urdf_content = urdf_path.read_text(encoding="utf-8")

        import xml.etree.ElementTree as ET
        root = ET.fromstring(urdf_content)

        # 检查所有 joint 的 rpy
        joints = root.findall(".//joint")
        nonzero_rpy_joints = []
        wheel_joints = []

        for joint_el in joints:
            origin = joint_el.find("origin")
            if origin is not None:
                rpy_str = origin.get("rpy", "0 0 0")
                xyz_str = origin.get("xyz", "0 0 0")
                rpy_parts = [float(v) for v in rpy_str.split()]
                xyz_parts = [float(v) for v in xyz_str.split()]
                joint_name = joint_el.get("name", "")
                joint_type = joint_el.get("type", "")

                has_nonzero_rpy = any(abs(v) > 1e-6 for v in rpy_parts)

                child_el = joint_el.find("child")
                child_name = child_el.get("link", "") if child_el is not None else ""

                if "wheel" in child_name.lower() or "wheel" in joint_name.lower():
                    wheel_joints.append({
                        "name": joint_name,
                        "xyz": xyz_parts,
                        "rpy": rpy_parts,
                    })

                if has_nonzero_rpy:
                    nonzero_rpy_joints.append(joint_name)

        check("urdf_has_nonzero_rpy",
              len(nonzero_rpy_joints) > 0,
              f"含非零 rpy 的关节数: {len(nonzero_rpy_joints)}/{len(joints)}")

        check("wheel_joints_found",
              len(wheel_joints) >= 2,
              f"轮子关节数: {len(wheel_joints)} (期望 >=2)")

        # 检查轮子关节的 rpy 是否非零
        wheel_joints_with_rpy = sum(
            1 for wj in wheel_joints if any(abs(v) > 1e-4 for v in wj["rpy"])
        )
        check("wheel_joint_rpy",
              wheel_joints_with_rpy > 0,
              f"轮子关节含非零 rpy: {wheel_joints_with_rpy}/{len(wheel_joints)}")

        # 打印轮子关节详情
        print(f"\n  --- 轮子关节 URDF 详情 ---")
        for wj in wheel_joints:
            rpy = wj["rpy"]
            xyz = wj["xyz"]
            print(f"    {wj['name']}")
            print(f"      xyz = ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}) m")
            print(f"      rpy = ({rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}) rad")
    else:
        skip("urdf_check", f"URDF文件不存在: {urdf_path}")

    # ================================================================
    # PHASE 5: Production Render 质量
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 5: Production Render 质量检查")
    print(f"{'='*70}")

    production_render_dir = result.get("production_render_dir", "")
    if production_render_dir and os.path.isdir(production_render_dir):
        renders = list(Path(production_render_dir).glob("*.png"))
        total_size = sum(f.stat().st_size for f in renders)
        avg_size = total_size / len(renders) if renders else 0

        check("render_count", len(renders) >= 3,
              f"Production渲染视角: {len(renders)} (期望 >=3)")
        check("render_quality", avg_size > 10000,
              f"平均大小: {avg_size/1024:.1f}KB (期望 >10KB)")

        for r in renders:
            size_kb = r.stat().st_size / 1024
            check(f"render_{r.stem}", size_kb > 5, f"{r.name}: {size_kb:.1f}KB")
    else:
        # Fallback to VLM renders
        render_dir = result.get("render_dir", "")
        if render_dir and os.path.isdir(render_dir):
            round_dirs = sorted(
                [d for d in Path(render_dir).iterdir() if d.is_dir() and d.name.startswith("round_")],
                key=lambda d: d.name,
            )
            if round_dirs:
                last_round = round_dirs[-1]
                renders = list(last_round.glob("*.png"))
                check("render_count_fallback", len(renders) >= 3,
                      f"渲染视角(VLM fallback): {len(renders)}")
        else:
            skip("render_check", "无渲染目录")

    # ================================================================
    # PHASE 6: 工程包完整性
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 6: 工程包完整性检查")
    print(f"{'='*70}")

    export_dir = result.get("export_dir", "")
    if export_dir and os.path.isdir(export_dir):
        expected_files = [
            "design_report.json",
            "bom.md",
            "assembly_guide.md",
            "urdf.xml",
            "README.md",
        ]
        for fname in expected_files:
            fpath = Path(export_dir) / fname
            exists = fpath.exists()
            size = fpath.stat().st_size if exists else 0
            check(f"pkg_{fname}", exists and size > 50,
                  f"{fname}: {'exists' if exists else 'MISSING'} ({size/1024:.1f}KB)")

        expected_dirs = {
            "freecad_scripts": "FreeCAD脚本",
            "stl_parts": "STL零件",
        }
        for dname, label in expected_dirs.items():
            dpath = Path(export_dir) / dname
            exists = dpath.exists() and dpath.is_dir()
            n_files = len(list(dpath.iterdir())) if exists else 0
            check(f"pkg_dir_{dname}", exists and n_files > 0,
                  f"{dname}/: {n_files} files ({label})")

        # design_report 检查
        report_path = Path(export_dir) / "design_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            check("report_mass", report.get("total_mass_kg", 0) > 0,
                  f"总质量: {report.get('total_mass_kg', 'N/A')} kg")
            check("report_parts", report.get("total_parts", 0) >= 10,
                  f"零件数: {report.get('total_parts', 'N/A')}")
    else:
        skip("package_check", "无工程包目录")

    # ================================================================
    # 总结
    # ================================================================
    _print_summary(description, assembly, output_dir, export_dir)


def _print_summary(description, assembly, output_dir, export_dir=""):
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")

    passed_count = sum(1 for r in results if r["status"] == PASS)
    failed_count = sum(1 for r in results if r["status"] == FAIL)
    warn_count = sum(1 for r in results if r["status"] == WARN)
    skip_count = sum(1 for r in results if r["status"] == SKIP)
    total = len(results)

    score = passed_count / total * 100 if total > 0 else 0

    print(f"\n  输入: {description}")
    if assembly:
        print(f"  装配体: {assembly.name}, {len(assembly.parts)} parts, {len(assembly.joints)} joints")
    print(f"  工程包: {export_dir or 'N/A'}")
    print(f"\n  检查总数: {total}")
    print(f"  通过: {passed_count}")
    print(f"  失败: {failed_count}")
    print(f"  警告: {warn_count}")
    print(f"  跳过: {skip_count}")
    print(f"\n  得分: {score:.1f}%")

    if failed_count > 0:
        print(f"\n  FAILED items:")
        for r in results:
            if r["status"] == FAIL:
                print(f"    FAIL {r['step']}: {r['detail']}")

    # 保存报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input": description,
        "assembly_name": assembly.name if assembly else None,
        "part_count": len(assembly.parts) if assembly else 0,
        "joint_count": len(assembly.joints) if assembly else 0,
        "score": score,
        "total_checks": total,
        "passed": passed_count,
        "failed": failed_count,
        "warnings": warn_count,
        "skipped": skip_count,
        "results": results,
    }

    report_path = os.path.join(output_dir, "e2e_4wheel_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  报告: {report_path}")
    print(f"  工程包: {export_dir}")


if __name__ == "__main__":
    main()
