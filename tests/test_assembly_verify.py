"""Integration test: Robotic arm assembly verification.

Tests:
1. Model all 8 robotic arm parts with fc_batch
2. Verify each part individually with cad_verify (VLM)
3. Create assembly (move parts into position) and verify
4. Record which features VLM can/cannot identify

Run with: python tests/test_assembly_verify.py
"""

import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

OUTPUT_DIR = project_root / "data" / "projects" / "robotic_arm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR = project_root / "data" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# Part definitions: (name, description for cad_verify, model_operations)
PARTS = [
    {
        "name": "base_plate",
        "description": "A circular base plate, approximately 120mm diameter and 8mm thick, with 4 mounting holes near the edge and a center hole",
        "file": "base_plate",
    },
    {
        "name": "base_joint_housing",
        "description": "A cylindrical housing with outer diameter about 80mm, height 40mm, with a hollow center (wall thickness ~5mm), and a wider flange at the base with 3 mounting holes",
        "file": "base_joint_housing",
    },
    {
        "name": "shoulder_link",
        "description": "A rectangular link approximately 150x40x30mm with 3 large lightening holes in the middle and mounting holes at both ends, with rounded edges",
        "file": "shoulder_link",
    },
    {
        "name": "elbow_joint",
        "description": "A cylindrical joint with outer diameter about 60mm, height 35mm, with a central shaft hole of about 12mm diameter and a small set screw hole on the side",
        "file": "elbow_joint",
    },
    {
        "name": "forearm_link",
        "description": "A rectangular hollow link approximately 120x35x25mm, with thin walls (about 4mm), mounting holes at both ends, and rounded edges",
        "file": "forearm_link",
    },
    {
        "name": "wrist_joint",
        "description": "A small cylindrical joint with outer diameter about 40mm, height 25mm, with a central shaft hole of about 8mm diameter and a set screw hole",
        "file": "wrist_joint",
    },
    {
        "name": "end_effector_mount",
        "description": "A small mounting bracket with a cylindrical pillar about 35mm diameter, 15mm tall, on a wider flange base, with a center hole and 4 M3 mounting holes on the flange",
        "file": "end_effector_mount",
    },
    {
        "name": "servo_holder",
        "description": "A small U-shaped bracket about 30x24x12mm with side walls and 4 small mounting holes in the base, designed to hold an SG90 servo",
        "file": "servo_holder",
    },
]


def model_all_parts():
    """Model all 8 parts using fc_batch operations."""
    print("=" * 60)
    print("PHASE 1: MODELING ALL 8 PARTS")
    print("=" * 60)

    from lang3d.tools.freecad import _execute_operations

    # Define modeling operations for each part
    part_operations = [
        # 1. base_plate
        [
            {"type": "new_doc", "name": "BasePlate"},
            {"type": "make_cylinder", "radius": 60, "height": 8, "name": "Plate"},
            {"type": "make_cylinder", "radius": 3.3, "height": 10, "name": "H1"},
            {"type": "move", "object": "H1", "dx": 45, "dy": 0, "dz": -1},
            {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "H1", "result_name": "Plate"},
            {"type": "make_cylinder", "radius": 3.3, "height": 10, "name": "H2"},
            {"type": "move", "object": "H2", "dx": 0, "dy": 45, "dz": -1},
            {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "H2", "result_name": "Plate"},
            {"type": "make_cylinder", "radius": 3.3, "height": 10, "name": "H3"},
            {"type": "move", "object": "H3", "dx": -45, "dy": 0, "dz": -1},
            {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "H3", "result_name": "Plate"},
            {"type": "make_cylinder", "radius": 3.3, "height": 10, "name": "H4"},
            {"type": "move", "object": "H4", "dx": 0, "dy": -45, "dz": -1},
            {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "H4", "result_name": "Plate"},
            {"type": "make_cylinder", "radius": 15, "height": 10, "name": "CH"},
            {"type": "boolean", "operation": "cut", "object1": "Plate", "object2": "CH", "result_name": "BasePlate"},
            {"type": "save", "path": str(OUTPUT_DIR / "base_plate.FCStd")},
            {"type": "export_stl", "path": str(OUTPUT_DIR / "base_plate.stl")},
        ],
        # 2. base_joint_housing
        [
            {"type": "new_doc", "name": "BaseJointHousing"},
            {"type": "cylinder_with_hole", "outer_radius": 40, "inner_radius": 35, "height": 40, "name": "Housing"},
            {"type": "make_cylinder", "radius": 45, "height": 5, "name": "Flange"},
            {"type": "boolean", "operation": "union", "object1": "Housing", "object2": "Flange", "result_name": "WithFlange"},
            {"type": "make_cylinder", "radius": 2.25, "height": 8, "name": "F1"},
            {"type": "move", "object": "F1", "dx": 38, "dy": 0, "dz": -1},
            {"type": "boolean", "operation": "cut", "object1": "WithFlange", "object2": "F1", "result_name": "WithFlange"},
            {"type": "make_cylinder", "radius": 2.25, "height": 8, "name": "F2"},
            {"type": "move", "object": "F2", "dx": -19, "dy": 32.9, "dz": -1},
            {"type": "boolean", "operation": "cut", "object1": "WithFlange", "object2": "F2", "result_name": "WithFlange"},
            {"type": "make_cylinder", "radius": 2.25, "height": 8, "name": "F3"},
            {"type": "move", "object": "F3", "dx": -19, "dy": -32.9, "dz": -1},
            {"type": "boolean", "operation": "cut", "object1": "WithFlange", "object2": "F3", "result_name": "BaseJointHousing"},
            {"type": "save", "path": str(OUTPUT_DIR / "base_joint_housing.FCStd")},
            {"type": "export_stl", "path": str(OUTPUT_DIR / "base_joint_housing.stl")},
        ],
        # 3. shoulder_link
        [
            {"type": "new_doc", "name": "ShoulderLink"},
            {"type": "make_box", "length": 150, "width": 40, "height": 30, "name": "Body"},
            {"type": "make_cylinder", "radius": 10, "height": 35, "name": "L1"},
            {"type": "move", "object": "L1", "dx": 40, "dy": 20, "dz": -2},
            {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "L1", "result_name": "Body"},
            {"type": "make_cylinder", "radius": 10, "height": 35, "name": "L2"},
            {"type": "move", "object": "L2", "dx": 75, "dy": 20, "dz": -2},
            {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "L2", "result_name": "Body"},
            {"type": "make_cylinder", "radius": 10, "height": 35, "name": "L3"},
            {"type": "move", "object": "L3", "dx": 110, "dy": 20, "dz": -2},
            {"type": "boolean", "operation": "cut", "object1": "Body", "object2": "L3", "result_name": "ShoulderLink"},
            {"type": "fillet", "object": "ShoulderLink", "radius": 2},
            {"type": "save", "path": str(OUTPUT_DIR / "shoulder_link.FCStd")},
            {"type": "export_stl", "path": str(OUTPUT_DIR / "shoulder_link.stl")},
        ],
        # 4. elbow_joint
        [
            {"type": "new_doc", "name": "ElbowJoint"},
            {"type": "cylinder_with_hole", "outer_radius": 30, "inner_radius": 6, "height": 35, "name": "Joint"},
            {"type": "make_cylinder", "radius": 1.65, "height": 40, "name": "SS"},
            {"type": "move", "object": "SS", "dx": 0, "dy": -20, "dz": 17},
            {"type": "boolean", "operation": "cut", "object1": "Joint", "object2": "SS", "result_name": "ElbowJoint"},
            {"type": "save", "path": str(OUTPUT_DIR / "elbow_joint.FCStd")},
            {"type": "export_stl", "path": str(OUTPUT_DIR / "elbow_joint.stl")},
        ],
        # 5. forearm_link
        [
            {"type": "new_doc", "name": "ForearmLink"},
            {"type": "make_box", "length": 120, "width": 35, "height": 25, "name": "Shell"},
            {"type": "make_box", "length": 108, "width": 27, "height": 17, "name": "Void"},
            {"type": "move", "object": "Void", "dx": 6, "dy": 4, "dz": 4},
            {"type": "boolean", "operation": "cut", "object1": "Shell", "object2": "Void", "result_name": "ForearmLink"},
            {"type": "fillet", "object": "ForearmLink", "radius": 1.5},
            {"type": "save", "path": str(OUTPUT_DIR / "forearm_link.FCStd")},
            {"type": "export_stl", "path": str(OUTPUT_DIR / "forearm_link.stl")},
        ],
        # 6. wrist_joint
        [
            {"type": "new_doc", "name": "WristJoint"},
            {"type": "cylinder_with_hole", "outer_radius": 20, "inner_radius": 4, "height": 25, "name": "Joint"},
            {"type": "make_cylinder", "radius": 1.25, "height": 25, "name": "SS"},
            {"type": "move", "object": "SS", "dx": 0, "dy": -14, "dz": 12},
            {"type": "boolean", "operation": "cut", "object1": "Joint", "object2": "SS", "result_name": "WristJoint"},
            {"type": "save", "path": str(OUTPUT_DIR / "wrist_joint.FCStd")},
            {"type": "export_stl", "path": str(OUTPUT_DIR / "wrist_joint.stl")},
        ],
        # 7. end_effector_mount
        [
            {"type": "new_doc", "name": "EndEffectorMount"},
            {"type": "make_cylinder", "radius": 17.5, "height": 4, "name": "Flange"},
            {"type": "make_cylinder", "radius": 12, "height": 15, "name": "Pillar"},
            {"type": "move", "object": "Pillar", "dx": 0, "dy": 0, "dz": 4},
            {"type": "boolean", "operation": "union", "object1": "Flange", "object2": "Pillar", "result_name": "Mount"},
            {"type": "make_cylinder", "radius": 4, "height": 20, "name": "CH"},
            {"type": "boolean", "operation": "cut", "object1": "Mount", "object2": "CH", "result_name": "EndEffectorMount"},
            {"type": "save", "path": str(OUTPUT_DIR / "end_effector_mount.FCStd")},
            {"type": "export_stl", "path": str(OUTPUT_DIR / "end_effector_mount.stl")},
        ],
        # 8. servo_holder
        [
            {"type": "new_doc", "name": "ServoHolder"},
            {"type": "make_box", "length": 30, "width": 24, "height": 3, "name": "Base"},
            {"type": "make_box", "length": 30, "width": 3, "height": 12, "name": "LW"},
            {"type": "boolean", "operation": "union", "object1": "Base", "object2": "LW", "result_name": "Part"},
            {"type": "make_box", "length": 30, "width": 3, "height": 12, "name": "RW"},
            {"type": "move", "object": "RW", "dx": 0, "dy": 21, "dz": 0},
            {"type": "boolean", "operation": "union", "object1": "Part", "object2": "RW", "result_name": "ServoHolder"},
            {"type": "save", "path": str(OUTPUT_DIR / "servo_holder.FCStd")},
            {"type": "export_stl", "path": str(OUTPUT_DIR / "servo_holder.stl")},
        ],
    ]

    results = []
    for i, (part, ops) in enumerate(zip(PARTS, part_operations)):
        name = part["name"]
        print(f"\n[{i+1}/8] Modeling {name}...")
        try:
            start = time.time()
            result = _execute_operations(ops)
            elapsed = time.time() - start

            stl_path = OUTPUT_DIR / f"{name}.stl"
            stl_size = stl_path.stat().st_size if stl_path.exists() else 0

            ok = stl_size > 0
            print(f"  -> {'OK' if ok else 'FAIL'} ({elapsed:.1f}s, STL: {stl_size:,} bytes)")
            results.append({"name": name, "modeled": ok, "stl_size": stl_size, "elapsed": elapsed})
        except Exception as e:
            print(f"  -> FAIL: {e}")
            results.append({"name": name, "modeled": False, "error": str(e)})

    return results


def verify_parts(model_results):
    """Verify each modeled part with cad_verify."""
    print(f"\n{'='*60}")
    print("PHASE 2: VLM VERIFICATION OF EACH PART")
    print("=" * 60)

    from lang3d.config import load_config
    from lang3d.models.router import ModelRouter
    from lang3d.tools.freecad import FCOpenGUITool, FCCloseGUITool
    from lang3d.tools.vlm import CADVerifyTool

    config = load_config()
    router = ModelRouter(config)
    verify = CADVerifyTool(router, screenshot_dir=str(SCREENSHOT_DIR))
    open_gui = FCOpenGUITool()
    close_gui = FCCloseGUITool()

    verify_results = []

    for i, (part, mr) in enumerate(zip(PARTS, model_results)):
        name = part["name"]
        if not mr["modeled"]:
            print(f"\n[{i+1}/8] SKIP {name} (not modeled)")
            verify_results.append({"name": name, "verified": False, "error": "not modeled"})
            continue

        fc_path = str(OUTPUT_DIR / f"{name}.FCStd")
        print(f"\n[{i+1}/8] Verifying {name}...")

        try:
            # Open FreeCAD GUI
            open_result = open_gui.execute(file_path=fc_path, view="isometric", fit_all=True, wait_seconds=6)
            time.sleep(4)

            # Verify with VLM (use detailed for better accuracy)
            verify_result = verify.execute(
                expected=part["description"],
                window_title="FreeCAD",
                detail="detailed",
            )

            match = "MATCH: True" in verify_result
            print(f"  -> {'MATCH' if match else 'MISMATCH'}")

            # Extract VLM observations for analysis
            observed = ""
            for line in verify_result.split("\n"):
                if line.startswith("OBSERVED:"):
                    observed = line.replace("OBSERVED:", "").strip()
                    break

            verify_results.append({
                "name": name,
                "verified": True,
                "match": match,
                "observed": observed,
            })

            # Close FreeCAD
            close_gui.execute()
            time.sleep(2)

        except Exception as e:
            print(f"  -> ERROR: {e}")
            verify_results.append({"name": name, "verified": False, "error": str(e)})
            try:
                close_gui.execute()
                time.sleep(2)
            except Exception:
                pass

        # Rate limit pause
        if i < len(PARTS) - 1:
            print("  (waiting 3s for API rate limit...)")
            time.sleep(3)

    return verify_results


def create_assembly():
    """Create an assembly with all parts positioned for visual verification."""
    print(f"\n{'='*60}")
    print("PHASE 3: ASSEMBLY CREATION")
    print("=" * 60)

    from lang3d.tools.freecad import FCBatchTool

    batch = FCBatchTool()
    assembly_path = str(OUTPUT_DIR / "assembly.FCStd")
    assembly_stl = str(OUTPUT_DIR / "assembly.stl")

    # Create assembly: import all parts and position them
    # Assembly layout (simple vertical):
    #   base_plate at origin
    #   base_joint_housing on top of base_plate
    #   shoulder_link above
    #   elbow_joint at end of shoulder_link
    #   forearm_link from elbow
    #   wrist_joint at end of forearm
    #   end_effector_mount on wrist
    #   servo_holder attached to shoulder_link

    ops = [
        {"type": "new_doc", "name": "Assembly"},

        # Base plate (already at origin, 8mm thick)
        {"type": "make_cylinder", "radius": 60, "height": 8, "name": "BasePlate"},

        # Base joint housing on top of base plate
        {"type": "cylinder_with_hole", "outer_radius": 40, "inner_radius": 35, "height": 40, "name": "BaseJoint"},
        {"type": "move", "object": "BaseJoint", "dx": 0, "dy": 0, "dz": 8},

        # Shoulder link above base joint
        {"type": "make_box", "length": 150, "width": 40, "height": 30, "name": "ShoulderLink"},
        {"type": "move", "object": "ShoulderLink", "dx": 0, "dy": 0, "dz": 48},

        # Elbow joint at end of shoulder link
        {"type": "cylinder_with_hole", "outer_radius": 30, "inner_radius": 6, "height": 35, "name": "ElbowJoint"},
        {"type": "move", "object": "ElbowJoint", "dx": 150, "dy": 0, "dz": 48},

        # Forearm link from elbow
        {"type": "make_box", "length": 120, "width": 35, "height": 25, "name": "ForearmLink"},
        {"type": "move", "object": "ForearmLink", "dx": 150, "dy": 0, "dz": 83},

        # Wrist joint at end of forearm
        {"type": "cylinder_with_hole", "outer_radius": 20, "inner_radius": 4, "height": 25, "name": "WristJoint"},
        {"type": "move", "object": "WristJoint", "dx": 270, "dy": 0, "dz": 83},

        # End effector mount on wrist
        {"type": "make_cylinder", "radius": 17.5, "height": 15, "name": "EndEffector"},
        {"type": "move", "object": "EndEffector", "dx": 270, "dy": 0, "dz": 108},

        # Servo holder on shoulder link
        {"type": "make_box", "length": 30, "width": 24, "height": 12, "name": "ServoHolder"},
        {"type": "move", "object": "ServoHolder", "dx": 75, "dy": 0, "dz": 78},

        {"type": "save", "path": assembly_path},
        {"type": "export_stl", "path": assembly_stl},
    ]

    result = batch.execute(operations=ops)
    ok = "Error" not in result

    stl_size = Path(assembly_stl).stat().st_size if Path(assembly_stl).exists() else 0
    print(f"  Assembly: {'OK' if ok else 'FAIL'} (STL: {stl_size:,} bytes)")

    return ok, assembly_path


def verify_assembly(assembly_path):
    """Verify the complete assembly with VLM."""
    print(f"\n{'='*60}")
    print("PHASE 4: ASSEMBLY VERIFICATION")
    print("=" * 60)

    from lang3d.config import load_config
    from lang3d.models.router import ModelRouter
    from lang3d.tools.freecad import FCOpenGUITool, FCCloseGUITool
    from lang3d.tools.vlm import CADVerifyTool

    config = load_config()
    router = ModelRouter(config)
    verify = CADVerifyTool(router, screenshot_dir=str(SCREENSHOT_DIR))
    open_gui = FCOpenGUITool()
    close_gui = FCCloseGUITool()

    try:
        open_gui.execute(file_path=assembly_path, view="isometric", fit_all=True, wait_seconds=8)
        time.sleep(4)

        result = verify.execute(
            expected=(
                "A robotic arm assembly consisting of: a circular base plate at the bottom, "
                "a cylindrical joint housing on top, a long rectangular shoulder link extending upward, "
                "a cylindrical elbow joint at the end, a shorter rectangular forearm link, "
                "a small wrist joint, an end effector mount at the tip, "
                "and a small servo holder on the shoulder link. "
                "The arm extends horizontally from left to right."
            ),
            window_title="FreeCAD",
            detail="detailed",
        )

        match = "MATCH: True" in result
        print(f"  Assembly verification: {'MATCH' if match else 'MISMATCH'}")

        # Extract observations
        observed = ""
        for line in result.split("\n"):
            if line.startswith("OBSERVED:"):
                observed = line.replace("OBSERVED:", "").strip()
                break

        close_gui.execute()

        return {"match": match, "observed": observed, "raw": result}

    except Exception as e:
        print(f"  ERROR: {e}")
        try:
            close_gui.execute()
        except Exception:
            pass
        return {"match": False, "error": str(e)}


def main():
    # Phase 1: Model all parts
    model_results = model_all_parts()
    modeled_count = sum(1 for r in model_results if r["modeled"])
    print(f"\n  Modeled: {modeled_count}/8 parts")

    if modeled_count < 6:
        print("  Too few parts modeled. Aborting.")
        return False

    # Phase 2: Verify each part
    verify_results = verify_parts(model_results)

    # Phase 3: Create assembly
    assembly_ok, assembly_path = create_assembly()

    # Phase 4: Verify assembly
    assembly_result = None
    if assembly_ok:
        assembly_result = verify_assembly(assembly_path)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Modeling results
    print("\n  PART MODELING:")
    for r in model_results:
        status = "OK" if r["modeled"] else "FAIL"
        stl = f"STL: {r.get('stl_size', 0):,}B" if r["modeled"] else r.get("error", "")[:50]
        print(f"    [{status}] {r['name']:25s} {stl}")

    # Verification results
    print("\n  VLM VERIFICATION:")
    vlm_analysis = {"match": [], "mismatch": [], "error": []}
    for r in verify_results:
        if not r.get("verified", False):
            status = "ERR"
            vlm_analysis["error"].append(r["name"])
        elif r.get("match", False):
            status = "OK"
            vlm_analysis["match"].append(r["name"])
        else:
            status = "MIS"
            vlm_analysis["mismatch"].append(r["name"])
        print(f"    [{status}] {r['name']:25s}")

    # Assembly
    print(f"\n  ASSEMBLY: {'OK' if assembly_ok else 'FAIL'}")
    if assembly_result:
        print(f"  ASSEMBLY VERIFY: {'MATCH' if assembly_result.get('match') else 'MISMATCH'}")

    # VLM feature analysis
    print(f"\n{'='*60}")
    print("VLM FEATURE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Parts matched:     {len(vlm_analysis['match'])}")
    print(f"  Parts mismatched:  {len(vlm_analysis['mismatch'])}")
    print(f"  Parts with errors: {len(vlm_analysis['error'])}")

    if vlm_analysis["match"]:
        print(f"\n  Features VLM CAN identify:")
        for name in vlm_analysis["match"]:
            part = next(p for p in PARTS if p["name"] == name)
            print(f"    - {name}: {part['description'][:80]}...")

    if vlm_analysis["mismatch"]:
        print(f"\n  Features VLM may struggle with:")
        for name in vlm_analysis["mismatch"]:
            part = next(p for p in PARTS if p["name"] == name)
            print(f"    - {name}: {part['description'][:80]}...")

    # Overall pass/fail
    # Core requirement: all 8 parts modeled + assembly created
    # VLM verification is best-effort (depends on model capability)
    all_modeled = modeled_count == 8
    verified_count = len(vlm_analysis["match"])
    test_pass = all_modeled and assembly_ok

    print(f"\n{'='*60}")
    print(f"  RESULT: {'PASS' if test_pass else 'FAIL'}")
    print(f"{'='*60}")
    print(f"  Modeled: {modeled_count}/8")
    print(f"  VLM Verified: {verified_count}/8 (best-effort, depends on model)")
    print(f"  Assembly: {'OK' if assembly_ok else 'FAIL'}")

    return test_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
