"""Natural language → Assembly generation tool.

Uses LLM to generate structured Assembly definitions (Part[] + Joint[])
from natural language descriptions.  This is the critical missing layer
between the agent's text planning and the concrete assembly pipeline.

The generated JSON can be consumed by export_package's _resolve_assembly_input()
and fed into the full pipeline: solver → FreeCAD → STL → VLM verification → export.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any

from ..knowledge.mechanics import Assembly, ConnectionMethod, Joint, Part
from ..knowledge.fastener_catalog import get_torque
from ..knowledge.arm_topology import build_arm_example, parse_dof, zigzag_angles
from ..knowledge.mobile_base_gen import build_wheeled_base, parse_drive_type
from .assembly_solver import ANCHOR_DIM_KEYS
from .base import Tool, ToolDefinition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for assembly generation
# ---------------------------------------------------------------------------

ASSEMBLY_GEN_SYSTEM_PROMPT = """\
You are an expert mechanical design engineer.  Given a natural language
description of a robot, you output a complete assembly definition as JSON.

## Output Format

Return ONLY a JSON object with this exact structure:

{
  "name": "robot_name (snake_case)",
  "description": "one-line description",
  "default_angles": {"joint_child_name": angle_degrees},
  "parts": [
    {
      "name": "part_name (snake_case)",
      "category": "structural|actuator|sensor|electronics|mechanical",
      "description": "中文简要说明",
      "material": "Aluminum|PLA|ABS|Steel|Carbon Fiber|PCB",
      "dimensions": {"length": mm, "width": mm, "height": mm}
    }
  ],
  "joints": [
    {
      "type": "fixed|revolute|prismatic",
      "parent": "parent_part_name",
      "child": "child_part_name",
      "range_deg": [min, max],
      "description": "关节说明",
      "parent_anchor": "top|bottom|left|right|front|back",
      "child_anchor": "top|bottom|left|right|front|back",
      "axis": "x|y|z",
      "offset": [dx, dy, dz],
      "distribution_group": "group_name",
      "no_distribute": false,
      "connection_method": "bolted|press_fit|snap_fit|adhesive|welded|magnetic",
      "connection_detail": {
        "bolt_size": "M3|M4|M5|M6",
        "bolt_count": 2
      }
    }
  ]
}

### Connection Method Guidance (per joint type)
- **fixed** joints (housing->motor, plate->bracket, standoffs): use "bolted"
  with connection_detail.bolt_size ("M3"/"M4") and bolt_count.
- **revolute** joints (rotation between two parts, e.g. arm segment to arm
  segment via a motor/bearing assembly): use "bolted" for structural mounting
  of the motor housing. The system automatically adds a central bearing bore
  (10mm) so the shaft can pass through. Do NOT use "press_fit" unless both
  parts are cylindrical bearings with matching bore diameters.
- **prismatic** joints (sliding gripper fingers on a rail): OMIT
  connection_method entirely. A sliding interface is not a fastening; leaving
  it null is correct.
Only set connection_method where it is physically meaningful.

## Dimension Rules

- Box parts: {"length": Xmm, "width": Ymm, "height": Zmm}
- Cylindrical parts (wheels, shafts, motors): {"diameter": Dmm, "height": Hmm}
  or {"outer_diameter": Dmm, "height": Hmm} for hollow cylinders

## Engineering Guidelines

### Part Classification Rules (CRITICAL)

Parts fall into two classes with DIFFERENT dimension rules:

**Functional Parts (motors, servos, sensors, bearings) — dimensions are FIXED:**
- SG90 servo: 22.2×11.8×31.0mm, shaft Ø4.6mm, shaft length 5mm
- MG996R servo: 40.7×19.7×42.9mm, shaft Ø5.8mm, shaft length 6mm
- NEMA17 stepper: 42.3×42.3×40mm, shaft Ø5mm, shaft length 24mm
- 608 bearing: Ø22×7mm (ID=8mm)
- 625 bearing: Ø16×5mm (ID=5mm)
- DO NOT invent dimensions for these parts. Use EXACT real-world specs.
- category should be "actuator" for motors/servos, "bearing" for bearings

**Structural Parts (brackets, plates, links, housings) — dimensions are PARAMETRIC:**
- Can be freely scaled within reasonable engineering limits
- Base plate: 200-400mm long, 150-250mm wide, 3-8mm thick, Aluminum
- Wheels: 50-100mm diameter, 20-40mm wide, rubber/PLA
- Standoffs: M3/M4, 6-8mm diameter, 40-60mm tall
- Arm links: 25-40mm wide, 15-25mm high, 60-120mm long
- category should be "structural" for custom parts

### General Guidelines

1. **Base plate**: 200-400mm long, 150-250mm wide, 3-8mm thick, Aluminum
2. **Wheels**: 50-100mm diameter, 20-40mm wide, rubber/PLA
3. **Motors**: match wheel size, ~40x40x30mm for standard TT motors
4. **Standoffs**: M3/M4, 6-8mm diameter, 40-60mm tall
5. **Arm links**: 25-40mm wide, 15-25mm high, 60-120mm long
6. **Arm joints**: revolute with ±180° range. Axis convention (see "ARM ANCHORS & AXIS" rule 12 for details): pitch joints (shoulder/elbow/wrist up-down bend) use axis="x"; base yaw uses axis="z"; wrist roll uses axis="y". NEVER use axis="y" for a front/back pitch joint.
7. **Sensors**: small mounts, 20-40mm range

## Joint Anchor Semantics

- parent_anchor: which face of the PARENT the joint connects to
- child_anchor: which face of the CHILD connects to the joint
- "top" = +Z face, "bottom" = -Z face
- "front" = -Y face, "back" = +Y face (Y-up convention)
- "left" = -X face, "right" = +X face

## Critical Rules

1. EVERY part must be connected via joints (no floating parts)
2. The first joint must connect to the base_plate (root of the tree)
3. **JOINT COUNT must equal PART COUNT minus 1** (N parts needs N-1 joints for a connected tree)
4. Part names must be unique and match exactly in joints
5. Wheels attach to motors (revolute, axis=y for horizontal axle)
6. Arms attach to top plate (revolute, axis=z for shoulder yaw rotation)
7. Use distribution_group to group symmetric siblings (e.g., "arms", "standoffs", "wheels_fl_fr")
8. Use offset for parts that need to be shifted from their anchor position
9. The assembly must form a connected tree (no cycles)
10. **DOUBLE CHECK**: count joints, if fewer than parts-1, add missing joints
11. **ARM TOPOLOGY**: Parts MUST alternate joint→link→joint→link (never link→link directly)
12. **ARM ANCHORS & AXIS (CLEAN CHAIN CONVENTION)**: The arm extends
    HORIZONTALLY along the parts' `length` axis so each link's `length`
    dimension is exactly the axis-to-axis distance the IK solver needs.
    - **Base rotation (yaw)**: the ONLY top/bottom joint. Use
      `parent_anchor="top" / child_anchor="bottom"` with `axis="z"` — the
      base sits on the plate and spins about the vertical.
    - **Every arm-segment joint** (servo→link, link→servo, link→gripper):
      use `parent_anchor="front" / child_anchor="back"` so the part's
      `length` dimension participates in positioning the next axis.
    - **Pitch joints** (shoulder/elbow/wrist up-down bending): `axis="x"`.
      X is perpendicular to the Y-extending arm direction, so rotation
      produces vertical bending. **CRITICAL: NEVER use axis="y" for a
      front/back pitch joint** — Y is parallel to the arm direction, so
      rotation about Y produces NO vertical displacement and the arm cannot
      reach targets.
    - **Wrist roll** (spinning the end effector about the arm axis): use
      `parent_anchor="front" / child_anchor="back"` with `axis="y"` — Y is
      along the horizontal arm direction, so this rolls the gripper.
13. **ARM DEFAULT ANGLES**: Provide non-zero default_angles (e.g., -45, -30, 15) so the arm
    has a bent posture instead of a straight horizontal rod. Zero pitch angles would collapse
    every link into a single straight line with no vertical extent — always bend the
    pitch joints (`axis="x"`).
14. **WHEEL ORIENTATION**: Wheels MUST use child_anchor='center', axis='y',
    parent_anchor='left' or 'right'. This ensures correct cylinder orientation.
15. **WHEEL-MOTOR CHAIN**: Always use base_plate→motor→wheel topology.
    Never attach wheels directly to base_plate.
16. **NO WHEELS IN ARMS**: Robotic arms (N-DOF arm, robotic arm, 机械臂) must NOT include
    wheel parts. An arm has: base_plate, joints (support/housing), links (连杆), and end_effector.
    NEVER generate "wheel" parts for arm assemblies.
17. **ARM LINK DIMENSIONS**: Link parts must have length 60-200mm, width 20-50mm, height 12-30mm.
    End effectors must be 20-60mm in each dimension. Never create parts with dimensions > 300mm.
18. **ARM CHAIN ANCHORS (FRONT/BACK)**: EVERY joint along the arm chain
    (base→housing, housing→link, link→joint, joint→link, link→gripper) MUST use
    `parent_anchor="front" / child_anchor="back"` so the link parts extend
    along their `length` dimension and the solver stacks axes at the correct
    pitch points. Combined with non-zero default_angles on the pitch joints
    (`axis="x"`), the arm bends into a natural 3D posture. The ONLY
    top/bottom joint in the arm is the base yaw (`axis="z"`).
    Exceptions (keep as-is): motor mounts inside housings
    (`parent_anchor="back" / child_anchor="front"`, type="fixed"), bearing
    press-fits (center/center), and prismatic gripper fingers (handled
    separately). Wrist roll keeps front/back but uses `axis="y"`.
19. **GRIPPER DECOMPOSITION (CRITICAL)**: End-effectors/grippers MUST be decomposed into
    4 separate parts: gripper_servo + gripper_base + gripper_finger_left + gripper_finger_right.
    The gripper must be VISUALLY DISTINCT from arm links: wider, taller, and shorter in the
    arm direction so it reads as a gripper block, not another link segment.
    - gripper_servo (category: actuator): Small SG90-style servo motor that DRIVES the
      fingers. Dimensions: {"length": 23, "width": 12, "height": 22}. Mounted on TOP of
      gripper_base (fixed joint, parent_anchor="top", child_anchor="bottom") so it is
      visually prominent and clearly identifies the gripper as servo-actuated.
    - gripper_base (category: mechanical): Mounts the servo and guides the fingers on
      linear rails. Dimensions: {"length": 28, "width": 50, "height": 32}. The base must
      be WIDER (50mm) and TALLER (32mm) than arm links, but SHORTER in the arm direction
      (28mm) so it looks like a blocky gripper housing, not a flat arm segment. Must have
      visible features: servo cavity on top, 2 parallel rail grooves on front face for
      finger sliding, M3 mounting holes.
    - gripper_finger_left attaches to gripper_base via a prismatic joint
      (axis="x", offset=[-16,0,0], range_deg=[-8,8]) so it slides left to open.
      Dimensions: {"length": 60, "width": 10, "height": 28}. Fingers must be LONG (60mm
      forward extension, clearly protruding past the base) and TALL (28mm) so they are
      visually prominent. Must have L-shaped tip and rail tab that fits into the rail groove.
    - gripper_finger_right attaches to gripper_base via a prismatic joint
      (axis="x", offset=[16,0,0], range_deg=[-8,8]) so it slides right to open.
      Dimensions: {"length": 60, "width": 10, "height": 28}. Must have L-shaped tip
      and rail tab. MUST specify mimic_joint="gripper_finger_left" with
      mimic_multiplier=-1 so the two fingers open/close symmetrically (coupled motion).
      The ±16mm offset creates a ~22mm visible gap between inner finger faces — this gap
      is what makes the gripper look like a real two-finger gripper.
    - gripper_base attaches to wrist_link via a fixed joint (front→back).
    The gripper MUST be a real, functional mechanism: the prismatic joints must have
    correct axis and limits so fingers actually slide open/close in simulation. The
    servo on top makes the actuation visible and realistic. NEVER model a gripper as
    a single fused "end_effector" part.
20. **DOF CORRECTNESS**: N自由度 (N-DOF) = the number of revolute joints in the arm chain
    (NOT counting gripper prismatic joints). A motor housing→output link joint (e.g.
    elbow_joint→elbow_link) is a structural mount and MUST be "fixed", NEVER "revolute" —
    a revolute there would create a phantom second elbow DOF. Typical 4-DOF topology:
    base→shoulder_servo=revolute Z, shoulder_servo→shoulder_link=revolute X,
    shoulder_link→elbow_servo=revolute X, wrist_servo→wrist_link=revolute Y, with all
    intermediate motor-housing mounts (elbow_servo→elbow_link, elbow_link→wrist_servo) as fixed.

## Connection Methods (physical joining)

Every joint should specify how the parts are physically connected:
- **bolted** (most common): structural plates, motor mounts, sensor brackets
  → Requires bolt_size (M3/M4/M5/M6) and bolt_count (typically 2-4)
- **press_fit**: bearings into housings, shafts into hubs
  → No additional fasteners needed
- **snap_fit**: 3D printed enclosures, battery doors, cable clips
  → Integrated into part geometry, no fasteners
- **adhesive**: bonding panels, permanent joins
  → Specify adhesive_type (epoxy/cyanoacrylate)
- **welded**: permanent structural frames (steel/aluminum)
  → Specify weld_type (fillet/butt/spot)

Default: structural→bolted, bearing→press_fit, servo→bolted, wheel→press_fit
"""

# ---------------------------------------------------------------------------
# Few-shot examples (compressed for prompt efficiency)
# ---------------------------------------------------------------------------

EXAMPLE_4W_ROBOT = """\
{
  "name": "4w_diff_robot",
  "description": "4轮差速移动底盘",
  "default_angles": {},
  "parts": [
    {"name": "base_plate", "category": "structural", "description": "主底盘板", "material": "Aluminum", "dimensions": {"length": 300, "width": 200, "height": 5}},
    {"name": "top_plate", "category": "structural", "description": "上盖板", "material": "Aluminum", "dimensions": {"length": 280, "width": 180, "height": 3}},
    {"name": "standoff_fl", "category": "structural", "description": "前左铜柱", "material": "Steel", "dimensions": {"length": 8, "diameter": 6, "height": 50}},
    {"name": "standoff_fr", "category": "structural", "description": "前右铜柱", "material": "Steel", "dimensions": {"length": 8, "diameter": 6, "height": 50}},
    {"name": "standoff_rl", "category": "structural", "description": "后左铜柱", "material": "Steel", "dimensions": {"length": 8, "diameter": 6, "height": 50}},
    {"name": "standoff_rr", "category": "structural", "description": "后右铜柱", "material": "Steel", "dimensions": {"length": 8, "diameter": 6, "height": 50}},
    {"name": "motor_fl", "category": "actuator", "description": "前左驱动电机", "material": "Steel", "dimensions": {"length": 40, "width": 30, "height": 25}},
    {"name": "motor_fr", "category": "actuator", "description": "前右驱动电机", "material": "Steel", "dimensions": {"length": 40, "width": 30, "height": 25}},
    {"name": "motor_rl", "category": "actuator", "description": "后左驱动电机", "material": "Steel", "dimensions": {"length": 40, "width": 30, "height": 25}},
    {"name": "motor_rr", "category": "actuator", "description": "后右驱动电机", "material": "Steel", "dimensions": {"length": 40, "width": 30, "height": 25}},
    {"name": "wheel_fl", "category": "mechanical", "description": "前左轮", "material": "Rubber", "dimensions": {"diameter": 65, "height": 26}},
    {"name": "wheel_fr", "category": "mechanical", "description": "前右轮", "material": "Rubber", "dimensions": {"diameter": 65, "height": 26}},
    {"name": "wheel_rl", "category": "mechanical", "description": "后左轮", "material": "Rubber", "dimensions": {"diameter": 65, "height": 26}},
    {"name": "wheel_rr", "category": "mechanical", "description": "后右轮", "material": "Rubber", "dimensions": {"diameter": 65, "height": 26}},
    {"name": "battery_box", "category": "electronics", "description": "电池盒", "material": "PLA", "dimensions": {"length": 150, "width": 60, "height": 30}}
  ],
  "joints": [
    {"type": "fixed", "parent": "base_plate", "child": "standoff_fl", "parent_anchor": "top", "child_anchor": "bottom", "distribution_group": "standoffs"},
    {"type": "fixed", "parent": "base_plate", "child": "standoff_fr", "parent_anchor": "top", "child_anchor": "bottom", "distribution_group": "standoffs"},
    {"type": "fixed", "parent": "base_plate", "child": "standoff_rl", "parent_anchor": "top", "child_anchor": "bottom", "distribution_group": "standoffs"},
    {"type": "fixed", "parent": "base_plate", "child": "standoff_rr", "parent_anchor": "top", "child_anchor": "bottom", "distribution_group": "standoffs"},
    {"type": "fixed", "parent": "standoff_fl", "child": "top_plate", "parent_anchor": "top", "child_anchor": "bottom", "no_distribute": true},
    {"type": "fixed", "parent": "base_plate", "child": "motor_fl", "parent_anchor": "bottom", "child_anchor": "top", "distribution_group": "motors"},
    {"type": "fixed", "parent": "base_plate", "child": "motor_fr", "parent_anchor": "bottom", "child_anchor": "top", "distribution_group": "motors"},
    {"type": "fixed", "parent": "base_plate", "child": "motor_rl", "parent_anchor": "bottom", "child_anchor": "top", "distribution_group": "motors"},
    {"type": "fixed", "parent": "base_plate", "child": "motor_rr", "parent_anchor": "bottom", "child_anchor": "top", "distribution_group": "motors"},
    {"type": "revolute", "parent": "motor_fl", "child": "wheel_fl", "axis": "y", "range_deg": [-360, 360], "parent_anchor": "left", "child_anchor": "center", "no_distribute": true},
    {"type": "revolute", "parent": "motor_fr", "child": "wheel_fr", "axis": "y", "range_deg": [-360, 360], "parent_anchor": "right", "child_anchor": "center", "no_distribute": true},
    {"type": "revolute", "parent": "motor_rl", "child": "wheel_rl", "axis": "y", "range_deg": [-360, 360], "parent_anchor": "left", "child_anchor": "center", "no_distribute": true},
    {"type": "revolute", "parent": "motor_rr", "child": "wheel_rr", "axis": "y", "range_deg": [-360, 360], "parent_anchor": "right", "child_anchor": "center", "no_distribute": true},
    {"type": "fixed", "parent": "base_plate", "child": "battery_box", "parent_anchor": "top", "child_anchor": "bottom", "no_distribute": true}
  ]
}
"""

EXAMPLE_DUAL_ARM = """
Extra parts to add for dual-arm configuration:

Parts:
  arm_l_base, arm_r_base (structural, 40x40x15mm, Aluminum)
  arm_l_shoulder, arm_r_shoulder (actuator, servo MG996R, 40x20x38mm)
  arm_l_upper_link, arm_r_upper_link (structural, 100x25x15mm, Aluminum)
  arm_l_elbow, arm_r_elbow (actuator, servo, 40x20x38mm)
  arm_l_forearm, arm_r_forearm (structural, 80x25x12mm, Aluminum)
  arm_l_wrist, arm_r_wrist (actuator, servo, 28x20x28mm)
  arm_l_gripper, arm_r_gripper (mechanical, 60x20x15mm, PLA)

Joints:
  top_plate → arm_l_base (fixed, offset=[0,-70,0], distribution_group="arms")
  top_plate → arm_r_base (fixed, offset=[0,70,0], distribution_group="arms")
  arm_l_base → arm_l_shoulder (revolute, axis=z, range=[-180,180])
  arm_l_shoulder → arm_l_upper_link (revolute, axis=x, range=[-120,120], default_angle=-30)
  arm_l_upper_link → arm_l_elbow (revolute, axis=x, range=[-150,150])
  arm_l_elbow → arm_l_forearm (revolute, axis=x, range=[-150,150])
  arm_l_forearm → arm_l_wrist (revolute, axis=x, range=[-180,180])
  arm_l_wrist → arm_l_gripper (fixed)
  (mirror for arm_r_* with symmetric Y offsets)
"""

EXAMPLE_ARM_STANDALONE = """\
{
  "name": "4dof_robot_arm",
  "description": "4自由度单机械臂",
  "default_angles": {"shoulder_joint": 0, "shoulder_link": -45, "elbow_joint": -30, "wrist_link": 15},
  "parts": [
    {"name": "base_plate", "category": "structural", "description": "底座安装板", "material": "Aluminum", "dimensions": {"length": 200, "width": 150, "height": 8}},
    {"name": "shoulder_joint", "category": "actuator", "description": "肩部旋转舵机", "material": "Steel", "dimensions": {"diameter": 40, "height": 35}},
    {"name": "shoulder_link", "category": "structural", "description": "肩部连杆", "material": "Aluminum", "dimensions": {"length": 120, "width": 25, "height": 15}},
    {"name": "elbow_joint", "category": "actuator", "description": "肘部舵机", "material": "Steel", "dimensions": {"diameter": 36, "height": 30}},
    {"name": "elbow_link", "category": "structural", "description": "肘部连杆", "material": "Aluminum", "dimensions": {"length": 100, "width": 25, "height": 15}},
    {"name": "wrist_joint", "category": "actuator", "description": "腕部舵机", "material": "Steel", "dimensions": {"diameter": 28, "height": 28}},
    {"name": "wrist_link", "category": "structural", "description": "腕部连杆", "material": "Aluminum", "dimensions": {"length": 60, "width": 20, "height": 12}},
    {"name": "gripper_base", "category": "mechanical", "description": "夹爪基座(含直线导轨槽和舵机安装座)", "material": "PLA", "dimensions": {"length": 28, "width": 50, "height": 32}},
    {"name": "gripper_servo", "category": "actuator", "description": "夹爪驱动舵机SG90", "material": "Steel", "dimensions": {"length": 23, "width": 12, "height": 22}},
    {"name": "gripper_finger_left", "category": "mechanical", "description": "夹爪左手指(含滑动导轨和L形指尖)", "material": "PLA", "dimensions": {"length": 60, "width": 14, "height": 28}},
    {"name": "gripper_finger_right", "category": "mechanical", "description": "夹爪右手指(含滑动导轨和L形指尖)", "material": "PLA", "dimensions": {"length": 60, "width": 14, "height": 28}}
  ],
  "joints": [
    {"type": "revolute", "parent": "base_plate", "child": "shoulder_joint", "axis": "z", "range_deg": [-180, 180], "parent_anchor": "top", "child_anchor": "bottom"},
    {"type": "revolute", "parent": "shoulder_joint", "child": "shoulder_link", "axis": "x", "range_deg": [-120, 120], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "revolute", "parent": "shoulder_link", "child": "elbow_joint", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "elbow_joint", "child": "elbow_link", "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "elbow_link", "child": "wrist_joint", "parent_anchor": "front", "child_anchor": "back"},
    {"type": "revolute", "parent": "wrist_joint", "child": "wrist_link", "axis": "y", "range_deg": [-180, 180], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "wrist_link", "child": "gripper_base", "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "gripper_base", "child": "gripper_servo", "parent_anchor": "top", "child_anchor": "bottom", "connection_method": "bolted", "connection_detail": {"bolt_size": "M2", "bolt_count": 2}},
    {"type": "prismatic", "parent": "gripper_base", "child": "gripper_finger_left", "axis": "x", "range_deg": [-8, 12], "parent_anchor": "front", "child_anchor": "back", "offset": [-16, 0, 0]},
    {"type": "prismatic", "parent": "gripper_base", "child": "gripper_finger_right", "axis": "x", "range_deg": [-8, 12], "parent_anchor": "front", "child_anchor": "back", "offset": [16, 0, 0], "mimic_joint": "gripper_finger_left", "mimic_multiplier": -1.0, "mimic_offset": 0}
  ]
}
"""

# Example based on reverse-engineered BCN3D MOVEO pattern:
# 5-DOF arm using NEMA17 steppers, all PLA 3D-printed structural parts,
# M3 bolted connections, 608 bearing press-fit in joints.
EXAMPLE_5DOF_ARM_REALISTIC = """\
{
  "name": "5dof_printed_arm",
  "description": "5自由度3D打印机械臂（参考BCN3D MOVEO）",
  "default_angles": {"shoulder_link": -30, "elbow_link": -45, "wrist_link": 20},
  "parts": [
    {"name": "base", "category": "structural", "description": "底座（Φ180mm）", "material": "PLA", "dimensions": {"diameter": 180, "height": 8}},
    {"name": "base_rotation_motor", "category": "actuator", "description": "底座旋转NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 40}},
    {"name": "shoulder_joint_housing", "category": "structural", "description": "肩部关节壳体", "material": "PLA", "dimensions": {"length": 50, "width": 40, "height": 45}},
    {"name": "shoulder_motor", "category": "actuator", "description": "肩部NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 40}},
    {"name": "shoulder_link", "category": "structural", "description": "肩部连杆", "material": "PLA", "dimensions": {"length": 140, "width": 35, "height": 20}},
    {"name": "elbow_joint_housing", "category": "structural", "description": "肘部关节壳体", "material": "PLA", "dimensions": {"length": 45, "width": 35, "height": 40}},
    {"name": "elbow_motor", "category": "actuator", "description": "肘部NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 40}},
    {"name": "elbow_link", "category": "structural", "description": "肘部连杆", "material": "PLA", "dimensions": {"length": 120, "width": 30, "height": 18}},
    {"name": "wrist_joint_housing", "category": "structural", "description": "腕部关节壳体", "material": "PLA", "dimensions": {"length": 40, "width": 30, "height": 35}},
    {"name": "wrist_motor", "category": "actuator", "description": "腕部NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 34}},
    {"name": "wrist_link", "category": "structural", "description": "腕部连杆", "material": "PLA", "dimensions": {"length": 60, "width": 25, "height": 15}},
    {"name": "wrist_rotate_motor", "category": "actuator", "description": "腕部旋转NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 34}},
    {"name": "end_effector_mount", "category": "structural", "description": "末端安装座", "material": "PLA", "dimensions": {"length": 40, "width": 30, "height": 10}},
    {"name": "bearing_base", "category": "bearing", "description": "底座轴承608", "material": "Steel", "dimensions": {"diameter": 22, "height": 7}},
    {"name": "bearing_shoulder", "category": "bearing", "description": "肩部轴承608", "material": "Steel", "dimensions": {"diameter": 22, "height": 7}}
  ],
  "joints": [
    {"type": "fixed", "parent": "base", "child": "base_rotation_motor", "parent_anchor": "bottom", "child_anchor": "top", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "revolute", "parent": "base", "child": "shoulder_joint_housing", "axis": "z", "range_deg": [-180, 180], "parent_anchor": "top", "child_anchor": "bottom"},
    {"type": "fixed", "parent": "shoulder_joint_housing", "child": "shoulder_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "shoulder_joint_housing", "child": "bearing_shoulder", "parent_anchor": "center", "child_anchor": "center", "connection_method": "press_fit"},
    {"type": "revolute", "parent": "shoulder_joint_housing", "child": "shoulder_link", "axis": "x", "range_deg": [-120, 120], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "shoulder_link", "child": "elbow_joint_housing", "parent_anchor": "front", "child_anchor": "back", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "elbow_joint_housing", "child": "elbow_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "revolute", "parent": "elbow_joint_housing", "child": "elbow_link", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "elbow_link", "child": "wrist_joint_housing", "parent_anchor": "front", "child_anchor": "back", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "wrist_joint_housing", "child": "wrist_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "revolute", "parent": "wrist_joint_housing", "child": "wrist_link", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "revolute", "parent": "wrist_link", "child": "wrist_rotate_motor", "axis": "y", "range_deg": [-180, 180], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "wrist_rotate_motor", "child": "end_effector_mount", "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "base", "child": "bearing_base", "parent_anchor": "center", "child_anchor": "center", "connection_method": "press_fit"}
  ]
}
"""

# Example based on PAROL6 (PCrnjak/PAROL6-Desktop-robot-arm):
# 6-DOF desktop arm using NEMA17 steppers, GT2 timing belt transmission,
# MR105/608 bearings, all 3D-printed structural parts.
EXAMPLE_6DOF_BELT_DRIVE_ARM = """\
{
  "name": "6dof_belt_drive_arm",
  "description": "6自由度同步带驱动桌面机械臂（参考PAROL6）",
  "default_angles": {"shoulder_link": -45, "elbow_link": -60, "wrist_pitch_link": 30, "wrist_roll_housing": 0, "wrist_yaw_link": 0},
  "parts": [
    {"name": "base", "category": "structural", "description": "底座壳体", "material": "PLA", "dimensions": {"length": 140, "width": 140, "height": 40}},
    {"name": "base_motor", "category": "actuator", "description": "底座旋转NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 34}},
    {"name": "shoulder_housing", "category": "structural", "description": "肩部关节壳体（含轴承座）", "material": "PLA", "dimensions": {"length": 55, "width": 45, "height": 50}},
    {"name": "shoulder_motor", "category": "actuator", "description": "肩部NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 40}},
    {"name": "shoulder_link", "category": "structural", "description": "肩部连杆", "material": "PLA", "dimensions": {"length": 150, "width": 35, "height": 22}},
    {"name": "elbow_housing", "category": "structural", "description": "肘部关节壳体", "material": "PLA", "dimensions": {"length": 48, "width": 40, "height": 45}},
    {"name": "elbow_motor", "category": "actuator", "description": "肘部NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 34}},
    {"name": "elbow_link", "category": "structural", "description": "肘部连杆", "material": "PLA", "dimensions": {"length": 120, "width": 30, "height": 18}},
    {"name": "wrist_pitch_housing", "category": "structural", "description": "腕部俯仰壳体", "material": "PLA", "dimensions": {"length": 42, "width": 35, "height": 38}},
    {"name": "wrist_pitch_motor", "category": "actuator", "description": "腕部俯仰NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 28}},
    {"name": "wrist_pitch_link", "category": "structural", "description": "腕部俯仰连杆", "material": "PLA", "dimensions": {"length": 60, "width": 28, "height": 15}},
    {"name": "wrist_yaw_housing", "category": "structural", "description": "腕部偏航壳体", "material": "PLA", "dimensions": {"length": 38, "width": 30, "height": 32}},
    {"name": "wrist_yaw_motor", "category": "actuator", "description": "腕部偏航NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 28}},
    {"name": "wrist_roll_housing", "category": "structural", "description": "腕部旋转壳体", "material": "PLA", "dimensions": {"length": 32, "width": 28, "height": 28}},
    {"name": "wrist_roll_motor", "category": "actuator", "description": "腕部旋转NEMA17", "material": "Steel", "dimensions": {"length": 42.3, "width": 42.3, "height": 28}},
    {"name": "end_effector_mount", "category": "structural", "description": "末端安装法兰", "material": "PLA", "dimensions": {"length": 35, "width": 35, "height": 8}},
    {"name": "bearing_base", "category": "bearing", "description": "底座轴承608", "material": "Steel", "dimensions": {"diameter": 22, "height": 7}},
    {"name": "bearing_shoulder_upper", "category": "bearing", "description": "肩部上轴承MR105", "material": "Steel", "dimensions": {"diameter": 10, "height": 4}},
    {"name": "bearing_shoulder_lower", "category": "bearing", "description": "肩部下轴承MR105", "material": "Steel", "dimensions": {"diameter": 10, "height": 4}},
    {"name": "bearing_elbow", "category": "bearing", "description": "肘部轴承MR105", "material": "Steel", "dimensions": {"diameter": 10, "height": 4}}
  ],
  "joints": [
    {"type": "fixed", "parent": "base", "child": "base_motor", "parent_anchor": "bottom", "child_anchor": "top", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "base", "child": "bearing_base", "parent_anchor": "center", "child_anchor": "center", "connection_method": "press_fit"},
    {"type": "revolute", "parent": "base", "child": "shoulder_housing", "axis": "z", "range_deg": [-180, 180], "parent_anchor": "top", "child_anchor": "bottom"},
    {"type": "fixed", "parent": "shoulder_housing", "child": "shoulder_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "shoulder_housing", "child": "bearing_shoulder_upper", "parent_anchor": "center", "child_anchor": "center", "connection_method": "press_fit"},
    {"type": "fixed", "parent": "shoulder_housing", "child": "bearing_shoulder_lower", "parent_anchor": "center", "child_anchor": "center", "connection_method": "press_fit"},
    {"type": "revolute", "parent": "shoulder_housing", "child": "shoulder_link", "axis": "x", "range_deg": [-120, 120], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "shoulder_link", "child": "elbow_housing", "parent_anchor": "front", "child_anchor": "back", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "elbow_housing", "child": "elbow_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "elbow_housing", "child": "bearing_elbow", "parent_anchor": "center", "child_anchor": "center", "connection_method": "press_fit"},
    {"type": "revolute", "parent": "elbow_housing", "child": "elbow_link", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "elbow_link", "child": "wrist_pitch_housing", "parent_anchor": "front", "child_anchor": "back", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "wrist_pitch_housing", "child": "wrist_pitch_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "revolute", "parent": "wrist_pitch_housing", "child": "wrist_pitch_link", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "wrist_pitch_link", "child": "wrist_yaw_housing", "parent_anchor": "front", "child_anchor": "back", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "wrist_yaw_housing", "child": "wrist_yaw_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "revolute", "parent": "wrist_yaw_housing", "child": "wrist_roll_housing", "axis": "z", "range_deg": [-180, 180], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "wrist_roll_housing", "child": "wrist_roll_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "revolute", "parent": "wrist_roll_housing", "child": "end_effector_mount", "axis": "y", "range_deg": [-180, 180], "parent_anchor": "front", "child_anchor": "back"}
  ]
}
"""


# ---------------------------------------------------------------------------
# Assembly Generator
# ---------------------------------------------------------------------------


def generate_assembly_from_nl(
    description: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "GLM-5.2",
    temperature: float = 0.3,
    system_prompt: str | None = None,
    few_shot_extras: str = "",
) -> Assembly:
    """Generate an Assembly from natural language description using LLM.

    Args:
        description: Natural language robot description (Chinese or English).
        api_key: API key for the LLM. Defaults to GLM_API_KEY env var.
        base_url: API base URL. Defaults to GLM_BASE_URL env var.
        model: Model name to use.
        temperature: Generation temperature (lower = more deterministic).
        system_prompt: Optional system prompt override. When ``None`` (the
            default) the built-in ``ASSEMBLY_GEN_SYSTEM_PROMPT`` is used,
            preserving the behaviour every existing caller relies on. The
            multi-expert ``AssemblyPipeline`` passes its Architect persona
            (``StageAgent.system_prompt(ASSEMBLY_GEN_SYSTEM_PROMPT)``) so the
            role identity reaches the model on the *first* generation round
            too, not only on repair rounds (see ``_regenerate_with_feedback``).
        few_shot_extras: Optional pre-formatted block of *retrieved* past
            verified-good cases, supplied by the experience store
            (:mod:`lang3d.experience`). When non-empty, it is appended to the
            user prompt as additional few-shot precedent. Empty by default —
            callers that don't use the experience store see no behaviour
            change.

    Returns:
        Assembly object with parts and joints.

    Raises:
        RuntimeError: If generation fails or output is invalid.
    """
    from ..models.base import Message
    from ..models.glm import GLMBackend

    api_key = api_key or os.environ.get("GLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("GLM_API_KEY not set")

    base_url = base_url or os.environ.get(
        "GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"
    )

    backend = GLMBackend(api_key=api_key, base_url=base_url, model=model)

    # Build prompt with few-shot examples
    desc_lower = description.lower()
    is_arm = any(kw in desc_lower for kw in ["臂", "arm", "机械手", "机械臂", "抓手", "gripper", "自由度"])
    is_wheeled = any(kw in desc_lower for kw in ["轮", "wheel", "差速", "移动", "底盘"])

    user_prompt = (
        f"请根据以下描述生成机器人装配体定义 JSON：\n\n"
        f"「{description}」\n\n"
        f"要求：\n"
        f"1. 返回纯 JSON，不要包裹在 ```json``` 代码块中\n"
        f"2. 包含完整的 parts 和 joints 数组\n"
        f"3. 所有关节形成连通树（以 base_plate 为根）\n"
        f"4. 零件名称使用 snake_case\n"
    )

    if is_arm:
        # Arm-specific instructions for correct topology.
        # NOTE: these apply to ANY robot that has arms — a fixed-base
        # standalone arm (is_wheeled=False) AND a wheeled dual-arm robot
        # (is_wheeled=True).  The earlier guard `if is_arm and not
        # is_wheeled` silently dropped these rules for dual-arm robots,
        # which then failed to generate a plausible 15-part structure.
        desc_lower_for_arm = description.lower()
        # Determine the requested DOF numerically. This replaces the legacy
        # keyword-substring detector (is_5dof/is_6dof/else→4dof) which (a)
        # false-positived on "5V"/"6 wheels" and (b) collapsed every other
        # request (2/3/4/7-DOF) onto the 4-DOF exemplar. parse_dof reads an
        # explicit "N自由度"/"N-dof"/Chinese-numeral and returns None when
        # unstated, defaulting to 4 (the most common, well-validated case).
        n_dof = parse_dof(description) or 4
        # Real-machine references (BCN3D MOVEO / PAROL6) are shown as
        # *additional* engineering context when the DOF matches a known
        # profile, but the primary topology example is always the
        # parametrically-generated one so every DOF gets a correct scaffold.
        real_machine_ref = ""
        if n_dof == 5:
            real_machine_ref = (
                f"\n（工程参考：5自由度3D打印机械臂，参考BCN3D MOVEO）"
                f"\n{EXAMPLE_5DOF_ARM_REALISTIC}\n"
            )
        elif n_dof == 6:
            real_machine_ref = (
                f"\n（工程参考：6自由度同步带驱动机械臂，参考PAROL6）"
                f"\n{EXAMPLE_6DOF_BELT_DRIVE_ARM}\n"
            )
        parametric_example = build_arm_example(n_dof)

        user_prompt += (
            f"\n5. **机械臂拓扑规则**（必须严格遵守）：\n"
            f"   - 零件必须按 joint→link→joint→link→... 交替排列\n"
            f"   - 不要出现 link→link 直接连接！\n"
            f"   - **base yaw 关节（底座旋转）**：唯一用 'top'/'bottom' 的关节，axis='z'（绕垂直轴旋转整个臂）\n"
            f"   - **所有臂段关节**（servo→link、link→servo、link→gripper）都用 'front'/'back'\n"
            f"     （这样连杆的 length 维度参与定位，连杆沿长度方向延伸，IK 求解器才能读到正确的轴到轴距离）\n"
            f"   - **pitch 关节**（肩/肘/腕俯仰）用 axis='x'（垂直于水平臂方向，旋转产生上下弯曲）\n"
            f"     **绝对不要给 front/back 的 pitch 关节用 axis='y'**——Y 平行于臂方向，旋转不产生上下位移，臂无法到达目标！\n"
            f"   - **wrist roll 关节**（绕臂方向滚转）用 'front'/'back' + axis='y'（Y 沿水平臂方向）\n"
            f"   - 提供 default_angles 让臂有弯曲姿态（不要全是0度，否则臂会平铺成一条直线）\n"
            f"   - 关节零件用 cylindrical dimensions（diameter + height）\n"
            f"   - 连杆零件用 box dimensions（length >> width, height）\n"
            f"   - **夹爪必须拆成 4 个零件**：gripper_servo + gripper_base + gripper_finger_left + gripper_finger_right，\n"
            f"     gripper_servo（SG90舵机 23×12×22mm）固定在 gripper_base 顶部（top→bottom），\n"
            f"     gripper_base 宽50mm×高32mm×长28mm（比臂连杆更宽更高更短，看起来像夹爪基座而非连杆），\n"
            f"     两个手指用 prismatic 关节（axis='x'，offset 左[-16,0,0] 右[16,0,0]），手指长60mm 宽10mm 高28mm，\n"
            f"     两个手指的 offset 必须是 ±16（即手指中心距=32mm，>25mm 几何阈值），绝对不能用更小的 offset！间距太小会被判定为融合！夹爪必须是实际可动的！\n"
        )
        if not is_wheeled:
            # Fixed-base standalone arm: NO wheels. This is reinforced hard
            # because GLM-5.2 has a persistent tendency to hallucinate
            # wheel/motor_mount parts on arms (e2e observed 3 consecutive
            # rounds generating wheels despite the rule). A positive
            # allow-list + explicit negative example is stronger than a
            # buried one-liner.
            user_prompt += (
                f"\n   ╔══════════════════════════════════════════════════════════╗\n"
                f"   ║  固定底座机械臂 —— 零件白名单（违反即错误）              ║\n"
                f"   ║  固定臂【只能】包含以下零件类别：                        ║\n"
                f"   ║    1. base_plate（底座板，唯一接地零件）                 ║\n"
                f"   ║    2. *_servo / *_joint（旋转关节舵机，cylinder）       ║\n"
                f"   ║    3. *_link（连杆，box，承力结构）                      ║\n"
                f"   ║    4. gripper_*（夹爪 4 件套）                           ║\n"
                f"   ║  【绝对禁止】的零件：wheel / tire / motor_mount / 履带 / ║\n"
                f"   ║    track / 螺旋桨 / propeller / 任何移动底盘零件。       ║\n"
                f"   ║  原因：固定臂靠 base_plate 螺栓固定在工作台，不需要移动。║\n"
                f"   ║  违规会被几何检查拦截并强制重生成（浪费 API 调用）。    ║\n"
                f"   ╚══════════════════════════════════════════════════════════╝\n"
            )
        else:
            # Wheeled robot with arms (e.g. dual-arm): arms mount on the
            # chassis top plate.  Wheel/chassis structure comes from the
            # wheeled-base example below.
            user_prompt += (
                f"   - **带轮机械臂**：每条臂通过 arm_l_base/arm_r_base（fixed 关节）安装在底盘 top_plate 上，\n"
                f"     左右臂用 distribution_group=\"arms\" 镜像（左臂 offset Y<0，右臂 Y>0，如 ±70mm）。\n"
                f"     臂本身遵循上面的拓扑规则；底盘+轮子遵循下面的轮式底盘规则。\n"
            )

        # Primary topology example: parametrically generated for the exact
        # requested DOF (2-7). Every DOF now gets a structurally-correct
        # scaffold instead of only 5/6 getting bespoke JSONs.
        user_prompt += (
            f"\n参考示例（{n_dof}自由度机械臂，关节拓扑已按规则生成，可在此基础上调整尺寸/名称）：\n"
            f"{parametric_example}\n"
        )
        if real_machine_ref:
            user_prompt += real_machine_ref
        # For wheeled robots, use the parametric chassis generator instead of
        # the hard-coded EXAMPLE_4W_ROBOT. The generator bakes in the
        # axis=y/center/no_distribute wheel conventions that the LLM kept
        # mutating (→ wheels vertical, flung 600mm out, Z misaligned) when
        # given EXAMPLE_4W_ROBOT as advisory text. Generated sizes also adapt
        # to payload (heavier → bigger wheels), which the hard-coded example
        # never did.
        if is_wheeled:
            drive = parse_drive_type(description)
            chassis_example = build_wheeled_base(
                wheel_count=4, drive_type=drive, payload_kg=5.0,
            )
            if is_arm:
                # Dual-arm: compose the full chassis+arms deterministically.
                # This gives the LLM ONE structurally-correct dual-arm JSON
                # (arms symmetric, wheels flat, no part flung out) rather
                # than asking it to merge three separate examples — which is
                # where every dual-arm failure originated.
                from ..agent.assembly_compose import compose_dual_arm_assembly
                dual_example = compose_dual_arm_assembly(
                    chassis_example, parametric_example, arm_dof=n_dof,
                )
                user_prompt += (
                    f"\n参考示例（完整双臂轮式机器人，底盘+双{n_dof}自由度臂已确定性组装，"
                    f"轮子齐平/双臂对称/拓扑正确，在此基础上调整尺寸）：\n{dual_example}\n"
                )
            else:
                user_prompt += (
                    f"\n参考示例（{drive}底盘+轮子结构，轮子轴=y/center/no_distribute，"
                    f"在此基础上调整尺寸）：\n{chassis_example}\n"
                )
    else:
        # Non-arm, non-wheeled fallback (shouldn't normally hit).
        user_prompt += (
            f"\n参考示例（4轮差速底盘）：\n{EXAMPLE_4W_ROBOT}\n"
        )

    # Experience-store injection (retrieve-before). The pipeline passes a
    # pre-formatted block of past *verified-good* cases for similar prompts;
    # these act as additional few-shot precedent on top of the parametric
    # examples above. Skipped when empty (no experience yet, or the store is
    # disabled) so the prompt is byte-identical to the pre-store behaviour.
    if few_shot_extras:
        user_prompt += f"\n{few_shot_extras}\n"

    response = backend.chat(
        messages=[Message(role="user", content=user_prompt)],
        system=system_prompt if system_prompt is not None else ASSEMBLY_GEN_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=16384,
        # Assembly generation is a STRUCTURED-JSON task — GLM-5.2's
        # chain-of-thought reasoning adds minutes of latency without
        # improving JSON correctness (the schema is fully specified in the
        # system prompt). Disabling thinking returns the JSON in seconds
        # instead of hanging the e2e loop for 5+ minutes. Verified: with
        # thinking disabled, the dual-arm prompt returns ~23k chars of valid
        # JSON in ~70s; with default thinking it stalls past the SDK timeout.
        thinking={"type": "disabled"},
    )

    raw_text = response.content.strip()
    logger.info("Assembly generator raw response length: %d chars", len(raw_text))

    # Parse the JSON response
    assembly = _parse_assembly_json(raw_text)

    # Apply normalizing sanitizers (non-raising).  Raising validators
    # (_raise_on_wheel_in_arm, _validate_proportions) are applied in the
    # VLM retry loop (generate_assembly_with_vlm_loop) so their errors
    # enter problems_history and the LLM can regenerate.  They are
    # intentionally NOT called here: a standalone call to this function
    # returns a parsed, normalised assembly without needing a surrounding
    # try/except, and the loop consolidates all validation errors in one
    # place (Step A.5).
    assembly = _normalize_gripper_fingers(assembly)
    assembly = _normalize_wheel_positions(assembly)
    if is_arm:
        assembly = _ensure_arm_default_angles(assembly)

    return assembly


# ============================================================================
# Catalog binding — replace LLM-invented dimensions with real standard specs
# ============================================================================


def _map_catalog_dims(catalog_dims: dict, part_class: str = "") -> dict:
    """Map catalog parameter keys to solver/feature-engine dimension keys.

    The catalog uses keys like ``body_length``, ``outer_diameter``.
    The solver expects ``length``, ``width``, ``height``, ``diameter``.
    """
    result: dict[str, float] = {}
    has_body_size = "body_size" in catalog_dims

    for k, v in catalog_dims.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if k == "body_size":
            # NEMA steppers: body_size is the square face dimension
            result["length"] = v
            result["width"] = v
        elif k == "body_length":
            if has_body_size:
                # Stepper depth (along shaft axis)
                result["height"] = v
            else:
                # Servo longest dimension
                result["length"] = v
        elif k == "body_width":
            result["width"] = v
        elif k == "body_height":
            result["height"] = v
        elif k == "body_depth":
            result["depth"] = v
        elif k == "pcb_length":
            result["length"] = v
        elif k == "pcb_width":
            result["width"] = v
        elif k == "pcb_thickness":
            result["height"] = v
        elif k == "outer_diameter":
            result["diameter"] = v
        elif k == "width" and part_class == "functional":
            result.setdefault("height", v)
        elif k in ("length", "width", "height", "diameter", "depth", "thickness"):
            result[k] = v
        else:
            result[k] = v
    return result


def _bind_catalog_part(
    name: str,
    description: str,
    category: str,
    llm_dims: dict,
) -> tuple[dict, str | None]:
    """Try to bind a part to a catalog standard part by model number.

    When the LLM mentions a known model number (SG90, MG996R, NEMA17,
 Dynamixel, bearing 608, etc.), replace the LLM-invented dimensions
    with real catalog specifications.

    Returns ``(dimensions, catalog_id)``.  If no match, returns
    ``(llm_dims, None)``.
    """
    try:
        from ..knowledge.parts_catalog import get_all_templates
    except ImportError:
        return llm_dims, None

    text = f"{name} {description}".upper().replace(" ", "").replace("-", "").replace("_", "")

    best_match = None
    best_model_len = 0

    for template in get_all_templates():
        if template.part_class != "functional":
            continue
        model = template.model_number.upper().replace(" ", "").replace("-", "").replace("_", "")
        if not model or len(model) < 3:
            continue
        # Use word-boundary-aware matching: the model string must appear as a
        # distinct token, not as a substring of a longer number.
        # Simple substring check works because model numbers are distinctive
        # (SG90, MG996R, NEMA17, 608-2RS, etc.)
        if model in text:
            # Prefer the longest model number match (e.g. MG996R over MG)
            if len(model) > best_model_len:
                best_match = template
                best_model_len = len(model)

    if best_match is None:
        return llm_dims, None

    if not best_match.standard_sizes:
        return llm_dims, None

    real_dims = _map_catalog_dims(best_match.standard_sizes[0], best_match.part_class)
    if not real_dims:
        return llm_dims, None

    logger.info(
        "Catalog binding: '%s' matched %s (model=%s), replacing dims %s -> %s",
        name, best_match.id, best_match.model_number, llm_dims, real_dims,
    )
    return real_dims, best_match.id


def _parse_assembly_json(raw_text: str) -> Assembly:
    """Parse LLM response text into an Assembly object.

    Handles various response formats:
    - Pure JSON
    - JSON wrapped in ```json ... ```
    - JSON with leading/trailing text
    """
    json_str = raw_text.strip()

    # Strip code block markers
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    elif json_str.startswith("```"):
        json_str = json_str[3:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    json_str = json_str.strip()

    # Try direct parse
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw_text[start:end + 1])
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse assembly JSON: {e}\n"
                    f"Raw text (first 500 chars): {raw_text[:500]}"
                ) from e
        else:
            raise RuntimeError(
                f"No JSON object found in response.\n"
                f"Raw text (first 500 chars): {raw_text[:500]}"
            )

    # Convert to Assembly
    parts: list[Part] = []
    for pd in data.get("parts", []):
        llm_dims = pd.get("dimensions", {})
        # Try to bind to catalog standard part (replaces LLM-invented
        # dimensions with real specs when a model number is detected)
        real_dims, catalog_id = _bind_catalog_part(
            name=pd["name"],
            description=pd.get("description", ""),
            category=pd.get("category", "structural"),
            llm_dims=llm_dims,
        )
        parts.append(Part(
            name=pd["name"],
            category=pd.get("category", "structural"),
            description=pd.get("description", ""),
            material=pd.get("material", "PLA"),
            dimensions=real_dims,
            notes=(f"catalog:{catalog_id}" if catalog_id else pd.get("notes", "")),
        ))

    joints: list[Joint] = []
    for jd in data.get("joints", []):
        jtype = jd.get("type", "fixed")
        range_deg = tuple(jd.get("range_deg", [-180, 180]))
        offset = tuple(jd["offset"]) if jd.get("offset") else None

        # Parse connection method from LLM output
        connection = None
        cm_type = jd.get("connection_method", "")
        if cm_type:
            cd = jd.get("connection_detail", {}) or {}
            _bolt_size = cd.get("bolt_size", "M3")
            # Default torque from catalog so metadata and step text agree (P0-2).
            _torque = cd.get("torque_nm", 0.0)
            if cm_type == "bolted" and _torque == 0.0:
                _torque = get_torque(_bolt_size, "PLA")
            connection = ConnectionMethod(
                type=cm_type,
                bolt_size=_bolt_size,
                bolt_count=cd.get("bolt_count", 0),
                torque_nm=_torque,
                interference_mm=cd.get("interference_mm", 0.0),
                snap_count=cd.get("snap_count", 0),
                snap_force_n=cd.get("snap_force_n", 0.0),
                adhesive_type=cd.get("adhesive_type", ""),
                bond_area_mm2=cd.get("bond_area_mm2", 0.0),
                weld_type=cd.get("weld_type", ""),
            )

        joints.append(Joint(
            type=jtype,
            parent=jd["parent"],
            child=jd["child"],
            range_deg=range_deg,
            description=jd.get("description", ""),
            axis=jd.get("axis", "auto"),
            parent_anchor=jd.get("parent_anchor", "top"),
            child_anchor=jd.get("child_anchor", "bottom"),
            offset=offset,
            no_distribute=jd.get("no_distribute", False),
            distribution_group=jd.get("distribution_group", ""),
            connection=connection,
            mimic_joint=jd.get("mimic_joint", ""),
            mimic_multiplier=jd.get("mimic_multiplier", 1.0),
            mimic_offset=jd.get("mimic_offset", 0.0),
        ))

    # Post-parse anchor fixup: correct common LLM mistakes for arm chains.
    _fix_arm_chain_anchors(joints, parts)

    # Default connection_method for joints that lack one.
    # See apply_default_connection_methods() for the shared rule set used by
    # both the LLM assembly path and build_complex_robot().
    apply_default_connection_methods(joints, parts=parts)

    assembly = Assembly(
        name=data.get("name", "generated_assembly"),
        parts=parts,
        joints=joints,
        description=data.get("description", ""),
    )

    # Set default angles if provided
    if "default_angles" in data:
        assembly.default_angles = data["default_angles"]

    return assembly


# Patterns that identify link-like parts (extend horizontally in arm chains).
_LINK_PATTERNS = ("link", "arm", "forearm", "upper_arm", "bracket")
_JOINT_PATTERNS = ("joint", "support", "housing", "servo", "motor")


# ---------------------------------------------------------------------------
# Re-exports from extracted sub-modules (P1-1 God Module split).
# These names are defined in assembly_gen/sanitizers.py and
# assembly_gen/vlm_verify.py but re-exported here so all existing
# imports (`from .assembly_generator import _validate_assembly`, etc.)
# continue to work without changes.
# ---------------------------------------------------------------------------

from .assembly_gen.sanitizers import apply_default_connection_methods  # noqa: F401
from .assembly_gen.sanitizers import _is_link_like  # noqa: F401
from .assembly_gen.sanitizers import _is_end_effector  # noqa: F401
from .assembly_gen.sanitizers import _is_joint_like  # noqa: F401
from .assembly_gen.sanitizers import _fix_arm_chain_anchors  # noqa: F401
from .assembly_gen.sanitizers import _find_best_parent  # noqa: F401
from .assembly_gen.sanitizers import _ensure_connected  # noqa: F401
from .assembly_gen.sanitizers import _raise_on_wheel_in_arm  # noqa: F401
from .assembly_gen.sanitizers import _strip_wheel_parts  # noqa: F401
from .assembly_gen.sanitizers import _normalize_wheel_positions  # noqa: F401
from .assembly_gen.sanitizers import wheel_joint_suffixes_map  # noqa: F401
from .assembly_gen.sanitizers import _normalize_gripper_fingers  # noqa: F401
from .assembly_gen.sanitizers import _validate_proportions  # noqa: F401
from .assembly_gen.sanitizers import _ensure_arm_default_angles  # noqa: F401
from .assembly_gen.sanitizers import _validate_assembly  # noqa: F401

from .assembly_gen.vlm_verify import _classify_robot  # noqa: F401
from .assembly_gen.vlm_verify import _build_verify_prompt  # noqa: F401
from .assembly_gen.vlm_verify import _CATEGORY_EXPECTATIONS  # noqa: F401
from .assembly_gen.vlm_verify import _is_gripper_false_alarm  # noqa: F401
from .assembly_gen.vlm_verify import _is_floating_false_alarm  # noqa: F401
from .assembly_gen.vlm_verify import _classify_vlm_complaint  # noqa: F401
from .assembly_gen.vlm_verify import _SOFT_VLM_MARKERS  # noqa: F401
from .assembly_gen.vlm_verify import _HARD_VLM_MARKERS  # noqa: F401
from .assembly_gen.vlm_verify import _assembly_has_wheels  # noqa: F401
from .assembly_gen.vlm_verify import _is_wheel_false_alarm  # noqa: F401
from .assembly_gen.vlm_verify import _WHEEL_FALSE_ALARM_PATTERNS  # noqa: F401
from .assembly_gen.vlm_verify import _WHEEL_PART_STEMS  # noqa: F401
from .assembly_gen.vlm_verify import _GRIPPER_FALSE_ALARM_PATTERNS  # noqa: F401
from .assembly_gen.vlm_verify import _FLOATING_FALSE_ALARM_PATTERNS  # noqa: F401
from .assembly_gen.vlm_verify import _geometric_prevalidation  # noqa: F401
from .assembly_gen.vlm_verify import _fcl_confirm_intersections  # noqa: F401
from .assembly_gen.vlm_verify import _build_box_preview_mesh  # noqa: F401
from .assembly_gen.vlm_verify import _build_cylinder_preview_mesh  # noqa: F401
from .assembly_gen.vlm_verify import _build_finger_preview_mesh  # noqa: F401
from .assembly_gen.vlm_verify import _generate_preview_stls  # noqa: F401
from .assembly_gen.vlm_verify import _vlm_check_assembly  # noqa: F401
from .assembly_gen.vlm_verify import _VLM_VERIFY_PROMPT  # noqa: F401
from .assembly_gen.vlm_verify import _VLM_GRIPPER_CLOSEUP_PROMPT  # noqa: F401
from .assembly_gen.vlm_verify import _VLM_FIX_PROMPT  # noqa: F401
from .assembly_gen.vlm_verify import _DEFAULT_VERIFIER_VISION_MODEL  # noqa: F401

class AssemblyGenerateTool(Tool):
    """Generate a robotic assembly from natural language with VLM auto-fix loop."""

    name = "assembly_generate"
    description = (
        "Generate a robotic assembly from natural language. Runs closed-loop: "
        "LLM generates → solver positions → VTK renders → VLM verifies → "
        "if problems found, LLM regenerates with feedback (up to 3 rounds). "
        "Then exports complete engineering package. "
        "Returns JSON with assembly, render paths, and export directory."
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "Natural language description of the robot. "
                            "Example: '设计一个4轮差速移动机器人，带双臂和传感器塔'"
                        ),
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory (default: auto-generated)",
                    },
                    "max_rounds": {
                        "type": "integer",
                        "description": "Max generate-verify-fix rounds (default: 3)",
                    },
                },
                "required": ["description"],
            },
        )

    def execute(self, **kwargs: Any) -> str:
        description = kwargs.get("description", "")
        if not description:
            return "Error: 'description' parameter is required"

        output_dir = kwargs.get("output_dir", "")
        max_rounds = int(kwargs.get("max_rounds", 3))

        # Production path: the multi-expert AssemblyPipeline (architect →
        # solver → cad → verifier → fixer, with role personas, tool
        # whitelists, SAGE geometric-vs-VLM arbitration, and selective
        # re-run routing). It returns a dict with the same keys the legacy
        # loop returned, so the summary builder below is unchanged.
        #
        # If the pipeline raises, the DEFAULT is to let the error propagate
        # (fail loud, AGENTS.md §1.1). Previously a bare ``except Exception``
        # silently fell back to the legacy monolithic loop, which meant any
        # pipeline improvement (cad_failed propagation, severity-graded fix
        # loop, deterministic compose) could be bypassed by a single KeyError
        # without the operator ever knowing the legacy path ran instead (the
        # audit's "legacy fallback black hole", P0-5). The legacy loop is
        # still reachable for diagnosis via LANG3D_LEGACY_FALLBACK=1, but it
        # is no longer the silent default.
        from ..agent.pipeline import AssemblyPipeline, PipelineContext

        ctx = PipelineContext(
            description=description,
            output_dir=output_dir,
            max_rounds=max_rounds,
        )
        try:
            result = AssemblyPipeline(ctx).run()
        except Exception as e:
            if os.environ.get("LANG3D_LEGACY_FALLBACK", "").lower() in (
                "1", "true", "yes",
            ):
                logger.error(
                    "AssemblyPipeline failed (%s); LANG3D_LEGACY_FALLBACK is "
                    "set — falling back to legacy generate_assembly_with_vlm_"
                    "loop. NOTE: the legacy loop lacks the pipeline's "
                    "severity-graded fix loop, deterministic compose path, "
                    "and cad_failed propagation.", e,
                )
                try:
                    result = generate_assembly_with_vlm_loop(
                        description=description,
                        output_dir=output_dir,
                        max_rounds=max_rounds,
                    )
                except Exception as legacy_err:
                    # Both paths failed — honor the Executor contract
                    # (tool returns str, not raises).
                    logger.error("Legacy loop also failed: %s", legacy_err)
                    return f"Error: {legacy_err}"
            else:
                logger.error("AssemblyPipeline failed: %s", e)
                raise

        try:
            summary = {
                "passed": result["passed"],
                "final_status": result["final_status"],
                "rounds": result["rounds"],
                "assembly_name": result["assembly"].name if result["assembly"] else None,
                "part_count": len(result["assembly"].parts) if result["assembly"] else 0,
                "joint_count": len(result["assembly"].joints) if result["assembly"] else 0,
                "export_dir": result["export_dir"],
                "problems_per_round": {
                    f"round_{i+1}": p for i, p in enumerate(result["problems_history"])
                },
            }
            return json.dumps(summary, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Assembly result summarisation failed: %s", e)
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# VLM auto-fix closed loop
# ---------------------------------------------------------------------------

# Each robot category carries DIFFERENT structural expectations.  Feeding the
# VLM a category-blind prompt caused a stable false-negative loop on wheeled
# dual-arm robots (data/runs/4wheel_dual_arm/20260627_001843): the VLM saw arm
# links, classified the assembly as a "fixed-base arm", and every round demanded
# removal of the legitimately-generated chassis wheels — geometry that was in
# fact correct (wheel_fl.stl is a horizontal Ø90 cylinder rolling on Y).  The
# generator kept "fixing" by regenerating, hit max_rounds, and aborted with no
# positions.json.  Injecting the category up-front lets the VLM apply the right
# expectations instead of guessing from pixels.
#
# ``robot_category`` is one of: "fixed_arm", "wheeled", "wheeled_arm",
# "assembly" (the generic fallback).  See ``_classify_robot`` below.
# Dedicated gripper-evaluation prompt — used ONLY for the gripper_closeup
# view.  Whole-assembly views cannot resolve ~46mm finger gaps at the edge
# of a 490mm-tall frame, so they false-negative the gripper as a "solid
# block".  The close-up zooms to a 120mm window around the finger centroid
# so the two prongs and the gap between them are clearly visible.  Only
# this view is authoritative for the gripper question.
# ---------------------------------------------------------------------------
# VLM gripper-false-alarm detection
# ---------------------------------------------------------------------------
# VLM (a lightweight vision model) frequently FALSE-NEGATIVES the gripper,
# reporting "solid block / no separated prongs / no gripper at tip" even
# when the solved finger positions are clearly 32mm apart (verified by
# _geometric_prevalidation Check 5).  Since solver positions are
# deterministic ground truth, these complaints are treated as false alarms
# and removed from all_problems when geometry confirms fingers are present
# and separated.  Non-gripper problems (floating parts, wrong orientation,
# collisions) are never matched.

# Context words: the text must ALSO mention one of these to be classified
# as a gripper complaint (prevents "base plate does not have two mounting
# holes" from being filtered).  The GLM-4.6V consistently uses "tip",
# "gripper", "finger", "prong", or "end-effector" when complaining about
# the gripper, so the double-condition is reliable.
_GRIPPER_CONTEXT_WORDS = (
    "gripper", "finger", "prong", "effector", "tip", "claw", "爪",
)


# Markers that unambiguously indicate a HARD geometry defect regardless of
# framing — these MUST be addressed, never treated as soft.
# Part-name stems that indicate a genuine wheel part. If ANY part matches,
# the assembly really has wheels and wheel-orientation complaints are legit.
def generate_assembly_with_vlm_loop(
    description: str,
    output_dir: str = "",
    max_rounds: int = 3,
    api_key: str | None = None,
    base_url: str | None = None,
    text_model: str = "GLM-5.2",
    vision_model: str = _DEFAULT_VERIFIER_VISION_MODEL,
    temperature: float = 0.3,
) -> dict:
    """Full closed-loop: NL → generate → solve → render → VLM verify → fix → loop.

    This is the main entry point for end-to-end NL→Assembly with automatic
    VLM-based quality verification and LLM-based correction.

    Flow:
        1. LLM generates Assembly JSON from description
        2. Assembly solver computes positions
        3. VTK renders 4 views
        4. VLM checks renders for structural problems
        5. If problems found → LLM regenerates with feedback → goto 2
        6. If passed → run full export pipeline
        7. Returns result dict with assembly, positions, render paths, export path

    Args:
        description: Natural language robot description.
        output_dir: Base output directory. Defaults to data/generated_<timestamp>.
        max_rounds: Maximum generate-verify-fix rounds (default 3).
        api_key: LLM API key.
        base_url: LLM API base URL.
        text_model: Model for text generation.
        vision_model: Model for visual verification.
        temperature: Generation temperature.

    Returns:
        Dict with keys: passed, rounds, assembly, positions, problems_history,
        render_dir, export_dir.
    """
    import shutil
    import tempfile
    import time

    from ..models.base import Message
    from ..models.glm import GLMBackend

    from .pipeline_context import AssemblyContext

    api_key = api_key or os.environ.get("GLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("GLM_API_KEY not set")
    base_url = base_url or os.environ.get(
        "GLM_BASE_URL", "https://open.bigmodel.cn/api/coding/paas/v4"
    )

    # Classify the description once so every round and the validation
    # try-block (Step A.5) share the same is_arm / is_wheeled decision.
    # The category feeds BOTH the run-directory naming AND the VLM verify
    # prompt — without category context the VLM mis-classifies wheeled
    # dual-arm robots as fixed-base arms and loops forever demanding removal
    # of legitimate wheels (see ``_build_verify_prompt``).
    robot_category = _classify_robot(description)
    is_arm = robot_category in ("fixed_arm", "wheeled_arm")
    is_wheeled = robot_category in ("wheeled", "wheeled_arm")

    # Output directory — canonical layout: data/runs/<case_id>/<timestamp>/
    # NOTE: ``case_id`` keeps the HISTORIC directory names (arm / wheeled /
    # wheeled_arm / assembly) so existing data/runs/<case>/ folders stay
    # compatible.  ``robot_category`` is the richer semantic label used only
    # for the VLM prompt; the two diverge only for fixed arms
    # (case_id="arm" vs robot_category="fixed_arm").
    if not output_dir:
        try:
            from ..config import make_run_dir
            if is_arm and is_wheeled:
                case_id = "wheeled_arm"
            elif is_arm:
                case_id = "arm"
            elif is_wheeled:
                case_id = "wheeled"
            else:
                case_id = "assembly"
            output_dir = str(make_run_dir(case_id))
        except Exception:
            # Fallback if config import fails — keep layout runnable
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("data", "runs", "assembly", ts)
    os.makedirs(output_dir, exist_ok=True)

    text_backend = GLMBackend(api_key=api_key, base_url=base_url,
                                model=text_model)

    assembly = None
    positions = None
    problems_history: list[list[str]] = []
    render_dir = os.path.join(output_dir, "vlm_renders")
    passed = False
    # Track whether Round 1 used the deterministic compose path. When True,
    # subsequent rounds must NOT fall back to LLM regeneration (which
    # corrupts the wheel/motor topology). Only targeted fixes are allowed,
    # and soft VLM complaints (gripper closeup framing, arm posture) do not
    # trigger regeneration at all — the compose geometry is the engineering
    # ground truth and a VLM framing nitpick must not overturn it.
    _round1_compose = False

    for round_num in range(1, max_rounds + 1):
        logger.info("=== Round %d/%d ===", round_num, max_rounds)
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{max_rounds}")
        print(f"{'='*60}")

        # --- Step A: Generate (or re-generate with feedback) ---
        if round_num == 1:
            # DETERMINISTIC PATH for wheeled dual-arm robots (option A):
            # ``compose_dual_arm_assembly`` produces a structurally-correct
            # chassis+dual-arm layout (wheels grounded at Z≈radius, arms
            # symmetric on the deck, no overlap). The LLM historically
            # REWRITES the wheel/motor joint offsets when given this as an
            # example, flinging wheels to Z=222+ and dead-looping the VLM
            # ("wheels vertical / above Z / missing chassis") for all 3
            # rounds — the regression from the 2026-06-26 working run.
            # Verified: compose output solves to wheel Z=47.5 (ground
            # contact), 200mm wheel-pair spacing (no overlap), arms at ±70mm.
            # Skip LLM topology mutation entirely; VLM still verifies, and if
            # it surfaces a real (non-false-alarm) issue, rounds 2+ can
            # regenerate via the normal LLM fix path.
            if robot_category == "wheeled_arm":
                try:
                    from ..knowledge.mobile_base_gen import build_wheeled_base
                    from ..agent.assembly_compose import compose_dual_arm_assembly
                    _chassis = build_wheeled_base(
                        wheel_count=4, drive_type=parse_drive_type(description),
                        payload_kg=5.0, arm_mount_points=["left", "right"],
                    )
                    _arm = build_arm_example(n_dof=3, profile="mobile")
                    _dual_json = compose_dual_arm_assembly(
                        _chassis, _arm, arm_dof=3,
                    )
                    assembly = _parse_assembly_json(_dual_json)
                    # Re-apply the non-raising sanitizers so any minor
                    # normalisation (gripper fingers, wheel-position guard)
                    # still runs on the deterministic output.
                    assembly = _normalize_gripper_fingers(assembly)
                    assembly = _normalize_wheel_positions(assembly)
                    assembly = _ensure_arm_default_angles(assembly)
                    logger.info(
                        "wheeled_arm: using deterministic compose output "
                        "(skipping LLM topology mutation to avoid wheel Z "
                        "regression) — %d parts, %d joints",
                        len(assembly.parts), len(assembly.joints),
                    )
                    _round1_compose = True
                except Exception as e:
                    # Fall back to LLM generation if compose fails for any
                    # reason (don't let a compose bug block all wheeled runs).
                    logger.warning(
                        "wheeled_arm deterministic compose failed (%s); "
                        "falling back to LLM generation", e,
                    )
                    assembly = None

            if assembly is None:
                try:
                    assembly = generate_assembly_from_nl(
                        description=description,
                        api_key=api_key,
                        base_url=base_url,
                        model=text_model,
                        temperature=temperature,
                    )
                except Exception as e:
                    # GLM occasionally returns an empty body for large structured
                    # prompts (observed: ~16k-char dual-arm prompt returns "" in
                    # ~80s).  Without this guard the RuntimeError from JSON parsing
                    # escapes the whole loop, so rounds 2/3 never run and the e2e
                    # fails with no retry.  Record the failure and let the loop
                    # retry generation next round (or via the fix path).
                    logger.warning("Round 1 generation failed: %s", e)
                    problems_history.append([f"Round 1 generation error: {e}"])
                    assembly = None
                    continue
        else:
            # Round 2+: fix or regenerate. If assembly is None (Round 1
            # generation failed entirely), skip this round's fix path.
            if assembly is None:
                problems_history.append(["Round generation produced no assembly"])
                continue
            # Try deterministic targeted fix FIRST (new path, 2026-06-18).
            # Falls back to LLM regeneration only when no targeted fix applies.
            prev_problems = problems_history[-1]
            targeted_applied = False
            try:
                from ..agent.modifier import apply_targeted_fix_from_vlm
                new_assembly, targeted_applied = apply_targeted_fix_from_vlm(
                    assembly, prev_problems,
                )
                if targeted_applied:
                    assembly = new_assembly
                    print(
                        f"  → Applied targeted fix ({len(prev_problems)} problems) "
                        f"instead of regenerating."
                    )
            except Exception as e:
                logger.warning("Targeted fix path failed: %s", e)

            if not targeted_applied:
                # Severity-gated regeneration (audit: fix-loop had no grading).
                # Classify the remaining problems. If Round 1 was the
                # deterministic compose output AND only SOFT complaints
                # remain (gripper-closeup framing, arm posture), do NOT
                # regenerate — the compose geometry is engineering-correct
                # and a VLM framing nitpick must not overturn it (the
                # regression that flung wheels to Z=222 / X=591). Keep the
                # compose assembly and let the loop's VLM re-verify; if the
                # only failures are soft, the run effectively preserves the
                # correct geometry. HARD defects (collision, misplaced
                # parts) still regenerate on LLM-sourced assemblies.
                if _round1_compose:
                    severities = [_classify_vlm_complaint(p) for p in prev_problems]
                    only_soft = severities and all(s == "SOFT" for s in severities)
                    if only_soft:
                        print(
                            f"  → Compose output retained — only soft VLM "
                            f"complaints ({len(prev_problems)}), not "
                            f"regenerating (would corrupt deterministic "
                            f"topology). Problems: {prev_problems[:2]}"
                        )
                        logger.info(
                            "wheeled_arm: retaining compose output — VLM "
                            "complaints are all SOFT (%s); skipping LLM "
                            "regeneration to preserve correct topology.",
                            prev_problems,
                        )
                        # Keep the compose assembly as-is for re-verification.
                    else:
                        # HARD defects on compose output: log but still
                        # avoid LLM regen (compose is the best geometry we
                        # have; targeted fix already tried). Re-verify.
                        print(
                            f"  → Compose output retained despite HARD/unknown "
                            f"complaints — LLM regen would corrupt topology. "
                            f"Will re-verify: {prev_problems[:2]}"
                        )
                        logger.warning(
                            "wheeled_arm: compose output has HARD complaints "
                            "(%s) but LLM regeneration is suppressed to avoid "
                            "topology corruption; targeted fix did not apply.",
                            prev_problems,
                        )
                else:
                    # LLM-sourced assembly: regenerate with VLM feedback.
                    fix_prompt = _VLM_FIX_PROMPT.format(
                        problems="\n".join(f"- {p}" for p in prev_problems),
                        description=description,
                    )
                    # Include the previous assembly JSON as reference
                    prev_json = _assembly_to_json(assembly)
                    fix_prompt += f"\nPrevious assembly (for reference):\n{prev_json}\n"

                    resp = text_backend.chat(
                        messages=[Message(role="user", content=fix_prompt)],
                        system=ASSEMBLY_GEN_SYSTEM_PROMPT,
                        temperature=min(temperature + 0.2 * (round_num - 1), 0.7),
                        max_tokens=16384,
                        # Structured-JSON regen — disable reasoning for speed
                        # (see the initial generate call for rationale).
                        thinking={"type": "disabled"},
                    )
                    assembly = _parse_assembly_json(resp.content)
            # Re-apply normalizing sanitizers (non-raising).  Raising
            # validators are consolidated in Step A.5 below so their
            # errors enter problems_history instead of escaping the loop.
            assembly = _normalize_gripper_fingers(assembly)
            assembly = _normalize_wheel_positions(assembly)
            if is_arm:
                assembly = _ensure_arm_default_angles(assembly)

        # --- Step A.5: Validate assembly (errors enter LLM retry loop) ---
        # All raising validators run inside this single try so their
        # RuntimeErrors enter problems_history and the LLM gets a chance
        # to regenerate with the error messages as feedback.  Previously
        # only _validate_assembly was guarded; the raising sanitizers
        # (_raise_on_wheel_in_arm, _validate_proportions) sat OUTSIDE the
        # try, so a wheel hallucination or a proportion violation killed
        # the whole pipeline — the exact failure mode that originally
        # crashed 4wheel_dual_arm.
        try:
            # _raise_on_wheel_in_arm rejects wheel parts — only correct for
            # fixed-base arms.  A wheeled dual-arm robot legitimately has
            # chassis wheels, so skip this check when is_wheeled.
            if is_arm and not is_wheeled:
                _raise_on_wheel_in_arm(assembly)
            if is_arm:
                _validate_proportions(assembly)
            _validate_assembly(assembly)
        except RuntimeError as e:
            logger.warning(
                "Assembly validation failed in round %d: %s", round_num, e
            )
            problems_history.append([f"Assembly validation error: {e}"])
            continue

        print(f"  Assembly: {assembly.name}, {len(assembly.parts)} parts, "
              f"{len(assembly.joints)} joints")

        # Dump the assembly JSON to disk for diagnostics (F1).  This captures
        # the exact parts/joints/default_angles the LLM produced (after
        # sanitization) so that downstream issues (e.g. solver position
        # blow-ups like the 467mm gripper offset) can be reproduced and
        # root-caused without re-running the LLM.
        _dump_assembly_json(assembly, output_dir, round_num)

        # --- Step B: Solve positions ---
        try:
            ctx = AssemblyContext(assembly=assembly)
            positions = ctx.ensure_positions()
        except Exception as e:
            logger.warning("Solver failed: %s", e)
            problems_history.append([f"Solver error: {e}"])
            continue

        print(f"  Solved: {len(positions)} positions")

        # --- Step B.5: Collision check + auto-resolve ---
        # Run the mesh-collision detector before VLM so simple overlaps
        # (which the VLM may not see clearly) are caught and fixed
        # deterministically.  Only auto-resolve in early rounds; the
        # final round lets the LLM handle it via the normal VLM loop.
        if round_num < max_rounds:
            try:
                from .mesh_collision import MeshCollisionChecker
                _collision_checker = MeshCollisionChecker()
                _collision_result = _collision_checker.check_assembly_collisions(
                    assembly, positions, skip_adjacent=True,
                )
                _severe = [
                    p for p in _collision_result.pairs
                    if p.is_collision and p.penetration_depth_mm > 1.0
                ]
                if _severe:
                    print(
                        f"  Collisions: {len(_severe)} severe pairs "
                        f"(of {_collision_result.pairs_checked} checked)"
                    )
                    from .collision_resolver import CollisionResolver
                    _resolver = CollisionResolver(max_rounds=2)
                    _resolution = _resolver.resolve(assembly, positions)
                    if _resolution.modified_assembly is not None:
                        assembly = _resolution.modified_assembly
                        positions = _resolution.modified_positions
                        # Refresh the solver context so downstream renders
                        # use the resolved positions.
                        ctx = AssemblyContext(assembly=assembly)
                        ctx._positions = positions
                    if _resolution.resolved:
                        print(
                            f"  Collision-resolved in "
                            f"{_resolution.rounds_used} round(s)"
                        )
                    else:
                        print(
                            f"  Collision partial: "
                            f"{_resolution.remaining_count} remain "
                            f"(history {_resolution.collision_history})"
                        )
            except ImportError as _e:
                # trimesh/python-fcl not installed. Previously a bare
                # ``pass`` (AGENTS.md §1.1) silently dropped ALL collision
                # checking, so a severely self-colliding assembly sailed
                # through the VLM loop untouched. Log it loudly so the run
                # record shows collisions were NOT checked (audit P0-6).
                logger.warning(
                    "Collision pre-check SKIPPED in round %d — python-fcl/"
                    "trimesh not installed. Self-collisions will NOT be "
                    "caught before VLM verification: %s",
                    round_num, _e,
                )
            except Exception as _e:
                logger.warning(
                    "Collision check failed in round %d: %s", round_num, _e,
                )

        # --- Step B.6: Collision-aware joint range clamp ---
        # Narrow each arm joint's range_deg to its maximal collision-free
        # sub-range so the arm cannot sweep into its own base during sim.
        # Mirrors AssemblyPipeline.run_solver's _clamp_arm_ranges.  No-op
        # when FCL is absent (numeric sanitizer caps remain as fallback).
        try:
            from ..agent.assembly_compose import (
                clamp_assembly_joint_ranges_collision_free,
            )
            _n_clamped = clamp_assembly_joint_ranges_collision_free(assembly)
            if _n_clamped:
                print(
                    f"  Collision-aware range clamp: {_n_clamped} joint(s) narrowed"
                )
        except ImportError:
            pass

        # --- Step C: Render + VLM check ---
        parts_dicts = [
            {"name": p.name, "category": p.category, "dimensions": p.dimensions}
            for p in assembly.parts
        ]
        round_render_dir = os.path.join(render_dir, f"round_{round_num}")

        # --- Step B2: Generate real FreeCAD STLs for VLM rendering ---
        # Produces real C-channel / servo-housing / gripper-finger geometry
        # so the VLM sees an actual robot arm instead of a pile of blocks.
        # Falls back to trimesh previews when FreeCAD is unavailable.
        round_stl_dir = os.path.join(round_render_dir, "stl_parts")
        real_stl_dir = None
        try:
            from .export_package import generate_part_stls
            stl_path, val_report = generate_part_stls(
                assembly=assembly, stl_dir=round_stl_dir
            )
            if not val_report.get("skipped"):
                real_stl_dir = stl_path
        except Exception as e:
            logger.warning(
                "Real STL generation failed for round %d: %s", round_num, e
            )

        try:
            passed, problems = _vlm_check_assembly(
                positions=positions,
                parts=parts_dicts,
                render_dir=round_render_dir,
                api_key=api_key,
                base_url=base_url,
                vision_model=vision_model,
                round_num=round_num,
                real_stl_dir=real_stl_dir,
                joints=assembly.joints,
                robot_category=robot_category,
            )
        except Exception as e:
            logger.warning("VLM check error: %s", e)
            problems = [f"VLM check exception: {e}"]
            passed = False

        problems_history.append(problems)

        if passed:
            print(f"  VLM: PASSED")
            break
        else:
            print(f"  VLM: FAILED — {len(problems)} problems:")
            for p in problems:
                print(f"    - {p}")
            print(f"  → Will regenerate with feedback...")

    # --- Step C.5: VLM loop summary (diagnostic) ---
    # Persist a single summary of the whole verify-and-fix loop so the per-round
    # problem trajectory can be inspected without grepping logs.
    summary = {
        "test_id": description[:40],
        "total_rounds": round_num,
        "final_passed": passed,
        "verification_status": (
            "PASSED" if passed else "FAILED_MAX_ROUNDS"
        ),
        "problems_history": problems_history,
    }
    try:
        with open(
            os.path.join(output_dir, "vlm_loop_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to write vlm_loop_summary.json: %s", e)

    # --- Step D: Export (whether passed or max rounds reached) ---
    # Stamp the package with verification status so downstream consumers
    # can detect unverified outputs. We still export on failure to enable
    # debugging, but the design_report.json will flag it.
    export_dir = os.path.join(output_dir, "engineering_package")
    export_success = False
    verification_status = "PASSED" if passed else "FAILED_MAX_ROUNDS"
    last_warnings = problems_history[-1] if problems_history else []

    # Reuse the last round's real STLs when they exist so export skips the
    # redundant ~2 min FreeCAD subprocess run.
    existing_stl_for_export: str | None = None
    _last_round_stl = os.path.join(render_dir, f"round_{round_num}", "stl_parts")
    if os.path.isdir(_last_round_stl) and any(
        f.endswith(".stl") for f in os.listdir(_last_round_stl)
    ):
        existing_stl_for_export = _last_round_stl

    if assembly and positions:
        try:
            from .export_package import export_engineering_package
            result = export_engineering_package(
                assembly=assembly,
                output_dir=export_dir,
                actuator_ids=None,  # Let export_engineering_package derive from assembly
                controller="esp32",
                verification_status=verification_status,
                verification_warnings=last_warnings,
                existing_stl_dir=existing_stl_for_export,
            )
            export_success = result is not None
        except Exception as e:
            logger.warning("Export failed: %s", e)

    # --- Step E: Production render with real STL geometry ---
    production_render_dir = None
    if export_success:
        stl_dir = os.path.join(export_dir, "stl_parts")
        if os.path.isdir(stl_dir):
            from .vtk_renderer import render_assembly_from_positions
            production_render_dir = os.path.join(output_dir, "production_renders")
            try:
                render_assembly_from_positions(
                    parts=parts_dicts,
                    positions=positions,
                    output_dir=production_render_dir,
                    stl_dir=stl_dir,
                    width=1920, height=1080,
                    joints=assembly.joints,
                )
                logger.info("Production renders saved to %s", production_render_dir)
            except Exception as e:
                logger.warning("Production render failed: %s", e)
                production_render_dir = None

    # Save assembly.json + positions.json to the run root so the web 3D
    # viewer (/api/runs/{case}/{ts}/positions) can load the solved assembly
    # without depending on FreeCAD internals. Without these files the web
    # positions API returns 404 and the /simulate page stacks every STL at
    # the origin. (Mirrors the same save in agent/pipeline.py run_export.)
    try:
        _asm_dict = {
            "name": assembly.name,
            "parts": [
                {"name": p.name, "category": p.category,
                 "description": p.description, "material": p.material,
                 "dimensions": dict(p.dimensions)}
                for p in assembly.parts
            ],
            "joints": [
                {k: v for k, v in _j.__dict__.items()
                 if k in ("type","parent","child","axis","parent_anchor",
                          "child_anchor","offset","range_deg","no_distribute",
                          "distribution_group","mimic_joint","mimic_multiplier",
                          "mimic_offset")}
                for _j in assembly.joints
            ],
            "default_angles": dict(assembly.default_angles),
        }
        with open(os.path.join(output_dir, "assembly.json"), "w", encoding="utf-8") as _f:
            json.dump(_asm_dict, _f, ensure_ascii=False, indent=2)
        _pos_dict = {
            n: {"position": list(p.get("position", (0, 0, 0))),
                "rotation": list(p.get("rotation", (0, 0, 1, 0)))}
            for n, p in positions.items()
        }
        with open(os.path.join(output_dir, "positions.json"), "w", encoding="utf-8") as _f:
            json.dump(_pos_dict, _f, ensure_ascii=False, indent=2)
        logger.info("Saved assembly.json + positions.json for web viewer")
    except Exception as e:
        logger.warning("Failed to save assembly.json/positions.json for web: %s", e)

    # Summary
    final_status = "PASSED" if passed else "MAX_ROUNDS_REACHED"
    print(f"\n{'='*60}")
    print(f"Result: {final_status} after {len(problems_history)} rounds")
    print(f"Assembly: {assembly.name if assembly else 'N/A'}, "
          f"{len(assembly.parts) if assembly else 0} parts")
    print(f"Export: {'SUCCESS' if export_success else 'FAILED'} → {export_dir}")
    print(f"Production renders: {production_render_dir or 'N/A'}")
    print(f"{'='*60}")

    return {
        "passed": passed,
        "final_status": final_status,
        "rounds": len(problems_history),
        "assembly": assembly,
        "positions": positions,
        "problems_history": problems_history,
        "render_dir": render_dir,
        "export_dir": export_dir if export_success else None,
        "production_render_dir": production_render_dir,
    }


def _dump_assembly_json(assembly: Assembly, output_dir: str, round_num: int) -> None:
    """Persist the assembly definition to ``{output_dir}/assembly.json``.

    Written every round (overwriting) so the final on-disk JSON reflects the
    assembly that was actually fed to the solver.  This is critical for
    diagnosing solver position anomalies (e.g. the 467mm gripper offset)
    because the LLM-generated assembly is otherwise never saved to disk.

    F1: observability — without this dump, failures inside the solver cannot
    be reproduced from the recorded artifacts.
    """
    try:
        data = {
            "name": assembly.name,
            "description": assembly.description,
            "default_angles": dict(assembly.default_angles),
            "parts": [
                {
                    "name": p.name,
                    "category": p.category,
                    "description": p.description,
                    "material": p.material,
                    "dimensions": dict(p.dimensions),
                    "notes": p.notes,
                }
                for p in assembly.parts
            ],
            "joints": [
                {
                    "type": j.type,
                    "parent": j.parent,
                    "child": j.child,
                    "range_deg": list(j.range_deg),
                    "description": j.description,
                    "axis": j.axis,
                    "parent_anchor": j.parent_anchor,
                    "child_anchor": j.child_anchor,
                    "offset": list(j.offset) if j.offset else None,
                    "no_distribute": j.no_distribute,
                    "distribution_group": j.distribution_group,
                    "mimic_joint": j.mimic_joint,
                    "mimic_multiplier": j.mimic_multiplier,
                    "mimic_offset": j.mimic_offset,
                    "connection_method": (
                        j.connection.type if j.connection else None
                    ),
                    "connection_detail": (
                        {
                            "bolt_size": j.connection.bolt_size,
                            "bolt_count": j.connection.bolt_count,
                        }
                        if j.connection and j.connection.type == "bolted"
                        else {}
                    ),
                }
                for j in assembly.joints
            ],
            "_meta": {"round": round_num},
        }
        path = os.path.join(output_dir, "assembly.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info("Dumped assembly.json (round %d) → %s", round_num, path)
    except Exception as e:
        logger.warning("Failed to dump assembly.json: %s", e)


def _assembly_to_json(assembly: Assembly) -> str:
    """Serialize an Assembly to compact JSON string."""
    data = {
        "name": assembly.name,
        "parts": [
            {"name": p.name, "category": p.category,
             "description": p.description, "material": p.material,
             "dimensions": p.dimensions}
            for p in assembly.parts
        ],
        "joints": [
            {"type": j.type, "parent": j.parent, "child": j.child,
             "range_deg": list(j.range_deg), "axis": j.axis,
             "parent_anchor": j.parent_anchor, "child_anchor": j.child_anchor,
             "distribution_group": j.distribution_group,
             "no_distribute": j.no_distribute,
             "connection_method": j.connection.type if j.connection else "",
             "connection_detail": {
                 "bolt_size": j.connection.bolt_size,
                 "bolt_count": j.connection.bolt_count,
             } if j.connection and j.connection.type == "bolted" else {}}
            for j in assembly.joints
        ],
    }
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_assembly_generator_tools(registry: Any) -> None:
    """Register assembly generator tools."""
    registry.register(AssemblyGenerateTool())
