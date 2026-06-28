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


def apply_default_connection_methods(joints: list, parts: list | None = None) -> None:
    """Assign a default ``ConnectionMethod`` to joints that lack one.

    Dispatches by part category (when ``parts`` is provided) on top of the
    existing anchor-geometry rule:

    - **Bearing** parent or child (any joint type) → ``press_fit`` H7/js6.
      Bearings are never bolted; they press into housings.
    - **Servo** child (name contains "servo", e.g. SG90) → ``bolted M2×2``
      with ``hole_type="threaded_hole"`` — SG90 servos have tapped holes
      for self-tapping M2 screws, not through holes.
    - **Actuator** child (motors like NEMA17) → ``bolted M3×4`` with
      ``hole_type="threaded_hole"`` — motor flanges have tapped holes.
    - Other fixed/revolute joints with face anchors → ``bolted M3×4``
      ``through_hole`` (structural bracket mounting).
    - Revolute with center/center anchors → ``press_fit`` (bearing seat).
    - Prismatic → null (sliding fit, not a fastening).

    Mutates *joints* in place.  ``parts`` is optional for backward
    compatibility; without it the function falls back to the original
    geometry-only dispatch.
    """
    _face_anchors = {"front", "back", "top", "bottom", "left", "right"}
    _parts_by_name = {p.name: p for p in parts} if parts else {}

    def _category(name: str) -> str:
        p = _parts_by_name.get(name)
        return (p.category or "").lower() if p else ""

    def _is_servo(name: str) -> bool:
        return "servo" in name.lower()

    def _bolted(size: str, count: int, hole_type: str) -> ConnectionMethod:
        return ConnectionMethod(
            type="bolted", bolt_size=size, bolt_count=count,
            hole_type=hole_type, torque_nm=get_torque(size, "PLA"),
        )

    for joint in joints:
        if joint.connection is not None:
            continue

        child_cat = _category(joint.child)
        parent_cat = _category(joint.parent)

        # Bearing → always press_fit regardless of joint type
        if child_cat == "bearing" or parent_cat == "bearing":
            if joint.type in ("fixed", "revolute"):
                joint.connection = ConnectionMethod(
                    type="press_fit", interference_mm=0.02,
                )
                logger.debug(
                    "Defaulted joint %s->%s to press_fit (bearing seat)",
                    joint.parent, joint.child,
                )
                continue

        if joint.type == "fixed":
            if _is_servo(joint.child):
                # SG90-style servo: M2 into tapped holes
                joint.connection = _bolted("M2", 2, "threaded_hole")
            elif child_cat == "actuator":
                # Larger motors (NEMA17 etc.): M3 into tapped flange
                joint.connection = _bolted("M3", 4, "threaded_hole")
            else:
                # Structural fixed joint: through hole + nut
                joint.connection = _bolted("M3", 4, "through_hole")
            logger.debug(
                "Defaulted fixed joint %s->%s to %s %s (%s)",
                joint.parent, joint.child,
                joint.connection.bolt_size,
                joint.connection.hole_type,
                joint.connection.type,
            )
        elif joint.type == "revolute":
            uses_face_anchor = (
                joint.parent_anchor in _face_anchors
                or joint.child_anchor in _face_anchors
            )
            if uses_face_anchor:
                if _is_servo(joint.child):
                    joint.connection = _bolted("M2", 2, "threaded_hole")
                elif child_cat == "actuator":
                    joint.connection = _bolted("M3", 4, "threaded_hole")
                else:
                    joint.connection = _bolted("M3", 4, "through_hole")
                logger.debug(
                    "Defaulted revolute joint %s->%s to bolted %s %s "
                    "(face anchor %s/%s)",
                    joint.parent, joint.child,
                    joint.connection.bolt_size,
                    joint.connection.hole_type,
                    joint.parent_anchor, joint.child_anchor,
                )
            else:
                # Center/center: bearing press-fit into a housing bore.
                joint.connection = ConnectionMethod(
                    type="press_fit", interference_mm=0.01,
                )
                logger.debug(
                    "Defaulted revolute joint %s->%s to press_fit "
                    "(bearing seat, center anchors)",
                    joint.parent, joint.child,
                )
        elif joint.type == "prismatic":
            # Sliding interface is not a fastening method; null is intentional.
            logger.info(
                "Prismatic joint %s->%s has no connection_method "
                "(sliding fit, expected)",
                joint.parent, joint.child,
            )

    # Safety: clear connections on ALL prismatic joints.  Sliding
    # interfaces must never have bolted/press-fit fasteners — a bolt
    # through a rail would prevent sliding.  The LLM sometimes marks
    # non-gripper prismatic joints as "bolted"; this ensures they are
    # always null.
    for joint in joints:
        if joint.type == "prismatic" and joint.connection is not None:
            logger.info(
                "Safety: cleared %s connection on prismatic joint "
                "%s->%s (sliding fit)",
                joint.connection.type, joint.parent, joint.child,
            )
            joint.connection = None


def generate_assembly_from_nl(
    description: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "GLM-5.2",
    temperature: float = 0.3,
    system_prompt: str | None = None,
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


def _is_link_like(name: str) -> bool:
    """Check if a part name looks like an arm link (extends horizontally)."""
    n = name.lower()
    return any(p in n for p in _LINK_PATTERNS)


def _is_end_effector(name: str) -> bool:
    """Check if a part name is an end effector."""
    n = name.lower()
    return "end_effector" in n or "gripper" in n or "effector" in n


def _is_joint_like(name: str) -> bool:
    """Check if a part name looks like an arm joint/housing/servo (rotary node).

    Used by ``_fix_arm_chain_anchors`` to decide whether a top/bottom joint is
    part of the arm kinematic chain (should be normalised to front/back) versus
    a motor mount inside a housing (should stay back/front).
    """
    n = name.lower()
    return any(p in n for p in _JOINT_PATTERNS)


def _fix_arm_chain_anchors(joints: list[Joint], parts: list[Part]) -> None:
    """Normalise arm-chain joints to the clean horizontal (front/back) convention.

    The target arm geometry extends horizontally so each link's ``length``
    dimension positions the next pitch axis: pitch joints use ``front/back``
    anchors with ``axis="x"`` (see the 4dof_arm template in
    assembly_templates.py). The base yaw is the only ``top/bottom`` joint.

    Two LLM patterns are fixed:

    1. **top/bottom + axis=y → front/back + axis=x**: Legacy prompt rules
       told the LLM to stack links vertically via top/bottom anchors; that
       built the arm as a tower of thin plates whose ``length`` dimension
       never participated in positioning, collapsing IK link lengths and
       producing a vertical column that does not move like a real arm.
       Converted to the clean pitch convention. base yaw (axis=z) and wrist
       roll are left untouched.

    2. **top/top → top/bottom**: When the LLM uses ``child_anchor="top"`` the
       solver places the child's top face at the parent's top face, so the
       child extends DOWNWARD — the arm folds back on itself like an
       accordion (workspace collapses to ~47mm instead of ~200mm).

    Conservative filters — the following joints are LEFT UNTOUCHED:

    * prismatic joints (gripper fingers are handled by
      ``_normalize_gripper_fingers``).
    * joints already using top/bottom.
    * center/center joints (bearing press-fits).
    * bottom/top joints (motor mounted under the base).
    * fixed joints with parent_anchor="back" / child_anchor="front" (motor
      mounted behind a housing face — e.g. NEMA17 on the BCN3D MOVEO).
    * joints where neither parent nor child is a joint/link/effector-like part
      (avoids converting unrelated structural brackets).
    """
    if len(joints) < 3:
        return

    for i, joint in enumerate(joints):
        # Prismatic fingers are normalized separately.
        if joint.type == "prismatic":
            continue

        parent_name = joint.parent.lower()
        child_name = joint.child.lower()

        # At least one endpoint must be an arm-chain part (joint/housing/servo/
        # motor/link/effector). This avoids rewriting unrelated brackets.
        if not (_is_joint_like(parent_name) or _is_joint_like(child_name)
                or _is_link_like(parent_name) or _is_link_like(child_name)
                or _is_end_effector(parent_name) or _is_end_effector(child_name)):
            continue

        # --- Pattern 2: top/top → top/bottom (fix fold-back) ---
        # child_anchor="top" makes the solver place the child hanging
        # downward from the parent's top, collapsing the arm. Fix first so
        # the subsequent pitch-normalisation can still apply. No `continue`
        # — fall through to Pattern 1.
        if joint.parent_anchor == "top" and joint.child_anchor == "top":
            logger.info(
                "Fixing arm-chain joint %d ('%s'→'%s'): top/top → top/bottom"
                " (child was folding back)",
                i, joint.parent, joint.child,
            )
            joint.child_anchor = "bottom"

        # --- Pattern 1: legacy top/bottom arm-chain joints → clean front/back ---
        # The clean arm convention uses front/back so each link's `length`
        # dimension positions the next pitch axis and the arm extends
        # horizontally as a real, movable arm. Convert any legacy top/bottom
        # arm-chain joint (pitch revolute with axis=y, or fixed link
        # connectors with axis=auto). base yaw (axis=z) is left untouched.
        # The gripper-servo mount (servo atop gripper_base) is preserved.
        is_gripper_servo_mount = (
            "servo" in child_name and "grip" in parent_name
        )
        if (joint.parent_anchor == "top" and joint.child_anchor == "bottom"
                and joint.axis in ("y", "auto")
                and not is_gripper_servo_mount):
            old_axis = joint.axis
            joint.parent_anchor = "front"
            joint.child_anchor = "back"
            if old_axis == "y":
                joint.axis = "x"
            logger.info(
                "Normalising arm-chain joint %d ('%s'→'%s'): top/bottom → "
                "front/back (clean convention, axis %s→%s)",
                i, joint.parent, joint.child, old_axis, joint.axis,
            )
            continue


def _find_best_parent(part_name: str, part_names: set[str], visited: set[str]) -> str | None:
    """Find the best parent for an orphaned part using prefix-based heuristics.

    Maps common component prefixes to their natural parent patterns:
    sensor_* → sensor_tower, camera_* → sensor_tower, lidar_* → sensor_tower,
    imu_* → sensor_tower, battery_* → base_plate, pcb_* → top_plate,
    controller_* → top_plate, power_* → base_plate, servo_* → matching arm link,
    arm_* → matching base, gripper_* → matching wrist/link.
    """
    n = part_name.lower()

    # Prefix-to-parent mapping with candidate patterns
    prefix_map: dict[str, list[str]] = {
        "sensor_": ["sensor_tower", "sensor_mount", "top_plate", "base_plate"],
        "camera_": ["sensor_tower", "sensor_mount", "top_plate", "base_plate"],
        "lidar_": ["sensor_tower", "sensor_mount", "top_plate", "base_plate"],
        "imu_": ["sensor_tower", "sensor_mount", "top_plate", "base_plate"],
        "battery_": ["base_plate", "bottom_plate", "chassis", "top_plate"],
        "pcb_": ["top_plate", "base_plate", "main_board"],
        "controller_": ["top_plate", "base_plate", "main_board"],
        "power_": ["base_plate", "bottom_plate", "battery_box"],
        "servo_": ["base_plate", "top_plate"],
        "arm_": ["base_plate", "top_plate"],
        "gripper_": ["wrist_link", "wrist", "end_effector"],
    }

    for prefix, candidates in prefix_map.items():
        if n.startswith(prefix):
            for candidate in candidates:
                if candidate in visited:
                    return candidate
            return None

    return None


def _ensure_connected(assembly: Assembly, part_names: set[str]) -> None:
    """Auto-fix: connect orphaned parts to the nearest reachable parent.

    When the LLM generates fewer joints than needed, some parts have no
    parent in the joint tree.  This function detects them and adds fixed
    joints to the nearest structural part that IS reachable from the root.
    """
    # Build parent→children map from existing joints
    children_map: dict[str, set[str]] = {}
    child_to_parent: dict[str, str] = {}
    for j in assembly.joints:
        children_map.setdefault(j.parent, set()).add(j.child)
        child_to_parent[j.child] = j.parent

    # BFS from the true kinematic root.
    # The root is the part that is NEVER a joint child (nothing parents onto
    # it) — e.g. base_footprint in a Husky chassis, which is the parent of
    # base_plate but never a child. Hard-coding "base_plate" as the BFS root
    # was wrong: it left base_footprint unvisited, so this auto-fixer added a
    # spurious base_plate→base_footprint joint, creating a cycle (no root in
    # the URDF → MuJoCo "URDF body not found" parse failure). If multiple
    # parts have no parent, prefer base_footprint/base_plate for stability.
    children_of_joints = {j.child for j in assembly.joints}
    candidate_roots = part_names - children_of_joints
    if not candidate_roots:
        # Cycle (every part is a child) — fall back to base_plate/first part.
        root = "base_plate" if "base_plate" in part_names else assembly.parts[0].name
    elif "base_footprint" in candidate_roots:
        root = "base_footprint"
    elif "base_plate" in candidate_roots:
        root = "base_plate"
    else:
        root = next(iter(candidate_roots))
    visited = {root}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for child in children_map.get(current, set()):
            if child not in visited:
                visited.add(child)
                queue.append(child)

    unconnected = part_names - visited
    if not unconnected:
        return

    logger.info("Auto-fixing %d unconnected parts: %s", len(unconnected), unconnected)

    # For each unconnected part, find the best parent and add a fixed joint
    for part_name in unconnected:
        # Find a suitable parent: prefer base_plate, then first reachable structural part
        parent = root
        # Check if the part name hints at its natural parent
        n = part_name.lower()
        if n.startswith("motor_"):
            # Motors go under base_plate bottom
            assembly.joints.append(Joint(
                type="fixed", parent="base_plate", child=part_name,
                parent_anchor="bottom", child_anchor="top",
                distribution_group="motors",
            ))
            continue
        elif n.startswith("wheel_"):
            # Wheels go on motors: wheel_fl → motor_fl
            suffix = n.replace("wheel_", "")
            motor_name = f"motor_{suffix}"
            if motor_name in part_names:
                assembly.joints.append(Joint(
                    type="revolute", parent=motor_name, child=part_name,
                    axis="y", range_deg=(-360, 360),
                    parent_anchor="left", child_anchor="center",
                    no_distribute=True,
                ))
                continue
        elif n.startswith("encoder_"):
            suffix = n.replace("encoder_", "")
            motor_name = f"motor_{suffix}"
            if motor_name in part_names:
                parent = motor_name
        else:
            best = _find_best_parent(part_name, part_names, visited)
            if best is not None:
                parent = best

        # Default: attach to root with a fixed joint
        assembly.joints.append(Joint(
            type="fixed", parent=parent, child=part_name,
            parent_anchor="top", child_anchor="bottom",
        ))
        logger.info("  Auto-connected '%s' -> '%s'", part_name, parent)


def _raise_on_wheel_in_arm(assembly: Assembly) -> None:
    """P1-1: detect hallucinated wheel parts in arm assemblies and raise.

    Per CLAUDE.md: "LLM 给出离谱尺寸/位置时，应该报错让 LLM 重试，
    而不是悄悄修正".  Previously ``_strip_wheel_parts`` silently deleted
    wheel/motor_mount parts, hiding the error from the VLM feedback loop
    so the LLM never learned to stop generating them.

    This function raises ``RuntimeError`` so the error enters
    ``problems_history`` via ``_validate_assembly``'s pattern, giving the
    LLM a chance to regenerate without wheels.
    """
    wheel_keywords = ("wheel", "motor_mount", "电机座", "轮")
    found = [
        p.name for p in assembly.parts
        if any(kw in p.name.lower() for kw in wheel_keywords)
    ]
    if found:
        raise RuntimeError(
            f"Arm assembly contains wheel/motor_mount parts that should "
            f"not exist in a fixed-base arm: {found}. Remove these parts "
            f"and their joints — a fixed-base arm has only base_plate, "
            f"joints (housings), links, and end_effector."
        )


def _strip_wheel_parts(assembly: Assembly) -> Assembly:
    """Remove wheel and wheel-motor parts from the assembly in-place.

    .. deprecated:: P1-1
       Silent deletion of LLM-hallucinated parts violates the CLAUDE.md
       principle "不要在代码里加 hack 让 LLM/外部输入看起来对".  Use
       :func:`_raise_on_wheel_in_arm` instead — it feeds the error back
       into the VLM retry loop so the LLM can correct itself.

    The LLM sometimes hallucinates wheel/motor_mount parts for fixed-base arms,
    which causes VLM verification failures (overlapping parts, wrong wheel
    orientation feedback that confuses the regeneration loop).

    This sanitizer strips any part whose name matches wheel keywords and
    removes their associated joints. Returns the (mutated) assembly.
    """
    wheel_keywords = ("wheel", "motor_mount", "电机座", "轮")
    removed_names: set[str] = set()
    kept_parts: list[Part] = []
    for p in assembly.parts:
        name_lower = p.name.lower()
        if any(kw in name_lower for kw in wheel_keywords):
            removed_names.add(p.name)
            logger.info(
                "Sanitizer: removed wheel part '%s' from arm assembly", p.name
            )
        else:
            kept_parts.append(p)
    if not removed_names:
        return assembly
    kept_joints: list[Joint] = []
    for j in assembly.joints:
        if j.parent in removed_names or j.child in removed_names:
            logger.info(
                "Sanitizer: removed joint '%s' -> '%s' (references stripped part)",
                j.parent, j.child,
            )
        else:
            kept_joints.append(j)
    assembly.parts = kept_parts
    assembly.joints = kept_joints
    logger.info(
        "Sanitizer: stripped %d wheel part(s), %d parts / %d joints remain",
        len(removed_names), len(assembly.parts), len(assembly.joints),
    )
    return assembly


def _normalize_wheel_positions(assembly: Assembly) -> Assembly:
    """Fix wheel joint offsets ONLY when the solved wheels are actually wrong.

    The LLM frequently emits wheel joint ``offset`` values that overlap the
    four wheels (``wheel_fr``/``wheel_rr`` collide) or fling them far outside
    the chassis. BUT ``build_wheeled_base`` / ``compose_dual_arm_assembly``
    already produce correct wheel layouts — and a previous revision of this
    sanitizer UNCONDITIONALLY overwrote those correct offsets, breaking the
    2026-06-26 working run (wheels went from Z=47.6 ground-contact to Z=96
    floating). Regression confirmed by re-solving build_wheeled_base with the
    old sanitizer.

    So this is now CONDITIONAL: solve the assembly first, measure the actual
    wheel positions, and only override when there is a genuine defect —
    overlapping wheels, or wheels far from a sensible 4-corner layout. A
    correct layout is left untouched.
    """
    wheel_joint_suffixes = ("_fl", "_fr", "_rl", "_rr")
    wheel_joints = {
        j.child[-3:].lower(): j for j in assembly.joints
        if j.child[-3:].lower() in wheel_joint_suffixes
        and "wheel" in j.child.lower()
    }
    if len(wheel_joints) < 4:
        return assembly  # not a 4-wheel layout

    base = next(
        (p for p in assembly.parts
         if "base" in p.name.lower() and "plate" in p.name.lower()),
        None,
    )
    if base is None:
        return assembly
    bd = base.dimensions
    base_l = float(bd.get("length", 0) or 0)
    base_w = float(bd.get("width", 0) or 0)
    if base_l < 50 or base_w < 50:
        return assembly  # base dims unreliable

    # Wheel radius for overlap/ground checks.
    wheel_part = next(
        (p for p in assembly.parts if "wheel" in p.name.lower()), None,
    )
    wheel_r = float(
        (wheel_part.dimensions.get("diameter", 0) or 0) / 2.0
        if wheel_part else 45.0
    ) or 45.0

    # --- Solve and inspect the ACTUAL wheel positions before touching anything.
    try:
        from .assembly_solver import AssemblySolver
        solved = AssemblySolver(assembly).solve()
    except Exception:
        return assembly  # can't verify — don't risk corrupting the layout

    wheel_pos = {
        suf: solved[j.child]["position"]
        for suf, j in wheel_joint_suffixes_map(wheel_joints).items()
        if j.child in solved
    }
    if len(wheel_pos) < 4:
        return assembly

    # Defect 1: any pair of wheels overlaps (centers closer than wheel diameter).
    import itertools
    min_pair_dist = min(
        ((wheel_pos[a][0]-wheel_pos[b][0])**2
         + (wheel_pos[a][1]-wheel_pos[b][1])**2)**0.5
        for a, b in itertools.combinations(wheel_pos, 2)
    )
    # Defect 2: wheels not near the ground (Z far from wheel_r).
    avg_z = sum(p[2] for p in wheel_pos.values()) / 4.0
    z_bad = abs(avg_z - wheel_r) > max(wheel_r * 0.6, 30.0)

    if min_pair_dist >= (wheel_r * 1.5) and not z_bad:
        # Layout is fine — do NOT touch it (preserves correct chassis builds).
        return assembly

    logger.info(
        "Sanitizer: wheel layout defective (min_pair_dist=%.0fmm, avg_Z=%.0f "
        "vs wheel_r=%.0f) — overriding offsets to canonical 4-corner layout",
        min_pair_dist, avg_z, wheel_r,
    )

    # Canonical corner offsets (solver X=lateral, Y=forward/back, Z=up).
    half_w = base_w / 2.0
    half_l = base_l / 2.0
    fy = half_l * 0.78
    corners = {"_fl": (-1, +1), "_fr": (+1, +1), "_rl": (-1, -1), "_rr": (+1, -1)}

    fixed = []
    for suf, (xs, ys) in corners.items():
        j = wheel_joints.get(suf)
        if j is None:
            continue
        # Wheel sits coaxial with its parent motor (shared axle). Z=0 relative
        # to the motor center; the motor's own position sets the axle height.
        new_offset = [xs * half_w, ys * fy, 0.0]
        if j.offset != new_offset:
            old = list(j.offset) if j.offset else None
            j.offset = new_offset
            j.parent_anchor = "center"
            j.child_anchor = "center"
            fixed.append((j.child, old, new_offset))

    if fixed:
        logger.info(
            "Sanitizer: reset %d wheel joint offset(s) to canonical 4-corner "
            "layout (base %.0f×%.0f): %s",
            len(fixed), base_l, base_w,
            [(c, [round(v, 1) for v in (o or [])], [round(v, 1) for v in n])
             for c, o, n in fixed],
        )
    return assembly


def wheel_joint_suffixes_map(wheel_joints: dict) -> dict:
    """Helper: return the suffix->joint dict as-is (kept for clarity)."""
    return wheel_joints


def _normalize_gripper_fingers(assembly: Assembly) -> Assembly:
    """Ensure gripper fingers are symmetrically separated, anchored at center.

    The LLM often emits left/right finger joints with face anchors such as
    ``"front"``/``"back"`` (or inconsistent pairs).  Face anchors contribute a
    non-zero positional offset via ``_anchor_offset_for_part`` (e.g. ±Y for
    front/back).  When that anchor offset is then *added* to the explicit
    lateral ``offset`` (±X) by the solver, the resulting 3D displacement can
    exceed the ``_clamp_child_offset`` threshold and be scaled down — but
    worse, in the 4dof_arm audit the LLM's "front"/"back" anchors produced a
    ±Y displacement that, combined with the ±X finger offset, summed to a
    ~477mm vector which ``_clamp_child_offset`` truncated to ~330mm and the
    URDF exporter then emitted as a **322mm** joint origin for
    ``gripper_finger_left``.  The fingers ended up far from the gripper base.

    Root cause: anchor (rotational face) and offset (lateral position) both
    move the child, so they compound.  The fix is to make the anchor
    contribute **rotation only** by forcing ``"center"`` for both parent and
    child anchors.  ``_anchor_offset_for_part(part, "center")`` returns
    ``(0, 0, 0)``, so the solver computes ``child_center = parent_center +
    rot @ offset`` — offset becomes the sole position determinant and the
    intended symmetric ±X gap is preserved exactly.

    This sanitizer:
    1. Detects left/right finger pairs by name.
    2. Forces ``parent_anchor == child_anchor == "center"`` on both joints
       (root-cause fix for the 4dof_arm 322mm URDF origin).
    3. Sets ``no_distribute=True`` to prevent auto-distribution.
    4. Sets explicit lateral (**Y**) offsets perpendicular to the finger
       length, and switches the prismatic axis to ``"y"`` so the grip
       opens/closes in the correct direction.
    """
    finger_left_kw = ("finger_left", "left_finger", "left_gripper",
                      "gripper_left", "左爪", "左指", "左夹", "左手指")
    finger_right_kw = ("finger_right", "right_finger", "right_gripper",
                       "gripper_right", "右爪", "右指", "右夹", "右手指")

    parts_by_name = {p.name: p for p in assembly.parts}

    left_name = None
    right_name = None
    for p in assembly.parts:
        nl = p.name.lower()
        if left_name is None and any(kw in nl for kw in finger_left_kw):
            left_name = p.name
        if right_name is None and any(kw in nl for kw in finger_right_kw):
            right_name = p.name

    if not left_name or not right_name:
        return assembly

    left_joint = None
    right_joint = None
    for j in assembly.joints:
        if j.child == left_name:
            left_joint = j
        elif j.child == right_name:
            right_joint = j

    if not left_joint or not right_joint:
        return assembly

    # L1 fix: anchors contribute rotation only — offset is the sole position
    # determinant.  Prevents anchor (front/back -> +/-Y displacement) + offset
    # (+/-X) from compounding into the 300mm+ URDF origins observed in the
    # 4dof_arm audit (gripper_finger_left measured at 322mm).  With "center"
    # anchors, ``_anchor_offset_for_part(part, "center")`` returns (0,0,0), so
    # the solver computes child_center = parent_center + rot*offset — exactly
    # the intended symmetric geometry.
    for j in (left_joint, right_joint):
        j.parent_anchor = "center"
        j.child_anchor = "center"

    # Disable auto-distribution so explicit offsets are the sole lateral factor
    left_joint.no_distribute = True
    right_joint.no_distribute = True

    # Compute the lateral gap between the two fingers.
    #
    # Coordinate convention — TWO frames (fixed 2026-06-22):
    #
    #   PART-LOCAL (FreeCAD makeBox, the STL mesh):
    #     finger length → X,  finger width → Y,  height → Z
    #
    #   SOLVER/WORLD (where this offset lives; what the renderer sees after
    #   swap_xy R_z(-90°) maps part-local X → world -Y, Y → world +X):
    #     forward (front/back) → Y    (ANCHOR_DIRECTIONS: front=(0,-1,0))
    #     lateral (left/right) → X    (ANCHOR_DIRECTIONS: left=(-1,0,0))
    #     up/down             → Z
    #
    # The finger offset is in WORLD coords: gap on X (lateral, so fingers
    # straddle the arm centreline) and forward on Y (so both fingers
    # protrude ahead of the base).  After swap_xy the finger STL's length
    # (part-local X) renders along world Y — aligned with the forward
    # offset — and width (part-local Y) renders along world X — aligned
    # with the gap.  So the two fingers appear as parallel bars extending
    # forward, separated left/right.  This reads as a gripper.
    #
    # PREVIOUS BUG (pre-2026-06-22): offset was (forward_x, ±gap, 0) —
    # forward on X, gap on Y.  After swap_xy both fingers landed on the
    # same side of the arm (world -Y), separated along the arm-length
    # axis — they read as two extra links, not a gripper.
    #
    # Geometric invariant: gap > finger_width guarantees the world-X AABBs
    # (finger width on world X after swap) do not overlap.  6mm clearance.
    parent_part = parts_by_name.get(left_joint.parent)
    gap = 22.0
    base_length = 28.0
    finger_w = 14.0
    finger_l = 60.0
    if parent_part and parent_part.dimensions:
        w = parent_part.dimensions.get("width",
                    parent_part.dimensions.get("depth", 50))
        base_length = parent_part.dimensions.get("length", 28.0)
        finger_part = parts_by_name.get(left_joint.child)
        if finger_part and finger_part.dimensions:
            finger_w = finger_part.dimensions.get("width", 14.0)
            finger_l = finger_part.dimensions.get("length", 60.0)
        # gap must exceed finger width so AABBs separate on Y.  Cap by the
        # parent base width so fingers stay within the gripper footprint.
        min_gap = finger_w + 6.0          # guarantee > width + 6mm grip gap
        max_gap = w / 2.0 - 2.0           # finger stays within base
        if max_gap < min_gap:
            # Base too narrow to fit both fingers inside — prefer the
            # geometric invariant (no intersection) over footprint fit.
            gap = min_gap
        else:
            gap = max(min_gap, min(min_gap * 1.25, max_gap))

    # Prismatic axis is X (lateral): fingers slide toward/away from each
    # other along X (open/close the grip), perpendicular to the forward Y
    # direction.  This matches the gap axis so closing the grip moves each
    # finger toward the centreline on X.
    for j in (left_joint, right_joint):
        j.axis = "x"

    # Push fingers forward along Y so the main bar fully protrudes beyond
    # the gripper base face — but WHICH direction is "forward" depends on
    # where the parent chain attaches to the gripper base.  The fingers
    # must point AWAY from the arm (the parent link), not back into it.
    #
    # The arm-side link (e.g. wrist_link) connects to gripper_base via a
    # joint whose child_anchor names the face it mounts on.  If the parent
    # attaches on the 'back' face, fingers go to 'front' (+X); if on
    # 'front', fingers go to 'back' (-X).  This was previously hardcoded
    # to -X, which drove the fingers back into the wrist_link whenever the
    # arm attached on the back face (the 4dof_arm topology) — causing the
    # wrist_link/gripper_finger intersection that the VLM loop could never
    # resolve.
    parent_face = None
    for j in assembly.joints:
        if j.child == left_joint.parent and j.child_anchor != "center":
            parent_face = j.child_anchor
            break
    # ANCHOR_DIRECTIONS (assembly_solver): front=(0,-1,0), back=(0,1,0).
    # Fingers protrude ALONG Y (forward/back), gap is on X (lateral).
    # Parent on back (+Y)  -> fingers toward front (-Y), away from arm.
    # Parent on front (-Y) -> fingers toward back (+Y), away from arm.
    # This convention matches the renderer's swap_xy (R_z(-90°)): the
    # finger STL has length on FreeCAD-X, which swap_xy maps to world -Y,
    # so the finger's long edge visually extends along world Y — the same
    # axis as the forward offset.  Gap on X means the two fingers straddle
    # the arm centreline left/right (world X), which reads as a gripper
    # in the render.  (The old convention put forward on X and gap on Y,
    # which after swap_xy placed both fingers on the same side of the arm
    # — they read as two extra links lined up, not a gripper.)
    if parent_face == "back":
        forward_sign = -1.0    # arm at +Y → fingers to -Y (front)
    elif parent_face == "front":
        forward_sign = +1.0    # arm at -Y → fingers to +Y (back)
    else:
        forward_sign = -1.0    # default: fingers forward (-Y)
    forward_y = forward_sign * (base_length / 2.0 + finger_l / 2.0)
    left_joint.offset = (-gap, forward_y, 0.0)
    right_joint.offset = (gap, forward_y, 0.0)

    # Dynamic range clamp: prevent finger collision.
    # The closing displacement moves both fingers toward center (mimic=-1).
    # Max safe close = gap - finger_w/2 - 1mm_margin.
    # At this displacement, inner faces have >= 2mm clearance.
    max_close = gap - finger_w / 2.0 - 1.0
    for j in (left_joint, right_joint):
        if j.type == "prismatic" and j.range_deg:
            lo, hi = j.range_deg
            hi = min(hi, max_close)
            lo = min(lo, -1.0)  # ensure at least 1mm opening range
            j.range_deg = (lo, hi)
            logger.info(
                "Sanitizer: clamped gripper finger %s range to "
                "(%.1f, %.1f) mm (gap=%.1f, finger_w=%.1f)",
                j.child, lo, hi, gap, finger_w,
            )

    # Clear connection_method on prismatic finger joints — sliding interfaces
    # are not fastenings.  The LLM frequently marks them "bolted" which is
    # mechanically wrong (you cannot bolt a sliding finger to the rail) and
    # causes the CAD feature engine to generate spurious bolt holes on the
    # linear rail grooves.
    for j in (left_joint, right_joint):
        if j.type == "prismatic" and j.connection is not None:
            logger.info(
                "Sanitizer: cleared connection_method on prismatic joint "
                "%s->%s (sliding fit, not a fastening)",
                j.parent, j.child,
            )
            j.connection = None

    # Wire the right finger to mimic the left (antagonistic grip).
    # Without this, the URDF exporter emits two independent prismatic joints,
    # so opening/closing one finger does not move the other — the gripper
    # cannot actually grasp.  mimic_multiplier=-1 makes them move symmetrically
    # toward/away from centre.
    if not left_joint.mimic_joint and not right_joint.mimic_joint:
        right_joint.mimic_joint = left_joint.child
        right_joint.mimic_multiplier = -1.0
        right_joint.mimic_offset = 0.0
        logger.info(
            "Sanitizer: set %s to mimic %s (multiplier=-1.0) for "
            "antagonistic grip",
            right_joint.child, left_joint.child,
        )

    logger.info(
        "Sanitizer: normalized gripper fingers '%s'/'%s' — "
        "anchors=center/center, axis=x, gap=±%.1fmm (X), z_lift=0",
        left_name, right_name, gap,
    )
    return assembly


def _validate_proportions(assembly: Assembly) -> Assembly:
    """Validate part proportions and raise on physically bad ratios.

    P1-1: previously this sanitizer SILENTLY CLAMPED disproportionate
    dimensions (gripper width, link length, link cross-section) so the
    rendered assembly "looked right".  Per CLAUDE.md ("不要在代码里加
    hack 让 LLM/外部输入看起来对"), clamp-and-pretend masks the real
    data-quality issue from the VLM retry loop so the LLM never learns
    to produce coherent dimensions.

    Now the function COLLECTS every proportion violation and raises a
    single RuntimeError describing all of them, so the error enters
    ``problems_history`` and the LLM gets a chance to regenerate with
    corrected dimensions.  Returns ``assembly`` unchanged when valid.

    Checks:
    1. gripper_base width ≤ 1.8 × parent link width
    2. Consecutive link length ratio < 3.0
    3. link cross-section ≥ 0.55 × joint diameter (width) / 0.50× (height)
    """
    problems: list[str] = []
    parts_by_name = {p.name: p for p in assembly.parts}

    for joint in assembly.joints:
        parent = parts_by_name.get(joint.parent)
        child = parts_by_name.get(joint.child)
        if not parent or not child:
            continue
        if not parent.dimensions or not child.dimensions:
            continue

        parent_w = parent.dimensions.get("width", 0)
        child_w = child.dimensions.get("width", 0)
        parent_l = parent.dimensions.get("length", 0)
        child_l = child.dimensions.get("length", 0)

        # Check 1: gripper_base width should not dwarf the parent link.
        # P1 correction: the original 1.8x threshold was too tight — a
        # real gripper base houses a servo (SG90 = 22mm wide) plus linear
        # guide rails plus finger mounts, so it is naturally 2-2.5x the
        # wrist link width.  3.0x captures grossly oversized grippers
        # (e.g. 90mm gripper on a 20mm wrist) without rejecting the
        # standard SG90 grip-per-base (50mm on a 22mm wrist = 2.27x).
        child_nl = child.name.lower()
        if ("gripper" in child_nl and "base" in child_nl
                and parent_w > 0 and child_w > 0):
            max_w = parent_w * 3.0
            if child_w > max_w:
                problems.append(
                    f"gripper_base '{child.name}' width {child_w:.0f}mm > "
                    f"3.0x parent '{parent.name}' width {parent_w:.0f}mm "
                    f"(limit {max_w:.0f}mm); reduce the gripper width or "
                    f"widen the parent link"
                )

        # Check 2: consecutive link length ratio
        parent_nl = parent.name.lower()
        if (parent_l > 0 and child_l > 0
                and "link" in parent_nl and "link" in child_nl):
            ratio = max(parent_l, child_l) / min(parent_l, child_l)
            if ratio > 3.0:
                problems.append(
                    f"consecutive links '{parent.name}' ({parent_l:.0f}mm) "
                    f"and '{child.name}' ({child_l:.0f}mm) have length "
                    f"ratio {ratio:.1f} > 3.0; make adjacent link lengths "
                    f"comparable (ratio < 3.0)"
                )

        # Check 3: joint-link cross-section consistency.
        # Joint cylinders (with "diameter") are often much fatter than the
        # links they connect to (e.g. diameter=40 vs link 25×15).  When the
        # joint is centred on the link's end face, the joint body extends
        # well beyond the link profile on all sides, making it look like the
        # joint "swallows" the link — visually read as parts intersecting.
        # Enforce: link cross-section ≥ 0.55 × joint diameter in both width
        # and height.  This keeps the link profile visually comparable to the
        # joint so the connection looks clean rather than overlapping.
        parent_d = parent.dimensions.get("diameter", 0)
        child_d = child.dimensions.get("diameter", 0)
        link_part = None
        joint_d = 0
        joint_name = ""
        # The cross-section rule targets ARM LINKS (the bars between joints),
        # not chassis/base plates.  A base_plate is "structural" but is a
        # thin slab by design (prompt allows 3-8mm) — applying the arm-link
        # 0.50×joint-diameter rule to it rejects every legitimate base
        # plate (8mm < 0.5×40mm = 20mm).  Restrict to parts whose name
        # actually reads as an arm link.
        def _is_arm_link(pt) -> bool:
            nl = pt.name.lower()
            return (
                pt.category in ("structural", "link")
                and ("link" in nl or "arm" in nl)
                and not any(b in nl for b in (
                    "base", "plate", "chassis", "foot", "mount",
                ))
            )
        if parent_d > 0 and "joint" in parent_nl and _is_arm_link(child):
            link_part = child
            joint_d = parent_d
            joint_name = parent.name
        elif child_d > 0 and "joint" in child_nl and _is_arm_link(parent):
            # parent is the link, child is the joint
            link_part = parent
            joint_d = child_d
            joint_name = child.name
        if link_part is not None and joint_d > 0:
            min_w = joint_d * 0.55
            min_h = joint_d * 0.50
            link_w = link_part.dimensions.get("width", 0)
            link_h = link_part.dimensions.get("height", 0)
            if link_w > 0 and link_w < min_w:
                problems.append(
                    f"link '{link_part.name}' width {link_w:.0f}mm < "
                    f"0.55x joint '{joint_name}' diameter {joint_d:.0f}mm "
                    f"(need >= {min_w:.0f}mm); the joint visually swallows "
                    f"the link — widen the link"
                )
            if link_h > 0 and link_h < min_h:
                problems.append(
                    f"link '{link_part.name}' height {link_h:.0f}mm < "
                    f"0.50x joint '{joint_name}' diameter {joint_d:.0f}mm "
                    f"(need >= {min_h:.0f}mm); the joint visually swallows "
                    f"the link — increase the link height"
                )

    if problems:
        raise RuntimeError(
            "Proportion validation failed: " + "; ".join(problems)
        )

    return assembly


def _ensure_arm_default_angles(assembly: Assembly) -> Assembly:
    """Inject non-zero default_angles for arm pitch joints that lack them.

    Even with prompt rules asking for bent postures, the LLM frequently emits
    all-zero default_angles for the pitch joints.  Combined with vertical
    top/bottom anchors, zero angles stack every part into a single straight
    column with no 3D extent — the VLM then sees "1 part" and the
    motion-collision sweep flags self collisions because the links overlap
    end-to-end.

    This sanitizer operates **per joint**, not all-or-nothing:

    1. **Clean**: remove default_angles entries whose key is NOT the child of a
       revolute joint.  The LLM sometimes emits entries for fixed-joint
       children (e.g. ``gripper_base``) or random structural parts — these are
       meaningless (fixed joints cannot rotate) and pollute the pose.
    2. **Preserve**: for each revolute joint where the LLM supplied a non-zero
       angle, keep it unchanged.
    3. **Inject**: for each revolute pitch joint that is zero or missing,
       synthesise a natural zig-zag bend by alternating the sign.

    A single stray non-zero value (e.g. the LLM setting only the wrist roll)
    no longer causes the sanitizer to skip every other pitch joint.
    """
    existing = dict(assembly.default_angles or {})

    # Build the set of child names that belong to revolute joints.
    revolute_children = {j.child for j in assembly.joints if j.type == "revolute"}

    # --- Clean: strip keys that are not revolute joint children. ---
    cleaned: dict[str, float] = {}
    removed: list[str] = []
    for k, v in existing.items():
        if k in revolute_children:
            try:
                cleaned[k] = float(v)
            except (TypeError, ValueError):
                removed.append(k)
        else:
            removed.append(k)
    if removed:
        logger.info(
            "Sanitizer: removed non-revolute default_angles keys: %s",
            removed,
        )

    # Zero out roll/yaw default angles that are NOT the base yaw joint.
    # Only pitch joints (axis=x, the up/down bend) and the base yaw
    # (first axis=z joint, turning the whole arm left/right) should
    # carry a non-zero default — they define the arm's reach pose.
    # Other roll (axis=y, spinning the end effector) and yaw joints
    # rotate the END of the arm, which breaks gripper symmetry in the
    # default pose (a 35° wrist roll tilts the whole gripper sideways,
    # making the two fingers project asymmetrically in world space).
    # The LLM frequently emits non-zero wrist roll; zero it here.
    revolute_joints_tmp = [
        j for j in assembly.joints if j.type == "revolute"
    ]
    # Base yaw = the FIRST z-axis joint of EACH arm chain (not just the
    # global first). A dual-arm assembly has two base yaws (arm_l + arm_r),
    # both carrying the collision-aware splay — zeroing the right one
    # destroyed the symmetric park pose and made the arms asymmetric. Detect
    # base-yaw children per arm-prefix so every arm keeps its base yaw.
    import re as _re_yaw
    def _arm_yaw_prefix(name: str) -> str:
        m = _re_yaw.match(r"(arm_[lr]|left_|right_)", name.lower())
        return m.group(1) if m else ""
    base_yaw_children: set[str] = set()
    _seen_yaw_prefixes: set[str] = set()
    for j in revolute_joints_tmp:
        if j.axis == "z":
            pref = _arm_yaw_prefix(j.child)
            if pref not in _seen_yaw_prefixes:
                _seen_yaw_prefixes.add(pref)
                base_yaw_children.add(j.child)
            elif not pref:
                # No-prefix arm: only the very first z joint is base yaw.
                if not base_yaw_children:
                    base_yaw_children.add(j.child)
    zeroed_roll = []
    for j in revolute_joints_tmp:
        if j.axis in ("y", "z") and j.child not in base_yaw_children:
            cur = cleaned.get(j.child)
            if cur is not None and abs(float(cur)) > 1e-6:
                zeroed_roll.append((j.child, cur))
                cleaned[j.child] = 0.0
    if zeroed_roll:
        logger.info(
            "Sanitizer: zeroed non-base roll/yaw default_angles "
            "(they tilt the gripper in the default pose): %s",
            zeroed_roll,
        )

    # Detect an arm-like assembly: at least 2 revolute joints and at least 1
    # link-like structural part.
    revolute_joints = revolute_joints_tmp
    has_link = any(
        _is_link_like(p.name) or _is_end_effector(p.name)
        for p in assembly.parts
    )
    if len(revolute_joints) < 2 or not has_link:
        assembly.default_angles = cleaned
        return assembly

    # Helper: identify ARM pitch joints while EXCLUDING drive-train joints.
    # Wheels/tires are axis=x (the axle runs along X for differential drive)
    # and motors sit on the same axle — they are NOT arm bend joints. Treating
    # them as pitch joints gave wheels spurious ±35° angles (rotated render)
    # AND pushed the real arm joints to higher pitch_idx, skipping the
    # shoulder's "force negative" branch so the arm drooped. Defined early so
    # both the zig-zag check and the pitch_children list use the same filter.
    def _is_arm_pitch_joint(j) -> bool:
        if j.axis != "x" or j.type != "revolute":
            return False
        c = j.child.lower()
        if any(c.startswith(p) for p in ("wheel_", "tire_")):
            return False
        if "motor" in c:
            return False
        return True

    # --- Anchor-consistency check (clean arm convention). ---
    # Under the clean convention, pitch joints (axis=x) must use front/back
    # anchors so the link's `length` dimension positions the next axis. A
    # pitch joint on top/bottom anchors means the LLM ignored the rule; the
    # IK link lengths still come from part dimensions, but the solver will
    # stack the arm into a vertical column. Warn (non-blocking) so this is
    # visible in logs without rejecting the assembly.
    mismatched = [
        j.child for j in revolute_joints
        if j.axis == "x"
        and {j.parent_anchor, j.child_anchor} != {"front", "back"}
        and j.parent_anchor in ("top", "bottom")
    ]
    if mismatched:
        logger.warning(
            "Sanitizer: %d arm pitch joint(s) use top/bottom anchors instead "
            "of front/back (clean convention): %s. Link lengths remain correct "
            "(read from part dimensions) but the solver may stack the arm "
            "vertically.",
            len(mismatched), mismatched,
        )

    # --- Per-joint fill: inject bends for pitch joints that are zero/missing. ---
    injected: dict[str, float] = dict(cleaned)
    pitch_index = 0
    filled: list[tuple[str, float]] = []

    # --- Over-fold detection: an arm whose pitch joints all bend the same
    # direction curls up on itself, crushing the end-effector into the
    # base and making the gripper impossible to resolve in renders (the
    # VLM then reports "gripper missing").  The LLM frequently emits
    # -35/-35/-35 (cumulative -105°), folding the arm into a tight coil.
    # A natural arm pose alternates direction (zig-zag): -45/-30/+15.
    # If the cumulative same-sign pitch exceeds 90°, override with a
    # canonical zig-zag so the arm extends enough to expose the gripper.
    # Only pitch joints (axis=x) are considered — yaw (axis=z) and roll
    # are not "bend" joints and must be left alone. Wheels/motors are also
    # axis=x (axle along X) but are drive-train parts, NOT arm bends —
    # exclude them (same filter as pitch_children above).
    pitch_joints_arm = [
        j for j in revolute_joints if _is_arm_pitch_joint(j) and j.child
    ]
    pitch_vals = [
        float(injected.get(j.child, 0.0) or 0.0) for j in pitch_joints_arm
    ]
    nonzero_pitch = [v for v in pitch_vals if abs(v) > 1e-6]
    _overrode_to_zigzag = False
    if len(nonzero_pitch) >= 2:
        all_same_sign = (
            all(v < -1e-6 for v in nonzero_pitch)
            or all(v > 1e-6 for v in nonzero_pitch)
        )
        cumulative = sum(nonzero_pitch)
        # Threshold scales with joint count: 2 pitch joints folding
        # -30°/-30° = -60° already curls the forearm back; 3 joints
        # need more room.  ~25° per joint is the "natural bend" ceiling
        # before the arm starts folding on itself.
        fold_threshold = 25.0 * len(nonzero_pitch)
        if all_same_sign and abs(cumulative) > fold_threshold:
            logger.warning(
                "Sanitizer: arm over-folded (all pitch joints same sign, "
                "cumulative %.0f° > %.0f° threshold for %d joints). "
                "Overriding with zig-zag so the gripper is visible: %s",
                cumulative, fold_threshold, len(nonzero_pitch),
                [j.child for j in pitch_joints_arm],
            )
            # Zig-zag template: alternate sign so the arm extends
            # outward instead of curling in.  Generalised to any pitch-joint
            # count via knowledge/arm_topology.zigzag_angles (the legacy
            # hard-coded [-45,30] / [-45,-30,15,-10] wrapped arbitrarily for
            # >4 pitch joints, producing a curled pose on 7-DOF arms).
            n = len(pitch_joints_arm)
            _zigzag_seq = zigzag_angles(n)
            for idx, j in enumerate(pitch_joints_arm):
                injected[j.child] = _zigzag_seq[idx % len(_zigzag_seq)]
            _overrode_to_zigzag = True
    for j in revolute_joints:
        # Skip drive-train joints (wheels/tires/motors) — they are axis=x like
        # arm pitch joints but must NOT get a bend angle (a rotated wheel
        # renders as a tilted disc). Only fill arm-chain joints.
        if not _is_arm_pitch_joint(j) and j.axis != "z" and j.axis != "y":
            continue
        # Base yaw (axis=z): clamp to ±10° for a SINGLE arm so it points
        # forward. BUT for a wheeled DUAL-arm assembly, the base yaw carries
        # the collision-aware splay (±30° outward) configured by
        # _configure_collision_aware_dual_arm — zeroing it would destroy the
        # anti-collision park pose and make the arms overlap. Detect dual-arm
        # (both arm_l_ and arm_r_ prefixes present) and PRESERVE those yaws.
        _is_dual_arm = (
            any(p.name.startswith("arm_l_") for p in assembly.parts)
            and any(p.name.startswith("arm_r_") for p in assembly.parts)
        )
        if j.axis == "z" and pitch_index == 0 and not _is_dual_arm:
            # Base yaw: the default (home) pose should point the arm
            # straight forward (yaw = 0).  Any non-zero yaw rotates the
            # whole arm, which carries through to the gripper and makes
            # the two fingers project asymmetrically in world space
            # (a 10° yaw offsets the ±20mm gap into a 39mm world-Y
            # difference).  The VLM verifies the home pose from a world
            # view, so symmetry matters here.  Yaw for reaching different
            # directions is exercised by the workspace/motion-sweep
            # checks, not baked into the default pose.
            injected[j.child] = 0.0
            pitch_index += 1
            continue

        # Roll joints (axis=y, spinning the end effector) must stay at 0°
        # in the default pose — a non-zero roll tilts the gripper and
        # breaks finger symmetry.  They were zeroed in the Clean phase
        # above; skip the fill so we don't reinject a value.
        if j.axis == "y":
            if injected.get(j.child) is None:
                injected[j.child] = 0.0
            continue

        current = injected.get(j.child)
        if current is not None and abs(float(current)) > 1e-6:
            # LLM explicitly gave a non-zero angle for this joint — keep it.
            pitch_index += 1
            continue

        # Compute a bend from the joint's range.  Cap at 35 degrees — larger
        # values (e.g. 90° from 30% of a 300° range) fold the arm back on
        # itself, causing motion-collision sweep failures and COM instability.
        lo, hi = j.range_deg if j.range_deg else (-120.0, 120.0)
        try:
            lo_f, hi_f = float(lo), float(hi)
        except (TypeError, ValueError):
            lo_f, hi_f = -120.0, 120.0
        span = hi_f - lo_f
        magnitude = max(15.0, min(abs(span) * 0.20, 35.0))
        # Clamp into the legal range so the angle is realisable.
        magnitude = min(magnitude, abs(span) / 2.0 - 1.0) if span > 2.0 else 15.0
        if magnitude < 5.0:
            magnitude = 15.0
        # Alternate sign to produce a zig-zag (natural-looking) posture.
        sign = -1.0 if (pitch_index % 2 == 0) else 1.0
        angle = sign * magnitude
        # Keep inside [lo, hi].
        angle = max(lo_f + 1.0, min(hi_f - 1.0, angle))
        injected[j.child] = round(angle, 1)
        filled.append((j.child, round(angle, 1)))
        pitch_index += 1

    if filled:
        logger.info(
            "Sanitizer: injected default_angles bends for arm '%s': %s",
            assembly.name, filled,
        )

    # --- Rising-arm pose (front/back convention). ---
    # With front/back anchors + axis=x, 0° = the link lies horizontal
    # (extending forward in -Y).  Pitch angles tilt each link upward.
    # For the arm to rise in Z (look 3D, not flat) ALL pitch joints must
    # tilt the SAME direction (negative = upward) so their effects
    # reinforce rather than cancel.  This is the opposite of the old
    # zig-zag logic, which was designed for the top/bottom convention
    # (where the arm starts vertical and alternating signs create bends).
    #
    # Each pitch is clamped to a moderate magnitude so the arm doesn't
    # fold back on itself (too steep) or stay flat (too horizontal).
    #
    # SKIPPED when the over-fold detector already applied a zig-zag: the
    # zig-zag alternates signs on purpose (to extend the arm and expose
    # the gripper), so forcing all-same-sign here would undo it and fold
    # the arm right back. The two pose strategies are mutually exclusive.
    # IMPORTANT: wheels are also axis=x (the axle runs along X for a
    # differential drive), so they must be EXCLUDED — otherwise they get
    # mis-treated as arm pitch joints, given spurious ±35° angles (the
    # wheels render rotated), AND they push the real arm joints to higher
    # pitch_idx so the shoulder's "force negative" branch is skipped and
    # the arm gets forced positive (drooping). The _is_arm_pitch_joint
    # helper (defined above, near the early-return) excludes wheel_/tire_/
    # motor_ so both the zig-zag check and this list stay drive-train-free.
    pitch_children = [
        j.child for j in revolute_joints
        if _is_arm_pitch_joint(j) and j.child in injected
    ]
    range_limit: dict[str, float] = {}
    for j in revolute_joints:
        if _is_arm_pitch_joint(j) and j.range_deg:
            try:
                lo_r, hi_r = float(j.range_deg[0]), float(j.range_deg[1])
                range_limit[j.child] = min(abs(lo_r), abs(hi_r)) - 1.0
            except (TypeError, ValueError):
                pass
    adjusted: list[tuple[str, float, float]] = []
    # Group pitch joints per-arm so EACH arm's first pitch joint (the
    # shoulder) is treated as shoulder (force negative = tilt up), not just
    # the global first joint. Without this, a dual-arm assembly treats the
    # right arm's shoulder as an elbow (pitch_idx > 0) and forces it positive
    # → the right arm droops while the left rises ("一个向上一个向下").
    # Arm prefix = the leading "arm_l" / "arm_r" / "left_" / "right_" token;
    # fall back to "" (treat as one chain) if no prefix.
    import re as _re
    def _arm_prefix(name: str) -> str:
        m = _re.match(r"(arm_[lr]|left_|right_)", name.lower())
        return m.group(1) if m else ""
    _seen_shoulders: set[str] = set()
    pitch_idx = 0
    for child in pitch_children:
        val = float(injected[child])
        # Is this the FIRST pitch joint of its arm chain? → shoulder.
        pref = _arm_prefix(child)
        is_shoulder = pref not in _seen_shoulders
        _seen_shoulders.add(pref)
        # Moderate cap so the arm tilts up without folding back.
        if is_shoulder:
            cap = 35.0      # Shoulder: sets the overall reach angle.
            min_mag = 30.0  # Minimum shoulder tilt for a working-arm look.
        else:
            cap = 40.0      # Subsequent pitches reinforce the rise.
            min_mag = 30.0  # Minimum elbow/wrist tilt.
        rl = range_limit.get(child)
        if rl is not None and rl > 0:
            cap = min(cap, rl)
        clamped = max(-cap, min(cap, val))
        if is_shoulder:
            # Shoulder (first pitch of THIS arm): force negative (upward tilt)
            # so the arm rises in Z.  This is the only joint that MUST be
            # same-sign for the arm to point up rather than lie flat. Applied
            # per-arm so a dual-arm assembly tilts BOTH shoulders up.
            if clamped > 0:
                clamped = -clamped
            if abs(clamped) < min_mag:
                clamped = -min(min_mag, cap)
        else:
            # Subsequent pitch joints (elbow, wrist): force POSITIVE
            # (opposite of shoulder).  The home pose must show an
            # EXTENDED arm, not a folded one.  The LLM systematically
            # emits same-sign angles (e.g. shoulder -30 + elbow -35),
            # which fold the arm into a coil and crush the gripper into
            # the base.  Forcing elbow/wrist positive guarantees the arm
            # extends outward regardless of what the LLM emitted.
            # Previous "preserve sign for large values" logic still
            # produced folds when the LLM gave -35 (>= min_mag), so the
            # pose ended up coiled in every E2E run.
            clamped = abs(clamped)  # force positive
            if clamped < min_mag:
                clamped = min_mag
        if abs(clamped - val) > 0.05:
            adjusted.append((child, val, clamped))
            injected[child] = round(clamped, 1)
        pitch_idx += 1

    if adjusted:
        logger.info(
            "Sanitizer: rising-arm default_angles for '%s': %s",
            assembly.name,
            [(c, f"{old:.0f}->{new:.0f}") for c, old, new in adjusted],
        )

    assembly.default_angles = injected

    # --- Base-plate sizing (COM stability, added 2026-06-22). ---
    # A 4-DOF arm with ~400mm of link reach generates a center of mass that
    # projects ~100mm forward of the base origin in the home pose.  When the
    # LLM emits a small base plate (e.g. 200×150), the support polygon
    # (±length/2 = ±100mm) lands exactly on the COM edge, so the
    # com_stability check fails by 1-2mm on a geometrically-correct arm.
    # Ensure the base plate's LENGTH (solver Y, the arm reach direction) is
    # at least 60% of the total arm link reach so the support polygon has
    # margin.  Width (solver X) is left alone — lateral COM offset is small.
    #
    # WHEELED-ROBOT GUARD: a wheeled chassis's base_plate is the CHASSIS deck,
    # sized by mobile_base_gen from real UGV proportions (Husky-class), with
    # wheels attached as children. Enlarging it here to chase an arm-COM
    # offset blows the deck up to 1176/1400mm (observed in
    # data/runs/wheeled_arm/20260627_*), which pushes the child wheels up to
    # Z≈248mm (floating ~200mm off the ground) and outward to XY≈246mm — the
    # "wheels above Z=248 / parts misplaced" VLM failures. Wheeled robots are
    # supported by their WHEELS (ground contact), not by a bench base plate,
    # so the fixed-arm COM-on-plate assumption does not apply. Skip entirely
    # when the assembly has wheels.
    has_wheels = any(
        "wheel" in p.name.lower() for p in assembly.parts
    )
    base_part = next(
        (p for p in assembly.parts
         if "base" in p.name.lower() and "plate" in p.name.lower()),
        None,
    )
    link_parts = [
        p for p in assembly.parts
        if _is_link_like(p.name) or "joint" in p.name.lower()
    ]
    if base_part and link_parts and not has_wheels:
        # Effective forward COM offset of a zig-zag arm. Real solves show the
        # COM projects ~0.30× the raw link-sum forward of the base (measured:
        # 400mm links → COM Y≈-110 to -132mm across LLM-generated poses).
        # Using 0.32 with a margin covers the observed variance.
        #
        # AXIS MAPPING (verified against AssemblySolver + assembly_verifier):
        #   - The arm reaches along solver -Y (front/back anchors).
        #   - base LENGTH is the Y (forward) dimension; WIDTH is X (lateral).
        #   - assembly_verifier.build_support_polygon uses (cy ± length/2)
        #     for the forward extent, so LENGTH must cover the COM offset.
        #   - An earlier revision enlarged WIDTH by mistake (treating width as
        #     Y), which left the forward support edge too short → com_stability
        #     failed at margin -10.6mm. Enlarge LENGTH here.
        total_reach = sum(
            float(p.dimensions.get("length", 0) or 0)
            + float(p.dimensions.get("diameter", 0) or 0)
            for p in link_parts
        )
        com_forward_mm = total_reach * 0.40
        # Support polygon must extend past the COM by a 35mm stability margin
        # (LLM pose variance can shift COM ~20mm run-to-run, and the verifier
        # uses a slightly different mass model than this estimate), so base
        # LENGTH (solver Y, the reach direction) >= 2*(COM + margin).
        min_base_length = max(320.0, 2.0 * (com_forward_mm + 35.0))
        cur_length = float(base_part.dimensions.get("length", 0) or 0)
        if 0 < cur_length < min_base_length:
            base_part.dimensions["length"] = min_base_length
            logger.info(
                "Sanitizer: enlarged base_plate '%s' length %.0f → %.0fmm "
                "so the support polygon covers the arm COM (forward≈%.0fmm "
                "along Y, need length≥%.0fmm)",
                base_part.name, cur_length, min_base_length,
                com_forward_mm, min_base_length,
            )

    # Clamp arm revolute joint ranges to a physically-reasonable workspace
    # so the e2e motion-collision sweep does not articulate into a
    # self-collision extreme. A ±180° base yaw or ±150° elbow is a servo
    # spec limit, not a usable workspace — at those extremes the arm hits
    # the chassis or the other arm (穿模). Real robots ship with narrower
    # software limits than the raw servo range. Cap pitch to ±90°.
    #
    # Base yaw (axis=z) is clamped ASYMMETRICALLY: the home angle's sign
    # sets the allowed direction, and the range must not cross 0° (the
    # midline). A left arm yawed -30° at home may rotate further outward
    # (more negative) but must not cross into +territory where it swings
    # into the right arm's workspace (the collision at +60° on
    # arm_l_base_yaw). Symmetric cap on yaw still let the arm reach the
    # other side → collision. So: yaw range = [min(home, -10), max(home,
    # home)] clamped to ±90°, never containing both signs.
    _ARM_PITCH_CAP = 90.0   # forward (downward reach) — generous
    _ARM_PITCH_BACK_CAP = 30.0  # backward (folds toward base) — tight, avoids 穿模
    _ARM_YAW_CAP = 90.0
    for j in revolute_joints:
        if not _is_arm_pitch_joint(j) and j.axis != "z":
            continue
        if not j.range_deg:
            continue
        lo, hi = float(j.range_deg[0]), float(j.range_deg[1])
        home = float(injected.get(j.child, 0.0) or 0.0)
        if j.axis == "z":
            # Base yaw: keep on the home side of the midline.
            if home >= 0:
                new_lo, new_hi = max(lo, 0.0), min(hi, _ARM_YAW_CAP)
            else:
                new_lo, new_hi = max(lo, -_ARM_YAW_CAP), min(hi, 0.0)
        else:
            # Pitch: asymmetric. The "forward" direction (the sign that
            # reaches into the workspace, = home sign) is generous (±90°).
            # The "backward" direction (opposite sign, folds the arm back
            # toward/into the base_plate) is tight (±30°). A fixed ±60° cap
            # was unstable across LLM-generated arm lengths: a long arm at
            # +60° backward swings the upper_arm_link into the base_plate
            # (base_plate ↔ upper_arm_link collision). The asymmetric cap
            # adapts: forward reach stays wide, backward fold is bounded.
            if home >= 0:
                new_lo, new_hi = max(lo, -_ARM_PITCH_BACK_CAP), min(hi, _ARM_PITCH_CAP)
            else:
                new_lo, new_hi = max(lo, -_ARM_PITCH_CAP), min(hi, _ARM_PITCH_BACK_CAP)
        if new_hi - new_lo < 10.0:
            continue  # range already tiny — don't collapse it
        if (new_lo, new_hi) != (lo, hi):
            logger.info(
                "Sanitizer: clamped arm joint '%s' range [%.0f, %.0f] → "
                "[%.0f, %.0f] (home %.0f, %s, avoid workspace-extreme "
                "self-collision)",
                j.child, lo, hi, new_lo, new_hi, home,
                "yaw midline-safe" if j.axis == "z" else f"cap ±{_ARM_PITCH_CAP:.0f}°",
            )
            j.range_deg = (new_lo, new_hi)

    return assembly


def _validate_assembly(assembly: Assembly) -> None:
    """Validate an Assembly for basic correctness.

    Raises RuntimeError for critical issues, logs warnings for minor ones.
    """
    if not assembly.parts:
        raise RuntimeError("Assembly has no parts")

    part_names = {p.name for p in assembly.parts}

    # Check all joints reference existing parts
    for i, joint in enumerate(assembly.joints):
        if joint.parent not in part_names:
            raise RuntimeError(
                f"Joint #{i}: parent '{joint.parent}' not in parts list"
            )
        if joint.child not in part_names:
            raise RuntimeError(
                f"Joint #{i}: child '{joint.child}' not in parts list"
            )

    # Check joint.type is valid (LLM sometimes hallucinates 'rotary' etc.)
    _VALID_JOINT_TYPES = {"fixed", "revolute", "prismatic", "continuous"}
    for i, joint in enumerate(assembly.joints):
        if joint.type not in _VALID_JOINT_TYPES:
            raise RuntimeError(
                f"Joint #{i} ('{joint.description}'): invalid type "
                f"'{joint.type}'. Must be one of {sorted(_VALID_JOINT_TYPES)}"
            )

    # Check range_deg well-formed for movable joints
    for i, joint in enumerate(assembly.joints):
        if joint.type in ("revolute", "continuous", "prismatic"):
            if not joint.range_deg or len(joint.range_deg) != 2:
                raise RuntimeError(
                    f"Joint #{i} ('{joint.description}'): range_deg missing "
                    f"or not a 2-tuple"
                )
            lo, hi = joint.range_deg
            if lo >= hi:
                raise RuntimeError(
                    f"Joint #{i} ('{joint.description}'): range_deg "
                    f"({lo}, {hi}) invalid, min must be < max"
                )
            if abs(lo) > 360 or abs(hi) > 360:
                raise RuntimeError(
                    f"Joint #{i} ('{joint.description}'): range_deg "
                    f"({lo}, {hi}) exceeds +/-360 degrees"
                )

    # Check all parts are connected (reachable from root via joints)
    # Auto-fix: connect any orphaned parts to the nearest reachable parent
    if assembly.joints:
        _ensure_connected(assembly, part_names)

    # Check dimensions
    for part in assembly.parts:
        if not part.dimensions:
            logger.warning("Part '%s' has no dimensions", part.name)
        else:
            for key, val in part.dimensions.items():
                if val <= 0:
                    raise RuntimeError(
                        f"Part '{part.name}' dimension '{key}' = {val} "
                        f"(must be > 0)"
                    )

    # Check anchor-dimension compatibility
    _parts_by_name = {p.name: p for p in assembly.parts}
    for joint in assembly.joints:
        for part_name, anchor in [(joint.parent, joint.parent_anchor), (joint.child, joint.child_anchor)]:
            part = _parts_by_name.get(part_name)
            if part and anchor in ("front", "back", "left", "right"):
                dim_keys = ANCHOR_DIM_KEYS.get(anchor, [])
                has_match = any(k in part.dimensions for k in dim_keys)
                if not has_match and not any(
                    k in part.dimensions for k in ("diameter", "outer_diameter")
                ):
                    logger.warning(
                        "Joint '%s': part '%s' uses anchor '%s' but has no matching dimensions %s",
                        joint.description, part_name, anchor, dim_keys,
                    )

    logger.info(
        "Assembly '%s' validated: %d parts, %d joints",
        assembly.name, len(assembly.parts), len(assembly.joints),
    )

    # DOF sanity check (warning only, non-blocking)
    revolute_count = sum(1 for j in assembly.joints if j.type == "revolute")
    if revolute_count == 0:
        logger.warning("Assembly has 0 revolute DOF — all joints fixed")
    elif revolute_count > 8:
        logger.warning(
            "Assembly has %d revolute DOF — verify design intent", revolute_count,
        )

    # P0-4: check joint offsets against the 3.0× max-dimension bound the
    # solver uses in _clamp_child_offset.  Without this, extreme offsets
    # pass validation, pass the VLM loop (because default_angles bend the
    # arm and keep the offset within bounds), then crash the all-zero
    # home-pose solve at export time — killing the entire engineering
    # package output.  Raising here feeds the error back to the LLM via
    # the VLM retry loop so it can regenerate with corrected offsets.
    for i, joint in enumerate(assembly.joints):
        if not joint.offset:
            continue
        parent_part = _parts_by_name.get(joint.parent)
        child_part = _parts_by_name.get(joint.child)
        if not parent_part or not child_part:
            continue
        parent_max = max(parent_part.dimensions.values()) if parent_part.dimensions else 0
        child_max = max(child_part.dimensions.values()) if child_part.dimensions else 0
        if parent_max < 1 and child_max < 1:
            continue
        offset_mag = math.sqrt(sum(c ** 2 for c in joint.offset))
        max_allowed = 3.0 * (parent_max + child_max)
        if max_allowed < 1.0:
            max_allowed = 500.0
        if offset_mag > max_allowed:
            raise RuntimeError(
                f"Joint #{i} ('{joint.description}'): offset "
                f"{joint.offset} (magnitude {offset_mag:.1f}mm) exceeds "
                f"3.0× (parent+child) = {max_allowed:.1f}mm. "
                f"The offset is physically inconsistent with the part "
                f"dimensions; reduce the offset or increase part sizes."
            )


# ---------------------------------------------------------------------------
# Tool class for agent integration
# ---------------------------------------------------------------------------

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
_CATEGORY_EXPECTATIONS = {
    "fixed_arm": (
        "=== ROBOT CATEGORY: FIXED-BASE ARM ===\n"
        "This assembly is a FIXED-BASE ARM (bolted to a workbench). Expected:\n"
        "  - A large base plate at the BOTTOM (the workbench mount).\n"
        "  - Vertical stack of servo joints + links reaching UP and OUT.\n"
        "  - A gripper at the tip.\n"
        "Wheels are NOT expected here — flag any wheel/tire as an error.\n"
        "Do NOT report 'missing wheels' for a fixed-base arm.\n"
    ),
    "wheeled": (
        "=== ROBOT CATEGORY: WHEELED MOBILE BASE (no arm) ===\n"
        "This assembly is a WHEELED MOBILE BASE / CHASSIS. Expected:\n"
        "  - 2 or 4 wheels as HORIZONTAL CYLINDERS resting ON or NEAR the "
        "ground (Z near wheel radius). Wheels roll forward — they must look "
        "like discs/tyres lying on their side, NOT standing upright like "
        "spinning tops.\n"
        "  - A chassis body / deck plate ABOVE the wheels.\n"
        "  - Motors mounted coaxially with each wheel axle.\n"
        "Wheels near the ground are CORRECT — do NOT flag them as errors.\n"
        "Flag wheels only if they FLOAT above the ground or stand vertically.\n"
    ),
    "wheeled_arm": (
        "=== ROBOT CATEGORY: WHEELED MOBILE MANIPULATOR (wheels + arm(s)) ===\n"
        "This assembly is a MOBILE MANIPULATOR: a wheeled chassis with one or "
        "more arms on top. Expected ALL of:\n"
        "  - Wheels as HORIZONTAL CYLINDERS near the ground (rolling discs, "
        "not upright). Wheels at Z ≈ wheel radius are CORRECT.\n"
        "  - A chassis body / deck above the wheels.\n"
        "  - Arm(s) mounted ON TOP of the chassis (base yaw servo -> links "
        "-> gripper), reaching upward/forward.\n"
        "BOTH the wheels AND the arms are EXPECTED here. Do NOT flag the "
        "wheels as 'arms should not have wheels' — this category legitimately "
        "combines them. Do NOT flag the arms as 'floating' if they sit on the "
        "chassis deck.\n"
        "Flag ONLY genuine defects: wheels floating above ground, wheels "
        "vertical, arms intersecting the chassis, parts disconnected.\n"
    ),
    "assembly": (
        "=== ROBOT CATEGORY: GENERAL ASSEMBLY ===\n"
        "No specific category hint. Judge structural integrity generically.\n"
        "If wheels are present they should be near the ground as horizontal "
        "cylinders; if this is an arm with no wheels, do not require wheels.\n"
    ),
}


def _classify_robot(description: str) -> str:
    """Classify the NL description into a robot category for the VLM prompt.

    Mirrors the ``is_arm`` / ``is_wheeled`` decision in
    :func:`generate_assembly_with_vlm_loop` so the prompt and the generator
    agree on what kind of robot is being built.  Returns one of the keys of
    :data:`_CATEGORY_EXPECTATIONS`.
    """
    d = (description or "").lower()
    is_arm = any(kw in d for kw in [
        "臂", "arm", "机械手", "机械臂", "抓手", "gripper", "自由度",
    ])
    is_wheeled = any(kw in d for kw in [
        "轮", "wheel", "差速", "移动", "底盘",
    ])
    if is_arm and is_wheeled:
        return "wheeled_arm"
    if is_wheeled:
        return "wheeled"
    if is_arm:
        return "fixed_arm"
    return "assembly"


def _build_verify_prompt(robot_category: str) -> str:
    """Build the whole-assembly VLM prompt with category context injected.

    Replaces the old static ``_VLM_VERIFY_PROMPT``.  The category block goes
    at the TOP of the prompt so the VLM reads the expectations before judging.
    """
    category_block = _CATEGORY_EXPECTATIONS.get(
        robot_category, _CATEGORY_EXPECTATIONS["assembly"],
    )
    return (
        "You are a STRICT robot assembly quality inspector. Examine the 3D "
        "render.\n\n"
        + category_block
        + "\n"
        "This is a WHOLE-ASSEMBLY view. Judge ONLY structural integrity — do "
        "NOT attempt to evaluate the gripper fingers from this view (they are "
        "too small to resolve here; a dedicated close-up view covers the "
        "gripper).\n\n"
        "=== STRUCTURAL INTEGRITY ===\n"
        "Check for:\n"
        "1. Parts floating in mid-air with no support\n"
        "2. Parts intersecting / overlapping each other\n"
        "3. Arms pointing in impossible directions (e.g. going through the "
        "body)\n"
        "4. Critical parts missing (no base plate, no main body)\n"
        "5. Overall structural coherence\n"
        "6. Parts with WRONG ORIENTATION (e.g. wheels standing vertical "
        "instead of lying horizontal; arms pointing down into the ground)\n\n"
        "IMPORTANT: Only report a problem if it is a GENUINE defect for THIS "
        "robot category. Parts that match the category expectations above are "
        "CORRECT and must NOT be flagged.\n\n"
        "Reply with JSON only:\n"
        '{"passed": true/false, '
        '"problems": ["list of specific structural issues found"], '
        '"description": "brief assessment"}\n'
    )


# Backwards-compat alias.  Some external callers / tests may still reference
# ``_VLM_VERIFY_PROMPT`` as a string; keep it as the generic-assembly variant
# so they get a valid prompt without the category injection.
_VLM_VERIFY_PROMPT = _build_verify_prompt("assembly")

# Dedicated gripper-evaluation prompt — used ONLY for the gripper_closeup
# view.  Whole-assembly views cannot resolve ~46mm finger gaps at the edge
# of a 490mm-tall frame, so they false-negative the gripper as a "solid
# block".  The close-up zooms to a 120mm window around the finger centroid
# so the two prongs and the gap between them are clearly visible.  Only
# this view is authoritative for the gripper question.
_VLM_GRIPPER_CLOSEUP_PROMPT = (
    "You are inspecting the GRIPPER at the tip of a robotic arm. This is a "
    "CLOSE-UP view zoomed in on the gripper — the rest of the arm is out of "
    "frame, which is intentional.\n\n"
    "Judge ONLY the gripper. The gripper passes if you can see TWO clearly "
    "separated, parallel finger prongs that face each other with a VISIBLE "
    "OPEN GAP between them (like a claw, chopsticks, or pliers).\n\n"
    "AUTOMATIC FAIL (passed=false) if ANY of these are true:\n"
    "- The tip is a single solid block, box, cylinder, sphere, or housing\n"
    "- The tip is just another arm link or segment\n"
    "- There are NOT two clearly separated parallel prongs\n"
    "- The end-effector is a single chunky mass with no visible gap/split\n\n"
    "Do NOT rationalize. If you cannot clearly see TWO separate finger prongs "
    "with a gap between them, the gripper FAILS.\n\n"
    "Reply with JSON only:\n"
    '{"passed": true/false, '
    '"problems": ["gripper-specific issues only"], '
    '"description": "brief gripper assessment"}\n'
)


_VLM_FIX_PROMPT = (
    "You previously generated a robot assembly JSON, but the visual "
    "verification found problems.\n\n"
    "Problems found:\n{problems}\n\n"
    "Original description: {description}\n\n"
    "Please regenerate the COMPLETE assembly JSON that fixes these problems. "
    "Apply the relevant fixes below based on each problem type:\n"
    "- Floating / disconnected part: adjust its position offset so it connects "
    "to its parent anchor point, or add/fix the joint referencing it.\n"
    "- Overlapping parts: increase the child part's position offset along the "
    "joint axis so the parts no longer intersect.\n"
    "- Wrong joint type (e.g. revolute where continuous is needed, or fixed "
    "where rotation is needed): change the joint \"type\" field accordingly.\n"
    "- Wrong orientation (e.g. cylinder axis pointing the wrong way): swap the "
    "dimension keys (diameter/height/length/width) or adjust the rotation so "
    "the part aligns with its joint axis.\n"
    "- Missing gripper/claw / end-effector not a gripper: The VERY END of the "
    "arm MUST have TWO clearly separated opposing finger parts named "
    "'gripper_finger_left' and 'gripper_finger_right' (or equivalent). Each "
    "finger MUST be at least 15mm wide and 40mm long so it is clearly visible. "
    "Connect each finger to the gripper_base via a 'prismatic' joint with "
    "'parent_anchor':'center','child_anchor':'center', and set the 'offset' "
    "to [0, +/-gap, z_lift] where gap >= 35mm so the two fingers are clearly "
    "separated with a visible opening between them. Remove any solid block, "
    "cylinder, or extra arm link that is currently at the arm tip.\n"
    "- Unstable / insufficient base: enlarge the base plate dimensions "
    "(length & width) so the assembly center of mass stays over it.\n"
    "- Wheels off the ground: lower the wheel parts' Z position so they "
    "contact the ground plane (Z ≈ wheel radius).\n\n"
    "Return only the JSON, no code blocks, no explanation.\n"
)


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

_GRIPPER_FALSE_ALARM_PATTERNS = (
    "solid block", "solid mass", "solid chunk", "chunky mass", "fused",
    "single curved mass", "single chunky", "single solid",
    "no visible gap", "no gap", "no separated",
    "not two clearly separated", "no two clearly separated",
    "does not have two", "does not terminate in a gripper",
    "no clearly separated parallel prongs", "parallel prongs",
    "no gripper at the tip", "no gripper", "not a gripper",
    "absence of a functional gripper", "no functional gripper",
    "tip of the arm does not have", "tip is a solid",
    "tip of the arm terminates in a solid",
    "tip of the arm does not terminate in a gripper",
    "end effector is a solid", "end-effector is a solid",
    "fails functional gripper", "fails category 2",
    "end-effector is a single", "end effector is a single",
)
# Context words: the text must ALSO mention one of these to be classified
# as a gripper complaint (prevents "base plate does not have two mounting
# holes" from being filtered).  The GLM-4.6V consistently uses "tip",
# "gripper", "finger", "prong", or "end-effector" when complaining about
# the gripper, so the double-condition is reliable.
_GRIPPER_CONTEXT_WORDS = (
    "gripper", "finger", "prong", "effector", "tip", "claw", "爪",
)


def _is_gripper_false_alarm(problem_text: str) -> bool:
    """Return True if a VLM problem is a gripper-finger complaint.

    Uses a double condition — the text must mention a gripper context word
    (gripper/finger/prong/effector/tip/claw) AND match a finger-fusion /
    missing-gripper pattern.  This prevents structural problems like
    "base plate does not have two mounting holes" from being filtered.
    """
    t = problem_text.lower()
    has_context = any(w in t for w in _GRIPPER_CONTEXT_WORDS)
    has_pattern = any(p in t for p in _GRIPPER_FALSE_ALARM_PATTERNS)
    return has_context and has_pattern


# Floating false-alarm patterns (added 2026-06-22, Plan B+C).
#
# VLMs catastrophically misjudge "floating / disconnected" when a part
# occupies <1% of the frame (TDBench, arXiv 2504.03748).  In a long
# robotic arm, the shoulder servo (Ø40mm) linking the base plate to the
# arm chain is exactly such a small part — GLM-4.6V reliably reports
# the arm as "floating with no support" even when the joint graph
# confirms every part is connected to the base.  When the joint-graph
# connectivity check (Check 7 in _geometric_prevalidation) returns
# clean, these reports are viewpoint artifacts and must be filtered so
# they do not trigger a corrupting regeneration round.
#
# Unlike the gripper filter, no context-word double condition is needed:
# "floating" / "mid-air" / "disconnected" are unambiguously geometric
# connectivity complaints, not structural details that could be real.
_FLOATING_FALSE_ALARM_PATTERNS = (
    "floating", "floats", "floated",
    "mid-air", "mid air", "in mid air", "in mid-air",
    "no support", "not supported", "unsupported",
    "disconnected", "not connected", "no visible connection",
    "no visible support", "no physical connection",
    "悬空", "悬浮", "未连接", "无支撑",
)


def _is_floating_false_alarm(problem_text: str) -> bool:
    """Return True if a VLM problem is a floating / disconnected complaint.

    These are filtered ONLY when the joint-graph connectivity check
    (Check 7) confirms every part is reachable from the root — i.e. the
    assembly is genuinely connected and the VLM report is a viewpoint
    artifact.  See ``_vlm_check_assembly`` for the gating logic.
    """
    t = problem_text.lower()
    return any(p in t for p in _FLOATING_FALSE_ALARM_PATTERNS)


# ---------------------------------------------------------------------------
# VLM complaint severity classification (audit: fix-loop has no grading)
# ---------------------------------------------------------------------------
# The fix loop previously treated ALL VLM complaints identically: try a
# targeted fix, then fall back to LLM regeneration. But a gripper-closeup
# FRAMING complaint ("not a close-up view") is fundamentally different from
# a HARD geometry defect ("wheel_fr is 591mm from center"). Conflating them
# meant a deterministic compose output (engineering-correct) got overturned
# by a framing nitpick and regenerated by the LLM into garbage.
#
# Severity grades:
#   HARD  — structural/geometric defect that MUST be fixed (collision,
#           disconnected parts, parts far from center, missing critical
#           parts, wheels misplaced/floating, finger overlap). Triggers
#           targeted fix, and on LLM-sourced assemblies, regeneration.
#   SOFT  — framing / posture / orientation nitpick that does NOT indicate
#           a geometric defect (gripper closeup framing, "arm too
#           flat/horizontal", "not a close-up view"). On a deterministic
#           compose output these are NOT a reason to regenerate — the
#           geometry is correct and the VLM is critiquing the render.
_SOFT_VLM_MARKERS = (
    "not a close-up", "not close-up", "rest of arm is in frame",
    "arm too flat", "arm too horizontal", "arm too vertical",
    "closeup framing", "close-up framing", "view is not",
)

# Markers that unambiguously indicate a HARD geometry defect regardless of
# framing — these MUST be addressed, never treated as soft.
_HARD_VLM_MARKERS = (
    "overlap", "intersect", "collision", "穿模", "penetrat",
    "floating", "disconnected", "not attached", "no support",
    "missing wheel", "missing chassis", "missing arm", "missing critical",
    "misplaced", "far from center", "from center",
    "above z", "below ground", "underground",
    "single chunky mass", "no visible gap", "no separated prongs",
    "vertical instead of horizontal", "wrong orientation",
)


def _classify_vlm_complaint(problem_text: str) -> str:
    """Classify a VLM complaint as 'HARD', 'SOFT', or 'UNKNOWN'.

    HARD = structural defect (must fix). SOFT = framing/posture nitpick.
    UNKNOWN = unclassified; treated conservatively as HARD (safer to fix
    than to ignore an unknown complaint).
    """
    t = problem_text.lower()
    if any(m in t for m in _HARD_VLM_MARKERS):
        return "HARD"
    if any(m in t for m in _SOFT_VLM_MARKERS):
        return "SOFT"
    return "UNKNOWN"


# Wheel false-alarm patterns (added 2026-06-24).
#
# GLM-4.6V reliably mistakes the cylindrical servo housings of a fixed-base
# arm (base_yaw_servo Ø40, pitch_servo Ø36) for "wheels" and reports their
# vertical orientation as "incorrect — wheels should be horizontal to roll
# on the ground". On an arm assembly there are no wheels at all, so the
# entire complaint is a hallucination. When the part list confirms there is
# no wheel/tire part (and the assembly is not a wheeled robot), these
# reports must be filtered instead of triggering a corrupting regeneration
# that "fixes" non-existent wheels by mis-orienting the servos.
#
# IMPORTANT: patterns are deliberately *orientation/rolling specific*
# ("wheels ... oriented", "should roll on ground"), NOT bare "wheel".
# A bare "wheel" would match real collision reports like "base_plate and
# wheel_fr overlap by 65mm", which must be kept even on a part list that
# happens to lack wheels (the part name alone may differ).
_WHEEL_FALSE_ALARM_PATTERNS = (
    "wheel.*orient", "wheels.*orient",
    "tire.*orient", "tires.*orient",
    "rolling on ground", "roll on the ground",
    "should be horizontal", "axis.*perpendicular to ground",
    "轮.*方向", "轮.*朝向", "轮胎.*方向",
)
# Part-name stems that indicate a genuine wheel part. If ANY part matches,
# the assembly really has wheels and wheel-orientation complaints are legit.
_WHEEL_PART_STEMS = ("wheel", "tire", "轮")


def _assembly_has_wheels(parts: list[dict]) -> bool:
    """Return True if any part is a genuine wheel/tire."""
    for p in parts:
        name = (p.get("name", "") or "").lower()
        if any(stem in name for stem in _WHEEL_PART_STEMS):
            return True
    return False


def _is_wheel_false_alarm(
    problem_text: str,
    parts: list[dict],
    positions: dict[str, dict] | None = None,
) -> bool:
    """Return True if a VLM problem is a spurious wheel complaint.

    Two cases:
      (1) The assembly has NO wheel parts but the VLM hallucinated wheels
          onto cylindrical servos ("wheels oriented vertically"). Filter.
      (2) The assembly HAS wheels and they are GROUNDED (solved Z ≈ wheel
          radius, within tolerance), but the VLM misreads the render and
          reports "wheels above Z / floating / wrong orientation". This is
          the recurring wheeled-dual-arm false-negative — the deterministic
          compose path produces correct Z=47.5 wheels (verified), but GLM-4.6V
          still reports "wheels above Z=178". When geometry confirms the
          wheels are grounded, the VLM complaint is overruled.
    """
    import re
    t = problem_text.lower()
    # Case 1: wheel complaint on an assembly with no wheels (hallucination).
    if not _assembly_has_wheels(parts):
        return any(re.search(p, t) for p in _WHEEL_FALSE_ALARM_PATTERNS)

    # Case 2: assembly has wheels. Check whether the complaint is about
    # wheel position/orientation/presence, and whether geometry refutes it.
    is_wheel_position_complaint = any(re.search(p, t) for p in _WHEEL_FALSE_ALARM_PATTERNS) or (
        "wheel" in t and ("above z" in t or "floating" in t
                          or "near ground" in t or "vertical" in t
                          or "horizontal" in t or "missing wheel" in t
                          or "no wheel" in t or "not present" in t
                          or "not visible" in t or "absent" in t)
    )
    # Also catch "No <part> present / visible / found" patterns that name
    # wheels without the word "wheel" adjacent (e.g. "No horizontal cylinders
    # near ground present" — the VLM describing wheels it cannot see).
    if not is_wheel_position_complaint and (
        ("no " in t and ("cylinder" in t or "tire" in t))
        or "not present" in t or "critical part missing" in t
    ):
        is_wheel_position_complaint = "wheel" in t or "cylinder" in t
    if not is_wheel_position_complaint:
        return False

    # Geometric oracle: are the wheels actually grounded?
    if positions is None:
        return False  # can't verify — don't filter
    wheel_positions = {
        name: pose["position"]
        for name, pose in positions.items()
        if "wheel" in name.lower()
    }
    if not wheel_positions:
        return False
    # Wheel radius from part dims (fall back to 45mm, the default).
    wheel_part = next((p for p in parts if "wheel" in (p.get("name","") or "").lower()), None)
    wheel_r = ((wheel_part.get("dimensions", {}).get("diameter", 0) or 0) / 2.0
               if wheel_part else 45.0) or 45.0
    # Grounded = every wheel's Z is within ±40% of wheel_r of wheel_r (i.e.
    # the wheel bottom is near the ground). Tolerance is generous because the
    # solver reports the wheel CENTER; bottom = center - radius.
    grounded = all(
        abs(z - wheel_r) < wheel_r * 0.8
        for (_, _, z) in wheel_positions.values()
    )
    return grounded


def _geometric_prevalidation(
    parts: list[dict],
    positions: dict[str, dict],
    joints: list[dict] | None = None,
) -> list[str]:
    """Deterministic geometric checks. Returns problem descriptions."""
    import math as _math
    problems = []

    # Build adjacency set from joints (parent-child pairs are expected
    # to be close — they are connected and should not trigger overlap warns).
    # Adjacency here mirrors mesh_collision.MeshCollisionChecker._build_adjacent_pairs:
    # direct parent↔child, transitive 2-hop (grandparent↔grandchild, e.g. motor↔wheel
    # through a suspension link), and siblings (same parent, e.g. chassis_body↔motor
    # both mounted on base_plate — a real chassis shell CONTAINS its motors, so their
    # AABB overlap is the intended enclosure, not a collision).
    _adjacent_pairs: set[tuple[str, str]] = set()
    _children_by_parent: dict[str, list[str]] = {}
    if joints:
        for j in joints:
            if isinstance(j, dict):
                p = j.get("parent", "")
                c = j.get("child", "")
            else:
                p = getattr(j, "parent", "")
                c = getattr(j, "child", "")
            if p and c:
                _adjacent_pairs.add((p, c))
                _adjacent_pairs.add((c, p))
                _children_by_parent.setdefault(p, []).append(c)
        # Transitive 2-hop: grandparent↔grandchild.
        for gp, mids in _children_by_parent.items():
            for mid in mids:
                for gc in _children_by_parent.get(mid, []):
                    _adjacent_pairs.add((gp, gc))
                    _adjacent_pairs.add((gc, gp))
        # Siblings: same parent (e.g. chassis_body & motor_* on base_plate).
        for _parent, siblings in _children_by_parent.items():
            for i in range(len(siblings)):
                for k in range(i + 1, len(siblings)):
                    _adjacent_pairs.add((siblings[i], siblings[k]))
                    _adjacent_pairs.add((siblings[k], siblings[i]))

    # 1. Collision proxy: parts at same position.
    # Parts joined by a joint (parent↔child) legitimately share a position —
    # e.g. a suspension_link co-located with its motor (a prismatic joint with
    # zero travel sits at the motor), or two links meeting at a coincident
    # anchor. Skip such pairs; only flag UNRELATED parts that overlap.
    seen: dict[str, str] = {}
    for name, pdata in positions.items():
        pos = pdata.get("position", [0, 0, 0])
        key = f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"
        if key in seen:
            other = seen[key]
            if (name, other) not in _adjacent_pairs:
                problems.append(f"Parts '{name}' and '{other}' at same position")
        else:
            seen[key] = name

    # 2. Outlier: parts >500mm from centroid
    if positions:
        vals = list(positions.values())
        cx = sum(v["position"][0] for v in vals) / len(vals)
        cy = sum(v["position"][1] for v in vals) / len(vals)
        cz = sum(v["position"][2] for v in vals) / len(vals)
        for name, pdata in positions.items():
            p = pdata["position"]
            dist = _math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2 + (p[2]-cz)**2)
            if dist > 500:
                problems.append(f"Part '{name}' is {dist:.0f}mm from center - misplaced")

    # 3. Wheels near ground
    wheel_names = [n for n in positions if "wheel" in n.lower()]
    if wheel_names:
        min_z = min(positions[n]["position"][2] for n in wheel_names)
        if min_z > 100:
            problems.append(f"All wheels above Z={min_z:.0f}mm - should be near ground")

    # 4. Arm-too-flat detection: a robotic arm must have real 3D (Z) extent,
    # not lie as a flat bar along the ground. This catches the systematic
    # failure where prompt rules + sanitizer produced a completely flat arm
    # (e.g. 4dof_arm: Z span 54mm over a 589mm Y span) that the VLM could not
    # recognise as a 3D structure.
    _ARM_PART_KEYWORDS = (
        "link", "joint", "shoulder", "elbow", "wrist", "arm",
        "gripper", "servo", "housing",
    )
    arm_names = [
        n for n in positions
        if any(kw in n.lower() for kw in _ARM_PART_KEYWORDS)
    ]
    if len(arm_names) >= 4:
        xs = [positions[n]["position"][0] for n in arm_names]
        ys = [positions[n]["position"][1] for n in arm_names]
        zs = [positions[n]["position"][2] for n in arm_names]
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        z_span = max(zs) - min(zs)
        horiz_span = max(x_span, y_span)
        if horiz_span > 100:
            if z_span < 30:
                problems.append(
                    f"Arm too flat: Z span {z_span:.0f}mm but horizontal span "
                    f"{horiz_span:.0f}mm — arm lies as a flat bar. Use "
                    f"top/bottom anchors for arm-chain joints and non-zero "
                    f"default_angles so links bend upward into 3D."
                )
            elif z_span < 0.25 * horiz_span:
                problems.append(
                    f"Arm too horizontal: Z span {z_span:.0f}mm is <25% of "
                    f"horizontal span {horiz_span:.0f}mm. Bend the pitch "
                    f"joints (axis='x', front/back anchors) with non-zero "
                    f"default_angles so the arm rises in Z instead of lying flat."
                )
            elif z_span > 2.0 * horiz_span:
                problems.append(
                    f"Arm too vertical: Z span {z_span:.0f}mm is >200% of "
                    f"horizontal span {horiz_span:.0f}mm — the arm looks like "
                    f"a vertical tower instead of a reaching arm. Reduce the "
                    f"pitch default_angles (shoulder/elbow) and use alternating "
                    f"signs (zig-zag) so the elbow bends back toward horizontal."
                )

    # 5. Gripper finger visibility: if the assembly describes an arm with
    # gripper fingers, verify that (a) there are >= 2 finger parts, and
    # (b) the solved finger positions are separated by >= 25mm so they read
    # as distinct opposing prongs rather than a fused block.  This is a
    # deterministic safety net for the VLM, which tends to rationalise a
    # non-visible gripper as "physically plausible".
    finger_names = [n for n in positions if "finger" in n.lower()]
    is_arm = len(arm_names) >= 4
    if is_arm:
        if len(finger_names) < 2:
            problems.append(
                "Arm is missing a functional gripper: fewer than 2 finger "
                "parts found. Add 'gripper_finger_left' and "
                "'gripper_finger_right' parts at the end of the arm."
            )
        else:
            import math as _math2
            for i in range(len(finger_names)):
                for j in range(i + 1, len(finger_names)):
                    p1 = positions[finger_names[i]]["position"]
                    p2 = positions[finger_names[j]]["position"]
                    dist = _math2.sqrt(
                        (p1[0] - p2[0]) ** 2
                        + (p1[1] - p2[1]) ** 2
                        + (p1[2] - p2[2]) ** 2
                    )
                    if dist < 25.0:
                        problems.append(
                            f"Gripper fingers '{finger_names[i]}' and "
                            f"'{finger_names[j]}' are only {dist:.1f}mm apart "
                            f"— they fuse into a single block. Increase the "
                            f"lateral offset so fingers are clearly separated "
                            f"(>= 35mm gap)."
                        )

    # 6. Bounding-box overlap detection for non-adjacent parts.
    #    Parts connected by joints are expected to touch.  Non-adjacent
    #    parts whose rotated world AABBs overlap are likely intersecting
    #    and must be flagged for the VLM fix loop.
    #
    #    P1: previously this check used a crude centre-distance heuristic
    #    (dist < 0.2 * (max_dim_a + max_dim_b)).  That MISSED real
    #    collisions when a long thin part (e.g. a 60mm finger) is rotated
    #    so its long axis sweeps across a sibling part — the centres can
    #    be 32mm apart while the rotated boxes overlap 39mm.  The fix
    #    computes each part's world AABB by rotating its 8 local corners
    #    by the solved axis-angle rotation, then tests axis-aligned
    #    overlap.  This is conservative (AABB ⊇ OBB) but never misses a
    #    real collision, which is the correct direction for a safety net.
    _part_dims = {}
    for p in parts:
        pname = p.get("name", "")
        pdims = p.get("dimensions", {})
        if pname and pdims:
            _part_dims[pname] = pdims

    def _world_aabb(pname):
        """World-space AABB of pname after rotating its local box by the
        solved rotation.  Returns (xmin,ymin,zmin,xmax,ymax,zmax) or None.

        Axis convention matches the renderer's swap_xy (R_z(-90°)): the
        STL's part-local length (FreeCAD X) renders along world Y, and
        width (FreeCAD Y) renders along world X.  So the world AABB maps
        length→world Y extent and width→world X extent.  A cylinder uses
        diameter for X&Y and height for Z.
        """
        pd = positions.get(pname, {})
        center = pd.get("position", [0, 0, 0])
        dims = _part_dims.get(pname)
        if not dims:
            return None
        if "diameter" in dims:
            hx = hy = dims["diameter"] / 2
            hz = dims.get("height", 0) / 2
        else:
            # Match swap_xy: world X = part width, world Y = part length.
            hx = dims.get("width", 0) / 2
            hy = dims.get("length", 0) / 2
            hz = dims.get("height", 0) / 2
        corners = [
            (sx * hx, sy * hy, sz * hz)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        ]
        rot = pd.get("rotation", [0, 0, 1, 0])
        ax, ay, az, ang = rot
        try:
            ang = _math.radians(float(ang))
        except (TypeError, ValueError):
            ang = 0.0
        n = _math.sqrt(ax * ax + ay * ay + az * az)
        if n < 1e-9 or abs(ang) < 1e-9:
            xs = [c[0] + center[0] for c in corners]
            ys = [c[1] + center[1] for c in corners]
            zs = [c[2] + center[2] for c in corners]
            return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
        ax, ay, az = ax / n, ay / n, az / n
        c = _math.cos(ang); s = _math.sin(ang); C = 1 - c
        R = (
            (ax * ax * C + c,       ax * ay * C - az * s, ax * az * C + ay * s),
            (ay * ax * C + az * s,  ay * ay * C + c,      ay * az * C - ax * s),
            (az * ax * C - ay * s,  az * ay * C + ax * s, az * az * C + c),
        )
        wxs = []; wys = []; wzs = []
        for (lx, ly, lz) in corners:
            rx = R[0][0] * lx + R[0][1] * ly + R[0][2] * lz
            ry = R[1][0] * lx + R[1][1] * ly + R[1][2] * lz
            rz = R[2][0] * lx + R[2][1] * ly + R[2][2] * lz
            wxs.append(rx + center[0])
            wys.append(ry + center[1])
            wzs.append(rz + center[2])
        return (min(wxs), min(wys), min(wzs), max(wxs), max(wys), max(wzs))

    _pos_list = list(positions.items())
    _aabb_cache: dict[str, tuple | None] = {}
    # Collect AABB-candidate intersections first; FCL confirms them below.
    # AABB (axis-aligned bbox of the rotated box) is a conservative
    # over-approximation: a 45°-rotated slender bar has an AABB ~41%
    # larger than its actual swept volume, so AABB flags many pairs that
    # do not truly intersect.  Feeding those false positives to the LLM
    # as "physically intersect" feedback caused the VLM loop to chase
    # phantom collisions.  FCL (oriented bounding-box + exact contact)
    # is the ground truth; we use it to filter the AABB candidates.
    _aabb_candidates: list[tuple[str, str, float, float, float]] = []
    for i in range(len(_pos_list)):
        na = _pos_list[i][0]
        box_a = _aabb_cache.get(na)
        if box_a is None:
            box_a = _world_aabb(na)
            _aabb_cache[na] = box_a
        if box_a is None:
            continue
        for j_idx in range(i + 1, len(_pos_list)):
            nb = _pos_list[j_idx][0]
            if (na, nb) in _adjacent_pairs:
                continue
            # Skip container↔internal: a structural shell (chassis_body, etc.)
            # intentionally encloses internal parts, so their AABB overlap is
            # the designed enclosure, not a collision.
            if any(kw in na.lower() for kw in ("chassis_body", "body_shell", "housing")) or \
               any(kw in nb.lower() for kw in ("chassis_body", "body_shell", "housing")):
                continue
            box_b = _aabb_cache.get(nb)
            if box_b is None:
                box_b = _world_aabb(nb)
                _aabb_cache[nb] = box_b
            if box_b is None:
                continue
            ox = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
            oy = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
            oz = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
            if ox > 1.0 and oy > 1.0 and oz > 1.0:
                _aabb_candidates.append((na, nb, ox, oy, oz))

    # FCL confirmation: only report a collision if the oriented bounding
    # boxes truly overlap (penetration > 1mm).  Falls back to reporting
    # all AABB candidates if FCL/trimesh is unavailable, preserving the
    # original conservative behaviour for dependency-free environments.
    if _aabb_candidates:
        confirmed = _fcl_confirm_intersections(
            _aabb_candidates, parts, positions,
        )
        # confirmed is None when FCL is unavailable -> keep all candidates.
        if confirmed is not None:
            report_pairs = confirmed
        else:
            report_pairs = _aabb_candidates
        for na, nb, ox, oy, oz in report_pairs:
            problems.append(
                f"Parts '{na}' and '{nb}' overlap by "
                f"{ox:.0f}x{oy:.0f}x{oz:.0f}mm in their rotated "
                f"world bounding boxes — they physically intersect. "
                f"Increase the offset between them or reduce their "
                f"dimensions so they do not collide."
            )

    # 7. Connectivity — every part must be reachable from an arbitrary
    #    root via the joint graph (BFS).  This is the GROUND-TRUTH
    #    arbiter for VLM "floating / disconnected" reports: if the joint
    #    graph says the assembly is a single connected component, then
    #    no part is genuinely floating, regardless of how the render
    #    angle makes it look (TDBench, arXiv 2504.03748 shows VLMs
    #    catastrophically misjudge "floating" when parts are <1% of the
    #    frame — this check overrides those false negatives).
    #
    #    Without this check, `_vlm_check_assembly` has no geometric
    #    signal to refute VLM "floating" false-alarms, so they enter the
    #    LLM regeneration loop and corrupt an otherwise-correct assembly
    #    (observed: 4dof_arm round 2 correct → VLM false "floating" →
    #    round 3 regeneration broke the gripper).
    if joints:
        part_names = {p.get("name", "") for p in parts if p.get("name")}
        adj: dict[str, set[str]] = {n: set() for n in part_names}
        for j in joints:
            if isinstance(j, dict):
                jp, jc = j.get("parent", ""), j.get("child", "")
            else:
                jp, jc = getattr(j, "parent", ""), getattr(j, "child", "")
            if jp in adj and jc in adj:
                adj[jp].add(jc)
                adj[jc].add(jp)
        if part_names:
            root = next(iter(part_names))
            visited: set[str] = set()
            queue = [root]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                queue.extend(adj.get(node, set()) - visited)
            disconnected = part_names - visited
            if disconnected:
                problems.append(
                    f"Parts {sorted(disconnected)} are not connected to "
                    f"the root '{root}' via the joint graph — they are "
                    f"genuinely floating (no joint path from the base)."
                )

    return problems


def _fcl_confirm_intersections(
    candidates: list[tuple[str, str, float, float, float]],
    parts: list[dict],
    positions: dict[str, dict],
) -> list[tuple[str, str, float, float, float]] | None:
    """Filter AABB candidate pairs through exact FCL collision tests.

    Returns the subset of candidates whose oriented bounding boxes truly
    intersect (penetration > 1mm), or ``None`` if FCL/trimesh is not
    installed (caller falls back to the full AABB candidate list).
    """
    try:
        from .mesh_collision import MeshCollisionChecker
        from ..knowledge.mechanics import Assembly, Joint, Part
    except ImportError:
        return None

    # MeshCollisionChecker needs an Assembly + placements.  Reconstruct
    # lightweight Part objects from the dict list; the checker only reads
    # name + dimensions, so category/material defaults are fine.
    part_objs: list[Part] = []
    name_to_dict: dict[str, dict] = {}
    for p in parts:
        name = p.get("name", "")
        dims = p.get("dimensions", {})
        if not name or not dims:
            continue
        part_objs.append(Part(
            name=name, category="mechanical", description="",
            dimensions=dict(dims),
        ))
        name_to_dict[name] = p
    if len(part_objs) < 2:
        return [c for c in candidates]  # nothing to check

    # Joints are needed only for adjacency filtering, which the caller
    # has already applied via _adjacent_pairs, so pass an empty list.
    asm = Assembly(name="prevalidation", parts=part_objs, joints=[])

    try:
        checker = MeshCollisionChecker()
    except Exception:
        return None

    result = checker.check_assembly_collisions(
        asm, positions, skip_adjacent=False, min_penetration_mm=1.0,
    )
    colliding_names: set[tuple[str, str]] = set()
    for pair in result.pairs:
        if pair.is_collision:
            a, b = pair.part_a, pair.part_b
            colliding_names.add((a, b))
            colliding_names.add((b, a))

    # Keep only candidates FCL confirms; preserve AABB overlap dims for
    # the message severity.
    confirmed = [
        (na, nb, ox, oy, oz)
        for (na, nb, ox, oy, oz) in candidates
        if (na, nb) in colliding_names
    ]
    return confirmed


# ---------------------------------------------------------------------------
# Preview STL generation (trimesh) for VLM verification
# ---------------------------------------------------------------------------
#
# Production STLs are exported by Phase 4 (engineering package), which runs
# AFTER the VLM loop.  During verification the renderer therefore falls back
# to box/cylinder approximations, where gripper fingers become near-invisible
# ~6mm boxes and cylindrical servos become boxes.  That makes prompt checks
# like "gripper should look like a gripper" always fail.
#
# These fast trimesh previews are drop-in replacements that follow the same
# axis convention as FreeCAD STLs (X=length, Y=width, Z=height) so the
# renderer's swap_xy=True path aligns them where the solver expects.


def _build_box_preview_mesh(dims: dict):
    """Box preview mesh with extents [L, W, H] → X=L, Y=W, Z=H."""
    import trimesh

    if "length" in dims and "width" in dims:
        l = dims["length"]
        w = dims["width"]
        h = dims.get("height", dims.get("thickness", 5))
    else:
        l = dims.get("length", dims.get("diameter", 20))
        w = dims.get("width", l)
        h = dims.get("height", dims.get("thickness", 20))
    return trimesh.creation.box(extents=[l, w, h])


def _build_cylinder_preview_mesh(dims: dict):
    """Cylinder preview mesh along Z (matches FreeCAD cylinder convention)."""
    import trimesh

    d = dims.get("outer_diameter", dims.get("diameter", 20))
    h = dims.get("height", dims.get("length", d))
    return trimesh.creation.cylinder(radius=d / 2.0, height=h)


def _build_finger_preview_mesh(name: str, dims: dict):
    """L-shaped gripper finger preview matching _gripper_finger_ops shape.

    Two fused boxes (concatenated): a main bar extending in +X and an
    inward-hooking tip at the front end.  Left/right tip direction is
    detected from the name, mirroring _gripper_finger_ops.
    """
    import trimesh

    L = dims.get("length", 35)
    W = dims.get("width", 6)
    H = dims.get("height", 15)

    n_lower = name.lower()
    is_left = "left" in n_lower
    # Match _gripper_finger_ops: left finger tip hooks toward +Y, right
    # finger tip hooks toward -Y (in FreeCAD coords).
    tip_dir = 1.0 if is_left else -1.0

    # Main bar: makeBox(L, W, H) has its corner at the origin in FreeCAD,
    # so translate the centred trimesh box to match.
    bar = trimesh.creation.box(extents=[L, W, H])
    bar.apply_translation([L / 2.0, W / 2.0, H / 2.0])

    # L-shaped tip at the front end, hooking inward.
    # tip_w mirrors _gripper_finger_ops (part_feature_engine.py): keep the
    # finger slender.  W*2.0 inflated the total Y extent to 3W (e.g.
    # 14→42mm), making the rendered finger a stubby block that the VLM
    # reads as a "solid sphere" instead of a gripper prong.
    tip_l = L * 0.25
    tip_w = max(4.0, W * 0.4)
    tip_y = W if tip_dir > 0 else -tip_w
    tip = trimesh.creation.box(extents=[tip_l, tip_w, H])
    tip.apply_translation([L - tip_l / 2.0, tip_y + tip_w / 2.0, H / 2.0])

    # NOTE: bar and tip touch on a coincident face (Y=W for left, Y=0
    # for right).  ``trimesh.util.concatenate`` does NOT boolean-merge,
    # so this leaves a non-manifold edge (euler=4, two separate bodies).
    # A real boolean union would fix it but requires the optional
    # ``manifold3d``/``blender`` backend, which is not a project
    # dependency.  This preview path is only a FALLBACK when FreeCAD is
    # unavailable — the production path uses FreeCAD-generated STLs
    # (where ``Part.fuse`` produces a clean watertight union), so the
    # non-manifold preview is acceptable.  See C1 (part_feature_engine
    # _gripper_finger_ops) for the real water-tightness fix.
    return trimesh.util.concatenate([bar, tip])


def _generate_preview_stls(parts: list[dict], output_dir: str) -> str:
    """Generate fast trimesh STL previews for VLM rendering.

    Writes one ``{part_name}.stl`` per part into ``{output_dir}/preview_stls``
    and returns that directory path.  Returns an empty string (so the renderer
    falls back to dimension boxes) when trimesh is unavailable.

    The renderer's existing fallback handles any part missing a preview STL
    gracefully (it just builds a dimension box), so partial generation is safe.
    """
    try:
        import trimesh  # noqa: F401
    except ImportError:
        logger.warning(
            "trimesh not installed — VLM will fall back to box approximations"
        )
        return ""

    preview_dir = os.path.join(output_dir, "preview_stls")
    os.makedirs(preview_dir, exist_ok=True)

    for idx, part in enumerate(parts):
        name = part.get("name", f"part_{idx}")
        dims = part.get("dimensions", {}) or {}
        n_lower = name.lower()

        if "finger" in n_lower:
            mesh = _build_finger_preview_mesh(name, dims)
        elif "diameter" in dims or "outer_diameter" in dims:
            mesh = _build_cylinder_preview_mesh(dims)
        else:
            mesh = _build_box_preview_mesh(dims)

        if mesh is None:
            continue

        # Centre on bounding-box centre so the renderer's load_stl centering
        # is a no-op and the part lands exactly where the solver expects.
        try:
            mesh.apply_translation(-mesh.bounding_box.centroid)
        except Exception as _e:
            logger.debug("mesh bounding-box center recentre failed: %s", _e)

        stl_path = os.path.join(preview_dir, f"{name}.stl")
        try:
            mesh.export(stl_path)
        except Exception as e:
            logger.warning("Preview STL export failed for %s: %s", name, e)

    return preview_dir


# Default vision model for the closed-loop assembly verifier.
#
# MUST be GLM-4.6V (the MAXIMUM-tier model), not GLM-4.6V-Flash or
# GLM-4V-Plus.  Empirically verified 2026-06-21: on the 4dof_arm
# gripper close-up render (two 14mm fingers separated by 46mm), only
# GLM-4.6V reliably identifies the two parallel finger prongs and the
# visible gap.  GLM-4.6V-Flash (free tier) and GLM-4V-Plus both
# false-negative the gripper as a "solid block / no separated prongs",
# which causes the VLM loop to fail all 3 rounds on a geometrically
# correct assembly (verification_status=FAILED_MAX_ROUNDS, e2e
# 89.5% blocked at the single critical check).  Using an underpowered
# model as the sole arbiter of fine geometric features defeats the
# purpose of the vision channel — see AGENTS.md §5.2.
_DEFAULT_VERIFIER_VISION_MODEL = "GLM-4.6V"


def _vlm_check_assembly(
    positions: dict[str, dict],
    parts: list[dict],
    render_dir: str,
    api_key: str,
    base_url: str,
    vision_model: str = _DEFAULT_VERIFIER_VISION_MODEL,
    round_num: int = 0,
    real_stl_dir: str | None = None,
    joints: list | None = None,
    robot_category: str = "assembly",
) -> tuple[bool, list[str]]:
    """Render assembly and run VLM verification. Returns (passed, problems).

    ``robot_category`` selects the expectations block injected into the
    whole-assembly prompt (see :func:`_build_verify_prompt`).  This MUST match
    the generator's own classification — passing a category-blind prompt
    historically caused the wheeled-dual-arm false-negative loop.

    Per-view raw VLM responses are accumulated and written to
    ``{render_dir}/vlm_responses.json`` so failures can be debugged across
    rounds without re-running the model.
    """
    from ..models.base import Message
    from ..models.glm import GLMBackend

    from .vtk_renderer import render_assembly_from_positions

    # Use real FreeCAD STLs when available (produced by generate_part_stls
    # before the VLM loop); fall back to fast trimesh preview STLs when
    # FreeCAD is not installed or generation failed.
    stl_dir_for_render = real_stl_dir or _generate_preview_stls(parts, render_dir)

    # Render 4 views
    # Render 4 standard views PLUS a gripper close-up.  The close-up aims
    # the camera at the finger centroid with a tight parallel scale so the
    # ~32mm finger gap is clearly resolvable — without it the VLM sees
    # fingers as sub-pixel slivers at the edge of the full-arm frame and
    # false-negatives the gripper as "single solid mass".
    rendered = render_assembly_from_positions(
        parts=parts,
        positions=positions,
        output_dir=render_dir,
        views=["isometric", "front", "top", "right", "gripper_closeup"],
        stl_dir=stl_dir_for_render,
        width=1600,
        height=1200,
        joints=joints,
        gripper_closeup=True,
    )
    if not rendered:
        return False, ["VTK rendering produced no images"]

    # Check each view with VLM — split by responsibility (2026-06-23).
    # Whole-assembly views (iso/front/top/right) judge STRUCTURAL integrity
    # only — they cannot resolve a ~46mm finger gap at the edge of a
    # 490mm-tall frame, so asking them about the gripper produced false
    # "solid block" negatives.  The gripper_closeup view is the SOLE
    # authority on the gripper question: it zooms to a 120mm window.
    backend = GLMBackend(api_key=api_key, base_url=base_url,
                          vision_model=vision_model)
    all_problems: list[str] = []
    # Track whole-assembly and gripper verdicts separately.
    structural_views: list[str] = []
    structural_pass_count = 0
    gripper_view_passed: bool | None = None  # None = no closeup rendered
    total_views = len(rendered)
    view_logs: list[dict] = []

    for view_path in rendered:
        view_name = os.path.splitext(os.path.basename(view_path))[0]
        is_closeup = view_name == "gripper_closeup"
        prompt = (
            _VLM_GRIPPER_CLOSEUP_PROMPT if is_closeup
            else _build_verify_prompt(robot_category)
        )
        entry: dict = {
            "view": view_name,
            "prompt_role": "gripper" if is_closeup else "structural",
            "raw_response": None,
            "parsed": None,
            "passed": False,
        }
        try:
            resp = backend.vision(
                image_path=view_path,
                prompt=prompt,
            )
            entry["raw_response"] = str(resp)
            text = str(resp).lower()
            view_passed = ('"passed": true' in text) or ('"passed":true' in text)
            entry["passed"] = view_passed
            # Always extract problems (even from "passed" views)
            try:
                start = str(resp).find("{")
                end = str(resp).rfind("}") + 1
                data = json.loads(str(resp)[start:end])
                entry["parsed"] = data
                for p in data.get("problems", []):
                    if p and p not in all_problems:
                        all_problems.append(p)
            except (json.JSONDecodeError, ValueError):
                pass
            # Route the verdict by responsibility.
            if is_closeup:
                gripper_view_passed = view_passed
            else:
                structural_views.append(view_name)
                if view_passed:
                    structural_pass_count += 1
        except Exception as e:
            logger.warning("VLM check failed for %s: %s", view_path, e)
            entry["raw_response"] = f"ERROR: {e}"
        view_logs.append(entry)

    # Pass requires BOTH responsibilities to pass:
    #  - STRUCTURAL: majority of whole-assembly views pass
    #  - GRIPPER: the close-up view passes (sole authority)
    # When no close-up was rendered, fall back to majority vote.
    if gripper_view_passed is not None and structural_views:
        structural_majority = structural_pass_count > len(structural_views) / 2
        passed = structural_majority and bool(gripper_view_passed)
    else:
        passed = (structural_pass_count + (1 if gripper_view_passed else 0)) > total_views / 2

    # Geometric pre-validation as safety net AND ground-truth arbitrator.
    geo_problems = _geometric_prevalidation(parts, positions, joints)

    # WHEEL FALSE-ALARM FILTER (geometric oracle, runs BEFORE the hard_geo
    # gate). The deterministic compose path produces grounded wheels (Z≈
    # radius, verified), but GLM-4.6V still reports "wheels above Z / wrong
    # orientation / missing". When geometry confirms the wheels are grounded,
    # these VLM complaints are overruled REGARDLESS of other geometry issues
    # — otherwise a single arm-motor overlap (hard_geo) would skip the
    # false-alarm filter and the wheeled-dual-arm e2e dead-loops on wheel
    # false-negatives. Filter from both all_problems (VLM) and geo_problems
    # so neither path keeps the refuted complaint.
    all_problems = [
        p for p in all_problems
        if not _is_wheel_false_alarm(p, parts, positions)
    ]
    geo_problems = [
        p for p in geo_problems
        if not _is_wheel_false_alarm(p, parts, positions)
    ]

    # Separate HARD geometry failures (collision, disconnection, absurd
    # positions) from SOFT pose warnings ("arm too flat/horizontal").
    # Soft warnings describe the arm's posture, not a geometric defect —
    # they should NOT block the gripper/floating false-alarm filtering
    # below.  Hard failures force FAIL + skip filtering; soft warnings
    # are appended to all_problems but still allow the filtering branch.
    _SOFT_POSE_MARKERS = ("arm too flat", "arm too horizontal", "arm too vertical")
    hard_geo = [p for p in geo_problems
                if not any(m in p.lower() for m in _SOFT_POSE_MARKERS)]
    soft_geo = [p for p in geo_problems
                if any(m in p.lower() for m in _SOFT_POSE_MARKERS)]

    if hard_geo:
        # Hard geometry problem (collision, disconnection, fused fingers,
        # outlier) → force failure.
        passed = False
        for p in geo_problems:  # include soft warnings too
            if p not in all_problems:
                all_problems.append(p)
    else:
        # No hard geometry failures.  Append soft pose warnings but still
        # run the false-alarm filtering (the arm posture is a suggestion,
        # not a reason to block a geometrically-correct assembly).
        for p in soft_geo:
            if p not in all_problems:
                all_problems.append(p)
        # Geometry is clean — including Check 7 (joint-graph connectivity:
        # every part reachable from the root).  Two classes of VLM false
        # alarm are filtered here, each backed by a geometric oracle:
        #
        #  (a) Gripper "solid block / no separated prongs": Check 5
        #      confirmed >= 2 fingers separated by >= 25mm, so the VLM
        #      (which catastrophically misjudges sub-1%-of-frame features,
        #      per TDBench arXiv 2504.03748) is overruled.
        #
        #  (b) "Floating / disconnected / no support": Check 7 confirmed
        #      the whole assembly is one connected component via the joint
        #      graph, so the VLM report is a viewpoint artifact.
        filtered = [
            p for p in all_problems
            if not _is_gripper_false_alarm(p)
            and not _is_floating_false_alarm(p)
            and not _is_wheel_false_alarm(p, parts, positions)
            and not any(m in p.lower() for m in _SOFT_POSE_MARKERS)
        ]
        if len(filtered) < len(all_problems):
            all_problems = filtered
            if not all_problems:
                # Every remaining problem was a geometrically-refuted
                # false alarm or a soft pose warning → pass.
                passed = True

    # Persist per-view VLM responses for debugging across rounds.
    vlm_log = {
        "round": round_num,
        "views": view_logs,
        "pass_count": structural_pass_count + (1 if gripper_view_passed else 0),
        "total_views": total_views,
        "final_passed": passed,
        "all_problems": all_problems,
    }
    try:
        with open(
            os.path.join(render_dir, "vlm_responses.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(vlm_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning("Failed to write vlm_responses.json: %s", e)

    return passed, all_problems


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
