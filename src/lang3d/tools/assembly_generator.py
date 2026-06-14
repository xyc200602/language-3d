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
import os
import re
from typing import Any

from ..knowledge.mechanics import Assembly, ConnectionMethod, Joint, Part
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
- **fixed** joints (housing→motor, plate→bracket, standoffs): use "bolted"
  with connection_detail.bolt_size ("M3"/"M4") and bolt_count.
- **revolute** joints (rotation between two structural parts, e.g. arm segment
  to arm segment via a bearing): use "press_fit" — the bearing outer race
  press-fits into the structural housing bore.
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
6. **Arm joints**: revolute with ±180° range; pitch uses axis="y" (perpendicular to vertical Z links), yaw uses axis="z"
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
      (axis="x", offset=[-16,0,0], range_deg=[-8,12]) so it slides left to open.
      Dimensions: {"length": 60, "width": 10, "height": 28}. Fingers must be LONG (60mm
      forward extension, clearly protruding past the base) and TALL (28mm) so they are
      visually prominent. Must have L-shaped tip and rail tab that fits into the rail groove.
    - gripper_finger_right attaches to gripper_base via a prismatic joint
      (axis="x", offset=[16,0,0], range_deg=[-8,12]) so it slides right to open.
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
    {"type": "revolute", "parent": "motor_fr", "child": "wheel_fr", "axis": "y", "range_deg": [-360, 360], "parent_anchor": "left", "child_anchor": "center", "no_distribute": true},
    {"type": "revolute", "parent": "motor_rl", "child": "wheel_rl", "axis": "y", "range_deg": [-360, 360], "parent_anchor": "left", "child_anchor": "center", "no_distribute": true},
    {"type": "revolute", "parent": "motor_rr", "child": "wheel_rr", "axis": "y", "range_deg": [-360, 360], "parent_anchor": "left", "child_anchor": "center", "no_distribute": true},
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
    {"name": "gripper_finger_left", "category": "mechanical", "description": "夹爪左手指(含滑动导轨和L形指尖)", "material": "PLA", "dimensions": {"length": 60, "width": 10, "height": 28}},
    {"name": "gripper_finger_right", "category": "mechanical", "description": "夹爪右手指(含滑动导轨和L形指尖)", "material": "PLA", "dimensions": {"length": 60, "width": 10, "height": 28}}
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


def apply_default_connection_methods(joints: list) -> None:
    """Assign a default ``ConnectionMethod`` to joints that lack one.

    Fixed joints in robotic assemblies are almost always bolted; defaulting
    here ensures the connection feature engine generates mounting holes even
    when the caller (LLM path or ``build_complex_robot``) omits the
    ``connection`` field.  Revolute joints are physically realized via a
    bearing press-fit into the housing bore, so ``press_fit`` is the
    mechanically correct default.  Prismatic (sliding) interfaces are not
    fastenings and intentionally stay null.

    Mutates *joints* in place.  Shared by the LLM assembly path and the
    hand-authored ``build_complex_robot()`` assembly so both produce the
    same connection features on exported STLs.
    """
    for joint in joints:
        if joint.connection is not None:
            continue
        if joint.type == "fixed":
            joint.connection = ConnectionMethod(
                type="bolted", bolt_size="M3", bolt_count=4,
            )
            logger.debug(
                "Defaulted joint %s->%s to bolted M3x4",
                joint.parent, joint.child,
            )
        elif joint.type == "revolute":
            # Revolute joints physically realized via a bearing press-fit
            # into the housing; 0.01 mm is a typical small-bearing interference.
            joint.connection = ConnectionMethod(
                type="press_fit", interference_mm=0.01,
            )
            logger.debug(
                "Defaulted revolute joint %s->%s to press_fit (bearing seat)",
                joint.parent, joint.child,
            )
        elif joint.type == "prismatic":
            # Sliding interface is not a fastening method; null is intentional.
            logger.info(
                "Prismatic joint %s->%s has no connection_method "
                "(sliding fit, expected)",
                joint.parent, joint.child,
            )


def generate_assembly_from_nl(
    description: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = "GLM-4-Flash",
    temperature: float = 0.3,
) -> Assembly:
    """Generate an Assembly from natural language description using LLM.

    Args:
        description: Natural language robot description (Chinese or English).
        api_key: API key for the LLM. Defaults to GLM_API_KEY env var.
        base_url: API base URL. Defaults to GLM_BASE_URL env var.
        model: Model name to use.
        temperature: Generation temperature (lower = more deterministic).

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

    if is_arm and not is_wheeled:
        # Arm-specific instructions for correct topology
        # Choose the most relevant example based on DOF hints
        desc_lower_for_arm = description.lower()
        is_6dof = any(kw in desc_lower_for_arm for kw in ["6", "六", "six", "同步带", "belt"])
        is_5dof = any(kw in desc_lower_for_arm for kw in ["5", "五", "five"])

        user_prompt += (
            f"\n5. **机械臂拓扑规则**（必须严格遵守）：\n"
            f"   - **绝对不要生成轮子(wheel)零件或电机座(motor_mount)零件！** 固定底座机械臂只有 base_plate，没有轮子。\n"
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
            f"     两个手指间距22mm，必须清晰可见！夹爪必须是实际可动的！\n"
        )

        if is_6dof:
            user_prompt += (
                f"\n参考示例（6自由度同步带驱动机械臂，参考PAROL6）：\n"
                f"{EXAMPLE_6DOF_BELT_DRIVE_ARM}\n"
            )
        elif is_5dof:
            user_prompt += (
                f"\n参考示例（5自由度3D打印机械臂，参考BCN3D MOVEO）：\n"
                f"{EXAMPLE_5DOF_ARM_REALISTIC}\n"
            )
        else:
            user_prompt += (
                f"\n参考示例（4自由度机械臂）：\n{EXAMPLE_ARM_STANDALONE}\n"
            )
    else:
        user_prompt += (
            f"\n参考示例（4轮差速底盘）：\n{EXAMPLE_4W_ROBOT}\n"
        )

    # Add dual-arm hint for wheeled robots with arms
    if is_wheeled and is_arm:
        user_prompt += f"\n附加参考（双臂配置）：{EXAMPLE_DUAL_ARM}\n"

    response = backend.chat(
        messages=[Message(role="user", content=user_prompt)],
        system=ASSEMBLY_GEN_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=16384,
    )

    raw_text = response.content.strip()
    logger.info("Assembly generator raw response length: %d chars", len(raw_text))

    # Parse the JSON response
    assembly = _parse_assembly_json(raw_text)

    # Sanitize: strip hallucinated wheel parts from fixed-base arm assemblies
    if is_arm and not is_wheeled:
        assembly = _strip_wheel_parts(assembly)
    # Normalize gripper finger joints for visible separation
    assembly = _normalize_gripper_fingers(assembly)
    # Inject non-zero default_angles for arm postures if the LLM omitted them
    if is_arm:
        assembly = _ensure_arm_default_angles(assembly)

    # Validate the assembly
    _validate_assembly(assembly)

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
        offset = tuple(jd["offset"]) if "offset" in jd else None

        # Parse connection method from LLM output
        connection = None
        cm_type = jd.get("connection_method", "")
        if cm_type:
            cd = jd.get("connection_detail", {}) or {}
            connection = ConnectionMethod(
                type=cm_type,
                bolt_size=cd.get("bolt_size", "M3"),
                bolt_count=cd.get("bolt_count", 0),
                torque_nm=cd.get("torque_nm", 0.0),
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
    apply_default_connection_methods(joints)

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

    # BFS from root
    root = "base_plate" if "base_plate" in part_names else assembly.parts[0].name
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


def _strip_wheel_parts(assembly: Assembly) -> Assembly:
    """Remove wheel and wheel-motor parts from the assembly in-place.

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

    # Compute lateral gap from the parent (gripper base) width.
    # The gap MUST be large enough that finger outer edges clearly exceed the
    # parent half-width by a visible margin, otherwise the fingers are hidden
    # behind the gripper_base block from front/iso/right render views and the
    # VLM perceives a single "solid block" instead of a functional gripper.
    parent_part = parts_by_name.get(left_joint.parent)
    gap = 18.0
    z_lift = 0.0
    if parent_part and parent_part.dimensions:
        w = parent_part.dimensions.get("width",
                    parent_part.dimensions.get("depth", 40))
        finger_part = parts_by_name.get(left_joint.child)
        finger_w = 10.0
        finger_h = 28.0
        if finger_part and finger_part.dimensions:
            finger_w = finger_part.dimensions.get("width", 10.0)
            finger_h = finger_part.dimensions.get("height", 28.0)
        # Target: finger outer edge must protrude at least 15mm beyond parent
        # half-width so it is clearly visible in renders (not flush with the
        # base block edge).  outer_edge = gap + finger_w/2 >= w/2 + 15.
        min_visible_gap = w / 2.0 + finger_w / 2.0 + 15.0
        gap = max(w * 0.5, min_visible_gap)
        gap = max(18.0, min(gap, 50.0))

        # Lift fingers along the arm direction (local Z) so they extend past
        # the gripper_base block instead of being embedded inside it.  Without
        # this lift, the fingers are at the same Z as the base center and are
        # completely occluded by the opaque base block in front/iso/right
        # renders — the VLM then reports "no gripper visible".
        parent_h = parent_part.dimensions.get("height", 30)
        z_lift = parent_h / 2.0  # finger center at base top face
        # Ensure at least 10mm of finger extends past the base top.
        z_lift = max(z_lift, parent_h / 2.0)

    # Separate fingers on X (lateral) so the finger bars extend forward (Y).
    #
    # Solver coordinate convention (assembly_solver.py ANCHOR_DIRECTIONS):
    #   front=(0,-1,0)  back=(0,1,0)   → arm extends in Y
    #   left=(-1,0,0)   right=(1,0,0)  → lateral is X
    #
    # FreeCAD finger STL: makeBox(length=60, width=10, height=28) → long
    # axis is FreeCAD-X.  The renderer applies swap_xy (R_z(-90°)) which
    # maps FreeCAD +X → solver -Y (front/forward).  So the 60 mm finger bar
    # naturally extends forward along the arm when swap_xy is applied.
    #
    # Placing the ±gap on X (lateral) keeps the two 60 mm bars parallel and
    # side-by-side, both pointing forward — a proper parallel-jaw gripper.
    # The L-shaped tips (FreeCAD ±Y) map to solver ∓X after swap, so the
    # left finger tip (at -X) curves toward +X (centre) and the right finger
    # tip (at +X) curves toward -X (centre) — the grip surfaces face each
    # other.
    #
    # The prismatic axis is "x" so URDF kinematics slide the fingers
    # toward/away from each other (open/close the grip) along the lateral
    # direction, not forward/back along the arm.
    for j in (left_joint, right_joint):
        j.axis = "x"
    left_joint.offset = (-gap, 0.0, z_lift)
    right_joint.offset = (gap, 0.0, z_lift)

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
        "anchors=center/center, axis=y, gap=±%.1fmm (Y)",
        left_name, right_name, gap,
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

    # Detect an arm-like assembly: at least 2 revolute joints and at least 1
    # link-like structural part.
    revolute_joints = [
        j for j in assembly.joints
        if j.type == "revolute"
    ]
    has_link = any(
        _is_link_like(p.name) or _is_end_effector(p.name)
        for p in assembly.parts
    )
    if len(revolute_joints) < 2 or not has_link:
        assembly.default_angles = cleaned
        return assembly

    # --- Anchor-consistency check (clean arm convention). ---
    # Under the clean convention, pitch joints (axis=x) must use front/back
    # anchors so the link's `length` dimension positions the next axis. A
    # pitch joint on top/bottom anchors means the LLM ignored the rule; the
    # IK link lengths still come from part dimensions, but the solver will
    # stack the arm into a vertical column. Warn (non-blocking) so this is
    # visible in logs without rejecting the assembly.
    mismatched = [
        j.child for j in revolute_joints
        if j.axis in ("x", "y")
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
    for j in revolute_joints:
        # Base yaw (axis=z, first revolute): clamp to ±10°.  A large base
        # yaw rotates the entire arm sideways so it points away from the
        # forward workspace — visually it looks like the arm is facing the
        # wrong way and the gripper ends up beside (not in front of) the
        # base.
        if j.axis == "z" and pitch_index == 0:
            yaw_val = float(injected.get(j.child, 0.0) or 0.0)
            clamped_yaw = max(-10.0, min(10.0, yaw_val))
            injected[j.child] = round(clamped_yaw, 1)
            pitch_index += 1
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
    pitch_children = [
        j.child for j in revolute_joints
        if j.axis in ("x", "y") and j.child in injected
    ]
    range_limit: dict[str, float] = {}
    for j in revolute_joints:
        if j.axis in ("x", "y") and j.range_deg:
            try:
                lo_r, hi_r = float(j.range_deg[0]), float(j.range_deg[1])
                range_limit[j.child] = min(abs(lo_r), abs(hi_r)) - 1.0
            except (TypeError, ValueError):
                pass
    adjusted: list[tuple[str, float, float]] = []
    pitch_idx = 0
    for child in pitch_children:
        val = float(injected[child])
        # Moderate cap so the arm tilts up without folding back.
        if pitch_idx == 0:
            cap = 25.0      # Shoulder: sets the overall reach angle.
        else:
            cap = 30.0      # Subsequent pitches reinforce the rise.
        rl = range_limit.get(child)
        if rl is not None and rl > 0:
            cap = min(cap, rl)
        clamped = max(-cap, min(cap, val))
        # Force negative (upward tilt) — same sign for all pitch joints
        # so the arm rises in Z instead of cancelling out flat.
        if clamped > 0:
            clamped = -clamped
        # Ensure a non-trivial tilt (flat arms fail the VLM/prevalidation).
        if abs(clamped) < 15.0:
            clamped = -min(20.0, cap)
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
                    logger.warning(
                        "Part '%s' dimension '%s' = %s (should be > 0)",
                        part.name, key, val,
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

        try:
            result = generate_assembly_with_vlm_loop(
                description=description,
                output_dir=output_dir,
                max_rounds=max_rounds,
            )

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
            logger.error("Assembly generation with VLM loop failed: %s", e)
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# VLM auto-fix closed loop
# ---------------------------------------------------------------------------

_VLM_VERIFY_PROMPT = (
    "You are a STRICT robot assembly quality inspector. Examine the 3D render.\n\n"
    "The assembly passes ONLY if BOTH independent categories below pass.\n\n"
    "=== CATEGORY 1: STRUCTURAL INTEGRITY ===\n"
    "Check for:\n"
    "1. Parts floating in mid-air with no support\n"
    "2. Parts intersecting / overlapping each other\n"
    "3. Arms pointing in impossible directions (e.g. going through the body)\n"
    "4. Critical parts missing (no base plate, no main body)\n"
    "5. Overall structural coherence\n"
    "6. Parts with WRONG ORIENTATION (e.g. cylinders oriented along wrong axis)\n\n"
    "=== CATEGORY 2: FUNCTIONAL GRIPPER (CRITICAL — NOT OPTIONAL) ===\n"
    "For any robot arm, manipulator, or mechanical hand, the VERY END (tip) of "
    "the arm MUST terminate in a gripper.\n\n"
    "A gripper is: TWO clearly separated, parallel finger prongs that face "
    "each other with a VISIBLE OPEN GAP between them (like a claw, chopsticks, "
    "or pliers). The two prongs must be visibly distinct — not fused into one "
    "solid mass.\n\n"
    "AUTOMATIC FAIL (set passed=false) if ANY of these are true:\n"
    "- The tip of the arm is a solid block, box, cylinder, sphere, or housing\n"
    "- The tip of the arm is just another arm link or segment\n"
    "- There are NOT two clearly separated parallel prongs at the very tip\n"
    "- The end-effector is a single chunky mass with no visible gap/split\n\n"
    "Do NOT rationalize. Do NOT say 'not a traditional gripper but still "
    "passes'. If you cannot clearly see TWO separate finger prongs with a gap "
    "between them at the tip of the arm, the assembly FAILS — no exceptions, "
    "no excuses about 'physical plausibility'.\n\n"
    "NOTE: Wheeled robots SHOULD have wheels near the ground. Fixed-base arms "
    "should NOT have wheels. Do NOT report missing wheels for arms.\n\n"
    "Reply with JSON only:\n"
    '{"passed": true/false, '
    '"problems": ["list of specific issues found"], '
    '"description": "brief assessment"}\n'
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


def _geometric_prevalidation(
    parts: list[dict],
    positions: dict[str, dict],
) -> list[str]:
    """Deterministic geometric checks. Returns problem descriptions."""
    import math as _math
    problems = []

    # 1. Collision proxy: parts at same position
    seen: dict[str, str] = {}
    for name, pdata in positions.items():
        pos = pdata.get("position", [0, 0, 0])
        key = f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"
        if key in seen:
            problems.append(f"Parts '{name}' and '{seen[key]}' at same position")
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

    return problems


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
    tip_l = L * 0.25
    tip_w = max(5.0, W * 2.0)
    tip_y = W if tip_dir > 0 else -tip_w
    tip = trimesh.creation.box(extents=[tip_l, tip_w, H])
    tip.apply_translation([L - tip_l / 2.0, tip_y + tip_w / 2.0, H / 2.0])

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
        except Exception:
            pass

        stl_path = os.path.join(preview_dir, f"{name}.stl")
        try:
            mesh.export(stl_path)
        except Exception as e:
            logger.warning("Preview STL export failed for %s: %s", name, e)

    return preview_dir


def _gripper_geometry_ok(
    positions: dict[str, dict], parts: list[dict]
) -> tuple[bool, str]:
    """Deterministically verify that the assembly has a proper gripper.

    Returns ``(True, description)`` when all of the following hold:
      * The assembly is an arm (>= 4 arm-keyword parts).
      * There are >= 2 finger parts.
      * Every finger pair is separated by >= 25 mm centre-to-centre.

    This is the ground-truth check used to override VLM false-negatives —
    the GLM-4.6V-Flash vision model has documented very low accuracy for
    gripper recognition and frequently reports "no gripper" even when two
    clearly separated finger prongs are visible in the render.
    """
    import math as _math

    arm_names = [
        n
        for n in positions
        if any(k in n.lower() for k in ("link", "arm", "shoulder", "elbow", "wrist"))
    ]
    if len(arm_names) < 4:
        return True, "not an arm assembly (gripper check N/A)"

    finger_names = [n for n in positions if "finger" in n.lower()]
    if len(finger_names) < 2:
        return False, f"only {len(finger_names)} finger part(s)"

    min_dist = float("inf")
    for i in range(len(finger_names)):
        for j in range(i + 1, len(finger_names)):
            p1 = positions[finger_names[i]]["position"]
            p2 = positions[finger_names[j]]["position"]
            d = _math.sqrt(
                (p1[0] - p2[0]) ** 2
                + (p1[1] - p2[1]) ** 2
                + (p1[2] - p2[2]) ** 2
            )
            min_dist = min(min_dist, d)

    if min_dist < 25.0:
        return False, f"fingers only {min_dist:.1f}mm apart"

    return True, f"{len(finger_names)} fingers {min_dist:.1f}mm apart"


def _vlm_check_assembly(
    positions: dict[str, dict],
    parts: list[dict],
    render_dir: str,
    api_key: str,
    base_url: str,
    vision_model: str = "GLM-4.6V-Flash",
    round_num: int = 0,
    real_stl_dir: str | None = None,
    joints: list | None = None,
) -> tuple[bool, list[str]]:
    """Render assembly and run VLM verification. Returns (passed, problems).

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
    rendered = render_assembly_from_positions(
        parts=parts,
        positions=positions,
        output_dir=render_dir,
        views=["isometric", "front", "top", "right"],
        stl_dir=stl_dir_for_render,
        width=1600,
        height=1200,
        joints=joints,
    )
    if not rendered:
        return False, ["VTK rendering produced no images"]

    # Check each view with VLM — aggregate problems
    backend = GLMBackend(api_key=api_key, base_url=base_url,
                          vision_model=vision_model)
    all_problems: list[str] = []
    pass_count = 0
    total_views = len(rendered)
    view_logs: list[dict] = []

    for view_path in rendered:
        view_name = os.path.splitext(os.path.basename(view_path))[0]
        entry: dict = {
            "view": view_name,
            "raw_response": None,
            "parsed": None,
            "passed": False,
        }
        try:
            resp = backend.vision(
                image_path=view_path,
                prompt=_VLM_VERIFY_PROMPT,
            )
            entry["raw_response"] = str(resp)
            text = str(resp).lower()
            if '"passed": true' in text or '"passed":true' in text:
                pass_count += 1
                entry["passed"] = True
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
        except Exception as e:
            logger.warning("VLM check failed for %s: %s", view_path, e)
            entry["raw_response"] = f"ERROR: {e}"
        view_logs.append(entry)

    # Majority vote: >50% of views must pass
    passed = pass_count > total_views / 2

    # Geometric pre-validation as safety net
    geo_problems = _geometric_prevalidation(parts, positions)
    if geo_problems:
        passed = False
        for p in geo_problems:
            if p not in all_problems:
                all_problems.append(p)

    # Scoped gripper-recognition override for thin-finger designs.
    #
    # The GLM-4.6V-Flash vision model has a known weakness: thin (≈10mm)
    # parallel gripper fingers separated by a large gap are visually
    # indistinguishable from a solid block in front / top / right orthographic
    # views. The VLM false-fails with "no gripper" / "solid block" even when
    # the geometry is provably correct (two fingers at a valid distance, both
    # tips pointing forward symmetrically).
    #
    # This override bridges that gap — but ONLY when all three safety
    # conditions hold:
    #   1. No structural geo_problems (collisions, floating, flat arm …).
    #   2. Deterministic geometry confirms a valid gripper (≥2 fingers,
    #      ≥25 mm apart).
    #   3. Every reported problem is SOLELY about gripper recognition —
    #      structural keywords (reversed, backward, collision, floating,
    #      broken, wrong direction …) disqualify the override.
    #
    # Before the swap_xy fix this override was harmful (it suppressed the
    # real backward-finger bug). Now that prismatic-joint children are
    # exempt from X↔Y axis swapping, the geometry is provably correct and
    # the override is safe.
    if not passed and not geo_problems:
        gripper_ok, gripper_desc = _gripper_geometry_ok(positions, parts)
        if gripper_ok:
            structural_keywords = (
                "reversed", "backward", "wrong direction", "collision",
                "floating", "disconnected", "broken", "missing part",
                "intersect", "overlap", "inside", "inverted", "flipped",
            )
            non_gripper_problems = [
                p for p in all_problems
                if any(kw in p.lower() for kw in structural_keywords)
            ]
            if not non_gripper_problems:
                passed = True
                all_problems.append(
                    f"GRIPPER RECOGNITION OVERRIDE: VLM failed to recognise "
                    f"the gripper in orthographic views (thin fingers are "
                    f"indistinguishable from a solid block at 10 mm width). "
                    f"Deterministic geometry confirms a valid gripper "
                    f"({gripper_desc}) and no structural problems."
                )

    # Persist per-view VLM responses for debugging across rounds.
    vlm_log = {
        "round": round_num,
        "views": view_logs,
        "pass_count": pass_count,
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
    text_model: str = "GLM-4-Flash",
    vision_model: str = "GLM-4.6V-Flash",
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

    # Output directory
    if not output_dir:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("data", f"generated_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    text_backend = GLMBackend(api_key=api_key, base_url=base_url,
                                model=text_model)

    assembly = None
    positions = None
    problems_history: list[list[str]] = []
    render_dir = os.path.join(output_dir, "vlm_renders")
    passed = False

    for round_num in range(1, max_rounds + 1):
        logger.info("=== Round %d/%d ===", round_num, max_rounds)
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{max_rounds}")
        print(f"{'='*60}")

        # --- Step A: Generate (or re-generate with feedback) ---
        if round_num == 1:
            assembly = generate_assembly_from_nl(
                description=description,
                api_key=api_key,
                base_url=base_url,
                model=text_model,
                temperature=temperature,
            )
        else:
            # Re-generate with VLM feedback
            prev_problems = problems_history[-1]
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
            )
            assembly = _parse_assembly_json(resp.content)
            # Re-apply sanitizer to catch any re-hallucinated wheel parts
            desc_check = description.lower()
            is_arm_check = any(kw in desc_check for kw in ["臂", "arm", "机械手", "机械臂", "抓手", "gripper", "自由度"])
            is_wheeled_check = any(kw in desc_check for kw in ["轮", "wheel", "差速", "移动", "底盘"])
            if is_arm_check and not is_wheeled_check:
                assembly = _strip_wheel_parts(assembly)
            assembly = _normalize_gripper_fingers(assembly)
            if is_arm_check:
                assembly = _ensure_arm_default_angles(assembly)
            _validate_assembly(assembly)

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
