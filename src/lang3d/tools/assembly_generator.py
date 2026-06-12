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
6. **Arm joints**: revolute with ±180° range; pitch uses axis="x", yaw uses axis="z"
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
12. **ARM ANCHORS & AXIS**: Arm links extend along Y via parent_anchor="front" / child_anchor="back".
    Pitch joints (up/down bending) MUST use axis="x" — the X axis is perpendicular to the Y link
    direction, so rotation creates bending. NEVER use axis="y" for front→back pitch joints, because
    Y is parallel to the link direction and rotation produces no displacement.
    Base rotation (yaw) uses axis="z" with top→bottom anchors.
13. **ARM DEFAULT ANGLES**: Provide non-zero default_angles (e.g., -45, -30, 15) so the arm
    has a bent posture instead of a straight vertical tower
14. **WHEEL ORIENTATION**: Wheels MUST use child_anchor='center', axis='y',
    parent_anchor='left' or 'right'. This ensures correct cylinder orientation.
15. **WHEEL-MOTOR CHAIN**: Always use base_plate→motor→wheel topology.
    Never attach wheels directly to base_plate.
16. **NO WHEELS IN ARMS**: Robotic arms (N-DOF arm, robotic arm, 机械臂) must NOT include
    wheel parts. An arm has: base_plate, joints (support/housing), links (连杆), and end_effector.
    NEVER generate "wheel" parts for arm assemblies.
17. **ARM LINK DIMENSIONS**: Link parts must have length 60-200mm, width 20-50mm, height 12-30mm.
    End effectors must be 20-60mm in each dimension. Never create parts with dimensions > 300mm.
18. **EVERY joint after the first 2 in an arm chain MUST use parent_anchor="front" / child_anchor="back".
    The first 2 joints (base→housing, housing→first_link) use top→bottom. All subsequent joints
    use front→back so the arm extends horizontally.
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
  "default_angles": {"shoulder_link": -45, "elbow_link": -30, "wrist_link": 15},
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
    {"type": "fixed", "parent": "base_plate", "child": "shoulder_joint", "parent_anchor": "top", "child_anchor": "bottom"},
    {"type": "revolute", "parent": "shoulder_joint", "child": "shoulder_link", "axis": "x", "range_deg": [-180, 180], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "revolute", "parent": "shoulder_link", "child": "elbow_joint", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "revolute", "parent": "elbow_joint", "child": "elbow_link", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "revolute", "parent": "elbow_link", "child": "wrist_joint", "axis": "x", "range_deg": [-180, 180], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "wrist_joint", "child": "wrist_link", "parent_anchor": "front", "child_anchor": "back"},
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
    {"type": "revolute", "parent": "shoulder_joint_housing", "child": "shoulder_link", "axis": "x", "range_deg": [-120, 120], "parent_anchor": "front", "child_anchor": "back", "offset": [0, 0, -20]},
    {"type": "fixed", "parent": "shoulder_link", "child": "elbow_joint_housing", "parent_anchor": "front", "child_anchor": "back", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "elbow_joint_housing", "child": "elbow_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "revolute", "parent": "elbow_joint_housing", "child": "elbow_link", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "fixed", "parent": "elbow_link", "child": "wrist_joint_housing", "parent_anchor": "front", "child_anchor": "back", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "fixed", "parent": "wrist_joint_housing", "child": "wrist_motor", "parent_anchor": "back", "child_anchor": "front", "connection_method": "bolted", "connection_detail": {"bolt_size": "M3", "bolt_count": 4}},
    {"type": "revolute", "parent": "wrist_joint_housing", "child": "wrist_link", "axis": "x", "range_deg": [-150, 150], "parent_anchor": "front", "child_anchor": "back"},
    {"type": "revolute", "parent": "wrist_link", "child": "wrist_rotate_motor", "axis": "z", "range_deg": [-180, 180], "parent_anchor": "front", "child_anchor": "back"},
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
    {"type": "revolute", "parent": "wrist_roll_housing", "child": "end_effector_mount", "axis": "z", "range_deg": [-180, 180], "parent_anchor": "front", "child_anchor": "back"}
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
            f"   - 连杆(link)的 child_anchor 和下一个关节的 parent_anchor 用 'front'/'back'\n"
            f"     （这样连杆沿长度方向水平延伸，而不是垂直堆叠）\n"
            f"   - 肩部旋转关节之后的第一段连接可以用 'top'/'bottom'（向上）\n"
            f"   - 提供 default_angles 让臂有弯曲姿态（不要全是0度）\n"
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

    # Default connection_method for fixed joints that lack one.
    # Fixed joints in robotic assemblies are almost always bolted; defaulting
    # here ensures the connection feature engine generates mounting holes
    # even when the LLM omits the connection_method field.
    for joint in joints:
        if joint.type == "fixed" and joint.connection is None:
            joint.connection = ConnectionMethod(
                type="bolted", bolt_size="M3", bolt_count=4,
            )
            logger.debug(
                "Defaulted joint %s->%s to bolted M3x4",
                joint.parent, joint.child,
            )

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


def _fix_arm_chain_anchors(joints: list[Joint], parts: list[Part]) -> None:
    """Fix anchor assignments for arm-chain joints.

    Heuristic: In a proper arm chain, joints after the first 2 (base→housing,
    housing→first_link which use top→bottom) should use front→back for
    horizontal extension.  If the LLM used default top→bottom for link joints,
    correct them.

    Also ensures end-effectors use front→back anchors.
    """
    if len(joints) < 3:
        return

    parts_by_name = {p.name: p for p in parts}

    for i, joint in enumerate(joints):
        # Skip joints that already have non-default anchors
        if joint.parent_anchor != "top" or joint.child_anchor != "bottom":
            continue

        parent_name = joint.parent.lower()
        child_name = joint.child.lower()

        # Case 1: parent is a link/bracket → child should extend from front
        # (link→joint, link→link, link→effector)
        if _is_link_like(parent_name):
            logger.info(
                "Fixing anchors for joint %d ('%s'→'%s'): top/bottom → front/back",
                i, joint.parent, joint.child,
            )
            joint.parent_anchor = "front"
            joint.child_anchor = "back"
            continue

        # Case 2: child is end_effector/gripper → use front/back
        if _is_end_effector(child_name):
            logger.info(
                "Fixing anchors for end-effector joint %d ('%s'→'%s'): top/bottom → front/back",
                i, joint.parent, joint.child,
            )
            joint.parent_anchor = "front"
            joint.child_anchor = "back"
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
    """Ensure gripper fingers are symmetrically separated for render visibility.

    The LLM often generates gripper finger joints with inconsistent anchors or
    relies on auto-distribution that places fingers too close together.  This
    causes the fingers to appear as a single solid block in box-based VLM
    renders, failing verification.

    This sanitizer:
    1. Detects left/right finger pairs by name.
    2. Overrides their joints to use identical front/back anchors.
    3. Sets ``no_distribute=True`` to prevent auto-distribution.
    4. Sets explicit lateral (**X**) offsets so fingers are visibly separated.

    The offset is applied in X.  An earlier attempt used Y separation, but
    this triggered the solver's ``_clamp_child_offset`` because the Y
    displacement combined with anchor offsets produced a large 3D distance,
    causing the entire displacement vector to be scaled down and the fingers
    to collapse together.  X separation avoids the clamping path while still
    producing a visible gap between the fingers in box-based VLM renders.
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

    # Use a consistent anchor pair — both fingers attach to the same face
    parent_anchor = left_joint.parent_anchor
    child_anchor = left_joint.child_anchor
    right_joint.parent_anchor = parent_anchor
    right_joint.child_anchor = child_anchor

    # Disable auto-distribution so explicit offsets are the sole lateral factor
    left_joint.no_distribute = True
    right_joint.no_distribute = True

    # Compute lateral gap from the parent (gripper base) width, fallback 18mm.
    parent_part = parts_by_name.get(left_joint.parent)
    gap = 18.0
    if parent_part and parent_part.dimensions:
        for key in ("width", "depth"):
            if key in parent_part.dimensions:
                w = parent_part.dimensions[key]
                gap = max(12.0, min(w * 0.35, 30.0))
                break

    # Offset in X.  See docstring for why X is used instead of Y (Y triggers
    # the solver's offset-clamping path and collapses the fingers).
    left_joint.offset = (-gap, 0.0, 0.0)
    right_joint.offset = (gap, 0.0, 0.0)

    logger.info(
        "Sanitizer: normalized gripper fingers '%s'/'%s' — "
        "anchors=%s/%s, gap=±%.1fmm",
        left_name, right_name, parent_anchor, child_anchor, gap,
    )
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
    "You are a robot assembly quality inspector.  Examine the 3D render and "
    "check whether this robot assembly is physically plausible.\n\n"
    "Check for:\n"
    "1. Parts floating in mid-air with no support\n"
    "2. Parts intersecting / overlapping each other\n"
    "3. Arms pointing in impossible directions (e.g. going through the body)\n"
    "4. Critical parts missing (no base plate, no main body)\n"
    "5. Overall structural coherence\n"
    "6. Parts with WRONG ORIENTATION (e.g. cylinders oriented along wrong axis)\n"
    "7. Gripper at the end of arm should look like a gripper (two fingers "
    "or a claw), not just another arm link\n\n"
    "NOTE: Wheeled robots SHOULD have wheels near the ground. Fixed-base "
    "arms should NOT have wheels. Do NOT report missing wheels for arms.\n\n"
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

    return problems


def _vlm_check_assembly(
    positions: dict[str, dict],
    parts: list[dict],
    render_dir: str,
    api_key: str,
    base_url: str,
    vision_model: str = "GLM-4.6V-Flash",
) -> tuple[bool, list[str]]:
    """Render assembly and run VLM verification. Returns (passed, problems)."""
    from ..models.base import Message
    from ..models.glm import GLMBackend

    from .vtk_renderer import render_assembly_from_positions

    # Render 4 views
    rendered = render_assembly_from_positions(
        parts=parts,
        positions=positions,
        output_dir=render_dir,
        views=["isometric", "front", "top", "right"],
    )
    if not rendered:
        return False, ["VTK rendering produced no images"]

    # Check each view with VLM — aggregate problems
    backend = GLMBackend(api_key=api_key, base_url=base_url,
                          vision_model=vision_model)
    all_problems: list[str] = []
    pass_count = 0
    total_views = len(rendered)

    for view_path in rendered:
        try:
            resp = backend.vision(
                image_path=view_path,
                prompt=_VLM_VERIFY_PROMPT,
            )
            text = str(resp).lower()
            if '"passed": true' in text or '"passed":true' in text:
                pass_count += 1
            # Always extract problems (even from "passed" views)
            try:
                start = str(resp).find("{")
                end = str(resp).rfind("}") + 1
                data = json.loads(str(resp)[start:end])
                for p in data.get("problems", []):
                    if p and p not in all_problems:
                        all_problems.append(p)
            except (json.JSONDecodeError, ValueError):
                pass
        except Exception as e:
            logger.warning("VLM check failed for %s: %s", view_path, e)

    # Majority vote: >50% of views must pass
    passed = pass_count > total_views / 2

    # Geometric pre-validation as safety net
    geo_problems = _geometric_prevalidation(parts, positions)
    if geo_problems:
        passed = False
        for p in geo_problems:
            if p not in all_problems:
                all_problems.append(p)

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
                temperature=temperature,
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
            _validate_assembly(assembly)

        print(f"  Assembly: {assembly.name}, {len(assembly.parts)} parts, "
              f"{len(assembly.joints)} joints")

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
        try:
            passed, problems = _vlm_check_assembly(
                positions=positions,
                parts=parts_dicts,
                render_dir=round_render_dir,
                api_key=api_key,
                base_url=base_url,
                vision_model=vision_model,
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

    # --- Step D: Export (whether passed or max rounds reached) ---
    # Stamp the package with verification status so downstream consumers
    # can detect unverified outputs. We still export on failure to enable
    # debugging, but the design_report.json will flag it.
    export_dir = os.path.join(output_dir, "engineering_package")
    export_success = False
    verification_status = "PASSED" if passed else "FAILED_MAX_ROUNDS"
    last_warnings = problems_history[-1] if problems_history else []
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
             "no_distribute": j.no_distribute}
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
