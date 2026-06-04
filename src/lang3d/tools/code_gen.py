"""Firmware code generation for robotic assemblies.

Generates Arduino/ESP32 C/C++ code including:
  - IK solver (from assembly chain parameters)
  - PWM servo control
  - Serial communication protocol
  - Trapezoidal velocity interpolation

Tools:
  gen_firmware        - Generate complete firmware (.ino + .cpp + .h)
  gen_wiring_diagram   - Generate wiring instructions
  gen_test_sequence    - Generate test motion sequence
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..knowledge.actuators import get_actuator
from ..knowledge.mechanics import Assembly
from ..models.base import ToolDefinition
from .assembly_solver import _resolve_assembly
from .base import Tool
from .ik_solver import _extract_chain


# ---------------------------------------------------------------------------
# Firmware generation
# ---------------------------------------------------------------------------

def generate_firmware(
    assembly: Assembly,
    actuator_ids: list[str],
    controller: str = "esp32",
    baud_rate: int = 115200,
) -> dict[str, str]:
    """Generate firmware files for a robotic assembly.

    Returns dict of {filename: content}.
    """
    links, base_height = _extract_chain(assembly)

    # Collect joint info
    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
    joint_names = [j.child for j in revolute_joints]
    joint_ranges = [(j.range_deg[0], j.range_deg[1]) for j in revolute_joints]
    joint_axes = [j.axis for j in revolute_joints]

    # Link lengths for IK (only pitch-axis links)
    pitch_links = [l for l in links if l.axis == "y"]
    L1 = pitch_links[0].length if len(pitch_links) > 0 else 100.0
    L2 = pitch_links[1].length if len(pitch_links) > 1 else 80.0

    # Has base yaw joint?
    has_base_yaw = any(l.axis == "z" for l in links[:1])

    # Servo pins
    servo_pins = _assign_pins(len(joint_names), controller)

    files: dict[str, str] = {}

    # Main .ino file
    files["robot_arm.ino"] = _gen_main_ino(
        joint_names=joint_names,
        servo_pins=servo_pins,
        controller=controller,
        baud_rate=baud_rate,
    )

    # IK solver
    files["ik_solver.h"] = _gen_ik_header(joint_names)
    files["ik_solver.cpp"] = _gen_ik_cpp(
        joint_names=joint_names,
        joint_ranges=joint_ranges,
        L1=L1,
        L2=L2,
        base_height=base_height,
        has_base_yaw=has_base_yaw,
    )

    # Servo driver
    files["servo_driver.h"] = _gen_servo_header(joint_names)
    files["servo_driver.cpp"] = _gen_servo_cpp(
        joint_names=joint_names,
        joint_ranges=joint_ranges,
        servo_pins=servo_pins,
    )

    return files


def _assign_pins(num_joints: int, controller: str) -> list[int]:
    """Assign servo PWM pins based on controller type."""
    if controller == "esp32":
        # ESP32 PWM-capable GPIO pins
        esp32_pins = [2, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33]
        return esp32_pins[:num_joints]
    else:
        # Arduino Uno PWM pins
        uno_pins = [3, 5, 6, 9, 10, 11]
        return (uno_pins * 4)[:num_joints]


def _gen_main_ino(
    joint_names: list[str],
    servo_pins: list[int],
    controller: str,
    baud_rate: int,
) -> str:
    n = len(joint_names)
    servo_init = "\n".join(
        f"  servo_{i}.attach({pin});"
        for i, pin in enumerate(servo_pins)
    )
    servo_decls = "\n".join(
        f"Servo servo_{i};  // {joint_names[i]}"
        for i in range(n)
    )
    includes_lib = "ESP32Servo.h" if controller == "esp32" else "Servo.h"

    return f"""\
// Robot Arm Controller - Auto-generated
// Controller: {controller}
// Joints: {n}

#include <{includes_lib}>
#include "ik_solver.h"
#include "servo_driver.h"

{servo_decls}

// Current joint angles (degrees)
float current_angles[NUM_JOINTS] = {{0}};

// Target joint angles (degrees)
float target_angles[NUM_JOINTS] = {{0}};

// Interpolation state
unsigned long interp_start_ms = 0;
unsigned long interp_duration_ms = 1000;  // 1s default move time
bool is_moving = false;

void setup() {{
  Serial.begin({baud_rate});
  Serial.println("Robot Arm Ready");
  Serial.println("Commands: G<joint_angles_csv> | H<home> | T<duration_ms>");

  // Initialize servos
{servo_init}

  // Move to home position
  for (int i = 0; i < NUM_JOINTS; i++) {{
    current_angles[i] = 0;
    target_angles[i] = 0;
    servo_write(i, 0);
  }}
}}

void loop() {{
  // Check serial input
  if (Serial.available()) {{
    parse_command(Serial.readStringUntil('\\n'));
  }}

  // Interpolate towards target
  if (is_moving) {{
    unsigned long now = millis();
    float t = (float)(now - interp_start_ms) / interp_duration_ms;

    if (t >= 1.0) {{
      t = 1.0;
      is_moving = false;
    }}

    // Trapezoidal velocity profile (smooth start/stop)
    float s = smooth_step(t);

    for (int i = 0; i < NUM_JOINTS; i++) {{
      float angle = current_angles[i] + s * (target_angles[i] - current_angles[i]);
      servo_write(i, angle);
    }}

    if (!is_moving) {{
      // Snap to final position
      for (int i = 0; i < NUM_JOINTS; i++) {{
        current_angles[i] = target_angles[i];
        servo_write(i, current_angles[i]);
      }}
      Serial.println("OK");
    }}
  }}

  delay(10);
}}

// Smooth step: trapezoidal velocity approximation (Hermite interpolation)
float smooth_step(float t) {{
  return t * t * (3.0 - 2.0 * t);
}}

void parse_command(String cmd) {{
  cmd.trim();
  if (cmd.length() == 0) return;

  char mode = cmd.charAt(0);
  String args = cmd.substring(1);

  switch (mode) {{
    case 'G':  // Go to angles: G0,45.0,-30.0,0
      parse_angles(args, target_angles);
      interp_start_ms = millis();
      is_moving = true;
      Serial.print("Moving to: ");
      for (int i = 0; i < NUM_JOINTS; i++) {{
        Serial.print(target_angles[i], 1);
        if (i < NUM_JOINTS - 1) Serial.print(",");
      }}
      Serial.println();
      break;

    case 'H':  // Home
      for (int i = 0; i < NUM_JOINTS; i++) target_angles[i] = 0;
      interp_start_ms = millis();
      is_moving = true;
      Serial.println("Going home");
      break;

    case 'T':  // Set interpolation time (ms)
      interp_duration_ms = args.toInt();
      Serial.print("Move time: ");
      Serial.print(interp_duration_ms);
      Serial.println("ms");
      break;

    case 'P':  // Print current angles
      for (int i = 0; i < NUM_JOINTS; i++) {{
        Serial.print(current_angles[i], 1);
        if (i < NUM_JOINTS - 1) Serial.print(",");
      }}
      Serial.println();
      break;

    default:
      Serial.print("Unknown command: ");
      Serial.println(cmd);
      break;
  }}
}}

void parse_angles(String args, float *angles) {{
  int idx = 0;
  int start = 0;
  for (int i = 0; i <= args.length() && idx < NUM_JOINTS; i++) {{
    if (i == args.length() || args.charAt(i) == ',') {{
      angles[idx++] = args.substring(start, i).toFloat();
      start = i + 1;
    }}
  }}
}}
"""


def _gen_ik_header(joint_names: list[str]) -> str:
    n = len(joint_names)
    names_str = "  // " + ", ".join(f"{i}: {nm}" for i, nm in enumerate(joint_names))
    return f"""\
// IK Solver - Auto-generated
// Solves inverse kinematics for target end-effector position

#ifndef IK_SOLVER_H
#define IK_SOLVER_H

#include <math.h>

#define NUM_JOINTS {n}
#define PI 3.14159265358979323846

{names_str}

struct IKSolution {{
  float angles[NUM_JOINTS];
  float error_mm;
  bool reachable;
}};

// Solve IK for target position (mm)
// Returns joint angles in degrees
IKSolution ik_solve(float target_x, float target_y, float target_z);

// Forward kinematics: given angles, compute end-effector position
void fk_compute(const float angles[NUM_JOINTS], float pos[3]);

#endif
"""


def _gen_ik_cpp(
    joint_names: list[str],
    joint_ranges: list[tuple[float, float]],
    L1: float,
    L2: float,
    base_height: float,
    has_base_yaw: bool,
) -> str:
    # Joint limit arrays
    min_angles = ", ".join(f"{r[0]:.1f}f" for r in joint_ranges)
    max_angles = ", ".join(f"{r[1]:.1f}f" for r in joint_ranges)

    return f"""\
// IK Solver Implementation - Auto-generated

#include "ik_solver.h"

// Link lengths (mm)
static const float LINK_1 = {L1:.1f}f;  // Shoulder to elbow
static const float LINK_2 = {L2:.1f}f;  // Elbow to wrist
static const float BASE_HEIGHT = {base_height:.1f}f;

// Joint limits (degrees)
static const float JOINT_MIN[NUM_JOINTS] = {{{min_angles}}};
static const float JOINT_MAX[NUM_JOINTS] = {{{max_angles}}};

// Clamp angle to joint limits
static float clamp_angle(int joint, float angle) {{
  if (angle < JOINT_MIN[joint]) return JOINT_MIN[joint];
  if (angle > JOINT_MAX[joint]) return JOINT_MAX[joint];
  return angle;
}}

IKSolution ik_solve(float target_x, float target_y, float target_z) {{
  IKSolution sol;
  for (int i = 0; i < NUM_JOINTS; i++) sol.angles[i] = 0;
  sol.error_mm = 0;
  sol.reachable = true;

  // Step 1: Base rotation (yaw around Z)
  float theta0 = atan2(target_y, target_x) * 180.0f / PI;

  // Horizontal distance from base axis
  float r = sqrt(target_x * target_x + target_y * target_y);

  // Height relative to shoulder
  float z_rel = target_z - BASE_HEIGHT;

  // Step 2: 2-link planar IK in vertical plane
  float D = sqrt(r * r + z_rel * z_rel);
  float max_reach = LINK_1 + LINK_2;

  if (D > max_reach * 0.99f) {{
    // Target beyond reach — stretch towards it
    D = max_reach * 0.99f;
    float scale = D / sqrt(r * r + z_rel * z_rel + 0.001f);
    r *= scale;
    z_rel *= scale;
    sol.reachable = false;
  }}

  float D2 = D * D;
  float L1_2 = LINK_1 * LINK_1;
  float L2_2 = LINK_2 * LINK_2;

  // Elbow angle (cosine law)
  float cos_elbow = (L1_2 + L2_2 - D2) / (2.0f * LINK_1 * LINK_2);
  if (cos_elbow > 1.0f) cos_elbow = 1.0f;
  if (cos_elbow < -1.0f) cos_elbow = -1.0f;
  float elbow_angle = PI - acos(cos_elbow);

  // Shoulder angle
  float alpha = atan2(z_rel, r);
  float cos_beta = (L1_2 + D2 - L2_2) / (2.0f * LINK_1 * D + 0.001f);
  if (cos_beta > 1.0f) cos_beta = 1.0f;
  if (cos_beta < -1.0f) cos_beta = -1.0f;
  float beta = acos(cos_beta);
  float shoulder_angle = alpha + beta;

  // Convert to degrees
  sol.angles[0] = clamp_angle(0, theta0);
  sol.angles[1] = clamp_angle(1, shoulder_angle * 180.0f / PI);
  sol.angles[2] = clamp_angle(2, elbow_angle * 180.0f / PI);

  // Remaining joints: set to 0 (or interpolate)
  for (int i = 3; i < NUM_JOINTS; i++) {{
    sol.angles[i] = 0;
  }}

  // FK verification to compute error
  float actual[3];
  fk_compute(sol.angles, actual);
  float dx = target_x - actual[0];
  float dy = target_y - actual[1];
  float dz = target_z - actual[2];
  sol.error_mm = sqrt(dx * dx + dy * dy + dz * dz);

  return sol;
}}

void fk_compute(const float angles[NUM_JOINTS], float pos[3]) {{
  // Simplified FK: assumes base yaw + 2 pitch links
  float theta0 = angles[0] * PI / 180.0f;
  float theta1 = angles[1] * PI / 180.0f;
  float theta2 = angles[2] * PI / 180.0f;

  // Projected reach in horizontal plane
  float reach = 0;
  reach += LINK_1 * cos(theta1);
  reach += LINK_2 * cos(theta1 - theta2);

  // Height
  float height = BASE_HEIGHT;
  height += LINK_1 * sin(theta1);
  height += LINK_2 * sin(theta1 - theta2);

  pos[0] = reach * cos(theta0);
  pos[1] = reach * sin(theta0);
  pos[2] = height;
}}
"""


def _gen_servo_header(joint_names: list[str]) -> str:
    n = len(joint_names)
    names_str = "  // " + ", ".join(f"{i}: {nm}" for i, nm in enumerate(joint_names))
    return f"""\
// Servo Driver - Auto-generated

#ifndef SERVO_DRIVER_H
#define SERVO_DRIVER_H

#include <Arduino.h>

#define NUM_SERVOS {n}

{names_str}

// Initialize all servos
void servo_init();

// Write angle (degrees) to servo
void servo_write(int servo_id, float angle_deg);

// Read current angle from servo (if supported)
float servo_read(int servo_id);

#endif
"""


def _gen_servo_cpp(
    joint_names: list[str],
    joint_ranges: list[tuple[float, float]],
    servo_pins: list[int],
) -> str:
    n = len(joint_names)
    min_angles = ", ".join(f"{r[0]:.1f}f" for r in joint_ranges)
    max_angles = ", ".join(f"{r[1]:.1f}f" for r in joint_ranges)
    pin_list = ", ".join(str(p) for p in servo_pins)

    return f"""\
// Servo Driver Implementation - Auto-generated

#include "servo_driver.h"

// PWM pulse width range (microseconds)
#define SERVO_MIN_PULSE 500
#define SERVO_MAX_PULSE 2500

// Joint angle limits (degrees)
static const float SERVO_MIN_ANGLE[NUM_SERVOS] = {{{min_angles}}};
static const float SERVO_MAX_ANGLE[NUM_SERVOS] = {{{max_angles}}};

// Servo pins
static const int SERVO_PINS[NUM_SERVOS] = {{{pin_list}}};

// Map angle to PWM pulse width
static int angle_to_pwm(int servo_id, float angle_deg) {{
  // Clamp to limits
  if (angle_deg < SERVO_MIN_ANGLE[servo_id]) angle_deg = SERVO_MIN_ANGLE[servo_id];
  if (angle_deg > SERVO_MAX_ANGLE[servo_id]) angle_deg = SERVO_MAX_ANGLE[servo_id];

  // Normalize to 0-1 range
  float range = SERVO_MAX_ANGLE[servo_id] - SERVO_MIN_ANGLE[servo_id];
  float norm = (range > 0) ? (angle_deg - SERVO_MIN_ANGLE[servo_id]) / range : 0.5f;

  // Map to pulse width
  return (int)(SERVO_MIN_PULSE + norm * (SERVO_MAX_PULSE - SERVO_MIN_PULSE));
}}

void servo_init() {{
  // Pin mode is set by Servo library attach()
}}

void servo_write(int servo_id, float angle_deg) {{
  if (servo_id < 0 || servo_id >= NUM_SERVOS) return;
  // Use the main .ino servo objects via extern
  extern void servo_raw_write(int id, int pin, int pwm);
  int pwm = angle_to_pwm(servo_id, angle_deg);
  servo_raw_write(servo_id, SERVO_PINS[servo_id], pwm);
}}

float servo_read(int servo_id) {{
  // Most hobby servos don't support angle readback
  // Return 0 as placeholder
  return 0.0f;
}}
"""


# ---------------------------------------------------------------------------
# Wiring diagram
# ---------------------------------------------------------------------------

def generate_wiring(
    actuator_ids: list[str],
    controller: str = "esp32",
) -> str:
    """Generate text-based wiring instructions."""
    lines = [
        f"# Wiring Diagram — {controller.upper()}",
        "",
        "## Components",
        f"- Controller: {controller.upper()}",
    ]

    # Collect actuators
    actuators_used: dict[str, int] = {}
    for aid in actuator_ids:
        a = get_actuator(aid)
        if a:
            key = a.id
            actuators_used[key] = actuators_used.get(key, 0) + 1

    pins = _assign_pins(sum(actuators_used.values()), controller)
    pin_idx = 0

    lines.append("")
    lines.append("## Power Supply")
    voltages = set()
    for aid, count in actuators_used.items():
        a = get_actuator(aid)
        if a:
            voltages.add(a.voltage)
    for v in sorted(voltages):
        lines.append(f"- {v}V power supply (shared ground with {controller.upper()})")

    lines.append("")
    lines.append("## Connections")
    lines.append("```")
    lines.append(f"{'Servo':<20} {'Signal Pin':<12} {'VCC':<8} {'GND':<8}")
    lines.append("-" * 50)

    for aid, count in actuators_used.items():
        a = get_actuator(aid)
        if not a:
            continue
        for c in range(count):
            label = f"{a.name} #{c + 1}" if count > 1 else a.name
            pin = pins[pin_idx] if pin_idx < len(pins) else "?"
            pin_idx += 1
            lines.append(f"{label:<20} GPIO{pin:<7} {a.voltage}V     GND")

    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append(f"- All servo GND wires connect to {controller.upper()} GND")
    lines.append("- Use separate power supply for servos (do NOT power from USB)")
    lines.append("- Add 100uF capacitor across servo power rails")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test sequence
# ---------------------------------------------------------------------------

def generate_test_sequence(
    assembly: Assembly,
    steps: int = 5,
) -> str:
    """Generate a test motion sequence for the assembly."""
    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
    n = len(revolute_joints)

    lines = [
        f"# Test Sequence — {assembly.name}",
        "",
        f"Joints: {n}",
        "",
    ]

    # Step 1: Individual joint test
    lines.append("## Phase 1: Individual Joint Test")
    lines.append("```")
    for i, j in enumerate(revolute_joints):
        mid = (j.range_deg[0] + j.range_deg[1]) / 2
        lines.append(f"# Test {j.description or j.child}")
        angles = [0.0] * n
        angles[i] = mid * 0.5
        lines.append("G" + ",".join(f"{a:.1f}" for a in angles))
        lines.append("# Wait 2 seconds, observe movement")
        lines.append("G" + ",".join("0.0" for _ in range(n)))
        lines.append("")
    lines.append("```")

    # Step 2: Range of motion
    lines.append("")
    lines.append("## Phase 2: Range of Motion")
    lines.append("```")
    for j in revolute_joints:
        lines.append(f"# {j.description or j.child}: range {j.range_deg[0]} to {j.range_deg[1]} deg")
    lines.append("```")

    # Step 3: Combined motion
    lines.append("")
    lines.append("## Phase 3: Combined Motion Test")
    lines.append("```")
    for step_i in range(steps):
        t = step_i / max(steps - 1, 1)
        angles = []
        for j in revolute_joints:
            mid = (j.range_deg[0] + j.range_deg[1]) / 2
            amp = (j.range_deg[1] - j.range_deg[0]) / 4
            angle = mid + amp * math.sin(2 * math.pi * t)
            angle = max(j.range_deg[0], min(j.range_deg[1], angle))
            angles.append(round(angle, 1))
        lines.append(f"G" + ",".join(str(a) for a in angles))
    lines.append("G" + ",".join("0" for _ in range(n)))
    lines.append("```")

    lines.append("")
    lines.append("## Check")
    lines.append("- [ ] All joints move smoothly")
    lines.append("- [ ] No grinding or binding sounds")
    lines.append("- [ ] Home position is correct")
    lines.append("- [ ] No abnormal heating")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class GenFirmwareTool(Tool):
    name = "gen_firmware"
    description = (
        "生成机器人控制固件代码（Arduino/ESP32）。"
        "输出 IK 求解器、舵机驱动、串口通信、平滑插值。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "assembly_name": {"type": "string", "description": "装配体名称"},
                "actuator_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "执行器 ID 列表（按关节顺序）",
                },
                "controller": {
                    "type": "string", "enum": ["esp32", "arduino"],
                    "description": "控制器类型（默认 esp32）",
                },
                "output_dir": {"type": "string", "description": "输出目录路径"},
            }, "required": ["actuator_ids"]},
        )

    def execute(self, *, assembly_name: str = "robotic_arm",
                actuator_ids: list[str] | None = None,
                controller: str = "esp32", output_dir: str = "",
                **kwargs: Any) -> str:
        if not actuator_ids:
            return "错误：未指定执行器 ID 列表"

        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        files = generate_firmware(assembly, actuator_ids, controller)

        # Optionally write to disk
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for fname, content in files.items():
                (out / fname).write_text(content, encoding="utf-8")

        lines = [
            f"[Firmware Generated] {assembly.name}",
            f"Controller: {controller}",
            f"Actuators: {', '.join(actuator_ids)}",
            f"Files: {len(files)}",
            "",
            "--- Files ---",
        ]
        for fname, content in files.items():
            lines.append(f"  {fname} ({len(content.splitlines())} lines)")

        if output_dir:
            lines.append(f"\nSaved to: {output_dir}")

        lines.append("\n--- Preview: ik_solver.h ---")
        lines.append(files.get("ik_solver.h", "")[:500])

        return "\n".join(lines)


class GenWiringDiagramTool(Tool):
    name = "gen_wiring_diagram"
    description = "生成执行器接线说明（引脚对应表 + 电源配置）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "actuator_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "执行器 ID 列表",
                },
                "controller": {
                    "type": "string", "enum": ["esp32", "arduino"],
                    "description": "控制器类型",
                },
            }, "required": ["actuator_ids"]},
        )

    def execute(self, *, actuator_ids: list[str] | None = None,
                controller: str = "esp32", **kwargs: Any) -> str:
        if not actuator_ids:
            return "错误：未指定执行器 ID"
        return generate_wiring(actuator_ids, controller)


class GenTestSequenceTool(Tool):
    name = "gen_test_sequence"
    description = "生成机器人测试动作序列（逐关节测试 → 全联动测试）"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "assembly_name": {"type": "string", "description": "装配体名称"},
                "steps": {"type": "integer", "description": "联动测试步数（默认 5）"},
            }, "required": []},
        )

    def execute(self, *, assembly_name: str = "robotic_arm",
                steps: int = 5, **kwargs: Any) -> str:
        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"
        return generate_test_sequence(assembly, steps)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_code_gen_tools(registry: Any) -> None:
    """Register code generation tools."""
    registry.register(GenFirmwareTool())
    registry.register(GenWiringDiagramTool())
    registry.register(GenTestSequenceTool())
