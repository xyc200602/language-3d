"""Firmware code generation for robotic assemblies.

Generates Arduino/ESP32 C/C++ code including:
  - IK solver (from assembly chain parameters)
  - PWM servo control
  - Serial communication protocol
  - Trapezoidal velocity interpolation
  - Sensor integration (limit switches, encoders, IMU)

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

from ..knowledge.actuators import get_actuator, get_motor_pid_spec, DCMotorPIDSpec
from ..knowledge.mechanics import Assembly
from ..knowledge.sensors import (
    Sensor,
    get_sensor,
    list_sensors,
    recommend_sensors_for_joints,
    get_encoder_spec,
    EncoderSpec,
)
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
    sensor_ids: list[str] | None = None,
) -> dict[str, str]:
    """Generate firmware files for a robotic assembly.

    Args:
        assembly: The mechanical assembly definition.
        actuator_ids: List of actuator IDs (one per joint).
        controller: "esp32" or "arduino".
        baud_rate: Serial baud rate.
        sensor_ids: Optional list of sensor IDs to integrate (e.g.
            ["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"]).

    Returns dict of {filename: content}.
    """
    links, base_height = _extract_chain(assembly)

    # Collect joint info
    revolute_joints = [j for j in assembly.joints if j.type == "revolute"]
    joint_names = [j.child for j in revolute_joints]
    joint_ranges = [(j.range_deg[0], j.range_deg[1]) for j in revolute_joints]
    joint_axes = [j.axis for j in revolute_joints]

    # Link lengths for IK (only pitch-axis links).
    # Under the clean arm convention pitch joints use axis="x"; legacy
    # assemblies used axis="y". Accept both, but prefer "x" when both are
    # present so mixed conventions resolve to the cleaner chain.
    pitch_links_x = [l for l in links if l.axis == "x"]
    pitch_links_y = [l for l in links if l.axis == "y"]
    pitch_links = pitch_links_x if pitch_links_x else pitch_links_y
    L1 = pitch_links[0].length if len(pitch_links) > 0 else 100.0
    L2 = pitch_links[1].length if len(pitch_links) > 1 else 80.0

    # Has base yaw joint?
    has_base_yaw = any(l.axis == "z" for l in links[:1])

    # Servo pins
    servo_pins = _assign_pins(len(joint_names), controller)

    # Resolve sensor objects
    sensors: list[Sensor] = []
    if sensor_ids:
        for sid in sensor_ids:
            s = get_sensor(sid)
            if s:
                sensors.append(s)

    files: dict[str, str] = {}

    # Main .ino file
    files["robot_arm.ino"] = _gen_main_ino(
        joint_names=joint_names,
        servo_pins=servo_pins,
        controller=controller,
        baud_rate=baud_rate,
        sensors=sensors,
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

    # Sensor driver (only if sensors are specified)
    if sensors:
        files["sensor_driver.h"] = _gen_sensor_header(sensors)
        files["sensor_driver.cpp"] = _gen_sensor_cpp(sensors, controller)

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
    sensors: list[Sensor] | None = None,
) -> str:
    n = len(joint_names)
    servo_init = "\n".join(
        f"  servo_{i}.attach({pin});"
        for i, pin in enumerate(servo_pins)
    )
    servo_decls_lines = [f"Servo servo_{i};  // {joint_names[i]}" for i in range(n)]
    servo_decls_lines.append(
        f"Servo* const SERVO_OBJECTS[NUM_JOINTS] = {{"
        + ", ".join(f"&servo_{i}" for i in range(n)) + "};"
    )
    servo_decls = "\n".join(servo_decls_lines)
    includes_lib = "ESP32Servo.h" if controller == "esp32" else "Servo.h"

    # Sensor includes and init
    sensor_include = ""
    sensor_init_block = ""
    sensor_read_block = ""
    sensor_status_cmd = ""

    if sensors:
        sensor_include = '#include "sensor_driver.h"'
        sensor_init_lines = ["  sensor_init();"]
        sensor_init_block = "\n".join(sensor_init_lines)

        # Periodic sensor reading in loop
        sensor_read_lines = [
            "  // Read sensors",
            "  sensor_read_all();",
        ]
        # Check limit switches before moving
        has_limit = any(s.category == "limit_switch" for s in sensors)
        if has_limit:
            sensor_read_lines.extend([
                "",
                "  // Safety: check limit switches before moving",
                "  if (is_moving) {",
                "    for (int i = 0; i < NUM_JOINTS; i++) {",
                "      if (sensor_limit_triggered(i)) {",
                "        is_moving = false;",
                "        Serial.print(\"LIMIT: joint \");",
                "        Serial.println(i);",
                "        break;",
                "      }",
                "    }",
                "  }",
            ])
        sensor_read_block = "\n".join(sensor_read_lines)

        sensor_status_cmd = """
    case 'S':  // Sensor status
      sensor_print_status();
      break;
"""

    # Build the complete .ino
    return f"""\
// Robot Arm Controller - Auto-generated
// Controller: {controller}
// Joints: {n}
{"// Sensors: " + ", ".join(s.name for s in sensors) if sensors else ""}

#include <{includes_lib}>
#include "ik_solver.h"
#include "servo_driver.h"
{sensor_include}

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
  Serial.println("Commands: G<joint_angles_csv> | H<home> | T<duration_ms> | P<print> | S<sensor status>");

  // Initialize servos
{servo_init}
{sensor_init_block}

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

{sensor_read_block}

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
{sensor_status_cmd}
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

// Direct PWM write bridge for servo_driver.cpp
void servo_raw_write(int id, int pin, int pwm) {{
  if (id < 0 || id >= NUM_JOINTS) return;
  SERVO_OBJECTS[id]->writeMicroseconds(pwm);
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
# Sensor driver generation
# ---------------------------------------------------------------------------

def _gen_sensor_header(sensors: list[Sensor]) -> str:
    """Generate sensor_driver.h."""
    sensor_list = "\n".join(
        f"  // {i}: {s.name} ({s.category})" for i, s in enumerate(sensors)
    )
    has_encoder = any(s.category == "encoder" for s in sensors)
    has_limit = any(s.category == "limit_switch" for s in sensors)
    has_imu = any(s.category == "imu" for s in sensors)
    has_analog = any(s.category == "potentiometer" for s in sensors)

    decls = ""
    if has_encoder:
        decls += """
// Read encoder angle for a joint (degrees)
float sensor_read_encoder(int joint_index);
"""
    if has_limit:
        decls += """
// Check if limit switch is triggered for a joint
bool sensor_limit_triggered(int joint_index);
"""
    if has_imu:
        decls += """
// Read IMU orientation (roll, pitch, yaw in degrees)
void sensor_read_imu(float orientation[3]);
"""
    if has_analog:
        decls += """
// Read potentiometer angle for a joint (degrees)
float sensor_read_potentiometer(int joint_index);
"""

    return f"""\
// Sensor Driver - Auto-generated
// Sensors: {len(sensors)}
{sensor_list}

#ifndef SENSOR_DRIVER_H
#define SENSOR_DRIVER_H

#include <Arduino.h>

#define NUM_SENSORS {len(sensors)}

// Initialize all sensors
void sensor_init(void);

// Read all sensors (call in main loop)
void sensor_read_all(void);

// Print sensor status to Serial
void sensor_print_status(void);
{decls}

#endif
"""


def _gen_sensor_cpp(sensors: list[Sensor], controller: str) -> str:
    """Generate sensor_driver.cpp."""
    includes = '#include "sensor_driver.h"\n'
    init_lines: list[str] = []
    read_lines: list[str] = []
    status_lines: list[str] = []

    has_encoder = any(s.category == "encoder" for s in sensors)
    has_limit = any(s.category == "limit_switch" for s in sensors)
    has_imu = any(s.category == "imu" for s in sensors)
    has_analog = any(s.category == "potentiometer" for s in sensors)

    # Collect pin assignments
    encoder_sensors = [s for s in sensors if s.category == "encoder"]
    limit_sensors = [s for s in sensors if s.category == "limit_switch"]
    imu_sensors = [s for s in sensors if s.category == "imu"]
    pot_sensors = [s for s in sensors if s.category == "potentiometer"]

    # I2C includes
    needs_i2c = has_encoder or has_imu
    if needs_i2c:
        includes += "#include <Wire.h>\n"

    if has_imu:
        includes += "#include <MPU6050.h>\n"

    # Pin definitions for limit switches
    if has_limit:
        limit_pins = ", ".join(
            str(25 + i) if controller == "esp32" else str(2 + i)
            for i in range(len(limit_sensors))
        )
        limit_decls = f"""\
static const int NUM_LIMIT_SWITCHES = {len(limit_sensors)};
static const int LIMIT_SWITCH_PINS[NUM_LIMIT_SWITCHES] = {{{limit_pins}}};
static bool limit_state[NUM_LIMIT_SWITCHES] = {{false}};
"""
    else:
        limit_decls = ""

    # Pin definitions for potentiometers
    if has_analog:
        pot_pins = ", ".join(
            str(36 + i) if controller == "esp32" else str(14 + i)
            for i in range(len(pot_sensors))
        )
        pot_decls = f"""\
static const int NUM_POTS = {len(pot_sensors)};
static const int POT_PINS[NUM_POTS] = {{{pot_pins}}};
static float pot_angles[NUM_POTS] = {{0}};
"""
    else:
        pot_decls = ""

    # Encoder state
    if has_encoder:
        encoder_decls = f"""\
static const int NUM_ENCODERS = {len(encoder_sensors)};
static float encoder_angles[NUM_ENCODERS] = {{0}};
// AS5600 I2C address
#define AS5600_ADDR 0x36
#define AS5600_RAW_ANGLE_H 0x0C
#define AS5600_RAW_ANGLE_L 0x0D
"""
    else:
        encoder_decls = ""

    # IMU state
    if has_imu:
        imu_decls = """\
static float imu_orientation[3] = {0, 0, 0};  // roll, pitch, yaw
"""
    else:
        imu_decls = ""

    # Init code
    init_lines.append("void sensor_init() {")
    if needs_i2c:
        init_lines.append("  Wire.begin();")
    if has_limit:
        init_lines.append(f"  for (int i = 0; i < NUM_LIMIT_SWITCHES; i++) {{")
        init_lines.append(f"    pinMode(LIMIT_SWITCH_PINS[i], INPUT_PULLUP);")
        init_lines.append(f"  }}")
    if has_imu:
        init_lines.append("  // MPU6050 init is handled by library begin()")
    init_lines.append("}")

    # Read all sensors
    read_lines.append("void sensor_read_all() {")
    if has_encoder:
        read_lines.append("  for (int i = 0; i < NUM_ENCODERS; i++) {")
        read_lines.append("    encoder_angles[i] = read_as5600_angle(AS5600_ADDR);")
        read_lines.append("  }")
    if has_limit:
        read_lines.append("  for (int i = 0; i < NUM_LIMIT_SWITCHES; i++) {")
        read_lines.append("    limit_state[i] = (digitalRead(LIMIT_SWITCH_PINS[i]) == LOW);")
        read_lines.append("  }")
    if has_analog:
        read_lines.append("  for (int i = 0; i < NUM_POTS; i++) {")
        read_lines.append("    int raw = analogRead(POT_PINS[i]);")
        read_lines.append("    pot_angles[i] = (float)raw / 4095.0f * 300.0f;  // Map ADC to 0-300 deg")
        read_lines.append("  }")
    read_lines.append("}")

    # Status print
    status_lines.append("void sensor_print_status() {")
    if has_encoder:
        status_lines.append("  Serial.println(\"--- Encoders ---\");")
        status_lines.append("  for (int i = 0; i < NUM_ENCODERS; i++) {")
        status_lines.append("    Serial.print(\"  Joint \"); Serial.print(i);")
        status_lines.append("    Serial.print(\": \"); Serial.print(encoder_angles[i], 1);")
        status_lines.append("    Serial.println(\" deg\");")
        status_lines.append("  }")
    if has_limit:
        status_lines.append("  Serial.println(\"--- Limit Switches ---\");")
        status_lines.append("  for (int i = 0; i < NUM_LIMIT_SWITCHES; i++) {")
        status_lines.append("    Serial.print(\"  Joint \"); Serial.print(i);")
        status_lines.append("    Serial.print(\": \"); Serial.println(limit_state[i] ? \"TRIGGERED\" : \"OK\");")
        status_lines.append("  }")
    if has_imu:
        status_lines.append("  Serial.println(\"--- IMU ---\");")
        status_lines.append("  Serial.print(\"  R: \"); Serial.print(imu_orientation[0], 1);")
        status_lines.append("  Serial.print(\" P: \"); Serial.print(imu_orientation[1], 1);")
        status_lines.append("  Serial.print(\" Y: \"); Serial.println(imu_orientation[2], 1);")
    status_lines.append("}")

    # Per-sensor-type functions
    func_blocks: list[str] = []

    if has_encoder:
        func_blocks.append("""
float sensor_read_encoder(int joint_index) {
  if (joint_index < 0 || joint_index >= NUM_ENCODERS) return 0;
  return encoder_angles[joint_index];
}

// Read AS5600 angle via I2C
static float read_as5600_angle(int addr) {
  Wire.beginTransmission(addr);
  Wire.write(AS5600_RAW_ANGLE_H);
  Wire.endTransmission(false);
  Wire.requestFrom(addr, 2);
  uint16_t raw = (Wire.read() << 8) | Wire.read();
  raw &= 0x0FFF;  // 12-bit mask
  return (float)raw / 4096.0f * 360.0f;
}
""")

    if has_limit:
        func_blocks.append("""
bool sensor_limit_triggered(int joint_index) {
  if (joint_index < 0 || joint_index >= NUM_LIMIT_SWITCHES) return false;
  return limit_state[joint_index];
}
""")

    if has_imu:
        func_blocks.append("""
void sensor_read_imu(float orientation[3]) {
  orientation[0] = imu_orientation[0];
  orientation[1] = imu_orientation[1];
  orientation[2] = imu_orientation[2];
}
""")

    if has_analog:
        func_blocks.append("""
float sensor_read_potentiometer(int joint_index) {
  if (joint_index < 0 || joint_index >= NUM_POTS) return 0;
  return pot_angles[joint_index];
}
""")

    init_block = "\n".join(init_lines)
    read_block = "\n".join(read_lines)
    status_block = "\n".join(status_lines)
    func_block = "\n".join(func_blocks)

    return f"""\
// Sensor Driver Implementation - Auto-generated

{includes}

{encoder_decls}
{limit_decls}
{pot_decls}
{imu_decls}

{init_block}

{read_block}

{status_block}
{func_block}
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
                "sensor_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "传感器 ID 列表（如 AS5600, LIMIT_SWITCH_MICRO, MPU6050）",
                },
                "output_dir": {"type": "string", "description": "输出目录路径"},
            }, "required": ["actuator_ids"]},
        )

    def execute(self, *, assembly_name: str = "robotic_arm",
                actuator_ids: list[str] | None = None,
                controller: str = "esp32", output_dir: str = "",
                sensor_ids: list[str] | None = None,
                **kwargs: Any) -> str:
        if not actuator_ids:
            return "错误：未指定执行器 ID 列表"

        assembly = _resolve_assembly(assembly_name, "")
        if assembly is None:
            return f"错误：未找到装配体 '{assembly_name}'"

        files = generate_firmware(assembly, actuator_ids, controller, sensor_ids=sensor_ids)

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

# ---------------------------------------------------------------------------
# DC Motor PID Controller
# ---------------------------------------------------------------------------


def gen_motor_driver_code(
    motors: list[dict[str, Any]],
    controller: str = "esp32",
) -> dict[str, str]:
    """Generate DC motor driver code with PID speed/position control.

    Args:
        motors: List of motor configs, each with keys:
            motor_id (str), encoder_id (str), pwm_pin (int),
            dir_pin1 (int), dir_pin2 (int), enc_a_pin (int), enc_b_pin (int)
        controller: "esp32" or "arduino".

    Returns dict of {filename: content}.
    """
    n = len(motors)

    # Resolve PID specs for each motor
    pid_specs: list[DCMotorPIDSpec | None] = []
    encoder_specs: list[EncoderSpec | None] = []
    for m in motors:
        pid_specs.append(get_motor_pid_spec(m.get("motor_id", "")))
        encoder_specs.append(get_encoder_spec(m.get("encoder_id", "")))

    # Generate header
    header = f"""\
// DC Motor Driver - Auto-generated
// Motors: {n}, Controller: {controller}

#ifndef DC_MOTOR_DRIVER_H
#define DC_MOTOR_DRIVER_H

#include <Arduino.h>

#define NUM_MOTORS {n}

struct MotorPins {{
  int pwm_pin;
  int dir1_pin;
  int dir2_pin;
  int enc_a_pin;
  int enc_b_pin;
}};

struct PIDState {{
  float kp, ki, kd;
  float integral;
  float prev_error;
  float output;
  int max_pwm;
  int sample_period_ms;
}};

// Initialize motor driver
void motor_init();

// Set motor speed (-255 to 255)
void motor_set_speed(int motor_id, int speed);

// PID update (call in timer interrupt or loop at sample_period_ms)
void motor_pid_update(int motor_id, float target_speed_rpm);

// Set PID target speed
void motor_set_target_rpm(int motor_id, float rpm);

// Set PID target position (encoder ticks)
void motor_set_target_ticks(int motor_id, long ticks);

// Read encoder count
long motor_read_encoder(int motor_id);

// Read current speed (RPM)
float motor_read_speed(int motor_id);

// Reset PID state
void motor_pid_reset(int motor_id);

#endif
"""

    # Build arrays for C code
    pwm_pins = ", ".join(str(m.get("pwm_pin", 0)) for m in motors)
    dir1_pins = ", ".join(str(m.get("dir_pin1", 0)) for m in motors)
    dir2_pins = ", ".join(str(m.get("dir_pin2", 0)) for m in motors)
    enc_a = ", ".join(str(m.get("enc_a_pin", 0)) for m in motors)
    enc_b = ", ".join(str(m.get("enc_b_pin", 0)) for m in motors)

    # PID defaults
    kp_vals = ", ".join(f"{(p.kp if p else 1.0):.2f}f" for p in pid_specs)
    ki_vals = ", ".join(f"{(p.ki if p else 0.3):.2f}f" for p in pid_specs)
    kd_vals = ", ".join(f"{(p.kd if p else 0.05):.2f}f" for p in pid_specs)
    max_pwm_vals = ", ".join(f"{(p.max_pwm if p else 200)}" for p in pid_specs)
    sample_ms = ", ".join(f"{(p.sample_period_ms if p else 20)}" for p in pid_specs)

    # Encoder PPR and gear ratios
    ppr_vals = ", ".join(f"{(e.ppr if e else 11)}" for e in encoder_specs)
    quad_vals = ", ".join("true" if (e and e.quadrature) else "false" for e in encoder_specs)
    gear_ratios = ", ".join(f"{(p.gear_ratio if p else 1.0):.1f}f" for p in pid_specs)

    # PID initializer block for C struct array
    pid_init_parts = []
    for p in pid_specs:
        if p:
            pid_init_parts.append(f"{{{p.kp:.2f}f, {p.ki:.2f}f, {p.kd:.2f}f, 0, 0, 0, {p.max_pwm}, {p.sample_period_ms}}}")
        else:
            pid_init_parts.append("{1.0f, 0.3f, 0.05f, 0, 0, 0, 200, 20}")
    pid_init_block = ", ".join(pid_init_parts)

    impl = f"""\
// DC Motor Driver Implementation - Auto-generated
// Motors: {n}

#include "dc_motor_driver.h"

// Pin assignments
static const MotorPins MOTORS[NUM_MOTORS] = {{
  {{{ pwm_pins.replace(', ', '}, {') }}}
}};
// Expanded for clarity:
static const int PWM_PINS[{n}] = {{{ pwm_pins }}};
static const int DIR1_PINS[{n}] = {{{ dir1_pins }}};
static const int DIR2_PINS[{n}] = {{{ dir2_pins }}};
static const int ENC_A_PINS[{n}] = {{{ enc_a }}};
static const int ENC_B_PINS[{n}] = {{{ enc_b }}};

// Encoder state (volatile for ISR)
static volatile long encoder_counts[{n}] = {{0}};
static long last_counts[{n}] = {{0}};

// Speed measurement
static float current_speed_rpm[{n}] = {{0}};

// PID state per motor (initialized properly in motor_init)
static PIDState pid[{n}] = {{
  {pid_init_block}
}};
// Initialize with proper values
static bool pid_initialized = false;

static const int MAX_PWM[{n}] = {{{ max_pwm_vals }}};
static const int SAMPLE_MS[{n}] = {{{ sample_ms }}};
static const int ENCODER_PPR[{n}] = {{{ ppr_vals }}};
static const bool ENCODER_QUAD[{n}] = {{{ quad_vals }}};
static const float GEAR_RATIO[{n}] = {{{ gear_ratios }}};

// Target
static float target_rpm[{n}] = {{0}};
static long target_ticks[{n}] = {{0}};
static bool position_mode[{n}] = {{false}};

// --- Encoder ISRs ---
"""

    # Generate ISR functions for each motor
    for i in range(n):
        impl += f"""
void IRAM_ATTR encoder_isr_{i}() {{
  static int last_state_{i} = 0;
  int a = digitalRead(ENC_A_PINS[{i}]);
  int b = digitalRead(ENC_B_PINS[{i}]);
  int state = (a << 1) | b;
  // Quadrature decode
  if (state != last_state_{i}) {{
    if ((state ^ last_state_{i}) == 0x03) {{
      encoder_counts[{i}]++;
    }} else {{
      encoder_counts[{i}]--;
    }}
    last_state_{i} = state;
  }}
}}
"""

    impl += """
void motor_init() {
"""

    for i in range(n):
        impl += f"""  pinMode(PWM_PINS[{i}], OUTPUT);
  pinMode(DIR1_PINS[{i}], OUTPUT);
  pinMode(DIR2_PINS[{i}], OUTPUT);
  pinMode(ENC_A_PINS[{i}], INPUT_PULLUP);
  pinMode(ENC_B_PINS[{i}], INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_A_PINS[{i}]), encoder_isr_{i}, CHANGE);
"""
        p = pid_specs[i]
        if p:
            impl += f"  pid[{i}] = {{{p.kp:.2f}f, {p.ki:.2f}f, {p.kd:.2f}f, 0, 0, 0, {p.max_pwm}, {p.sample_period_ms}}};\n"
        else:
            impl += f"  pid[{i}] = {{1.0f, 0.3f, 0.05f, 0, 0, 0, 200, 20}};\n"

    impl += """  pid_initialized = true;
}

void motor_set_speed(int motor_id, int speed) {
  if (motor_id < 0 || motor_id >= NUM_MOTORS) return;
  int abs_speed = (speed > 0) ? speed : -speed;
  if (abs_speed > 255) abs_speed = 255;

  if (speed >= 0) {
    digitalWrite(DIR1_PINS[motor_id], HIGH);
    digitalWrite(DIR2_PINS[motor_id], LOW);
  } else {
    digitalWrite(DIR1_PINS[motor_id], LOW);
    digitalWrite(DIR2_PINS[motor_id], HIGH);
  }
  analogWrite(PWM_PINS[motor_id], abs_speed);
}

void motor_pid_update(int motor_id, float target) {
  if (motor_id < 0 || motor_id >= NUM_MOTORS) return;

  // Read current encoder
  long current = encoder_counts[motor_id];
  float dt_sec = (float)pid[motor_id].sample_period_ms / 1000.0f;

  // Compute speed (RPM at output shaft)
  long delta = current - last_counts[motor_id];
  last_counts[motor_id] = current;
  int cpr = ENCODER_PPR[motor_id] * (ENCODER_QUAD[motor_id] ? 4 : 1);
  if (cpr == 0) cpr = 1;
  current_speed_rpm[motor_id] = (float)delta / (float)cpr * 60.0f / dt_sec / GEAR_RATIO[motor_id];

  if (position_mode[motor_id]) {
    // Position PID
    float error = (float)(target_ticks[motor_id] - current);
    pid[motor_id].integral += error * dt_sec;
    pid[motor_id].integral = constrain(pid[motor_id].integral, -1000, 1000);
    float deriv = (error - pid[motor_id].prev_error) / dt_sec;
    pid[motor_id].prev_error = error;
    pid[motor_id].output = pid[motor_id].kp * error +
                           pid[motor_id].ki * pid[motor_id].integral +
                           pid[motor_id].kd * deriv;
  } else {
    // Speed PID
    float error = target - current_speed_rpm[motor_id];
    pid[motor_id].integral += error * dt_sec;
    pid[motor_id].integral = constrain(pid[motor_id].integral, -500, 500);
    float deriv = (error - pid[motor_id].prev_error) / dt_sec;
    pid[motor_id].prev_error = error;
    pid[motor_id].output = pid[motor_id].kp * error +
                           pid[motor_id].ki * pid[motor_id].integral +
                           pid[motor_id].kd * deriv;
  }

  int pwm = (int)pid[motor_id].output;
  pwm = constrain(pwm, -pid[motor_id].max_pwm, pid[motor_id].max_pwm);
  motor_set_speed(motor_id, pwm);
}

void motor_set_target_rpm(int motor_id, float rpm) {
  if (motor_id < 0 || motor_id >= NUM_MOTORS) return;
  target_rpm[motor_id] = rpm;
  position_mode[motor_id] = false;
}

void motor_set_target_ticks(int motor_id, long ticks) {
  if (motor_id < 0 || motor_id >= NUM_MOTORS) return;
  target_ticks[motor_id] = ticks;
  position_mode[motor_id] = true;
}

long motor_read_encoder(int motor_id) {
  if (motor_id < 0 || motor_id >= NUM_MOTORS) return 0;
  return encoder_counts[motor_id];
}

float motor_read_speed(int motor_id) {
  if (motor_id < 0 || motor_id >= NUM_MOTORS) return 0;
  return current_speed_rpm[motor_id];
}

void motor_pid_reset(int motor_id) {
  if (motor_id < 0 || motor_id >= NUM_MOTORS) return;
  pid[motor_id].integral = 0;
  pid[motor_id].prev_error = 0;
  pid[motor_id].output = 0;
  encoder_counts[motor_id] = 0;
  last_counts[motor_id] = 0;
  target_rpm[motor_id] = 0;
  target_ticks[motor_id] = 0;
  position_mode[motor_id] = false;
}
"""

    return {
        "dc_motor_driver.h": header,
        "dc_motor_driver.cpp": impl,
    }


# ---------------------------------------------------------------------------
# Odometry (differential drive)
# ---------------------------------------------------------------------------


def gen_odometry_code(
    wheel_radius_mm: float = 32.5,
    wheel_base_mm: float = 150.0,
    encoder_ppr: int = 7,
    gear_ratio: float = 48.0,
    quadrature: bool = True,
) -> str:
    """Generate differential drive odometry code (C/C++ for Arduino/ESP32).

    Args:
        wheel_radius_mm: Wheel radius in mm.
        wheel_base_mm: Distance between left and right wheel centers (mm).
        encoder_ppr: Encoder pulses per revolution (before quadrature).
        gear_ratio: Motor gear ratio (output_rev / motor_rev).
        quadrature: True if encoder has A+B channels (4x decoding).

    Returns:
        C++ header+implementation as a single string.
    """
    cpr = encoder_ppr * (4 if quadrature else 1) * gear_ratio
    mm_per_tick = 2.0 * math.pi * wheel_radius_mm / cpr if cpr > 0 else 1.0

    return f"""\
// Differential Drive Odometry - Auto-generated
// Wheel radius: {wheel_radius_mm} mm
// Wheel base: {wheel_base_mm} mm
// CPR (counts per output revolution): {cpr:.0f}
// MM per tick: {mm_per_tick:.4f}

#ifndef ODOMETRY_H
#define ODOMETRY_H

#include <Arduino.h>
#include <math.h>

static const float WHEEL_RADIUS_MM = {wheel_radius_mm:.2f}f;
static const float WHEEL_BASE_MM = {wheel_base_mm:.2f}f;
static const long CPR = {cpr:.0f}L;
static const float MM_PER_TICK = {mm_per_tick:.6f}f;

struct Pose {{
  float x;     // mm
  float y;     // mm
  float theta; // radians
}};

// Update odometry from encoder readings
// left_ticks, right_ticks: current encoder counts
// Returns updated pose
Pose odometry_update(long left_ticks, long right_ticks);

// Get current pose
Pose odometry_get_pose();

// Reset odometry to origin
void odometry_reset();

// Convert velocity commands to wheel speeds
// v_linear: mm/s, v_angular: rad/s
// Returns (left_rpm, right_rpm)
void odometry_velocity_to_wheels(float v_linear, float v_angular,
                                  float *left_rpm, float *right_rpm);

#endif

// === Implementation ===

static Pose _pose = {{0, 0, 0}};
static long _last_left = 0;
static long _last_right = 0;

Pose odometry_update(long left_ticks, long right_ticks) {{
  long dl = left_ticks - _last_left;
  long dr = right_ticks - _last_right;
  _last_left = left_ticks;
  _last_right = right_ticks;

  float dist_l = (float)dl * MM_PER_TICK;
  float dist_r = (float)dr * MM_PER_TICK;

  float dist = (dist_r + dist_l) / 2.0f;
  float dtheta = (dist_r - dist_l) / WHEEL_BASE_MM;

  // Update pose
  if (fabsf(dtheta) < 0.0001f) {{
    // Straight line
    _pose.x += dist * cosf(_pose.theta);
    _pose.y += dist * sinf(_pose.theta);
  }} else {{
    // Arc
    float r = dist / dtheta;
    _pose.x += r * (sinf(_pose.theta + dtheta) - sinf(_pose.theta));
    _pose.y += r * (cosf(_pose.theta) - cosf(_pose.theta + dtheta));
    _pose.theta += dtheta;
  }}

  // Normalize theta to [-PI, PI]
  while (_pose.theta > M_PI) _pose.theta -= 2.0f * M_PI;
  while (_pose.theta < -M_PI) _pose.theta += 2.0f * M_PI;

  return _pose;
}}

Pose odometry_get_pose() {{
  return _pose;
}}

void odometry_reset() {{
  _pose.x = 0;
  _pose.y = 0;
  _pose.theta = 0;
  _last_left = 0;
  _last_right = 0;
}}

void odometry_velocity_to_wheels(float v_linear, float v_angular,
                                  float *left_rpm, float *right_rpm) {{
  // v_linear: mm/s, v_angular: rad/s
  // Differential drive: v_l = v - omega * L/2, v_r = v + omega * L/2
  float v_left = v_linear - v_angular * WHEEL_BASE_MM / 2.0f;
  float v_right = v_linear + v_angular * WHEEL_BASE_MM / 2.0f;

  // Convert mm/s to RPM at wheel shaft
  // RPM = (v_mm/s) / (2*PI*r_mm) * 60
  float circumference = 2.0f * M_PI * WHEEL_RADIUS_MM;
  *left_rpm = (v_left / circumference) * 60.0f;
  *right_rpm = (v_right / circumference) * 60.0f;
}}
"""


# ---------------------------------------------------------------------------
# Tools: gen_motor_driver / gen_odometry
# ---------------------------------------------------------------------------


class GenMotorDriverTool(Tool):
    name = "gen_motor_driver"
    description = (
        "生成直流电机驱动代码（PID 速度/位置控制 + 编码器反馈）。"
        "支持 Arduino/ESP32，自动匹配电机 PID 参数。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "motors": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "电机配置列表 [{motor_id, encoder_id, pwm_pin, dir_pin1, dir_pin2, enc_a_pin, enc_b_pin}]",
                },
                "controller": {
                    "type": "string", "enum": ["esp32", "arduino"],
                    "description": "控制器类型（默认 esp32）",
                },
                "output_dir": {"type": "string", "description": "输出目录路径"},
            }, "required": ["motors"]},
        )

    def execute(self, *, motors: list[dict] | None = None,
                controller: str = "esp32", output_dir: str = "",
                **kwargs: Any) -> str:
        if not motors:
            return "错误：未指定电机配置"

        files = gen_motor_driver_code(motors, controller)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for fname, content in files.items():
                (out / fname).write_text(content, encoding="utf-8")

        lines = [
            f"[Motor Driver Generated]",
            f"Controller: {controller}",
            f"Motors: {len(motors)}",
            f"Files: {len(files)}",
            "",
            "--- Motor Config ---",
        ]
        for i, m in enumerate(motors):
            lines.append(f"  Motor {i}: {m.get('motor_id', '?')} + {m.get('encoder_id', '?')}")
            pid = get_motor_pid_spec(m.get("motor_id", ""))
            if pid:
                lines.append(f"    PID: Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}")

        if output_dir:
            lines.append(f"\nSaved to: {output_dir}")

        lines.append("\n--- Preview: dc_motor_driver.h ---")
        lines.append(files["dc_motor_driver.h"][:600])

        return "\n".join(lines)


class GenOdometryTool(Tool):
    name = "gen_odometry"
    description = (
        "生成差速底盘里程计代码（位置推算+速度分解）。"
        "输出 C++ 头文件，含 Pose 更新和轮速分解函数。"
    )

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name, description=self.description,
            parameters={"type": "object", "properties": {
                "wheel_radius_mm": {
                    "type": "number",
                    "description": "轮子半径 mm（默认 32.5）",
                },
                "wheel_base_mm": {
                    "type": "number",
                    "description": "轮距 mm（默认 150）",
                },
                "encoder_ppr": {
                    "type": "integer",
                    "description": "编码器 PPR（默认 7）",
                },
                "gear_ratio": {
                    "type": "number",
                    "description": "减速比（默认 48）",
                },
                "quadrature": {
                    "type": "boolean",
                    "description": "是否正交编码（默认 true）",
                },
                "output_dir": {
                    "type": "string",
                    "description": "输出目录路径",
                },
            }, "required": []},
        )

    def execute(self, *, wheel_radius_mm: float = 32.5,
                wheel_base_mm: float = 150.0,
                encoder_ppr: int = 7,
                gear_ratio: float = 48.0,
                quadrature: bool = True,
                output_dir: str = "",
                **kwargs: Any) -> str:
        code = gen_odometry_code(
            wheel_radius_mm=wheel_radius_mm,
            wheel_base_mm=wheel_base_mm,
            encoder_ppr=encoder_ppr,
            gear_ratio=gear_ratio,
            quadrature=quadrature,
        )

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "odometry.h").write_text(code, encoding="utf-8")

        cpr = encoder_ppr * (4 if quadrature else 1) * gear_ratio
        mm_per_tick = 2.0 * math.pi * wheel_radius_mm / cpr if cpr > 0 else 0

        lines = [
            "[Odometry Code Generated]",
            f"Wheel radius: {wheel_radius_mm} mm",
            f"Wheel base: {wheel_base_mm} mm",
            f"CPR: {cpr:.0f}",
            f"MM/tick: {mm_per_tick:.4f}",
            "",
            "--- Preview ---",
            code[:800],
        ]

        if output_dir:
            lines.append(f"\nSaved to: {output_dir}/odometry.h")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_code_gen_tools(registry: Any) -> None:
    """Register code generation tools."""
    registry.register(GenFirmwareTool())
    registry.register(GenWiringDiagramTool())
    registry.register(GenTestSequenceTool())
    registry.register(GenMotorDriverTool())
    registry.register(GenOdometryTool())
