"""Actuator (servo, DC motor, stepper) knowledge base.

Provides a database of common hobbyist and industrial actuators with
torque, speed, voltage, weight, and price specifications.

Used by actuator_tools.py for selection, analysis, and power budgeting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Actuator:
    """Specification sheet for a single actuator."""

    id: str                    # Unique identifier (e.g. "SG90")
    name: str                  # Human-readable name
    category: str              # "servo", "dc_motor", "stepper", "bldc"
    torque_kgcm: float         # Stall / holding torque (kg·cm)
    speed_s_per_60deg: float   # Seconds per 60° at no-load (servos)
    rpm: float                 # No-load speed (motors) in RPM
    voltage: float             # Nominal voltage (V)
    voltage_range: tuple[float, float] = (0, 0)  # (min, max) operating voltage
    current_idle_ma: float = 0     # Idle / no-load current (mA)
    current_stall_ma: float = 0    # Stall current (mA)
    weight_g: float = 0            # Weight (grams)
    price_cny: float = 0           # Approximate price (CNY)
    rotation_range: tuple[float, float] = (0, 180)  # Usable angle range (servos)
    interface: str = "pwm"         # "pwm", "i2c", "uart", "step_dir"
    description: str = ""
    dimensions_mm: dict[str, float] = field(default_factory=dict)
    shaft_diameter_mm: float = 0


# ============================================================================
# Servo Database
# ============================================================================

ACTUATORS: dict[str, Actuator] = {
    # ---- Micro / Mini servos ----
    "SG90": Actuator(
        id="SG90", name="Tower Pro SG90", category="servo",
        torque_kgcm=1.8, speed_s_per_60deg=0.1, rpm=0,
        voltage=5.0, voltage_range=(4.8, 6.0),
        current_idle_ma=5, current_stall_ma=750,
        weight_g=9, price_cny=5,
        rotation_range=(0, 180), interface="pwm",
        description="微型舵机，适用于轻负载关节和小型机器人",
        # Matches parts_catalog servo_sg90 (body_height=31.0 incl. tabs).
        dimensions_mm={"length": 22.2, "width": 11.8, "height": 31.0},
        shaft_diameter_mm=4.6,
    ),
    "MG90S": Actuator(
        id="MG90S", name="Tower Pro MG90S", category="servo",
        torque_kgcm=2.2, speed_s_per_60deg=0.1, rpm=0,
        voltage=5.0, voltage_range=(4.8, 6.0),
        current_idle_ma=5, current_stall_ma=800,
        weight_g=14, price_cny=12,
        rotation_range=(0, 180), interface="pwm",
        description="金属齿轮微型舵机，比 SG90 更耐用",
        dimensions_mm={"length": 22.7, "width": 12.2, "height": 23.0},
        shaft_diameter_mm=4.8,
    ),
    "ES08A": Actuator(
        id="ES08A", name="E-SKY ES08A", category="servo",
        torque_kgcm=1.6, speed_s_per_60deg=0.12, rpm=0,
        voltage=5.0, voltage_range=(4.8, 6.0),
        current_idle_ma=5, current_stall_ma=600,
        weight_g=8, price_cny=6,
        rotation_range=(0, 180), interface="pwm",
        description="超微型舵机，空间受限场景",
        dimensions_mm={"length": 20.0, "width": 8.5, "height": 20.0},
        shaft_diameter_mm=4.0,
    ),

    # ---- Standard servos ----
    "MG996R": Actuator(
        id="MG996R", name="Tower Pro MG996R", category="servo",
        torque_kgcm=11.0, speed_s_per_60deg=0.17, rpm=0,
        voltage=6.0, voltage_range=(4.8, 7.2),
        current_idle_ma=10, current_stall_ma=2500,
        weight_g=55, price_cny=25,
        rotation_range=(0, 180), interface="pwm",
        description="标准舵机，桌面机械臂最常用",
        dimensions_mm={"length": 40.7, "width": 19.7, "height": 42.9},
        shaft_diameter_mm=5.8,
    ),
    "DS3218": Actuator(
        id="DS3218", name="DS Power DS3218", category="servo",
        torque_kgcm=20.0, speed_s_per_60deg=0.2, rpm=0,
        voltage=6.8, voltage_range=(6.0, 7.4),
        current_idle_ma=10, current_stall_ma=3000,
        weight_g=60, price_cny=35,
        rotation_range=(0, 180), interface="pwm",
        description="20kg 大扭矩数字舵机，适合中负载机械臂",
        # Matches parts_catalog servo_ds3218 (body_height=38.5, fixed=True).
        dimensions_mm={"length": 40.0, "width": 20.0, "height": 38.5},
        shaft_diameter_mm=5.8,
    ),
    "DS3225": Actuator(
        id="DS3225", name="DS Power DS3225", category="servo",
        torque_kgcm=25.0, speed_s_per_60deg=0.2, rpm=0,
        voltage=6.8, voltage_range=(6.0, 7.4),
        current_idle_ma=12, current_stall_ma=3200,
        weight_g=62, price_cny=45,
        rotation_range=(0, 270), interface="pwm",
        description="25kg 超大扭矩，270° 范围",
        dimensions_mm={"length": 40.0, "width": 20.0, "height": 40.5},
        shaft_diameter_mm=6.0,
    ),

    # ---- Continuous rotation servos ----
    "FS90R": Actuator(
        id="FS90R", name="Feetech FS90R", category="servo",
        torque_kgcm=1.3, speed_s_per_60deg=0.08, rpm=70,
        voltage=5.0, voltage_range=(4.0, 6.0),
        current_idle_ma=5, current_stall_ma=600,
        weight_g=9, price_cny=15,
        rotation_range=(0, 0), interface="pwm",
        description="连续旋转舵机，适合轮式机器人驱动",
        dimensions_mm={"length": 23.0, "width": 12.5, "height": 22.0},
        shaft_diameter_mm=4.8,
    ),

    # ---- DC Motors ----
    "TT_MOTOR": Actuator(
        id="TT_MOTOR", name="TT 减速电机 (1:48)", category="dc_motor",
        torque_kgcm=0.8, speed_s_per_60deg=0, rpm=200,
        voltage=3.0, voltage_range=(1.5, 4.5),
        current_idle_ma=70, current_stall_ma=1200,
        weight_g=18, price_cny=5,
        rotation_range=(0, 0), interface="pwm",
        description="TT 黄色减速电机，低成本轮式机器人首选",
        dimensions_mm={"length": 36.0, "width": 26.0, "height": 20.0},
        shaft_diameter_mm=3.175,
    ),
    "JGB37_520": Actuator(
        id="JGB37_520", name="JGB37-520 减速电机 (1:30)", category="dc_motor",
        torque_kgcm=5.0, speed_s_per_60deg=0, rpm=350,
        voltage=12.0, voltage_range=(6.0, 18.0),
        current_idle_ma=100, current_stall_ma=3000,
        weight_g=120, price_cny=35,
        rotation_range=(0, 0), interface="pwm",
        description="12V 减速电机，AGV/机器人底盘常用",
        dimensions_mm={"length": 72.0, "width": 37.0, "height": 37.0},
        shaft_diameter_mm=6.0,
    ),
    "GA25_370_6V": Actuator(
        id="GA25_370_6V", name="GA25-370 减速电机 6V (1:30)", category="dc_motor",
        torque_kgcm=3.0, speed_s_per_60deg=0, rpm=300,
        voltage=6.0, voltage_range=(3.0, 9.0),
        current_idle_ma=80, current_stall_ma=2000,
        weight_g=60, price_cny=20,
        rotation_range=(0, 0), interface="pwm",
        description="6V 减速电机 (注意：mobile_base.py 中的 GA25-370 为 12V 变体)，带霍尔编码器，桌面机器人底盘",
        dimensions_mm={"length": 54.0, "width": 25.0, "height": 25.0},
        shaft_diameter_mm=4.0,
    ),

    # ---- Stepper Motors ----
    "28BYJ48": Actuator(
        id="28BYJ48", name="28BYJ-48 步进电机", category="stepper",
        torque_kgcm=0.35, speed_s_per_60deg=0, rpm=15,
        voltage=5.0, voltage_range=(5.0, 7.0),
        current_idle_ma=40, current_stall_ma=300,
        weight_g=30, price_cny=5,
        rotation_range=(0, 0), interface="step_dir",
        description="5V 步进电机，精确角度控制，成本低",
        dimensions_mm={"diameter": 28.0, "height": 19.0},
        shaft_diameter_mm=5.0,
    ),
    "NEMA17": Actuator(
        id="NEMA17", name="NEMA 17 步进电机", category="stepper",
        torque_kgcm=4.4, speed_s_per_60deg=0, rpm=200,
        voltage=12.0, voltage_range=(12.0, 24.0),
        current_idle_ma=200, current_stall_ma=1500,
        weight_g=280, price_cny=45,
        rotation_range=(0, 0), interface="step_dir",
        description="42mm 步进电机，3D 打印机/CNC 常用",
        dimensions_mm={"length": 42.3, "width": 42.3, "height": 40.0},
        shaft_diameter_mm=5.0,
    ),
    "NEMA23": Actuator(
        id="NEMA23", name="NEMA 23 步进电机", category="stepper",
        torque_kgcm=12.0, speed_s_per_60deg=0, rpm=300,
        voltage=24.0, voltage_range=(24.0, 48.0),
        current_idle_ma=300, current_stall_ma=3000,
        weight_g=700, price_cny=120,
        rotation_range=(0, 0), interface="step_dir",
        description="57mm 步进电机，CNC/工业级应用",
        dimensions_mm={"length": 56.4, "width": 56.4, "height": 56.0},
        shaft_diameter_mm=6.35,
    ),

    # ---- BLDC Motors ----
    "2204_2300KV": Actuator(
        id="2204_2300KV", name="2204 2300KV 无刷电机", category="bldc",
        torque_kgcm=0.15, speed_s_per_60deg=0, rpm=28000,
        voltage=7.4, voltage_range=(7.0, 12.0),
        current_idle_ma=200, current_stall_ma=8000,
        weight_g=18, price_cny=30,
        rotation_range=(0, 0), interface="pwm",
        description="无人机/竞速常用无刷电机",
        dimensions_mm={"diameter": 22.0, "height": 14.0},
        shaft_diameter_mm=3.0,
    ),
}


def get_actuator(actuator_id: str) -> Actuator | None:
    """Look up an actuator by ID."""
    return ACTUATORS.get(actuator_id.upper())


def list_actuators(category: str = "") -> list[Actuator]:
    """List actuators, optionally filtered by category."""
    cats = [category] if category else []
    return [
        a for a in ACTUATORS.values()
        if not cats or a.category in cats
    ]


def torque_to_nm(torque_kgcm: float) -> float:
    """Convert kg·cm to N·m."""
    return torque_kgcm * 9.80665 / 100.0


def nm_to_torque(nm: float) -> float:
    """Convert N·m to kg·cm."""
    return nm * 100.0 / 9.80665


# ============================================================================
# DC Motor PID Tuning Defaults
# ============================================================================

@dataclass
class DCMotorPIDSpec:
    """Recommended PID tuning parameters for a DC motor with encoder."""
    motor_id: str
    encoder_ppr: int          # Encoder pulses per revolution
    gear_ratio: float         # Gear reduction ratio (output_rev / motor_rev)
    kp: float                 # Proportional gain
    ki: float                 # Integral gain
    kd: float                 # Derivative gain
    max_pwm: int              # Max PWM duty cycle (0-255 for 8-bit)
    sample_period_ms: int     # PID loop period in ms
    description: str = ""


DC_MOTOR_PID_SPECS: dict[str, DCMotorPIDSpec] = {
    "TT_MOTOR": DCMotorPIDSpec(
        motor_id="TT_MOTOR", encoder_ppr=7, gear_ratio=48.0,
        kp=1.2, ki=0.3, kd=0.05,
        max_pwm=200, sample_period_ms=20,
        description="TT 减速电机 1:48，低速高扭矩，适合差速底盘",
    ),
    "GA25_370_6V": DCMotorPIDSpec(
        motor_id="GA25_370_6V", encoder_ppr=11, gear_ratio=30.0,
        kp=1.5, ki=0.4, kd=0.08,
        max_pwm=220, sample_period_ms=20,
        description="GA25-370 6V 减速电机 1:30，中等精度差速底盘",
    ),
    "JGB37_520": DCMotorPIDSpec(
        motor_id="JGB37_520", encoder_ppr=11, gear_ratio=30.0,
        kp=2.0, ki=0.5, kd=0.1,
        max_pwm=240, sample_period_ms=10,
        description="JGB37-520 12V 减速电机，AGV 底盘常用",
    ),
    "NEMA17_DRV": DCMotorPIDSpec(
        motor_id="NEMA17", encoder_ppr=400, gear_ratio=1.0,
        kp=3.0, ki=0.8, kd=0.15,
        max_pwm=255, sample_period_ms=5,
        description="NEMA17 步进（驱动器模式），高精度定位",
    ),
}


def get_motor_pid_spec(motor_id: str) -> DCMotorPIDSpec | None:
    """Look up PID tuning spec for a motor."""
    return DC_MOTOR_PID_SPECS.get(motor_id.upper())
