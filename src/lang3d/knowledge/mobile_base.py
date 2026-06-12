"""Mobile base design knowledge — kinematics, motor sizing, battery estimation.

Supports differential drive and omnidirectional (Mecanum) drive kinematics,
motor torque calculation, battery capacity sizing, and wheelbase parameter
derivation from high-level requirements (payload, speed, gradeability, runtime).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# DC Motor Catalog
# ============================================================================

@dataclass
class DCMotorSpec:
    """Specifications for a DC motor."""
    name: str
    nominal_voltage: float  # V
    no_load_speed_rpm: float
    stall_torque_kg_cm: float
    rated_torque_kg_cm: float
    rated_speed_rpm: float
    weight_g: float
    shaft_diameter_mm: float = 6.0
    price_cny: float = 0.0
    notes: str = ""


DC_MOTOR_CATALOG: dict[str, DCMotorSpec] = {
    "TT_Motor_130": DCMotorSpec(
        name="TT马达 130", nominal_voltage=6, no_load_speed_rpm=12000,
        stall_torque_kg_cm=0.5, rated_torque_kg_cm=0.1, rated_speed_rpm=9000,
        weight_g=15, shaft_diameter_mm=3.0, price_cny=5,
        notes="玩具级，适合轻载小车",
    ),
    "JGA25-370": DCMotorSpec(
        name="JGA25-370 减速电机", nominal_voltage=12, no_load_speed_rpm=200,
        stall_torque_kg_cm=8.0, rated_torque_kg_cm=1.5, rated_speed_rpm=170,
        weight_g=120, shaft_diameter_mm=6.0, price_cny=25,
        notes="常用机器人底盘电机，金属齿轮",
    ),
    "GA25-370": DCMotorSpec(
        name="GA25-370 减速电机", nominal_voltage=12, no_load_speed_rpm=100,
        stall_torque_kg_cm=12.0, rated_torque_kg_cm=2.0, rated_speed_rpm=85,
        weight_g=130, shaft_diameter_mm=6.0, price_cny=30,
        notes="高扭矩版，适合重载底盘",
    ),
    "NEMA17": DCMotorSpec(
        name="NEMA17 步进电机", nominal_voltage=12, no_load_speed_rpm=600,
        stall_torque_kg_cm=4.4, rated_torque_kg_cm=3.0, rated_speed_rpm=400,
        weight_g=280, shaft_diameter_mm=5.0, price_cny=35,
        notes="42步进，定位精度高",
    ),
    "NEMA23": DCMotorSpec(
        name="NEMA23 步进电机", nominal_voltage=24, no_load_speed_rpm=600,
        stall_torque_kg_cm=12.0, rated_torque_kg_cm=8.0, rated_speed_rpm=400,
        weight_g=700, shaft_diameter_mm=8.0, price_cny=120,
        notes="大功率步进，适合工业AGV",
    ),
}


# ============================================================================
# Differential Drive Kinematics
# ============================================================================

@dataclass
class DifferentialDriveKinematics:
    """Differential drive mobile base kinematics.

    Two driven wheels separated by track_width, with optional caster wheels.
    """
    wheel_radius_mm: float = 50.0  # mm
    track_width_mm: float = 300.0  # mm (distance between wheel centers)
    wheelbase_mm: float = 300.0  # mm (front-to-back distance)
    max_rpm: float = 200.0  # max motor RPM

    @property
    def wheel_circumference_mm(self) -> float:
        return 2 * math.pi * self.wheel_radius_mm

    def forward_kinematics(
        self, v_left_mm_s: float, v_right_mm_s: float
    ) -> tuple[float, float]:
        """Compute linear velocity (mm/s) and angular velocity (rad/s).

        Returns (v_linear, omega).
        """
        v = (v_left_mm_s + v_right_mm_s) / 2.0
        omega = (v_right_mm_s - v_left_mm_s) / self.track_width_mm
        return v, omega

    def inverse_kinematics(
        self, v_linear_mm_s: float, omega_rad_s: float
    ) -> tuple[float, float]:
        """Compute wheel velocities for desired linear and angular velocity.

        Returns (v_left_mm_s, v_right_mm_s).
        """
        v_left = v_linear_mm_s - omega_rad_s * self.track_width_mm / 2.0
        v_right = v_linear_mm_s + omega_rad_s * self.track_width_mm / 2.0
        return v_left, v_right

    def turning_radius_mm(self, v_linear_mm_s: float, omega_rad_s: float) -> float:
        """Compute instantaneous turning radius (mm). Inf for straight line."""
        if abs(omega_rad_s) < 1e-6:
            return float("inf")
        return abs(v_linear_mm_s / omega_rad_s)

    def max_linear_speed_mm_s(self) -> float:
        """Max linear speed from max motor RPM."""
        return self.max_rpm / 60.0 * self.wheel_circumference_mm

    def wheel_speed_from_omega(self, omega_rad_s: float) -> float:
        """Wheel speed difference needed for angular velocity (mm/s)."""
        return omega_rad_s * self.track_width_mm / 2.0

    def rpm_from_speed(self, v_mm_s: float) -> float:
        """Motor RPM needed for a given linear speed."""
        return v_mm_s / self.wheel_circumference_mm * 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "wheel_radius_mm": self.wheel_radius_mm,
            "track_width_mm": self.track_width_mm,
            "wheelbase_mm": self.wheelbase_mm,
            "max_rpm": self.max_rpm,
            "wheel_circumference_mm": round(self.wheel_circumference_mm, 2),
            "max_linear_speed_mm_s": round(self.max_linear_speed_mm_s(), 2),
        }


# ============================================================================
# Omnidirectional (Mecanum) Kinematics
# ============================================================================

@dataclass
class OmnidirectionalKinematics:
    """Mecanum wheel 4-wheel omnidirectional drive kinematics.

    Wheel order: [FL, FR, RL, RR] (front-left, front-right, rear-left, rear-right).
    """
    wheel_radius_mm: float = 50.0
    track_width_mm: float = 300.0  # lateral distance between wheel centers
    wheelbase_mm: float = 300.0  # longitudinal distance between wheel centers

    def inverse_kinematics(
        self, vx_mm_s: float, vy_mm_s: float, omega_rad_s: float
    ) -> list[float]:
        """Compute 4 wheel speeds (mm/s) for desired (vx, vy, omega).

        Returns [fl, fr, rl, rr] wheel linear speeds in mm/s.
        """
        L = self.wheelbase_mm
        W = self.track_width_mm
        r = self.wheel_radius_mm
        k = (L + W) / 2.0

        fl = vx_mm_s - vy_mm_s - k * omega_rad_s
        fr = vx_mm_s + vy_mm_s + k * omega_rad_s
        rl = vx_mm_s + vy_mm_s - k * omega_rad_s
        rr = vx_mm_s - vy_mm_s + k * omega_rad_s

        return [fl, fr, rl, rr]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "omnidirectional",
            "wheel_radius_mm": self.wheel_radius_mm,
            "track_width_mm": self.track_width_mm,
            "wheelbase_mm": self.wheelbase_mm,
        }

    def forward_kinematics(
        self, wheels_mm_s: list[float]
    ) -> tuple[float, float, float]:
        """Compute (vx, vy, omega) from 4 wheel speeds [fl, fr, rl, rr]."""
        L = self.wheelbase_mm
        W = self.track_width_mm
        k = (L + W) / 2.0

        fl, fr, rl, rr = wheels_mm_s
        vx = (fl + fr + rl + rr) / 4.0
        vy = (-fl + fr + rl - rr) / 4.0
        omega = (-fl + fr - rl + rr) / (4.0 * k)

        return vx, vy, omega


# ============================================================================
# Motor Torque Calculator
# ============================================================================

@dataclass
class MotorTorqueCalculator:
    """Calculate required motor torque from robot parameters.

    Torque = T_rolling + T_incline + T_acceleration
    """
    total_mass_kg: float = 10.0  # robot total mass
    wheel_radius_mm: float = 50.0
    wheel_count: int = 4
    rolling_resistance: float = 0.02  # rubber on concrete
    max_grade_deg: float = 10.0  # maximum climbable slope
    max_acceleration_mm_s2: float = 500.0  # target acceleration

    @property
    def wheel_radius_m(self) -> float:
        return self.wheel_radius_mm / 1000.0

    def rolling_torque_nm(self) -> float:
        """Torque to overcome rolling resistance (Nm per wheel)."""
        gravity = 9.81
        total_friction = self.total_mass_kg * gravity * self.rolling_resistance
        per_wheel = total_friction / self.wheel_count
        return per_wheel * self.wheel_radius_m

    def incline_torque_nm(self) -> float:
        """Torque to climb max grade (Nm per wheel)."""
        gravity = 9.81
        sin_angle = math.sin(math.radians(self.max_grade_deg))
        total_incline = self.total_mass_kg * gravity * sin_angle
        per_wheel = total_incline / self.wheel_count
        return per_wheel * self.wheel_radius_m

    def accel_torque_nm(self) -> float:
        """Torque for target acceleration (Nm per wheel)."""
        a = self.max_acceleration_mm_s2 / 1000.0  # m/s²
        total_force = self.total_mass_kg * a
        per_wheel = total_force / self.wheel_count
        return per_wheel * self.wheel_radius_m

    def total_torque_nm_per_wheel(self) -> float:
        """Total required torque per wheel (Nm)."""
        return self.rolling_torque_nm() + self.incline_torque_nm() + self.accel_torque_nm()

    def total_torque_kg_cm_per_wheel(self) -> float:
        """Total required torque per wheel (kg·cm)."""
        return self.total_torque_nm_per_wheel() / 9.81 * 100

    def select_motor(self) -> DCMotorSpec | None:
        """Select smallest motor from catalog that meets torque requirement."""
        required = self.total_torque_kg_cm_per_wheel()
        candidates = sorted(DC_MOTOR_CATALOG.values(), key=lambda m: m.weight_g)
        for motor in candidates:
            if motor.rated_torque_kg_cm >= required:
                return motor
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_mass_kg": self.total_mass_kg,
            "wheel_count": self.wheel_count,
            "rolling_torque_nm": round(self.rolling_torque_nm(), 4),
            "incline_torque_nm": round(self.incline_torque_nm(), 4),
            "accel_torque_nm": round(self.accel_torque_nm(), 4),
            "total_torque_nm_per_wheel": round(self.total_torque_nm_per_wheel(), 4),
            "total_torque_kg_cm_per_wheel": round(self.total_torque_kg_cm_per_wheel(), 2),
        }


# ============================================================================
# Battery Sizing Calculator
# ============================================================================

@dataclass
class BatterySizingCalculator:
    """Estimate battery capacity from power requirements."""
    motor_count: int = 4
    motor_power_w: float = 10.0  # W per motor (rated)
    electronics_power_w: float = 15.0  # IPC + sensors
    runtime_hours: float = 2.0
    voltage: float = 12.0  # system voltage
    safety_factor: float = 1.3  # derating

    def total_power_w(self) -> float:
        return self.motor_count * self.motor_power_w + self.electronics_power_w

    def capacity_ah(self) -> float:
        """Required battery capacity in Ah."""
        return self.total_power_w() * self.runtime_hours * self.safety_factor / self.voltage

    def energy_wh(self) -> float:
        return self.total_power_w() * self.runtime_hours * self.safety_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_power_w": round(self.total_power_w(), 1),
            "capacity_ah": round(self.capacity_ah(), 2),
            "energy_wh": round(self.energy_wh(), 1),
            "voltage": self.voltage,
            "runtime_hours": self.runtime_hours,
            "safety_factor": self.safety_factor,
        }


# ============================================================================
# Wheelbase Calculator
# ============================================================================

@dataclass
class WheelBaseCalculator:
    """Derive wheel diameter, track width, ground clearance from payload."""
    payload_kg: float = 5.0
    chassis_mass_kg: float = 3.0  # estimated chassis mass
    wheel_count: int = 4
    drive_type: str = "differential"  # differential / mecanum

    @property
    def total_mass_kg(self) -> float:
        return self.payload_kg + self.chassis_mass_kg

    def wheel_diameter_mm(self) -> float:
        """Estimate wheel diameter from total mass."""
        # Heuristic: heavier → larger wheels
        if self.total_mass_kg < 5:
            return 60.0
        elif self.total_mass_kg < 15:
            return 80.0
        elif self.total_mass_kg < 30:
            return 100.0
        elif self.total_mass_kg < 60:
            return 120.0
        else:
            return 150.0

    def track_width_mm(self) -> float:
        """Estimate track width based on mass and wheel count."""
        base = 200.0
        if self.total_mass_kg > 20:
            base = 300.0
        if self.total_mass_kg > 50:
            base = 400.0
        if self.drive_type == "mecanum":
            base *= 1.1  # Mecanum needs slightly wider stance
        return base

    def ground_clearance_mm(self) -> float:
        """Minimum ground clearance."""
        if self.total_mass_kg < 10:
            return 10.0
        elif self.total_mass_kg < 30:
            return 15.0
        else:
            return 20.0

    def wheelbase_mm(self) -> float:
        """Estimate wheelbase (front-to-back)."""
        return self.track_width_mm() * 1.0  # roughly square

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_mass_kg": self.total_mass_kg,
            "wheel_diameter_mm": self.wheel_diameter_mm(),
            "track_width_mm": self.track_width_mm(),
            "wheelbase_mm": self.wheelbase_mm(),
            "ground_clearance_mm": self.ground_clearance_mm(),
            "wheel_count": self.wheel_count,
            "drive_type": self.drive_type,
        }


# ============================================================================
# Mobile Base Templates
# ============================================================================

@dataclass
class MobileBaseTemplate:
    """A pre-defined mobile base configuration."""
    name: str
    description: str
    drive_type: str  # differential_4w, differential_2w, mecanum, omnidirectional, ackermann
    parts: list[dict[str, Any]]
    joints: list[dict[str, Any]]
    parameters: dict[str, Any]  # parameter ranges


MOBILE_BASE_TEMPLATES: dict[str, MobileBaseTemplate] = {
    "differential_4w": MobileBaseTemplate(
        name="差速4轮底盘",
        description="4轮差速驱动底盘，2个驱动轮+2个万向轮或4轮全驱",
        drive_type="differential_4w",
        parts=[
            {"name": "chassis", "category": "structural", "description": "底盘框架",
             "material": "PLA", "dimensions": {"length": 300, "width": 200, "height": 5}},
            {"name": "wheel_fl", "category": "wheel", "description": "左前轮",
             "material": "TPU", "dimensions": {"diameter": 80, "width": 25}},
            {"name": "wheel_fr", "category": "wheel", "description": "右前轮",
             "material": "TPU", "dimensions": {"diameter": 80, "width": 25}},
            {"name": "wheel_rl", "category": "wheel", "description": "左后轮",
             "material": "TPU", "dimensions": {"diameter": 80, "width": 25}},
            {"name": "wheel_rr", "category": "wheel", "description": "右后轮",
             "material": "TPU", "dimensions": {"diameter": 80, "width": 25}},
            {"name": "motor_mount_fl", "category": "structural", "description": "左前电机支架",
             "material": "PLA", "dimensions": {"length": 50, "width": 30, "height": 25}},
            {"name": "motor_mount_fr", "category": "structural", "description": "右前电机支架",
             "material": "PLA", "dimensions": {"length": 50, "width": 30, "height": 25}},
            {"name": "motor_mount_rl", "category": "structural", "description": "左后电机支架",
             "material": "PLA", "dimensions": {"length": 50, "width": 30, "height": 25}},
            {"name": "motor_mount_rr", "category": "structural", "description": "右后电机支架",
             "material": "PLA", "dimensions": {"length": 50, "width": 30, "height": 25}},
        ],
        joints=[
            {"type": "fixed", "parent": "chassis", "child": "motor_mount_fl",
             "parent_anchor": "front", "child_anchor": "bottom"},
            {"type": "fixed", "parent": "chassis", "child": "motor_mount_fr",
             "parent_anchor": "front", "child_anchor": "bottom"},
            {"type": "fixed", "parent": "chassis", "child": "motor_mount_rl",
             "parent_anchor": "back", "child_anchor": "bottom"},
            {"type": "fixed", "parent": "chassis", "child": "motor_mount_rr",
             "parent_anchor": "back", "child_anchor": "bottom"},
            {"type": "revolute", "parent": "motor_mount_fl", "child": "wheel_fl",
             "axis": "y"},
            {"type": "revolute", "parent": "motor_mount_fr", "child": "wheel_fr",
             "axis": "y"},
            {"type": "revolute", "parent": "motor_mount_rl", "child": "wheel_rl",
             "axis": "y"},
            {"type": "revolute", "parent": "motor_mount_rr", "child": "wheel_rr",
             "axis": "y"},
        ],
        parameters={
            "wheel_diameter_mm": [60, 150],
            "track_width_mm": [200, 500],
            "wheelbase_mm": [200, 500],
            "max_speed_mm_s": [200, 1000],
            "payload_kg": [2, 50],
        },
    ),
    "differential_2w": MobileBaseTemplate(
        name="差速2轮底盘",
        description="2轮差速驱动+1-2个万向轮，适合轻量级机器人",
        drive_type="differential_2w",
        parts=[
            {"name": "chassis", "category": "structural", "description": "底盘框架",
             "material": "PLA", "dimensions": {"length": 200, "width": 150, "height": 5}},
            {"name": "wheel_l", "category": "wheel", "description": "左驱动轮",
             "material": "TPU", "dimensions": {"diameter": 65, "width": 20}},
            {"name": "wheel_r", "category": "wheel", "description": "右驱动轮",
             "material": "TPU", "dimensions": {"diameter": 65, "width": 20}},
            {"name": "caster_front", "category": "wheel", "description": "前万向轮",
             "material": "Nylon", "dimensions": {"diameter": 30}},
            {"name": "motor_mount_l", "category": "structural", "description": "左电机支架",
             "material": "PLA", "dimensions": {"length": 40, "width": 25, "height": 20}},
            {"name": "motor_mount_r", "category": "structural", "description": "右电机支架",
             "material": "PLA", "dimensions": {"length": 40, "width": 25, "height": 20}},
        ],
        joints=[
            {"type": "fixed", "parent": "chassis", "child": "motor_mount_l"},
            {"type": "fixed", "parent": "chassis", "child": "motor_mount_r"},
            {"type": "fixed", "parent": "chassis", "child": "caster_front"},
            {"type": "revolute", "parent": "motor_mount_l", "child": "wheel_l", "axis": "y"},
            {"type": "revolute", "parent": "motor_mount_r", "child": "wheel_r", "axis": "y"},
        ],
        parameters={
            "wheel_diameter_mm": [40, 100],
            "track_width_mm": [150, 300],
            "payload_kg": [0.5, 10],
        },
    ),
    "mecanum": MobileBaseTemplate(
        name="Mecanum全向底盘",
        description="4轮Mecanum全向驱动底盘，支持平移和旋转",
        drive_type="mecanum",
        parts=[
            {"name": "chassis", "category": "structural", "description": "方形底盘",
             "material": "PLA", "dimensions": {"length": 300, "width": 300, "height": 5}},
            {"name": "wheel_fl", "category": "wheel", "description": "左前Mecanum轮",
             "material": "TPU", "dimensions": {"diameter": 80, "width": 25}},
            {"name": "wheel_fr", "category": "wheel", "description": "右前Mecanum轮",
             "material": "TPU", "dimensions": {"diameter": 80, "width": 25}},
            {"name": "wheel_rl", "category": "wheel", "description": "左后Mecanum轮",
             "material": "TPU", "dimensions": {"diameter": 80, "width": 25}},
            {"name": "wheel_rr", "category": "wheel", "description": "右后Mecanum轮",
             "material": "TPU", "dimensions": {"diameter": 80, "width": 25}},
        ],
        joints=[
            {"type": "revolute", "parent": "chassis", "child": "wheel_fl", "axis": "y"},
            {"type": "revolute", "parent": "chassis", "child": "wheel_fr", "axis": "y"},
            {"type": "revolute", "parent": "chassis", "child": "wheel_rl", "axis": "y"},
            {"type": "revolute", "parent": "chassis", "child": "wheel_rr", "axis": "y"},
        ],
        parameters={
            "wheel_diameter_mm": [60, 120],
            "track_width_mm": [250, 500],
            "wheelbase_mm": [250, 500],
            "payload_kg": [2, 30],
        },
    ),
}


def design_mobile_base(
    payload_kg: float = 5.0,
    max_speed_mm_s: float = 500.0,
    max_grade_deg: float = 10.0,
    runtime_hours: float = 2.0,
    drive_type: str = "differential_4w",
    wheel_count: int = 4,
) -> dict[str, Any]:
    """End-to-end mobile base design from high-level requirements.

    Returns a complete specification including chassis parameters, motor
    selection, battery sizing, and kinematic properties.
    """
    # Step 1: Wheelbase sizing
    wb_calc = WheelBaseCalculator(
        payload_kg=payload_kg,
        wheel_count=wheel_count,
        drive_type="mecanum" if "mecanum" in drive_type else "differential",
    )
    wheel_d = wb_calc.wheel_diameter_mm()
    track = wb_calc.track_width_mm()
    clearance = wb_calc.ground_clearance_mm()

    # Step 2: Motor torque
    torque_calc = MotorTorqueCalculator(
        total_mass_kg=wb_calc.total_mass_kg,
        wheel_radius_mm=wheel_d / 2,
        wheel_count=wheel_count,
        max_grade_deg=max_grade_deg,
        max_acceleration_mm_s2=max_speed_mm_s * 0.5,  # reach speed in 2s
    )
    selected_motor = torque_calc.select_motor()

    # Step 3: Kinematics
    max_rpm = selected_motor.rated_speed_rpm if selected_motor else 200
    if "mecanum" in drive_type:
        kinematics = OmnidirectionalKinematics(
            wheel_radius_mm=wheel_d / 2,
            track_width_mm=track,
            wheelbase_mm=wb_calc.wheelbase_mm(),
        )
    else:
        kinematics = DifferentialDriveKinematics(
            wheel_radius_mm=wheel_d / 2,
            track_width_mm=track,
            wheelbase_mm=wb_calc.wheelbase_mm(),
            max_rpm=max_rpm,
        )

    # Step 4: Battery sizing
    motor_power = selected_motor.rated_torque_kg_cm * 9.81 / 100 * (
        selected_motor.rated_speed_rpm * 2 * math.pi / 60
    ) if selected_motor else 10.0
    battery_calc = BatterySizingCalculator(
        motor_count=wheel_count,
        motor_power_w=motor_power,
        runtime_hours=runtime_hours,
        voltage=selected_motor.nominal_voltage if selected_motor else 12.0,
    )

    # Step 5: Template
    template = MOBILE_BASE_TEMPLATES.get(drive_type)

    result = {
        "requirements": {
            "payload_kg": payload_kg,
            "max_speed_mm_s": max_speed_mm_s,
            "max_grade_deg": max_grade_deg,
            "runtime_hours": runtime_hours,
            "drive_type": drive_type,
        },
        "chassis": wb_calc.to_dict(),
        "motor": {
            "required_torque_kg_cm": round(torque_calc.total_torque_kg_cm_per_wheel(), 2),
            "selected": selected_motor.name if selected_motor else "None found",
            "specs": {
                "voltage": selected_motor.nominal_voltage,
                "rated_speed_rpm": selected_motor.rated_speed_rpm,
                "rated_torque_kg_cm": selected_motor.rated_torque_kg_cm,
            } if selected_motor else {},
        },
        "battery": battery_calc.to_dict(),
        "kinematics": kinematics.to_dict(),
        "ground_clearance_mm": clearance,
    }
    if template:
        result["template"] = template.name
        result["parts_count"] = len(template.parts)

    return result
