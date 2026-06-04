"""Sensor knowledge base for robotic assemblies.

Provides sensor parameters for integration into firmware:
  - Limit switches (mechanical endstops)
  - Potentiometers (analog angle feedback)
  - Magnetic encoders (AS5600 high-resolution)
  - IMU (MPU6050 6-axis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sensor:
    """A sensor definition."""

    id: str
    name: str
    category: str  # "limit_switch", "potentiometer", "encoder", "imu", "proximity"
    description: str
    interface: str  # "digital", "analog", "i2c", "spi", "uart"
    voltage: float  # Operating voltage (V)
    current_ma: float  # Operating current (mA)
    resolution: str  # e.g. "12-bit", "N/A"
    range_deg: tuple[float, float] | None = None  # For angle sensors
    weight_g: float = 0
    price_cny: float = 0
    pins: list[str] = field(default_factory=list)  # Required MCU pins
    notes: str = ""


# ============================================================================
# Sensor Database
# ============================================================================

SENSORS: dict[str, Sensor] = {}

# --- Limit Switches ---

SENSORS["LIMIT_SWITCH_MICRO"] = Sensor(
    id="LIMIT_SWITCH_MICRO",
    name="微型限位开关",
    category="limit_switch",
    description="机械式限位开关，用于关节行程端点检测",
    interface="digital",
    voltage=3.3,
    current_ma=5,
    resolution="N/A",
    weight_g=2,
    price_cny=1,
    pins=["GPIO (digital input)", "GND"],
    notes="常开/常闭可选，需要上拉/下拉电阻",
)

SENSORS["LIMIT_SWITCH_ROLLER"] = Sensor(
    id="LIMIT_SWITCH_ROLLER",
    name="滚轮限位开关",
    category="limit_switch",
    description="滚轮式限位开关，适合运动部件触发",
    interface="digital",
    voltage=3.3,
    current_ma=5,
    resolution="N/A",
    weight_g=5,
    price_cny=3,
    pins=["GPIO (digital input)", "GND"],
    notes="触发力度低，适合连续触发场景",
)

# --- Potentiometers ---

SENSORS["POT_10K"] = Sensor(
    id="POT_10K",
    name="10K 电位器",
    category="potentiometer",
    description="单圈旋转电位器，用于关节角度模拟反馈",
    interface="analog",
    voltage=3.3,
    current_ma=0.5,
    resolution="10-bit (ADC)",
    range_deg=(0, 300),
    weight_g=3,
    price_cny=2,
    pins=["VCC", "GPIO (ADC)", "GND"],
    notes="线性/对数可选，推荐线性型",
)

# --- Magnetic Encoders ---

SENSORS["AS5600"] = Sensor(
    id="AS5600",
    name="AS5600 磁编码器",
    category="encoder",
    description="14-bit 非接触式磁编码器，高精度角度反馈",
    interface="i2c",
    voltage=3.3,
    current_ma=6,
    resolution="14-bit (0.022°)",
    range_deg=(0, 360),
    weight_g=1,
    price_cny=8,
    pins=["SDA", "SCL", "VCC", "GND"],
    notes="需要径向磁铁（ diametrically magnetized），支持绝对角度输出",
)

SENSORS["AS5048A"] = Sensor(
    id="AS5048A",
    name="AS5048A 磁编码器",
    category="encoder",
    description="14-bit PWM/SPI 磁编码器，高速角度反馈",
    interface="spi",
    voltage=3.3,
    current_ma=13,
    resolution="14-bit (0.022°)",
    range_deg=(0, 360),
    weight_g=1,
    price_cny=25,
    pins=["CS", "CLK", "MISO", "VCC", "GND"],
    notes="SPI 接口响应更快，适合高速控制回路",
)

# --- IMU ---

SENSORS["MPU6050"] = Sensor(
    id="MPU6050",
    name="MPU6050 六轴 IMU",
    category="imu",
    description="六轴惯性测量单元（3轴加速度+3轴陀螺仪），用于姿态反馈",
    interface="i2c",
    voltage=3.3,
    current_ma=4,
    resolution="16-bit (accel/gyro)",
    weight_g=2,
    price_cny=5,
    pins=["SDA", "SCL", "VCC", "GND", "INT"],
    notes="内置 DMP 可做姿态解算，加速度范围 ±2/4/8/16g，陀螺仪 ±250/500/1000/2000°/s",
)

# --- Proximity ---

SENSORS["VL53L0X"] = Sensor(
    id="VL53L0X",
    name="VL53L0X 激光测距",
    category="proximity",
    description="ToF 激光测距传感器，末端避障/抓取检测",
    interface="i2c",
    voltage=3.3,
    current_ma=10,
    resolution="N/A",
    weight_g=1,
    price_cny=8,
    pins=["SDA", "SCL", "VCC", "GND", "XSHUT"],
    notes="测量范围 30-1000mm，精度 ±3%",
)


# ============================================================================
# Lookup helpers
# ============================================================================

def get_sensor(sensor_id: str) -> Sensor | None:
    """Get a sensor by ID."""
    return SENSORS.get(sensor_id)


def list_sensors(category: str = "") -> list[Sensor]:
    """List sensors, optionally filtered by category."""
    sensors = list(SENSORS.values())
    if category:
        sensors = [s for s in sensors if s.category == category]
    return sensors


def recommend_sensors_for_joints(
    num_joints: int,
    *,
    feedback_type: str = "encoder",
    include_limit_switches: bool = True,
    include_imu: bool = False,
) -> list[dict[str, Any]]:
    """Recommend sensors for a robotic arm with N revolute joints.

    Args:
        num_joints: Number of revolute joints.
        feedback_type: "encoder" (AS5600), "potentiometer" (POT_10K), or "none".
        include_limit_switches: Add limit switches for joint endpoints.
        include_imu: Add MPU6050 for end-effector orientation feedback.

    Returns list of {joint_index, sensor_id, sensor, purpose}.
    """
    recommendations: list[dict[str, Any]] = []

    for i in range(num_joints):
        # Angle feedback
        if feedback_type == "encoder":
            recommendations.append({
                "joint_index": i,
                "sensor_id": "AS5600",
                "sensor": SENSORS["AS5600"],
                "purpose": "angle_feedback",
            })
        elif feedback_type == "potentiometer":
            recommendations.append({
                "joint_index": i,
                "sensor_id": "POT_10K",
                "sensor": SENSORS["POT_10K"],
                "purpose": "angle_feedback",
            })

        # Limit switches
        if include_limit_switches:
            recommendations.append({
                "joint_index": i,
                "sensor_id": "LIMIT_SWITCH_MICRO",
                "sensor": SENSORS["LIMIT_SWITCH_MICRO"],
                "purpose": "limit_switch",
            })

    # IMU on end-effector
    if include_imu:
        recommendations.append({
            "joint_index": -1,
            "sensor_id": "MPU6050",
            "sensor": SENSORS["MPU6050"],
            "purpose": "orientation_feedback",
        })

    return recommendations
