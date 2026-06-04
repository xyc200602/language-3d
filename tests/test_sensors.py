"""Tests for sensor integration.

Tests cover:
- Sensor knowledge base (lookup, listing, filtering)
- Sensor recommendations for joints
- Firmware generation with sensors
- Sensor driver code generation
- Power budget with sensors
"""

from __future__ import annotations

import pytest

from lang3d.knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
from lang3d.knowledge.sensors import (
    SENSORS,
    Sensor,
    get_sensor,
    list_sensors,
    recommend_sensors_for_joints,
)
from lang3d.tools.actuator_tools import power_budget
from lang3d.tools.code_gen import (
    GenFirmwareTool,
    generate_firmware,
    generate_wiring,
)


# ============================================================================
# Test: Sensor knowledge base
# ============================================================================

class TestSensorKnowledgeBase:
    """Test sensor database and lookup functions."""

    def test_sensors_dict_not_empty(self):
        assert len(SENSORS) > 0

    def test_get_sensor_known(self):
        s = get_sensor("AS5600")
        assert s is not None
        assert s.name == "AS5600 磁编码器"
        assert s.category == "encoder"
        assert s.interface == "i2c"

    def test_get_sensor_unknown(self):
        assert get_sensor("NONEXISTENT") is None

    def test_list_all_sensors(self):
        all_sensors = list_sensors()
        assert len(all_sensors) >= 7

    def test_list_by_category_encoder(self):
        encoders = list_sensors("encoder")
        assert len(encoders) >= 2
        assert all(s.category == "encoder" for s in encoders)

    def test_list_by_category_limit_switch(self):
        switches = list_sensors("limit_switch")
        assert len(switches) >= 2
        assert all(s.category == "limit_switch" for s in switches)

    def test_list_by_category_imu(self):
        imus = list_sensors("imu")
        assert len(imus) >= 1
        assert imus[0].id == "MPU6050"

    def test_list_by_category_empty(self):
        result = list_sensors("nonexistent")
        assert result == []

    def test_sensor_fields(self):
        s = get_sensor("AS5600")
        assert s.voltage > 0
        assert s.current_ma > 0
        assert s.resolution == "14-bit (0.022°)"
        assert s.range_deg == (0, 360)
        assert len(s.pins) > 0
        assert s.price_cny > 0

    def test_limit_switch_fields(self):
        s = get_sensor("LIMIT_SWITCH_MICRO")
        assert s.interface == "digital"
        assert s.range_deg is None

    def test_potentiometer_fields(self):
        s = get_sensor("POT_10K")
        assert s.interface == "analog"
        assert s.range_deg == (0, 300)

    def test_imu_fields(self):
        s = get_sensor("MPU6050")
        assert s.interface == "i2c"
        assert "INT" in s.pins

    def test_proximity_sensor(self):
        s = get_sensor("VL53L0X")
        assert s.category == "proximity"
        assert s.interface == "i2c"


# ============================================================================
# Test: Sensor recommendations
# ============================================================================

class TestSensorRecommendations:
    """Test automatic sensor recommendation for joints."""

    def test_encoder_recommendation(self):
        recs = recommend_sensors_for_joints(4, feedback_type="encoder")
        encoder_recs = [r for r in recs if r["purpose"] == "angle_feedback"]
        assert len(encoder_recs) == 4
        assert all(r["sensor_id"] == "AS5600" for r in encoder_recs)

    def test_potentiometer_recommendation(self):
        recs = recommend_sensors_for_joints(3, feedback_type="potentiometer")
        pot_recs = [r for r in recs if r["purpose"] == "angle_feedback"]
        assert len(pot_recs) == 3
        assert all(r["sensor_id"] == "POT_10K" for r in pot_recs)

    def test_no_feedback(self):
        recs = recommend_sensors_for_joints(4, feedback_type="none")
        feedback_recs = [r for r in recs if r["purpose"] == "angle_feedback"]
        assert len(feedback_recs) == 0

    def test_with_limit_switches(self):
        recs = recommend_sensors_for_joints(4, include_limit_switches=True)
        limit_recs = [r for r in recs if r["purpose"] == "limit_switch"]
        assert len(limit_recs) == 4

    def test_without_limit_switches(self):
        recs = recommend_sensors_for_joints(4, include_limit_switches=False)
        limit_recs = [r for r in recs if r["purpose"] == "limit_switch"]
        assert len(limit_recs) == 0

    def test_with_imu(self):
        recs = recommend_sensors_for_joints(4, include_imu=True)
        imu_recs = [r for r in recs if r["purpose"] == "orientation_feedback"]
        assert len(imu_recs) == 1
        assert imu_recs[0]["sensor_id"] == "MPU6050"
        assert imu_recs[0]["joint_index"] == -1

    def test_without_imu(self):
        recs = recommend_sensors_for_joints(4, include_imu=False)
        imu_recs = [r for r in recs if r["purpose"] == "orientation_feedback"]
        assert len(imu_recs) == 0

    def test_full_recommendation(self):
        recs = recommend_sensors_for_joints(
            4,
            feedback_type="encoder",
            include_limit_switches=True,
            include_imu=True,
        )
        # 4 encoders + 4 limit switches + 1 IMU = 9
        assert len(recs) == 9

    def test_joint_indices(self):
        recs = recommend_sensors_for_joints(3, feedback_type="encoder")
        indices = [r["joint_index"] for r in recs if r["purpose"] == "angle_feedback"]
        assert indices == [0, 1, 2]


# ============================================================================
# Test: Firmware generation with sensors
# ============================================================================

class TestFirmwareWithSensors:
    """Test firmware generation with sensor integration."""

    @pytest.fixture
    def firmware_no_sensors(self):
        return generate_firmware(
            ROBOTIC_ARM_ASSEMBLY,
            ["MG996R", "MG996R", "DS3218", "SG90"],
        )

    @pytest.fixture
    def firmware_with_sensors(self):
        return generate_firmware(
            ROBOTIC_ARM_ASSEMBLY,
            ["MG996R", "MG996R", "DS3218", "SG90"],
            sensor_ids=["AS5600", "LIMIT_SWITCH_MICRO", "MPU6050"],
        )

    def test_no_sensors_no_driver_files(self, firmware_no_sensors):
        """Without sensors, no sensor_driver files are generated."""
        assert "sensor_driver.h" not in firmware_no_sensors
        assert "sensor_driver.cpp" not in firmware_no_sensors

    def test_with_sensors_generates_driver_files(self, firmware_with_sensors):
        """With sensors, sensor_driver.h and .cpp are generated."""
        assert "sensor_driver.h" in firmware_with_sensors
        assert "sensor_driver.cpp" in firmware_with_sensors

    def test_sensor_header_has_guards(self, firmware_with_sensors):
        h = firmware_with_sensors["sensor_driver.h"]
        assert "SENSOR_DRIVER_H" in h
        assert "sensor_init" in h
        assert "sensor_read_all" in h
        assert "sensor_print_status" in h

    def test_sensor_header_has_encoder_func(self, firmware_with_sensors):
        h = firmware_with_sensors["sensor_driver.h"]
        assert "sensor_read_encoder" in h

    def test_sensor_header_has_limit_func(self, firmware_with_sensors):
        h = firmware_with_sensors["sensor_driver.h"]
        assert "sensor_limit_triggered" in h

    def test_sensor_header_has_imu_func(self, firmware_with_sensors):
        h = firmware_with_sensors["sensor_driver.h"]
        assert "sensor_read_imu" in h

    def test_sensor_cpp_has_i2c(self, firmware_with_sensors):
        cpp = firmware_with_sensors["sensor_driver.cpp"]
        assert "Wire.h" in cpp
        assert "Wire.begin" in cpp

    def test_sensor_cpp_has_as5600(self, firmware_with_sensors):
        cpp = firmware_with_sensors["sensor_driver.cpp"]
        assert "AS5600" in cpp
        assert "0x36" in cpp

    def test_sensor_cpp_has_limit_check(self, firmware_with_sensors):
        cpp = firmware_with_sensors["sensor_driver.cpp"]
        assert "LIMIT_SWITCH_PINS" in cpp
        assert "digitalRead" in cpp

    def test_ino_includes_sensor_driver(self, firmware_with_sensors):
        ino = firmware_with_sensors["robot_arm.ino"]
        assert "sensor_driver.h" in ino
        assert "sensor_init()" in ino
        assert "sensor_read_all()" in ino

    def test_ino_has_limit_safety(self, firmware_with_sensors):
        ino = firmware_with_sensors["robot_arm.ino"]
        assert "sensor_limit_triggered" in ino

    def test_ino_has_sensor_status_command(self, firmware_with_sensors):
        ino = firmware_with_sensors["robot_arm.ino"]
        assert "case 'S'" in ino
        assert "sensor_print_status" in ino

    def test_ino_no_sensor_code_without_sensors(self, firmware_no_sensors):
        ino = firmware_no_sensors["robot_arm.ino"]
        assert "sensor_driver.h" not in ino
        assert "sensor_init" not in ino

    def test_sensor_driver_nonempty(self, firmware_with_sensors):
        for fname in ["sensor_driver.h", "sensor_driver.cpp"]:
            content = firmware_with_sensors[fname]
            assert len(content) > 50, f"{fname} is too short"


# ============================================================================
# Test: Firmware with potentiometers only
# ============================================================================

class TestFirmwareWithPotentiometers:
    """Test firmware generation with potentiometer sensors."""

    @pytest.fixture
    def firmware(self):
        return generate_firmware(
            ROBOTIC_ARM_ASSEMBLY,
            ["MG996R", "MG996R", "DS3218", "SG90"],
            sensor_ids=["POT_10K"],
        )

    def test_has_sensor_files(self, firmware):
        assert "sensor_driver.h" in firmware
        assert "sensor_driver.cpp" in firmware

    def test_has_potentiometer_func(self, firmware):
        h = firmware["sensor_driver.h"]
        assert "sensor_read_potentiometer" in h

    def test_has_analog_read(self, firmware):
        cpp = firmware["sensor_driver.cpp"]
        assert "analogRead" in cpp


# ============================================================================
# Test: Power budget with sensors
# ============================================================================

class TestPowerBudgetWithSensors:
    """Test power budget calculation including sensor power."""

    def test_power_budget_no_sensors(self):
        result = power_budget(["MG996R", "SG90"])
        assert result["total_power_w"] > 0
        assert len(result["sensors"]) == 0

    def test_power_budget_with_sensors(self):
        result = power_budget(
            ["MG996R", "SG90"],
            sensor_ids=["AS5600", "AS5600", "MPU6050"],
        )
        assert len(result["sensors"]) == 3
        assert result["total_power_w"] > 0

    def test_power_budget_sensors_increase_total(self):
        without = power_budget(["MG996R"])
        with_sensors = power_budget(["MG996R"], sensor_ids=["AS5600", "MPU6050"])
        assert with_sensors["total_power_w"] > without["total_power_w"]

    def test_power_budget_only_sensors(self):
        result = power_budget([], sensor_ids=["AS5600", "MPU6050"])
        assert result["count"] == 0
        assert len(result["sensors"]) == 2
        assert result["total_power_w"] > 0

    def test_power_budget_unknown_sensor(self):
        result = power_budget(["MG996R"], sensor_ids=["FAKE_SENSOR"])
        assert len(result["sensors"]) == 0

    def test_power_budget_sensor_details(self):
        result = power_budget(
            ["MG996R"],
            sensor_ids=["AS5600"],
        )
        assert result["sensors"][0]["id"] == "AS5600"
        assert result["sensors"][0]["power_w"] > 0


# ============================================================================
# Test: GenFirmwareTool with sensors
# ============================================================================

class TestGenFirmwareToolWithSensors:
    """Test the GenFirmwareTool with sensor_ids parameter."""

    def test_execute_with_sensors(self):
        tool = GenFirmwareTool()
        result = tool.execute(
            actuator_ids=["MG996R", "SG90"],
            sensor_ids=["AS5600", "LIMIT_SWITCH_MICRO"],
        )
        assert "Firmware Generated" in result
        assert "sensor_driver" in result

    def test_execute_without_sensors(self):
        tool = GenFirmwareTool()
        result = tool.execute(actuator_ids=["MG996R", "SG90"])
        assert "Firmware Generated" in result
        # sensor_driver may or may not appear in preview
