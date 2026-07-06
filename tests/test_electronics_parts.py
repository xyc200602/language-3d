"""Tests for Task 75: Electronics & sensor parts library."""

import pytest

from lang3d.knowledge.parts_catalog import (
    CATEGORY_TREE,
    PART_CATALOG,
    PartTemplate,
    format_fc_script,
    get_template,
    resolve_parameters,
    search_by_subsystem,
    search_parts,
    find_compatible_parts,
)

EXPECTED_IDS = [
    "driver_l298n", "driver_tb6612fng",
    "controller_arduino_uno", "controller_arduino_nano", "controller_esp32_devkit",
    "encoder_as5600", "limit_switch_kw12",
    "power_lm2596_buck",
    "connector_xt60", "connector_jst_xh",
]


# =====================================================================
# 1. Catalog registration
# =====================================================================

class TestCatalogRegistration:

    def test_all_parts_registered(self):
        for pid in EXPECTED_IDS:
            assert pid in PART_CATALOG, f"Missing: {pid}"

    def test_total_catalog_size(self):
        assert len(PART_CATALOG) >= 59

    def test_category_tree_has_electronics(self):
        assert "electronics" in CATEGORY_TREE
        assert "motor_driver" in CATEGORY_TREE["electronics"]
        assert "controller" in CATEGORY_TREE["electronics"]
        assert "power_module" in CATEGORY_TREE["electronics"]
        assert "connector" in CATEGORY_TREE["electronics"]

    def test_sensor_subcategories_updated(self):
        assert "encoder" in CATEGORY_TREE["sensor"]
        assert "limit_switch" in CATEGORY_TREE["sensor"]

    def test_electronics_subsystem(self):
        results = search_by_subsystem("electronics")
        ids = {t.id for t in results}
        for pid in ["driver_l298n", "controller_arduino_uno", "power_lm2596_buck",
                     "connector_xt60", "connector_jst_xh"]:
            assert pid in ids


# =====================================================================
# 2. Part classification
# =====================================================================

class TestPartClassification:

    @pytest.mark.parametrize("pid", EXPECTED_IDS)
    def test_is_functional(self, pid):
        t = get_template(pid)
        assert t.part_class == "functional"
        assert t.scalable is False
        assert t.real_part is True

    @pytest.mark.parametrize("pid", EXPECTED_IDS)
    def test_float_params_fixed(self, pid):
        t = get_template(pid)
        for p in t.parameters:
            if p.param_type == "float":
                assert p.fixed is True, f"{pid}: '{p.name}' not fixed"


# =====================================================================
# 3. Motor drivers
# =====================================================================

class TestL298N:

    def test_metadata(self):
        t = get_template("driver_l298n")
        assert t.category == "electronics"
        assert t.subcategory == "motor_driver"

    def test_dimensions(self):
        t = get_template("driver_l298n")
        s = t.standard_sizes[0]
        assert s["pcb_length"] == 43.0
        assert s["pcb_width"] == 43.0

    def test_script_generation(self):
        t = get_template("driver_l298n")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Heatsink" in script
        assert "TerminalBlock" in script
        assert "PCB" in script


class TestTB6612FNG:

    def test_metadata(self):
        t = get_template("driver_tb6612fng")
        assert t.subcategory == "motor_driver"

    def test_smaller_than_l298n(self):
        t_l298 = get_template("driver_l298n")
        t_tb = get_template("driver_tb6612fng")
        area_l298 = t_l298.standard_sizes[0]["pcb_length"] * t_l298.standard_sizes[0]["pcb_width"]
        area_tb = t_tb.standard_sizes[0]["pcb_length"] * t_tb.standard_sizes[0]["pcb_width"]
        assert area_tb < area_l298

    def test_script_generation(self):
        t = get_template("driver_tb6612fng")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "IC" in script
        assert "PinHeader" in script


# =====================================================================
# 4. Controllers
# =====================================================================

class TestArduinoUno:

    def test_metadata(self):
        t = get_template("controller_arduino_uno")
        assert t.subcategory == "controller"
        assert "Arduino" in " ".join(t.tags)

    def test_dimensions(self):
        t = get_template("controller_arduino_uno")
        s = t.standard_sizes[0]
        assert s["pcb_length"] == 68.6
        assert s["pcb_width"] == 53.4

    def test_script_has_usb(self):
        t = get_template("controller_arduino_uno")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "USB" in script
        assert "MCU" in script
        assert "PinHeader" in script


class TestArduinoNano:

    def test_metadata(self):
        t = get_template("controller_arduino_nano")
        assert t.subcategory == "controller"

    def test_smaller_than_uno(self):
        t_uno = get_template("controller_arduino_uno")
        t_nano = get_template("controller_arduino_nano")
        assert t_nano.standard_sizes[0]["pcb_length"] < t_uno.standard_sizes[0]["pcb_length"]

    def test_script_generation(self):
        t = get_template("controller_arduino_nano")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "USB" in script


class TestESP32DevKit:

    def test_metadata(self):
        t = get_template("controller_esp32_devkit")
        assert t.subcategory == "controller"
        assert "Wi-Fi" in " ".join(t.tags) or "ESP32" in " ".join(t.tags)

    def test_script_has_module(self):
        t = get_template("controller_esp32_devkit")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "ESP32Module" in script
        assert "USB" in script


# =====================================================================
# 5. Encoder & limit switch
# =====================================================================

class TestAS5600Encoder:

    def test_metadata(self):
        t = get_template("encoder_as5600")
        assert t.category == "sensor"
        assert t.subcategory == "encoder"

    def test_14bit_resolution_noted(self):
        t = get_template("encoder_as5600")
        assert "14-bit" in t.notes or "14 bit" in t.notes

    def test_center_hole(self):
        t = get_template("encoder_as5600")
        s = t.standard_sizes[0]
        assert s["center_hole_diameter"] > 0  # Shaft pass-through

    def test_script_has_magnet(self):
        t = get_template("encoder_as5600")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Magnet" in script
        assert "center_hole" in script


class TestLimitSwitch:

    def test_metadata(self):
        t = get_template("limit_switch_kw12")
        assert t.category == "sensor"
        assert t.subcategory == "limit_switch"

    def test_has_lever(self):
        t = get_template("limit_switch_kw12")
        param_names = {p.name for p in t.parameters}
        assert "lever_length" in param_names

    def test_script_has_lever(self):
        t = get_template("limit_switch_kw12")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Lever" in script
        assert "Pin" in script

    def test_two_lever_lengths(self):
        t = get_template("limit_switch_kw12")
        lever_lengths = {s["lever_length"] for s in t.standard_sizes}
        assert len(lever_lengths) >= 2


# =====================================================================
# 6. Power module
# =====================================================================

class TestLM2596Buck:

    def test_metadata(self):
        t = get_template("power_lm2596_buck")
        assert t.category == "electronics"
        assert t.subcategory == "power_module"

    def test_script_has_inductor(self):
        t = get_template("power_lm2596_buck")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Inductor" in script
        assert "Potentiometer" in script


# =====================================================================
# 7. Connectors
# =====================================================================

class TestXT60:

    def test_metadata(self):
        t = get_template("connector_xt60")
        assert t.category == "electronics"
        assert t.subcategory == "connector"

    def test_script_generation(self):
        t = get_template("connector_xt60")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "XT60" in script
        assert "contact" in script.lower()


class TestJSTXH:

    def test_metadata(self):
        t = get_template("connector_jst_xh")
        assert t.category == "electronics"
        assert t.subcategory == "connector"

    def test_pin_variants(self):
        t = get_template("connector_jst_xh")
        pins = {s["num_pins"] for s in t.standard_sizes}
        assert 2 in pins
        assert 4 in pins
        assert 6 in pins

    def test_script_with_4_pins(self):
        t = get_template("connector_jst_xh")
        params = resolve_parameters(t, {"num_pins": 4})
        script = format_fc_script(t, params)
        assert "JST_XH" in script
        assert "Pin_" in script


# =====================================================================
# 8. Search & compatibility
# =====================================================================

class TestSearchAndCompatibility:

    def test_search_motor_driver(self):
        results = search_parts(query="motor driver")
        ids = {t.id for t in results}
        assert "driver_l298n" in ids
        assert "driver_tb6612fng" in ids

    def test_search_arduino(self):
        results = search_parts(query="Arduino")
        ids = {t.id for t in results}
        assert "controller_arduino_uno" in ids
        assert "controller_arduino_nano" in ids

    def test_search_encoder(self):
        results = search_parts(query="encoder")
        ids = {t.id for t in results}
        assert "encoder_as5600" in ids

    def test_search_chinese(self):
        results = search_parts(query="限位开关")
        ids = {t.id for t in results}
        assert "limit_switch_kw12" in ids

    def test_search_connector(self):
        results = search_parts(query="connector")
        ids = {t.id for t in results}
        assert "connector_xt60" in ids
        assert "connector_jst_xh" in ids

    def test_compatible_parts_l298n(self):
        compat = find_compatible_parts("driver_l298n")
        ids = {t.id for t in compat}
        assert len(compat) > 0

    def test_electronics_subsystem_search(self):
        results = search_by_subsystem("electronics")
        assert len(results) >= 8

    def test_sensor_category_count(self):
        results = search_parts(category="sensor")
        # lidar, imu, camera, encoder, limit_switch = 5+ original sensors
        assert len(results) >= 5


# =====================================================================
# 9. Parameter resolution & script integration
# =====================================================================

class TestParameterResolution:

    def test_resolve_arduino_uno(self):
        t = get_template("controller_arduino_uno")
        params = resolve_parameters(t)
        assert params["pcb_length"] == 68.6

    def test_resolve_jst_xh_pins(self):
        t = get_template("connector_jst_xh")
        params = resolve_parameters(t, {"num_pins": 3})
        assert params["num_pins"] == 3

    def test_invalid_pin_count(self):
        t = get_template("connector_jst_xh")
        with pytest.raises(ValueError):
            resolve_parameters(t, {"num_pins": 99})


class TestScriptGeneration:

    def test_all_simplified_scripts(self):
        for pid in EXPECTED_IDS:
            t = get_template(pid)
            if t.standard_sizes:
                params = resolve_parameters(t, t.standard_sizes[0])
            else:
                params = resolve_parameters(t)
            script = format_fc_script(t, params)
            assert len(script) > 30, f"Script too short for {pid}"

    def test_controller_scripts_have_usb(self):
        for pid in ["controller_arduino_uno", "controller_arduino_nano", "controller_esp32_devkit"]:
            t = get_template(pid)
            params = resolve_parameters(t)
            script = format_fc_script(t, params)
            assert "USB" in script

    def test_all_electronics_have_pcb(self):
        """Most electronics parts should have PCB in their scripts."""
        for pid in ["driver_l298n", "driver_tb6612fng", "controller_arduino_uno",
                     "controller_arduino_nano", "controller_esp32_devkit",
                     "power_lm2596_buck"]:
            t = get_template(pid)
            params = resolve_parameters(t)
            script = format_fc_script(t, params)
            assert "PCB" in script, f"{pid} missing PCB in script"
