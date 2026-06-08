"""Tests for Task 74: Linear motion & advanced actuator parts library."""

import pytest

from src.lang3d.knowledge.parts_catalog import (
    CATEGORY_TREE,
    PART_CATALOG,
    PartTemplate,
    ParamDef,
    format_fc_script,
    get_template,
    resolve_parameters,
    search_by_subsystem,
    search_parts,
    find_compatible_parts,
    get_functional_parts,
)


# =====================================================================
# 1. Catalog registration
# =====================================================================

EXPECTED_IDS = [
    "linear_bearing_lm8uu", "linear_bearing_lm10uu", "linear_bearing_lm12uu",
    "linear_guide_mgn12h", "t8_leadscrew", "t8_nut",
    "bldc_motor_5010", "bldc_motor_2208",
    "compression_spring", "damper_shock_absorber",
]


class TestCatalogRegistration:

    def test_all_parts_registered(self):
        for pid in EXPECTED_IDS:
            assert pid in PART_CATALOG, f"Missing: {pid}"

    def test_total_catalog_size(self):
        assert len(PART_CATALOG) == 49

    def test_category_tree_updates(self):
        assert "linear_bearing" in CATEGORY_TREE["bearing"]
        assert "leadscrew" in CATEGORY_TREE["shaft"]
        assert "bldc" in CATEGORY_TREE["actuator"]

    def test_linear_motion_subsystem(self):
        results = search_by_subsystem("linear_motion")
        ids = {t.id for t in results}
        for pid in EXPECTED_IDS:
            assert pid in ids, f"Missing in linear_motion subsystem: {pid}"


# =====================================================================
# 2. Part classification — all functional, fixed params
# =====================================================================

class TestPartClassification:

    @pytest.mark.parametrize("pid", EXPECTED_IDS)
    def test_is_functional(self, pid):
        t = get_template(pid)
        assert t.part_class == "functional", f"{pid}: expected functional"
        assert t.scalable is False, f"{pid}: should not be scalable"
        assert t.real_part is True, f"{pid}: should be real_part"

    @pytest.mark.parametrize("pid", EXPECTED_IDS)
    def test_float_params_fixed(self, pid):
        t = get_template(pid)
        for p in t.parameters:
            if p.param_type == "float":
                assert p.fixed is True, f"{pid}: param '{p.name}' not fixed"


# =====================================================================
# 3. Linear bearings
# =====================================================================

class TestLinearBearingLM8UU:

    def test_metadata(self):
        t = get_template("linear_bearing_lm8uu")
        assert t.category == "bearing"
        assert t.subcategory == "linear_bearing"
        assert t.model_number == "LM8UU"

    def test_dimensions(self):
        t = get_template("linear_bearing_lm8uu")
        dims = t.standard_sizes[0]
        assert dims["inner_diameter"] == 8.0
        assert dims["outer_diameter"] == 15.0
        assert dims["length"] == 24.0

    def test_simplified_script(self):
        t = get_template("linear_bearing_lm8uu")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "LinearBearing" in script
        assert "makeCylinder" in script

    def test_realistic_script(self):
        t = get_template("linear_bearing_lm8uu")
        params = resolve_parameters(t, {"bearing_detail": "realistic"})
        script = format_fc_script(t, params)
        assert "OuterRing" in script
        assert "InnerRing" in script
        assert "Ball" in script

    def test_quality_levels(self):
        t = get_template("linear_bearing_lm8uu")
        assert "simplified" in t.quality_levels
        assert "realistic" in t.quality_levels


class TestLinearBearingLM10UU:

    def test_dimensions(self):
        t = get_template("linear_bearing_lm10uu")
        dims = t.standard_sizes[0]
        assert dims["inner_diameter"] == 10.0
        assert dims["outer_diameter"] == 19.0

    def test_script(self):
        t = get_template("linear_bearing_lm10uu")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "LinearBearing" in script


class TestLinearBearingLM12UU:

    def test_dimensions(self):
        t = get_template("linear_bearing_lm12uu")
        dims = t.standard_sizes[0]
        assert dims["inner_diameter"] == 12.0
        assert dims["outer_diameter"] == 21.0

    def test_script(self):
        t = get_template("linear_bearing_lm12uu")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "LinearBearing" in script


class TestLinearBearingFit:
    """Verify bearing-to-shaft dimension compatibility."""

    def test_lm8uu_fits_8mm_shaft(self):
        t = get_template("linear_bearing_lm8uu")
        shaft = get_template("linear_shaft")
        bearing_bore = t.standard_sizes[0]["inner_diameter"]
        # Find a standard shaft size that matches
        matching = [s for s in shaft.standard_sizes if s["diameter"] == bearing_bore]
        assert len(matching) >= 1, "LM8UU bore should match 8mm shaft"

    def test_lm10uu_fits_10mm_shaft(self):
        t = get_template("linear_bearing_lm10uu")
        shaft = get_template("linear_shaft")
        bearing_bore = t.standard_sizes[0]["inner_diameter"]
        matching = [s for s in shaft.standard_sizes if s["diameter"] == bearing_bore]
        assert len(matching) >= 1, "LM10UU bore should match 10mm shaft"


# =====================================================================
# 4. Linear guide (MGN12H)
# =====================================================================

class TestLinearGuideMGN12H:

    def test_metadata(self):
        t = get_template("linear_guide_mgn12h")
        assert t.category == "bearing"
        assert t.subcategory == "linear_bearing"
        assert "MGN12" in t.tags[1] or "MGN12" in " ".join(t.tags)

    def test_standard_sizes(self):
        t = get_template("linear_guide_mgn12h")
        assert len(t.standard_sizes) >= 3  # 200mm, 300mm, 500mm

    def test_rail_length_variants(self):
        t = get_template("linear_guide_mgn12h")
        lengths = {s["rail_length"] for s in t.standard_sizes}
        assert 200 in lengths
        assert 300 in lengths
        assert 500 in lengths

    def test_script_generation(self):
        t = get_template("linear_guide_mgn12h")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Rail" in script
        assert "makeBox" in script

    def test_mounting_holes_in_script(self):
        t = get_template("linear_guide_mgn12h")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "hole" in script.lower()


# =====================================================================
# 5. T8 Leadscrew & Nut
# =====================================================================

class TestT8Leadscrew:

    def test_metadata(self):
        t = get_template("t8_leadscrew")
        assert t.category == "shaft"
        assert t.subcategory == "leadscrew"
        assert "丝杠" in " ".join(t.tags)

    def test_lead_variants(self):
        t = get_template("t8_leadscrew")
        leads = {s["lead"] for s in t.standard_sizes}
        assert 2.0 in leads
        assert 4.0 in leads
        assert 8.0 in leads

    def test_simplified_script(self):
        t = get_template("t8_leadscrew")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Leadscrew" in script
        assert "makeCylinder" in script

    def test_realistic_script(self):
        t = get_template("t8_leadscrew")
        params = resolve_parameters(t, {"leadscrew_detail": "realistic"})
        script = format_fc_script(t, params)
        assert "Helix" in script or "makeHelix" in script or "realistic" in script

    def test_quality_levels(self):
        t = get_template("t8_leadscrew")
        assert "simplified" in t.quality_levels
        assert "realistic" in t.quality_levels


class TestT8Nut:

    def test_metadata(self):
        t = get_template("t8_nut")
        assert t.category == "shaft"
        assert t.subcategory == "leadscrew"

    def test_standard_size(self):
        t = get_template("t8_nut")
        s = t.standard_sizes[0]
        assert s["bore_diameter"] == 8.0
        assert s["flange_diameter"] > s["outer_diameter"]  # Flange is wider

    def test_script_has_flange(self):
        t = get_template("t8_nut")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "T8Nut" in script
        assert "flange" in script.lower()


# =====================================================================
# 6. BLDC Motors
# =====================================================================

class TestBLDCMotor5010:

    def test_metadata(self):
        t = get_template("bldc_motor_5010")
        assert t.category == "actuator"
        assert t.subcategory == "bldc"
        assert "无刷" in " ".join(t.tags) or "BLDC" in " ".join(t.tags)

    def test_outrunner_rotor_larger(self):
        """Outrunner: rotor OD > stator OD."""
        t = get_template("bldc_motor_5010")
        s = t.standard_sizes[0]
        assert s["rotor_outer_diameter"] > s["stator_outer_diameter"]

    def test_script_generation(self):
        t = get_template("bldc_motor_5010")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Stator" in script
        assert "Rotor" in script
        assert "Shaft" in script


class TestBLDCMotor2208:

    def test_metadata(self):
        t = get_template("bldc_motor_2208")
        assert t.category == "actuator"
        assert t.subcategory == "bldc"

    def test_smaller_than_5010(self):
        t2208 = get_template("bldc_motor_2208")
        t5010 = get_template("bldc_motor_5010")
        rotor_2208 = t2208.standard_sizes[0]["rotor_outer_diameter"]
        rotor_5010 = t5010.standard_sizes[0]["rotor_outer_diameter"]
        assert rotor_2208 < rotor_5010

    def test_script_generation(self):
        t = get_template("bldc_motor_2208")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Stator" in script
        assert "Rotor" in script


# =====================================================================
# 7. Compression spring
# =====================================================================

class TestCompressionSpring:

    def test_metadata(self):
        t = get_template("compression_spring")
        assert "弹簧" in " ".join(t.tags)
        assert "compression" in " ".join(t.tags)

    def test_standard_sizes(self):
        t = get_template("compression_spring")
        assert len(t.standard_sizes) >= 4

    def test_script_generation(self):
        t = get_template("compression_spring")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Spring" in script
        assert "makeHelix" in script or "makeCylinder" in script  # Either helix or fallback

    def test_coil_param(self):
        t = get_template("compression_spring")
        param_names = {p.name for p in t.parameters}
        assert "active_coils" in param_names
        assert "wire_diameter" in param_names


# =====================================================================
# 8. Damper / shock absorber
# =====================================================================

class TestDamper:

    def test_metadata(self):
        t = get_template("damper_shock_absorber")
        assert "阻尼器" in " ".join(t.tags) or "damper" in " ".join(t.tags)

    def test_standard_sizes(self):
        t = get_template("damper_shock_absorber")
        assert len(t.standard_sizes) >= 2

    def test_script_generation(self):
        t = get_template("damper_shock_absorber")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "Cylinder" in script
        assert "Rod" in script
        assert "Mount" in script

    def test_mount_holes_in_script(self):
        t = get_template("damper_shock_absorber")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        assert "TopMount" in script
        assert "BottomMount" in script


# =====================================================================
# 9. Search and compatibility
# =====================================================================

class TestSearchAndCompatibility:

    def test_search_linear_bearing(self):
        results = search_parts(query="linear bearing")
        ids = {t.id for t in results}
        assert "linear_bearing_lm8uu" in ids
        assert "linear_bearing_lm10uu" in ids
        assert "linear_bearing_lm12uu" in ids

    def test_search_leadscrew(self):
        results = search_parts(query="leadscrew")
        ids = {t.id for t in results}
        assert "t8_leadscrew" in ids
        assert "t8_nut" in ids

    def test_search_bldc(self):
        results = search_parts(query="BLDC")
        ids = {t.id for t in results}
        assert "bldc_motor_5010" in ids
        assert "bldc_motor_2208" in ids

    def test_search_spring_chinese(self):
        results = search_parts(query="弹簧")
        ids = {t.id for t in results}
        assert "compression_spring" in ids

    def test_compatible_parts_bearing(self):
        compat = find_compatible_parts("linear_bearing_lm8uu")
        assert len(compat) > 0

    def test_compatible_parts_leadscrew(self):
        compat = find_compatible_parts("t8_leadscrew")
        ids = {t.id for t in compat}
        assert "t8_nut" in ids  # Leadscrew should be compatible with its nut

    def test_linear_motion_subsystem_search(self):
        results = search_by_subsystem("linear_motion")
        assert len(results) >= 10


# =====================================================================
# 10. Parameter resolution & script integration
# =====================================================================

class TestParameterResolution:

    def test_resolve_lm8uu_defaults(self):
        t = get_template("linear_bearing_lm8uu")
        params = resolve_parameters(t)
        assert params["inner_diameter"] == 8.0
        assert params["outer_diameter"] == 15.0

    def test_resolve_t8_leadscrew_custom(self):
        t = get_template("t8_leadscrew")
        params = resolve_parameters(t, {"lead": 2.0, "length": 400})
        assert params["lead"] == 2.0
        assert params["length"] == 400

    def test_resolve_mgn12_defaults(self):
        t = get_template("linear_guide_mgn12h")
        params = resolve_parameters(t)
        assert params["rail_width"] == 12.0
        assert params["carriage_length"] == 40.3

    @pytest.mark.parametrize("pid", EXPECTED_IDS)
    def test_all_standard_sizes_resolve(self, pid):
        t = get_template(pid)
        for size in t.standard_sizes:
            params = resolve_parameters(t, size)
            for key in size:
                assert key in params


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

    def test_realistic_scripts_for_bearings(self):
        for pid in ["linear_bearing_lm8uu", "linear_bearing_lm10uu", "linear_bearing_lm12uu"]:
            t = get_template(pid)
            params = resolve_parameters(t, {"bearing_detail": "realistic"})
            script = format_fc_script(t, params)
            assert "Ball" in script

    def test_realistic_leadscrew_script(self):
        t = get_template("t8_leadscrew")
        params = resolve_parameters(t, {"leadscrew_detail": "realistic"})
        script = format_fc_script(t, params)
        assert "Helix" in script or "makeHelix" in script or "realistic" in script

    def test_spring_helix_or_fallback(self):
        t = get_template("compression_spring")
        params = resolve_parameters(t)
        script = format_fc_script(t, params)
        # Should contain either helix (realistic) or cylinder (fallback)
        assert "makeHelix" in script or "makeCylinder" in script
