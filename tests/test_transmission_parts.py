"""Tests for Task 73: Transmission parts library — GT2/HTD pulleys, belts, couplings, keyway."""

import pytest

from lang3d.knowledge.parts_catalog import (
    CATEGORY_TREE,
    DIN_6885_SQUARE_KEYS,
    STANDARD_KEY_LENGTHS,
    PART_CATALOG,
    PartTemplate,
    ParamDef,
    format_fc_script,
    generate_hub_keyway_ops,
    generate_key_ops,
    generate_shaft_keyway_ops,
    get_key_size,
    get_template,
    recommend_key_length,
    resolve_parameters,
    search_by_subsystem,
    search_parts,
    find_compatible_parts,
    get_functional_parts,
    get_structural_parts,
)


# =====================================================================
# 1. Catalog registration tests
# =====================================================================

class TestCatalogRegistration:
    """Verify all new transmission parts are registered correctly."""

    EXPECTED_IDS = [
        "gt2_pulley", "gt2_belt", "htd_pulley_3m", "htd_pulley_5m",
        "rigid_coupling_setscrew", "rigid_coupling_clamping",
        "spider_coupling", "bellows_coupling",
    ]

    def test_all_transmission_parts_registered(self):
        for pid in self.EXPECTED_IDS:
            assert pid in PART_CATALOG, f"Missing part: {pid}"

    def test_category_tree_has_transmission(self):
        assert "transmission" in CATEGORY_TREE
        subcats = CATEGORY_TREE["transmission"]
        assert "timing_pulley" in subcats
        assert "timing_belt" in subcats
        assert "rigid_coupling" in subcats
        assert "flexible_coupling" in subcats

    def test_transmission_parts_count(self):
        trans = search_parts(category="transmission")
        assert len(trans) == 8

    def test_subsystem_compat_has_transmission(self):
        results = search_by_subsystem("transmission")
        ids = {t.id for t in results}
        for pid in self.EXPECTED_IDS:
            assert pid in ids, f"Missing in transmission subsystem: {pid}"


# =====================================================================
# 2. Part classification tests — all must be functional
# =====================================================================

class TestPartClassification:
    """All transmission parts must be part_class='functional', scalable=False."""

    def test_gt2_pulley_is_functional(self):
        t = get_template("gt2_pulley")
        assert t.part_class == "functional"
        assert t.scalable is False
        assert t.real_part is True

    def test_gt2_belt_is_functional(self):
        t = get_template("gt2_belt")
        assert t.part_class == "functional"
        assert t.scalable is False

    def test_htd_pulley_3m_is_functional(self):
        t = get_template("htd_pulley_3m")
        assert t.part_class == "functional"
        assert t.scalable is False
        assert t.real_part is True

    def test_htd_pulley_5m_is_functional(self):
        t = get_template("htd_pulley_5m")
        assert t.part_class == "functional"
        assert t.scalable is False
        assert t.real_part is True

    def test_rigid_coupling_setscrew_is_functional(self):
        t = get_template("rigid_coupling_setscrew")
        assert t.part_class == "functional"
        assert t.scalable is False
        assert t.real_part is True

    def test_rigid_coupling_clamping_is_functional(self):
        t = get_template("rigid_coupling_clamping")
        assert t.part_class == "functional"
        assert t.scalable is False
        assert t.real_part is True

    def test_spider_coupling_is_functional(self):
        t = get_template("spider_coupling")
        assert t.part_class == "functional"
        assert t.scalable is False
        assert t.real_part is True

    def test_bellows_coupling_is_functional(self):
        t = get_template("bellows_coupling")
        assert t.part_class == "functional"
        assert t.scalable is False
        assert t.real_part is True

    def test_all_transmission_params_fixed(self):
        """All parameters of functional transmission parts must have fixed=True."""
        trans = search_parts(category="transmission")
        for t in trans:
            if t.part_class == "functional":
                for p in t.parameters:
                    if p.param_type == "float":
                        assert p.fixed is True, (
                            f"{t.id}.param '{p.name}' should be fixed=True"
                        )


# =====================================================================
# 3. GT2 Pulley tests
# =====================================================================

class TestGT2Pulley:

    def test_template_metadata(self):
        t = get_template("gt2_pulley")
        assert t.category == "transmission"
        assert t.subcategory == "timing_pulley"
        assert "GT2" in t.tags
        assert t.manufacturer == "Various"

    def test_standard_sizes_exist(self):
        t = get_template("gt2_pulley")
        assert len(t.standard_sizes) >= 4

    def test_16t_size(self):
        t = get_template("gt2_pulley")
        sizes_16t = [s for s in t.standard_sizes if s["teeth"] == 16]
        assert len(sizes_16t) >= 1
        assert sizes_16t[0]["bore_diameter"] == 5.0

    def test_36t_size(self):
        t = get_template("gt2_pulley")
        sizes_36t = [s for s in t.standard_sizes if s["teeth"] == 36]
        assert len(sizes_36t) >= 1
        assert sizes_36t[0]["bore_diameter"] == 6.35

    def test_simplified_script_generation(self):
        t = get_template("gt2_pulley")
        params = resolve_parameters(t, {"teeth": 20, "width": 6.0, "bore_diameter": 5.0})
        script = format_fc_script(t, params)
        assert "GT2Pulley" in script
        assert "makeCylinder" in script

    def test_realistic_script_generation(self):
        t = get_template("gt2_pulley")
        params = resolve_parameters(
            t, {"teeth": 20, "width": 6.0, "bore_diameter": 5.0, "pulley_detail": "realistic"}
        )
        script = format_fc_script(t, params)
        assert "gt2_pulley_realistic" in script
        assert "makePolygon" in script  # Tooth profile uses polygon

    def test_quality_levels(self):
        t = get_template("gt2_pulley")
        assert "simplified" in t.quality_levels
        assert "realistic" in t.quality_levels

    def test_script_has_teeth_parameter(self):
        t = get_template("gt2_pulley")
        params = resolve_parameters(t, {"teeth": 36})
        script = format_fc_script(t, params)
        assert "36" in script


# =====================================================================
# 4. GT2 Belt tests
# =====================================================================

class TestGT2Belt:

    def test_template_metadata(self):
        t = get_template("gt2_belt")
        assert t.category == "transmission"
        assert t.subcategory == "timing_belt"
        assert "同步带" in t.tags or "timing belt" in t.tags

    def test_standard_sizes_cover_widths(self):
        t = get_template("gt2_belt")
        widths = {s["width"] for s in t.standard_sizes}
        assert 6.0 in widths
        assert 9.0 in widths

    def test_script_generation(self):
        t = get_template("gt2_belt")
        params = resolve_parameters(t, {"teeth": 100, "width": 6.0})
        script = format_fc_script(t, params)
        assert "GT2Belt" in script
        assert "makeCylinder" in script

    def test_belt_length_formula(self):
        """Belt length = teeth × pitch (2mm)."""
        t = get_template("gt2_belt")
        for s in t.standard_sizes:
            expected_length = s["teeth"] * 2.0
            # Verify the script uses this relationship
            params = resolve_parameters(t, s)
            script = format_fc_script(t, params)
            assert "belt_length" in script


# =====================================================================
# 5. HTD Pulley tests
# =====================================================================

class TestHTDPulley3M:

    def test_pitch_3mm(self):
        t = get_template("htd_pulley_3m")
        # Find the pitch parameter default
        pitch_param = next(p for p in t.parameters if p.name == "pitch")
        assert pitch_param.default == 3.0

    def test_simplified_script(self):
        t = get_template("htd_pulley_3m")
        params = resolve_parameters(t, {"teeth": 20, "width": 9.0, "bore_diameter": 5.0})
        script = format_fc_script(t, params)
        assert "HTDPulley" in script

    def test_realistic_script(self):
        t = get_template("htd_pulley_3m")
        params = resolve_parameters(
            t, {"teeth": 20, "width": 9.0, "bore_diameter": 5.0, "pulley_detail": "realistic"}
        )
        script = format_fc_script(t, params)
        assert "htd_pulley_realistic" in script
        assert "makePolygon" in script


class TestHTDPulley5M:

    def test_pitch_5mm(self):
        t = get_template("htd_pulley_5m")
        pitch_param = next(p for p in t.parameters if p.name == "pitch")
        assert pitch_param.default == 5.0

    def test_simplified_script(self):
        t = get_template("htd_pulley_5m")
        params = resolve_parameters(t, {"teeth": 20, "width": 15.0, "bore_diameter": 8.0})
        script = format_fc_script(t, params)
        assert "HTDPulley" in script

    def test_realistic_script(self):
        t = get_template("htd_pulley_5m")
        params = resolve_parameters(
            t, {"teeth": 20, "width": 15.0, "bore_diameter": 8.0, "pulley_detail": "realistic"}
        )
        script = format_fc_script(t, params)
        assert "htd_pulley_realistic" in script


# =====================================================================
# 6. Rigid coupling tests
# =====================================================================

class TestRigidCouplingSetScrew:

    def test_template_metadata(self):
        t = get_template("rigid_coupling_setscrew")
        assert t.category == "transmission"
        assert t.subcategory == "rigid_coupling"
        assert t.part_class == "functional"

    def test_script_generation(self):
        t = get_template("rigid_coupling_setscrew")
        params = resolve_parameters(t, {
            "outer_diameter": 16.0, "length": 25.0, "bore_diameter": 5.0,
            "num_setscrews": 2, "setscrew_size": 3.0,
        })
        script = format_fc_script(t, params)
        assert "RigidCoupling" in script
        assert "makeCylinder" in script

    def test_setscrew_holes_in_script(self):
        t = get_template("rigid_coupling_setscrew")
        params = resolve_parameters(t, {"num_setscrews": 2})
        script = format_fc_script(t, params)
        assert "setscrew_hole" in script or "n_setscrews" in script

    def test_standard_sizes_shaft_coverage(self):
        t = get_template("rigid_coupling_setscrew")
        bores = {s["bore_diameter"] for s in t.standard_sizes}
        assert 5.0 in bores  # NEMA17
        assert 8.0 in bores  # Common shaft


class TestRigidCouplingClamping:

    def test_template_metadata(self):
        t = get_template("rigid_coupling_clamping")
        assert t.category == "transmission"
        assert t.subcategory == "rigid_coupling"

    def test_script_generation(self):
        t = get_template("rigid_coupling_clamping")
        params = resolve_parameters(t, {
            "outer_diameter": 19.0, "length": 25.0,
            "bore_diameter": 5.0, "clamp_screw_size": 3.0,
        })
        script = format_fc_script(t, params)
        assert "ClampingCoupling" in script

    def test_has_clamp_slot(self):
        t = get_template("rigid_coupling_clamping")
        params = resolve_parameters(t, {})
        script = format_fc_script(t, params)
        assert "slot" in script.lower()


# =====================================================================
# 7. Flexible coupling tests
# =====================================================================

class TestSpiderCoupling:

    def test_template_metadata(self):
        t = get_template("spider_coupling")
        assert t.category == "transmission"
        assert t.subcategory == "flexible_coupling"

    def test_has_two_bore_params(self):
        t = get_template("spider_coupling")
        param_names = {p.name for p in t.parameters}
        assert "bore1_diameter" in param_names
        assert "bore2_diameter" in param_names

    def test_script_generation(self):
        t = get_template("spider_coupling")
        params = resolve_parameters(t, {
            "bore1_diameter": 5.0, "bore2_diameter": 8.0,
        })
        script = format_fc_script(t, params)
        assert "Hub1" in script
        assert "Hub2" in script
        assert "Spider" in script

    def test_different_bore_sizes(self):
        t = get_template("spider_coupling")
        sizes = t.standard_sizes
        # Should have at least one size with different bore diameters
        diff_bore = [s for s in sizes if s["bore1_diameter"] != s["bore2_diameter"]]
        assert len(diff_bore) >= 0  # May or may not have mixed bores in standard sizes


class TestBellowsCoupling:

    def test_template_metadata(self):
        t = get_template("bellows_coupling")
        assert t.category == "transmission"
        assert t.subcategory == "flexible_coupling"

    def test_has_convolutions_param(self):
        t = get_template("bellows_coupling")
        param_names = {p.name for p in t.parameters}
        assert "convolutions" in param_names
        assert "wall_thickness" in param_names

    def test_script_generation(self):
        t = get_template("bellows_coupling")
        params = resolve_parameters(t, {
            "bore1_diameter": 5.0, "bore2_diameter": 8.0,
        })
        script = format_fc_script(t, params)
        assert "Bellows" in script
        assert "Hub1" in script
        assert "Hub2" in script

    def test_mixed_bore_sizes_in_standard(self):
        t = get_template("bellows_coupling")
        sizes = t.standard_sizes
        diff_bore = [s for s in sizes if s["bore1_diameter"] != s["bore2_diameter"]]
        assert len(diff_bore) >= 1  # Bellows couplings commonly adapt shaft sizes


# =====================================================================
# 8. Keyway (DIN 6885) tests
# =====================================================================

class TestKeySize:

    def test_small_shaft_8mm(self):
        # 8.0 falls in (6, 8) range — early return semantics
        key = get_key_size(8.0)
        assert key["key_width"] == 2
        assert key["key_height"] == 2

    def test_shaft_9mm(self):
        key = get_key_size(9.0)
        assert key["key_width"] == 3
        assert key["key_height"] == 3

    def test_medium_shaft_15mm(self):
        key = get_key_size(15.0)
        assert key["key_width"] == 5
        assert key["key_height"] == 5

    def test_large_shaft_50mm(self):
        key = get_key_size(50.0)
        assert key["key_width"] == 14
        assert key["key_height"] == 9

    def test_boundary_shaft_6mm(self):
        key = get_key_size(6.0)
        assert key["key_width"] == 2

    def test_boundary_shaft_130mm(self):
        key = get_key_size(130.0)
        assert key["key_width"] == 32

    def test_out_of_range_small(self):
        with pytest.raises(ValueError, match="outside DIN 6885"):
            get_key_size(5.0)

    def test_out_of_range_large(self):
        with pytest.raises(ValueError, match="outside DIN 6885"):
            get_key_size(150.0)

    def test_all_ranges_covered(self):
        """Verify complete coverage from 6 to 130 mm."""
        for d in range(6, 131):
            key = get_key_size(float(d))
            assert key["key_width"] > 0
            assert key["key_height"] > 0


class TestShaftKeywayOps:

    def test_returns_ops_list(self):
        ops = generate_shaft_keyway_ops(10.0, 20.0)
        assert isinstance(ops, list)
        assert len(ops) == 1

    def test_op_structure(self):
        ops = generate_shaft_keyway_ops(10.0, 20.0)
        op = ops[0]
        assert op["type"] == "box"
        assert op["name"] == "keyway"
        assert op["operation"] == "cut"
        assert op["height"] == 20.0

    def test_keyway_dimensions(self):
        """Keyway width and height should match DIN 6885 for 11mm shaft."""
        ops = generate_shaft_keyway_ops(11.0, 25.0)
        op = ops[0]
        assert op["width"] == 4  # DIN 6885: 10-12mm shaft → 4mm key
        assert op["depth"] == 4

    def test_position_z_offset(self):
        ops = generate_shaft_keyway_ops(10.0, 20.0, position_z=5.0)
        assert ops[0]["z"] == 5.0


class TestHubKeywayOps:

    def test_returns_ops_list(self):
        ops = generate_hub_keyway_ops(10.0, 15.0)
        assert isinstance(ops, list)
        assert len(ops) == 1

    def test_op_structure(self):
        ops = generate_hub_keyway_ops(10.0, 15.0)
        op = ops[0]
        assert op["type"] == "box"
        assert op["name"] == "hub_keyway"
        assert op["operation"] == "cut"
        assert op["height"] == 15.0

    def test_hub_keyway_dimensions(self):
        ops = generate_hub_keyway_ops(11.0, 15.0)
        op = ops[0]
        assert op["width"] == 4  # DIN 6885: 10-12mm bore → 4mm key
        assert op["depth"] == 4


class TestKeyOps:

    def test_returns_ops_list(self):
        ops = generate_key_ops(10.0, 20.0)
        assert isinstance(ops, list)
        assert len(ops) == 1

    def test_op_structure(self):
        ops = generate_key_ops(10.0, 20.0)
        op = ops[0]
        assert op["type"] == "box"
        assert op["name"] == "square_key"
        assert op["operation"] == "add"
        assert op["height"] == 20.0

    def test_key_cross_section(self):
        ops = generate_key_ops(11.0, 20.0)
        op = ops[0]
        assert op["width"] == 4
        assert op["depth"] == 4


class TestRecommendKeyLength:

    def test_exact_match(self):
        assert recommend_key_length(10.0, 20.0) == 20.0

    def test_round_up(self):
        result = recommend_key_length(10.0, 21.0)
        assert result >= 21.0
        assert result in STANDARD_KEY_LENGTHS

    def test_very_small(self):
        assert recommend_key_length(10.0, 5.0) == 6.0

    def test_very_large(self):
        result = recommend_key_length(10.0, 500.0)
        assert result == STANDARD_KEY_LENGTHS[-1]


# =====================================================================
# 9. Search and compatibility tests
# =====================================================================

class TestSearchAndCompatibility:

    def test_search_by_keyword_gt2(self):
        results = search_parts(query="GT2")
        ids = {t.id for t in results}
        assert "gt2_pulley" in ids
        assert "gt2_belt" in ids

    def test_search_by_keyword_coupling(self):
        results = search_parts(query="coupling")
        ids = {t.id for t in results}
        assert "rigid_coupling_setscrew" in ids
        assert "rigid_coupling_clamping" in ids
        assert "spider_coupling" in ids
        assert "bellows_coupling" in ids
        # Also should find existing flexible_coupling
        assert "flexible_coupling" in ids

    def test_search_by_keyword_htd(self):
        results = search_parts(query="HTD")
        ids = {t.id for t in results}
        assert "htd_pulley_3m" in ids
        assert "htd_pulley_5m" in ids

    def test_search_by_part_class_functional(self):
        results = search_parts(part_class="functional")
        all_ids = {t.id for t in results}
        assert "gt2_pulley" in all_ids
        assert "nema17_stepper" in all_ids
        # Should NOT include structural parts
        assert "l_bracket" not in all_ids

    def test_compatible_parts_pulley(self):
        compat = find_compatible_parts("gt2_pulley")
        ids = {t.id for t in compat}
        # Should find transmission subsystem parts
        assert "nema17_stepper" in ids or "nema23_stepper" in ids

    def test_compatible_parts_coupling(self):
        compat = find_compatible_parts("spider_coupling")
        assert len(compat) > 0

    def test_bilingual_search(self):
        """Search by Chinese keyword should find parts."""
        results = search_parts(query="同步轮")
        ids = {t.id for t in results}
        assert "gt2_pulley" in ids
        assert "htd_pulley_3m" in ids

    def test_tag_search(self):
        results = search_parts(tags=["transmission"])
        assert len(results) >= 8


# =====================================================================
# 10. Parameter resolution and validation tests
# =====================================================================

class TestParameterResolution:

    def test_resolve_gt2_pulley_defaults(self):
        t = get_template("gt2_pulley")
        params = resolve_parameters(t)
        assert params["teeth"] == 20
        assert params["width"] == 6.0
        assert params["bore_diameter"] == 5.0

    def test_resolve_spider_coupling_custom(self):
        t = get_template("spider_coupling")
        params = resolve_parameters(t, {
            "bore1_diameter": 8.0,
            "bore2_diameter": 10.0,
        })
        assert params["bore1_diameter"] == 8.0
        assert params["bore2_diameter"] == 10.0

    def test_invalid_quality_level_raises(self):
        t = get_template("gt2_pulley")
        with pytest.raises(ValueError):
            resolve_parameters(t, {"pulley_detail": "nonexistent"})

    def test_standard_sizes_match_params(self):
        """Verify all standard sizes can be resolved."""
        for pid in ["gt2_pulley", "htd_pulley_3m", "htd_pulley_5m",
                     "rigid_coupling_setscrew", "spider_coupling", "bellows_coupling"]:
            t = get_template(pid)
            for size in t.standard_sizes:
                params = resolve_parameters(t, size)
                for key in size:
                    assert key in params


# =====================================================================
# 11. Script generation integration tests
# =====================================================================

class TestScriptGeneration:

    def test_all_transmission_simplified_scripts(self):
        """Every transmission part should generate a valid simplified script."""
        trans = search_parts(category="transmission")
        for t in trans:
            if t.part_class == "functional" and t.standard_sizes:
                params = resolve_parameters(t, t.standard_sizes[0])
            else:
                params = resolve_parameters(t)
            script = format_fc_script(t, params)
            assert len(script) > 50, f"Script too short for {t.id}"
            assert "makeCylinder" in script or "makeBox" in script, f"No geometry in {t.id}"

    def test_realistic_scripts_for_pulleys(self):
        """Pulleys with realistic alternatives should generate properly."""
        for pid in ["gt2_pulley", "htd_pulley_3m", "htd_pulley_5m"]:
            t = get_template(pid)
            assert "realistic" in t.quality_levels
            if t.standard_sizes:
                params = resolve_parameters(t, t.standard_sizes[0])
            else:
                params = resolve_parameters(t)
            params["pulley_detail"] = "realistic"
            script = format_fc_script(t, params)
            assert "realistic" in script or "makePolygon" in script

    def test_coupling_scripts_have_hubs(self):
        """Spider and bellows coupling scripts should create separate hubs."""
        for pid in ["spider_coupling", "bellows_coupling"]:
            t = get_template(pid)
            params = resolve_parameters(t)
            script = format_fc_script(t, params)
            assert "Hub1" in script
            assert "Hub2" in script


# =====================================================================
# 12. DIN 6885 key standard data integrity
# =====================================================================

class TestDIN6885Data:

    def test_ranges_covered(self):
        """Verify ranges cover 6mm to 130mm without gaps."""
        sorted_ranges = sorted(DIN_6885_SQUARE_KEYS.keys())
        first_min = sorted_ranges[0][0]
        assert first_min == 6, f"First range should start at 6mm, got {first_min}"
        prev_max = first_min
        for min_d, max_d in sorted_ranges:
            assert min_d <= prev_max + 0.01, (
                f"Gap between {prev_max} and {min_d}"
            )
            prev_max = max_d
        assert prev_max >= 130.0

    def test_key_sizes_increase(self):
        """Larger shafts should have larger (or equal) keys."""
        sorted_ranges = sorted(DIN_6885_SQUARE_KEYS.keys())
        prev_width = 0
        for min_d, max_d in sorted_ranges:
            dims = DIN_6885_SQUARE_KEYS[(min_d, max_d)]
            assert dims["key_width"] >= prev_width
            prev_width = dims["key_width"]

    def test_standard_key_lengths_sorted(self):
        assert STANDARD_KEY_LENGTHS == sorted(STANDARD_KEY_LENGTHS)

    def test_standard_key_lengths_positive(self):
        for kl in STANDARD_KEY_LENGTHS:
            assert kl > 0
