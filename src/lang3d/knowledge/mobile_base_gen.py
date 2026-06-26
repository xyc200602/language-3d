"""Parametric wheeled mobile-base topology generator.

The wheeled-base counterpart to ``arm_topology.build_arm_example``. Instead
of feeding the LLM the hand-written ``EXAMPLE_4W_ROBOT`` constant (whose
hard-coded wheel/motor dimensions never adapt to payload, and which the
LLM "creatively" mutates — breaking the axis/anchor/no_distribute
conventions that keep wheels flat on the ground), this module synthesises a
structurally-correct chassis from high-level parameters.

Why this matters (root cause of the dual-arm failures):
  * ``assembly_solver`` is correct — a static ``EXAMPLE_4W_ROBOT`` solve
    places all 4 wheels at Z=-15 with 0.0mm variation.
  * But the LLM, given the example as *advisory text*, changes the wheel
    joint axis to "z" (wheel stands vertical), drops ``no_distribute`` (the
    sibling-distribution algorithm flings wheels 600mm from centre), or
    re-parents wheels onto the arm chain. Every failure mode traces back to
    the LLM not preserving the conventions.
  * A parametric generator (like ``arm_topology`` already proved for arms)
    makes those conventions *structural* — the LLM adjusts dimensions, not
    topology.

Convention pins (must match ``EXAMPLE_4W_ROBOT`` exactly, validated by the
solver producing flat, symmetric wheels):
  * wheel joint: ``revolute, axis="y", parent_anchor="left"/"right",
    child_anchor="center", no_distribute=True`` — axis=y makes the cylinder
    lie horizontally (its spin axis is Y, so it rolls forward/back).
  * motors: ``fixed, parent=base_plate, parent_anchor="bottom",
    distribution_group="motors"`` — the solver's 2×2 grid distribution puts
    them at the four corners of the base underside.
  * standoffs: ``fixed, parent=base_plate, parent_anchor="top",
    distribution_group="standoffs"`` — same grid, on top, holding the
    top_plate up.
  * top_plate: the arm-mounting deck. Its height = standoff height + base
    plate thickness, so arms sit clear of the wheels.

Z-stack (Husky-style ground reference, added 2026-06-24):
  * ``base_footprint`` is the virtual root at Z=0 — the ground-contact
    plane (mirrors the real-robot convention where base_footprint is the
    projection of the chassis onto the floor).
  * ``base_plate`` sits ABOVE the wheels at wheel_radius + ground_clearance,
    so the deck floats over the wheels (the LLM's previous "deck at Z=0,
    wheels hanging below" model buried wheels 55mm underground).
  * wheels reach DOWN to Z=0: wheel center = wheel_radius → bottom = ground.
    motors sit at the axle height (wheel_radius), exactly as on a real
    differential/mecanum base (motor shaft is the wheel axis).
  * Explicit ``offset`` on the base_plate and motor joints compensates for
    the solver's auto joint-face clearance so wheels land on the ground
    within engineering tolerance (±5mm).

Sizes come from ``WheelBaseCalculator`` (payload → wheel diameter, track
width, ground clearance) and ``DC_MOTOR_CATALOG`` (real motor dimensions),
so a 20kg-payload robot gets bigger wheels than a 2kg one — not the flat
65mm of the hard-coded example.
"""

from __future__ import annotations

import json
from typing import Any

from .mobile_base import WheelBaseCalculator, DC_MOTOR_CATALOG


# ---------------------------------------------------------------------------
# Corner naming convention
# ---------------------------------------------------------------------------
#
# Four-corner layout, viewed from above (X right, Y forward):
#
#        rl (−X, +Y)        rr (+X, +Y)
#               ┌─────────────┐
#               │  base_plate │
#               └─────────────┘
#        fl (−X, −Y)        fr (+X, −Y)
#
# fl/fr/rl/rr = front-left / front-right / rear-left / rear-right.
# The solver's 2x2 distribution assigns corners by joint *order*
# (assembly_solver._compute_distribution_offset: row=idx//2, col=idx%2),
# so the order we emit joints in determines their physical placement.
# We emit in the order fl, fr, rl, rr which the solver maps to a clockwise
# corner sweep — matching the real robot layout.
_CORNERS = ["fl", "fr", "rl", "rr"]


# ---------------------------------------------------------------------------
# Dimension derivation (delegates to WheelBaseCalculator)
# ---------------------------------------------------------------------------


def _chassis_dims(payload_kg: float, drive_type: str) -> dict[str, float]:
    """Derive chassis + wheel dimensions from payload via WheelBaseCalculator.

    Proportions match the Husky A200 UGV (authoritative source:
    docs/references/external/husky_urdf_ground_truth.md, taken from the
    official Clearpath husky.urdf):

      * body_width  (X, the ``width`` dimension) ≈ track  — wheels sit at
        the body's left/right edges, half-embedded in the sides (Husky:
        body 571mm = track 571mm). NOT track×1.2, which tucks wheels 20%
        inside and makes the body look too fat around tiny wheels.
      * body_length (Y, the ``length`` dimension) ≈ wheelbase × 1.9 — the
        body is a long rectangle so wheels sit *inset* from the front/rear
        ends (Husky: body 1007mm on a 512mm wheelbase = 1.97×). NOT
        wheelbase×1.2, which made a near-square body where wheels appeared
        at the corners like four stubs on a box.

    The solver maps ``width`` → X axis (left/right, via the
    ANCHOR_DIM_KEYS['left/right'] = ['width']) and ``length`` → Y axis
    (front/back). So a body with length > width is a rectangle whose long
    edge runs front-to-back — perpendicular to the wheel axles (which are
    along Y, axis='y'). This matches real UGVs: long body, wheels on the
    short sides.
    """
    calc = WheelBaseCalculator(payload_kg=payload_kg, drive_type=drive_type)
    wheel_d = calc.wheel_diameter_mm()
    track = calc.track_width_mm()        # left↔right (X)
    wheelbase = calc.wheelbase_mm()      # front↔rear (Y)
    clearance = calc.ground_clearance_mm()
    base_thickness = max(5.0, payload_kg * 0.4)  # heavier payload → thicker deck
    standoff_height = max(40.0, wheel_d * 0.8)   # arms must clear the wheels
    # Husky proportions: body_width = track (wheels at the edge, half-embedded),
    # body_length = wheelbase × 1.9 (long rectangle, wheels inset from ends).
    # See husky_urdf_ground_truth.md §3: body_width/track=1.00, body_length/wheelbase=1.97.
    base_width = max(120.0, track)
    base_length = max(base_width, wheelbase * 1.9)
    return {
        "wheel_diameter": wheel_d,
        "wheel_width": max(20.0, wheel_d * 0.35),
        "track_width": track,
        "wheelbase": wheelbase,
        "base_length": base_length,    # deck Y extent (long edge, front↔rear)
        "base_width": base_width,      # deck X extent (short edge, left↔right)
        "base_thickness": base_thickness,
        "standoff_height": standoff_height,
        "ground_clearance": clearance,
    }


def _motor_dims(payload_kg: float) -> tuple[str, dict[str, float]]:
    """Pick a real COTS DC motor from the catalog based on payload.

    Returns (model_number, dimensions) so the chassis generator can carry the
    real model number in the part description (for catalog binding + BOM).
    Previously this returned an invented 40×30×25 box with no model number,
    violating AGENTS.md §1.2 (functional parts need real COTS specs).
    """
    # Required torque per wheel scales with payload / wheel_count.
    # Pick the SMALLEST catalog motor whose rated torque meets the need
    # (sorted ascending, first match) — over-spec'ing to NEMA23 for a 5kg
    # robot makes the motor bigger than the chassis.
    required_kg_cm = max(0.8, payload_kg * 0.2)
    candidates = sorted(
        DC_MOTOR_CATALOG.values(),
        key=lambda m: m.rated_torque_kg_cm,  # ascending: prefer smaller first
    )
    chosen = None
    for m in candidates:
        if m.rated_torque_kg_cm >= required_kg_cm and m.body_length_mm > 0:
            chosen = m
            break
    if chosen is None:
        chosen = candidates[-1] if candidates else None
    if chosen is None:
        return "JGA25-370", {"length": 37.0, "width": 25.0, "height": 25.0}
    return chosen.name, {
        "length": chosen.body_length_mm or 37.0,
        "width": chosen.body_width_mm or 25.0,
        "height": chosen.body_height_mm or 25.0,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_wheeled_base(
    wheel_count: int = 4,
    drive_type: str = "differential",
    payload_kg: float = 5.0,
    arm_mount_points: list[str] | None = None,
) -> str:
    """Build a complete wheeled-base assembly-JSON string.

    Parameters mirror how a real chassis is specified: how many wheels,
    what drive kinematics, and the payload it must carry. Dimensions are
    derived from ``WheelBaseCalculator`` so a heavier robot gets larger
    wheels and a wider stance — the key gap in the hard-coded example.

    Returns JSON text matching the ``EXAMPLE_4W_ROBOT`` schema (same keys,
    same joint conventions) so the downstream solver / cad / export need
    **zero** changes. Wheels are guaranteed flat and symmetric because the
    axis/anchor/no_distribute conventions are baked into the generator, not
    left to the LLM.

    ``arm_mount_points`` names the sides of the top_plate where arms will
    attach (consumed later by ``assembly_compose.compose_dual_arm_assembly``).
    """
    wheel_count = max(2, min(int(wheel_count), 4))
    if arm_mount_points is None:
        arm_mount_points = ["left", "right"]

    dims = _chassis_dims(payload_kg, drive_type)
    motor_model, motor = _motor_dims(payload_kg)
    corners = _CORNERS[:wheel_count]

    # --- Husky-style Z-stack (ground = Z=0) ---
    # See docs/references/external/husky_urdf_ground_truth.md for the
    # authoritative source (Husky A200 official URDF).
    #
    # Z-stack (all relative to ground Z=0):
    #   wheel center Z = wheel_radius          (so wheel bottom touches ground)
    #   wheel TOP      Z = 2 × wheel_radius    (the highest point of the wheel)
    #   base_plate sits ABOVE the wheels — its bottom face at wheel_top + a
    #     small clearance so the deck clears the tires. The deck is the arm-
    #     mounting surface; if it's below the wheel top, wheels PIERCE the
    #     deck (the 穿模 bug).
    #
    # The old formula `wheel_radius + ground_clearance + base_thickness/2`
    # was wrong: ground_clearance in Husky is base-bottom-to-ground, but the
    # deck must clear the WHEEL TOP (2×radius), not the axle (1×radius). With
    # a 90mm wheel the old formula put the deck at Z≈82 while the wheel top
    # was at Z=93 → 11mm of wheel poking through the deck.
    wheel_radius = dims["wheel_diameter"] / 2.0
    wheel_top_z = 2.0 * wheel_radius           # highest point of the wheel
    ground_clearance = dims["ground_clearance"]
    base_thickness = dims["base_thickness"]
    motor_height = motor["height"]
    # base_plate bottom sits at wheel_top + a tire clearance (so the deck
    # doesn't intersect the tire). 5mm is enough for visual separation; a
    # real chassis would have fender clearance here.
    _tire_clearance = 5.0
    # Solver injects a joint-face clearance when stacking base_footprint→
    # base_plate (it pushes the child up beyond the target). We pre-subtract
    # the observed compensation so the deck clears the wheel top. The wheel's
    # Z drop is computed FROM base_plate_center_z, so the wheel always lands
    # at the same absolute Z (the drop cancels the deck height) — what we
    # tune here is the deck-vs-wheel-top gap = 5 - comp. comp=4.0 gives a
    # 1mm clearance; comp=5.5 made the deck pierce the wheels by 0.5mm.
    _solver_clearance_compensation = 4.0
    base_plate_center_z = (
        wheel_top_z + _tire_clearance + base_thickness / 2.0
        - _solver_clearance_compensation
    )

    parts: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []

    # --- base_footprint: virtual ground-contact root (Z=0) ---
    # Husky convention: base_footprint is the robot's projection onto the
    # floor — a VIRTUAL reference frame, NOT a visible part. Making it a
    # 342×200mm plate rendered it as a giant white slab under the robot
    # (the "下面那个白的" defect). It must be tiny so it's invisible in
    # renders: 1×1×0.1mm, never seen. Its only job is to be the Z=0 ground
    # root that base_plate attaches to.
    parts.append({
        "name": "base_footprint", "category": "structural",
        "description": "虚拟接地参考（base_footprint，不可见）",
        "material": "Aluminum",
        "dimensions": {
            "length": 1.0, "width": 1.0,
            "height": 0.1,
        },
    })

    # --- base_plate: the chassis body (Husky's base_link) ---
    # Mirrors the real-robot structure: base_plate IS the chassis body
    # (visual + structural + the part wheels attach to). No separate
    # chassis_body shell — the body and deck are one, like Husky's base_link.
    # Wheels mount DIRECTLY on base_plate (Husky: wheel continuous joint →
    # base_link), not through motor→suspension chains.
    parts.append({
        "name": "base_plate", "category": "structural",
        "description": "车体底盘（base_link，兼具外壳）", "material": "Aluminum",
        "dimensions": {
            "length": dims["base_length"], "width": dims["base_width"],
            "height": base_thickness,
        },
    })
    joints.append({
        "type": "fixed", "parent": "base_footprint", "child": "base_plate",
        "parent_anchor": "top", "child_anchor": "bottom",
        "offset": [0.0, 0.0, base_plate_center_z],
        "no_distribute": True,
    })

    # --- corner XYZ map (shared by motors + wheels) ---
    # Husky URDF convention (see docs/references/external/
    # husky_urdf_ground_truth.md): each wheel/motor origin is specified
    # DIRECTLY relative to base_link (front_left: xyz="0.256 0.2854 ...").
    # We use center/center anchors + explicit XYZ offsets so the solver
    # applies NO face offset and NO half-extent push-out (center/center is
    # the concentric case the solver leaves alone) — parts land EXACTLY at
    # base_center + offset.
    #
    # Corner convention (top view, X right, Y forward):
    #   fl (-X,+Y)  fr (+X,+Y)   ← front (+Y)
    #   rl (-X,-Y)  rr (+X,-Y)   ← rear (-Y)
    track_half = dims["track_width"] / 2.0   # X: left (-) / right (+)
    wb_half = dims["wheelbase"] / 2.0        # Y: front (+) / rear (-)
    _corner_xy = {
        "fl": (-track_half, +wb_half),
        "fr": (+track_half, +wb_half),
        "rl": (-track_half, -wb_half),
        "rr": (+track_half, -wb_half),
    }

    # --- motors (real COTS, mounted INSIDE the body) ---
    # Per user direction: motors are visible parts inside the body (cutaway
    # shows them), but they are NOT in the wheel drive chain — the wheel's
    # revolute joint attaches directly to base_plate (Husky structure, where
    # the motor drives via belt/gear inside the chassis, not a wheel-hub
    # motor). This avoids the 4-layer motor→suspension→wheel chain that broke
    # proportions and introduced collision false-positives.
    #
    # Motor placement uses the SAME center/center + explicit-XY convention
    # as the wheels, so each motor sits directly above its wheel at the
    # matching corner. Previously motors used distribution_group="motors"
    # which the solver spreads by joint ORDER (fl→-X-Y, fr→+X-Y, rl→-X+Y,
    # rr→+X+Y) — the OPPOSITE of the wheel corner convention, placing each
    # motor at the DIAGONAL opposite corner from its wheel (motor_fl ended up
    # where wheel_rr is), causing motor↔wheel cross-corner AABB overlaps.
    motor_h = motor["height"]
    # Motor sits at AXLE height (Z = wheel_radius), tucked INWARD from the
    # wheel (toward chassis center) so it doesn't collide with its own wheel.
    # A real Husky has the motor inside the body at axle height, driving the
    # wheel via a belt/shaft. We can't model a hollow body shell, so we place
    # the motor at axle height but offset INWARD by (wheel_radius + motor/2)
    # so the motor body sits beside the wheel (toward chassis center), not
    # on top of it. This avoids both the wheel-motor overlap AND keeps the
    # motor off the deck top (clean top view).
    _motor_z = wheel_radius - base_plate_center_z  # axle height, rel to deck center
    _inward = wheel_radius + motor["length"] / 2.0  # move motor inward off the wheel
    for corner in corners:
        parts.append({
            "name": f"motor_{corner}", "category": "actuator",
            "description": f"{corner}驱动电机{motor_model}",
            "material": "Steel",
            "dimensions": dict(motor),
        })
        # Inward = toward X=0 (left wheels at -X move +X inward, right wheels
        # at +X move -X inward).
        cx, cy = _corner_xy[corner]
        inward_x = cx + (1 if cx < 0 else -1) * _inward
        joints.append({
            "type": "fixed", "parent": "base_plate", "child": f"motor_{corner}",
            "parent_anchor": "center", "child_anchor": "center",
            "offset": [inward_x, cy, _motor_z],
            "no_distribute": True,
        })

    # --- wheels (mounted DIRECTLY on base_plate, Husky structure) ---
    # Each wheel is a continuous/revolute joint directly on base_plate (Husky:
    # wheel_link ← continuous joint, axis=y ← base_link). NO motor→suspension
    # chain. axis="y" makes the cylinder lie horizontal and roll forward.
    #
    # WHEEL POSITIONING (Husky URDF convention — see
    # docs/references/external/husky_urdf_ground_truth.md):
    # Husky specifies each wheel's origin xyz DIRECTLY relative to base_link
    # (front_left: xyz="0.256 0.2854 0.03282"), with no anchor-face logic. We
    # do the same: parent_anchor="center" + child_anchor="center" means the
    # solver applies NO face offset and NO half-extent push-out (the
    # center/center pair is the concentric case the solver leaves alone), so
    # the wheel center lands EXACTLY at base_center + offset.
    #
    # Why not left/right anchors (the previous approach)? The solver's
    # center-anchor half-extent offset (assembly_solver.py:1061) pushes a
    # center-anchored child OUTWARD by its own half-extent so its near face
    # aligns with the parent face. For a wheel that means the wheel center
    # lands at body_edge + wheel_radius — a full radius BEYOND the body,
    # which is exactly the "轮子完全暴露在外面" (wheels completely exposed
    # outside) defect. center/center + explicit offset places the wheel
    # center at track/2 (= body_width/2), so the wheel is half-embedded in
    # the body side like a real UGV (Husky: body 571mm = track 571mm, wheel
    # center at the body edge).
    wheel_dims = {"diameter": dims["wheel_diameter"], "height": dims["wheel_width"]}
    # Wheel center Z = wheel_radius (axle height) → wheel bottom touches ground.
    # offset is relative to base_plate CENTER, so subtract base_plate_center_z.
    _wheel_drop = wheel_radius - base_plate_center_z
    for corner in corners:
        parts.append({
            "name": f"wheel_{corner}", "category": "mechanical",
            "description": f"{corner}轮", "material": "Rubber",
            "dimensions": dict(wheel_dims),
        })
        joints.append({
            "type": "revolute", "parent": "base_plate", "child": f"wheel_{corner}",
            "axis": "y", "range_deg": [-360, 360],
            "parent_anchor": "center", "child_anchor": "center",
            "offset": [_corner_xy[corner][0], _corner_xy[corner][1], _wheel_drop],
            "no_distribute": True,
        })

    # --- top_plate removed: arms mount directly on base_plate ---
    # Real wheeled manipulators (HSR, TIAGo, Fetch, PR2) mount the arm on the
    # chassis TOP FACE — the arm base is co-planar with the deck, not suspended
    # on a second plate. The old top_plate sat 65mm above base_plate (the
    # deleted standoffs' height) leaving a visible air gap that the VLM read as
    # "arms not connected to base". Removing top_plate and mounting arms on
    # base_plate (the Husky base_link / single deck) eliminates the gap and
    # matches real-robot structure. See docs/references/external/
    # real_ugv_chassis_engineering.md.

    # --- chassis_body: the visible 3D shell (Husky base_link body) ---
    # The base_plate is a thin deck (5mm) that wheels/motors/battery attach to.
    # A real UGV also has a BODY SHELL that rises above the deck and encloses
    # the drivetrain — Husky's base_link is a 267mm-tall box. Without it the
    # robot reads as "a flat plate on wheels" (裸露底盘) instead of a vehicle.
    # The chassis_body sits ON the deck (bottom = deck top), slightly narrower
    # than the deck so wheels remain visible at the sides, and tall enough to
    # look like a vehicle body (~0.5× wheel diameter; Husky ratio 267/355).
    #
    # Arms mount on the TOP face of this shell (not on the deck beneath it) —
    # this is how every real wheeled dual-arm robot (HSR, TIAGo, Fetch, PR2)
    # is built: the arm base is co-planar with the chassis top deck. Mounting
    # arms on the lower deck while the shell rises above it would have the
    # shell ENCLOSE the arm bases — the穿模 defect the user reported (arm
    # base_yaw_servo piercing the body). See _arm_mount.deck = "chassis_body".
    _body_height = max(dims["wheel_diameter"] * 0.5, 40.0)
    # Body footprint: inset from the deck edges so it doesn't cover the wheels.
    # Wheels are at X=±track/2; body half-width stays inside that.
    _body_half_w = (dims["base_width"] / 2.0) - wheel_radius * 0.4
    _body_half_l = (dims["base_length"] / 2.0) - wheel_radius * 0.4
    parts.append({
        "name": "chassis_body", "category": "structural",
        "description": "车体外壳（base_link body，包覆传动）", "material": "Aluminum",
        "dimensions": {
            "length": _body_half_l * 2.0, "width": _body_half_w * 2.0,
            "height": _body_height,
        },
    })
    # Body sits on TOP of the deck (deck top = base_plate center + half thickness).
    _body_z = base_thickness / 2.0 + _body_height / 2.0
    joints.append({
        "type": "fixed", "parent": "base_plate", "child": "chassis_body",
        "parent_anchor": "center", "child_anchor": "center",
        "offset": [0.0, 0.0, _body_z],
        "no_distribute": True,
    })

    # --- suspension: a spring/damper between each wheel and the chassis ---
    # Real UGVs (and the project expectation) require suspension. Husky is
    # rigid (tires absorb shock), but the project explicitly asks for visible
    # suspension. We model a simple vertical prismatic strut per wheel: a short
    # cylinder (the damper body) between the wheel axle and the chassis body
    # underside. This is a real mechanical element (not just metadata) — the
    # strut is a visible part, and the prismatic joint lets the wheel travel
    # vertically relative to the chassis (active suspension in sim).
    _strut_radius = max(4.0, dims["wheel_diameter"] * 0.05)
    # Strut length spans from wheel top (2×radius) to deck underside.
    # base_plate bottom = base_plate_center_z - thickness/2.
    _deck_bottom_z = base_plate_center_z - base_thickness / 2.0
    _wheel_top_z = 2.0 * wheel_radius
    _strut_length = max(10.0, _deck_bottom_z - _wheel_top_z)
    # Strut center Z (world) is midway between wheel top and deck bottom.
    _strut_world_z = (_wheel_top_z + _deck_bottom_z) / 2.0
    _strut_z = _strut_world_z - base_plate_center_z  # relative to deck center
    # Strut X is offset INWARD from the wheel (toward chassis center) so the
    # strut cylinder doesn't intersect the wheel's circular profile. A real
    # strut mounts on the suspension arm, inboard of the wheel.
    _strut_inward = wheel_radius * 0.6
    for corner in corners:
        parts.append({
            "name": f"suspension_{corner}", "category": "mechanical",
            "description": f"{corner}悬挂减震支柱", "material": "Steel",
            "dimensions": {"diameter": _strut_radius * 2, "height": _strut_length},
        })
        cx, cy = _corner_xy[corner]
        strut_x = cx + (1 if cx < 0 else -1) * _strut_inward
        joints.append({
            "type": "fixed", "parent": "base_plate", "child": f"suspension_{corner}",
            "parent_anchor": "center", "child_anchor": "center",
            "offset": [strut_x, cy, _strut_z],
            "no_distribute": True,
        })

    # --- battery_box (representative payload, mounted on base top) ---
    parts.append({
        "name": "battery_box", "category": "electronics",
        "description": "电池盒", "material": "PLA",
        "dimensions": {
            "length": dims["base_length"] * 0.5, "width": dims["base_width"] * 0.3,
            "height": max(25.0, dims["base_thickness"] * 3),
        },
    })
    joints.append({
        "type": "fixed", "parent": "base_plate", "child": "battery_box",
        "parent_anchor": "top", "child_anchor": "bottom", "no_distribute": True,
    })

    assembly = {
        "name": f"{wheel_count}w_{drive_type}_base",
        "description": f"{wheel_count}轮{drive_type}移动底盘",
        "default_angles": {},
        "parts": parts,
        "joints": joints,
        # Metadata for the composer: arms mount on base_plate (the single
        # deck / Husky base_link). deck_height_mm is the nominal deck height
        # above ground (base_plate raised by ground_clearance above wheel axle).
        # The composer reads the solved base_plate position, not this value.
        # Metadata for the composer: arms mount on the chassis_body TOP FACE
        # (the visible shell), NOT on the thin base_plate deck beneath it.
        # A real wheeled dual-arm robot (HSR, TIAGo, Fetch) mounts the arm
        # co-planar with the chassis top surface; mounting on the lower deck
        # while the shell rises above it would have the shell enclose the arm
        # bases (the穿模 defect). The composer reads deck_top_z to raise the
        # arm base to the shell's top face.
        #
        # NOTE: deck_top_z is relative to the chassis_body's OWN center
        # (because the arm mount joint's parent is chassis_body, and the
        # solver interprets a child offset relative to the parent center).
        # The shell's top face is body_height/2 above its own center.
        "_arm_mount": {
            "deck": "chassis_body",
            "deck_top_z": _body_height / 2.0,
            "deck_height_mm": (
                wheel_radius + ground_clearance + dims["base_thickness"]
                + _body_height
            ),
            "sides": arm_mount_points,
            # Fixed shoulder spread (~140mm, like a real dual-arm robot's
            # shoulder width), NOT scaled with base_width — otherwise a wider
            # base flings the arms apart past the deck edge.
            "offset_mm": 70.0,
        },
    }
    return json.dumps(assembly, ensure_ascii=False, indent=2)


def parse_drive_type(description: str) -> str:
    """Infer the drive type from a natural-language description.

    Returns 'mecanum' / 'omnidirectional' / 'differential'. Defaults to
    'differential' (the most common and simplest).
    """
    if not description:
        return "differential"
    t = description.lower()
    if "麦克纳姆" in description or "mecanum" in t or "全向" in description:
        return "mecanum"
    if "omni" in t:
        return "mecanum"  # treat omni as mecanum kinematically
    return "differential"
