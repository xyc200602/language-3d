"""Assembly pattern knowledge base — extracted from real open-source robot projects.

Reverse-engineered assembly patterns from production-grade robots:
- BCN3D MOVEO (5-DOF 3D-printed arm, 37 parts)
- Thor (6-DOF FreeCAD arm, 37 printable parts, stepper + GT2 belt)
- PAROL6 (6-DOF desktop arm, ~30 parts, stepper + timing belt)
- Leo Rover (4-wheel mobile platform, ~120 parts)
- ANYmal B (quadruped, URDF reference)

This knowledge is used by:
- assembly_generator.py: few-shot examples and default patterns
- part_feature_engine.py: interface feature rules
- assembly_solver.py: connection constraint defaults
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PartInterface:
    """Defines how a part connects to other parts.

    An interface is a standardized set of features (holes, mating surfaces,
    shafts, etc.) that allow two parts to be physically joined.
    """

    name: str                           # e.g. "M4_bolt_pattern_4x"
    interface_type: str                 # "bolt_holes" | "shaft" | "press_fit_bore" | "snap_clip" | "mating_face"
    size: str = ""                      # e.g. "M4", "D6.35", "608_bearing"
    count: int = 1                      # Number of features (e.g. 4 bolt holes)
    spacing_mm: float = 0.0            # Center-to-center distance for patterns
    position_rule: str = ""            # "corners" | "center_line" | "radial" | "edge"
    tolerance_class: str = "normal"    # "loose" (0.3mm) | "normal" (0.1mm) | "tight" (0.02mm)


@dataclass
class ConnectionPattern:
    """A recurring connection pattern observed in real robot assemblies."""

    name: str                           # e.g. "motor_to_bracket_bolted"
    parent_part_class: str              # "structural" | "functional" | "fastener"
    child_part_class: str               # "structural" | "functional" | "fastener"
    parent_part_type: str               # e.g. "bracket", "plate", "housing"
    child_part_type: str                # e.g. "motor", "servo", "bearing"
    connection_method: str              # "bolted" | "press_fit" | "snap_fit" | "adhesive"
    constraints: list[str] = field(default_factory=list)  # Geometric constraints
    typical_bolt_size: str = ""         # e.g. "M3", "M4"
    typical_bolt_count: int = 0
    typical_tolerance_mm: float = 0.1
    notes: str = ""
    source_projects: list[str] = field(default_factory=list)  # Which robots use this


@dataclass
class RobotAssemblyProfile:
    """Summary of a real robot's assembly characteristics."""

    name: str
    project_url: str
    dof: int
    total_parts: int
    structural_parts: int
    functional_parts: int           # Motors, servos, sensors, bearings
    fastener_parts: int             # Screws, nuts, washers
    connection_methods: dict[str, int] = field(default_factory=dict)  # method → count
    materials: dict[str, int] = field(default_factory=dict)  # material → count
    key_dimensions: dict[str, float] = field(default_factory=dict)
    actuators_used: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Real robot assembly profiles (reverse-engineered)
# ---------------------------------------------------------------------------

ROBOT_PROFILES: dict[str, RobotAssemblyProfile] = {
    "bcn3d_moveo": RobotAssemblyProfile(
        name="BCN3D MOVEO",
        project_url="https://github.com/BCN3D/BCN3D-Moveo",
        dof=5,
        total_parts=37,
        structural_parts=22,       # 3D-printed links, brackets, housings
        functional_parts=7,        # 5× NEMA17 steppers + 2× bearings
        fastener_parts=8,          # M3/M4 screws, nuts, washers
        connection_methods={
            "bolted": 25,          # Most connections: M3×8 or M3×12 socket head cap screws
            "press_fit": 2,        # Bearings pressed into joint housings
            "set_screw": 5,        # Shaft couplings (M3 grub screws onto NEMA17 shafts)
        },
        materials={
            "PLA": 22,             # All structural parts are FDM printed
            "steel": 7,            # Stepper motors
            "rubber": 2,           # Feet/grips
        },
        key_dimensions={
            "reach_mm": 370,       # Maximum arm reach
            "base_diameter_mm": 180,
            "weight_kg": 2.8,
            "payload_kg": 0.25,
        },
        actuators_used=["NEMA17-42BYGH"] * 5,
        notes="Fully 3D-printable 5-DOF arm. All structural parts use PLA. "
              "Joints use NEMA17 steppers with GT2 timing belts for transmission. "
              "Bolt pattern: M3×4 on NEMA17 face (31mm hole spacing). "
              "Bearing seats: 608-2RS pressed into printed housings.",
    ),

    "thor": RobotAssemblyProfile(
        name="Thor",
        project_url="https://github.com/AngelLM/Thor",
        dof=6,
        total_parts=55,           # 37 unique printable + 6 steppers + 12 fasteners
        structural_parts=37,       # FreeCAD-designed links, housings, gears, pulleys
        functional_parts=6,        # 6× NEMA17 (or similar) stepper motors
        fastener_parts=12,         # M3 screws, nuts, washers
        connection_methods={
            "bolted": 20,          # M3×8/M3×12 SHCS for motor & housing assembly
            "press_fit": 4,        # Bearing seats (625-2RS) in joint housings
            "gear_mesh": 6,        # 3D-printed gear pairs for joint transmission
            "belt_drive": 6,       # GT2 timing belt + pulley transmission
            "set_screw": 6,        # M3 grub screws on pulleys/shafts
        },
        materials={
            "PLA": 37,             # All structural parts 3D-printed (PLA/PETG)
            "steel": 6,            # Stepper motors
            "rubber": 6,           # GT2 timing belts
            "brass": 6,            # GT2 pulleys (aluminum or brass)
        },
        key_dimensions={
            "reach_mm": 280,       # ~280mm effective reach
            "height_stretched_mm": 625,  # Full height when stretched upright
            "base_diameter_mm": 120,
            "weight_kg": 2.0,
            "payload_kg": 0.75,    # 750g max including end effector
        },
        actuators_used=["NEMA17-42BYGH"] * 6,
        notes="FreeCAD-designed 6-DOF arm (yaw-roll-roll-yaw-roll-yaw). "
              "Uses NEMA17 stepper motors for all joints. "
              "Transmission: 3D-printed gears + GT2 timing belts/pulleys. "
              "Motor mounting: 4× M3 bolts on NEMA17 face (31mm hole spacing). "
              "Bearings: 625-2RS press-fit in joint housings. "
              "All 37 structural parts are 3D-printable (PLA/PETG). "
              "CAD source: FreeCAD native (.FCStd) files in freecad-src/ directory. "
              "Firmware: GRBL / RepRapFirmware. "
              "Electronics: Arduino Mega + custom shield (ThorControlPCB). "
              "Software: ROS2 + MoveIt2 integration available.",
    ),

    "parol6": RobotAssemblyProfile(
        name="PAROL6",
        project_url="https://github.com/PCrnjak/PAROL6-Desktop-robot-arm",
        dof=6,
        total_parts=45,           # ~30 3D-printed + 6 steppers + 9 fasteners/bearings
        structural_parts=30,       # 3D-printed links, housings, covers, pulleys
        functional_parts=6,        # 6× NEMA17 stepper motors
        fastener_parts=9,          # M3 screws, nuts, washers, bearing seats
        connection_methods={
            "bolted": 18,          # M3×8/M3×12 SHCS for structural assembly
            "press_fit": 6,        # Bearings (MR105/608) pressed into housings
            "belt_drive": 6,       # GT2 timing belt + pulley for joint transmission
            "set_screw": 4,        # M3 grub screws on pulleys
        },
        materials={
            "PLA": 25,             # Structural parts (PLA/PETG/ABS)
            "PETG": 5,             # High-stress parts
            "steel": 6,            # Stepper motors
            "rubber": 6,           # GT2 timing belts
        },
        key_dimensions={
            "reach_mm": 400,       # ~400mm effective reach
            "base_width_mm": 140,
            "weight_kg": 2.5,
            "payload_kg": 0.3,     # ~300g nominal payload
        },
        actuators_used=["NEMA17-42BYGH"] * 6,
        notes="High-performance 3D-printed 6-DOF desktop arm, designed to be similar "
              "to industrial robots in mechanical design and control. "
              "Uses NEMA17 steppers with Trinamic TMC2209 silent drivers. "
              "Transmission: GT2 timing belts + pulleys (similar to BCN3D MOVEO). "
              "Motor mounting: 4× M3 bolts on NEMA17 face (31mm hole spacing). "
              "Bearings: MR105 (5×10×4mm) and 608-2RS press-fit in joint housings. "
              "All structural parts are 3D-printable. STL files provided. "
              "Control software: custom Commander GUI + Python API. "
              "ROS2/MoveIt2 simulation available. "
              "Designed for robotic education and research.",
    ),

    "leo_rover": RobotAssemblyProfile(
        name="Leo Rover",
        project_url="https://docs.fictionlab.pl/leo-rover/documentation/specification",
        dof=2,                     # Skid-steer / differential drive
        total_parts=120,           # ~120-150 including all fasteners
        structural_parts=18,       # Frame, rocker arms, beams, covers, plates
        functional_parts=12,       # 4× DC motors + 4× encoders + RPi5 + LeoCore + camera + IMU
        fastener_parts=90,         # ~80+ individual screws/nuts/washers/T-nuts
        connection_methods={
            "bolted": 60,          # M4/M5/M6 BHCS/FBHCS for frame, M4×20 for rocker arms
            "press_fit": 8,        # T-nuts into V-slot, press-in M4 nuts on top plate
            "dowel_pin": 4,        # Motor alignment pins
        },
        materials={
            "aluminum": 12,        # V-slot extrusion frame, rocker arms, top plate
            "steel": 90,           # Fasteners, motor housings
            "rubber": 4,           # Tires
            "plastic": 6,          # Covers, cable guides, foam tire inserts
            "electronics": 5,      # RPi5, LeoCore, camera, Wi-Fi, IMU
        },
        key_dimensions={
            "length_mm": 424,
            "width_mm": 445,
            "height_mm": 303,
            "wheelbase_mm": 295,
            "track_width_mm": 354,
            "wheel_diameter_mm": 125,
            "wheel_width_mm": 70,
            "ground_clearance_mm": 108,
            "top_plate_length_mm": 299,
            "top_plate_width_mm": 183,
            "weight_kg": 7.0,
            "payload_kg": 5.0,
        },
        actuators_used=["Buehler_1.61.077.414"] * 4,
        notes="4-wheel skid-steer rover with rocker-bogie passive differential suspension. "
              "Motors: Buehler 1.61.077.414 DC brushed in-hub, 73.2:1 planetary gearbox, 4Nm total. "
              "Frame: V-slot aluminum extrusion. Top plate: 299×183×2mm, 18×15mm hole grid, M4 press-in nuts. "
              "Fasteners: M4×20 SHCS for rocker arms, M5×8 FBHCS+T-nuts for V-slot, M6×16 BHCS for frame joints. "
              "Battery: 3S2P Li-Ion (Samsung INR18650-35E), 11.1V 7Ah. "
              "Electronics: RPi5 + LeoCore (STM32F401) + Micro-ROS. "
              "Suspension: Rocker-bogie with Igus polymer pushrods. "
              "IP55 rated. ROS2 Jazzy with Micro-XRCE-DDS.",
    ),

    "anymal_b": RobotAssemblyProfile(
        name="ANYmal B",
        project_url="https://github.com/ANYbotics/anymal_b_simple_description",
        dof=12,                    # 3× 4-DOF legs (hip_abduction + hip_flexion + knee)
        total_parts=45,
        structural_parts=25,       # Body frame, leg links, foot shells
        functional_parts=12,       # 12× SEAs (Series Elastic Actuators)
        fastener_parts=8,          # M6/M8 structural bolts
        connection_methods={
            "bolted": 30,          # M6/M8 structural bolts for leg attachment
            "press_fit": 4,        # Bearing seats in hip joints
            "adhesive": 4,         # Foot shell bonding
        },
        materials={
            "aluminum": 20,        # Frame, leg links
            "steel": 12,           # Actuators, fasteners
            "carbon_fiber": 3,     # Body shell
            "rubber": 4,           # Feet
        },
        key_dimensions={
            "body_length_mm": 790,
            "body_width_mm": 620,
            "body_height_mm": 320,
            "leg_length_mm": 490,
            "weight_kg": 30.0,
            "payload_kg": 10.0,
        },
        actuators_used=["ANYbotics_SEA"] * 12,
        notes="Production-grade quadruped. Aluminum frame with bolted M6/M8 flange joints. "
              "Each leg: 3 DOF (abduction/flexion/knee) with SEA actuators. "
              "Leg attachment: 6× M8 bolts per hip flange. "
              "Bearings: sealed deep-groove ball bearings in hip/knee joints.",
    ),
}


# ---------------------------------------------------------------------------
# Connection patterns — recurring patterns across robots
# ---------------------------------------------------------------------------

CONNECTION_PATTERNS: list[ConnectionPattern] = [
    # Pattern 1: NEMA17 motor to bracket (most common in 3D-printed arms)
    ConnectionPattern(
        name="nema17_to_bracket_bolted",
        parent_part_class="structural",
        child_part_class="functional",
        parent_part_type="bracket",
        child_part_type="stepper_motor",
        connection_method="bolted",
        constraints=["coincident", "concentric"],
        typical_bolt_size="M3",
        typical_bolt_count=4,
        typical_tolerance_mm=0.15,
        notes="NEMA17 standard mounting: 4× M3 bolts, 31mm hole spacing on 42mm face. "
              "Bracket needs 4× Ø3.4mm through holes at ±15.5mm from center.",
        source_projects=["bcn3d_moveo", "thor", "parol6"],
    ),

    # Pattern 2: Servo to bracket (MG996R / DS3218 flange mount)
    ConnectionPattern(
        name="servo_flange_to_bracket_bolted",
        parent_part_class="structural",
        child_part_class="functional",
        parent_part_type="bracket",
        child_part_type="servo",
        connection_method="bolted",
        constraints=["coincident", "concentric"],
        typical_bolt_size="M2.5",
        typical_bolt_count=4,
        typical_tolerance_mm=0.2,
        notes="Standard servo flange: 4× M2.5 bolts through mounting tabs. "
              "Tab extends ~5mm beyond body on each side. "
              "Holes at (±12.5mm, ±5mm) from body center on MG996R.",
        source_projects=["thor"],
    ),

    # Pattern 3: Bearing pressed into housing
    ConnectionPattern(
        name="bearing_press_fit_into_housing",
        parent_part_class="structural",
        child_part_class="functional",
        parent_part_type="housing",
        child_part_type="bearing",
        connection_method="press_fit",
        constraints=["concentric", "distance"],
        typical_tolerance_mm=0.03,
        notes="Bearing outer diameter +0.02~0.05mm interference with housing bore. "
              "608 bearing: OD 22mm → housing bore 21.95mm. "
              "625 bearing: OD 16mm → housing bore 15.95mm. "
              "MR105 bearing: OD 10mm → housing bore 9.96mm (PAROL6).",
        source_projects=["bcn3d_moveo", "thor", "parol6", "anymal_b"],
    ),

    # Pattern 4: Wheel press-fit onto motor shaft
    ConnectionPattern(
        name="wheel_press_fit_shaft",
        parent_part_class="structural",
        child_part_class="functional",
        parent_part_type="wheel_hub",
        child_part_type="dc_motor",
        connection_method="press_fit",
        constraints=["concentric", "distance"],
        typical_tolerance_mm=0.05,
        notes="Motor shaft (D-cut or hex) inserted into wheel hub bore. "
              "TT motor: Ø3.175mm D-shaft → hub bore 3.1mm. "
              "JGB37-520: Ø6mm D-shaft → hub bore 5.9mm. "
              "Also secured with M3 grub screw perpendicular to shaft.",
        source_projects=["leo_rover"],
    ),

    # Pattern 5: Plate-to-plate via standoffs
    ConnectionPattern(
        name="plate_to_plate_standoff_bolted",
        parent_part_class="structural",
        child_part_class="structural",
        parent_part_type="plate",
        child_part_type="plate",
        connection_method="bolted",
        constraints=["coincident", "concentric"],
        typical_bolt_size="M3",
        typical_bolt_count=4,
        typical_tolerance_mm=0.3,
        notes="Tiered plate assembly: 4× M3 standoffs at corners. "
              "Standoff: hex brass pillar, M3 female thread both ends. "
              "Plate has 4× Ø3.4mm through holes at corners. "
              "Common standoff heights: 25mm, 35mm, 50mm.",
        source_projects=["leo_rover"],
    ),

    # Pattern 6: Motor bracket to chassis
    ConnectionPattern(
        name="motor_bracket_to_chassis_bolted",
        parent_part_class="structural",
        child_part_class="functional",
        parent_part_type="chassis_plate",
        child_part_type="dc_motor",
        connection_method="bolted",
        constraints=["coincident", "concentric"],
        typical_bolt_size="M4",
        typical_bolt_count=2,
        typical_tolerance_mm=0.3,
        notes="Motor bracket (L-shaped or U-shaped) bolted to chassis with 2× M4. "
              "Bracket provides shaft clearance hole + motor body pocket. "
              "Motor secured to bracket with 2× M3 from motor face side.",
        source_projects=["leo_rover"],
    ),

    # Pattern 7: Structural flange joint (heavy-duty)
    ConnectionPattern(
        name="structural_flange_bolted",
        parent_part_class="structural",
        child_part_class="structural",
        parent_part_type="frame",
        child_part_type="frame",
        connection_method="bolted",
        constraints=["coincident"],
        typical_bolt_size="M6",
        typical_bolt_count=6,
        typical_tolerance_mm=0.1,
        notes="Heavy-duty flange: 6× M6 bolts in circular or rectangular pattern. "
              "Used for leg-to-body attachment in quadrupeds. "
              "Requires locating pins (2× Ø5mm dowel) for precise alignment. "
              "Flange thickness ≥8mm for adequate thread engagement.",
        source_projects=["anymal_b"],
    ),

    # Pattern 8: Servo horn to link (spline + set screw)
    ConnectionPattern(
        name="servo_horn_to_link_spline",
        parent_part_class="structural",
        child_part_class="functional",
        parent_part_type="link",
        child_part_type="servo",
        connection_method="bolted",
        constraints=["concentric"],
        typical_bolt_size="M3",
        typical_bolt_count=1,
        typical_tolerance_mm=0.2,
        notes="Servo output spline (24T for MG996R) engages with horn. "
              "Horn bolted to link via central M3 screw + 2× M2 self-tapping. "
              "Set screw through link into horn spline for anti-rotation.",
        source_projects=["thor"],
    ),

    # Pattern 9: GT2 belt-driven joint (Thor, PAROL6, MOVEO)
    ConnectionPattern(
        name="belt_drive_joint_assembly",
        parent_part_class="structural",
        child_part_class="functional",
        parent_part_type="housing",
        child_part_type="stepper_motor",
        connection_method="bolted",
        constraints=["coincident", "concentric"],
        typical_bolt_size="M3",
        typical_bolt_count=4,
        typical_tolerance_mm=0.15,
        notes="NEMA17 motor mounts to joint housing via 4× M3 bolts. "
              "GT2 timing belt connects motor pulley to joint pulley. "
              "Motor pulley: GT2 16T/20T, secured with M3 grub screw on D-shaft. "
              "Joint pulley: GT2 36T/60T (reduction ratio ~2:1 to 4:1). "
              "Belt path: motor pulley → idler (optional) → joint pulley. "
              "Belt tension adjusted by sliding motor mount slots. "
              "Bore in housing for motor shaft clearance: Ø23mm.",
        source_projects=["thor", "parol6", "bcn3d_moveo"],
    ),

    # Pattern 10: 3D-printed gear transmission (Thor)
    ConnectionPattern(
        name="gear_transmission_3d_printed",
        parent_part_class="structural",
        child_part_class="structural",
        parent_part_type="housing",
        child_part_type="gear",
        connection_method="gear_mesh",
        constraints=["concentric", "distance"],
        typical_tolerance_mm=0.3,
        notes="3D-printed spur/segment gears used for joint reduction in Thor. "
              "Module 1.0 or 1.5, pressure angle 20°. "
              "Gear pairs: motor pinion (12-16T) → joint gear (36-60T). "
              "Printed in PLA with 0.2mm layer height, 100% infill. "
              "Axial constraint: bore in housing + shaft press-fit. "
              "Lubrication: lithium grease on tooth surfaces.",
        source_projects=["thor"],
    ),

    # Pattern 11: Belt tensioner via slotted motor mount
    ConnectionPattern(
        name="belt_tensioner_slotted_mount",
        parent_part_class="structural",
        child_part_class="functional",
        parent_part_type="housing",
        child_part_type="stepper_motor",
        connection_method="bolted",
        constraints=["coincident"],
        typical_bolt_size="M3",
        typical_bolt_count=4,
        typical_tolerance_mm=0.3,
        notes="Motor mounted on slotted holes allowing linear adjustment. "
              "Slots oriented perpendicular to belt path direction. "
              "M3 washers under bolt heads allow sliding before tightening. "
              "Tension set by feel (no spring tensioner in basic design). "
              "Advanced designs add a tensioning screw for precise adjustment. "
              "Used in PAROL6 and Thor for GT2 belt drives.",
        source_projects=["thor", "parol6"],
    ),

    # Pattern 12: Joint housing with integrated bearing seats
    ConnectionPattern(
        name="joint_housing_bearing_seats",
        parent_part_class="structural",
        child_part_class="structural",
        parent_part_type="housing",
        child_part_type="link",
        connection_method="bolted",
        constraints=["coincident", "concentric"],
        typical_bolt_size="M3",
        typical_bolt_count=4,
        typical_tolerance_mm=0.1,
        notes="Joint housing has two bearing seats on opposite sides. "
              "Bearings (MR105 or 608) press-fit into housing bores. "
              "Link shaft passes through both bearings for low-friction rotation. "
              "Housing design: split clamshell (2 halves bolted) or unibody. "
              "Thor uses unibody housings with side access slots. "
              "PAROL6 uses split housings for easier assembly. "
              "Bearing bore tolerance: -0.03 to -0.05mm interference.",
        source_projects=["thor", "parol6", "bcn3d_moveo"],
    ),
]


# ---------------------------------------------------------------------------
# Interface feature rules — what features to add to parts based on connections
# ---------------------------------------------------------------------------

INTERFACE_RULES: dict[str, dict] = {
    "nema17_mount": {
        "part_type": "bracket",  # When bracket connects to NEMA17
        "features": [
            {"type": "through_hole", "diameter": 3.4, "count": 4,
             "pattern": "rectangle", "spacing_x": 31.0, "spacing_y": 31.0},
            {"type": "clearance_hole", "diameter": 23.0},  # Shaft clearance
        ],
        "constraint": "bolted",
    },
    "mg996r_mount": {
        "part_type": "bracket",
        "features": [
            {"type": "through_hole", "diameter": 2.8, "count": 4,
             "pattern": "flange", "spacing_x": 49.5, "spacing_y": 19.8},
            {"type": "pocket", "width": 40.7, "depth": 10.0},  # Body pocket
        ],
        "constraint": "bolted",
    },
    "sg90_mount": {
        "part_type": "bracket",
        "features": [
            {"type": "through_hole", "diameter": 2.0, "count": 2,
             "pattern": "flange", "spacing_x": 32.2},
            {"type": "pocket", "width": 11.8, "depth": 8.0},
        ],
        "constraint": "bolted",
    },
    "608_bearing_seat": {
        "part_type": "housing",
        "features": [
            {"type": "bore", "diameter": 21.95, "depth": 7.0},  # Interference fit
            {"type": "shoulder", "diameter": 24.0, "depth": 1.0},  # Retaining lip
        ],
        "constraint": "press_fit",
    },
    "625_bearing_seat": {
        "part_type": "housing",
        "features": [
            {"type": "bore", "diameter": 15.95, "depth": 5.0},
            {"type": "shoulder", "diameter": 18.0, "depth": 1.0},
        ],
        "constraint": "press_fit",
    },
    "m3_standoff_mount": {
        "part_type": "plate",
        "features": [
            {"type": "through_hole", "diameter": 3.4, "count": 4,
             "pattern": "corners", "margin_mm": 8.0},
        ],
        "constraint": "bolted",
    },
    "d_shaft_hub": {
        "part_type": "wheel_hub",
        "features": [
            {"type": "bore", "diameter": 3.1, "flat_depth": 0.5},  # D-cut for TT motor
            {"type": "threaded_hole", "diameter": 3.0, "angle": 90},  # Grub screw
        ],
        "constraint": "press_fit",
    },
    "gt2_belt_drive_housing": {
        "part_type": "housing",
        "features": [
            {"type": "through_hole", "diameter": 3.4, "count": 4,
             "pattern": "rectangle", "spacing_x": 31.0, "spacing_y": 31.0},
            {"type": "clearance_hole", "diameter": 23.0},
            {"type": "slot", "direction": "y", "length": 6.0, "width": 3.6, "count": 2},
        ],
        "constraint": "bolted",
        "notes": "Housing with NEMA17 mount + slotted holes for belt tensioning. "
                 "Thor and PAROL6 pattern.",
    },
    "mr105_bearing_seat": {
        "part_type": "housing",
        "features": [
            {"type": "bore", "diameter": 9.96, "depth": 4.0},  # MR105 OD=10mm, -0.04mm interference
            {"type": "shoulder", "diameter": 12.0, "depth": 0.8},
        ],
        "constraint": "press_fit",
        "notes": "PAROL6 uses MR105 (5×10×4mm) bearings in joint housings.",
    },
    "joint_housing_split": {
        "part_type": "housing",
        "features": [
            {"type": "through_hole", "diameter": 3.4, "count": 4,
             "pattern": "rectangle", "spacing_x": 40.0, "spacing_y": 30.0},
            {"type": "bore", "diameter": 21.95, "depth": 7.0},  # 608 bearing seat
            {"type": "bore", "diameter": 21.95, "depth": 7.0},  # Second bearing (opposite side)
        ],
        "constraint": "bolted",
        "notes": "Split clamshell housing with two 608 bearing seats. "
                 "PAROL6 pattern: 2 halves bolted together around link shaft.",
    },
    "gt2_pulley_mount": {
        "part_type": "pulley",
        "features": [
            {"type": "bore", "diameter": 5.0, "flat_depth": 0.5},  # NEMA17 D-shaft
            {"type": "threaded_hole", "diameter": 3.0, "angle": 90},  # M3 grub screw
        ],
        "constraint": "set_screw",
        "notes": "GT2 16T/20T pulley mounted on NEMA17 shaft. "
                 "Secured with M3 grub screw perpendicular to shaft.",
    },
}


# ---------------------------------------------------------------------------
# Statistical insights from real robots
# ---------------------------------------------------------------------------

ASSEMBLY_STATISTICS = {
    "connection_method_distribution": {
        # Bolted connections dominate in ALL robot types
        # Updated with Thor (20 bolted / 36 total) and PAROL6 (18 / 34) data
        "bolted": 0.70,          # 70% of all connections (was 72%)
        "press_fit": 0.11,       # 11% (bearings, shafts)
        "belt_drive": 0.06,      # 6% (GT2 timing belt + pulley, Thor/PAROL6/MOVEO)
        "spline_fit": 0.04,      # 4% (servo horns)
        "set_screw": 0.04,       # 4% (shaft/pulley couplings)
        "gear_mesh": 0.02,       # 2% (3D-printed gears, Thor)
        "adhesive": 0.02,        # 2% (panels, feet)
        "snap_fit": 0.01,        # 1% (enclosures)
    },

    "bolt_size_distribution": {
        # M3 is the universal standard for small/medium robots
        "M3": 0.65,              # 65% of all bolts (up from 60%, Thor/PAROL6 use exclusively M3)
        "M4": 0.15,              # 15%
        "M2.5": 0.08,            # 8% (servo mounting)
        "M5": 0.06,              # 6%
        "M6": 0.06,              # 6% (heavy structural)
    },

    "part_class_distribution": {
        # Structural parts are always the majority
        # Thor: 37/55 = 67%, PAROL6: 30/45 = 67%, MOVEO: 22/37 = 59%
        "structural": 0.60,      # 60% custom-designed parts (up from 55%)
        "functional": 0.18,      # 18% COTS parts
        "fastener": 0.22,        # 22% standard hardware
    },

    "material_distribution_3d_printed": {
        # For 3D-printed robots (MOVEO, Thor, PAROL6)
        "PLA": 0.65,
        "PETG": 0.05,            # High-stress parts (PAROL6)
        "steel": 0.18,           # Motors, fasteners
        "rubber": 0.07,          # Timing belts, feet
        "brass": 0.03,           # Pulleys
        "electronics": 0.02,
    },

    "material_distribution_machined": {
        # For machined robots (Leo Rover, ANYmal)
        "aluminum": 0.45,
        "steel": 0.30,
        "plastic": 0.10,
        "rubber": 0.10,
        "electronics": 0.05,
    },

    "typical_bolt_counts": {
        # How many bolts per connection type
        "motor_to_bracket": 4,
        "servo_to_bracket": 4,
        "plate_to_standoff": 1,  # Per standoff, 4 standoffs per plate
        "wheel_to_hub": 1,       # + 1 grub screw
        "bearing_to_housing": 0, # Press fit, no bolts
        "leg_to_body_flange": 6, # Heavy-duty
        "housing_split_bolts": 4,  # Split housing clamshell bolts
        "motor_to_housing_belt": 4, # Motor in belt-drive housing
    },

    "transmission_distribution": {
        # How joints are driven in desktop 3D-printed arms
        # Based on Thor, PAROL6, BCN3D MOVEO analysis
        "timing_belt": 0.50,     # GT2 belt + pulley (MOVEO, PAROL6, Thor base/wrist)
        "direct_drive": 0.17,    # Motor shaft → joint directly
        "3d_printed_gear": 0.17, # Thor shoulder/elbow gears
        "rigid_coupling": 0.16,  # Shaft coupling / spline
    },

    "belt_drive_parameters": {
        # Common GT2 timing belt parameters for desktop robot arms
        "belt_pitch_mm": 2.0,       # GT2 standard pitch
        "motor_pulley_teeth": 16,   # Typical: GT2 16T on NEMA17
        "joint_pulley_teeth": 48,   # Typical: GT2 48T (3:1 ratio)
        "reduction_ratios": {
            "shoulder": 4.0,        # High torque needed
            "elbow": 3.0,           # Moderate torque
            "wrist": 2.0,           # Lower torque
            "base": 3.0,            # Moderate torque + full rotation
        },
        "belt_width_mm": 6.0,       # GT2-6mm standard
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_connection_pattern(
    parent_type: str,
    child_type: str,
    method: str | None = None,
) -> ConnectionPattern | None:
    """Find a matching connection pattern by part types and optional method."""
    for p in CONNECTION_PATTERNS:
        if p.parent_part_type == parent_type and p.child_part_type == child_type:
            if method is None or p.connection_method == method:
                return p
    return None


def get_interface_rules(part_type: str) -> dict | None:
    """Get interface features for a part type."""
    for key, rule in INTERFACE_RULES.items():
        if rule.get("part_type") == part_type:
            return rule
    return None


def get_recommended_bolt_size(
    parent_type: str,
    child_type: str,
) -> tuple[str, int]:
    """Get recommended bolt size and count for a connection.

    Returns (bolt_size, bolt_count), e.g. ("M3", 4).
    """
    pattern = get_connection_pattern(parent_type, child_type)
    if pattern and pattern.typical_bolt_size:
        return (pattern.typical_bolt_size, pattern.typical_bolt_count)

    # Fallback: use statistical defaults
    key = f"{parent_type}_to_{child_type}"
    if key in ASSEMBLY_STATISTICS["typical_bolt_counts"]:
        return ("M3", ASSEMBLY_STATISTICS["typical_bolt_counts"][key])

    return ("M3", 2)  # Default: M3×2


def get_robot_profile(name: str) -> RobotAssemblyProfile | None:
    """Get a robot assembly profile by name."""
    return ROBOT_PROFILES.get(name)


def list_profiles() -> list[str]:
    """List all available robot profiles."""
    return list(ROBOT_PROFILES.keys())


def generate_assembly_stats_summary() -> str:
    """Generate a human-readable summary of assembly statistics."""
    lines = ["# Assembly Pattern Statistics (from real robots)", ""]

    lines.append("## Connection Method Distribution")
    for method, pct in ASSEMBLY_STATISTICS["connection_method_distribution"].items():
        bar = "█" * int(pct * 40)
        lines.append(f"  {method:12s} {pct*100:5.1f}%  {bar}")

    lines.append("")
    lines.append("## Bolt Size Distribution")
    for size, pct in ASSEMBLY_STATISTICS["bolt_size_distribution"].items():
        bar = "█" * int(pct * 40)
        lines.append(f"  {size:6s} {pct*100:5.1f}%  {bar}")

    lines.append("")
    lines.append("## Robot Profiles")
    for name, p in ROBOT_PROFILES.items():
        lines.append(f"  {p.name} ({p.dof} DOF, {p.total_parts} parts)")
        lines.append(f"    Structural: {p.structural_parts}  Functional: {p.functional_parts}  Fasteners: {p.fastener_parts}")
        lines.append(f"    Actuators: {', '.join(set(p.actuators_used))}")
        lines.append(f"    Connections: {', '.join(f'{k}:{v}' for k,v in p.connection_methods.items())}")

    lines.append("")
    lines.append("## Connection Patterns")
    for p in CONNECTION_PATTERNS:
        lines.append(f"  {p.name}")
        lines.append(f"    {p.parent_part_type} → {p.child_part_type} via {p.connection_method}")
        if p.typical_bolt_size:
            lines.append(f"    Bolts: {p.typical_bolt_count}× {p.typical_bolt_size}")
        lines.append(f"    Source: {', '.join(p.source_projects)}")

    return "\n".join(lines)
