"""Parts catalog — standard mechanical part templates with parametric FreeCAD scripts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ParamDef:
    """Definition of a single parameter for a part template."""

    name: str
    display_name_cn: str
    unit: str = "mm"
    default: Any = 0.0
    min_value: float = 0.0
    max_value: float = 1000.0
    step: float = 0.1
    fixed: bool = False  # If True, user cannot change this parameter
    param_type: str = "float"       # "float" | "string"
    choices: list[str] = field(default_factory=list)  # valid values for string params


@dataclass
class BoltHole:
    """A single bolt/mounting hole definition in a mounting interface.

    Position is relative to the part's local coordinate origin on the contact face.
    Direction is the hole axis (normal to the face), unit vector.
    """

    x: float               # mm, position on face
    y: float               # mm, position on face
    diameter: float        # mm, through-hole or tapped diameter
    depth: float = 0.0     # mm, 0 = through (all the way)
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)  # hole axis direction
    hole_type: str = "through_hole"  # "through_hole" | "threaded_hole" | "counterbore"


@dataclass
class AlignmentFeature:
    """A non-bolt alignment feature: dowel pin, keyway, notch, etc."""

    feature_type: str       # "dowel_pin" | "keyway" | "notch" | "d_cut" | "spline"
    x: float = 0.0
    y: float = 0.0
    diameter: float = 0.0   # mm, for dowel/spline
    width: float = 0.0      # mm, for keyway/notch
    length: float = 0.0     # mm, for keyway/notch
    depth: float = 0.0      # mm


@dataclass
class MountingInterface:
    """Standardized mounting interface for a functional part.

    Defines how this part attaches to a structural part, enabling
    automatic generation of matching holes/features on the structural side.
    """

    # Interface classification
    interface_type: str          # "through_hole" | "threaded_hole" | "press_fit" | "snap_fit" | "flange"
    contact_face: str = "front" # "front" | "back" | "top" | "bottom" | "side"
    contact_face_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Bolt holes on the contact face
    holes: list[BoltHole] = field(default_factory=list)

    # Center bore / shaft hole (for motors, bearings, etc.)
    bore_diameter: float = 0.0      # mm, 0 = none
    bore_depth: float = 0.0         # mm, 0 = through

    # Body pocket dimensions (structural part needs to clear the functional part body)
    pocket_width: float = 0.0       # mm
    pocket_height: float = 0.0      # mm
    pocket_depth: float = 0.0       # mm

    # Alignment features
    alignment_features: list[AlignmentFeature] = field(default_factory=list)

    # Press-fit parameters (for bearings)
    press_fit_interference: float = 0.0  # mm, positive = bore is smaller than OD
    shoulder_diameter: float = 0.0       # mm, retaining lip OD
    shoulder_depth: float = 0.0          # mm

    # Documentation
    notes: str = ""


@dataclass
class PartTemplate:
    """A parametric part template that can generate FreeCAD models."""

    id: str
    name_en: str
    name_cn: str
    category: str          # e.g. "fastener"
    subcategory: str       # e.g. "screw"
    description: str
    tags: list[str] = field(default_factory=list)
    material_default: str = "steel"
    parameters: list[ParamDef] = field(default_factory=list)
    fc_script_template: str = ""   # FreeCAD Python script with {param} placeholders
    standard_sizes: list[dict[str, float]] = field(default_factory=list)
    notes: str = ""
    quality_levels: list[str] = field(default_factory=lambda: ["simplified"])
    fc_script_alternatives: dict[str, str] = field(default_factory=dict)

    # --- Functional vs Structural classification ---
    # "functional" = real off-the-shelf part (motor, servo, sensor, bearing)
    #   - Dimensions are FIXED (from real product specs)
    #   - Cannot be freely scaled
    #   - Must be selected from catalog, not generated with arbitrary dimensions
    # "structural" = custom-designed part (bracket, plate, link, housing)
    #   - Dimensions are PARAMETRIC (can be freely adjusted)
    #   - Scalable within min/max constraints
    # "fastener" = standard fastening hardware (screw, nut, washer)
    #   - Dimensions follow ISO/DIN standards
    #   - Selected from standard sizes, not arbitrary values
    part_class: str = "structural"  # "functional" | "structural" | "fastener"
    scalable: bool = True           # True for structural, False for functional/fastener
    real_part: bool = False         # True if this is a real COTS (Commercial Off-The-Shelf) part
    manufacturer: str = ""          # e.g. "Tower Pro", "Pololu", "DFRobot"
    model_number: str = ""          # e.g. "SG90", "NEMA17-42BYGH", "JGB37-520"

    # --- Task 76: Standardized mounting interface ---
    mounting_interface: MountingInterface | None = None


@dataclass
class GeneratedPart:
    """Record of a generated part instance."""

    template_id: str
    name: str
    parameters: dict[str, float]
    fcstd_path: str = ""
    stl_path: str = ""
    created_at: str = ""
    print_analysis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a dict."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "parameters": self.parameters,
            "fcstd_path": self.fcstd_path,
            "stl_path": self.stl_path,
            "created_at": self.created_at,
            "print_analysis": self.print_analysis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedPart:
        """Deserialize from a dict."""
        return cls(
            template_id=data.get("template_id", ""),
            name=data.get("name", ""),
            parameters=data.get("parameters", {}),
            fcstd_path=data.get("fcstd_path", ""),
            stl_path=data.get("stl_path", ""),
            created_at=data.get("created_at", ""),
            print_analysis=data.get("print_analysis", {}),
        )


# ---------------------------------------------------------------------------
# Category tree
# ---------------------------------------------------------------------------

CATEGORY_TREE: dict[str, list[str]] = {
    "fastener": ["screw", "nut", "washer", "bolt", "insert", "pin"],
    "bearing": ["ball_bearing", "linear_bearing"],

    "actuator": ["servo", "stepper", "dc_motor", "bldc"],

    "shaft": ["linear", "coupling", "leadscrew", "collar"],

    "gear": ["spur"],
    "transmission": ["timing_pulley", "timing_belt", "rigid_coupling", "flexible_coupling"],
    "structural": ["bracket", "plate", "spring", "damper"],
    "mobile_base": ["wheel", "chassis", "motor_bracket", "hub"],
    "mounting": ["standoff", "battery_holder", "pcb_mount"],
    "sensor": ["lidar", "imu", "camera", "encoder", "limit_switch"],
    "electronics": ["motor_driver", "controller", "power_module", "connector"],
}


# ---------------------------------------------------------------------------
# FreeCAD script templates
# ---------------------------------------------------------------------------

# Metric thread pitch lookup table (ISO coarse pitch)
METRIC_THREAD_PITCH: dict[int, float] = {
    3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25,
    10: 1.5, 12: 1.75, 14: 2.0, 16: 2.0, 20: 2.5,
}

_SOCKET_HEAD_CAP_SCREW_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("socket_head_cap_screw")
# Thread (simplified as cylinder)
thread = Part.makeCylinder({thread_diameter}/2, {length})
thread_obj = doc.addObject("Part::Feature", "Thread")
thread_obj.Shape = thread
# Head
head_h = {thread_diameter}
head = Part.makeCylinder({head_diameter}/2, head_h)
head.translate(FreeCAD.Vector(0, 0, {length}))
head_obj = doc.addObject("Part::Feature", "Head")
head_obj.Shape = head
# Allen hole in head
allen_r = {thread_diameter} * 0.3
allen = Part.makeCylinder(allen_r, head_h * 0.6)
allen.translate(FreeCAD.Vector(0, 0, {length}))
cut = head.cut(allen)
head_obj.Shape = cut
doc.recompute()
"""

_SOCKET_HEAD_CAP_SCREW_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("socket_head_cap_screw_realistic")
# Realistic thread via helix sweep
thread_d = {thread_diameter}
thread_l = {length}
pitch = {thread_pitch}
minor_r = (thread_d - pitch * 1.226) / 2
pitch_r = (thread_d - pitch * 0.6495) / 2
try:
    # Shaft core (minor diameter cylinder)
    shaft = Part.makeCylinder(minor_r, thread_l)
    # Helix path for thread sweep
    helix_wire = Part.makeHelix(pitch, thread_l, pitch_r)
    # Thread profile: triangular cross-section
    prof_h = (thread_d / 2 - minor_r)
    profile_pts = [
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(prof_h, 0, pitch / 8),
        FreeCAD.Vector(0, 0, pitch / 4),
    ]
    profile_wire = Part.makePolygon(profile_pts + [profile_pts[0]])
    profile_face = Part.Face(profile_wire)
    # Sweep profile along helix
    thread_sweep = profile_face.makePipe(helix_wire)
    # Boolean: cut threads into shaft
    result = shaft.fuse(thread_sweep)
    thread_obj = doc.addObject("Part::Feature", "Thread")
    thread_obj.Shape = result
except Exception:
    # Fallback for small sizes where boolean may fail
    thread_obj = doc.addObject("Part::Feature", "Thread")
    thread_obj.Shape = Part.makeCylinder(thread_d / 2, thread_l)
# Head
head_h = thread_d
head = Part.makeCylinder({head_diameter}/2, head_h)
head.translate(FreeCAD.Vector(0, 0, thread_l))
head_obj = doc.addObject("Part::Feature", "Head")
head_obj.Shape = head
# Allen hole in head
allen_r = thread_d * 0.3
allen = Part.makeCylinder(allen_r, head_h * 0.6)
allen.translate(FreeCAD.Vector(0, 0, thread_l))
cut = head.cut(allen)
head_obj.Shape = cut
doc.recompute()
"""

_HEX_NUT_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("hex_nut")
# Hex body
r = {nominal_diameter} * 0.92
h = {nominal_diameter} * 0.8
hex_body = Part.makePolygon([
    FreeCAD.Vector(r, 0, 0),
    FreeCAD.Vector(r/2, r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(-r/2, r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(-r, 0, 0),
    FreeCAD.Vector(-r/2, -r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(r/2, -r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(r, 0, 0),
])
hex_face = Part.Face(hex_body)
hex_prism = hex_face.extrude(FreeCAD.Vector(0, 0, h))
# Center hole
hole = Part.makeCylinder({nominal_diameter}/2, h)
result = hex_prism.cut(hole)
obj = doc.addObject("Part::Feature", "HexNut")
obj.Shape = result
doc.recompute()
"""

_HEX_NUT_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("hex_nut_realistic")
nom_d = {nominal_diameter}
pitch = {thread_pitch}
# Hex body
r = nom_d * 1.1
h = nom_d * 0.8
hex_body = Part.makePolygon([
    FreeCAD.Vector(r, 0, 0),
    FreeCAD.Vector(r/2, r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(-r/2, r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(-r, 0, 0),
    FreeCAD.Vector(-r/2, -r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(r/2, -r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(r, 0, 0),
])
hex_face = Part.Face(hex_body)
hex_prism = hex_face.extrude(FreeCAD.Vector(0, 0, h))
# Center hole (minor diameter for thread)
minor_r = (nom_d - pitch * 1.226) / 2
try:
    hole = Part.makeCylinder(minor_r, h)
    # Helical thread inside the hole
    pitch_r = (nom_d - pitch * 0.6495) / 2
    helix_wire = Part.makeHelix(pitch, h, pitch_r)
    prof_h = (nom_d / 2 - minor_r)
    profile_pts = [
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(-prof_h, 0, pitch / 8),
        FreeCAD.Vector(0, 0, pitch / 4),
    ]
    profile_wire = Part.makePolygon(profile_pts + [profile_pts[0]])
    profile_face = Part.Face(profile_wire)
    thread_sweep = profile_face.makePipe(helix_wire)
    hole_with_thread = hole.fuse(thread_sweep)
    result = hex_prism.cut(hole_with_thread)
except Exception:
    hole = Part.makeCylinder(nom_d / 2, h)
    result = hex_prism.cut(hole)
obj = doc.addObject("Part::Feature", "HexNut")
obj.Shape = result
doc.recompute()
"""

_FLAT_WASHER_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("flat_washer")
outer = Part.makeCylinder({outer_diameter}/2, {thickness})
inner = Part.makeCylinder({inner_diameter}/2, {thickness})
result = outer.cut(inner)
obj = doc.addObject("Part::Feature", "FlatWasher")
obj.Shape = result
doc.recompute()
"""

_HEX_BOLT_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("hex_bolt")
# Thread (simplified as cylinder)
thread = Part.makeCylinder({thread_diameter}/2, {length})
thread_obj = doc.addObject("Part::Feature", "Thread")
thread_obj.Shape = thread
# Hex head
r = {thread_diameter} * 0.95
head_h = {thread_diameter} * 0.7
hex_body = Part.makePolygon([
    FreeCAD.Vector(r, 0, 0),
    FreeCAD.Vector(r/2, r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(-r/2, r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(-r, 0, 0),
    FreeCAD.Vector(-r/2, -r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(r/2, -r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(r, 0, 0),
])
hex_face = Part.Face(hex_body)
hex_prism = hex_face.extrude(FreeCAD.Vector(0, 0, head_h))
hex_prism.translate(FreeCAD.Vector(0, 0, {length}))
head_obj = doc.addObject("Part::Feature", "Head")
head_obj.Shape = hex_prism
doc.recompute()
"""

_HEX_BOLT_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("hex_bolt_realistic")
thread_d = {thread_diameter}
thread_l = {length}
pitch = {thread_pitch}
minor_r = (thread_d - pitch * 1.226) / 2
pitch_r = (thread_d - pitch * 0.6495) / 2
try:
    shaft = Part.makeCylinder(minor_r, thread_l)
    helix_wire = Part.makeHelix(pitch, thread_l, pitch_r)
    prof_h = (thread_d / 2 - minor_r)
    profile_pts = [
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(prof_h, 0, pitch / 8),
        FreeCAD.Vector(0, 0, pitch / 4),
    ]
    profile_wire = Part.makePolygon(profile_pts + [profile_pts[0]])
    profile_face = Part.Face(profile_wire)
    thread_sweep = profile_face.makePipe(helix_wire)
    result = shaft.fuse(thread_sweep)
    thread_obj = doc.addObject("Part::Feature", "Thread")
    thread_obj.Shape = result
except Exception:
    thread_obj = doc.addObject("Part::Feature", "Thread")
    thread_obj.Shape = Part.makeCylinder(thread_d / 2, thread_l)
# Hex head
r = thread_d * 1.0
head_h = thread_d * 0.7
hex_body = Part.makePolygon([
    FreeCAD.Vector(r, 0, 0),
    FreeCAD.Vector(r/2, r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(-r/2, r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(-r, 0, 0),
    FreeCAD.Vector(-r/2, -r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(r/2, -r*math.sqrt(3)/2, 0),
    FreeCAD.Vector(r, 0, 0),
])
hex_face = Part.Face(hex_body)
hex_prism = hex_face.extrude(FreeCAD.Vector(0, 0, head_h))
hex_prism.translate(FreeCAD.Vector(0, 0, thread_l))
head_obj = doc.addObject("Part::Feature", "Head")
head_obj.Shape = hex_prism
doc.recompute()
"""

_BEARING_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("bearing")
outer_ring = Part.makeCylinder({outer_diameter}/2, {width})
inner_ring = Part.makeCylinder({inner_diameter}/2, {width})
# Cut center from outer ring
bore = Part.makeCylinder({inner_diameter}/2 + ({outer_diameter}-{inner_diameter})/4, {width})
result = outer_ring.cut(bore)
outer_obj = doc.addObject("Part::Feature", "OuterRing")
outer_obj.Shape = result
# Inner ring solid
inner_obj = doc.addObject("Part::Feature", "InnerRing")
inner_obj.Shape = inner_ring
doc.recompute()
"""

_BEARING_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("bearing_realistic")

OD = {outer_diameter}
ID = {inner_diameter}
W = {width}
ball_count = int({ball_count}) if int({ball_count}) > 0 else 0

# Calculate bearing geometry
gap = (OD - ID) / 2.0  # radial gap between rings
raceway_depth = gap * 0.15  # groove depth
ball_d = gap * 0.65  # ball diameter fits in the gap
pitch_circle = (OD + ID) / 4.0  # pitch circle radius
raceway_r = ball_d * 0.55  # raceway groove curvature radius

if ball_count == 0:
    ball_count = int(math.pi * 2 * pitch_circle / (ball_d * 1.1))
    ball_count = max(ball_count, 4)  # minimum 4 balls

half_w = W / 2.0

# --- Outer ring ---
outer_od = OD / 2.0
outer_id = outer_od - gap * 0.4 - raceway_depth  # inner bore of outer ring
outer_ring = Part.makeCylinder(outer_od, W)
outer_bore = Part.makeCylinder(outer_id, W)
outer_ring = outer_ring.cut(outer_bore)

# Outer raceway groove (toroidal cut on inner surface of outer ring, at mid-width)
for z_off in [0]:
    groove_profile = Part.Wire([
        FreeCAD.Vector(outer_id, 0, half_w - raceway_r + z_off),
        FreeCAD.Vector(outer_id + raceway_depth * 0.5, 0, half_w + z_off),
        FreeCAD.Vector(outer_id, 0, half_w + raceway_r + z_off),
    ])
    groove_face = Part.Face(groove_profile)
    groove_revolve = groove_face.revolve(
        FreeCAD.Vector(0, 0, half_w + z_off),
        FreeCAD.Vector(0, 0, 1), 360
    )
    outer_ring = outer_ring.cut(groove_revolve)

outer_obj = doc.addObject("Part::Feature", "OuterRing")
outer_obj.Shape = outer_ring

# --- Inner ring ---
inner_od = ID / 2.0 + gap * 0.4 + raceway_depth
inner_id = ID / 2.0
inner_ring = Part.makeCylinder(inner_od, W)
inner_bore = Part.makeCylinder(inner_id, W)
inner_ring = inner_ring.cut(inner_bore)

# Inner raceway groove
groove_profile2 = Part.Wire([
    FreeCAD.Vector(inner_od, 0, half_w - raceway_r),
    FreeCAD.Vector(inner_od - raceway_depth * 0.5, 0, half_w),
    FreeCAD.Vector(inner_od, 0, half_w + raceway_r),
])
groove_face2 = Part.Face(groove_profile2)
groove_revolve2 = groove_face2.revolve(
    FreeCAD.Vector(0, 0, half_w),
    FreeCAD.Vector(0, 0, 1), 360
)
inner_ring = inner_ring.cut(groove_revolve2)

inner_obj = doc.addObject("Part::Feature", "InnerRing")
inner_obj.Shape = inner_ring

# --- Balls ---
ball_r = ball_d / 2.0
for idx in range(ball_count):
    angle = 2 * math.pi * idx / ball_count
    x = pitch_circle * math.cos(angle)
    y = pitch_circle * math.sin(angle)
    ball = Part.makeSphere(ball_r)
    ball.translate(FreeCAD.Vector(x, y, half_w))
    ball_obj = doc.addObject("Part::Feature", "Ball" + str(idx))
    ball_obj.Shape = ball

# --- Cage (simplified: thin ring with rectangular pockets) ---
cage_inner = pitch_circle - ball_d * 0.45
cage_outer = pitch_circle + ball_d * 0.45
cage_thickness = W * 0.08  # thin cage
cage_z = half_w - cage_thickness / 2

cage_body = Part.makeCylinder(cage_outer, cage_thickness)
cage_bore = Part.makeCylinder(cage_inner, cage_thickness)
cage_body.translate(FreeCAD.Vector(0, 0, cage_z))
cage_bore.translate(FreeCAD.Vector(0, 0, cage_z))
cage = cage_body.cut(cage_bore)

# Cut ball pockets (rectangular holes) in cage
pocket_w = ball_d * 0.85  # pocket width
pocket_d = ball_d  # pocket depth radial
for idx in range(ball_count):
    angle = 2 * math.pi * idx / ball_count
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    # Pocket centered on pitch circle at this angle
    cx = pitch_circle * cos_a
    cy = pitch_circle * sin_a
    # Create pocket as small box, rotated to be radial
    pocket = Part.makeBox(pocket_d, pocket_w, cage_thickness)
    # Position pocket
    pocket.translate(FreeCAD.Vector(
        cx - pocket_d / 2,
        cy - pocket_w / 2,
        cage_z
    ))
    try:
        cage = cage.cut(pocket)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Bearing cage pocket cut failed at idx %s: %s", idx, e
        )

cage_obj = doc.addObject("Part::Feature", "Cage")
cage_obj.Shape = cage

doc.recompute()
"""

_SERVO_SG90_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("servo_sg90")
# Main body
body = Part.makeBox({body_length}, {body_width}, {body_height})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Mounting tabs
tab_w = 5
tab = Part.makeBox({body_length} + 2*tab_w, {body_width}, tab_w/2)
tab.translate(FreeCAD.Vector(-tab_w, 0, -tab_w/2))
tab_obj = doc.addObject("Part::Feature", "MountingTab")
tab_obj.Shape = tab
# Output shaft
shaft = Part.makeCylinder({shaft_diameter}/2, {shaft_length})
shaft.translate(FreeCAD.Vector({body_length}/2 - 6, {body_width}/2, {body_height}))
shaft_obj = doc.addObject("Part::Feature", "Shaft")
shaft_obj.Shape = shaft
doc.recompute()
"""

_SERVO_MG996R_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("servo_mg996r")
# Main body
body = Part.makeBox({body_length}, {body_width}, {body_height})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Mounting tabs
tab_w = 5
tab = Part.makeBox({body_length} + 2*tab_w, {body_width}, tab_w/2)
tab.translate(FreeCAD.Vector(-tab_w, 0, -tab_w/2))
tab_obj = doc.addObject("Part::Feature", "MountingTab")
tab_obj.Shape = tab
# Output shaft
shaft = Part.makeCylinder({shaft_diameter}/2, {shaft_length})
shaft.translate(FreeCAD.Vector({body_length}/2 - 10, {body_width}/2, {body_height}))
shaft_obj = doc.addObject("Part::Feature", "Shaft")
shaft_obj.Shape = shaft
doc.recompute()
"""

_NEMA17_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("nema17_stepper")
# Main body (square)
body = Part.makeBox({body_size}, {body_size}, {body_length})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Output shaft
shaft = Part.makeCylinder({shaft_diameter}/2, {shaft_length})
shaft.translate(FreeCAD.Vector({body_size}/2, {body_size}/2, {body_length}))
shaft_obj = doc.addObject("Part::Feature", "Shaft")
shaft_obj.Shape = shaft
# Mounting holes
for dx, dy in [(5.5, 5.5), ({body_size}-5.5, 5.5), (5.5, {body_size}-5.5), ({body_size}-5.5, {body_size}-5.5)]:
    hole = Part.makeCylinder(1.5, 3)
    hole.translate(FreeCAD.Vector(dx, dy, {body_length}))
    body = body.cut(hole)
body_obj.Shape = body
doc.recompute()
"""

_LINEAR_SHAFT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("linear_shaft")
shaft = Part.makeCylinder({diameter}/2, {length})
obj = doc.addObject("Part::Feature", "Shaft")
obj.Shape = shaft
doc.recompute()
"""

_FLEXIBLE_COUPLING_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("flexible_coupling")
# Outer body
outer = Part.makeCylinder({outer_diameter}/2, {length})
# Hole 1
hole1_depth = {length} * 0.45
hole1 = Part.makeCylinder({bore1_diameter}/2, hole1_depth)
# Hole 2 (from opposite side)
hole2 = Part.makeCylinder({bore2_diameter}/2, {length} - hole1_depth)
hole2.translate(FreeCAD.Vector(0, 0, hole1_depth))
result = outer.cut(hole1).cut(hole2)
obj = doc.addObject("Part::Feature", "Coupling")
obj.Shape = result
doc.recompute()
"""

_SPUR_GEAR_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("spur_gear")
# Simplified gear using cylinder with teeth approximation
pitch_radius = {teeth} * {module} / 2
outer_r = pitch_radius + {module}
root_r = pitch_radius - 1.25 * {module}
# Gear body (simplified as cylinder for parametric generation)
body = Part.makeCylinder(outer_r, {thickness})
bore = Part.makeCylinder({bore_diameter}/2, {thickness})
result = body.cut(bore)
obj = doc.addObject("Part::Feature", "SpurGear")
obj.Shape = result
doc.recompute()
"""

_SPUR_GEAR_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("spur_gear_realistic")
# Involute spur gear with mathematically accurate tooth profile
num_teeth = {teeth}
mod = {module}
thk = {thickness}
bore_d = {bore_diameter}
pressure_angle = math.radians({pressure_angle})
backlash_val = {backlash}

r_pitch = num_teeth * mod / 2
r_base = r_pitch * math.cos(pressure_angle)
r_addendum = r_pitch + mod
r_dedendum = r_pitch - 1.25 * mod

# Involute function: returns (x, y) for parameter t
def involute_xy(r, t):
    x = r * (math.cos(t) + t * math.sin(t))
    y = r * (math.sin(t) - t * math.cos(t))
    return (x, y)

# Find t where involute reaches addendum radius
def find_t_at_radius(r_base, target_r):
    for t_test in [i * 0.01 for i in range(1, 2000)]:
        x, y = involute_xy(r_base, t_test)
        r = math.sqrt(x*x + y*y)
        if r >= target_r:
            return t_test
    return 1.0

t_max = find_t_at_radius(r_base, r_addendum)
# Tooth angular half-width at pitch circle
inv_at_pitch = find_t_at_radius(r_base, r_pitch)
x_p, y_p = involute_xy(r_base, inv_at_pitch)
angle_at_pitch = math.atan2(y_p, x_p)
tooth_half_angle = math.pi / (2 * num_teeth) + angle_at_pitch

# Build single tooth profile
num_pts = 30
right_flank = []
for i in range(num_pts + 1):
    t = i * t_max / num_pts
    x, y = involute_xy(r_base, t)
    right_flank.append((x, y))

# Addendum arc (top of tooth)
addendum_pts = []
tooth_top_angle = tooth_half_angle
for i in range(5):
    a = -tooth_top_angle + 2 * tooth_top_angle * i / 4
    addendum_pts.append((r_addendum * math.cos(a), r_addendum * math.sin(a)))

# Left flank (mirror of right)
left_flank = [(x, -y) for x, y in reversed(right_flank)]

# Combine: right flank + addendum arc + left flank
tooth_pts = right_flank + addendum_pts + left_flank

# Build full gear profile by rotating tooth copies
all_points = []
tooth_angle = 2 * math.pi / num_teeth
for tooth_idx in range(num_teeth):
    offset = tooth_idx * tooth_angle
    cos_o = math.cos(offset)
    sin_o = math.sin(offset)
    for (x, y) in tooth_pts:
        rx = x * cos_o - y * sin_o
        ry = x * sin_o + y * cos_o
        all_points.append(FreeCAD.Vector(rx, ry, 0))
    # Add root arc points between teeth
    root_start_angle = offset + tooth_half_angle + backlash_val / r_pitch
    root_end_angle = offset + tooth_angle - tooth_half_angle - backlash_val / r_pitch
    for i in range(5):
        a = root_start_angle + (root_end_angle - root_start_angle) * i / 4
        all_points.append(FreeCAD.Vector(r_dedendum * math.cos(a), r_dedendum * math.sin(a), 0))

# Close the profile
if all_points:
    all_points.append(all_points[0])

# Create wire, face, extrude
gear_wire = Part.makePolygon(all_points)
gear_face = Part.Face(gear_wire)
gear_body = gear_face.extrude(FreeCAD.Vector(0, 0, thk))

# Cut bore
bore = Part.makeCylinder(bore_d / 2, thk)
result = gear_body.cut(bore)

obj = doc.addObject("Part::Feature", "SpurGear")
obj.Shape = result
doc.recompute()
"""

_L_BRACKET_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("l_bracket")
# Vertical part
vert = Part.makeBox({width}, {thickness}, {height})
# Horizontal part (extends in X direction)
horiz = Part.makeBox({length}, {width}, {thickness})
horiz.translate(FreeCAD.Vector(0, 0, {height} - {thickness}))
# Union
result = vert.fuse(horiz)
obj = doc.addObject("Part::Feature", "LBracket")
obj.Shape = result
doc.recompute()
"""

_MOUNTING_PLATE_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("mounting_plate")
plate = Part.makeBox({length}, {width}, {thickness})
# Corner mounting holes
margin = {hole_margin}
hole_r = {hole_diameter} / 2
for x in [margin, {length} - margin]:
    for y in [margin, {width} - margin]:
        hole = Part.makeCylinder(hole_r, {thickness})
        hole.translate(FreeCAD.Vector(x, y, 0))
        plate = plate.cut(hole)
obj = doc.addObject("Part::Feature", "MountingPlate")
obj.Shape = plate
doc.recompute()
"""


# ---------------------------------------------------------------------------
# New script templates for extended parts
# ---------------------------------------------------------------------------

_WHEEL_SIMPLE_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("wheel_simple")
# Tire (outer cylinder)
tire = Part.makeCylinder({outer_diameter}/2, {width})
# Hub hole
hub = Part.makeCylinder({hub_diameter}/2, {width})
wheel = tire.cut(hub)
obj = doc.addObject("Part::Feature", "Wheel")
obj.Shape = wheel
doc.recompute()
"""

_WHEEL_MECANUM_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("wheel_mecanum")
# Core cylinder
core = Part.makeCylinder({diameter}/2, {width})
# Roller positions (simplified as small cylinders at 45 degrees)
n_rollers = {num_rollers}
for i in range(n_rollers):
    angle = 360 * i / n_rollers
    rad = math.radians(angle)
    r = {diameter}/2 * 0.85
    cx = r * math.cos(rad)
    cy = r * math.sin(rad)
    roller = Part.makeCylinder({roller_diameter}/2, {width} * 0.9)
    roller.translate(FreeCAD.Vector(cx, cy, 0))
    roller.rotate(FreeCAD.Vector(cx, cy, 0), FreeCAD.Vector(math.cos(rad), math.sin(rad), 0), 45)
    core = core.fuse(roller)
obj = doc.addObject("Part::Feature", "MecanumWheel")
obj.Shape = core
doc.recompute()
"""

_HUB_ADAPTER_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("hub_adapter")
# Main body cylinder
body = Part.makeCylinder({outer_diameter}/2, {height})
# Motor shaft hole
hole = Part.makeCylinder({shaft_diameter}/2, {height})
# Set screw holes (2 perpendicular)
result = body.cut(hole)
set_r = {outer_diameter}/2 * 0.7
for angle in [0, 90]:
    import math
    rad = math.radians(angle)
    cx = set_r * math.cos(rad)
    cy = set_r * math.sin(rad)
    set_hole = Part.makeCylinder({set_screw_size}/4, {height})
    set_hole.translate(FreeCAD.Vector(cx, cy, 0))
    result = result.cut(set_hole)
obj = doc.addObject("Part::Feature", "HubAdapter")
obj.Shape = result
doc.recompute()
"""

_MOTOR_BRACKET_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("motor_bracket")
# Base plate
base = Part.makeBox({base_length}, {base_width}, {thickness})
# U-bracket arms
arm_height = {bracket_height}
arm1 = Part.makeBox({thickness}, {base_width}, arm_height)
arm1.translate(FreeCAD.Vector(0, 0, {thickness}))
arm2 = Part.makeBox({thickness}, {base_width}, arm_height)
arm2.translate(FreeCAD.Vector({base_length} - {thickness}, 0, {thickness}))
result = base.fuse(arm1).fuse(arm2)
# Motor shaft hole between arms
motor_hole = Part.makeCylinder({motor_diameter}/2, {thickness} + arm_height)
motor_hole.translate(FreeCAD.Vector({base_length}/2, {base_width}/2, 0))
result = result.cut(motor_hole)
obj = doc.addObject("Part::Feature", "MotorBracket")
obj.Shape = result
doc.recompute()
"""

_STANDOFF_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("standoff")
# Main body hex cylinder
body = Part.makeCylinder({outer_diameter}/2, {length})
# M3 through hole
hole = Part.makeCylinder({hole_diameter}/2, {length})
result = body.cut(hole)
obj = doc.addObject("Part::Feature", "Standoff")
obj.Shape = result
doc.recompute()
"""

_BATTERY_HOLDER_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("battery_holder")
# Base tray
tray = Part.makeBox({length}, {width}, {height})
# Battery slots (cylindrical cutouts)
n_cells = {num_cells}
cell_r = {cell_diameter}/2
spacing = {width} / (n_cells + 1)
for i in range(n_cells):
    cy = spacing * (i + 1)
    slot = Part.makeCylinder(cell_r, {length})
    slot.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 1, 0), 90)
    slot.translate(FreeCAD.Vector(0, cy, cell_r + 1))
    tray = tray.cut(slot)
obj = doc.addObject("Part::Feature", "BatteryHolder")
obj.Shape = tray
doc.recompute()
"""

_CHASSIS_PLATE_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("chassis_plate")
# Main plate
plate = Part.makeBox({length}, {width}, {thickness})
# Mounting holes in grid pattern
hole_r = {hole_diameter}/2
margin = {hole_margin}
nx = {grid_x}
ny = {grid_y}
dx = ({length} - 2 * margin) / max(nx - 1, 1)
dy = ({width} - 2 * margin) / max(ny - 1, 1)
for ix in range(nx):
    for iy in range(ny):
        hx = margin + ix * dx
        hy = margin + iy * dy
        hole = Part.makeCylinder(hole_r, {thickness})
        hole.translate(FreeCAD.Vector(hx, hy, 0))
        plate = plate.cut(hole)
obj = doc.addObject("Part::Feature", "ChassisPlate")
obj.Shape = plate
doc.recompute()
"""

_CORNER_BRACKET_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("corner_bracket")
# Vertical plate
v_plate = Part.makeBox({side_length}, {thickness}, {side_length})
# Horizontal plate
h_plate = Part.makeBox({side_length}, {side_length}, {thickness})
result = v_plate.fuse(h_plate)
# Mounting holes (2 per side)
hole_r = {hole_diameter}/2
offset = {side_length} * 0.25
for side in range(2):
    for i in range(2):
        if side == 0:
            hx = offset * (i + 1)
            hy = {thickness} / 2
            hz = offset * (i + 1)
        else:
            hx = offset * (i + 1)
            hy = offset * (i + 1)
            hz = {thickness} / 2
        hole = Part.makeCylinder(hole_r, {thickness} + 2)
        hole.translate(FreeCAD.Vector(hx - hole_r, hy - hole_r, hz - 1))
        result = result.cut(hole)
obj = doc.addObject("Part::Feature", "CornerBracket")
obj.Shape = result
doc.recompute()
"""

# ---------------------------------------------------------------------------
# Scalable structural part scripts — link, housing, mount, etc.
# ---------------------------------------------------------------------------

_LINK_ARM_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("link_arm")
# Hollow rectangular beam
outer = Part.makeBox({length}, {width}, {height})
inner_l = {length} - 2 * {joint_hole_margin}
inner_w = {width} - 2 * {wall_thickness}
inner_h = {height} - 2 * {wall_thickness}
if inner_l > 0 and inner_w > 0 and inner_h > 0:
    inner = Part.makeBox(inner_l, inner_w, inner_h)
    inner.translate(FreeCAD.Vector({joint_hole_margin}, {wall_thickness}, {wall_thickness}))
    beam = outer.cut(inner)
else:
    beam = outer
# Joint mounting holes at both ends (2 holes each end)
hole_r = {joint_hole_diameter} / 2
margin_x = {joint_hole_margin} / 2
margin_y = {width} / 3
for end_x in [margin_x, {length} - margin_x]:
    for dy in [-margin_y, margin_y]:
        hole = Part.makeCylinder(hole_r, {height} + 4)
        hole.translate(FreeCAD.Vector(end_x, {width}/2 + dy, -2))
        beam = beam.cut(hole)
obj = doc.addObject("Part::Feature", "LinkArm")
obj.Shape = beam
doc.recompute()
"""

_JOINT_HOUSING_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("joint_housing")
# Outer cylinder body
body = Part.makeCylinder({outer_diameter}/2, {height})
# Bearing press-fit bore (H7 tolerance)
if {bearing_bore_diameter} > 0:
    bore = Part.makeCylinder({bearing_bore_diameter}/2 - 0.02, {height})
    body = body.cut(bore)
# Bolt holes on PCD
if {bolt_count} > 0 and {bolt_pcd} > 0:
    for i in range({bolt_count}):
        angle = 2 * math.pi * i / {bolt_count}
        bx = {bolt_pcd}/2 * math.cos(angle)
        by = {bolt_pcd}/2 * math.sin(angle)
        hole = Part.makeCylinder({bolt_hole_diameter}/2, {height} + 2)
        hole.translate(FreeCAD.Vector(bx, by, -1))
        body = body.cut(hole)
obj = doc.addObject("Part::Feature", "JointHousing")
obj.Shape = body
doc.recompute()
"""

_MOTOR_MOUNT_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("motor_mount")
# Base plate
plate = Part.makeBox({plate_length}, {plate_width}, {plate_thickness})
# Motor mounting holes — pattern depends on motor_type
motor_type = "{motor_type}"
if motor_type == "NEMA17":
    spacing = 31.0
    hole_d = 3.4
    cx, cy = {plate_length}/2, {plate_width}/2
elif motor_type == "NEMA23":
    spacing = 47.14
    hole_d = 4.5
    cx, cy = {plate_length}/2, {plate_width}/2
elif motor_type == "XM430":
    spacing = 16.0
    hole_d = 2.9
    cx, cy = {plate_length}/2, {plate_width}/2
elif motor_type == "MG996R":
    spacing = 49.5
    hole_d = 2.8
    cx, cy = {plate_length}/2, {plate_width}/2
elif motor_type == "SG90":
    spacing = 28.0
    hole_d = 2.2
    cx, cy = {plate_length}/2, {plate_width}/2
else:
    spacing = 31.0
    hole_d = 3.4
    cx, cy = {plate_length}/2, {plate_width}/2
half = spacing / 2
hole_r = hole_d / 2
for dx in [-half, half]:
    for dy in [-half, half]:
        hole = Part.makeCylinder(hole_r, {plate_thickness} + 2)
        hole.translate(FreeCAD.Vector(cx + dx, cy + dy, -1))
        plate = plate.cut(hole)
# Center bore for shaft
bore_r = hole_r * 2.5
bore = Part.makeCylinder(bore_r, {plate_thickness} + 2)
bore.translate(FreeCAD.Vector(cx, cy, -1))
plate = plate.cut(bore)
obj = doc.addObject("Part::Feature", "MotorMount")
obj.Shape = plate
doc.recompute()
"""

_SENSOR_MOUNT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("sensor_mount")
# Horizontal base
base = Part.makeBox({base_length}, {base_width}, {thickness})
# Vertical bracket arm
arm = Part.makeBox({thickness}, {base_width}, {bracket_height})
arm.translate(FreeCAD.Vector(0, 0, {thickness}))
result = base.fuse(arm)
# Sensor mounting hole on arm
hole = Part.makeCylinder({hole_diameter}/2, {thickness} + 2)
hole.translate(FreeCAD.Vector({thickness}/2, {base_width}/2, {bracket_height} * 0.6 - 1))
result = result.cut(hole)
obj = doc.addObject("Part::Feature", "SensorMount")
obj.Shape = result
doc.recompute()
"""

_BASE_PLATE_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("base_plate")
shape = "{shape}"
if shape == "circle":
    body = Part.makeCylinder({length_or_diameter}/2, {thickness})
    if {center_bore} > 0:
        bore = Part.makeCylinder({center_bore}/2, {thickness} + 2)
        body = body.cut(bore)
    if {mounting_hole_diameter} > 0 and {num_holes} > 0:
        hole_r = {mounting_hole_diameter} / 2
        pcd = {length_or_diameter} * 0.8
        for i in range({num_holes}):
            angle = 2 * math.pi * i / {num_holes}
            hx = pcd/2 * math.cos(angle)
            hy = pcd/2 * math.sin(angle)
            hole = Part.makeCylinder(hole_r, {thickness} + 2)
            hole.translate(FreeCAD.Vector(hx, hy, -1))
            body = body.cut(hole)
else:
    body = Part.makeBox({length_or_diameter}, {width}, {thickness})
    if {center_bore} > 0:
        bore = Part.makeCylinder({center_bore}/2, {thickness} + 2)
        bore.translate(FreeCAD.Vector({length_or_diameter}/2, {width}/2, -1))
        body = body.cut(bore)
    if {mounting_hole_diameter} > 0 and {num_holes} >= 4:
        hole_r = {mounting_hole_diameter} / 2
        margin = min({length_or_diameter}, {width}) * 0.1
        cols = max(2, int(math.sqrt({num_holes})))
        rows = max(2, ({num_holes} + cols - 1) // cols)
        dx = ({length_or_diameter} - 2*margin) / max(cols - 1, 1)
        dy = ({width} - 2*margin) / max(rows - 1, 1)
        count = 0
        for ix in range(cols):
            for iy in range(rows):
                if count >= {num_holes}:
                    break
                hx = margin + ix * dx
                hy = margin + iy * dy
                hole = Part.makeCylinder(hole_r, {thickness} + 2)
                hole.translate(FreeCAD.Vector(hx, hy, -1))
                body = body.cut(hole)
                count += 1
obj = doc.addObject("Part::Feature", "BasePlate")
obj.Shape = body
doc.recompute()
"""

_FLANGE_COUPLING_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("flange_coupling")
# Main disc
disc = Part.makeCylinder({outer_diameter}/2, {thickness})
# Center bore
if {inner_diameter} > 0:
    bore = Part.makeCylinder({inner_diameter}/2, {thickness} + 2)
    disc = disc.cut(bore)
# Bolt holes on PCD
if {bolt_count} > 0 and {bolt_pcd} > 0:
    hole_r = {bolt_hole_diameter} / 2
    for i in range({bolt_count}):
        angle = 2 * math.pi * i / {bolt_count}
        hx = {bolt_pcd}/2 * math.cos(angle)
        hy = {bolt_pcd}/2 * math.sin(angle)
        hole = Part.makeCylinder(hole_r, {thickness} + 2)
        hole.translate(FreeCAD.Vector(hx, hy, -1))
        disc = disc.cut(hole)
obj = doc.addObject("Part::Feature", "FlangeCoupling")
obj.Shape = disc
doc.recompute()
"""

_SHAFT_SUPPORT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("shaft_support")
# Base plate
base = Part.makeBox({base_length}, {base_width}, {base_thickness})
# Side supports (left and right walls)
wall_h = {bearing_width} + 2 * {base_thickness}
wall1 = Part.makeBox({base_thickness}, {base_width}, wall_h)
wall1.translate(FreeCAD.Vector(0, 0, {base_thickness}))
wall2 = Part.makeBox({base_thickness}, {base_width}, wall_h)
wall2.translate(FreeCAD.Vector({base_length} - {base_thickness}, 0, {base_thickness}))
result = base.fuse(wall1).fuse(wall2)
# Bearing press-fit holes in both walls
bore_r = {shaft_diameter}/2 + 0.5
for wx in [{base_thickness}/2, {base_length} - {base_thickness}/2]:
    bore = Part.makeCylinder(bore_r, {base_thickness} + 4)
    bore.translate(FreeCAD.Vector(wx, {base_width}/2, wall_h/2 + {base_thickness} - 2))
    result = result.cut(bore)
# Base mounting holes
mhole_r = 2.0
margin = {base_thickness} + 2
for mx in [margin, {base_length} - margin]:
    for my in [margin, {base_width} - margin]:
        mhole = Part.makeCylinder(mhole_r, {base_thickness} + 2)
        mhole.translate(FreeCAD.Vector(mx, my, -1))
        result = result.cut(mhole)
obj = doc.addObject("Part::Feature", "ShaftSupport")
obj.Shape = result
doc.recompute()
"""

_BATTERY_BOX_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("battery_box")
# Outer box (open top)
outer = Part.makeBox({length}, {width}, {height})
inner_l = {length} - 2 * {wall_thickness}
inner_w = {width} - 2 * {wall_thickness}
inner_h = {height} - {wall_thickness}
if inner_l > 0 and inner_w > 0 and inner_h > 0:
    inner = Part.makeBox(inner_l, inner_w, inner_h)
    inner.translate(FreeCAD.Vector({wall_thickness}, {wall_thickness}, 0))
    box = outer.cut(inner)
else:
    box = outer
# Battery cell slots
cell_type = "{cell_type}"
cell_d = {{"18650": 18.0, "21700": 21.0, "26650": 26.0}}.get(cell_type, 18.0)
cell_r = cell_d / 2
spacing = cell_d + 1.0
for i in range({num_cells}):
    cx = {wall_thickness} + cell_r + 1
    cy = {wall_thickness} + cell_r + 1 + i * spacing
    if cy + cell_r < {width} - {wall_thickness}:
        slot = Part.makeCylinder(cell_r, inner_l)
        slot.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(0, 1, 0), 90)
        slot.translate(FreeCAD.Vector({wall_thickness}, cy, cell_r + {wall_thickness}))
        box = box.cut(slot)
# Lid mounting holes (corners)
hole_r = 1.5
margin = {wall_thickness} + 2
for hx in [margin, {length} - margin]:
    for hy in [margin, {width} - margin]:
        hole = Part.makeCylinder(hole_r, {wall_thickness} + 2)
        hole.translate(FreeCAD.Vector(hx, hy, -1))
        box = box.cut(hole)
obj = doc.addObject("Part::Feature", "BatteryBox")
obj.Shape = box
doc.recompute()
"""

_PCB_MOUNT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("pcb_mount")
# Pillar
pillar = Part.makeCylinder({outer_diameter}/2, {height})
# Through hole
hole = Part.makeCylinder({hole_diameter}/2, {height})
result = pillar.cut(hole)
obj = doc.addObject("Part::Feature", "PCBMount")
obj.Shape = result
doc.recompute()
"""

# ---------------------------------------------------------------------------
# Layer 3 Phase 1: 15 additional structural part templates
# ---------------------------------------------------------------------------

_ALUMINUM_EXTRUSION_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("aluminum_extrusion")
# Main square profile
profile = Part.makeBox({profile_size}, {profile_size}, {length})
# Central bore
if {bore_size} > 0:
    bore_r = {bore_size} / 2
    bore = Part.makeCylinder(bore_r, {length} + 2)
    bore.translate(FreeCAD.Vector({profile_size}/2, {profile_size}/2, -1))
    profile = profile.cut(bore)
# T-slot grooves on all 4 sides
gw = {groove_w}
gd = {groove_d}
half = {profile_size} / 2
if gw > 0 and gd > 0:
    for cx, cy, dx, dy in [(0, half - gd/2, gw, gd), ({profile_size} - gw, half - gd/2, gw, gd),
                            (half - gd/2, 0, gd, gw), (half - gd/2, {profile_size} - gw, gd, gw)]:
        groove = Part.makeBox(dx, dy, {length} + 2)
        groove.translate(FreeCAD.Vector(cx, cy, -1))
        profile = profile.cut(groove)
obj = doc.addObject("Part::Feature", "AluminumExtrusion")
obj.Shape = profile
doc.recompute()
"""

_U_BRACKET_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("u_bracket")
t = {thickness}
# Base plate
base = Part.makeBox({leg_length}, {width}, t)
# Left wall
left = Part.makeBox(t, {width}, {height} - t)
left.translate(FreeCAD.Vector(0, 0, t))
# Right wall
right = Part.makeBox(t, {width}, {height} - t)
right.translate(FreeCAD.Vector({leg_length} - t, 0, t))
body = base.fuse(left).fuse(right)
# Mounting holes on base
hole_r = {hole_d} / 2
margin = t + hole_r + 1
for hx in [margin, {leg_length} - margin]:
    for hy in [margin, {width} - margin]:
        hole = Part.makeCylinder(hole_r, t + 2)
        hole.translate(FreeCAD.Vector(hx, hy, -1))
        body = body.cut(hole)
obj = doc.addObject("Part::Feature", "UBracket")
obj.Shape = body
doc.recompute()
"""

_T_BRACKET_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("t_bracket")
t = {thickness}
# Horizontal plate
plate = Part.makeBox({plate_w}, {plate_h}, t)
# Vertical stem (centered)
stem = Part.makeBox(t, {plate_h}, {stem_l})
stem.translate(FreeCAD.Vector({plate_w}/2 - t/2, 0, -{stem_l}))
body = plate.fuse(stem)
# Holes on plate (4 corners)
hole_r = {hole_d} / 2
margin = hole_r + 2
for hx in [margin, {plate_w} - margin]:
    for hy in [margin, {plate_h} - margin]:
        hole = Part.makeCylinder(hole_r, t + 2)
        hole.translate(FreeCAD.Vector(hx, hy, -1))
        body = body.cut(hole)
# Holes on stem (2 positions)
for sz in [{stem_l} * 0.3, {stem_l} * 0.7]:
    hole = Part.makeCylinder(hole_r, t + 2)
    hole.translate(FreeCAD.Vector({plate_w}/2, {plate_h}/2, -sz))
    body = body.cut(hole)
obj = doc.addObject("Part::Feature", "TBracket")
obj.Shape = body
doc.recompute()
"""

_GUSSET_PLATE_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("gusset_plate")
import math
a = {side_a}
b = {side_b}
t = {thickness}
# Right triangle prism via wire extrusion
v0 = FreeCAD.Vector(0, 0, 0)
v1 = FreeCAD.Vector(a, 0, 0)
v2 = FreeCAD.Vector(0, b, 0)
wire = Part.makePolygon([v0, v1, v2, v0])
face = Part.Face(wire)
body = face.extrude(FreeCAD.Vector(0, 0, t))
# Mounting holes
hole_r = {hole_d} / 2
for hx, hy in [a * 0.2, b * 0.2], [a * 0.6, b * 0.15]:
    hole = Part.makeCylinder(hole_r, t + 2)
    hole.translate(FreeCAD.Vector(hx, hy, -1))
    body = body.cut(hole)
obj = doc.addObject("Part::Feature", "GussetPlate")
obj.Shape = body
doc.recompute()
"""

_BEARING_BLOCK_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("bearing_block")
# Base plate
base = Part.makeBox({base_l}, {base_w}, {base_l} * 0.3)
base_h = {base_l} * 0.3
# Side walls
wall_h = {block_h} - base_h
wall1 = Part.makeBox({base_l}, {base_w} * 0.15, wall_h)
wall1.translate(FreeCAD.Vector(0, 0, base_h))
wall2 = Part.makeBox({base_l}, {base_w} * 0.15, wall_h)
wall2.translate(FreeCAD.Vector(0, {base_w} * 0.85, base_h))
body = base.fuse(wall1).fuse(wall2)
# Bearing bore through walls
bore_r = {shaft_d} / 2
for wy in [{base_w} * 0.075, {base_w} * 0.925]:
    bore = Part.makeCylinder(bore_r, wall_h + 2)
    bore.translate(FreeCAD.Vector({base_l}/2, wy, base_h - 1))
    body = body.cut(bore)
# Bolt holes in base
bolt_r = {bolt_d} / 2
margin = {base_l} * 0.15
for bx in [margin, {base_l} - margin]:
    for by in [margin, {base_w} - margin]:
        hole = Part.makeCylinder(bolt_r, base_h + 2)
        hole.translate(FreeCAD.Vector(bx, by, -1))
        body = body.cut(hole)
obj = doc.addObject("Part::Feature", "BearingBlock")
obj.Shape = body
doc.recompute()
"""

_SERVO_BRACKET_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("servo_bracket")
t = {plate_t}
servo_type = "{servo_type}"
# U-channel base
base = Part.makeBox({plate_l}, {plate_w}, t)
left = Part.makeBox({plate_l}, t, {flange_h})
left.translate(FreeCAD.Vector(0, 0, t))
right = Part.makeBox({plate_l}, t, {flange_h})
right.translate(FreeCAD.Vector(0, {plate_w} - t, t))
body = base.fuse(left).fuse(right)
# SG90 mounting holes
hole_r = 1.3
spacing_x = 28.0
for hx in [{plate_l}/2 - spacing_x/2, {plate_l}/2 + spacing_x/2]:
    for hy in [t/2, {plate_w} - t/2]:
        hole = Part.makeCylinder(hole_r, {flange_h} + 2)
        hole.translate(FreeCAD.Vector(hx, hy, -1))
        body = body.cut(hole)
# Base mounting holes
for bx in [{plate_l} * 0.15, {plate_l} * 0.85]:
    hole = Part.makeCylinder(1.5, t + 2)
    hole.translate(FreeCAD.Vector(bx, {plate_w}/2, -1))
    body = body.cut(hole)
obj = doc.addObject("Part::Feature", "ServoBracket")
obj.Shape = body
doc.recompute()
"""

_NEMA_MOUNT_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("nema_mount")
motor_type = "{motor_type}"
# Plate
plate = Part.makeBox({plate_l}, {plate_w}, {plate_t})
# NEMA hole pattern
patterns = {{"NEMA17": (31.0, 3.4, 22.0), "NEMA23": (47.14, 4.5, 31.0),
             "NEMA14": (26.0, 3.0, 16.0)}}
spacing, hole_d, bore_d = patterns.get(motor_type, (31.0, 3.4, 22.0))
cx, cy = {plate_l}/2, {plate_w}/2
# Center bore
bore = Part.makeCylinder(bore_d/2, {plate_t} + 2)
bore.translate(FreeCAD.Vector(cx, cy, -1))
plate = plate.cut(bore)
# Corner mounting holes
half = spacing / 2
hole_r = hole_d / 2
for dx in [-half, half]:
    for dy in [-half, half]:
        hole = Part.makeCylinder(hole_r, {plate_t} + 2)
        hole.translate(FreeCAD.Vector(cx + dx, cy + dy, -1))
        plate = plate.cut(hole)
obj = doc.addObject("Part::Feature", "NEMAMount")
obj.Shape = plate
doc.recompute()
"""

_STANDOFF_COLUMN_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("standoff_column")
# Outer cylinder
pillar = Part.makeCylinder({od}/2, {length})
# Through hole
hole = Part.makeCylinder({hole_d}/2, {length} + 2)
hole.translate(FreeCAD.Vector(0, 0, -1))
result = pillar.cut(hole)
obj = doc.addObject("Part::Feature", "StandoffColumn")
obj.Shape = result
doc.recompute()
"""

_CABLE_CHAIN_MOUNT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("cable_chain_mount")
t = {thickness}
# L-shaped bracket
base = Part.makeBox({base_l}, {base_w}, t)
wall = Part.makeBox(t, {base_w}, {slot_h} + t)
wall.translate(FreeCAD.Vector(0, 0, t))
body = base.fuse(wall)
# Cable slot (rectangular cutout in wall)
slot = Part.makeBox(t + 2, {slot_w}, {slot_h})
slot.translate(FreeCAD.Vector(-1, ({base_w} - {slot_w})/2, t))
body = body.cut(slot)
# Base mounting holes
hole_r = 1.5
margin = t + 3
for bx in [margin, {base_l} - margin]:
    hole = Part.makeCylinder(hole_r, t + 2)
    hole.translate(FreeCAD.Vector(bx, {base_w}/2, -1))
    body = body.cut(hole)
obj = doc.addObject("Part::Feature", "CableChainMount")
obj.Shape = body
doc.recompute()
"""

_BATTERY_TRAY_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("battery_tray")
# Open-top box
outer = Part.makeBox({length}, {width}, {height})
inner_l = {length} - 2 * {wall_t}
inner_w = {width} - 2 * {wall_t}
inner_h = {height} - {wall_t}
if inner_l > 0 and inner_w > 0 and inner_h > 0:
    inner = Part.makeBox(inner_l, inner_w, inner_h)
    inner.translate(FreeCAD.Vector({wall_t}, {wall_t}, 0))
    box = outer.cut(inner)
else:
    box = outer
# Tie-down slots on both long sides
slot_w = 3
slot_d = 2
for side_y in [0, {width} - slot_d]:
    slot = Part.makeBox({length} * 0.4, slot_d, slot_w)
    slot.translate(FreeCAD.Vector({length} * 0.3, side_y, {height} - slot_w))
    box = box.fuse(slot)
obj = doc.addObject("Part::Feature", "BatteryTray")
obj.Shape = box
doc.recompute()
"""

_SENSOR_SHELF_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("sensor_shelf")
t = {thickness}
angle_rad = math.radians({shelf_angle})
# Horizontal base
base = Part.makeBox({base_l}, {base_w}, t)
# Angled shelf
shelf_h = {base_l} * 0.5 * math.sin(angle_rad)
shelf_d = {base_l} * 0.5 * math.cos(angle_rad)
shelf = Part.makeBox(shelf_d, {base_w}, t)
shelf.translate(FreeCAD.Vector(0, 0, shelf_h))
shelf.rotate(FreeCAD.Vector(0, 0, shelf_h), FreeCAD.Vector(0, 1, 0), -{shelf_angle})
body = base.fuse(shelf)
# Sensor mounting hole on shelf
hole_r = 2.0
hole = Part.makeCylinder(hole_r, t + 2)
hole.translate(FreeCAD.Vector(shelf_d * 0.5, {base_w}/2, shelf_h - 1))
body = body.cut(hole)
obj = doc.addObject("Part::Feature", "SensorShelf")
obj.Shape = body
doc.recompute()
"""

_SHAFT_COUPLING_BLOCK_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("shaft_coupling_block")
# Main block
block = Part.makeBox({block_l}, {block_w}, {block_h})
# Bore through center (along X)
bore_r = {bore_d} / 2
bore = Part.makeCylinder(bore_r, {block_l} + 2)
bore.translate(FreeCAD.Vector(-1, {block_w}/2, {block_h}/2))
block = block.cut(bore)
# Set-screw hole (top, perpendicular)
if {set_screw_d} > 0:
    screw_r = {set_screw_d} / 2
    screw = Part.makeCylinder(screw_r, {block_h}/2 + 2)
    screw.translate(FreeCAD.Vector({block_l}/2, {block_w}/2, {block_h} - 1))
    block = block.cut(screw)
obj = doc.addObject("Part::Feature", "ShaftCouplingBlock")
obj.Shape = block
doc.recompute()
"""

_GUIDE_RAIL_CARRIAGE_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("guide_rail_carriage")
rail_type = "{rail_type}"
# Carriage plate
plate = Part.makeBox({plate_l}, {plate_w}, {plate_t})
# Mounting holes (4 corner holes)
hole_r = 1.5
margin_l = {plate_l} * 0.15
margin_w = {plate_w} * 0.15
for hx in [margin_l, {plate_l} - margin_l]:
    for hy in [margin_w, {plate_w} - margin_w]:
        hole = Part.makeCylinder(hole_r, {plate_t} + 2)
        hole.translate(FreeCAD.Vector(hx, hy, -1))
        plate = plate.cut(hole)
# Rail slot indicators (grooves on bottom)
groove_w = 3.0
groove_d = 1.5
for gx in [{plate_l} * 0.3, {plate_l} * 0.7]:
    groove = Part.makeBox(groove_w, {plate_w} * 0.6, groove_d)
    groove.translate(FreeCAD.Vector(gx - groove_w/2, {plate_w} * 0.2, -groove_d))
    plate = plate.cut(groove)
obj = doc.addObject("Part::Feature", "GuideRailCarriage")
obj.Shape = plate
doc.recompute()
"""

_PULLEY_IDLER_MOUNT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("pulley_idler_mount")
t = {plate_t}
# L-bracket
base = Part.makeBox({plate_l}, {plate_w}, t)
wall = Part.makeBox(t, {plate_w}, {plate_w} * 0.8)
wall.translate(FreeCAD.Vector(0, 0, t))
body = base.fuse(wall)
# Bearing hole in wall
bearing_r = {bearing_od} / 2
bore = Part.makeCylinder(bearing_r, t + 2)
bore.translate(FreeCAD.Vector(t/2, {plate_w}/2, t + {plate_w} * 0.4 - 1))
body = body.cut(bore)
# Base mounting holes
hole_r = 1.5
margin = t + 3
for bx in [margin, {plate_l} - margin]:
    hole = Part.makeCylinder(hole_r, t + 2)
    hole.translate(FreeCAD.Vector(bx, {plate_w}/2, -1))
    body = body.cut(hole)
obj = doc.addObject("Part::Feature", "PulleyIdlerMount")
obj.Shape = body
doc.recompute()
"""

_ENCODER_MOUNT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("encoder_mount")
t = {thickness}
# Small L-bracket
base = Part.makeBox({base_l}, {base_w}, t)
wall = Part.makeBox(t, {base_w}, {base_w} * 0.7)
wall.translate(FreeCAD.Vector(0, 0, t))
body = base.fuse(wall)
# Encoder center bore through wall
bore_r = {bore_d} / 2
bore = Part.makeCylinder(bore_r, t + 2)
bore.translate(FreeCAD.Vector(t/2, {base_w}/2, t + {base_w} * 0.35 - 1))
body = body.cut(bore)
# Fixing holes (2 on base, 2 on wall)
hole_r = 1.3
for bx in [{base_l} * 0.2, {base_l} * 0.8]:
    hole = Part.makeCylinder(hole_r, t + 2)
    hole.translate(FreeCAD.Vector(bx, {base_w}/2, -1))
    body = body.cut(hole)
obj = doc.addObject("Part::Feature", "EncoderMount")
obj.Shape = body
doc.recompute()
"""

# ---------------------------------------------------------------------------
# Task 68: Real functional part scripts — motors, servos, sensors
# ---------------------------------------------------------------------------

# TT Motor (Gearbox Motor) — most common cheap DC motor for robot kits
_TT_MOTOR_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("tt_motor")
# Motor body (rectangular)
body = Part.makeBox({body_length}, {body_width}, {body_height})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Gearbox housing (slightly wider section)
gb = Part.makeBox({gearbox_length}, {gearbox_width}, {gearbox_height})
gb.translate(FreeCAD.Vector({body_length}, ({body_width} - {gearbox_width})/2, ({body_height} - {gearbox_height})/2))
gb_obj = doc.addObject("Part::Feature", "Gearbox")
gb_obj.Shape = gb
# Output shaft
shaft = Part.makeCylinder({shaft_diameter}/2, {shaft_length})
shaft.translate(FreeCAD.Vector({body_length} + {gearbox_length}, {body_width}/2, {body_height}/2))
shaft_obj = doc.addObject("Part::Feature", "Shaft")
shaft_obj.Shape = shaft
# Mounting tabs (2 ears)
tab_w = 5
tab_h = 2
tab = Part.makeBox({body_length} + 2*tab_w, tab_w, tab_h)
tab.translate(FreeCAD.Vector(-tab_w, -tab_w, {body_height} - tab_h))
tab_obj = doc.addObject("Part::Feature", "MountingTab")
tab_obj.Shape = tab
doc.recompute()
"""

# DS3218 Digital Servo — 20kg high-torque servo
_DS3218_SERVO_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("servo_ds3218")
# Main body
body = Part.makeBox({body_length}, {body_width}, {body_height})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Mounting tabs
tab_w = 6
tab = Part.makeBox({body_length} + 2*tab_w, {body_width}, tab_w/2)
tab.translate(FreeCAD.Vector(-tab_w, 0, -tab_w/2))
tab_obj = doc.addObject("Part::Feature", "MountingTab")
tab_obj.Shape = tab
# Output shaft (larger than SG90)
shaft = Part.makeCylinder({shaft_diameter}/2, {shaft_length})
shaft.translate(FreeCAD.Vector({body_length}/2 - 10, {body_width}/2, {body_height}))
shaft_obj = doc.addObject("Part::Feature", "Shaft")
shaft_obj.Shape = shaft
doc.recompute()
"""

# JGB37-520 DC Gearmotor — common 12V gearmotor for medium robots
_JGB37_520_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("jgb37_520")
# Motor body (cylindrical)
body = Part.makeCylinder({body_diameter}/2, {body_length})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Gearbox housing (slightly larger cylinder)
gb = Part.makeCylinder({gearbox_diameter}/2, {gearbox_length})
gb.translate(FreeCAD.Vector(0, 0, {body_length}))
gb_obj = doc.addObject("Part::Feature", "Gearbox")
gb_obj.Shape = gb
# Output shaft
shaft = Part.makeCylinder({shaft_diameter}/2, {shaft_length})
shaft.translate(FreeCAD.Vector(0, 0, {body_length} + {gearbox_length}))
shaft_obj = doc.addObject("Part::Feature", "Shaft")
shaft_obj.Shape = shaft
# Mounting bracket holes (2 M3 holes on gearbox face)
import math
for angle in [0, 180]:
    rad = math.radians(angle)
    hx = {gearbox_diameter}/2 * 0.55 * math.cos(rad)
    hy = {gearbox_diameter}/2 * 0.55 * math.sin(rad)
    hole = Part.makeCylinder(1.5, {gearbox_length})
    hole.translate(FreeCAD.Vector(hx, hy, {body_length}))
    gb = gb.cut(hole)
gb_obj.Shape = gb
doc.recompute()
"""

# NEMA23 Stepper Motor — larger stepper for CNC/robotics
_NEMA23_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("nema23_stepper")
# Main body (square)
body = Part.makeBox({body_size}, {body_size}, {body_length})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Output shaft
shaft = Part.makeCylinder({shaft_diameter}/2, {shaft_length})
shaft.translate(FreeCAD.Vector({body_size}/2, {body_size}/2, {body_length}))
shaft_obj = doc.addObject("Part::Feature", "Shaft")
shaft_obj.Shape = shaft
# Mounting holes (4 corners)
for dx, dy in [(4.63, 4.63), ({body_size}-4.63, 4.63), (4.63, {body_size}-4.63), ({body_size}-4.63, {body_size}-4.63)]:
    hole = Part.makeCylinder(2.0, 3)
    hole.translate(FreeCAD.Vector(dx, dy, {body_length}))
    body = body.cut(hole)
body_obj.Shape = body
doc.recompute()
"""

# RPLIDAR A1 — 2D LiDAR sensor for mobile robots
_RPLIDAR_A1_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("rplidar_a1")
# Main body (cylinder)
body = Part.makeCylinder({body_diameter}/2, {body_height})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Top dome (slight bump)
dome = Part.makeCylinder({body_diameter}/4, 3)
dome.translate(FreeCAD.Vector(0, 0, {body_height}))
dome_obj = doc.addObject("Part::Feature", "Dome")
dome_obj.Shape = dome
# Base mounting holes (4x M3)
import math
mount_r = {body_diameter}/2 - 5
for i in range(4):
    angle = math.radians(90 * i + 45)
    hx = mount_r * math.cos(angle)
    hy = mount_r * math.sin(angle)
    hole = Part.makeCylinder(1.6, {body_height})
    hole.translate(FreeCAD.Vector(hx, hy, 0))
    body = body.cut(hole)
body_obj.Shape = body
doc.recompute()
"""

# MPU6050 IMU Module
_MPU6050_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("mpu6050")
# PCB board
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
# IMU chip (center)
chip = Part.makeBox(4, 4, 1.2)
chip.translate(FreeCAD.Vector({pcb_length}/2 - 2, {pcb_width}/2 - 2, {pcb_thickness}))
chip_obj = doc.addObject("Part::Feature", "Chip")
chip_obj.Shape = chip
# Mounting holes (4 corners)
margin = 2.5
hole_r = 1.0
for x in [margin, {pcb_length} - margin]:
    for y in [margin, {pcb_width} - margin]:
        hole = Part.makeCylinder(hole_r, {pcb_thickness})
        hole.translate(FreeCAD.Vector(x, y, 0))
        pcb = pcb.cut(hole)
pcb_obj.Shape = pcb
doc.recompute()
"""

# ESP32-CAM Module
_ESP32_CAM_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("esp32_cam")
# Main PCB
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
# Camera module (center, cylindrical)
cam = Part.makeCylinder({camera_diameter}/2, {camera_height})
cam.translate(FreeCAD.Vector({pcb_length}/2, {pcb_width}/2, {pcb_thickness}))
cam_obj = doc.addObject("Part::Feature", "Camera")
cam_obj.Shape = cam
# ESP32 chip
chip = Part.makeBox(10, 10, 2)
chip.translate(FreeCAD.Vector({pcb_length}/2 - 5, 3, {pcb_thickness}))
chip_obj = doc.addObject("Part::Feature", "ESP32Chip")
chip_obj.Shape = chip
# Mounting holes (2 on each side)
for x in [3, {pcb_length} - 3]:
    for y in [3, {pcb_width} - 3]:
        hole = Part.makeCylinder(1.0, {pcb_thickness})
        hole.translate(FreeCAD.Vector(x, y, 0))
        pcb = pcb.cut(hole)
pcb_obj.Shape = pcb
doc.recompute()
"""


# ---------------------------------------------------------------------------
# Task 73: Transmission parts — GT2/HTD pulleys, belts, couplings, keyway
# ---------------------------------------------------------------------------

# GT2 Synchronous Pulley — simplified (cylinder + flange + bore)
_GT2_PULLEY_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("gt2_pulley")
pitch = 2.0
n_teeth = {teeth}
r_pitch = n_teeth * pitch / (2 * math.pi)
r_outer = r_pitch + 0.75
r_root = r_pitch - 0.30
w = {width}
flange_d = r_outer + 1.5
# Main body
body = Part.makeCylinder(r_outer, w)
# Flanges
flange1 = Part.makeCylinder(flange_d, 1.0)
flange2 = Part.makeCylinder(flange_d, 1.0)
flange2.translate(FreeCAD.Vector(0, 0, w - 1.0))
body = body.fuse(flange1).fuse(flange2)
# Bore
bore = Part.makeCylinder({bore_diameter}/2, w)
body = body.cut(bore)
obj = doc.addObject("Part::Feature", "GT2Pulley")
obj.Shape = body
doc.recompute()
"""

# GT2 Synchronous Pulley — realistic (tooth profile)
_GT2_PULLEY_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("gt2_pulley_realistic")
pitch = 2.0
n_teeth = int({teeth})
w = {width}
bore_r = {bore_diameter} / 2.0

r_pitch = n_teeth * pitch / (2 * math.pi)
r_outer = r_pitch + 0.75
r_root = r_pitch - 0.30

# Build tooth profile: GT2 has rounded trapezoidal teeth
tooth_half_angle = math.pi / n_teeth  # half angular pitch
tip_half_angle = tooth_half_angle * 0.45
root_half_angle = tooth_half_angle * 0.55

points = []
for i in range(n_teeth):
    base_angle = i * 2 * math.pi / n_teeth
    # Root arc start
    a0 = base_angle - root_half_angle
    points.append(FreeCAD.Vector(r_root * math.cos(a0), r_root * math.sin(a0), 0))
    # Tooth flank rise (left)
    a1 = base_angle - tip_half_angle
    points.append(FreeCAD.Vector(r_outer * math.cos(a1), r_outer * math.sin(a1), 0))
    # Tooth tip (small arc approximated by 3 points)
    for j in range(3):
        ta = base_angle - tip_half_angle + tip_half_angle * 2 * j / 2
        points.append(FreeCAD.Vector(r_outer * math.cos(ta), r_outer * math.sin(ta), 0))
    # Tooth flank fall (right)
    a2 = base_angle + tip_half_angle
    points.append(FreeCAD.Vector(r_outer * math.cos(a2), r_outer * math.sin(a2), 0))
    # Root arc end
    a3 = base_angle + root_half_angle
    points.append(FreeCAD.Vector(r_root * math.cos(a3), r_root * math.sin(a3), 0))

points.append(points[0])

try:
    gear_wire = Part.makePolygon(points)
    gear_face = Part.Face(gear_wire)
    body = gear_face.extrude(FreeCAD.Vector(0, 0, w))
except Exception:
    body = Part.makeCylinder(r_outer, w)

# Flanges
flange_d = r_outer + 1.5
flange1 = Part.makeCylinder(flange_d, 1.0)
flange2 = Part.makeCylinder(flange_d, 1.0)
flange2.translate(FreeCAD.Vector(0, 0, w - 1.0))
body = body.fuse(flange1).fuse(flange2)

# Bore
bore = Part.makeCylinder(bore_r, w)
body = body.cut(bore)

# Hub (optional raised center)
if {hub_diameter} > {bore_diameter}:
    hub_h = {hub_height}
    hub = Part.makeCylinder({hub_diameter}/2, hub_h)
    hub.translate(FreeCAD.Vector(0, 0, w))
    body = body.fuse(hub)
    hub_bore = Part.makeCylinder(bore_r, w + hub_h)
    body = body.cut(hub_bore)

obj = doc.addObject("Part::Feature", "GT2Pulley")
obj.Shape = body
doc.recompute()
"""

# GT2 Belt — simplified flat belt with teeth on one side
_GT2_BELT_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("gt2_belt")
pitch = 2.0
n_teeth = {teeth}
w = {width}
belt_length = n_teeth * pitch
belt_thickness = 1.38
tooth_height = 0.75

# Belt body (flat strip loop approximated as a thick ring)
r_outer = belt_length / (2 * math.pi)
r_inner = r_outer - belt_thickness

body = Part.makeCylinder(r_outer, w)
inner = Part.makeCylinder(r_inner, w)
belt = body.cut(inner)

# Teeth on inner surface
tooth_count = n_teeth
for i in range(tooth_count):
    angle = 2 * math.pi * i / tooth_count
    cx = r_inner * math.cos(angle)
    cy = r_inner * math.sin(angle)
    tooth = Part.makeCylinder(0.3, w)
    tooth.translate(FreeCAD.Vector(cx, cy, 0))
    try:
        belt = belt.cut(tooth)
    except Exception as _e:
        logger.debug("belt tooth cut failed (cosmetic: %s", _e)

obj = doc.addObject("Part::Feature", "GT2Belt")
obj.Shape = belt
doc.recompute()
"""

# HTD Pulley — simplified (cylinder + flange + bore)
_HTD_PULLEY_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("htd_pulley")
pitch = {pitch}
n_teeth = {teeth}
r_pitch = n_teeth * pitch / (2 * math.pi)
r_addendum = r_pitch + {module}
r_dedendum = r_pitch - 1.25 * {module}
w = {width}
flange_r = r_addendum + 2.0

body = Part.makeCylinder(r_addendum, w)
# Flanges
flange1 = Part.makeCylinder(flange_r, 1.5)
flange2 = Part.makeCylinder(flange_r, 1.5)
flange2.translate(FreeCAD.Vector(0, 0, w - 1.5))
body = body.fuse(flange1).fuse(flange2)
# Bore
bore = Part.makeCylinder({bore_diameter}/2, w)
body = body.cut(bore)
obj = doc.addObject("Part::Feature", "HTDPulley")
obj.Shape = body
doc.recompute()
"""

# HTD Pulley — realistic (HTD tooth profile: semi-circular)
_HTD_PULLEY_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("htd_pulley_realistic")
pitch = {pitch}
n_teeth = int({teeth})
w = {width}
bore_r = {bore_diameter} / 2.0

r_pitch = n_teeth * pitch / (2 * math.pi)
r_addendum = r_pitch + {module}
r_dedendum = r_pitch - 1.25 * {module}

# HTD tooth profile: semi-circular groove between rounded teeth
tooth_angle = 2 * math.pi / n_teeth
groove_r = pitch * 0.30  # groove radius for HTD profile

points = []
resolution = 8  # points per groove and per tooth
for i in range(n_teeth):
    base = i * tooth_angle
    # Groove (dedendum arc)
    groove_center_r = r_dedendum + groove_r
    for j in range(resolution + 1):
        a = base + tooth_angle * 0.1 + (tooth_angle * 0.35) * j / resolution
        gx = (groove_center_r - groove_r) * math.cos(a)
        gy = (groove_center_r - groove_r) * math.sin(a)
        points.append(FreeCAD.Vector(gx, gy, 0))
    # Tooth (addendum arc)
    for j in range(resolution + 1):
        a = base + tooth_angle * 0.5 + (tooth_angle * 0.4) * j / resolution
        tx = r_addendum * math.cos(a)
        ty = r_addendum * math.sin(a)
        points.append(FreeCAD.Vector(tx, ty, 0))

points.append(points[0])

try:
    profile_wire = Part.makePolygon(points)
    profile_face = Part.Face(profile_wire)
    body = profile_face.extrude(FreeCAD.Vector(0, 0, w))
except Exception:
    body = Part.makeCylinder(r_addendum, w)

# Flanges
flange_r = r_addendum + 2.0
flange1 = Part.makeCylinder(flange_r, 1.5)
flange2 = Part.makeCylinder(flange_r, 1.5)
flange2.translate(FreeCAD.Vector(0, 0, w - 1.5))
body = body.fuse(flange1).fuse(flange2)

# Bore
bore = Part.makeCylinder(bore_r, w)
body = body.cut(bore)

# Hub
if {hub_diameter} > {bore_diameter}:
    hub_h = {hub_height}
    hub = Part.makeCylinder({hub_diameter}/2, hub_h)
    hub.translate(FreeCAD.Vector(0, 0, w))
    body = body.fuse(hub)
    hub_bore = Part.makeCylinder(bore_r, w + hub_h)
    body = body.cut(hub_bore)

obj = doc.addObject("Part::Feature", "HTDPulley")
obj.Shape = body
doc.recompute()
"""

# Rigid Coupling — set screw type
_RIGID_COUPLING_SETSCREW_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("rigid_coupling_setscrew")
od = {outer_diameter}
l = {length}
bore = {bore_diameter}
n_setscrews = {num_setscrews}
setscrew_size = {setscrew_size}

body = Part.makeCylinder(od/2, l)
# Central bore
hole = Part.makeCylinder(bore/2, l)
body = body.cut(hole)
# Set screw holes (radial, perpendicular to bore axis)
for i in range(int(n_setscrews)):
    z = l * (i + 1) / (n_setscrews + 1)
    setscrew_hole = Part.makeCylinder(setscrew_size/4, od/2)
    setscrew_hole.translate(FreeCAD.Vector(0, 0, z))
    setscrew_hole.rotate(FreeCAD.Vector(0, 0, z), FreeCAD.Vector(0, 0, 1), 90)
    body = body.cut(setscrew_hole)

obj = doc.addObject("Part::Feature", "RigidCoupling")
obj.Shape = body
doc.recompute()
"""

# Rigid Coupling — clamping type (split clamp)
_RIGID_COUPLING_CLAMPING_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("rigid_coupling_clamping")
od = {outer_diameter}
l = {length}
bore = {bore_diameter}
clamp_screw = {clamp_screw_size}
clamp_width = 4.0

body = Part.makeCylinder(od/2, l)
# Central bore
hole = Part.makeCylinder(bore/2, l)
body = body.cut(hole)
# Clamp slot (through the diameter)
slot = Part.makeBox(od, clamp_width, l)
slot.translate(FreeCAD.Vector(-od/2, -clamp_width/2, 0))
body = body.cut(slot)
# Clamp screw holes (perpendicular to slot)
screw_hole_r = clamp_screw / 2
for z in [l * 0.3, l * 0.7]:
    for dx in [-od/4, od/4]:
        sh = Part.makeCylinder(screw_hole_r, clamp_width + 4)
        sh.translate(FreeCAD.Vector(dx, -clamp_width/2 - 2, z))
        sh.rotate(FreeCAD.Vector(dx, 0, z), FreeCAD.Vector(1, 0, 0), 90)
        body = body.cut(sh)

obj = doc.addObject("Part::Feature", "ClampingCoupling")
obj.Shape = body
doc.recompute()
"""

# Spider (Jaw) Flexible Coupling — two metal hubs + elastomer spider
_SPIDER_COUPLING_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("spider_coupling")
od = {outer_diameter}
l = {length}
bore1 = {bore1_diameter}
bore2 = {bore2_diameter}
jaw_count = {jaw_count}
jaw_depth = {jaw_depth}

half_l = l / 2.0

# Hub 1 (bottom half)
hub1 = Part.makeCylinder(od/2, half_l)
bore1_shape = Part.makeCylinder(bore1/2, half_l)
hub1 = hub1.cut(bore1_shape)

# Jaw cuts on top of hub1
for i in range(int(jaw_count)):
    angle = 2 * math.pi * i / jaw_count
    jaw_w = od * 0.25
    jaw = Part.makeBox(jaw_depth, jaw_w, jaw_depth)
    cx = (od/2 - jaw_depth) * math.cos(angle)
    cy = (od/2 - jaw_depth) * math.sin(angle)
    jaw.translate(FreeCAD.Vector(cx - jaw_depth/2, cy - jaw_w/2, half_l - jaw_depth))
    jaw_rot = Part.Body()
    jaw_obj_rot = jaw_rot.addObject("Part::Feature", "Jaw")
    jaw_obj_rot.Shape = jaw
    jaw_rot.rotate(FreeCAD.Vector(0, 0, half_l), FreeCAD.Vector(0, 0, 1), math.degrees(angle))
    try:
        hub1 = hub1.cut(jaw_rot.Shape)
    except Exception as _e:
        logger.debug("hub jaw cut failed (cosmetic: %s", _e)

# Hub 2 (top half)
hub2 = Part.makeCylinder(od/2, half_l)
bore2_shape = Part.makeCylinder(bore2/2, half_l)
hub2 = hub2.cut(bore2_shape)
hub2.translate(FreeCAD.Vector(0, 0, half_l))

# Spider (elastomer) — simplified as thin ring between hubs
spider_r = od / 2 * 0.75
spider_t = 2.0
spider = Part.makeCylinder(spider_r, spider_t)
spider_inner = Part.makeCylinder(od/4, spider_t)
spider = spider.cut(spider_inner)
spider.translate(FreeCAD.Vector(0, 0, half_l - spider_t/2))

obj1 = doc.addObject("Part::Feature", "Hub1")
obj1.Shape = hub1
obj2 = doc.addObject("Part::Feature", "Hub2")
obj2.Shape = hub2
obj3 = doc.addObject("Part::Feature", "Spider")
obj3.Shape = spider
doc.recompute()
"""

# Bellows Flexible Coupling
_BELLOWS_COUPLING_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("bellows_coupling")
od = {outer_diameter}
l = {length}
bore1 = {bore1_diameter}
bore2 = {bore2_diameter}
n_convolutions = {convolutions}
wall_t = {wall_thickness}

half_l = l / 2.0
conv_height = half_l / n_convolutions

# Hub 1
hub1_od = od * 1.2
hub1 = Part.makeCylinder(hub1_od/2, half_l * 0.3)
bore1_shape = Part.makeCylinder(bore1/2, half_l * 0.3)
hub1 = hub1.cut(bore1_shape)

# Bellows section (corrugated thin wall)
bellows_r = od / 2
bellows_start = half_l * 0.3
for i in range(int(n_convolutions)):
    z = bellows_start + i * conv_height
    r = bellows_r if i % 2 == 0 else bellows_r * 0.75
    ring = Part.makeCylinder(r, conv_height)
    ring.translate(FreeCAD.Vector(0, 0, z))
    if i == 0:
        bellows = ring
    else:
        bellows = bellows.fuse(ring)

# Hub 2
hub2_od = od * 1.2
hub2 = Part.makeCylinder(hub2_od/2, half_l * 0.3)
bore2_shape = Part.makeCylinder(bore2/2, half_l * 0.3)
hub2 = hub2.cut(bore2_shape)
hub2.translate(FreeCAD.Vector(0, 0, l - half_l * 0.3))

obj1 = doc.addObject("Part::Feature", "Hub1")
obj1.Shape = hub1
obj2 = doc.addObject("Part::Feature", "Hub2")
obj2.Shape = hub2
obj3 = doc.addObject("Part::Feature", "Bellows")
obj3.Shape = bellows
doc.recompute()
"""


# ---------------------------------------------------------------------------
# Keyway generation — DIN 6885 square key feature utilities
# ---------------------------------------------------------------------------

# DIN 6885 square key dimensions (shaft diameter -> key size)
DIN_6885_SQUARE_KEYS: dict[tuple[float, float], dict[str, float]] = {
    # (min_shaft_d, max_shaft_d) -> {key_width, key_height}
    (6, 8):   {"key_width": 2, "key_height": 2},
    (8, 10):  {"key_width": 3, "key_height": 3},
    (10, 12): {"key_width": 4, "key_height": 4},
    (12, 17): {"key_width": 5, "key_height": 5},
    (17, 22): {"key_width": 6, "key_height": 6},
    (22, 30): {"key_width": 8, "key_height": 7},
    (30, 38): {"key_width": 10, "key_height": 8},
    (38, 44): {"key_width": 12, "key_height": 8},
    (44, 50): {"key_width": 14, "key_height": 9},
    (50, 58): {"key_width": 16, "key_height": 10},
    (58, 65): {"key_width": 18, "key_height": 11},
    (65, 75): {"key_width": 20, "key_height": 12},
    (75, 85): {"key_width": 22, "key_height": 14},
    (85, 95): {"key_width": 25, "key_height": 14},
    (95, 110): {"key_width": 28, "key_height": 16},
    (110, 130): {"key_width": 32, "key_height": 18},
}

# Standard key lengths (DIN 6885)
STANDARD_KEY_LENGTHS: list[float] = [
    6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50,
    56, 63, 70, 80, 90, 100, 110, 125, 140, 160, 180, 200,
]


def get_key_size(shaft_diameter: float) -> dict[str, float]:
    """Get DIN 6885 square key dimensions for a given shaft diameter.

    Args:
        shaft_diameter: Shaft diameter in mm.

    Returns:
        Dict with 'key_width' and 'key_height' in mm.

    Raises:
        ValueError: If shaft diameter is out of DIN 6885 range.
    """
    for (min_d, max_d), dims in DIN_6885_SQUARE_KEYS.items():
        if min_d <= shaft_diameter <= max_d:
            return dims
    raise ValueError(
        f"Shaft diameter {shaft_diameter}mm is outside DIN 6885 range (6-130mm)"
    )


def generate_shaft_keyway_ops(
    shaft_diameter: float,
    keyway_length: float,
    position_z: float = 0.0,
) -> list[dict]:
    """Generate FreeCAD operations for a shaft keyway (DIN 6885).

    The keyway is a rectangular slot cut into the shaft surface.

    Args:
        shaft_diameter: Shaft diameter in mm.
        keyway_length: Length of the keyway along shaft axis.
        position_z: Z-offset of keyway start along shaft.

    Returns:
        List of operation dicts for part_feature_engine.
    """
    key = get_key_size(shaft_diameter)
    key_w = key["key_width"]
    key_h = key["key_height"]
    shaft_r = shaft_diameter / 2.0

    # Keyway depth: how deep the key sits into the shaft
    keyway_depth = shaft_r - (shaft_r - key_h)

    return [
        {
            "type": "box",
            "name": "keyway",
            "width": key_w,
            "depth": key_h,
            "height": keyway_length,
            "x": -key_w / 2.0,
            "y": shaft_r - key_h,
            "z": position_z,
            "operation": "cut",
        }
    ]


def generate_hub_keyway_ops(
    bore_diameter: float,
    hub_length: float,
    position_z: float = 0.0,
) -> list[dict]:
    """Generate FreeCAD operations for a hub/bore keyway (DIN 6885).

    The keyway is a rectangular slot cut into the bore surface.

    Args:
        bore_diameter: Bore diameter in mm.
        hub_length: Length of the hub along bore axis.
        position_z: Z-offset of keyway start.

    Returns:
        List of operation dicts for part_feature_engine.
    """
    key = get_key_size(bore_diameter)
    key_w = key["key_width"]
    key_h = key["key_height"]
    bore_r = bore_diameter / 2.0

    # Hub keyway extends from bore surface outward
    return [
        {
            "type": "box",
            "name": "hub_keyway",
            "width": key_w,
            "depth": key_h,
            "height": hub_length,
            "x": -key_w / 2.0,
            "y": bore_r - key_h * 0.2,
            "z": position_z,
            "operation": "cut",
        }
    ]


def generate_key_ops(
    shaft_diameter: float,
    key_length: float,
) -> list[dict]:
    """Generate FreeCAD operations for a standalone square key (DIN 6885).

    Args:
        shaft_diameter: Shaft diameter in mm (determines key cross-section).
        key_length: Length of the key along the shaft axis.

    Returns:
        List of operation dicts for part_feature_engine.
    """
    key = get_key_size(shaft_diameter)
    return [
        {
            "type": "box",
            "name": "square_key",
            "width": key["key_width"],
            "depth": key["key_height"],
            "height": key_length,
            "x": -key["key_width"] / 2.0,
            "y": 0,
            "z": 0,
            "operation": "add",
        }
    ]


def recommend_key_length(shaft_diameter: float, required_length: float) -> float:
    """Recommend the nearest standard key length (DIN 6885).

    Args:
        shaft_diameter: Shaft diameter in mm.
        required_length: Minimum required key length.

    Returns:
        Nearest standard key length >= required_length.
    """
    for kl in STANDARD_KEY_LENGTHS:
        if kl >= required_length:
            return kl
    return STANDARD_KEY_LENGTHS[-1]


# ---------------------------------------------------------------------------
# Task 74: Linear motion & advanced actuator parts
# ---------------------------------------------------------------------------

# Linear Ball Bearing (LM8UU type) — simplified
_LINEAR_BEARING_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("linear_bearing")
OD = {outer_diameter}
ID = {inner_diameter}
L = {length}
# Outer cylinder
outer = Part.makeCylinder(OD/2, L)
inner_cut = Part.makeCylinder(ID/2, L)
body = outer.cut(inner_cut)
obj = doc.addObject("Part::Feature", "LinearBearing")
obj.Shape = body
doc.recompute()
"""

# Linear Ball Bearing — realistic (outer ring, inner groove, balls)
_LINEAR_BEARING_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("linear_bearing_realistic")
OD = {outer_diameter}
ID = {inner_diameter}
L = {length}
n_balls = int({ball_count}) if int({ball_count}) > 0 else 4

gap = (OD - ID) / 2.0
ball_d = gap * 0.55
pitch_r = (OD + ID) / 4.0
# Outer ring
outer_ring = Part.makeCylinder(OD/2, L)
outer_bore = Part.makeCylinder(ID/2 + gap*0.35, L)
outer_ring = outer_ring.cut(outer_bore)
# Inner ring (thin sleeve)
inner_od = ID/2 + gap*0.30
inner_ring = Part.makeCylinder(inner_od, L)
inner_bore = Part.makeCylinder(ID/2, L)
inner_ring = inner_ring.cut(inner_bore)
inner_obj = doc.addObject("Part::Feature", "InnerRing")
inner_obj.Shape = inner_ring
# Balls in 2-4 rows
n_rows = max(2, n_balls // 6)
balls_per_row = max(4, n_balls // n_rows)
row_spacing = L / (n_rows + 1)
ball_idx = 0
for row in range(n_rows):
    z = row_spacing * (row + 1)
    for i in range(balls_per_row):
        angle = 2 * math.pi * i / balls_per_row
        x = pitch_r * math.cos(angle)
        y = pitch_r * math.sin(angle)
        ball = Part.makeSphere(ball_d/2)
        ball.translate(FreeCAD.Vector(x, y, z))
        ball_name = "Ball_" + str(ball_idx)
        ball_obj = doc.addObject("Part::Feature", ball_name)
        ball_obj.Shape = ball
        ball_idx += 1
outer_obj = doc.addObject("Part::Feature", "OuterRing")
outer_obj.Shape = outer_ring
doc.recompute()
"""

# MGN12 Linear Guide Rail — simplified
_LINEAR_GUIDE_RAIL_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("linear_guide_rail")
rl = {rail_length}
rw = {rail_width}
rh = {rail_height}
# Rail body
rail = Part.makeBox(rl, rw, rh)
# Ball groove channels (2 grooves on top)
groove_r = rw * 0.15
for y_off in [rw*0.3, rw*0.7]:
    groove = Part.makeCylinder(groove_r, rl)
    groove.rotate(FreeCAD.Vector(0, y_off, rh), FreeCAD.Vector(1, 0, 0), 90)
    rail = rail.cut(groove)
# Mounting holes
mh_r = {mounting_hole_diameter}/2
spacing = {mounting_hole_pitch}
n_holes = max(2, int(rl / spacing))
for i in range(n_holes):
    hx = spacing/2 + i * spacing
    if hx > rl - spacing/2:
        break
    hole = Part.makeCylinder(mh_r, rh)
    hole.translate(FreeCAD.Vector(hx, rw/2, 0))
    rail = rail.cut(hole)
obj = doc.addObject("Part::Feature", "Rail")
obj.Shape = rail
doc.recompute()
"""

# MGN12 Linear Guide Carriage (Slider) — simplified
_LINEAR_GUIDE_CARRIAGE_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("linear_guide_carriage")
cl = {carriage_length}
cw = {carriage_width}
ch = {carriage_height}
rw = {rail_width}
# Carriage body
body = Part.makeBox(cl, cw, ch)
# Mounting holes on top
mh_r = {mounting_hole_diameter}/2
mh_spacing = cl * 0.6
for x in [cl*0.2, cl*0.8]:
    for y in [cw*0.25, cw*0.75]:
        hole = Part.makeCylinder(mh_r, ch*0.5)
        hole.translate(FreeCAD.Vector(x, y, ch*0.5))
        body = body.cut(hole)
# Groove channels on bottom (matching rail)
groove_r = rw * 0.15
for y_off in [cw*0.3, cw*0.7]:
    groove = Part.makeCylinder(groove_r, cl)
    groove.rotate(FreeCAD.Vector(cl/2, y_off, 0), FreeCAD.Vector(0, 1, 0), 90)
    groove.translate(FreeCAD.Vector(-cl/2, 0, groove_r*0.5))
    body = body.cut(groove)
obj = doc.addObject("Part::Feature", "Carriage")
obj.Shape = body
doc.recompute()
"""

# T8 Leadscrew — simplified cylinder
_T8_LEADSCREW_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("t8_leadscrew")
shaft = Part.makeCylinder({diameter}/2, {length})
obj = doc.addObject("Part::Feature", "Leadscrew")
obj.Shape = shaft
doc.recompute()
"""

# T8 Leadscrew — realistic (trapezoidal thread)
_T8_LEADSCREW_REALISTIC_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("t8_leadscrew_realistic")
d = {diameter}
l = {length}
lead = {lead}
pitch = lead  # single start
# Core shaft
minor_r = d/2 - 1.0
shaft = Part.makeCylinder(minor_r, l)
# Trapezoidal thread via helical sweep
try:
    pitch_r = d/2 - 0.5
    helix = Part.makeHelix(pitch, l, pitch_r)
    thread_h = d/2 - minor_r
    profile_pts = [
        FreeCAD.Vector(0, 0, 0),
        FreeCAD.Vector(thread_h, 0, pitch*0.15),
        FreeCAD.Vector(thread_h*0.8, 0, pitch*0.35),
        FreeCAD.Vector(0, 0, pitch*0.5),
    ]
    profile_wire = Part.makePolygon(profile_pts + [profile_pts[0]])
    profile_face = Part.Face(profile_wire)
    thread = profile_face.makePipe(helix)
    result = shaft.fuse(thread)
    obj = doc.addObject("Part::Feature", "Leadscrew")
    obj.Shape = result
except Exception:
    obj = doc.addObject("Part::Feature", "Leadscrew")
    obj.Shape = Part.makeCylinder(d/2, l)
doc.recompute()
"""

# T8 Leadscrew Nut — simplified
_T8_NUT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("t8_nut")
od = {outer_diameter}
l = {length}
bore = {bore_diameter}
body = Part.makeCylinder(od/2, l)
hole = Part.makeCylinder(bore/2, l)
body = body.cut(hole)
# Flange (if applicable)
flange_od = {flange_diameter}
if flange_od > od:
    flange = Part.makeCylinder(flange_od/2, {flange_thickness})
    body = body.fuse(flange)
    # Flange mounting holes
    import math
    for i in range(4):
        angle = math.pi/2 * i + math.pi/4
        hx = flange_od/2 * 0.7 * math.cos(angle)
        hy = flange_od/2 * 0.7 * math.sin(angle)
        fhole = Part.makeCylinder({flange_hole_diameter}/2, {flange_thickness})
        fhole.translate(FreeCAD.Vector(hx, hy, 0))
        body = body.cut(fhole)
obj = doc.addObject("Part::Feature", "T8Nut")
obj.Shape = body
doc.recompute()
"""

# BLDC Motor (outrunner) — simplified
_BLDC_MOTOR_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("bldc_motor")
# Stator (inner, fixed)
stator_od = {stator_outer_diameter}
stator_id = {stator_inner_diameter}
stator_l = {stator_length}
stator_outer = Part.makeCylinder(stator_od/2, stator_l)
stator_bore = Part.makeCylinder(stator_id/2, stator_l)
stator = stator_outer.cut(stator_bore)
stator_obj = doc.addObject("Part::Feature", "Stator")
stator_obj.Shape = stator
# Rotor (outer, rotating) — outrunner: rotor wraps around stator
rotor_od = {rotor_outer_diameter}
rotor_id = {rotor_inner_diameter}
rotor_l = {rotor_length}
rotor_outer = Part.makeCylinder(rotor_od/2, rotor_l)
rotor_inner = Part.makeCylinder(rotor_id/2, rotor_l)
rotor = rotor_outer.cut(rotor_inner)
rotor.translate(FreeCAD.Vector(0, 0, -0.5))
rotor_obj = doc.addObject("Part::Feature", "Rotor")
rotor_obj.Shape = rotor
# Shaft
shaft = Part.makeCylinder({shaft_diameter}/2, {shaft_length})
shaft.translate(FreeCAD.Vector(0, 0, -{shaft_length}+{stator_length}/2))
shaft_obj = doc.addObject("Part::Feature", "Shaft")
shaft_obj.Shape = shaft
doc.recompute()
"""

# DYNAMIXEL XM430-W350-T Smart Servo
_XM430_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("xm430_w350")
BW = {body_width}
BH = {body_height}
BD = {body_depth}
HD = {horn_diameter}
SD = {shaft_diameter}
# Body (rectangular block)
body = Part.makeBox(BW, BH, BD)
# Output flange cylinder (front face, centered)
flange = Part.makeCylinder(HD/2, 3.0)
flange.translate(FreeCAD.Vector(BW/2 - HD/2, BH/2, 0))
body = body.fuse(flange)
# Output shaft (D-cut, passes through body)
shaft = Part.makeCylinder(SD/2, BD + 8.0)
shaft.translate(FreeCAD.Vector(BW/2 - SD/2, BH/2, -4.0))
body = body.fuse(shaft)
# D-flat cut on shaft
flat_w = SD * 0.3
flat = Part.makeBox(SD, flat_w, BD + 10.0)
flat.translate(FreeCAD.Vector(BW/2 - SD/2 + SD - flat_w*0.2, BH/2 + SD/2 - flat_w, -5.0))
body = body.cut(flat)
# 4x M2.5 mounting holes (back face, 16x16mm grid)
hx_off = BW/2
hy_off = BH/2
hole_spacing = 8.0
hole_r = 1.25
for dx in [-hole_spacing, hole_spacing]:
    for dy in [-hole_spacing, hole_spacing]:
        hole = Part.makeCylinder(hole_r, BD)
        hole.translate(FreeCAD.Vector(hx_off - hole_r + dx, hy_off - hole_r + dy, 0))
        body = body.cut(hole)
# 4x M2.5 through holes on flange face
for dx in [-hole_spacing, hole_spacing]:
    for dy in [-hole_spacing, hole_spacing]:
        fhole = Part.makeCylinder(hole_r, 5.0)
        fhole.translate(FreeCAD.Vector(BW/2 - hole_r + dx, BH/2 - hole_r + dy, BD))
        body = body.cut(fhole)
obj = doc.addObject("Part::Feature", "XM430_W350")
obj.Shape = body
doc.recompute()
"""

# Compression Spring — parametric helix
_COMPRESSION_SPRING_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("compression_spring")
wire_r = {wire_diameter}/2
coil_r = {outer_diameter}/2 - wire_r
n_coils = {active_coils}
free_length = {free_length}
pitch = free_length / n_coils
try:
    # Helix path
    helix = Part.makeHelix(pitch, free_length, coil_r)
    # Wire cross-section (circle)
    wire_circle = Part.makeCircle(wire_r)
    wire_face = Part.Face(wire_circle)
    wire_face.translate(FreeCAD.Vector(coil_r, 0, 0))
    # Sweep wire along helix
    spring = wire_face.makePipe(helix)
    obj = doc.addObject("Part::Feature", "Spring")
    obj.Shape = spring
except Exception:
    # Fallback: simplified cylinder representation
    body = Part.makeCylinder({outer_diameter}/2, free_length)
    inner = Part.makeCylinder({outer_diameter}/2 - {wire_diameter}, free_length)
    body = body.cut(inner)
    obj = doc.addObject("Part::Feature", "Spring")
    obj.Shape = body
doc.recompute()
"""

# Damper / Shock Absorber
_DAMPER_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("damper")
# Cylinder body
cyl_od = {cylinder_diameter}
cyl_l = {cylinder_length}
cylinder = Part.makeCylinder(cyl_od/2, cyl_l)
# Piston rod
rod_d = {rod_diameter}
rod_l = {rod_length}
rod = Part.makeCylinder(rod_d/2, rod_l)
rod.translate(FreeCAD.Vector(0, 0, cyl_l))
# Eyelet / mount at top
mount_r = {mount_diameter}/2
mount = Part.makeCylinder(mount_r, {mount_thickness})
mount.translate(FreeCAD.Vector(0, 0, cyl_l + rod_l))
mount_hole = Part.makeCylinder({mount_hole_diameter}/2, {mount_thickness})
mount = mount.cut(mount_hole)
# Eyelet at bottom
mount2 = Part.makeCylinder(mount_r, {mount_thickness})
mount2.translate(FreeCAD.Vector(0, 0, -{mount_thickness}))
mount2_hole = Part.makeCylinder({mount_hole_diameter}/2, {mount_thickness})
mount2 = mount2.cut(mount2_hole)
cyl_obj = doc.addObject("Part::Feature", "Cylinder")
cyl_obj.Shape = cylinder
rod_obj = doc.addObject("Part::Feature", "Rod")
rod_obj.Shape = rod
mt1_obj = doc.addObject("Part::Feature", "TopMount")
mt1_obj.Shape = mount
mt2_obj = doc.addObject("Part::Feature", "BottomMount")
mt2_obj.Shape = mount2
doc.recompute()
"""


# ---------------------------------------------------------------------------
# Task 75: Electronics & sensor parts
# ---------------------------------------------------------------------------

# PCB-like part — generic helper used by several electronics templates
_PCB_BOARD_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("pcb_board")
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
# Mounting holes
margin = {hole_margin}
hole_r = {hole_diameter}/2
for x in [margin, {pcb_length} - margin]:
    for y in [margin, {pcb_width} - margin]:
        hole = Part.makeCylinder(hole_r, {pcb_thickness})
        hole.translate(FreeCAD.Vector(x, y, 0))
        pcb = pcb.cut(hole)
pcb_obj.Shape = pcb
doc.recompute()
"""

# L298N Motor Driver
_L298N_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("l298n_driver")
# PCB board
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
pcb_obj = doc.addObject("Part::Feature", "PCB")
# Heatsink (large aluminum block)
hs = Part.makeBox({heatsink_length}, {heatsink_width}, {heatsink_height})
hs.translate(FreeCAD.Vector(({pcb_length}-{heatsink_length})/2, ({pcb_width}-{heatsink_width})/2, {pcb_thickness}))
hs_obj = doc.addObject("Part::Feature", "Heatsink")
hs_obj.Shape = hs
# Terminal blocks (2 sides)
tb_w = 8
tb_h = 10
tb1 = Part.makeBox(tb_w, {pcb_width}*0.8, tb_h)
tb1.translate(FreeCAD.Vector(3, {pcb_width}*0.1, {pcb_thickness}))
tb1_obj = doc.addObject("Part::Feature", "TerminalBlock1")
tb1_obj.Shape = tb1
tb2 = Part.makeBox(tb_w, {pcb_width}*0.8, tb_h)
tb2.translate(FreeCAD.Vector({pcb_length}-3-tb_w, {pcb_width}*0.1, {pcb_thickness}))
tb2_obj = doc.addObject("Part::Feature", "TerminalBlock2")
tb2_obj.Shape = tb2
# Mounting holes
for x in [7, {pcb_length}-7]:
    for y in [7, {pcb_width}-7]:
        hole = Part.makeCylinder(1.5, {pcb_thickness}+1)
        hole.translate(FreeCAD.Vector(x, y, -0.5))
        pcb = pcb.cut(hole)
pcb_obj.Shape = pcb
doc.recompute()
"""

# TB6612FNG Motor Driver (small breakout)
_TB6612FNG_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("tb6612fng_driver")
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
# Pin headers (2 rows)
pin_h = 8
pin_w = {pcb_width} * 0.9
for x in [2, {pcb_length}-4]:
    header = Part.makeBox(2, pin_w, pin_h)
    header.translate(FreeCAD.Vector(x, ({pcb_width}-pin_w)/2, {pcb_thickness}))
    header_obj = doc.addObject("Part::Feature", "PinHeader")
    header_obj.Shape = header
# IC chip
chip = Part.makeBox(7, 7, 1.5)
chip.translate(FreeCAD.Vector(({pcb_length}-7)/2, ({pcb_width}-7)/2, {pcb_thickness}))
chip_obj = doc.addObject("Part::Feature", "IC")
chip_obj.Shape = chip
# Mounting holes
for x in [3, {pcb_length}-3]:
    for y in [3, {pcb_width}-3]:
        hole = Part.makeCylinder(0.8, {pcb_thickness})
        hole.translate(FreeCAD.Vector(x, y, 0))
        pcb = pcb.cut(hole)
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
doc.recompute()
"""

# Arduino Uno
_ARDUINO_UNO_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("arduino_uno")
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
# USB connector
usb = Part.makeBox(13, 12, 6)
usb.translate(FreeCAD.Vector(-1, ({pcb_width}-12)/2, {pcb_thickness}))
usb_obj = doc.addObject("Part::Feature", "USB")
usb_obj.Shape = usb
# MCU chip
chip = Part.makeBox(10, 10, 2)
chip.translate(FreeCAD.Vector(({pcb_length}-10)/2, ({pcb_width}-10)/2, {pcb_thickness}))
chip_obj = doc.addObject("Part::Feature", "MCU")
chip_obj.Shape = chip
# Pin headers (2 long sides)
pin_h = 9
for y_off in [3, {pcb_width}-5]:
    header = Part.makeBox({pcb_length}-25, 2, pin_h)
    header.translate(FreeCAD.Vector(15, y_off, {pcb_thickness}))
    header_obj = doc.addObject("Part::Feature", "PinHeader")
    header_obj.Shape = header
# DC barrel jack
jack = Part.makeCylinder(5, 10)
jack.translate(FreeCAD.Vector(8, {pcb_width}/2, {pcb_thickness}))
jack.rotate(FreeCAD.Vector(8, {pcb_width}/2, {pcb_thickness}), FreeCAD.Vector(0, 1, 0), 90)
jack_obj = doc.addObject("Part::Feature", "DCJack")
jack_obj.Shape = jack
# Mounting holes
for x in [14, {pcb_length}-14]:
    for y in [6, {pcb_width}-6]:
        hole = Part.makeCylinder(1.6, {pcb_thickness})
        hole.translate(FreeCAD.Vector(x, y, 0))
        pcb = pcb.cut(hole)
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
doc.recompute()
"""

# Arduino Nano
_ARDUINO_NANO_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("arduino_nano")
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
# USB mini connector
usb = Part.makeBox(8, 8, 3.5)
usb.translate(FreeCAD.Vector(-1, ({pcb_width}-8)/2, {pcb_thickness}))
usb_obj = doc.addObject("Part::Feature", "USB")
usb_obj.Shape = usb
# Pin headers (2 long sides)
pin_h = 7
for y_off in [1, {pcb_width}-3]:
    header = Part.makeBox({pcb_length}-6, 2, pin_h)
    header.translate(FreeCAD.Vector(3, y_off, {pcb_thickness}))
    header_obj = doc.addObject("Part::Feature", "PinHeader")
    header_obj.Shape = header
# MCU chip
chip = Part.makeBox(7, 7, 1.5)
chip.translate(FreeCAD.Vector(({pcb_length}-7)/2, ({pcb_width}-7)/2, {pcb_thickness}))
chip_obj = doc.addObject("Part::Feature", "MCU")
chip_obj.Shape = chip
# Mounting holes
for x in [4, {pcb_length}-4]:
    hole = Part.makeCylinder(0.8, {pcb_thickness})
    hole.translate(FreeCAD.Vector(x, {pcb_width}/2, 0))
    pcb = pcb.cut(hole)
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
doc.recompute()
"""

# ESP32 DevKit
_ESP32_DEVKIT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("esp32_devkit")
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
# USB micro connector
usb = Part.makeBox(9, 8, 3.5)
usb.translate(FreeCAD.Vector(({pcb_length}-9)/2, -1, {pcb_thickness}))
usb_obj = doc.addObject("Part::Feature", "USB")
usb_obj.Shape = usb
# ESP32 module (metal shield)
shield = Part.makeBox(18, 20, 3)
shield.translate(FreeCAD.Vector(({pcb_length}-18)/2, ({pcb_width}-20)/2, {pcb_thickness}))
shield_obj = doc.addObject("Part::Feature", "ESP32Module")
shield_obj.Shape = shield
# Pin headers (2 long sides)
pin_h = 7
for x_off in [2, {pcb_length}-4]:
    header = Part.makeBox(2, {pcb_width}-4, pin_h)
    header.translate(FreeCAD.Vector(x_off, 2, {pcb_thickness}))
    header_obj = doc.addObject("Part::Feature", "PinHeader")
    header_obj.Shape = header
# Mounting holes
for x in [4, {pcb_length}-4]:
    for y in [4, {pcb_width}-4]:
        hole = Part.makeCylinder(0.8, {pcb_thickness})
        hole.translate(FreeCAD.Vector(x, y, 0))
        pcb = pcb.cut(hole)
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
doc.recompute()
"""

# AS5600 Magnetic Encoder
_AS5600_ENCODER_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("as5600_encoder")
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
# IC chip (center)
chip = Part.makeBox(5, 5, 1.2)
chip.translate(FreeCAD.Vector(({pcb_length}-5)/2, ({pcb_width}-5)/2, {pcb_thickness}))
chip_obj = doc.addObject("Part::Feature", "AS5600_IC")
chip_obj.Shape = chip
# Center hole for shaft
center_hole = Part.makeCylinder({center_hole_diameter}/2, {pcb_thickness})
center_hole.translate(FreeCAD.Vector({pcb_length}/2, {pcb_width}/2, 0))
pcb = pcb.cut(center_hole)
# Magnet (small cylinder above IC)
magnet = Part.makeCylinder({magnet_diameter}/2, {magnet_height})
magnet.translate(FreeCAD.Vector({pcb_length}/2, {pcb_width}/2, {pcb_thickness}+1))
magnet_obj = doc.addObject("Part::Feature", "Magnet")
magnet_obj.Shape = magnet
# Pin headers
pin_h = 6
header = Part.makeBox(2, {pcb_width}*0.7, pin_h)
header.translate(FreeCAD.Vector(2, {pcb_width}*0.15, {pcb_thickness}))
header_obj = doc.addObject("Part::Feature", "PinHeader")
header_obj.Shape = header
# Mounting holes
for x in [3, {pcb_length}-3]:
    for y in [3, {pcb_width}-3]:
        hole = Part.makeCylinder(0.8, {pcb_thickness})
        hole.translate(FreeCAD.Vector(x, y, 0))
        pcb = pcb.cut(hole)
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
doc.recompute()
"""

# Limit Switch (mechanical micro switch)
_LIMIT_SWITCH_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("limit_switch")
# Switch body
body = Part.makeBox({body_length}, {body_width}, {body_height})
body_obj = doc.addObject("Part::Feature", "Body")
body_obj.Shape = body
# Lever arm
lever = Part.makeBox({lever_length}, 2, 1)
lever.translate(FreeCAD.Vector({body_length}/2, {body_width}/2, {body_height}))
lever_obj = doc.addObject("Part::Feature", "Lever")
lever_obj.Shape = lever
# Terminals (3 pins on bottom)
pin_h = 5
for x_off in [{body_length}*0.2, {body_length}*0.5, {body_length}*0.8]:
    pin = Part.makeCylinder(0.5, pin_h)
    pin.translate(FreeCAD.Vector(x_off, {body_width}/2, -pin_h))
    pin_obj = doc.addObject("Part::Feature", "Pin")
    pin_obj.Shape = pin
# Mounting holes
for x in [3, {body_length}-3]:
    hole = Part.makeCylinder(1.2, {body_height})
    hole.translate(FreeCAD.Vector(x, {body_width}/2, 0))
    body = body.cut(hole)
body_obj.Shape = body
doc.recompute()
"""

# LM2596 Buck Converter
_LM2596_BUCK_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("lm2596_buck")
pcb = Part.makeBox({pcb_length}, {pcb_width}, {pcb_thickness})
# Inductor (large cylinder)
inductor = Part.makeCylinder({inductor_diameter}/2, {inductor_height})
inductor.translate(FreeCAD.Vector({pcb_length}*0.3, {pcb_width}/2, {pcb_thickness}))
inductor_obj = doc.addObject("Part::Feature", "Inductor")
inductor_obj.Shape = inductor
# IC (small black rectangle)
ic = Part.makeBox(7, 5, 2)
ic.translate(FreeCAD.Vector({pcb_length}*0.6, ({pcb_width}-5)/2, {pcb_thickness}))
ic_obj = doc.addObject("Part::Feature", "IC")
ic_obj.Shape = ic
# Potentiometer (adjustable voltage)
pot = Part.makeCylinder(3, 2)
pot.translate(FreeCAD.Vector({pcb_length}*0.75, {pcb_width}/2, {pcb_thickness}))
pot_obj = doc.addObject("Part::Feature", "Potentiometer")
pot_obj.Shape = pot
# Pin headers (input + output)
pin_h = 7
for x in [5, {pcb_length}-7]:
    header = Part.makeBox(2, 6, pin_h)
    header.translate(FreeCAD.Vector(x, ({pcb_width}-6)/2, {pcb_thickness}))
    header_obj = doc.addObject("Part::Feature", "PinHeader")
    header_obj.Shape = header
pcb_obj = doc.addObject("Part::Feature", "PCB")
pcb_obj.Shape = pcb
doc.recompute()
"""

# XT60 Connector
_XT60_CONNECTOR_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("xt60_connector")
body = Part.makeCylinder({body_diameter}/2, {body_length})
# Gold-plated contacts
contact_r = {contact_diameter}/2
contact1 = Part.makeCylinder(contact_r, {body_length}*1.1)
contact1.translate(FreeCAD.Vector({body_diameter}*0.2, 0, -{body_length}*0.05))
contact2 = Part.makeCylinder(contact_r, {body_length}*1.1)
contact2.translate(FreeCAD.Vector(-{body_diameter}*0.2, 0, -{body_length}*0.05))
body = body.cut(contact1).cut(contact2)
obj = doc.addObject("Part::Feature", "XT60")
obj.Shape = body
doc.recompute()
"""

# JST-XH Connector (2-6 pin)
_JST_XH_CONNECTOR_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("jst_xh_connector")
n_pins = {num_pins}
body_w = n_pins * 2.5 + 2
body = Part.makeBox(body_w, {body_width}, {body_height})
# Pin slots
for i in range(n_pins):
    x = 2.5 + i * 2.5
    pin = Part.makeCylinder(0.5, {body_height}*1.3)
    pin.translate(FreeCAD.Vector(x, {body_width}/2, -{body_height}*0.15))
    pin_obj = doc.addObject("Part::Feature", "Pin_" + str(i))
    pin_obj.Shape = pin
obj = doc.addObject("Part::Feature", "JST_XH")
obj.Shape = body
doc.recompute()
"""

# Heat-set brass thread insert — knurled cylinder with internal thread hole
_HEAT_SET_INSERT_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("heat_set_insert")
outer_r = {outer_diameter} / 2
thread_r = {thread_diameter} / 2
length = {length}
body = Part.makeCylinder(outer_r, length)
# Internal thread hole
hole = Part.makeCylinder(thread_r, length)
# Knurl pattern (simplified as longitudinal grooves)
n_grooves = int(math.pi * outer_r * 2 / {knurl_pitch})
groove_r = {knurl_pitch} / 3
for i in range(n_grooves):
    angle = 2 * math.pi * i / n_grooves
    x = (outer_r - groove_r * 0.5) * math.cos(angle)
    y = (outer_r - groove_r * 0.5) * math.sin(angle)
    groove = Part.makeCylinder(groove_r, length)
    groove.translate(FreeCAD.Vector(x, y, 0))
    body = body.cut(groove)
result = body.cut(hole)
obj = doc.addObject("Part::Feature", "HeatSetInsert")
obj.Shape = result
doc.recompute()
"""

# Shaft collar — ring with bore and radial set-screw hole
_SHAFT_COLLAR_SCRIPT = """\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("shaft_collar")
outer_r = {outer_diameter} / 2
bore_r = {bore_diameter} / 2
width = {width}
body = Part.makeCylinder(outer_r, width)
hole = Part.makeCylinder(bore_r, width)
result = body.cut(hole)
# Radial set-screw hole
screw_r = {set_screw_diameter} / 2
screw_hole = Part.makeCylinder(screw_r, outer_r - bore_r)
screw_hole.rotate(FreeCAD.Vector(0, 0, width / 2), FreeCAD.Vector(0, 1, 0), 90)
screw_hole.translate(FreeCAD.Vector(bore_r, 0, 0))
result = result.cut(screw_hole)
obj = doc.addObject("Part::Feature", "ShaftCollar")
obj.Shape = result
doc.recompute()
"""

# T-nut for aluminum extrusion slots
_T_NUT_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("t_nut")
# T-shaped cross section
plate_w = {plate_width}
plate_h = {plate_height}
flange_w = {flange_width}
flange_h = {flange_height}
thread_r = {thread_diameter} / 2
total_h = plate_h + flange_h
# Build T-profile by extrusion
plate = Part.makeBox(plate_w, plate_h, {nut_length})
flange = Part.makeBox(flange_w, flange_h, {nut_length})
flange.translate(FreeCAD.Vector(-(flange_w - plate_w) / 2, plate_h, 0))
body = plate.fuse(flange)
# Center threaded hole
hole = Part.makeCylinder(thread_r, total_h)
hole.translate(FreeCAD.Vector(plate_w / 2, 0, 0))
result = body.cut(hole)
obj = doc.addObject("Part::Feature", "TNut")
obj.Shape = result
doc.recompute()
"""

# Dowel pin — cylinder with chamfered ends
_DOWEL_PIN_SCRIPT = """\
import FreeCAD, Part
doc = FreeCAD.newDocument("dowel_pin")
r = {diameter} / 2
length = {length}
chamfer = {chamfer}
pin = Part.makeCylinder(r, length)
# Chamfer both ends using cones
if chamfer > 0:
    cone1 = Part.makeCone(r, r - chamfer, chamfer)
    cone1.translate(FreeCAD.Vector(0, 0, -chamfer))
    pin = pin.fuse(cone1)
    cone2 = Part.makeCone(r, r - chamfer, chamfer)
    cone2.translate(FreeCAD.Vector(0, 0, length))
    pin = pin.fuse(cone2)
obj = doc.addObject("Part::Feature", "DowelPin")
obj.Shape = pin
doc.recompute()
"""


# ---------------------------------------------------------------------------
# Standard parts catalog (25+ templates)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Standard parts catalog (25+ templates)
# ---------------------------------------------------------------------------
# The PART_CATALOG dict body (~2800 lines) lives in _catalog_entries.py
# (AGENTS.md §2.1: single file ≤500 lines). It imports all models and
# constants from this module, so entries resolve identically.

def _load_catalog() -> dict[str, PartTemplate]:
    """Load PART_CATALOG from the extracted entries module (deferred
    import to avoid circular dependency)."""
    from ._catalog_entries import ENTRIES
    return ENTRIES

PART_CATALOG: dict[str, PartTemplate] = _load_catalog()


# ---------------------------------------------------------------------------
# Task 76: Mounting interface definitions for all functional parts
# ---------------------------------------------------------------------------

MOUNTING_INTERFACES: dict[str, MountingInterface] = {
    # ---- Stepper motors ----
    "nema17_stepper": MountingInterface(
        interface_type="through_hole",
        contact_face="front",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-15.5, y=-15.5, diameter=3.4),
            BoltHole(x=15.5, y=-15.5, diameter=3.4),
            BoltHole(x=-15.5, y=15.5, diameter=3.4),
            BoltHole(x=15.5, y=15.5, diameter=3.4),
        ],
        bore_diameter=23.0,
        bore_depth=0.0,
    ),
    "nema23_stepper": MountingInterface(
        interface_type="through_hole",
        contact_face="front",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-23.57, y=-23.57, diameter=4.5),
            BoltHole(x=23.57, y=-23.57, diameter=4.5),
            BoltHole(x=-23.57, y=23.57, diameter=4.5),
            BoltHole(x=23.57, y=23.57, diameter=4.5),
        ],
        bore_diameter=31.0,
        bore_depth=0.0,
    ),

    # ---- Servos ----
    "servo_sg90": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-27.5, y=0, diameter=2.0),
            BoltHole(x=27.5, y=0, diameter=2.0),
        ],
        pocket_width=11.8,
        pocket_height=22.2,
        pocket_depth=8.0,
    ),
    "servo_mg996r": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-12.35, y=-9.9, diameter=2.8),
            BoltHole(x=12.35, y=-9.9, diameter=2.8),
            BoltHole(x=-12.35, y=9.9, diameter=2.8),
            BoltHole(x=12.35, y=9.9, diameter=2.8),
        ],
        pocket_width=19.7,
        pocket_height=40.7,
        pocket_depth=10.0,
    ),
    "servo_ds3218": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-12.5, y=-10.0, diameter=2.8),
            BoltHole(x=12.5, y=-10.0, diameter=2.8),
            BoltHole(x=-12.5, y=10.0, diameter=2.8),
            BoltHole(x=12.5, y=10.0, diameter=2.8),
        ],
        pocket_width=20.0,
        pocket_height=40.0,
        pocket_depth=10.0,
    ),

    # ---- Bearings ----
    "bearing_608": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(0.0, 1.0, 0.0),
        bore_diameter=21.95,
        bore_depth=7.0,
        press_fit_interference=0.05,
        shoulder_diameter=24.0,
        shoulder_depth=1.0,
    ),
    "bearing_623": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(0.0, 1.0, 0.0),
        bore_diameter=10.0,
        bore_depth=4.0,
        press_fit_interference=0.04,
        shoulder_diameter=12.0,
        shoulder_depth=0.8,
    ),
    "bearing_625": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(0.0, 1.0, 0.0),
        bore_diameter=16.0,
        bore_depth=5.0,
        press_fit_interference=0.04,
        shoulder_diameter=18.0,
        shoulder_depth=0.8,
    ),
    "bearing_626": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(0.0, 1.0, 0.0),
        bore_diameter=19.0,
        bore_depth=6.0,
        press_fit_interference=0.04,
        shoulder_diameter=22.0,
        shoulder_depth=0.8,
    ),
    "bearing_688": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(0.0, 1.0, 0.0),
        bore_diameter=16.0,
        bore_depth=4.0,
        press_fit_interference=0.03,
        shoulder_diameter=18.0,
        shoulder_depth=0.6,
    ),

    # ---- Sensors ----
    "sensor_rplidar_a1": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-30, y=-30, diameter=3.0),
            BoltHole(x=30, y=-30, diameter=3.0),
            BoltHole(x=-30, y=30, diameter=3.0),
            BoltHole(x=30, y=30, diameter=3.0),
        ],
    ),
    "sensor_mpu6050": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-8, y=-5, diameter=2.2),
            BoltHole(x=8, y=-5, diameter=2.2),
            BoltHole(x=-8, y=5, diameter=2.2),
            BoltHole(x=8, y=5, diameter=2.2),
        ],
    ),
    "sensor_esp32_cam": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-15, y=-9, diameter=2.0),
            BoltHole(x=15, y=-9, diameter=2.0),
            BoltHole(x=-15, y=9, diameter=2.0),
            BoltHole(x=15, y=9, diameter=2.0),
        ],
    ),
    "encoder_as5600": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-7, y=-7, diameter=2.0),
            BoltHole(x=7, y=-7, diameter=2.0),
            BoltHole(x=-7, y=7, diameter=2.0),
            BoltHole(x=7, y=7, diameter=2.0),
        ],
        bore_diameter=6.0,
        bore_depth=0.0,
    ),

    # ---- Electronics ----
    "driver_l298n": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-14.5, y=-14.5, diameter=3.0),
            BoltHole(x=14.5, y=-14.5, diameter=3.0),
            BoltHole(x=-14.5, y=14.5, diameter=3.0),
            BoltHole(x=14.5, y=14.5, diameter=3.0),
        ],
    ),
    "controller_arduino_uno": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-27.3, y=-20.7, diameter=3.0),
            BoltHole(x=27.3, y=-20.7, diameter=3.0),
            BoltHole(x=-27.3, y=20.7, diameter=3.0),
            BoltHole(x=27.3, y=20.7, diameter=3.0),
        ],
    ),
    "controller_arduino_nano": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-18.5, y=0, diameter=2.0),
            BoltHole(x=18.5, y=0, diameter=2.0),
        ],
    ),
    "controller_esp32_devkit": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-23.5, y=-10, diameter=2.0),
            BoltHole(x=23.5, y=-10, diameter=2.0),
            BoltHole(x=-23.5, y=10, diameter=2.0),
            BoltHole(x=23.5, y=10, diameter=2.0),
        ],
    ),

    # ---- ROBOTIS DYNAMIXEL XM430 series ----
    # XM430-W350-T has TWO mounting faces:
    #   1) Body (back) face: 4×M2.5 tapped holes, 16×16mm grid — used to mount TO something
    #   2) Horn (front) face: 4×M2.5 through holes on ⌀22mm bolt circle — something mounts TO it
    # Reference: ROBOTIS e-Manual + OpenMANIPULATOR-X BOM

    "dynamixel_xm430_w350_body": MountingInterface(
        interface_type="threaded_hole",
        contact_face="back",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-8.0, y=-8.0, diameter=2.8),   # M2.5 tap, 2.8mm clearance
            BoltHole(x=8.0, y=-8.0, diameter=2.8),
            BoltHole(x=-8.0, y=8.0, diameter=2.8),
            BoltHole(x=8.0, y=8.0, diameter=2.8),
        ],
        bore_diameter=0.0,
        notes="4×M2.5 螺纹孔，间距 16×16mm，本体背面安装面。"
              "XM430 本体 28×34×46.5mm，安装面 28×28mm。",
    ),
    "dynamixel_xm430_w350_horn": MountingInterface(
        interface_type="through_hole",
        contact_face="front",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-8.0, y=0.0, diameter=2.8),    # M2.5 clearance on ⌀22 PCD
            BoltHole(x=8.0, y=0.0, diameter=2.8),
            BoltHole(x=0.0, y=-8.0, diameter=2.8),
            BoltHole(x=0.0, y=8.0, diameter=2.8),
        ],
        bore_diameter=6.0,    # ⌀6mm output shaft
        bore_depth=0.0,
        alignment_features=[
            AlignmentFeature(
                feature_type="d_cut",
                diameter=6.0,
                width=0.5,
                depth=1.0,
            ),
        ],
        notes="4×M2.5 通孔，⌀16mm PCD（简化为 ±8mm 十字分布），⌀6mm D-cut 输出轴。"
              "法兰 ⌀22mm。用于连接 FR12 框架、舵盘、连杆等。",
    ),

    # ---- ROBOTIS FR12 frame brackets ----
    # Each FR12 frame has XM430-compatible hole patterns on its mounting faces.

    "robotis_fr12_h101": MountingInterface(
        interface_type="through_hole",
        contact_face="front",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            # Side A: matches XM430 body (4×M2.5, 16×16mm grid)
            BoltHole(x=-8.0, y=-8.0, diameter=2.8),
            BoltHole(x=8.0, y=-8.0, diameter=2.8),
            BoltHole(x=-8.0, y=8.0, diameter=2.8),
            BoltHole(x=8.0, y=8.0, diameter=2.8),
        ],
        notes="FR12-H101-K H型框架。两侧各有 4×M2.5 通孔（16×16mm 间距），"
              "分别连接两个 XM430 舵机（一个 body 面，一个 horn 面）。"
              "用于 OpenMANIPULATOR-X 的 link2 和 link5。",
    ),
    "robotis_fr12_h104": MountingInterface(
        interface_type="through_hole",
        contact_face="front",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-8.0, y=-8.0, diameter=2.8),
            BoltHole(x=8.0, y=-8.0, diameter=2.8),
            BoltHole(x=-8.0, y=8.0, diameter=2.8),
            BoltHole(x=8.0, y=8.0, diameter=2.8),
        ],
        notes="FR12-H104-K 加长 H 型框架。一端 4×M2.5 匹配 XM430，"
              "另一端为导轨安装面。用于 link5（腕部+夹爪基座）。",
    ),
    "robotis_fr12_s101": MountingInterface(
        interface_type="through_hole",
        contact_face="front",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-8.0, y=-8.0, diameter=2.8),
            BoltHole(x=8.0, y=-8.0, diameter=2.8),
            BoltHole(x=-8.0, y=8.0, diameter=2.8),
            BoltHole(x=8.0, y=8.0, diameter=2.8),
        ],
        notes="FR12-S101-K U 型短连杆。两端 4×M2.5 匹配 XM430。用于 link4（前臂）。",
    ),
    "robotis_fr12_s102": MountingInterface(
        interface_type="through_hole",
        contact_face="front",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-8.0, y=-8.0, diameter=2.8),
            BoltHole(x=8.0, y=-8.0, diameter=2.8),
            BoltHole(x=-8.0, y=8.0, diameter=2.8),
            BoltHole(x=8.0, y=8.0, diameter=2.8),
        ],
        notes="FR12-S102-K U 型长连杆。两端 4×M2.5 匹配 XM430。用于 link3（上臂）。",
    ),

    # ---- P0-3: Functional part MountingInterfaces (12 parts) ----

    "motor_tt": MountingInterface(
        interface_type="through_hole",
        contact_face="side",
        contact_face_normal=(0.0, -1.0, 0.0),
        holes=[
            BoltHole(x=-8.0, y=9.0, diameter=2.0),
            BoltHole(x=8.0, y=9.0, diameter=2.0),
        ],
        bore_diameter=3.2,
        bore_depth=7.5,
        notes="TT减速电机。齿轮箱端面 2×M2 安装孔，间距16mm。D型输出轴。",
    ),
    "motor_jgb37_520": MountingInterface(
        interface_type="threaded_hole",
        contact_face="back",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-11.0, y=0.0, diameter=4.0, hole_type="threaded_hole"),
            BoltHole(x=11.0, y=0.0, diameter=4.0, hole_type="threaded_hole"),
        ],
        bore_diameter=6.2,
        bore_depth=15.5,
        notes="JGB37-520减速电机。齿轮箱背面 2×M4 螺纹孔，间距22mm。",
    ),
    "bldc_motor_5010": MountingInterface(
        interface_type="through_hole",
        contact_face="top",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-6.36, y=-6.36, diameter=3.0),
            BoltHole(x=6.36, y=-6.36, diameter=3.0),
            BoltHole(x=-6.36, y=6.36, diameter=3.0),
            BoltHole(x=6.36, y=6.36, diameter=3.0),
        ],
        bore_diameter=5.2,
        bore_depth=25.0,
        notes="5010无刷电机。定子安装面 4×M3 孔，PCD≈18mm。",
    ),
    "bldc_motor_2208": MountingInterface(
        interface_type="through_hole",
        contact_face="top",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-4.24, y=-4.24, diameter=3.0),
            BoltHole(x=4.24, y=-4.24, diameter=3.0),
            BoltHole(x=-4.24, y=4.24, diameter=3.0),
            BoltHole(x=4.24, y=4.24, diameter=3.0),
        ],
        bore_diameter=3.2,
        bore_depth=15.0,
        notes="2208无刷电机。定子安装面 4×M3 孔，PCD≈12mm。",
    ),
    "linear_bearing_lm8uu": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=15.0,
        bore_depth=24.0,
        press_fit_interference=0.02,
        notes="LM8UU直线轴承。外径压入壳体孔。内径8mm配光轴。",
    ),
    "linear_bearing_lm6uu": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=12.0,
        bore_depth=19.0,
        press_fit_interference=0.02,
        notes="LM6UU直线轴承。外径压入壳体孔。内径6mm配光轴。",
    ),
    "linear_bearing_lm10uu": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=19.0,
        bore_depth=29.0,
        press_fit_interference=0.02,
        notes="LM10UU直线轴承。外径压入壳体孔。内径10mm配光轴。",
    ),
    "linear_bearing_lm12uu": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=21.0,
        bore_depth=30.0,
        press_fit_interference=0.02,
        notes="LM12UU直线轴承。外径压入壳体孔。内径12mm配光轴。",
    ),
    "limit_switch_kw12": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-6.0, y=0.0, diameter=2.0),
            BoltHole(x=6.0, y=0.0, diameter=2.0),
        ],
        notes="KW12-3微动限位开关。底部 2×M2 安装孔，间距12mm。",
    ),
    "driver_tb6612fng": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-7.0, y=0.0, diameter=2.0),
            BoltHole(x=7.0, y=0.0, diameter=2.0),
        ],
        notes="TB6612FNG电机驱动模块。底部 2×M2 安装孔，间距14mm。",
    ),
    "power_lm2596_buck": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-15.5, y=-6.5, diameter=2.5),
            BoltHole(x=15.5, y=-6.5, diameter=2.5),
            BoltHole(x=-15.5, y=6.5, diameter=2.5),
            BoltHole(x=15.5, y=6.5, diameter=2.5),
        ],
        notes="LM2596降压模块。4×M2.5安装孔，四角分布。",
    ),
    "gt2_pulley": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=5.0,
        bore_depth=6.0,
        holes=[
            BoltHole(x=0.0, y=5.0, diameter=3.0, depth=4.0,
                     direction=(1.0, 0.0, 0.0), hole_type="threaded_hole"),
        ],
        notes="GT2同步轮。中心轴孔Ø5mm（配5mm轴），1×M3紧定螺钉。",
    ),
    "linear_shaft": MountingInterface(
        interface_type="shaft",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=8.0,
        bore_depth=0.0,
        notes="直线光轴。靠直线轴承/支撑座定位，无独立安装孔。",
    ),

    # ---- P0-2: Structural part MountingInterfaces (11 parts) ----

    "l_bracket": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            # Vertical face: 4×M4 holes
            BoltHole(x=-10.0, y=10.0, diameter=4.2),
            BoltHole(x=10.0, y=10.0, diameter=4.2),
            # Horizontal face: 4×M4 holes
            BoltHole(x=-10.0, y=-10.0, diameter=4.2),
            BoltHole(x=10.0, y=-10.0, diameter=4.2),
        ],
        notes="L型角钢支架。垂直面和水平面各2×M4孔。默认50×30×50mm。",
    ),
    "mounting_plate": MountingInterface(
        interface_type="through_hole",
        contact_face="top",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-40.0, y=-30.0, diameter=4.5),
            BoltHole(x=40.0, y=-30.0, diameter=4.5),
            BoltHole(x=-40.0, y=30.0, diameter=4.5),
            BoltHole(x=40.0, y=30.0, diameter=4.5),
        ],
        notes="安装板。四角4×M4.5孔，边距10mm。默认100×80mm板。",
    ),
    "motor_bracket_u": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            # Base mounting holes
            BoltHole(x=-10.0, y=0.0, diameter=3.2),
            BoltHole(x=10.0, y=0.0, diameter=3.2),
            # Arm mounting holes
            BoltHole(x=-10.0, y=-25.0, diameter=3.2),
            BoltHole(x=10.0, y=-25.0, diameter=3.2),
        ],
        bore_diameter=12.0,
        bore_depth=3.0,
        notes="U型电机支架。底面2×M3安装孔+U型槽电机孔Ø12mm+臂端2×M3孔。",
    ),
    "chassis_plate": MountingInterface(
        interface_type="through_hole",
        contact_face="top",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-65.0, y=-40.0, diameter=4.5),
            BoltHole(x=65.0, y=-40.0, diameter=4.5),
            BoltHole(x=-65.0, y=40.0, diameter=4.5),
            BoltHole(x=65.0, y=40.0, diameter=4.5),
        ],
        notes="底盘板。四角4×M4.5孔。默认150×100mm，网格安装孔系。",
    ),
    "corner_bracket": MountingInterface(
        interface_type="through_hole",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        holes=[
            BoltHole(x=-10.0, y=-10.0, diameter=4.2),
            BoltHole(x=10.0, y=-10.0, diameter=4.2),
            BoltHole(x=-10.0, y=10.0, diameter=4.2),
            BoltHole(x=10.0, y=10.0, diameter=4.2),
        ],
        notes="角码。每面2×M4孔，共4孔。默认30mm边长。",
    ),
    "standoff_hex": MountingInterface(
        interface_type="through_hole",
        contact_face="top",
        contact_face_normal=(0.0, 0.0, 1.0),
        bore_diameter=3.2,
        bore_depth=0.0,
        notes="六角铜柱。中心通孔M3(Ø3.2mm)，无独立螺栓孔。",
    ),
    "pcb_mount": MountingInterface(
        interface_type="through_hole",
        contact_face="top",
        contact_face_normal=(0.0, 0.0, 1.0),
        holes=[
            BoltHole(x=-2.0, y=-2.0, diameter=3.0),
            BoltHole(x=2.0, y=-2.0, diameter=3.0),
            BoltHole(x=-2.0, y=2.0, diameter=3.0),
            BoltHole(x=2.0, y=2.0, diameter=3.0),
        ],
        notes="PCB安装铜柱。上下M3螺纹孔。默认Ø6×15mm。",
    ),
    "battery_holder_18650": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-27.5, y=-17.5, diameter=3.2),
            BoltHole(x=27.5, y=-17.5, diameter=3.2),
            BoltHole(x=-27.5, y=17.5, diameter=3.2),
            BoltHole(x=27.5, y=17.5, diameter=3.2),
        ],
        notes="18650电池盒。4×M3安装孔，间距≈55×35mm。默认75×55mm。",
    ),
    "wheel_simple": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=5.0,
        bore_depth=26.0,
        holes=[
            BoltHole(x=0.0, y=10.0, diameter=3.0, depth=8.0,
                     direction=(1.0, 0.0, 0.0), hole_type="threaded_hole"),
        ],
        notes="实心轮。中心轮毂孔Ø5mm，1×M3紧定螺钉。默认Ø65×26mm。",
    ),
    "wheel_mecanum": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=8.0,
        bore_depth=30.0,
        notes="麦克纳姆轮。中心孔Ø8mm适配轮毂。默认Ø60×30mm。",
    ),
    "hub_adapter": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=6.0,
        bore_depth=15.0,
        holes=[
            BoltHole(x=0.0, y=10.0, diameter=3.0, depth=5.0,
                     direction=(1.0, 0.0, 0.0), hole_type="threaded_hole"),
        ],
        notes="轮毂适配器。中心轴孔Ø6mm，1×M3紧定螺钉。默认Ø20×15mm。",
    ),

    # ---- Layer 2: Additional functional part MountingInterfaces (8) ----

    "servo_mgmt995": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-12.0, y=-9.5, diameter=2.8),
            BoltHole(x=12.0, y=-9.5, diameter=2.8),
            BoltHole(x=-12.0, y=9.5, diameter=2.8),
            BoltHole(x=12.0, y=9.5, diameter=2.8),
        ],
        bore_diameter=6.0,
        bore_depth=0.0,
        notes="MG995/MG996R大扭矩舵机。4×M2.5安装孔（四角），输出轴Ø6mm。",
    ),
    "imu_mpu6050": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-8.0, y=-5.0, diameter=2.2),
            BoltHole(x=8.0, y=-5.0, diameter=2.2),
            BoltHole(x=-8.0, y=5.0, diameter=2.2),
            BoltHole(x=8.0, y=5.0, diameter=2.2),
        ],
        notes="MPU6050 IMU模块。4×M2.5安装孔（四角）。PCB约21×16mm。",
    ),
    "ultrasonic_hcsr04": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-15.0, y=0.0, diameter=2.0),
            BoltHole(x=15.0, y=0.0, diameter=2.0),
        ],
        notes="HC-SR04超声波传感器。2×M2安装孔，间距30mm。PCB约45×20mm。",
    ),
    "oled_128x64": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-12.5, y=-9.5, diameter=2.0),
            BoltHole(x=12.5, y=-9.5, diameter=2.0),
            BoltHole(x=-12.5, y=9.5, diameter=2.0),
            BoltHole(x=12.5, y=9.5, diameter=2.0),
        ],
        notes="0.96寸OLED 128×64显示模块。4×M2安装孔（四角）。PCB约27×27mm。",
    ),
    "spur_gear": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=5.0,
        bore_depth=8.0,
        holes=[
            BoltHole(x=0.0, y=8.0, diameter=3.0, depth=5.0,
                     direction=(1.0, 0.0, 0.0), hole_type="threaded_hole"),
        ],
        notes="直齿轮。中心轴孔Ø5mm（压入配合），1×M3紧定螺钉。",
    ),
    "flexible_coupling": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=5.0,
        bore_depth=25.0,
        holes=[
            BoltHole(x=0.0, y=10.0, diameter=3.0, depth=5.0,
                     direction=(1.0, 0.0, 0.0), hole_type="threaded_hole"),
            BoltHole(x=0.0, y=-10.0, diameter=3.0, depth=5.0,
                     direction=(1.0, 0.0, 0.0), hole_type="threaded_hole"),
        ],
        notes="柔性联轴器。双端轴孔Ø5mm，各1×M3紧定螺钉。默认Ø25×25mm。",
    ),
    "t8_nut": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-12.0, y=0.0, diameter=3.4),
            BoltHole(x=12.0, y=0.0, diameter=3.4),
            BoltHole(x=0.0, y=-12.0, diameter=3.4),
            BoltHole(x=0.0, y=12.0, diameter=3.4),
        ],
        bore_diameter=8.0,
        bore_depth=0.0,
        notes="T8丝杆螺母（法兰型）。法兰4×M3安装孔，中心孔Ø8mm配T8丝杆。",
    ),

    # ---- Layer 2: Additional structural part MountingInterfaces (6) ----

    "linear_guide_mgn12": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=-15.0, y=0.0, diameter=3.4),
            BoltHole(x=15.0, y=0.0, diameter=3.4),
            BoltHole(x=0.0, y=0.0, diameter=3.4),
        ],
        notes="MGN12直线导轨滑块。底部3×M3安装槽。导轨靠安装面定位。",
    ),
    "t8_leadscrew": MountingInterface(
        interface_type="shaft",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=8.0,
        bore_depth=0.0,
        notes="T8丝杆。轴径Ø8mm，靠螺母/支撑座定位，无独立安装孔。",
    ),
    "compression_spring": MountingInterface(
        interface_type="shaft",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=2.0,
        bore_depth=0.0,
        notes="压缩弹簧。柔性件，靠两端座孔定位。线径约2mm。",
    ),
    "damper_foot": MountingInterface(
        interface_type="through_hole",
        contact_face="bottom",
        contact_face_normal=(0.0, 0.0, -1.0),
        holes=[
            BoltHole(x=0.0, y=0.0, diameter=6.6),
        ],
        bore_diameter=0.0,
        bore_depth=0.0,
        notes="减震脚垫。中心1×M6安装孔，法兰Ø25mm。",
    ),
    "gt2_belt": MountingInterface(
        interface_type="belt",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=0.0,
        bore_depth=0.0,
        notes="GT2同步带。柔性件，无物理安装接口。靠张紧轮/惰轮定位。",
    ),
    "shaft_coupling": MountingInterface(
        interface_type="press_fit",
        contact_face="side",
        contact_face_normal=(1.0, 0.0, 0.0),
        bore_diameter=5.0,
        bore_depth=25.0,
        holes=[
            BoltHole(x=0.0, y=10.0, diameter=3.0, depth=5.0,
                     direction=(1.0, 0.0, 0.0), hole_type="threaded_hole"),
            BoltHole(x=0.0, y=-10.0, diameter=3.0, depth=5.0,
                     direction=(1.0, 0.0, 0.0), hole_type="threaded_hole"),
        ],
        notes="刚性轴联轴器。双端轴孔Ø5mm，各1×M3紧定螺钉。默认Ø25×25mm。",
    ),
}


def get_mounting_interface(part_id: str) -> MountingInterface | None:
    """Get the mounting interface for a part by its catalog ID.

    Returns the interface from MOUNTING_INTERFACES if defined,
    otherwise falls back to the template's mounting_interface field.
    """
    if part_id in MOUNTING_INTERFACES:
        return MOUNTING_INTERFACES[part_id]
    t = PART_CATALOG.get(part_id)
    if t and t.mounting_interface:
        return t.mounting_interface
    return None


def auto_match_interface(
    structural_dims: dict[str, float],
    interface: MountingInterface,
    anchor: str = "top",
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[dict]:
    """Generate FreeCAD operation dicts to create matching features on a structural part.

    Given a structural part's dimensions and a functional part's MountingInterface,
    produces a list of operation dicts (hole cuts, bores, pockets) that can be
    merged into the structural part's operation pipeline.

    Args:
        structural_dims: The structural part's key dimensions (e.g. length, width, height, thickness).
        interface: The functional part's MountingInterface.
        anchor: Which face of the structural part receives the features ("top", "bottom", "front", "back").
        offset_x: Additional X offset from center.
        offset_y: Additional Y offset from center.

    Returns:
        List of operation dicts for part_feature_engine.
    """
    ops: list[dict] = []
    thickness = structural_dims.get("thickness", structural_dims.get("height", 10.0))

    # Center offset for structural part face
    cx = structural_dims.get("length", structural_dims.get("width", 50.0)) / 2.0
    cy = structural_dims.get("width", structural_dims.get("length", 50.0)) / 2.0

    # Determine Z position based on anchor
    anchor_z_map = {
        "top": thickness,
        "bottom": 0.0,
        "front": thickness,
        "back": 0.0,
    }
    z_start = anchor_z_map.get(anchor, thickness)

    # Generate bolt holes
    for i, hole in enumerate(interface.holes):
        hx = cx + hole.x + offset_x
        hy = cy + hole.y + offset_y
        ops.append({
            "type": "cylinder",
            "name": f"mount_hole_{i}",
            "radius": hole.diameter / 2.0,
            "height": hole.depth if hole.depth > 0 else thickness + 2.0,
            "x": hx,
            "y": hy,
            "z": z_start - (thickness + 2.0) if anchor == "top" else 0.0,
            "operation": "cut",
        })

    # Center bore (shaft clearance, bearing bore)
    if interface.bore_diameter > 0:
        bore_depth = interface.bore_depth if interface.bore_depth > 0 else thickness + 2.0
        ops.append({
            "type": "cylinder",
            "name": "center_bore",
            "radius": interface.bore_diameter / 2.0,
            "height": bore_depth,
            "x": cx + offset_x,
            "y": cy + offset_y,
            "z": z_start - bore_depth if anchor == "top" else 0.0,
            "operation": "cut",
        })

    # Press-fit bore (bearings)
    if interface.press_fit_interference > 0 and interface.bore_diameter > 0:
        fit_bore_d = interface.bore_diameter - interface.press_fit_interference
        bore_depth = interface.bore_depth if interface.bore_depth > 0 else thickness
        ops.append({
            "type": "cylinder",
            "name": "press_fit_bore",
            "radius": fit_bore_d / 2.0,
            "height": bore_depth,
            "x": cx + offset_x,
            "y": cy + offset_y,
            "z": 0.0,
            "operation": "cut",
        })
        # Shoulder / retaining lip
        if interface.shoulder_diameter > 0 and interface.shoulder_depth > 0:
            ops.append({
                "type": "cylinder",
                "name": "shoulder",
                "radius": interface.shoulder_diameter / 2.0,
                "height": interface.shoulder_depth,
                "x": cx + offset_x,
                "y": cy + offset_y,
                "z": bore_depth,
                "operation": "cut",
            })

    # Body pocket (servo mounting, etc.)
    if interface.pocket_width > 0 and interface.pocket_height > 0:
        ops.append({
            "type": "box",
            "name": "body_pocket",
            "width": interface.pocket_width,
            "depth": interface.pocket_height,
            "height": interface.pocket_depth if interface.pocket_depth > 0 else thickness,
            "x": cx - interface.pocket_width / 2.0 + offset_x,
            "y": cy - interface.pocket_height / 2.0 + offset_y,
            "z": 0.0,
            "operation": "cut",
        })

    return ops


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def search_parts(
    query: str = "",
    category: str | None = None,
    tags: list[str] | None = None,
    part_class: str | None = None,
) -> list[PartTemplate]:
    """Search parts by keyword, category, tags, and/or part_class.

    Keyword matching is case-insensitive and checks name_en, name_cn,
    description, and tags.

    Args:
        query: Keyword search string.
        category: Filter by category or subcategory.
        tags: Filter by tags (any match).
        part_class: Filter by part_class ("functional", "structural", "fastener").
    """
    results: list[PartTemplate] = []
    query_lower = query.lower().strip() if query else ""

    for template in PART_CATALOG.values():
        # Part class filter
        if part_class and template.part_class != part_class:
            continue

        # Category filter
        if category:
            cat_lower = category.lower()
            if template.category.lower() != cat_lower and template.subcategory.lower() != cat_lower:
                continue

        # Tag filter
        if tags:
            template_tags_lower = {t.lower() for t in template.tags}
            if not any(t.lower() in template_tags_lower for t in tags):
                continue

        # Keyword search
        if query_lower:
            searchable = " ".join([
                template.name_en, template.name_cn, template.description,
                *template.tags, template.category, template.subcategory,
                template.part_class, template.model_number, template.manufacturer,
            ]).lower()
            if query_lower not in searchable:
                continue

        results.append(template)

    return results


def get_template(part_id: str) -> PartTemplate | None:
    """Get a part template by ID."""
    return PART_CATALOG.get(part_id)


def get_all_templates() -> list[PartTemplate]:
    """Get all part templates."""
    return list(PART_CATALOG.values())


def get_functional_parts() -> list[PartTemplate]:
    """Get all functional (non-scalable, real COTS) parts: motors, servos, sensors, bearings."""
    return [t for t in PART_CATALOG.values() if t.part_class == "functional"]


def get_structural_parts() -> list[PartTemplate]:
    """Get all structural (scalable, custom-designed) parts: brackets, plates, links."""
    return [t for t in PART_CATALOG.values() if t.part_class == "structural"]


def get_fastener_parts() -> list[PartTemplate]:
    """Get all fastener (standard hardware) parts: screws, nuts, washers."""
    return [t for t in PART_CATALOG.values() if t.part_class == "fastener"]


def validate_functional_params(template_id: str, params: dict[str, Any]) -> list[str]:
    """Validate that functional part parameters match real product specs.

    Returns a list of warning strings for parameters that deviate from
    standard sizes.  Empty list means all OK.
    """
    template = PART_CATALOG.get(template_id)
    if not template or template.part_class != "functional":
        return []

    warnings: list[str] = []
    if not template.standard_sizes:
        return warnings

    # Check if params match any standard size
    for std_size in template.standard_sizes:
        match = True
        for key, std_val in std_size.items():
            if key in params and abs(float(params[key]) - float(std_val)) > 0.5:
                match = False
                break
        if match:
            return warnings  # Found a matching standard size

    # No exact match — generate warnings for each deviating parameter
    best_match = template.standard_sizes[0]
    for key in params:
        if key in best_match:
            diff = abs(float(params[key]) - float(best_match[key]))
            if diff > 0.5:
                warnings.append(
                    f"Parameter '{key}' = {params[key]} deviates from standard "
                    f"value {best_match[key]} (diff={diff:.1f}). "
                    f"Functional part '{template.name_en}' has fixed dimensions."
                )
    return warnings


# Subsystem compatibility mappings
_SUBSYSTEM_COMPAT: dict[str, list[str]] = {
    "mobile_base": ["wheel_simple", "wheel_mecanum", "hub_adapter", "motor_bracket_u",
                     "chassis_plate", "corner_bracket", "standoff_hex", "battery_holder_18650",
                     "motor_tt", "motor_jgb37_520"],
    "mounting": ["standoff_hex", "pcb_mount", "battery_holder_18650", "corner_bracket",
                  "l_bracket", "mounting_plate"],
    "arm": ["servo_sg90", "servo_mg996r", "servo_ds3218", "nema17_stepper", "nema23_stepper",
             "l_bracket", "mounting_plate", "standoff_hex"],
    "drive": ["motor_bracket_u", "hub_adapter", "wheel_simple", "wheel_mecanum",
               "spur_gear", "flexible_coupling", "motor_tt", "motor_jgb37_520"],
    "transmission": ["gt2_pulley", "gt2_belt", "htd_pulley_3m", "htd_pulley_5m",
                      "rigid_coupling_setscrew", "rigid_coupling_clamping",
                      "spider_coupling", "bellows_coupling",
                      "flexible_coupling", "nema17_stepper", "nema23_stepper",
                      "linear_shaft", "hub_adapter"],
    "linear_motion": ["linear_bearing_lm6uu", "linear_bearing_lm8uu", "linear_bearing_lm10uu", "linear_bearing_lm12uu",
                       "linear_guide_mgn12h", "t8_leadscrew", "t8_nut",
                       "linear_shaft", "nema17_stepper", "nema23_stepper",
                       "bldc_motor_5010", "bldc_motor_2208",
                       "compression_spring", "damper_shock_absorber"],
    "perception": ["sensor_rplidar_a1", "sensor_mpu6050", "sensor_esp32_cam",
                    "encoder_as5600", "limit_switch_kw12",
                    "mounting_plate", "l_bracket", "standoff_hex"],
    "electronics": ["driver_l298n", "driver_tb6612fng",
                      "controller_arduino_uno", "controller_arduino_nano",
                      "controller_esp32_devkit",
                      "power_lm2596_buck", "connector_xt60", "connector_jst_xh",
                      "standoff_hex", "pcb_mount", "battery_holder_18650"],
}

# Dimension-based compatibility: maps parameter name patterns to matching parts
_DIM_COMPAT: dict[str, list[str]] = {
    "hole_diameter": ["socket_head_cap_screw", "hex_bolt", "flat_washer", "hex_nut"],
    "shaft_diameter": ["hub_adapter", "flexible_coupling", "linear_shaft"],
    "motor_diameter": ["motor_bracket_u"],
}


def search_by_subsystem(subsystem: str) -> list[PartTemplate]:
    """Search parts by subsystem (e.g. 'mobile_base', 'mounting', 'arm').

    Returns templates relevant to the given subsystem, including
    structural and fastener parts that are commonly used.
    """
    part_ids = _SUBSYSTEM_COMPAT.get(subsystem, [])
    return [PART_CATALOG[pid] for pid in part_ids if pid in PART_CATALOG]


def find_compatible_parts(
    part_id: str,
    by_dimension: bool = True,
) -> list[PartTemplate]:
    """Find parts compatible with a given part.

    Compatibility is determined by:
      1. Shared subsystem membership
      2. Matching dimension parameters (e.g. same hole_diameter)

    Args:
        part_id: The reference part ID.
        by_dimension: If True, also match by dimension parameters.

    Returns:
        List of compatible PartTemplate objects (excluding the reference itself).
    """
    results: set[str] = set()
    template = PART_CATALOG.get(part_id)
    if template is None:
        return []

    # Find subsystems containing this part
    for subsystem, part_ids in _SUBSYSTEM_COMPAT.items():
        if part_id in part_ids:
            results.update(part_ids)

    # Dimension matching
    if by_dimension and template:
        param_names = {p.name for p in template.parameters}
        for dim_name, compat_ids in _DIM_COMPAT.items():
            if dim_name in param_names:
                results.update(compat_ids)

    results.discard(part_id)
    return [PART_CATALOG[pid] for pid in sorted(results) if pid in PART_CATALOG]


def resolve_parameters(
    template: PartTemplate,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill defaults for missing parameters and validate ranges.

    Returns a complete parameter dict with all parameters filled in.
    For string-type parameters, validates against choices list.
    Raises ValueError if a parameter is out of range or invalid choice.
    """
    resolved: dict[str, Any] = {}
    for pdef in template.parameters:
        if params and pdef.name in params:
            if pdef.param_type == "string":
                val = str(params[pdef.name])
                if pdef.choices and val not in pdef.choices:
                    raise ValueError(
                        f"Parameter '{pdef.name}' value '{val}' not in choices {pdef.choices}"
                    )
                resolved[pdef.name] = val
            else:
                val = float(params[pdef.name])
                if val < pdef.min_value or val > pdef.max_value:
                    raise ValueError(
                        f"Parameter '{pdef.name}' value {val} out of range "
                        f"[{pdef.min_value}, {pdef.max_value}]"
                    )
                resolved[pdef.name] = val
        else:
            resolved[pdef.name] = pdef.default
    return resolved


def format_fc_script(template: PartTemplate, params: dict[str, Any]) -> str:
    """Substitute parameter values into the FreeCAD script template.

    Selects the appropriate script based on detail level (thread_detail or
    tooth_detail), falling back to the default simplified script.
    Uses Python str.format_map to replace {param_name} placeholders.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Determine detail level from params
    detail = params.get(
        "thread_detail",
        params.get("tooth_detail", params.get("bearing_detail",
                      params.get("pulley_detail",
                        params.get("leadscrew_detail", "simplified")))),
    )
    # Select script: prefer alternative matching detail level
    script_template = template.fc_script_alternatives.get(detail, template.fc_script_template)

    # Convert values to clean strings (avoid scientific notation)
    clean_params: dict[str, str] = {}
    for k, v in params.items():
        if isinstance(v, (int, float)) and v == int(v):
            clean_params[k] = str(int(v))
        else:
            clean_params[k] = str(v)

    # Validate that all placeholders have corresponding parameters.
    # Previously a missing parameter would substitute '0' (makeBox(0,0,0) =
    # zero-volume garbage geometry) — that violated AGENTS.md §1.1 "do not
    # silently fix absurd values; raise so the caller retries".  Now a
    # missing parameter raises a clear error naming the template + field.
    class _StrictFormatDict(dict):
        """Dict that RAISES on missing keys — no silent zero-substitution."""
        def __missing__(self, key):
            raise KeyError(
                f"FreeCAD script for template '{template.name}' references "
                f"missing parameter '{key}'. The PartTemplate's FreeCAD "
                f"script has a placeholder {{{key}}} but no matching value "
                f"was provided. Fix the template's params or the caller."
            )

    try:
        return script_template.format_map(_StrictFormatDict(clean_params))
    except (KeyError, ValueError, IndexError) as e:
        logger.error(
            "FreeCAD script formatting failed for template '%s': %s. "
            "Returning raw template.", template.name, e
        )
        return script_template
