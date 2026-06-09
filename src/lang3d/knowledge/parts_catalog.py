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
    "fastener": ["screw", "nut", "washer", "bolt"],
    "bearing": ["ball_bearing", "linear_bearing"],

    "actuator": ["servo", "stepper", "dc_motor", "bldc"],

    "shaft": ["linear", "coupling", "leadscrew"],

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
r = {nominal_diameter} * 1.1
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
r = {thread_diameter} * 1.0
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
    except Exception:
        pass

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
for dx, dy in [(7.5, 7.5), ({body_size}-7.5, 7.5), (7.5, {body_size}-7.5), ({body_size}-7.5, {body_size}-7.5)]:
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
    except Exception:
        pass

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
    except Exception:
        pass

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


# ---------------------------------------------------------------------------
# Standard parts catalog (25+ templates)
# ---------------------------------------------------------------------------

PART_CATALOG: dict[str, PartTemplate] = {
    "socket_head_cap_screw": PartTemplate(
        id="socket_head_cap_screw",
        name_en="Socket Head Cap Screw",
        name_cn="内六角圆柱头螺钉",
        category="fastener",
        subcategory="screw",
        description="DIN 912 / ISO 4762 内六角圆柱头螺钉，标准机械连接件",
        tags=["螺钉", "内六角", "DIN912", "紧固件", "screw", "socket head"],
        part_class="fastener", scalable=False,
        parameters=[
            ParamDef("thread_diameter", "螺纹直径", "mm", 3, 1, 30, 0.5),
            ParamDef("length", "螺钉长度", "mm", 10, 2, 200, 1),
            ParamDef("head_diameter", "头部直径", "mm", 5.5, 2, 50, 0.5, fixed=False),
            ParamDef("thread_detail", "螺纹细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("thread_pitch", "螺距", "mm", 1.0, 0.25, 4.0, 0.05),
        ],
        fc_script_template=_SOCKET_HEAD_CAP_SCREW_SCRIPT,
        fc_script_alternatives={"realistic": _SOCKET_HEAD_CAP_SCREW_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"thread_diameter": 3, "length": 10, "head_diameter": 5.5, "thread_pitch": 0.5},
            {"thread_diameter": 3, "length": 20, "head_diameter": 5.5, "thread_pitch": 0.5},
            {"thread_diameter": 4, "length": 16, "head_diameter": 7.0, "thread_pitch": 0.7},
            {"thread_diameter": 5, "length": 20, "head_diameter": 8.5, "thread_pitch": 0.8},
            {"thread_diameter": 6, "length": 25, "head_diameter": 10.0, "thread_pitch": 1.0},
            {"thread_diameter": 8, "length": 30, "head_diameter": 13.0, "thread_pitch": 1.25},
        ],
        notes="螺纹为简化圆柱体表示，非真实螺纹几何。选择 realistic 启用螺旋扫掠螺纹。",
    ),

    "hex_nut": PartTemplate(
        id="hex_nut",
        name_en="Hex Nut",
        name_cn="六角螺母",
        category="fastener",
        subcategory="nut",
        description="DIN 934 / ISO 4032 六角螺母，标准紧固件",
        tags=["螺母", "六角", "DIN934", "紧固件", "nut", "hex"],
        part_class="fastener", scalable=False,
        parameters=[
            ParamDef("nominal_diameter", "公称直径", "mm", 3, 1, 30, 0.5),
            ParamDef("thread_detail", "螺纹细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("thread_pitch", "螺距", "mm", 1.0, 0.25, 4.0, 0.05),
        ],
        fc_script_template=_HEX_NUT_SCRIPT,
        fc_script_alternatives={"realistic": _HEX_NUT_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"nominal_diameter": 3, "thread_pitch": 0.5},
            {"nominal_diameter": 4, "thread_pitch": 0.7},
            {"nominal_diameter": 5, "thread_pitch": 0.8},
            {"nominal_diameter": 6, "thread_pitch": 1.0},
            {"nominal_diameter": 8, "thread_pitch": 1.25},
            {"nominal_diameter": 10, "thread_pitch": 1.5},
        ],
    ),

    "flat_washer": PartTemplate(
        id="flat_washer",
        name_en="Flat Washer",
        name_cn="平垫圈",
        category="fastener",
        subcategory="washer",
        description="DIN 125 / ISO 7089 平垫圈，配合螺栓/螺钉使用",
        tags=["垫圈", "平垫", "DIN125", "紧固件", "washer", "flat"],
        part_class="fastener", scalable=False,
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 3.2, 1, 30, 0.1),
            ParamDef("outer_diameter", "外径", "mm", 7.0, 2, 60, 0.1),
            ParamDef("thickness", "厚度", "mm", 0.5, 0.1, 5, 0.1),
        ],
        fc_script_template=_FLAT_WASHER_SCRIPT,
        standard_sizes=[
            {"inner_diameter": 3.2, "outer_diameter": 7.0, "thickness": 0.5},
            {"inner_diameter": 4.3, "outer_diameter": 9.0, "thickness": 0.8},
            {"inner_diameter": 5.3, "outer_diameter": 10.0, "thickness": 1.0},
            {"inner_diameter": 6.4, "outer_diameter": 12.0, "thickness": 1.6},
        ],
    ),

    "hex_bolt": PartTemplate(
        id="hex_bolt",
        name_en="Hex Bolt",
        name_cn="六角螺栓",
        category="fastener",
        subcategory="bolt",
        description="DIN 933 / ISO 4014 六角螺栓，全牙标准件",
        tags=["螺栓", "六角", "DIN933", "紧固件", "bolt", "hex"],
        part_class="fastener", scalable=False,
        parameters=[
            ParamDef("thread_diameter", "螺纹直径", "mm", 4, 2, 30, 0.5),
            ParamDef("length", "螺栓长度", "mm", 20, 5, 300, 1),
            ParamDef("thread_detail", "螺纹细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("thread_pitch", "螺距", "mm", 1.0, 0.25, 4.0, 0.05),
        ],
        fc_script_template=_HEX_BOLT_SCRIPT,
        fc_script_alternatives={"realistic": _HEX_BOLT_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"thread_diameter": 4, "length": 20, "thread_pitch": 0.7},
            {"thread_diameter": 5, "length": 25, "thread_pitch": 0.8},
            {"thread_diameter": 6, "length": 30, "thread_pitch": 1.0},
            {"thread_diameter": 8, "length": 40, "thread_pitch": 1.25},
            {"thread_diameter": 10, "length": 50, "thread_pitch": 1.5},
        ],
        notes="螺纹为简化圆柱体表示。选择 realistic 启用螺旋扫掠螺纹。",
    ),

    "bearing_608": PartTemplate(
        id="bearing_608",
        name_en="608 Deep Groove Ball Bearing",
        name_cn="608 深沟球轴承",
        category="bearing",
        subcategory="ball_bearing",
        description="608 系列深沟球轴承，常用于滑轮、滑板轮、3D打印机",
        tags=["轴承", "608", "深沟球", "bearing", "ball bearing", "skateboard"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="608-2RS",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 8, 1, 100, 1),
            ParamDef("outer_diameter", "外径", "mm", 22, 5, 200, 1),
            ParamDef("width", "宽度", "mm", 7, 1, 50, 1),
            ParamDef("bearing_detail", "轴承细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("ball_count", "滚珠数量", "", 0, 0, 50, 1),
        ],
        fc_script_template=_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 8, "outer_diameter": 22, "width": 7},
        ],
    ),

    "bearing_623": PartTemplate(
        id="bearing_623",
        name_en="623 Deep Groove Ball Bearing",
        name_cn="623 深沟球轴承",
        category="bearing",
        subcategory="ball_bearing",
        description="623 系列深沟球轴承，小型精密轴承",
        tags=["轴承", "623", "深沟球", "bearing", "small"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="623-2RS",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 3, 1, 100, 1),
            ParamDef("outer_diameter", "外径", "mm", 10, 5, 200, 1),
            ParamDef("width", "宽度", "mm", 4, 1, 50, 1),
            ParamDef("bearing_detail", "轴承细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("ball_count", "滚珠数量", "", 0, 0, 50, 1),
        ],
        fc_script_template=_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 3, "outer_diameter": 10, "width": 4},
        ],
    ),

    "bearing_625": PartTemplate(
        id="bearing_625",
        name_en="625 Deep Groove Ball Bearing",
        name_cn="625 深沟球轴承",
        category="bearing",
        subcategory="ball_bearing",
        description="625 系列深沟球轴承，常用于3D打印机",
        tags=["轴承", "625", "深沟球", "bearing", "3D printer"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="625-2RS",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 5, 1, 100, 1),
            ParamDef("outer_diameter", "外径", "mm", 16, 5, 200, 1),
            ParamDef("width", "宽度", "mm", 5, 1, 50, 1),
            ParamDef("bearing_detail", "轴承细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("ball_count", "滚珠数量", "", 0, 0, 50, 1),
        ],
        fc_script_template=_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 5, "outer_diameter": 16, "width": 5},
        ],
    ),

    "servo_sg90": PartTemplate(
        id="servo_sg90",
        name_en="SG90 Micro Servo",
        name_cn="SG90 微型舵机",
        category="actuator",
        subcategory="servo",
        description="SG90 微型舵机，常用于小型机器人、航模",
        tags=["舵机", "SG90", "servo", "微型", "robot", "RC"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Tower Pro", model_number="SG90",
        parameters=[
            ParamDef("body_length", "机身长度", "mm", 22.2, 10, 100, 0.1),
            ParamDef("body_width", "机身宽度", "mm", 11.8, 5, 50, 0.1),
            ParamDef("body_height", "机身高度", "mm", 31.0, 10, 100, 0.1),
            ParamDef("shaft_diameter", "输出轴直径", "mm", 4.6, 1, 20, 0.1),
            ParamDef("shaft_length", "输出轴长度", "mm", 5.0, 1, 30, 0.1),
        ],
        fc_script_template=_SERVO_SG90_SCRIPT,
        standard_sizes=[
            {"body_length": 22.2, "body_width": 11.8, "body_height": 31.0,
             "shaft_diameter": 4.6, "shaft_length": 5.0},
        ],
    ),

    "servo_mg996r": PartTemplate(
        id="servo_mg996r",
        name_en="MG996R Servo",
        name_cn="MG996R 舵机",
        category="actuator",
        subcategory="servo",
        description="MG996R 大扭力金属齿轮舵机，常用于机器人关节",
        tags=["舵机", "MG996R", "servo", "大扭力", "robot"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Tower Pro", model_number="MG996R",
        parameters=[
            ParamDef("body_length", "机身长度", "mm", 40.7, 20, 100, 0.1),
            ParamDef("body_width", "机身宽度", "mm", 19.7, 10, 60, 0.1),
            ParamDef("body_height", "机身高度", "mm", 42.9, 20, 100, 0.1),
            ParamDef("shaft_diameter", "输出轴直径", "mm", 5.8, 2, 20, 0.1),
            ParamDef("shaft_length", "输出轴长度", "mm", 6.0, 2, 30, 0.1),
        ],
        fc_script_template=_SERVO_MG996R_SCRIPT,
        standard_sizes=[
            {"body_length": 40.7, "body_width": 19.7, "body_height": 42.9,
             "shaft_diameter": 5.8, "shaft_length": 6.0},
        ],
    ),

    "nema17_stepper": PartTemplate(
        id="nema17_stepper",
        name_en="NEMA17 Stepper Motor",
        name_cn="NEMA17 步进电机",
        category="actuator",
        subcategory="stepper",
        description="NEMA17 (42mm) 步进电机，常用于3D打印机和CNC",
        tags=["步进电机", "NEMA17", "stepper", "motor", "42mm", "3D printer"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="NEMA17-42BYGH",
        parameters=[
            ParamDef("body_size", "机身尺寸", "mm", 42.3, 20, 100, 0.1),
            ParamDef("body_length", "机身长度", "mm", 40.0, 20, 100, 1),
            ParamDef("shaft_diameter", "输出轴直径", "mm", 5.0, 2, 20, 0.1),
            ParamDef("shaft_length", "输出轴长度", "mm", 24.0, 5, 50, 0.1),
        ],
        fc_script_template=_NEMA17_SCRIPT,
        standard_sizes=[
            {"body_size": 42.3, "body_length": 40.0,
             "shaft_diameter": 5.0, "shaft_length": 24.0},
        ],
    ),

    "linear_shaft": PartTemplate(
        id="linear_shaft",
        name_en="Linear Shaft",
        name_cn="直线光轴",
        category="shaft",
        subcategory="linear",
        description="直线光轴，配合直线轴承使用，用于直线运动系统",
        tags=["光轴", "直线", "linear shaft", "guide rod", "bearing shaft"],
        parameters=[
            ParamDef("diameter", "直径", "mm", 8, 3, 50, 0.5),
            ParamDef("length", "长度", "mm", 300, 10, 2000, 1),
        ],
        fc_script_template=_LINEAR_SHAFT_SCRIPT,
        standard_sizes=[
            {"diameter": 6, "length": 300},
            {"diameter": 8, "length": 300},
            {"diameter": 8, "length": 500},
            {"diameter": 10, "length": 300},
            {"diameter": 12, "length": 500},
        ],
    ),

    "flexible_coupling": PartTemplate(
        id="flexible_coupling",
        name_en="Flexible Coupling",
        name_cn="弹性联轴器",
        category="shaft",
        subcategory="coupling",
        description="弹性联轴器，连接不同直径的轴，补偿对中偏差",
        tags=["联轴器", "弹性", "coupling", "flexible", "shaft connector"],
        parameters=[
            ParamDef("bore1_diameter", "孔1直径", "mm", 5, 2, 30, 0.5),
            ParamDef("bore2_diameter", "孔2直径", "mm", 8, 2, 30, 0.5),
            ParamDef("outer_diameter", "外径", "mm", 19, 10, 50, 0.5),
            ParamDef("length", "长度", "mm", 25, 10, 60, 1),
        ],
        fc_script_template=_FLEXIBLE_COUPLING_SCRIPT,
        standard_sizes=[
            {"bore1_diameter": 5, "bore2_diameter": 8, "outer_diameter": 19, "length": 25},
            {"bore1_diameter": 5, "bore2_diameter": 10, "outer_diameter": 25, "length": 30},
            {"bore1_diameter": 8, "bore2_diameter": 10, "outer_diameter": 25, "length": 30},
        ],
    ),

    "spur_gear": PartTemplate(
        id="spur_gear",
        name_en="Spur Gear",
        name_cn="直齿轮",
        category="gear",
        subcategory="spur",
        description="直齿轮，用于平行轴间的动力传递",
        tags=["齿轮", "直齿轮", "spur gear", "gear", "transmission"],
        parameters=[
            ParamDef("teeth", "齿数", "", 20, 8, 200, 1),
            ParamDef("module", "模数", "mm", 1.0, 0.3, 10, 0.1),
            ParamDef("thickness", "齿厚", "mm", 6.0, 1, 50, 0.5),
            ParamDef("bore_diameter", "轴孔直径", "mm", 8.0, 2, 50, 0.5),
            ParamDef("tooth_detail", "齿廓细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("pressure_angle", "压力角", "deg", 20.0, 14.5, 30.0, 0.5),
            ParamDef("backlash", "侧隙", "mm", 0.1, 0.0, 1.0, 0.01),
        ],
        fc_script_template=_SPUR_GEAR_SCRIPT,
        fc_script_alternatives={"realistic": _SPUR_GEAR_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"teeth": 20, "module": 1.0, "thickness": 6, "bore_diameter": 8,
             "pressure_angle": 20.0, "backlash": 0.1},
            {"teeth": 30, "module": 1.0, "thickness": 6, "bore_diameter": 8,
             "pressure_angle": 20.0, "backlash": 0.1},
            {"teeth": 16, "module": 1.5, "thickness": 8, "bore_diameter": 10,
             "pressure_angle": 20.0, "backlash": 0.1},
        ],
        notes="齿轮为简化圆柱体表示，非渐开线齿廓。选择 realistic 启用渐开线齿廓建模。",
    ),

    "l_bracket": PartTemplate(
        id="l_bracket",
        name_en="L-Bracket",
        name_cn="L型角钢支架",
        category="structural",
        subcategory="bracket",
        description="L型角钢支架，用于结构连接和加固",
        tags=["支架", "角钢", "L型", "bracket", "angle", "structural"],
        parameters=[
            ParamDef("length", "水平长度", "mm", 50, 10, 300, 1),
            ParamDef("width", "宽度", "mm", 30, 5, 100, 1),
            ParamDef("height", "垂直高度", "mm", 50, 10, 300, 1),
            ParamDef("thickness", "壁厚", "mm", 3, 1, 20, 0.5),
        ],
        fc_script_template=_L_BRACKET_SCRIPT,
        standard_sizes=[
            {"length": 50, "width": 30, "height": 50, "thickness": 3},
            {"length": 80, "width": 40, "height": 80, "thickness": 4},
        ],
    ),

    "mounting_plate": PartTemplate(
        id="mounting_plate",
        name_en="Mounting Plate",
        name_cn="安装板",
        category="structural",
        subcategory="plate",
        description="安装板，带四角安装孔，用于固定组件",
        tags=["安装板", "底板", "mounting plate", "base plate", "structural"],
        parameters=[
            ParamDef("length", "长度", "mm", 100, 10, 500, 1),
            ParamDef("width", "宽度", "mm", 80, 10, 500, 1),
            ParamDef("thickness", "厚度", "mm", 5, 1, 30, 0.5),
            ParamDef("hole_diameter", "安装孔直径", "mm", 4, 1, 20, 0.5),
            ParamDef("hole_margin", "孔边距", "mm", 10, 3, 50, 1),
        ],
        fc_script_template=_MOUNTING_PLATE_SCRIPT,
        standard_sizes=[
            {"length": 100, "width": 80, "thickness": 5, "hole_diameter": 4, "hole_margin": 10},
            {"length": 150, "width": 100, "thickness": 6, "hole_diameter": 5, "hole_margin": 12},
        ],
    ),

    # ---- Wheels ----
    "wheel_simple": PartTemplate(
        id="wheel_simple",
        name_en="Simple Wheel",
        name_cn="实心轮",
        category="mobile_base",
        subcategory="wheel",
        description="实心圆柱轮，适合小型差速/全向底盘",
        tags=["轮子", "实心轮", "wheel", "differential", "mobile base"],
        parameters=[
            ParamDef("outer_diameter", "外径", default=65.0, min_value=10, max_value=300),
            ParamDef("width", "宽度", default=26.0, min_value=5, max_value=100),
            ParamDef("hub_diameter", "轮毂孔径", default=5.0, min_value=2, max_value=30),
        ],
        fc_script_template=_WHEEL_SIMPLE_SCRIPT,
        standard_sizes=[
            {"outer_diameter": 65, "width": 26, "hub_diameter": 5},
            {"outer_diameter": 80, "width": 30, "hub_diameter": 6},
            {"outer_diameter": 100, "width": 35, "hub_diameter": 8},
        ],
    ),
    "wheel_mecanum": PartTemplate(
        id="wheel_mecanum",
        name_en="Mecanum Wheel",
        name_cn="麦克纳姆轮",
        category="mobile_base",
        subcategory="wheel",
        description="麦克纳姆轮，支持全向移动（前后/左右/原地旋转）",
        tags=["麦克纳姆", "全向轮", "mecanum", "omnidirectional"],
        parameters=[
            ParamDef("diameter", "直径", default=60.0, min_value=30, max_value=200),
            ParamDef("width", "宽度", default=30.0, min_value=10, max_value=80),
            ParamDef("num_rollers", "滚轮数", default=8, min_value=4, max_value=16),
            ParamDef("roller_diameter", "滚轮直径", default=10.0, min_value=3, max_value=30),
        ],
        fc_script_template=_WHEEL_MECANUM_SCRIPT,
        standard_sizes=[
            {"diameter": 60, "width": 30, "num_rollers": 8, "roller_diameter": 10},
            {"diameter": 80, "width": 35, "num_rollers": 9, "roller_diameter": 12},
        ],
    ),

    # ---- Hub / Adapter ----
    "hub_adapter": PartTemplate(
        id="hub_adapter",
        name_en="Hub Adapter",
        name_cn="轮毂适配器",
        category="mobile_base",
        subcategory="hub",
        description="电机轴到轮子的适配器，含紧定螺钉孔",
        tags=["轮毂", "适配器", "hub", "adapter", "coupling"],
        parameters=[
            ParamDef("outer_diameter", "外径", default=20.0, min_value=8, max_value=60),
            ParamDef("height", "高度", default=15.0, min_value=5, max_value=50),
            ParamDef("shaft_diameter", "轴径", default=6.0, min_value=2, max_value=20),
            ParamDef("set_screw_size", "紧定螺钉", default=3.0, min_value=1, max_value=8),
        ],
        fc_script_template=_HUB_ADAPTER_SCRIPT,
    ),

    # ---- Motor Brackets ----
    "motor_bracket_u": PartTemplate(
        id="motor_bracket_u",
        name_en="U-Motor Bracket",
        name_cn="U型电机支架",
        category="mobile_base",
        subcategory="motor_bracket",
        description="U型电机固定支架，适合 TT/N20 等小型电机",
        tags=["电机支架", "U型", "motor bracket", "TT motor"],
        parameters=[
            ParamDef("base_length", "底座长", default=30.0, min_value=10, max_value=100),
            ParamDef("base_width", "底座宽", default=25.0, min_value=10, max_value=80),
            ParamDef("thickness", "壁厚", default=3.0, min_value=1, max_value=10),
            ParamDef("bracket_height", "臂高", default=25.0, min_value=5, max_value=60),
            ParamDef("motor_diameter", "电机孔径", default=12.0, min_value=5, max_value=40),
        ],
        fc_script_template=_MOTOR_BRACKET_SCRIPT,
    ),

    # ---- Standoffs ----
    "standoff_hex": PartTemplate(
        id="standoff_hex",
        name_en="Hex Standoff",
        name_cn="六角铜柱",
        category="mounting",
        subcategory="standoff",
        description="六角铜柱/尼龙柱，PCB/层板间隔固定",
        tags=["铜柱", "六角柱", "standoff", "spacer", "PCB"],
        parameters=[
            ParamDef("outer_diameter", "外径", default=5.0, min_value=2, max_value=15),
            ParamDef("length", "长度", default=25.0, min_value=5, max_value=80),
            ParamDef("hole_diameter", "通孔径", default=3.0, min_value=1, max_value=8),
        ],
        fc_script_template=_STANDOFF_SCRIPT,
        standard_sizes=[
            {"outer_diameter": 5, "length": 10, "hole_diameter": 3},
            {"outer_diameter": 5, "length": 25, "hole_diameter": 3},
            {"outer_diameter": 5, "length": 40, "hole_diameter": 3},
        ],
    ),

    # ---- Battery Holder ----
    "battery_holder_18650": PartTemplate(
        id="battery_holder_18650",
        name_en="18650 Battery Holder",
        name_cn="18650 电池盒",
        category="mounting",
        subcategory="battery_holder",
        description="18650 锂电池槽座，可定制 cell 数量",
        tags=["电池盒", "18650", "battery", "holder"],
        parameters=[
            ParamDef("length", "长度", default=75.0, min_value=30, max_value=200),
            ParamDef("width", "宽度", default=55.0, min_value=15, max_value=100),
            ParamDef("height", "高度", default=20.0, min_value=10, max_value=40),
            ParamDef("num_cells", "电池数", default=2, min_value=1, max_value=6),
            ParamDef("cell_diameter", "电池直径", default=18.5, min_value=10, max_value=30),
        ],
        fc_script_template=_BATTERY_HOLDER_SCRIPT,
        standard_sizes=[
            {"length": 75, "width": 40, "height": 20, "num_cells": 2, "cell_diameter": 18.5},
            {"length": 75, "width": 55, "height": 20, "num_cells": 3, "cell_diameter": 18.5},
        ],
    ),

    # ---- Chassis Plate ----
    "chassis_plate": PartTemplate(
        id="chassis_plate",
        name_en="Chassis Plate",
        name_cn="底盘板",
        category="mobile_base",
        subcategory="chassis",
        description="带网格安装孔的底盘板，差速/全向底盘主体结构件",
        tags=["底盘", "安装板", "chassis", "plate", "base"],
        parameters=[
            ParamDef("length", "长度", default=150.0, min_value=30, max_value=500),
            ParamDef("width", "宽度", default=100.0, min_value=20, max_value=500),
            ParamDef("thickness", "厚度", default=3.0, min_value=1, max_value=10),
            ParamDef("hole_diameter", "孔径", default=4.0, min_value=2, max_value=10),
            ParamDef("hole_margin", "边距", default=10.0, min_value=5, max_value=30),
            ParamDef("grid_x", "列数", default=4, min_value=2, max_value=10),
            ParamDef("grid_y", "行数", default=3, min_value=2, max_value=10),
        ],
        fc_script_template=_CHASSIS_PLATE_SCRIPT,
        standard_sizes=[
            {"length": 150, "width": 100, "thickness": 3, "hole_diameter": 4, "hole_margin": 10, "grid_x": 4, "grid_y": 3},
            {"length": 200, "width": 150, "thickness": 5, "hole_diameter": 5, "hole_margin": 12, "grid_x": 5, "grid_y": 4},
        ],
    ),

    # ---- Corner Bracket ----
    "corner_bracket": PartTemplate(
        id="corner_bracket",
        name_en="Corner Bracket",
        name_cn="角码",
        category="structural",
        subcategory="bracket",
        description="L型角码连接件，铝型材/板材 90° 固定",
        tags=["角码", "L型", "corner bracket", "90 degree"],
        parameters=[
            ParamDef("side_length", "边长", default=30.0, min_value=10, max_value=80),
            ParamDef("thickness", "厚度", default=3.0, min_value=1, max_value=8),
            ParamDef("hole_diameter", "孔径", default=4.0, min_value=2, max_value=8),
        ],
        fc_script_template=_CORNER_BRACKET_SCRIPT,
    ),

    "link_arm": PartTemplate(
        id="link_arm",
        name_en="Link Arm",
        name_cn="机械臂连杆",
        category="structural",
        subcategory="bracket",
        description="参数化中空矩形连杆，两端带关节安装孔",
        tags=["连杆", "机械臂", "link", "arm", "beam", "structural"],
        parameters=[
            ParamDef("length", "长度", "mm", 150, 30, 500, 1),
            ParamDef("width", "宽度", "mm", 30, 15, 100, 1),
            ParamDef("height", "高度", "mm", 25, 10, 80, 1),
            ParamDef("wall_thickness", "壁厚", "mm", 3, 1, 10, 0.5),
            ParamDef("joint_hole_diameter", "关节孔径", "mm", 6, 2, 12, 0.5),
            ParamDef("joint_hole_margin", "关节孔边距", "mm", 15, 5, 50, 1),
        ],
        fc_script_template=_LINK_ARM_SCRIPT,
        standard_sizes=[
            {"length": 100, "width": 30, "height": 20, "wall_thickness": 3,
             "joint_hole_diameter": 6, "joint_hole_margin": 12},
            {"length": 200, "width": 40, "height": 30, "wall_thickness": 4,
             "joint_hole_diameter": 8, "joint_hole_margin": 18},
        ],
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="front",
            holes=[
                BoltHole(x=0, y=0, diameter=6.0),
            ],
        ),
    ),

    "joint_housing": PartTemplate(
        id="joint_housing",
        name_en="Joint Housing",
        name_cn="关节壳体",
        category="structural",
        subcategory="bracket",
        description="圆柱形关节壳体，含轴承压入孔和螺栓分布圆",
        tags=["壳体", "关节", "housing", "joint", "bearing seat", "structural"],
        parameters=[
            ParamDef("outer_diameter", "外径", "mm", 50, 25, 100, 1),
            ParamDef("height", "高度", "mm", 30, 15, 80, 1),
            ParamDef("wall_thickness", "壁厚", "mm", 4, 2, 8, 0.5),
            ParamDef("bearing_bore_diameter", "轴承孔径", "mm", 0, 0, 30, 0.5),
            ParamDef("bolt_hole_diameter", "螺栓孔径", "mm", 4, 2, 8, 0.5),
            ParamDef("bolt_count", "螺栓数", "", 6, 0, 8, 1),
            ParamDef("bolt_pcd", "螺栓PCD", "mm", 40, 0, 80, 1),
        ],
        fc_script_template=_JOINT_HOUSING_SCRIPT,
        standard_sizes=[
            {"outer_diameter": 50, "height": 30, "wall_thickness": 4,
             "bearing_bore_diameter": 22, "bolt_hole_diameter": 4,
             "bolt_count": 6, "bolt_pcd": 40},
        ],
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="top",
            holes=[
                BoltHole(x=20, y=0, diameter=4.0),
                BoltHole(x=-20, y=0, diameter=4.0),
            ],
        ),
    ),

    "motor_mount": PartTemplate(
        id="motor_mount",
        name_en="Motor Mount Plate",
        name_cn="电机安装座",
        category="structural",
        subcategory="bracket",
        description="参数化电机安装板，根据电机类型自动匹配孔位",
        tags=["电机", "安装座", "motor mount", "NEMA", "servo", "structural"],
        parameters=[
            ParamDef("motor_type", "电机类型", "", "NEMA17", "NEMA17", "SG90", 1,
                     param_type="string",
                     choices=["NEMA17", "NEMA23", "XM430", "MG996R", "SG90"]),
            ParamDef("plate_length", "板长", "mm", 60, 30, 200, 1),
            ParamDef("plate_width", "板宽", "mm", 60, 30, 200, 1),
            ParamDef("plate_thickness", "板厚", "mm", 5, 2, 10, 0.5),
        ],
        fc_script_template=_MOTOR_MOUNT_SCRIPT,
        standard_sizes=[
            {"motor_type": "NEMA17", "plate_length": 60, "plate_width": 60, "plate_thickness": 5},
            {"motor_type": "XM430", "plate_length": 45, "plate_width": 45, "plate_thickness": 4},
        ],
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="top",
            holes=[
                BoltHole(x=-15.5, y=-15.5, diameter=3.4),
                BoltHole(x=15.5, y=-15.5, diameter=3.4),
                BoltHole(x=-15.5, y=15.5, diameter=3.4),
                BoltHole(x=15.5, y=15.5, diameter=3.4),
            ],
            bore_diameter=23.0,
        ),
    ),

    "sensor_mount": PartTemplate(
        id="sensor_mount",
        name_en="Sensor Mount Bracket",
        name_cn="传感器安装支架",
        category="structural",
        subcategory="bracket",
        description="L型传感器安装支架，带传感器安装孔",
        tags=["传感器", "支架", "sensor mount", "bracket", "structural"],
        parameters=[
            ParamDef("base_length", "底座长度", "mm", 30, 15, 80, 1),
            ParamDef("base_width", "底座宽度", "mm", 25, 10, 60, 1),
            ParamDef("thickness", "厚度", "mm", 3, 2, 8, 0.5),
            ParamDef("bracket_height", "支架高度", "mm", 20, 5, 50, 1),
            ParamDef("hole_diameter", "安装孔径", "mm", 3, 1.5, 5, 0.5),
        ],
        fc_script_template=_SENSOR_MOUNT_SCRIPT,
        standard_sizes=[
            {"base_length": 30, "base_width": 25, "thickness": 3,
             "bracket_height": 20, "hole_diameter": 3},
        ],
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=10, y=12.5, diameter=3.0),
                BoltHole(x=20, y=12.5, diameter=3.0),
            ],
        ),
    ),

    "base_plate": PartTemplate(
        id="base_plate",
        name_en="Base Plate",
        name_cn="底座板",
        category="structural",
        subcategory="plate",
        description="矩形或圆形底座板，带安装孔和可选中心孔",
        tags=["底板", "底座", "base plate", "mounting", "structural"],
        parameters=[
            ParamDef("shape", "形状", "", "rect", "rect", "circle", 1,
                     param_type="string", choices=["rect", "circle"]),
            ParamDef("length_or_diameter", "长度/直径", "mm", 120, 50, 300, 1),
            ParamDef("width", "宽度", "mm", 80, 50, 300, 1),
            ParamDef("thickness", "厚度", "mm", 5, 3, 15, 0.5),
            ParamDef("center_bore", "中心孔径", "mm", 0, 0, 30, 1),
            ParamDef("mounting_hole_diameter", "安装孔径", "mm", 4, 2, 8, 0.5),
            ParamDef("num_holes", "安装孔数", "", 4, 0, 12, 1),
        ],
        fc_script_template=_BASE_PLATE_SCRIPT,
        standard_sizes=[
            {"shape": "rect", "length_or_diameter": 120, "width": 80,
             "thickness": 5, "center_bore": 0, "mounting_hole_diameter": 4, "num_holes": 4},
            {"shape": "circle", "length_or_diameter": 100, "width": 100,
             "thickness": 6, "center_bore": 10, "mounting_hole_diameter": 5, "num_holes": 6},
        ],
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="top",
            holes=[
                BoltHole(x=10, y=10, diameter=4.0),
                BoltHole(x=110, y=10, diameter=4.0),
                BoltHole(x=10, y=70, diameter=4.0),
                BoltHole(x=110, y=70, diameter=4.0),
            ],
        ),
    ),

    "flange_coupling": PartTemplate(
        id="flange_coupling",
        name_en="Flange Coupling",
        name_cn="法兰联轴器",
        category="structural",
        subcategory="bracket",
        description="圆盘法兰联轴器，带中心孔和螺栓分布圆",
        tags=["法兰", "联轴器", "flange", "coupling", "structural"],
        parameters=[
            ParamDef("outer_diameter", "外径", "mm", 50, 20, 120, 1),
            ParamDef("inner_diameter", "内径", "mm", 10, 3, 50, 1),
            ParamDef("thickness", "厚度", "mm", 8, 3, 20, 1),
            ParamDef("bolt_hole_diameter", "螺栓孔径", "mm", 4, 2, 8, 0.5),
            ParamDef("bolt_count", "螺栓数", "", 4, 3, 8, 1),
            ParamDef("bolt_pcd", "螺栓PCD", "mm", 35, 10, 100, 1),
        ],
        fc_script_template=_FLANGE_COUPLING_SCRIPT,
        standard_sizes=[
            {"outer_diameter": 50, "inner_diameter": 10, "thickness": 8,
             "bolt_hole_diameter": 4, "bolt_count": 4, "bolt_pcd": 35},
        ],
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="top",
            holes=[
                BoltHole(x=17.5, y=0, diameter=4.0),
                BoltHole(x=-17.5, y=0, diameter=4.0),
            ],
        ),
    ),

    "shaft_support": PartTemplate(
        id="shaft_support",
        name_en="Shaft Support / Bearing Block",
        name_cn="轴支撑座（轴承座）",
        category="structural",
        subcategory="bracket",
        description="轴支撑座，底板+两侧支撑+轴承压入孔",
        tags=["轴承座", "支撑座", "shaft support", "bearing block", "pillow block", "structural"],
        parameters=[
            ParamDef("shaft_diameter", "轴径", "mm", 8, 3, 30, 1),
            ParamDef("bearing_width", "轴承宽度", "mm", 7, 3, 20, 1),
            ParamDef("base_width", "底板宽度", "mm", 30, 15, 80, 1),
            ParamDef("base_length", "底板长度", "mm", 40, 20, 100, 1),
            ParamDef("base_thickness", "底板厚度", "mm", 5, 3, 15, 0.5),
        ],
        fc_script_template=_SHAFT_SUPPORT_SCRIPT,
        standard_sizes=[
            {"shaft_diameter": 8, "bearing_width": 7, "base_width": 30,
             "base_length": 40, "base_thickness": 5},
        ],
        mounting_interface=MountingInterface(
            interface_type="press_fit",
            contact_face="top",
            bore_diameter=8.5,
            holes=[
                BoltHole(x=7, y=5, diameter=4.0),
                BoltHole(x=33, y=5, diameter=4.0),
                BoltHole(x=7, y=25, diameter=4.0),
                BoltHole(x=33, y=25, diameter=4.0),
            ],
        ),
    ),

    "battery_box": PartTemplate(
        id="battery_box",
        name_en="Battery Box",
        name_cn="电池仓",
        category="structural",
        subcategory="bracket",
        description="参数化电池仓，支持18650/21700/26650，含盖板安装孔",
        tags=["电池", "电池仓", "battery box", "18650", "21700", "structural"],
        parameters=[
            ParamDef("length", "长度", "mm", 75, 30, 200, 1),
            ParamDef("width", "宽度", "mm", 55, 20, 150, 1),
            ParamDef("height", "高度", "mm", 30, 15, 100, 1),
            ParamDef("wall_thickness", "壁厚", "mm", 2, 1.5, 5, 0.5),
            ParamDef("num_cells", "电池数", "", 2, 1, 8, 1),
            ParamDef("cell_type", "电池型号", "", "18650", "18650", "26650", 1,
                     param_type="string", choices=["18650", "21700", "26650"]),
        ],
        fc_script_template=_BATTERY_BOX_SCRIPT,
        standard_sizes=[
            {"length": 75, "width": 55, "height": 30, "wall_thickness": 2,
             "num_cells": 2, "cell_type": "18650"},
            {"length": 155, "width": 55, "height": 30, "wall_thickness": 2,
             "num_cells": 4, "cell_type": "18650"},
        ],
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="top",
            holes=[
                BoltHole(x=4, y=4, diameter=3.0),
                BoltHole(x=71, y=4, diameter=3.0),
                BoltHole(x=4, y=51, diameter=3.0),
                BoltHole(x=71, y=51, diameter=3.0),
            ],
        ),
    ),

    # ---- PCB Mount ----
    "pcb_mount": PartTemplate(
        id="pcb_mount",
        name_en="PCB Mount Pillar",
        name_cn="PCB 安装铜柱",
        category="mounting",
        subcategory="pcb_mount",
        description="PCB 安装支柱，上下 M3 螺纹",
        tags=["PCB", "安装柱", "mount", "pillar"],
        parameters=[
            ParamDef("outer_diameter", "外径", default=6.0, min_value=3, max_value=15),
            ParamDef("height", "高度", default=15.0, min_value=5, max_value=50),
            ParamDef("hole_diameter", "孔径", default=3.0, min_value=1, max_value=8),
        ],
        fc_script_template=_PCB_MOUNT_SCRIPT,
        standard_sizes=[
            {"outer_diameter": 6, "height": 10, "hole_diameter": 3},
            {"outer_diameter": 6, "height": 15, "hole_diameter": 3},
            {"outer_diameter": 6, "height": 25, "hole_diameter": 3},
        ],
    ),

    # ---- Task 68: Real functional parts from product specs ----

    "motor_tt": PartTemplate(
        id="motor_tt",
        name_en="TT Motor (Gearbox Motor)",
        name_cn="TT 减速电机（黄电机）",
        category="actuator",
        subcategory="dc_motor",
        description="TT 减速电机（又称黄电机/130电机），最常用的低成本机器人驱动电机，配塑料减速齿轮箱，常见于 Arduino 机器人套件",
        tags=["电机", "TT", "黄电机", "直流电机", "减速电机", "130", "motor", "DC", "gearbox", "robot"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (Zhengzhou)", model_number="TT-130-1:48",
        parameters=[
            ParamDef("body_length", "机身长度", "mm", 26.5, 10, 100, 0.1, fixed=True),
            ParamDef("body_width", "机身宽度", "mm", 20.5, 5, 50, 0.1, fixed=True),
            ParamDef("body_height", "机身高度", "mm", 15.0, 5, 50, 0.1, fixed=True),
            ParamDef("gearbox_length", "齿轮箱长度", "mm", 10.0, 3, 50, 0.1, fixed=True),
            ParamDef("gearbox_width", "齿轮箱宽度", "mm", 22.0, 5, 50, 0.1, fixed=True),
            ParamDef("gearbox_height", "齿轮箱高度", "mm", 18.0, 5, 50, 0.1, fixed=True),
            ParamDef("shaft_diameter", "输出轴直径", "mm", 3.175, 1, 10, 0.01, fixed=True),
            ParamDef("shaft_length", "输出轴长度", "mm", 7.5, 2, 30, 0.1, fixed=True),
        ],
        fc_script_template=_TT_MOTOR_SCRIPT,
        standard_sizes=[
            # 1:48 ratio (most common)
            {"body_length": 26.5, "body_width": 20.5, "body_height": 15.0,
             "gearbox_length": 10.0, "gearbox_width": 22.0, "gearbox_height": 18.0,
             "shaft_diameter": 3.175, "shaft_length": 7.5},
            # 1:120 ratio (slower, more torque)
            {"body_length": 26.5, "body_width": 20.5, "body_height": 15.0,
             "gearbox_length": 10.0, "gearbox_width": 22.0, "gearbox_height": 18.0,
             "shaft_diameter": 3.175, "shaft_length": 7.5},
        ],
        notes="真实参数来源于产品手册。电压3-6V，空载转速200rpm(1:48)/90rpm(1:120)。D型输出轴。",
    ),

    "servo_ds3218": PartTemplate(
        id="servo_ds3218",
        name_en="DS3218 Digital Servo (20kg)",
        name_cn="DS3218 数字舵机（20kg）",
        category="actuator",
        subcategory="servo",
        description="DS3218 20kg 大扭力数字舵机，金属齿轮，常用于大型机器人关节、云台、机械爪",
        tags=["舵机", "DS3218", "servo", "20kg", "大扭力", "数字", "robot", "pan tilt"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="JX", model_number="DS3218",
        parameters=[
            ParamDef("body_length", "机身长度", "mm", 40.0, 20, 100, 0.1, fixed=True),
            ParamDef("body_width", "机身宽度", "mm", 20.0, 10, 60, 0.1, fixed=True),
            ParamDef("body_height", "机身高度", "mm", 38.5, 20, 100, 0.1, fixed=True),
            ParamDef("shaft_diameter", "输出轴直径", "mm", 5.8, 2, 20, 0.1, fixed=True),
            ParamDef("shaft_length", "输出轴长度", "mm", 6.0, 2, 30, 0.1, fixed=True),
        ],
        fc_script_template=_DS3218_SERVO_SCRIPT,
        standard_sizes=[
            {"body_length": 40.0, "body_width": 20.0, "body_height": 38.5,
             "shaft_diameter": 5.8, "shaft_length": 6.0},
        ],
        notes="真实参数来源于产品手册。电压6.8-7.4V，扭矩20kg·cm@6.8V，速度0.17s/60°@6.8V。",
    ),

    "motor_jgb37_520": PartTemplate(
        id="motor_jgb37_520",
        name_en="JGB37-520 DC Gearmotor",
        name_cn="JGB37-520 直流减速电机",
        category="actuator",
        subcategory="dc_motor",
        description="JGB37-520 直流减速电机，12V 金属齿轮箱，常用于中型 AGV、巡检机器人、服务机器人",
        tags=["电机", "JGB37-520", "直流减速电机", "gearmotor", "12V", "AGV", "robot"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="JGB37-520",
        parameters=[
            ParamDef("body_diameter", "机身直径", "mm", 37.0, 20, 80, 0.1, fixed=True),
            ParamDef("body_length", "机身长度", "mm", 50.0, 20, 100, 0.1, fixed=True),
            ParamDef("gearbox_diameter", "齿轮箱直径", "mm", 37.0, 20, 80, 0.1, fixed=True),
            ParamDef("gearbox_length", "齿轮箱长度", "mm", 18.0, 5, 50, 0.1, fixed=True),
            ParamDef("shaft_diameter", "输出轴直径", "mm", 6.0, 2, 20, 0.1, fixed=True),
            ParamDef("shaft_length", "输出轴长度", "mm", 15.5, 5, 30, 0.1, fixed=True),
        ],
        fc_script_template=_JGB37_520_SCRIPT,
        standard_sizes=[
            # 1:30 ratio @ 12V, 200rpm
            {"body_diameter": 37.0, "body_length": 50.0,
             "gearbox_diameter": 37.0, "gearbox_length": 18.0,
             "shaft_diameter": 6.0, "shaft_length": 15.5},
            # 1:50 ratio @ 12V, 130rpm
            {"body_diameter": 37.0, "body_length": 50.0,
             "gearbox_diameter": 37.0, "gearbox_length": 18.0,
             "shaft_diameter": 6.0, "shaft_length": 15.5},
            # 1:131 ratio @ 12V, 50rpm
            {"body_diameter": 37.0, "body_length": 50.0,
             "gearbox_diameter": 37.0, "gearbox_length": 18.0,
             "shaft_diameter": 6.0, "shaft_length": 15.5},
        ],
        notes="真实参数来源于产品手册。D型输出轴。齿轮箱端面2×M3安装螺孔。",
    ),

    "nema23_stepper": PartTemplate(
        id="nema23_stepper",
        name_en="NEMA23 Stepper Motor",
        name_cn="NEMA23 步进电机",
        category="actuator",
        subcategory="stepper",
        description="NEMA23 (57mm) 步进电机，大扭力，常用于CNC、大型3D打印机、工业机械臂",
        tags=["步进电机", "NEMA23", "stepper", "motor", "57mm", "CNC", "3D printer"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="NEMA23-57BYGH",
        parameters=[
            ParamDef("body_size", "机身尺寸", "mm", 56.4, 30, 100, 0.1, fixed=True),
            ParamDef("body_length", "机身长度", "mm", 56.0, 30, 150, 1, fixed=True),
            ParamDef("shaft_diameter", "输出轴直径", "mm", 6.35, 2, 20, 0.01, fixed=True),
            ParamDef("shaft_length", "输出轴长度", "mm", 21.0, 5, 50, 0.1, fixed=True),
        ],
        fc_script_template=_NEMA23_SCRIPT,
        standard_sizes=[
            # Standard NEMA23
            {"body_size": 56.4, "body_length": 56.0,
             "shaft_diameter": 6.35, "shaft_length": 21.0},
            # Short body variant
            {"body_size": 56.4, "body_length": 40.0,
             "shaft_diameter": 6.35, "shaft_length": 21.0},
            # Long body (high torque)
            {"body_size": 56.4, "body_length": 76.0,
             "shaft_diameter": 6.35, "shaft_length": 21.0},
        ],
        notes="真实参数来源于NEMA23标准。安装孔距47.14mm×47.14mm，4×M5安装孔。",
    ),

    "sensor_rplidar_a1": PartTemplate(
        id="sensor_rplidar_a1",
        name_en="RPLIDAR A1 2D LiDAR",
        name_cn="RPLIDAR A1 2D 激光雷达",
        category="sensor",
        subcategory="lidar",
        description="RPLIDAR A1 2D 激光雷达，360° 扫描，测距范围 12m，常用于移动机器人导航和建图",
        tags=["传感器", "激光雷达", "LiDAR", "RPLIDAR", "A1", "SLAM", "navigation", "robot"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Slamtec", model_number="RPLIDAR-A1",
        parameters=[
            ParamDef("body_diameter", "机身直径", "mm", 72.0, 30, 200, 0.1, fixed=True),
            ParamDef("body_height", "机身高度", "mm", 41.0, 10, 100, 0.1, fixed=True),
        ],
        fc_script_template=_RPLIDAR_A1_SCRIPT,
        standard_sizes=[
            {"body_diameter": 72.0, "body_height": 41.0},
        ],
        notes="真实参数来源于Slamtec官方规格。扫描频率5.5Hz，角度分辨率1°，测距范围0.15-12m。",
    ),

    "sensor_mpu6050": PartTemplate(
        id="sensor_mpu6050",
        name_en="MPU6050 IMU Module",
        name_cn="MPU6050 惯性测量模块",
        category="sensor",
        subcategory="imu",
        description="MPU6050 六轴惯性测量单元（3轴加速度+3轴陀螺仪），常用于机器人姿态估计和平衡控制",
        tags=["传感器", "IMU", "MPU6050", "加速度计", "陀螺仪", "姿态", "robot", "balancing"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="InvenSense (TDK)", model_number="MPU-6050",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 21.0, 10, 50, 0.1, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 16.0, 8, 40, 0.1, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
        ],
        fc_script_template=_MPU6050_SCRIPT,
        standard_sizes=[
            {"pcb_length": 21.0, "pcb_width": 16.0, "pcb_thickness": 1.6},
        ],
        notes="真实参数来源于模块尺寸。芯片本身4x4mm QFN封装。加速度范围±2/4/8/16g，陀螺仪范围±250/500/1000/2000°/s。",
    ),

    "sensor_esp32_cam": PartTemplate(
        id="sensor_esp32_cam",
        name_en="ESP32-CAM Module",
        name_cn="ESP32-CAM 摄像头模块",
        category="sensor",
        subcategory="camera",
        description="ESP32-CAM Wi-Fi 摄像头模块，集成 ESP32 + OV2640 摄像头，常用于机器人视觉、远程监控",
        tags=["传感器", "摄像头", "ESP32", "ESP32-CAM", "Wi-Fi", "camera", "vision", "robot"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Ai-Thinker", model_number="ESP32-CAM",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 40.0, 20, 80, 0.1, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 27.0, 10, 60, 0.1, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
            ParamDef("camera_diameter", "摄像头直径", "mm", 8.0, 3, 20, 0.1, fixed=True),
            ParamDef("camera_height", "摄像头高度", "mm", 5.0, 2, 15, 0.1, fixed=True),
        ],
        fc_script_template=_ESP32_CAM_SCRIPT,
        standard_sizes=[
            {"pcb_length": 40.0, "pcb_width": 27.0, "pcb_thickness": 1.6,
             "camera_diameter": 8.0, "camera_height": 5.0},
        ],
        notes="真实参数来源于Ai-Thinker官方尺寸。OV2640摄像头，支持JPEG/QT streaming。注意：模块无USB口，需FTTL下载器。",
    ),

    # ---- Task 73: Transmission parts ----

    "gt2_pulley": PartTemplate(
        id="gt2_pulley",
        name_en="GT2 Timing Pulley",
        name_cn="GT2 同步轮",
        category="transmission",
        subcategory="timing_pulley",
        description="GT2 同步轮，节距2mm，常用于3D打印机、小型CNC、机器人关节驱动链",
        tags=["同步轮", "GT2", "timing pulley", "同步带轮", "3D printer", "robot", "transmission"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="GT2",
        parameters=[
            ParamDef("teeth", "齿数", "", 20, 10, 80, 1, fixed=True),
            ParamDef("width", "宽度", "mm", 6.0, 3, 15, 0.5, fixed=True),
            ParamDef("bore_diameter", "轴孔直径", "mm", 5.0, 3, 12, 0.5, fixed=True),
            ParamDef("hub_diameter", "轮毂直径", "mm", 10.0, 5, 25, 0.5, fixed=True),
            ParamDef("hub_height", "轮毂高度", "mm", 5.0, 0, 15, 0.5, fixed=True),
            ParamDef("pulley_detail", "齿廓细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
        ],
        fc_script_template=_GT2_PULLEY_SCRIPT,
        fc_script_alternatives={"realistic": _GT2_PULLEY_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            # 16T for NEMA17 (5mm shaft)
            {"teeth": 16, "width": 6.0, "bore_diameter": 5.0,
             "hub_diameter": 10.0, "hub_height": 5.0},
            # 20T for NEMA17
            {"teeth": 20, "width": 6.0, "bore_diameter": 5.0,
             "hub_diameter": 12.0, "hub_height": 5.0},
            # 20T 9mm wide
            {"teeth": 20, "width": 9.0, "bore_diameter": 5.0,
             "hub_diameter": 12.0, "hub_height": 5.0},
            # 36T for NEMA23 (6.35mm shaft)
            {"teeth": 36, "width": 6.0, "bore_diameter": 6.35,
             "hub_diameter": 15.0, "hub_height": 7.0},
            # 36T 9mm wide
            {"teeth": 36, "width": 9.0, "bore_diameter": 6.35,
             "hub_diameter": 15.0, "hub_height": 7.0},
        ],
        notes="GT2节距2mm。铝合金材质最常用（标注中未区分，建模为统一外观）。"
              "轮毂侧有凸台（hub），带平键或紧定螺钉固定。",
    ),

    "gt2_belt": PartTemplate(
        id="gt2_belt",
        name_en="GT2 Timing Belt",
        name_cn="GT2 同步带",
        category="transmission",
        subcategory="timing_belt",
        description="GT2 同步带，节距2mm，玻璃纤维芯/钢芯，用于3D打印机和小型传动",
        tags=["同步带", "GT2", "timing belt", "传动带", "3D printer", "robot", "transmission"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (Gates/Continental clone)", model_number="GT2",
        parameters=[
            ParamDef("teeth", "齿数", "", 100, 20, 500, 1, fixed=True),
            ParamDef("width", "宽度", "mm", 6.0, 3, 15, 0.5, fixed=True),
        ],
        fc_script_template=_GT2_BELT_SCRIPT,
        standard_sizes=[
            # Common 3D printer belts
            {"teeth": 100, "width": 6.0},   # 200mm loop
            {"teeth": 150, "width": 6.0},   # 300mm loop
            {"teeth": 200, "width": 6.0},   # 400mm loop
            {"teeth": 100, "width": 9.0},   # 200mm loop (wide)
            {"teeth": 150, "width": 9.0},   # 300mm loop (wide)
            {"teeth": 200, "width": 9.0},   # 400mm loop (wide)
        ],
        notes="GT2同步带节距2mm。长度 = 齿数 × 2mm。建模为环形近似。"
              "常见宽度6mm和9mm。玻璃纤维芯抗拉伸。",
    ),

    "htd_pulley_3m": PartTemplate(
        id="htd_pulley_3m",
        name_en="HTD 3M Timing Pulley",
        name_cn="HTD 3M 同步轮",
        category="transmission",
        subcategory="timing_pulley",
        description="HTD 3M 同步轮，节距3mm，半圆齿形，适用于中小功率传动",
        tags=["同步轮", "HTD", "3M", "timing pulley", "传动", "robot", "CNC", "transmission"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="HTD-3M",
        parameters=[
            ParamDef("teeth", "齿数", "", 15, 10, 72, 1, fixed=True),
            ParamDef("width", "宽度", "mm", 9.0, 5, 20, 0.5, fixed=True),
            ParamDef("bore_diameter", "轴孔直径", "mm", 5.0, 3, 15, 0.5, fixed=True),
            ParamDef("hub_diameter", "轮毂直径", "mm", 12.0, 5, 30, 0.5, fixed=True),
            ParamDef("hub_height", "轮毂高度", "mm", 5.0, 0, 15, 0.5, fixed=True),
            ParamDef("pitch", "节距", "mm", 3.0, 3, 3, 0.0, fixed=True),
            ParamDef("module", "模数", "mm", 0.97, 0.5, 2.0, 0.01, fixed=True),
            ParamDef("pulley_detail", "齿廓细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
        ],
        fc_script_template=_HTD_PULLEY_SCRIPT,
        fc_script_alternatives={"realistic": _HTD_PULLEY_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"teeth": 15, "width": 9.0, "bore_diameter": 5.0,
             "hub_diameter": 12.0, "hub_height": 5.0},
            {"teeth": 20, "width": 9.0, "bore_diameter": 5.0,
             "hub_diameter": 14.0, "hub_height": 5.0},
            {"teeth": 30, "width": 9.0, "bore_diameter": 8.0,
             "hub_diameter": 18.0, "hub_height": 7.0},
        ],
        notes="HTD 3M节距3mm，半圆齿形，比GT2传递力矩更大。铝合金材质。",
    ),

    "htd_pulley_5m": PartTemplate(
        id="htd_pulley_5m",
        name_en="HTD 5M Timing Pulley",
        name_cn="HTD 5M 同步轮",
        category="transmission",
        subcategory="timing_pulley",
        description="HTD 5M 同步轮，节距5mm，半圆齿形，适用于大功率传动",
        tags=["同步轮", "HTD", "5M", "timing pulley", "传动", "robot", "CNC", "大功率", "transmission"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="HTD-5M",
        parameters=[
            ParamDef("teeth", "齿数", "", 15, 10, 72, 1, fixed=True),
            ParamDef("width", "宽度", "mm", 15.0, 9, 30, 0.5, fixed=True),
            ParamDef("bore_diameter", "轴孔直径", "mm", 8.0, 5, 20, 0.5, fixed=True),
            ParamDef("hub_diameter", "轮毂直径", "mm", 18.0, 10, 35, 0.5, fixed=True),
            ParamDef("hub_height", "轮毂高度", "mm", 7.0, 0, 20, 0.5, fixed=True),
            ParamDef("pitch", "节距", "mm", 5.0, 5, 5, 0.0, fixed=True),
            ParamDef("module", "模数", "mm", 1.60, 1.0, 3.0, 0.01, fixed=True),
            ParamDef("pulley_detail", "齿廓细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
        ],
        fc_script_template=_HTD_PULLEY_SCRIPT,
        fc_script_alternatives={"realistic": _HTD_PULLEY_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"teeth": 15, "width": 15.0, "bore_diameter": 8.0,
             "hub_diameter": 18.0, "hub_height": 7.0},
            {"teeth": 20, "width": 15.0, "bore_diameter": 8.0,
             "hub_diameter": 22.0, "hub_height": 8.0},
            {"teeth": 30, "width": 15.0, "bore_diameter": 10.0,
             "hub_diameter": 28.0, "hub_height": 10.0},
        ],
        notes="HTD 5M节距5mm，比3M更大承载能力。常用于CNC主轴、大型3D打印机、机器人关节。",
    ),

    "rigid_coupling_setscrew": PartTemplate(
        id="rigid_coupling_setscrew",
        name_en="Rigid Coupling (Set Screw)",
        name_cn="刚性联轴器（紧定螺钉型）",
        category="transmission",
        subcategory="rigid_coupling",
        description="刚性联轴器，紧定螺钉固定，适用于两轴刚性对中连接",
        tags=["联轴器", "刚性", "紧定螺钉", "rigid coupling", "set screw", "shaft", "transmission"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="RCS",
        parameters=[
            ParamDef("outer_diameter", "外径", "mm", 16.0, 8, 40, 0.5, fixed=True),
            ParamDef("length", "长度", "mm", 25.0, 10, 60, 1, fixed=True),
            ParamDef("bore_diameter", "轴孔直径", "mm", 5.0, 2, 20, 0.5, fixed=True),
            ParamDef("num_setscrews", "紧定螺钉数量", "", 2, 1, 4, 1, fixed=True),
            ParamDef("setscrew_size", "紧定螺钉尺寸", "mm", 3.0, 1.5, 6, 0.5, fixed=True),
        ],
        fc_script_template=_RIGID_COUPLING_SETSCREW_SCRIPT,
        standard_sizes=[
            # 5mm shaft
            {"outer_diameter": 16.0, "length": 25.0, "bore_diameter": 5.0,
             "num_setscrews": 2, "setscrew_size": 3.0},
            # 6mm shaft
            {"outer_diameter": 19.0, "length": 30.0, "bore_diameter": 6.0,
             "num_setscrews": 2, "setscrew_size": 3.0},
            # 8mm shaft
            {"outer_diameter": 22.0, "length": 35.0, "bore_diameter": 8.0,
             "num_setscrews": 2, "setscrew_size": 4.0},
        ],
        notes="紧定螺钉压入轴面固定，要求轴面有平面或D-cut。对中性要求高。",
    ),

    "rigid_coupling_clamping": PartTemplate(
        id="rigid_coupling_clamping",
        name_en="Rigid Coupling (Clamping)",
        name_cn="刚性联轴器（夹紧型）",
        category="transmission",
        subcategory="rigid_coupling",
        description="刚性联轴器，夹紧式固定，开缝设计，通过螺栓径向夹紧轴",
        tags=["联轴器", "刚性", "夹紧", "clamping coupling", "split clamp", "shaft", "transmission"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="RCC",
        parameters=[
            ParamDef("outer_diameter", "外径", "mm", 19.0, 10, 40, 0.5, fixed=True),
            ParamDef("length", "长度", "mm", 25.0, 10, 60, 1, fixed=True),
            ParamDef("bore_diameter", "轴孔直径", "mm", 5.0, 2, 20, 0.5, fixed=True),
            ParamDef("clamp_screw_size", "夹紧螺栓尺寸", "mm", 3.0, 2, 6, 0.5, fixed=True),
        ],
        fc_script_template=_RIGID_COUPLING_CLAMPING_SCRIPT,
        standard_sizes=[
            # 5mm shaft
            {"outer_diameter": 19.0, "length": 25.0, "bore_diameter": 5.0,
             "clamp_screw_size": 3.0},
            # 8mm shaft
            {"outer_diameter": 25.0, "length": 30.0, "bore_diameter": 8.0,
             "clamp_screw_size": 4.0},
            # 10mm shaft
            {"outer_diameter": 30.0, "length": 35.0, "bore_diameter": 10.0,
             "clamp_screw_size": 5.0},
        ],
        notes="开缝设计，通过M3/M4螺栓径向夹紧。无需D-cut轴面。拆装方便。",
    ),

    "spider_coupling": PartTemplate(
        id="spider_coupling",
        name_en="Spider (Jaw) Flexible Coupling",
        name_cn="梅花弹性联轴器",
        category="transmission",
        subcategory="flexible_coupling",
        description="梅花弹性联轴器，金属两端+弹性体中间，补偿轴向/径向/角向偏差",
        tags=["联轴器", "柔性", "梅花", "spider", "jaw coupling", "flexible", "elastomer", "transmission"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (Lovejoy type)", model_number="L-type",
        parameters=[
            ParamDef("outer_diameter", "外径", "mm", 19.0, 10, 50, 0.5, fixed=True),
            ParamDef("length", "长度", "mm", 25.0, 15, 60, 1, fixed=True),
            ParamDef("bore1_diameter", "孔1直径", "mm", 5.0, 2, 20, 0.5, fixed=True),
            ParamDef("bore2_diameter", "孔2直径", "mm", 5.0, 2, 20, 0.5, fixed=True),
            ParamDef("jaw_count", "爪数", "", 3, 2, 6, 1, fixed=True),
            ParamDef("jaw_depth", "爪深", "mm", 3.0, 1, 8, 0.5, fixed=True),
        ],
        fc_script_template=_SPIDER_COUPLING_SCRIPT,
        standard_sizes=[
            # L035 (5mm x 5mm)
            {"outer_diameter": 19.0, "length": 25.0,
             "bore1_diameter": 5.0, "bore2_diameter": 5.0,
             "jaw_count": 3, "jaw_depth": 3.0},
            # L050 (8mm x 8mm)
            {"outer_diameter": 25.0, "length": 30.0,
             "bore1_diameter": 8.0, "bore2_diameter": 8.0,
             "jaw_count": 3, "jaw_depth": 4.0},
            # L070 (10mm x 10mm)
            {"outer_diameter": 30.0, "length": 35.0,
             "bore1_diameter": 10.0, "bore2_diameter": 10.0,
             "jaw_count": 4, "jaw_depth": 5.0},
        ],
        notes="弹性体材质：聚氨酯(85A/95A/98A)。可补偿径向偏差0.1-0.3mm、角向偏差1-2°。",
    ),

    "bellows_coupling": PartTemplate(
        id="bellows_coupling",
        name_en="Bellows Flexible Coupling",
        name_cn="波纹管联轴器",
        category="transmission",
        subcategory="flexible_coupling",
        description="波纹管联轴器，不锈钢波纹管+两端铝合金夹紧头，高刚性高精度",
        tags=["联轴器", "柔性", "波纹管", "bellows", "flexible", "高精度", "servo", "transmission"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (Servo City/Misumi type)", model_number="BC",
        parameters=[
            ParamDef("outer_diameter", "外径", "mm", 19.0, 10, 40, 0.5, fixed=True),
            ParamDef("length", "长度", "mm", 30.0, 15, 60, 1, fixed=True),
            ParamDef("bore1_diameter", "孔1直径", "mm", 5.0, 2, 16, 0.5, fixed=True),
            ParamDef("bore2_diameter", "孔2直径", "mm", 8.0, 2, 16, 0.5, fixed=True),
            ParamDef("convolutions", "波纹数", "", 6, 3, 12, 1, fixed=True),
            ParamDef("wall_thickness", "壁厚", "mm", 0.3, 0.1, 1.0, 0.05, fixed=True),
        ],
        fc_script_template=_BELLOWS_COUPLING_SCRIPT,
        standard_sizes=[
            # 5mm to 8mm
            {"outer_diameter": 19.0, "length": 30.0,
             "bore1_diameter": 5.0, "bore2_diameter": 8.0,
             "convolutions": 6, "wall_thickness": 0.3},
            # 6.35mm to 8mm
            {"outer_diameter": 19.0, "length": 33.0,
             "bore1_diameter": 6.35, "bore2_diameter": 8.0,
             "convolutions": 7, "wall_thickness": 0.3},
            # 8mm to 8mm
            {"outer_diameter": 25.0, "length": 36.0,
             "bore1_diameter": 8.0, "bore2_diameter": 8.0,
             "convolutions": 6, "wall_thickness": 0.4},
        ],
        notes="不锈钢波纹管提供高扭转刚性和零背隙。适用于伺服电机、编码器、精密传动。",
    ),

    # ---- Task 74: Linear motion & advanced actuator parts ----

    "linear_bearing_lm8uu": PartTemplate(
        id="linear_bearing_lm8uu",
        name_en="LM8UU Linear Ball Bearing",
        name_cn="LM8UU 直线球轴承",
        category="bearing",
        subcategory="linear_bearing",
        description="LM8UU 直线运动球轴承，用于 8mm 光轴上的直线往复运动，3D打印机/CNC最常用",
        tags=["直线轴承", "LM8UU", "linear bearing", "ball bearing", "直线运动", "3D printer"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (MISUMI/THK clone)", model_number="LM8UU",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 8.0, 4, 60, 0.1, fixed=True),
            ParamDef("outer_diameter", "外径", "mm", 15.0, 8, 62, 0.1, fixed=True),
            ParamDef("length", "长度", "mm", 24.0, 10, 80, 0.5, fixed=True),
            ParamDef("ball_count", "滚珠数", "", 0, 0, 50, 1, fixed=True),
            ParamDef("bearing_detail", "细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
        ],
        fc_script_template=_LINEAR_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _LINEAR_BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 8.0, "outer_diameter": 15.0, "length": 24.0},
        ],
        notes="LM8UU是最常见的直线轴承。4~5列钢珠回路。外圈有微小间隙适应壳体。",
    ),

    "linear_bearing_lm10uu": PartTemplate(
        id="linear_bearing_lm10uu",
        name_en="LM10UU Linear Ball Bearing",
        name_cn="LM10UU 直线球轴承",
        category="bearing",
        subcategory="linear_bearing",
        description="LM10UU 直线运动球轴承，用于 10mm 光轴",
        tags=["直线轴承", "LM10UU", "linear bearing", "ball bearing", "直线运动"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="LM10UU",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 10.0, 4, 60, 0.1, fixed=True),
            ParamDef("outer_diameter", "外径", "mm", 19.0, 8, 62, 0.1, fixed=True),
            ParamDef("length", "长度", "mm", 29.0, 10, 80, 0.5, fixed=True),
            ParamDef("ball_count", "滚珠数", "", 0, 0, 50, 1, fixed=True),
            ParamDef("bearing_detail", "细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
        ],
        fc_script_template=_LINEAR_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _LINEAR_BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 10.0, "outer_diameter": 19.0, "length": 29.0},
        ],
        notes="LM10UU用于10mm光轴。比LM8UU承载力更大。",
    ),

    "linear_bearing_lm12uu": PartTemplate(
        id="linear_bearing_lm12uu",
        name_en="LM12UU Linear Ball Bearing",
        name_cn="LM12UU 直线球轴承",
        category="bearing",
        subcategory="linear_bearing",
        description="LM12UU 直线运动球轴承，用于 12mm 光轴",
        tags=["直线轴承", "LM12UU", "linear bearing", "ball bearing", "直线运动"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="LM12UU",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 12.0, 4, 60, 0.1, fixed=True),
            ParamDef("outer_diameter", "外径", "mm", 21.0, 8, 62, 0.1, fixed=True),
            ParamDef("length", "长度", "mm", 30.0, 10, 80, 0.5, fixed=True),
            ParamDef("ball_count", "滚珠数", "", 0, 0, 50, 1, fixed=True),
            ParamDef("bearing_detail", "细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
        ],
        fc_script_template=_LINEAR_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _LINEAR_BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 12.0, "outer_diameter": 21.0, "length": 30.0},
        ],
        notes="LM12UU用于12mm光轴。中载荷直线运动。",
    ),

    "linear_guide_mgn12h": PartTemplate(
        id="linear_guide_mgn12h",
        name_en="MGN12H Linear Guide (Rail + Carriage)",
        name_cn="MGN12H 直线导轨（轨道+滑块）",
        category="bearing",
        subcategory="linear_bearing",
        description="MGN12H 微型直线导轨，12mm轨宽，高载荷滑块，CNC/3D打印机/机械臂常用",
        tags=["直线导轨", "MGN12", "linear guide", "rail", "carriage", "CNC", "3D printer"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (HIWIN/THK clone)", model_number="MGN12H",
        parameters=[
            ParamDef("rail_length", "轨道长度", "mm", 200.0, 50, 2000, 1, fixed=True),
            ParamDef("rail_width", "轨道宽度", "mm", 12.0, 5, 30, 0.5, fixed=True),
            ParamDef("rail_height", "轨道高度", "mm", 8.0, 3, 20, 0.5, fixed=True),
            ParamDef("carriage_length", "滑块长度", "mm", 40.3, 15, 80, 0.1, fixed=True),
            ParamDef("carriage_width", "滑块宽度", "mm", 27.0, 10, 50, 0.1, fixed=True),
            ParamDef("carriage_height", "滑块高度", "mm", 10.0, 5, 25, 0.1, fixed=True),
            ParamDef("mounting_hole_diameter", "安装孔直径", "mm", 3.5, 2, 8, 0.5, fixed=True),
            ParamDef("mounting_hole_pitch", "安装孔距", "mm", 25.0, 10, 60, 0.5, fixed=True),
        ],
        fc_script_template=_LINEAR_GUIDE_RAIL_SCRIPT,
        quality_levels=["simplified"],
        standard_sizes=[
            # MGN12H — 200mm rail
            {"rail_length": 200, "rail_width": 12.0, "rail_height": 8.0,
             "carriage_length": 40.3, "carriage_width": 27.0, "carriage_height": 10.0,
             "mounting_hole_diameter": 3.5, "mounting_hole_pitch": 25.0},
            # MGN12H — 300mm rail
            {"rail_length": 300, "rail_width": 12.0, "rail_height": 8.0,
             "carriage_length": 40.3, "carriage_width": 27.0, "carriage_height": 10.0,
             "mounting_hole_diameter": 3.5, "mounting_hole_pitch": 25.0},
            # MGN12H — 500mm rail
            {"rail_length": 500, "rail_width": 12.0, "rail_height": 8.0,
             "carriage_length": 40.3, "carriage_width": 27.0, "carriage_height": 10.0,
             "mounting_hole_diameter": 3.5, "mounting_hole_pitch": 25.0},
        ],
        notes="MGN12H：H型高载荷滑块（4列钢珠）。额定动载荷1.67kN，静载荷2.56kN。"
              "轨道安装孔距25mm。滑块安装孔M3×4。",
    ),

    "t8_leadscrew": PartTemplate(
        id="t8_leadscrew",
        name_en="T8 Leadscrew (Tr8×8)",
        name_cn="T8 丝杠 (Tr8×8)",
        category="shaft",
        subcategory="leadscrew",
        description="T8 梯形螺纹丝杠，导程 2/4/8mm，3D打印机/CNC Z轴最常用",
        tags=["丝杠", "T8", "leadscrew", "梯形螺纹", "3D printer", "CNC"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="T8",
        parameters=[
            ParamDef("diameter", "直径", "mm", 8.0, 4, 20, 0.5, fixed=True),
            ParamDef("length", "长度", "mm", 300.0, 50, 2000, 1, fixed=True),
            ParamDef("lead", "导程", "mm", 8.0, 2, 20, 0.5, fixed=True),
            ParamDef("leadscrew_detail", "细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
        ],
        fc_script_template=_T8_LEADSCREW_SCRIPT,
        fc_script_alternatives={"realistic": _T8_LEADSCREW_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            # Most common: Tr8×8 (lead 8mm, single start, pitch 8mm)
            {"diameter": 8.0, "length": 300, "lead": 8.0},
            {"diameter": 8.0, "length": 400, "lead": 8.0},
            {"diameter": 8.0, "length": 500, "lead": 8.0},
            # High resolution: Tr8×2 (lead 2mm, 4 starts, pitch 2mm)
            {"diameter": 8.0, "length": 300, "lead": 2.0},
            # Medium: Tr8×4
            {"diameter": 8.0, "length": 300, "lead": 4.0},
        ],
        notes="T8丝杠是最常用的3D打印机Z轴丝杠。Tr8×8=导程8mm(单头)、Tr8×4=导程4mm(双头)、"
              "Tr8×2=导程2mm(四头)。材料一般为SUS304不锈钢或S45C碳钢。",
    ),

    "t8_nut": PartTemplate(
        id="t8_nut",
        name_en="T8 Leadscrew Nut (Flange)",
        name_cn="T8 丝杠螺母（法兰型）",
        category="shaft",
        subcategory="leadscrew",
        description="T8 丝杠配套法兰螺母，黄铜/聚甲醛材质，4×M3法兰安装孔",
        tags=["丝杠螺母", "T8", "leadscrew nut", "flange", "brass"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="T8-NUT-FLANGE",
        parameters=[
            ParamDef("outer_diameter", "外径", "mm", 14.0, 8, 30, 0.5, fixed=True),
            ParamDef("bore_diameter", "螺纹孔径", "mm", 8.0, 4, 20, 0.5, fixed=True),
            ParamDef("length", "长度", "mm", 15.0, 8, 30, 0.5, fixed=True),
            ParamDef("flange_diameter", "法兰直径", "mm", 22.0, 10, 40, 0.5, fixed=True),
            ParamDef("flange_thickness", "法兰厚度", "mm", 3.0, 1, 6, 0.5, fixed=True),
            ParamDef("flange_hole_diameter", "法兰安装孔径", "mm", 3.4, 2, 6, 0.5, fixed=True),
        ],
        fc_script_template=_T8_NUT_SCRIPT,
        standard_sizes=[
            {"outer_diameter": 14.0, "bore_diameter": 8.0, "length": 15.0,
             "flange_diameter": 22.0, "flange_thickness": 3.0, "flange_hole_diameter": 3.4},
        ],
        notes="黄铜材质最常见（耐磨）。法兰4×M3安装孔，孔距16mm×16mm方阵。"
              "POM材质版本更静音但寿命较短。",
    ),

    "bldc_motor_5010": PartTemplate(
        id="bldc_motor_5010",
        name_en="5010 BLDC Motor (Outrunner)",
        name_cn="5010 无刷电机（外转子）",
        category="actuator",
        subcategory="bldc",
        description="5010 外转子无刷电机，常用于无人机推进、云台、小型机器人关节",
        tags=["无刷电机", "BLDC", "5010", "外转子", "outrunner", "drone", "gimbal"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (SunnySky/T-Motor clone)", model_number="5010",
        parameters=[
            ParamDef("stator_outer_diameter", "定子外径", "mm", 27.0, 10, 60, 0.1, fixed=True),
            ParamDef("stator_inner_diameter", "定子内径", "mm", 12.0, 4, 30, 0.1, fixed=True),
            ParamDef("stator_length", "定子长度", "mm", 10.0, 5, 30, 0.1, fixed=True),
            ParamDef("rotor_outer_diameter", "转子外径", "mm", 50.0, 20, 100, 0.1, fixed=True),
            ParamDef("rotor_inner_diameter", "转子内径", "mm", 28.0, 10, 60, 0.1, fixed=True),
            ParamDef("rotor_length", "转子长度", "mm", 12.0, 5, 30, 0.1, fixed=True),
            ParamDef("shaft_diameter", "轴径", "mm", 5.0, 2, 10, 0.1, fixed=True),
            ParamDef("shaft_length", "轴长", "mm", 25.0, 5, 50, 0.1, fixed=True),
        ],
        fc_script_template=_BLDC_MOTOR_SCRIPT,
        standard_sizes=[
            {"stator_outer_diameter": 27.0, "stator_inner_diameter": 12.0, "stator_length": 10.0,
             "rotor_outer_diameter": 50.0, "rotor_inner_diameter": 28.0, "rotor_length": 12.0,
             "shaft_diameter": 5.0, "shaft_length": 25.0},
        ],
        notes="5010外转子无刷电机。KV值约280-360。定子12槽14极。"
              "用于无人机、云台稳定器。配ESC电调使用。",
    ),

    "bldc_motor_2208": PartTemplate(
        id="bldc_motor_2208",
        name_en="2208 BLDC Motor (Inrunner/Outrunner)",
        name_cn="2208 无刷电机",
        category="actuator",
        subcategory="bldc",
        description="2208 小型无刷电机，常用于小型无人机、舵机替换、小型机器人",
        tags=["无刷电机", "BLDC", "2208", "小型", "drone", "micro robot"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="2208",
        parameters=[
            ParamDef("stator_outer_diameter", "定子外径", "mm", 13.0, 5, 30, 0.1, fixed=True),
            ParamDef("stator_inner_diameter", "定子内径", "mm", 6.0, 2, 15, 0.1, fixed=True),
            ParamDef("stator_length", "定子长度", "mm", 8.0, 3, 20, 0.1, fixed=True),
            ParamDef("rotor_outer_diameter", "转子外径", "mm", 22.0, 10, 40, 0.1, fixed=True),
            ParamDef("rotor_inner_diameter", "转子内径", "mm", 14.0, 5, 20, 0.1, fixed=True),
            ParamDef("rotor_length", "转子长度", "mm", 10.0, 3, 20, 0.1, fixed=True),
            ParamDef("shaft_diameter", "轴径", "mm", 3.0, 1, 6, 0.1, fixed=True),
            ParamDef("shaft_length", "轴长", "mm", 15.0, 3, 30, 0.1, fixed=True),
        ],
        fc_script_template=_BLDC_MOTOR_SCRIPT,
        standard_sizes=[
            {"stator_outer_diameter": 13.0, "stator_inner_diameter": 6.0, "stator_length": 8.0,
             "rotor_outer_diameter": 22.0, "rotor_inner_diameter": 14.0, "rotor_length": 10.0,
             "shaft_diameter": 3.0, "shaft_length": 15.0},
        ],
        notes="2208小型无刷电机。KV值约1000-1500。用于小型无人机、小型机器人关节驱动。",
    ),

    "compression_spring": PartTemplate(
        id="compression_spring",
        name_en="Compression Spring (DIN 2098)",
        name_cn="压缩弹簧 (DIN 2098)",
        category="structural",
        subcategory="spring",
        description="圆柱压缩弹簧，参数化设计，线径/外径/自由长度/有效圈数可调",
        tags=["弹簧", "压缩弹簧", "spring", "compression", "DIN 2098"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="Custom",
        parameters=[
            ParamDef("wire_diameter", "线径", "mm", 1.0, 0.3, 5, 0.1, fixed=True),
            ParamDef("outer_diameter", "外径", "mm", 10.0, 3, 60, 0.5, fixed=True),
            ParamDef("free_length", "自由长度", "mm", 30.0, 5, 200, 1, fixed=True),
            ParamDef("active_coils", "有效圈数", "", 6, 2, 20, 1, fixed=True),
        ],
        fc_script_template=_COMPRESSION_SPRING_SCRIPT,
        standard_sizes=[
            # Common 3D printer springs
            {"wire_diameter": 1.0, "outer_diameter": 8.0, "free_length": 25.0, "active_coils": 6},
            {"wire_diameter": 1.2, "outer_diameter": 10.0, "free_length": 30.0, "active_coils": 7},
            {"wire_diameter": 1.5, "outer_diameter": 12.0, "free_length": 40.0, "active_coils": 8},
            {"wire_diameter": 2.0, "outer_diameter": 15.0, "free_length": 50.0, "active_coils": 6},
        ],
        notes="弹簧常数k = Gd⁴/(8D³n)，G为剪切模量(钢丝≈79GPa)。"
              "建模使用螺旋扫掠，若FreeCAD不支持则降级为空心圆柱。",
    ),

    "damper_shock_absorber": PartTemplate(
        id="damper_shock_absorber",
        name_en="Damper / Shock Absorber",
        name_cn="阻尼器/减震器",
        category="structural",
        subcategory="damper",
        description="液压/气压阻尼器，缸体+活塞杆+两端环耳，机器人悬挂/减震",
        tags=["阻尼器", "减震器", "damper", "shock absorber", "悬挂", "机器人"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="Custom",
        parameters=[
            ParamDef("cylinder_diameter", "缸体外径", "mm", 18.0, 8, 50, 0.5, fixed=True),
            ParamDef("cylinder_length", "缸体长度", "mm", 60.0, 20, 200, 1, fixed=True),
            ParamDef("rod_diameter", "活塞杆直径", "mm", 6.0, 3, 20, 0.5, fixed=True),
            ParamDef("rod_length", "活塞杆长度", "mm", 50.0, 10, 150, 1, fixed=True),
            ParamDef("mount_diameter", "安装环外径", "mm", 12.0, 5, 30, 0.5, fixed=True),
            ParamDef("mount_thickness", "安装环厚度", "mm", 4.0, 2, 10, 0.5, fixed=True),
            ParamDef("mount_hole_diameter", "安装孔径", "mm", 5.0, 2, 12, 0.5, fixed=True),
        ],
        fc_script_template=_DAMPER_SCRIPT,
        standard_sizes=[
            {"cylinder_diameter": 18.0, "cylinder_length": 60.0,
             "rod_diameter": 6.0, "rod_length": 50.0,
             "mount_diameter": 12.0, "mount_thickness": 4.0, "mount_hole_diameter": 5.0},
            {"cylinder_diameter": 22.0, "cylinder_length": 80.0,
             "rod_diameter": 8.0, "rod_length": 60.0,
             "mount_diameter": 15.0, "mount_thickness": 5.0, "mount_hole_diameter": 6.0},
        ],
        notes="液压阻尼器通过节流孔产生阻尼力。机器人常用型号行程25-100mm。"
              "两端环耳安装方式。缸体充氮气或液压油。",
    ),

    # ---- Task 75: Electronics & sensor parts ----

    "driver_l298n": PartTemplate(
        id="driver_l298n",
        name_en="L298N Motor Driver",
        name_cn="L298N 电机驱动板",
        category="electronics",
        subcategory="motor_driver",
        description="L298N 双H桥电机驱动板，驱动2路直流电机或1路步进电机，带散热片",
        tags=["电机驱动", "L298N", "motor driver", "H-bridge", "双路", "driver"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="STMicroelectronics", model_number="L298N",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 43.0, 20, 80, 0.5, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 43.0, 20, 80, 0.5, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
            ParamDef("heatsink_length", "散热片长", "mm", 20.0, 5, 40, 0.5, fixed=True),
            ParamDef("heatsink_width", "散热片宽", "mm", 15.0, 5, 30, 0.5, fixed=True),
            ParamDef("heatsink_height", "散热片高", "mm", 10.0, 3, 20, 0.5, fixed=True),
        ],
        fc_script_template=_L298N_SCRIPT,
        standard_sizes=[
            {"pcb_length": 43.0, "pcb_width": 43.0, "pcb_thickness": 1.6,
             "heatsink_length": 20.0, "heatsink_width": 15.0, "heatsink_height": 10.0},
        ],
        notes="逻辑电压5V，驱动电压5-35V，单路最大电流2A（峰值3A）。内置5V稳压输出。"
              "板载散热片必须安装。4×M3安装孔。",
    ),

    "driver_tb6612fng": PartTemplate(
        id="driver_tb6612fng",
        name_cn="TB6612FNG 电机驱动模块",
        name_en="TB6612FNG Motor Driver Breakout",
        category="electronics",
        subcategory="motor_driver",
        description="TB6612FNG 小型双H桥电机驱动模块，体积小巧，适合微型机器人",
        tags=["电机驱动", "TB6612FNG", "motor driver", "小型", "breakout"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Toshiba", model_number="TB6612FNG",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 20.0, 10, 40, 0.5, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 15.0, 8, 30, 0.5, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
        ],
        fc_script_template=_TB6612FNG_SCRIPT,
        standard_sizes=[
            {"pcb_length": 20.0, "pcb_width": 15.0, "pcb_thickness": 1.6},
        ],
        notes="驱动电压2.5-13.5V，单路最大电流1.2A（峰值3.2A）。比L298N更小更高效。"
              "需要PWM控制。无内置散热片。",
    ),

    "controller_arduino_uno": PartTemplate(
        id="controller_arduino_uno",
        name_cn="Arduino Uno 开发板",
        name_en="Arduino Uno Rev3",
        category="electronics",
        subcategory="controller",
        description="Arduino Uno Rev3 主控制器，ATmega328P，14路数字I/O+6路模拟输入",
        tags=["Arduino", "Uno", "主控制器", "开发板", "microcontroller", "ATmega328P"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Arduino", model_number="Uno-Rev3",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 68.6, 30, 100, 0.1, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 53.4, 20, 80, 0.1, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
        ],
        fc_script_template=_ARDUINO_UNO_SCRIPT,
        standard_sizes=[
            {"pcb_length": 68.6, "pcb_width": 53.4, "pcb_thickness": 1.6},
        ],
        notes="ATmega328P @ 16MHz。工作电压5V，输入电压7-12V。USB Type-B接口。"
              "4个安装孔M3。尺寸68.6×53.4mm。",
    ),

    "controller_arduino_nano": PartTemplate(
        id="controller_arduino_nano",
        name_cn="Arduino Nano 开发板",
        name_en="Arduino Nano",
        category="electronics",
        subcategory="controller",
        description="Arduino Nano 小型开发板，ATmega328P，USB Mini-B，适合空间受限场景",
        tags=["Arduino", "Nano", "小型", "开发板", "microcontroller", "紧凑"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Arduino", model_number="Nano",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 45.0, 20, 80, 0.1, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 18.0, 8, 40, 0.1, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
        ],
        fc_script_template=_ARDUINO_NANO_SCRIPT,
        standard_sizes=[
            {"pcb_length": 45.0, "pcb_width": 18.0, "pcb_thickness": 1.6},
        ],
        notes="ATmega328P @ 16MHz。尺寸45×18mm。30针排针接口。USB Mini-B供电/下载。"
              "适合嵌入小型机器人。",
    ),

    "controller_esp32_devkit": PartTemplate(
        id="controller_esp32_devkit",
        name_cn="ESP32 DevKit 开发板",
        name_en="ESP32 DevKit V1",
        category="electronics",
        subcategory="controller",
        description="ESP32 DevKit V1，双核Wi-Fi+BLE微控制器，适合IoT和机器人",
        tags=["ESP32", "Wi-Fi", "BLE", "开发板", "IoT", "robot", "microcontroller"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Espressif", model_number="ESP32-DevKitC-V4",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 55.0, 25, 80, 0.5, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 28.0, 10, 50, 0.5, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
        ],
        fc_script_template=_ESP32_DEVKIT_SCRIPT,
        standard_sizes=[
            {"pcb_length": 55.0, "pcb_width": 28.0, "pcb_thickness": 1.6},
        ],
        notes="ESP32-WROOM-32模组。双核240MHz，520KB SRAM，Wi-Fi 802.11 b/g/n，BLE 4.2。"
              "38针排针接口。USB Micro-B。尺寸55×28mm。",
    ),

    "encoder_as5600": PartTemplate(
        id="encoder_as5600",
        name_cn="AS5600 磁编码器模块",
        name_en="AS5600 Magnetic Encoder Module",
        category="sensor",
        subcategory="encoder",
        description="AS5600 14-bit 磁编码器，I2C/PWM输出，用于关节角度检测、闭环控制",
        tags=["编码器", "磁编码器", "AS5600", "角度", "encoder", "magnetic", "I2C"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="ams-OSRAM", model_number="AS5600",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 22.0, 10, 40, 0.5, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 22.0, 10, 40, 0.5, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
            ParamDef("center_hole_diameter", "中心孔径", "mm", 6.0, 2, 15, 0.5, fixed=True),
            ParamDef("magnet_diameter", "磁铁直径", "mm", 6.0, 2, 10, 0.5, fixed=True),
            ParamDef("magnet_height", "磁铁高度", "mm", 2.5, 1, 5, 0.5, fixed=True),
        ],
        fc_script_template=_AS5600_ENCODER_SCRIPT,
        standard_sizes=[
            {"pcb_length": 22.0, "pcb_width": 22.0, "pcb_thickness": 1.6,
             "center_hole_diameter": 6.0, "magnet_diameter": 6.0, "magnet_height": 2.5},
        ],
        notes="14-bit分辨率（0.022°/step）。I2C地址0x36。内径6mm中孔，用于安装在轴端。"
              "需配合径向磁化磁铁（直径6mm）使用。功耗约5mA@3.3V。",
    ),

    "limit_switch_kw12": PartTemplate(
        id="limit_switch_kw12",
        name_cn="KW12-3 微动限位开关",
        name_en="KW12-3 Limit Switch (Micro Switch)",
        category="sensor",
        subcategory="limit_switch",
        description="KW12-3 机械微动限位开关，用于机器人限位/零点检测",
        tags=["限位开关", "微动开关", "limit switch", "micro switch", "KW12"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="KW12-3",
        parameters=[
            ParamDef("body_length", "本体长度", "mm", 20.0, 8, 40, 0.5, fixed=True),
            ParamDef("body_width", "本体宽度", "mm", 6.5, 3, 15, 0.5, fixed=True),
            ParamDef("body_height", "本体高度", "mm", 10.5, 5, 20, 0.5, fixed=True),
            ParamDef("lever_length", "杠杆长度", "mm", 15.0, 5, 30, 0.5, fixed=True),
        ],
        fc_script_template=_LIMIT_SWITCH_SCRIPT,
        standard_sizes=[
            {"body_length": 20.0, "body_width": 6.5, "body_height": 10.5, "lever_length": 15.0},
            {"body_length": 20.0, "body_width": 6.5, "body_height": 10.5, "lever_length": 25.0},
        ],
        notes="SPDT（单刀双掷），3引脚（COM/NC/NO）。额定电流5A@250VAC。"
              "机械寿命100万次以上。杠杆长度有多种变体。",
    ),

    "power_lm2596_buck": PartTemplate(
        id="power_lm2596_buck",
        name_cn="LM2596 降压模块",
        name_en="LM2596 Buck Converter Module",
        category="electronics",
        subcategory="power_module",
        description="LM2596 可调降压模块，输入4-35V，输出1.5-30V可调，机器人电源常用",
        tags=["降压模块", "LM2596", "buck converter", "电源", "power", "voltage regulator"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (Texas Instruments clone)", model_number="LM2596",
        parameters=[
            ParamDef("pcb_length", "PCB长度", "mm", 43.0, 20, 80, 0.5, fixed=True),
            ParamDef("pcb_width", "PCB宽度", "mm", 21.0, 10, 50, 0.5, fixed=True),
            ParamDef("pcb_thickness", "PCB厚度", "mm", 1.6, 0.8, 3.0, 0.1, fixed=True),
            ParamDef("inductor_diameter", "电感直径", "mm", 15.0, 5, 30, 0.5, fixed=True),
            ParamDef("inductor_height", "电感高度", "mm", 8.0, 3, 15, 0.5, fixed=True),
        ],
        fc_script_template=_LM2596_BUCK_SCRIPT,
        standard_sizes=[
            {"pcb_length": 43.0, "pcb_width": 21.0, "pcb_thickness": 1.6,
             "inductor_diameter": 15.0, "inductor_height": 8.0},
        ],
        notes="输出电流最大3A（建议2A以内长期使用）。效率约80-90%。"
              "蓝色可调电位器调输出电压。输入输出各2针（IN+/IN-/OUT+/OUT-）。",
    ),

    "connector_xt60": PartTemplate(
        id="connector_xt60",
        name_cn="XT60 电源连接器",
        name_en="XT60 Power Connector",
        category="electronics",
        subcategory="connector",
        description="XT60 电源连接器，无人机/机器人电池常用，额定电流60A",
        tags=["连接器", "XT60", "电源", "connector", "battery", "power"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (AMASS clone)", model_number="XT60",
        parameters=[
            ParamDef("body_diameter", "本体直径", "mm", 12.0, 6, 20, 0.5, fixed=True),
            ParamDef("body_length", "本体长度", "mm", 16.0, 8, 30, 0.5, fixed=True),
            ParamDef("contact_diameter", "触点直径", "mm", 3.0, 1.5, 5, 0.5, fixed=True),
        ],
        fc_script_template=_XT60_CONNECTOR_SCRIPT,
        standard_sizes=[
            {"body_diameter": 12.0, "body_length": 16.0, "contact_diameter": 3.0},
        ],
        notes="额定电压600V，额定电流60A。镀金铜触点。阻燃PA材质。"
              "公头接电池端（红线+，黑线-）。常用于无人机/AGV电池接口。",
    ),

    "connector_jst_xh": PartTemplate(
        id="connector_jst_xh",
        name_cn="JST-XH 连接器",
        name_en="JST-XH Connector (2-6 pin)",
        category="electronics",
        subcategory="connector",
        description="JST-XH 连接器，2-6针，信号/传感器接线常用",
        tags=["连接器", "JST", "XH", "connector", "信号", "sensor"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="JST", model_number="XH",
        parameters=[
            ParamDef("num_pins", "针数", "", 4, 2, 6, 1, fixed=True),
            ParamDef("body_width", "本体宽度", "mm", 5.0, 3, 8, 0.5, fixed=True),
            ParamDef("body_height", "本体高度", "mm", 7.0, 4, 12, 0.5, fixed=True),
        ],
        fc_script_template=_JST_XH_CONNECTOR_SCRIPT,
        standard_sizes=[
            {"num_pins": 2, "body_width": 5.0, "body_height": 7.0},
            {"num_pins": 3, "body_width": 5.0, "body_height": 7.0},
            {"num_pins": 4, "body_width": 5.0, "body_height": 7.0},
            {"num_pins": 6, "body_width": 5.0, "body_height": 7.0},
        ],
        notes="额定电压250V，额定电流3A。间距2.5mm。常用于传感器、舵机、限位开关接线。"
              "白色外壳，带锁扣防松脱。",
    ),

    # ---- ROBOTIS DYNAMIXEL & OpenMANIPULATOR-X parts ----

    "dynamixel_xm430_w350": PartTemplate(
        id="dynamixel_xm430_w350",
        name_en="DYNAMIXEL XM430-W350-T",
        name_cn="DYNAMIXEL XM430-W350-T 智能舵机",
        category="actuator",
        subcategory="servo",
        description="ROBOTIS DYNAMIXEL XM430-W350-T 智能伺服舵机，OpenMANIPULATOR-X 标准执行器",
        tags=["DYNAMIXEL", "XM430", "ROBOTIS", "servo", "智能舵机", "actuator",
              "OpenMANIPULATOR", "伺服电机"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="ROBOTIS", model_number="XM430-W350-T",
        parameters=[
            ParamDef("body_width", "本体宽度", "mm", 28.0, 20, 40, 0.5, fixed=True),
            ParamDef("body_height", "本体高度", "mm", 46.5, 30, 60, 0.5, fixed=True),
            ParamDef("body_depth", "本体深度", "mm", 34.0, 20, 45, 0.5, fixed=True),
            ParamDef("horn_diameter", "输出轴法兰直径", "mm", 22.0, 10, 30, 0.5, fixed=True),
            ParamDef("shaft_diameter", "输出轴直径", "mm", 6.0, 3, 10, 0.5, fixed=True),
        ],
        fc_script_template=_XM430_SCRIPT,
        standard_sizes=[
            {"body_width": 28.0, "body_height": 46.5, "body_depth": 34.0,
             "horn_diameter": 22.0, "shaft_diameter": 6.0},
        ],
        notes="XM430-W350-T: 额定扭矩 4.1 Nm @ 12V, 空载转速 46 RPM, "
              "分辨率 0.088°, 质量 82g, TTL/RS485 通信, "
              "供电 10.0~14.8V, 待机电流 52mA, "
              "工作温度 -5~55°C, IP 等级无。"
              "安装面: 4×M2.5 螺栓, 间距 16×16mm。",
    ),

    "robotis_fr12_h101": PartTemplate(
        id="robotis_fr12_h101",
        name_en="ROBOTIS FR12-H101-K Frame",
        name_cn="ROBOTIS FR12-H101-K H型框架",
        category="structural",
        subcategory="bracket",
        description="ROBOTIS FR12-H101-K 铝合金H型连接框架，连接两个XM430舵机",
        tags=["ROBOTIS", "FR12", "frame", "bracket", "H型", "框架",
              "DYNAMIXEL", "OpenMANIPULATOR"],
        part_class="structural", scalable=False, real_part=True,
        manufacturer="ROBOTIS", model_number="FR12-H101-K",
        parameters=[
            ParamDef("length", "长度", "mm", 28.0, 15, 60, 1, fixed=True),
            ParamDef("width", "宽度", "mm", 28.0, 15, 40, 1, fixed=True),
            ParamDef("height", "高度", "mm", 22.0, 10, 40, 1, fixed=True),
        ],
        fc_script_template="""\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("fr12_h101")
# H-type block: 28x28x22mm
body = Part.makeBox({length}, {width}, {height})
# Front face: 4x M2.5 through holes at 16x16mm spacing
hole_r = 2.9 / 2
half = 8.0
cx, cy = {length}/2, {width}/2
for dx in [-half, half]:
    for dy in [-half, half]:
        hole = Part.makeCylinder(hole_r, {height} + 2)
        hole.translate(FreeCAD.Vector(cx + dx, cy + dy, -1))
        body = body.cut(hole)
# Center shaft bore (Ø6mm)
bore = Part.makeCylinder(3.0, {height} + 2)
bore.translate(FreeCAD.Vector(cx, cy, -1))
body = body.cut(bore)
# Back face: same 4x M2.5 pattern (holes already through)
obj = doc.addObject("Part::Feature", "FR12_H101")
obj.Shape = body
doc.recompute()
""",
        standard_sizes=[
            {"length": 28.0, "width": 28.0, "height": 22.0},
        ],
        notes="铝合金切削件。两侧各4×M2.5安装孔，间距16×16mm，匹配XM430安装面。"
              "用于OpenMANIPULATOR-X的link2和link5。",
    ),

    "robotis_fr12_h104": PartTemplate(
        id="robotis_fr12_h104",
        name_en="ROBOTIS FR12-H104-K Frame",
        name_cn="ROBOTIS FR12-H104-K 长H型框架",
        category="structural",
        subcategory="bracket",
        description="ROBOTIS FR12-H104-K 加长H型框架，连接XM430与夹爪机构",
        tags=["ROBOTIS", "FR12", "frame", "bracket", "H型", "长", "框架",
              "DYNAMIXEL", "OpenMANIPULATOR"],
        part_class="structural", scalable=False, real_part=True,
        manufacturer="ROBOTIS", model_number="FR12-H104-K",
        parameters=[
            ParamDef("length", "长度", "mm", 72.0, 30, 100, 1, fixed=True),
            ParamDef("width", "宽度", "mm", 28.0, 15, 40, 1, fixed=True),
            ParamDef("height", "高度", "mm", 22.0, 10, 40, 1, fixed=True),
        ],
        fc_script_template="""\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("fr12_h104")
# Extended H-type block: 72x28x22mm
body = Part.makeBox({length}, {width}, {height})
# Left face: 4x M2.5 through holes at 16x16mm spacing (XM430 mount)
hole_r = 2.9 / 2
half = 8.0
for dx in [-half, half]:
    for dy in [-half, half]:
        hole = Part.makeCylinder(hole_r, {height} + 2)
        hole.translate(FreeCAD.Vector(14.0 + dx, 14.0 + dy, -1))
        body = body.cut(hole)
# Left center bore
bore1 = Part.makeCylinder(3.0, {height} + 2)
bore1.translate(FreeCAD.Vector(14.0, 14.0, -1))
body = body.cut(bore1)
# Right face: rail mounting slot (simplified as 2x M2.5 holes)
for dx2 in [-6, 6]:
    hole2 = Part.makeCylinder(hole_r, {height} + 2)
    hole2.translate(FreeCAD.Vector({length} - 14.0 + dx2, 14.0, -1))
    body = body.cut(hole2)
# Right center bore
bore2 = Part.makeCylinder(3.0, {height} + 2)
bore2.translate(FreeCAD.Vector({length} - 14.0, 14.0, -1))
body = body.cut(bore2)
obj = doc.addObject("Part::Feature", "FR12_H104")
obj.Shape = body
doc.recompute()
""",
        standard_sizes=[
            {"length": 72.0, "width": 28.0, "height": 22.0},
        ],
        notes="FR12-H104-K 铝合金切削件。一端4×M2.5匹配XM430，另一端导轨安装面。"
              "用于OpenMANIPULATOR-X的link5（腕部+夹爪基座）。",
    ),

    "robotis_fr12_s101": PartTemplate(
        id="robotis_fr12_s101",
        name_en="ROBOTIS FR12-S101-K Frame",
        name_cn="ROBOTIS FR12-S101-K 短连杆框架",
        category="structural",
        subcategory="bracket",
        description="ROBOTIS FR12-S101-K U型短连杆框架，连接两个XM430",
        tags=["ROBOTIS", "FR12", "frame", "bracket", "U型", "短连杆",
              "DYNAMIXEL", "OpenMANIPULATOR"],
        part_class="structural", scalable=False, real_part=True,
        manufacturer="ROBOTIS", model_number="FR12-S101-K",
        parameters=[
            ParamDef("length", "长度", "mm", 48.0, 20, 80, 1, fixed=True),
            ParamDef("width", "宽度", "mm", 28.0, 15, 40, 1, fixed=True),
            ParamDef("height", "高度", "mm", 16.0, 8, 30, 1, fixed=True),
        ],
        fc_script_template="""\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("fr12_s101")
# U-type short link: 48x28x16mm
wall_t = 4.0
flange = 6.0
# Top flange
top = Part.makeBox({length}, {width}, flange)
# Bottom flange
bottom = Part.makeBox({length}, {width}, flange)
bottom.translate(FreeCAD.Vector(0, 0, {height} - flange))
# Left web
web_l = Part.makeBox(flange, {width}, {height} - 2 * flange)
web_l.translate(FreeCAD.Vector(0, 0, flange))
# Right web
web_r = Part.makeBox(flange, {width}, {height} - 2 * flange)
web_r.translate(FreeCAD.Vector({length} - flange, 0, flange))
body = top.fuse(bottom).fuse(web_l).fuse(web_r)
# Mounting holes on both flanges (4x M2.5 at 16x16mm spacing)
hole_r = 2.9 / 2
half = 8.0
# Left end holes (through both flanges)
for dx in [-half, half]:
    for dy in [-half, half]:
        hole = Part.makeCylinder(hole_r, {height} + 2)
        hole.translate(FreeCAD.Vector(14.0 + dx, 14.0 + dy, -1))
        body = body.cut(hole)
# Right end holes
for dx in [-half, half]:
    for dy in [-half, half]:
        hole = Part.makeCylinder(hole_r, {height} + 2)
        hole.translate(FreeCAD.Vector({length} - 14.0 + dx, 14.0 + dy, -1))
        body = body.cut(hole)
obj = doc.addObject("Part::Feature", "FR12_S101")
obj.Shape = body
doc.recompute()
""",
        standard_sizes=[
            {"length": 48.0, "width": 28.0, "height": 16.0},
        ],
        notes="FR12-S101-K 铝合金切削件。用于OpenMANIPULATOR-X的link4（前臂）。",
    ),

    "robotis_fr12_s102": PartTemplate(
        id="robotis_fr12_s102",
        name_en="ROBOTIS FR12-S102-K Frame",
        name_cn="ROBOTIS FR12-S102-K 长连杆框架",
        category="structural",
        subcategory="bracket",
        description="ROBOTIS FR12-S102-K U型长连杆框架，连接两个XM430",
        tags=["ROBOTIS", "FR12", "frame", "bracket", "U型", "长连杆",
              "DYNAMIXEL", "OpenMANIPULATOR"],
        part_class="structural", scalable=False, real_part=True,
        manufacturer="ROBOTIS", model_number="FR12-S102-K",
        parameters=[
            ParamDef("length", "长度", "mm", 96.0, 40, 150, 1, fixed=True),
            ParamDef("width", "宽度", "mm", 28.0, 15, 40, 1, fixed=True),
            ParamDef("height", "高度", "mm", 16.0, 8, 30, 1, fixed=True),
        ],
        fc_script_template="""\
import FreeCAD, Part, math
doc = FreeCAD.newDocument("fr12_s102")
# U-type long link: 96x28x16mm
wall_t = 4.0
flange = 6.0
# Top flange
top = Part.makeBox({length}, {width}, flange)
# Bottom flange
bottom = Part.makeBox({length}, {width}, flange)
bottom.translate(FreeCAD.Vector(0, 0, {height} - flange))
# Left web
web_l = Part.makeBox(flange, {width}, {height} - 2 * flange)
web_l.translate(FreeCAD.Vector(0, 0, flange))
# Right web
web_r = Part.makeBox(flange, {width}, {height} - 2 * flange)
web_r.translate(FreeCAD.Vector({length} - flange, 0, flange))
body = top.fuse(bottom).fuse(web_l).fuse(web_r)
# Mounting holes on both ends (4x M2.5 at 16x16mm spacing, through both flanges)
hole_r = 2.9 / 2
half = 8.0
# Left end holes
for dx in [-half, half]:
    for dy in [-half, half]:
        hole = Part.makeCylinder(hole_r, {height} + 2)
        hole.translate(FreeCAD.Vector(14.0 + dx, 14.0 + dy, -1))
        body = body.cut(hole)
# Right end holes
for dx in [-half, half]:
    for dy in [-half, half]:
        hole = Part.makeCylinder(hole_r, {height} + 2)
        hole.translate(FreeCAD.Vector({length} - 14.0 + dx, 14.0 + dy, -1))
        body = body.cut(hole)
obj = doc.addObject("Part::Feature", "FR12_S102")
obj.Shape = body
doc.recompute()
""",
        standard_sizes=[
            {"length": 96.0, "width": 28.0, "height": 16.0},
        ],
        notes="FR12-S102-K 铝合金切削件。用于OpenMANIPULATOR-X的link3（上臂）。",
    ),
}


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
        bore_diameter=22.0,
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
    "linear_motion": ["linear_bearing_lm8uu", "linear_bearing_lm10uu", "linear_bearing_lm12uu",
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

    return script_template.format_map(clean_params)
