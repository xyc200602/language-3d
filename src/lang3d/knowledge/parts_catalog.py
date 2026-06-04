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
    "bearing": ["ball_bearing"],
    "actuator": ["servo", "stepper"],
    "shaft": ["linear", "coupling"],
    "gear": ["spur"],
    "structural": ["bracket", "plate"],
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
# Standard parts catalog (15 templates)
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
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def search_parts(
    query: str = "",
    category: str | None = None,
    tags: list[str] | None = None,
) -> list[PartTemplate]:
    """Search parts by keyword, category, and/or tags.

    Keyword matching is case-insensitive and checks name_en, name_cn,
    description, and tags.
    """
    results: list[PartTemplate] = []
    query_lower = query.lower().strip() if query else ""

    for template in PART_CATALOG.values():
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
        params.get("tooth_detail", params.get("bearing_detail", "simplified")),
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
