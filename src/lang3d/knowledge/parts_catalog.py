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
    default: float = 0.0
    min_value: float = 0.0
    max_value: float = 1000.0
    step: float = 0.1
    fixed: bool = False  # If True, user cannot change this parameter


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


@dataclass
class GeneratedPart:
    """Record of a generated part instance."""

    template_id: str
    name: str
    parameters: dict[str, float]
    fcstd_path: str = ""
    stl_path: str = ""
    created_at: str = ""


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
        ],
        fc_script_template=_SOCKET_HEAD_CAP_SCREW_SCRIPT,
        standard_sizes=[
            {"thread_diameter": 3, "length": 10, "head_diameter": 5.5},
            {"thread_diameter": 3, "length": 20, "head_diameter": 5.5},
            {"thread_diameter": 4, "length": 16, "head_diameter": 7.0},
            {"thread_diameter": 5, "length": 20, "head_diameter": 8.5},
            {"thread_diameter": 6, "length": 25, "head_diameter": 10.0},
            {"thread_diameter": 8, "length": 30, "head_diameter": 13.0},
        ],
        notes="螺纹为简化圆柱体表示，非真实螺纹几何",
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
        ],
        fc_script_template=_HEX_NUT_SCRIPT,
        standard_sizes=[
            {"nominal_diameter": 3},
            {"nominal_diameter": 4},
            {"nominal_diameter": 5},
            {"nominal_diameter": 6},
            {"nominal_diameter": 8},
            {"nominal_diameter": 10},
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
        ],
        fc_script_template=_HEX_BOLT_SCRIPT,
        standard_sizes=[
            {"thread_diameter": 4, "length": 20},
            {"thread_diameter": 5, "length": 25},
            {"thread_diameter": 6, "length": 30},
            {"thread_diameter": 8, "length": 40},
            {"thread_diameter": 10, "length": 50},
        ],
        notes="螺纹为简化圆柱体表示",
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
        ],
        fc_script_template=_BEARING_SCRIPT,
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
        ],
        fc_script_template=_BEARING_SCRIPT,
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
        ],
        fc_script_template=_BEARING_SCRIPT,
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
        ],
        fc_script_template=_SPUR_GEAR_SCRIPT,
        standard_sizes=[
            {"teeth": 20, "module": 1.0, "thickness": 6, "bore_diameter": 8},
            {"teeth": 30, "module": 1.0, "thickness": 6, "bore_diameter": 8},
            {"teeth": 16, "module": 1.5, "thickness": 8, "bore_diameter": 10},
        ],
        notes="齿轮为简化圆柱体表示，非渐开线齿廓",
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
) -> dict[str, float]:
    """Fill defaults for missing parameters and validate ranges.

    Returns a complete parameter dict with all parameters filled in.
    Raises ValueError if a parameter is out of range.
    """
    resolved: dict[str, float] = {}
    for pdef in template.parameters:
        if params and pdef.name in params:
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


def format_fc_script(template: PartTemplate, params: dict[str, float]) -> str:
    """Substitute parameter values into the FreeCAD script template.

    Uses Python str.format_map to replace {param_name} placeholders.
    """
    # Convert float values to clean strings (avoid scientific notation)
    clean_params: dict[str, str] = {}
    for k, v in params.items():
        if v == int(v):
            clean_params[k] = str(int(v))
        else:
            clean_params[k] = str(v)

    return template.fc_script_template.format_map(clean_params)
