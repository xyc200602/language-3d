"""CAD modeling patterns and common operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelingPattern:
    """A reusable CAD modeling pattern."""

    name: str
    description: str
    steps: list[str]
    required_tools: list[str]


COMMON_PATTERNS: dict[str, ModelingPattern] = {
    "box": ModelingPattern(
        name="Box (立方体)",
        description="创建一个立方体",
        steps=[
            "选择 XY_Plane 基准面，进入草图模式",
            "绘制矩形，标注尺寸",
            "退出草图，拉伸到指定深度",
        ],
        required_tools=["fc_batch"],
    ),
    "cylinder": ModelingPattern(
        name="Cylinder (圆柱体)",
        description="创建一个圆柱体",
        steps=[
            "选择 XY_Plane 基准面，进入草图模式",
            "绘制圆，标注半径",
            "退出草图，拉伸到指定高度",
        ],
        required_tools=["fc_batch"],
    ),
    "plate_with_holes": ModelingPattern(
        name="Plate with Holes (带孔板)",
        description="创建一块带安装孔的板",
        steps=[
            "选择 XY_Plane 基准面，进入草图模式",
            "绘制矩形轮廓",
            "在四角绘制圆形（安装孔位置）",
            "使用布尔减去圆形区域",
            "退出草图，拉伸到板厚",
        ],
        required_tools=["fc_batch"],
    ),
    "hollow_cylinder": ModelingPattern(
        name="Hollow Cylinder (空心圆柱)",
        description="创建一个空心圆柱（管状）",
        steps=[
            "选择 XY_Plane 基准面，进入草图模式",
            "绘制外圆",
            "绘制内圆（同心）",
            "退出草图，拉伸到指定高度",
        ],
        required_tools=["fc_batch"],
    ),
    "servo_mount": ModelingPattern(
        name="Servo Mount (舵机安装座)",
        description="创建舵机安装座",
        steps=[
            "创建底板矩形 (30mm x 24mm)",
            "拉伸底板 (3mm 厚)",
            "在两侧创建侧板",
            "在侧板上钻螺丝孔",
            "倒角处理",
        ],
        required_tools=["fc_batch"],
    ),
}

# Feature naming conventions
FEATURE_NAMING = {
    "sketch": "Sketch{N}_{description}",
    "extrude": "Extrude{N}_{description}",
    "cut": "Cut{N}_{description}",
    "fillet": "Fillet{N}_{description}",
    "chamfer": "Chamfer{N}_{description}",
}

# Standard plane names
PLANES = {
    "front": "前视基准面 (Front Plane)",
    "top": "上视基准面 (Top Plane)",
    "right": "右视基准面 (Right Plane)",
}


def get_pattern(name: str) -> ModelingPattern | None:
    return COMMON_PATTERNS.get(name)


def list_patterns() -> list[str]:
    return list(COMMON_PATTERNS.keys())
