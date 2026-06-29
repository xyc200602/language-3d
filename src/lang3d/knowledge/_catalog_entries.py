"""PART_CATALOG dict entries — extracted from parts_catalog.py.

This file contains the ~2800-line dict body. It imports ALL names it
needs (data models, script templates, helper functions) from parts_catalog
so the entries resolve identically to the original inline dict.
"""

from __future__ import annotations

# Import everything the entries reference. Using a star-style import from
# parts_catalog would create a circular dependency (parts_catalog imports
# PART_CATALOG from here), so we do a deferred module-level import.
import lang3d.knowledge.parts_catalog as _pc

# Make all names from parts_catalog available at module level
PartTemplate = _pc.PartTemplate
ParamDef = _pc.ParamDef
BoltHole = _pc.BoltHole
AlignmentFeature = _pc.AlignmentFeature
MountingInterface = _pc.MountingInterface
GeneratedPart = _pc.GeneratedPart
if hasattr(_pc, "_ALUMINUM_EXTRUSION_SCRIPT"): _ALUMINUM_EXTRUSION_SCRIPT = _pc._ALUMINUM_EXTRUSION_SCRIPT
if hasattr(_pc, "_ARDUINO_NANO_SCRIPT"): _ARDUINO_NANO_SCRIPT = _pc._ARDUINO_NANO_SCRIPT
if hasattr(_pc, "_ARDUINO_UNO_SCRIPT"): _ARDUINO_UNO_SCRIPT = _pc._ARDUINO_UNO_SCRIPT
if hasattr(_pc, "_AS5600_ENCODER_SCRIPT"): _AS5600_ENCODER_SCRIPT = _pc._AS5600_ENCODER_SCRIPT
if hasattr(_pc, "_BASE_PLATE_SCRIPT"): _BASE_PLATE_SCRIPT = _pc._BASE_PLATE_SCRIPT
if hasattr(_pc, "_BATTERY_BOX_SCRIPT"): _BATTERY_BOX_SCRIPT = _pc._BATTERY_BOX_SCRIPT
if hasattr(_pc, "_BATTERY_HOLDER_SCRIPT"): _BATTERY_HOLDER_SCRIPT = _pc._BATTERY_HOLDER_SCRIPT
if hasattr(_pc, "_BATTERY_TRAY_SCRIPT"): _BATTERY_TRAY_SCRIPT = _pc._BATTERY_TRAY_SCRIPT
if hasattr(_pc, "_BEARING_BLOCK_SCRIPT"): _BEARING_BLOCK_SCRIPT = _pc._BEARING_BLOCK_SCRIPT
if hasattr(_pc, "_BEARING_REALISTIC_SCRIPT"): _BEARING_REALISTIC_SCRIPT = _pc._BEARING_REALISTIC_SCRIPT
if hasattr(_pc, "_BEARING_SCRIPT"): _BEARING_SCRIPT = _pc._BEARING_SCRIPT
if hasattr(_pc, "_BELLOWS_COUPLING_SCRIPT"): _BELLOWS_COUPLING_SCRIPT = _pc._BELLOWS_COUPLING_SCRIPT
if hasattr(_pc, "_BLDC_MOTOR_SCRIPT"): _BLDC_MOTOR_SCRIPT = _pc._BLDC_MOTOR_SCRIPT
if hasattr(_pc, "_CABLE_CHAIN_MOUNT_SCRIPT"): _CABLE_CHAIN_MOUNT_SCRIPT = _pc._CABLE_CHAIN_MOUNT_SCRIPT
if hasattr(_pc, "_CHASSIS_PLATE_SCRIPT"): _CHASSIS_PLATE_SCRIPT = _pc._CHASSIS_PLATE_SCRIPT
if hasattr(_pc, "_COMPRESSION_SPRING_SCRIPT"): _COMPRESSION_SPRING_SCRIPT = _pc._COMPRESSION_SPRING_SCRIPT
if hasattr(_pc, "_CORNER_BRACKET_SCRIPT"): _CORNER_BRACKET_SCRIPT = _pc._CORNER_BRACKET_SCRIPT
if hasattr(_pc, "_DAMPER_SCRIPT"): _DAMPER_SCRIPT = _pc._DAMPER_SCRIPT
if hasattr(_pc, "_DOWEL_PIN_SCRIPT"): _DOWEL_PIN_SCRIPT = _pc._DOWEL_PIN_SCRIPT
if hasattr(_pc, "_DS3218_SERVO_SCRIPT"): _DS3218_SERVO_SCRIPT = _pc._DS3218_SERVO_SCRIPT
if hasattr(_pc, "_ENCODER_MOUNT_SCRIPT"): _ENCODER_MOUNT_SCRIPT = _pc._ENCODER_MOUNT_SCRIPT
if hasattr(_pc, "_ESP32_CAM_SCRIPT"): _ESP32_CAM_SCRIPT = _pc._ESP32_CAM_SCRIPT
if hasattr(_pc, "_ESP32_DEVKIT_SCRIPT"): _ESP32_DEVKIT_SCRIPT = _pc._ESP32_DEVKIT_SCRIPT
if hasattr(_pc, "_FLANGE_COUPLING_SCRIPT"): _FLANGE_COUPLING_SCRIPT = _pc._FLANGE_COUPLING_SCRIPT
if hasattr(_pc, "_FLAT_WASHER_SCRIPT"): _FLAT_WASHER_SCRIPT = _pc._FLAT_WASHER_SCRIPT
if hasattr(_pc, "_FLEXIBLE_COUPLING_SCRIPT"): _FLEXIBLE_COUPLING_SCRIPT = _pc._FLEXIBLE_COUPLING_SCRIPT
if hasattr(_pc, "_GT2_BELT_SCRIPT"): _GT2_BELT_SCRIPT = _pc._GT2_BELT_SCRIPT
if hasattr(_pc, "_GT2_PULLEY_REALISTIC_SCRIPT"): _GT2_PULLEY_REALISTIC_SCRIPT = _pc._GT2_PULLEY_REALISTIC_SCRIPT
if hasattr(_pc, "_GT2_PULLEY_SCRIPT"): _GT2_PULLEY_SCRIPT = _pc._GT2_PULLEY_SCRIPT
if hasattr(_pc, "_GUIDE_RAIL_CARRIAGE_SCRIPT"): _GUIDE_RAIL_CARRIAGE_SCRIPT = _pc._GUIDE_RAIL_CARRIAGE_SCRIPT
if hasattr(_pc, "_GUSSET_PLATE_SCRIPT"): _GUSSET_PLATE_SCRIPT = _pc._GUSSET_PLATE_SCRIPT
if hasattr(_pc, "_HEAT_SET_INSERT_SCRIPT"): _HEAT_SET_INSERT_SCRIPT = _pc._HEAT_SET_INSERT_SCRIPT
if hasattr(_pc, "_HEX_BOLT_REALISTIC_SCRIPT"): _HEX_BOLT_REALISTIC_SCRIPT = _pc._HEX_BOLT_REALISTIC_SCRIPT
if hasattr(_pc, "_HEX_BOLT_SCRIPT"): _HEX_BOLT_SCRIPT = _pc._HEX_BOLT_SCRIPT
if hasattr(_pc, "_HEX_NUT_REALISTIC_SCRIPT"): _HEX_NUT_REALISTIC_SCRIPT = _pc._HEX_NUT_REALISTIC_SCRIPT
if hasattr(_pc, "_HEX_NUT_SCRIPT"): _HEX_NUT_SCRIPT = _pc._HEX_NUT_SCRIPT
if hasattr(_pc, "_HTD_PULLEY_REALISTIC_SCRIPT"): _HTD_PULLEY_REALISTIC_SCRIPT = _pc._HTD_PULLEY_REALISTIC_SCRIPT
if hasattr(_pc, "_HTD_PULLEY_SCRIPT"): _HTD_PULLEY_SCRIPT = _pc._HTD_PULLEY_SCRIPT
if hasattr(_pc, "_HUB_ADAPTER_SCRIPT"): _HUB_ADAPTER_SCRIPT = _pc._HUB_ADAPTER_SCRIPT
if hasattr(_pc, "_JGB37_520_SCRIPT"): _JGB37_520_SCRIPT = _pc._JGB37_520_SCRIPT
if hasattr(_pc, "_JOINT_HOUSING_SCRIPT"): _JOINT_HOUSING_SCRIPT = _pc._JOINT_HOUSING_SCRIPT
if hasattr(_pc, "_JST_XH_CONNECTOR_SCRIPT"): _JST_XH_CONNECTOR_SCRIPT = _pc._JST_XH_CONNECTOR_SCRIPT
if hasattr(_pc, "_L298N_SCRIPT"): _L298N_SCRIPT = _pc._L298N_SCRIPT
if hasattr(_pc, "_LIMIT_SWITCH_SCRIPT"): _LIMIT_SWITCH_SCRIPT = _pc._LIMIT_SWITCH_SCRIPT
if hasattr(_pc, "_LINEAR_BEARING_REALISTIC_SCRIPT"): _LINEAR_BEARING_REALISTIC_SCRIPT = _pc._LINEAR_BEARING_REALISTIC_SCRIPT
if hasattr(_pc, "_LINEAR_BEARING_SCRIPT"): _LINEAR_BEARING_SCRIPT = _pc._LINEAR_BEARING_SCRIPT
if hasattr(_pc, "_LINEAR_GUIDE_RAIL_SCRIPT"): _LINEAR_GUIDE_RAIL_SCRIPT = _pc._LINEAR_GUIDE_RAIL_SCRIPT
if hasattr(_pc, "_LINEAR_SHAFT_SCRIPT"): _LINEAR_SHAFT_SCRIPT = _pc._LINEAR_SHAFT_SCRIPT
if hasattr(_pc, "_LINK_ARM_SCRIPT"): _LINK_ARM_SCRIPT = _pc._LINK_ARM_SCRIPT
if hasattr(_pc, "_LM2596_BUCK_SCRIPT"): _LM2596_BUCK_SCRIPT = _pc._LM2596_BUCK_SCRIPT
if hasattr(_pc, "_L_BRACKET_SCRIPT"): _L_BRACKET_SCRIPT = _pc._L_BRACKET_SCRIPT
if hasattr(_pc, "_MOTOR_BRACKET_SCRIPT"): _MOTOR_BRACKET_SCRIPT = _pc._MOTOR_BRACKET_SCRIPT
if hasattr(_pc, "_MOTOR_MOUNT_SCRIPT"): _MOTOR_MOUNT_SCRIPT = _pc._MOTOR_MOUNT_SCRIPT
if hasattr(_pc, "_MOUNTING_PLATE_SCRIPT"): _MOUNTING_PLATE_SCRIPT = _pc._MOUNTING_PLATE_SCRIPT
if hasattr(_pc, "_MPU6050_SCRIPT"): _MPU6050_SCRIPT = _pc._MPU6050_SCRIPT
if hasattr(_pc, "_NEMA17_SCRIPT"): _NEMA17_SCRIPT = _pc._NEMA17_SCRIPT
if hasattr(_pc, "_NEMA23_SCRIPT"): _NEMA23_SCRIPT = _pc._NEMA23_SCRIPT
if hasattr(_pc, "_NEMA_MOUNT_SCRIPT"): _NEMA_MOUNT_SCRIPT = _pc._NEMA_MOUNT_SCRIPT
if hasattr(_pc, "_PCB_MOUNT_SCRIPT"): _PCB_MOUNT_SCRIPT = _pc._PCB_MOUNT_SCRIPT
if hasattr(_pc, "_PULLEY_IDLER_MOUNT_SCRIPT"): _PULLEY_IDLER_MOUNT_SCRIPT = _pc._PULLEY_IDLER_MOUNT_SCRIPT
if hasattr(_pc, "_RIGID_COUPLING_CLAMPING_SCRIPT"): _RIGID_COUPLING_CLAMPING_SCRIPT = _pc._RIGID_COUPLING_CLAMPING_SCRIPT
if hasattr(_pc, "_RIGID_COUPLING_SETSCREW_SCRIPT"): _RIGID_COUPLING_SETSCREW_SCRIPT = _pc._RIGID_COUPLING_SETSCREW_SCRIPT
if hasattr(_pc, "_RPLIDAR_A1_SCRIPT"): _RPLIDAR_A1_SCRIPT = _pc._RPLIDAR_A1_SCRIPT
if hasattr(_pc, "_SENSOR_MOUNT_SCRIPT"): _SENSOR_MOUNT_SCRIPT = _pc._SENSOR_MOUNT_SCRIPT
if hasattr(_pc, "_SENSOR_SHELF_SCRIPT"): _SENSOR_SHELF_SCRIPT = _pc._SENSOR_SHELF_SCRIPT
if hasattr(_pc, "_SERVO_BRACKET_SCRIPT"): _SERVO_BRACKET_SCRIPT = _pc._SERVO_BRACKET_SCRIPT
if hasattr(_pc, "_SERVO_MG996R_SCRIPT"): _SERVO_MG996R_SCRIPT = _pc._SERVO_MG996R_SCRIPT
if hasattr(_pc, "_SERVO_SG90_SCRIPT"): _SERVO_SG90_SCRIPT = _pc._SERVO_SG90_SCRIPT
if hasattr(_pc, "_SHAFT_COLLAR_SCRIPT"): _SHAFT_COLLAR_SCRIPT = _pc._SHAFT_COLLAR_SCRIPT
if hasattr(_pc, "_SHAFT_COUPLING_BLOCK_SCRIPT"): _SHAFT_COUPLING_BLOCK_SCRIPT = _pc._SHAFT_COUPLING_BLOCK_SCRIPT
if hasattr(_pc, "_SHAFT_SUPPORT_SCRIPT"): _SHAFT_SUPPORT_SCRIPT = _pc._SHAFT_SUPPORT_SCRIPT
if hasattr(_pc, "_SOCKET_HEAD_CAP_SCREW_REALISTIC_SCRIPT"): _SOCKET_HEAD_CAP_SCREW_REALISTIC_SCRIPT = _pc._SOCKET_HEAD_CAP_SCREW_REALISTIC_SCRIPT
if hasattr(_pc, "_SOCKET_HEAD_CAP_SCREW_SCRIPT"): _SOCKET_HEAD_CAP_SCREW_SCRIPT = _pc._SOCKET_HEAD_CAP_SCREW_SCRIPT
if hasattr(_pc, "_SPIDER_COUPLING_SCRIPT"): _SPIDER_COUPLING_SCRIPT = _pc._SPIDER_COUPLING_SCRIPT
if hasattr(_pc, "_SPUR_GEAR_REALISTIC_SCRIPT"): _SPUR_GEAR_REALISTIC_SCRIPT = _pc._SPUR_GEAR_REALISTIC_SCRIPT
if hasattr(_pc, "_SPUR_GEAR_SCRIPT"): _SPUR_GEAR_SCRIPT = _pc._SPUR_GEAR_SCRIPT
if hasattr(_pc, "_STANDOFF_COLUMN_SCRIPT"): _STANDOFF_COLUMN_SCRIPT = _pc._STANDOFF_COLUMN_SCRIPT
if hasattr(_pc, "_STANDOFF_SCRIPT"): _STANDOFF_SCRIPT = _pc._STANDOFF_SCRIPT
if hasattr(_pc, "_T8_LEADSCREW_REALISTIC_SCRIPT"): _T8_LEADSCREW_REALISTIC_SCRIPT = _pc._T8_LEADSCREW_REALISTIC_SCRIPT
if hasattr(_pc, "_T8_LEADSCREW_SCRIPT"): _T8_LEADSCREW_SCRIPT = _pc._T8_LEADSCREW_SCRIPT
if hasattr(_pc, "_T8_NUT_SCRIPT"): _T8_NUT_SCRIPT = _pc._T8_NUT_SCRIPT
if hasattr(_pc, "_TB6612FNG_SCRIPT"): _TB6612FNG_SCRIPT = _pc._TB6612FNG_SCRIPT
if hasattr(_pc, "_TT_MOTOR_SCRIPT"): _TT_MOTOR_SCRIPT = _pc._TT_MOTOR_SCRIPT
if hasattr(_pc, "_T_BRACKET_SCRIPT"): _T_BRACKET_SCRIPT = _pc._T_BRACKET_SCRIPT
if hasattr(_pc, "_T_NUT_SCRIPT"): _T_NUT_SCRIPT = _pc._T_NUT_SCRIPT
if hasattr(_pc, "_U_BRACKET_SCRIPT"): _U_BRACKET_SCRIPT = _pc._U_BRACKET_SCRIPT
if hasattr(_pc, "_WHEEL_MECANUM_SCRIPT"): _WHEEL_MECANUM_SCRIPT = _pc._WHEEL_MECANUM_SCRIPT
if hasattr(_pc, "_WHEEL_SIMPLE_SCRIPT"): _WHEEL_SIMPLE_SCRIPT = _pc._WHEEL_SIMPLE_SCRIPT
if hasattr(_pc, "_XM430_SCRIPT"): _XM430_SCRIPT = _pc._XM430_SCRIPT
if hasattr(_pc, "_XT60_CONNECTOR_SCRIPT"): _XT60_CONNECTOR_SCRIPT = _pc._XT60_CONNECTOR_SCRIPT
if hasattr(_pc, "body"): body = _pc.body
if hasattr(_pc, "bore"): bore = _pc.bore
if hasattr(_pc, "cut"): cut = _pc.cut
if hasattr(_pc, "ends"): ends = _pc.ends
if hasattr(_pc, "flanges"): flanges = _pc.flanges
if hasattr(_pc, "fuse"): fuse = _pc.fuse
if hasattr(_pc, "holes"): holes = _pc.holes
if hasattr(_pc, "loop"): loop = _pc.loop
if hasattr(_pc, "pattern"): pattern = _pc.pattern
if hasattr(_pc, "ratio"): ratio = _pc.ratio
if hasattr(_pc, "recompute"): recompute = _pc.recompute
if hasattr(_pc, "slot"): slot = _pc.slot
if hasattr(_pc, "spacing"): spacing = _pc.spacing
if hasattr(_pc, "translate"): translate = _pc.translate

ENTRIES: dict[str, PartTemplate] = {
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

    "bearing_626": PartTemplate(
        id="bearing_626",
        name_en="626 Deep Groove Ball Bearing",
        name_cn="626 深沟球轴承",
        category="bearing",
        subcategory="ball_bearing",
        description="626 系列深沟球轴承，6mm内径，小型电机/传动常用",
        tags=["轴承", "626", "深沟球", "bearing", "small motor"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="626-2RS",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 6, 1, 100, 1),
            ParamDef("outer_diameter", "外径", "mm", 19, 5, 200, 1),
            ParamDef("width", "宽度", "mm", 6, 1, 50, 1),
            ParamDef("bearing_detail", "轴承细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("ball_count", "滚珠数量", "", 0, 0, 50, 1),
        ],
        fc_script_template=_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 6, "outer_diameter": 19, "width": 6},
        ],
    ),

    "bearing_688": PartTemplate(
        id="bearing_688",
        name_en="688 Deep Groove Ball Bearing",
        name_cn="688 深沟球轴承",
        category="bearing",
        subcategory="ball_bearing",
        description="688 系列微型深沟球轴承，8mm内径薄型，小型滑轮/微型电机常用",
        tags=["轴承", "688", "深沟球", "bearing", "miniature", "thin section"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="688-2RS",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 8, 1, 100, 1),
            ParamDef("outer_diameter", "外径", "mm", 16, 5, 200, 1),
            ParamDef("width", "宽度", "mm", 4, 1, 50, 1),
            ParamDef("bearing_detail", "轴承细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
            ParamDef("ball_count", "滚珠数量", "", 0, 0, 50, 1),
        ],
        fc_script_template=_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 8, "outer_diameter": 16, "width": 4},
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
        part_class="structural", scalable=True,
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
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="Flexible-Coupling",
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
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="Spur-Gear",
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
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="Generic-Wheel",
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
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="Generic-Mecanum",
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
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="Generic-Hub",
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
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="Hex-Standoff",
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
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various", model_number="18650-Holder",
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

    "linear_bearing_lm6uu": PartTemplate(
        id="linear_bearing_lm6uu",
        name_en="LM6UU Linear Ball Bearing",
        name_cn="LM6UU 直线球轴承",
        category="bearing",
        subcategory="linear_bearing",
        description="LM6UU 直线运动球轴承，用于 6mm 光轴上的直线往复运动，微型3D打印机/CNC常用",
        tags=["直线轴承", "LM6UU", "linear bearing", "ball bearing", "直线运动", "3D printer"],
        part_class="functional", scalable=False, real_part=True,
        manufacturer="Various (MISUMI/THK clone)", model_number="LM6UU",
        parameters=[
            ParamDef("inner_diameter", "内径", "mm", 6.0, 4, 60, 0.1, fixed=True),
            ParamDef("outer_diameter", "外径", "mm", 12.0, 8, 62, 0.1, fixed=True),
            ParamDef("length", "长度", "mm", 19.0, 10, 80, 0.5, fixed=True),
            ParamDef("ball_count", "滚珠数", "", 0, 0, 50, 1, fixed=True),
            ParamDef("bearing_detail", "细节", param_type="string",
                     choices=["simplified", "realistic"], default="simplified"),
        ],
        fc_script_template=_LINEAR_BEARING_SCRIPT,
        fc_script_alternatives={"realistic": _LINEAR_BEARING_REALISTIC_SCRIPT},
        quality_levels=["simplified", "realistic"],
        standard_sizes=[
            {"inner_diameter": 6.0, "outer_diameter": 12.0, "length": 19.0},
        ],
        notes="LM6UU用于6mm光轴。最小型直线轴承，适合紧凑型机构。",
    ),

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

    # ===================================================================
    # Layer 3 Phase 1: 15 new structural templates
    # ===================================================================

    "aluminum_extrusion": PartTemplate(
        id="aluminum_extrusion",
        name_en="Aluminum Extrusion",
        name_cn="铝型材",
        category="structural",
        subcategory="bracket",
        description="参数化铝型材方管，含中心孔和T型槽",
        tags=["铝型材", "extrusion", "profile", "structural", "frame"],
        parameters=[
            ParamDef("profile_size", "截面尺寸", "mm", 20, 10, 80, 1),
            ParamDef("length", "长度", "mm", 300, 50, 2000, 1),
            ParamDef("bore_size", "中心孔径", "mm", 4, 0, 30, 0.5),
            ParamDef("groove_w", "T槽宽度", "mm", 5, 0, 15, 0.5),
            ParamDef("groove_d", "T槽深度", "mm", 2, 0, 10, 0.5),
        ],
        fc_script_template=_ALUMINUM_EXTRUSION_SCRIPT,
        standard_sizes=[
            {"profile_size": 20, "length": 300, "bore_size": 4, "groove_w": 5, "groove_d": 2},
            {"profile_size": 30, "length": 500, "bore_size": 6, "groove_w": 6, "groove_d": 3},
            {"profile_size": 40, "length": 500, "bore_size": 8, "groove_w": 8, "groove_d": 4},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="end",
            holes=[
                BoltHole(x=0, y=0, diameter=4.0),
            ],
            bore_diameter=4.0,
        ),
    ),

    "u_bracket": PartTemplate(
        id="u_bracket",
        name_en="U Bracket",
        name_cn="U型支架",
        category="structural",
        subcategory="bracket",
        description="参数化U型支架，底面带安装孔",
        tags=["支架", "U型", "bracket", "structural", "mount"],
        parameters=[
            ParamDef("width", "宽度", "mm", 30, 15, 100, 1),
            ParamDef("height", "高度", "mm", 40, 15, 120, 1),
            ParamDef("thickness", "壁厚", "mm", 3, 1, 10, 0.5),
            ParamDef("leg_length", "腿长", "mm", 30, 10, 80, 1),
            ParamDef("hole_d", "安装孔径", "mm", 4, 2, 10, 0.5),
        ],
        fc_script_template=_U_BRACKET_SCRIPT,
        standard_sizes=[
            {"width": 30, "height": 40, "thickness": 3, "leg_length": 30, "hole_d": 4},
            {"width": 40, "height": 50, "thickness": 4, "leg_length": 40, "hole_d": 5},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=8, y=8, diameter=4.0),
                BoltHole(x=22, y=8, diameter=4.0),
                BoltHole(x=8, y=22, diameter=4.0),
                BoltHole(x=22, y=22, diameter=4.0),
            ],
        ),
    ),

    "t_bracket": PartTemplate(
        id="t_bracket",
        name_en="T Bracket",
        name_cn="T型支架",
        category="structural",
        subcategory="bracket",
        description="参数化T型支架，横板和竖干均有安装孔",
        tags=["支架", "T型", "bracket", "structural", "mount"],
        parameters=[
            ParamDef("plate_w", "横板宽", "mm", 60, 20, 150, 1),
            ParamDef("plate_h", "横板高", "mm", 30, 15, 80, 1),
            ParamDef("stem_l", "竖干长", "mm", 40, 15, 100, 1),
            ParamDef("thickness", "板厚", "mm", 3, 1, 10, 0.5),
            ParamDef("hole_d", "孔径", "mm", 4, 2, 10, 0.5),
        ],
        fc_script_template=_T_BRACKET_SCRIPT,
        standard_sizes=[
            {"plate_w": 60, "plate_h": 30, "stem_l": 40, "thickness": 3, "hole_d": 4},
            {"plate_w": 80, "plate_h": 40, "stem_l": 50, "thickness": 4, "hole_d": 5},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=30, y=15, diameter=4.0),
                BoltHole(x=30, y=15, diameter=4.0),
            ],
        ),
    ),

    "gusset_plate": PartTemplate(
        id="gusset_plate",
        name_en="Gusset Plate",
        name_cn="加强筋板",
        category="structural",
        subcategory="bracket",
        description="参数化三角形加强筋板",
        tags=["加强筋", "gusset", "triangular", "structural", "reinforcement"],
        parameters=[
            ParamDef("side_a", "直角边A", "mm", 40, 15, 100, 1),
            ParamDef("side_b", "直角边B", "mm", 40, 15, 100, 1),
            ParamDef("thickness", "厚度", "mm", 3, 1, 10, 0.5),
            ParamDef("hole_d", "安装孔径", "mm", 4, 2, 10, 0.5),
        ],
        fc_script_template=_GUSSET_PLATE_SCRIPT,
        standard_sizes=[
            {"side_a": 40, "side_b": 40, "thickness": 3, "hole_d": 4},
            {"side_a": 60, "side_b": 40, "thickness": 4, "hole_d": 5},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=10, y=10, diameter=4.0),
            ],
        ),
    ),

    "bearing_block": PartTemplate(
        id="bearing_block",
        name_en="Bearing Block",
        name_cn="轴承座",
        category="structural",
        subcategory="bracket",
        description="参数化轴承座，底板+侧壁+轴孔",
        tags=["轴承座", "bearing", "block", "pillow", "structural"],
        parameters=[
            ParamDef("shaft_d", "轴径", "mm", 10, 3, 30, 0.5),
            ParamDef("block_h", "座高", "mm", 30, 15, 60, 1),
            ParamDef("base_l", "底板长", "mm", 40, 20, 80, 1),
            ParamDef("base_w", "底板宽", "mm", 30, 15, 60, 1),
            ParamDef("bolt_d", "螺栓孔径", "mm", 4, 2, 8, 0.5),
        ],
        fc_script_template=_BEARING_BLOCK_SCRIPT,
        standard_sizes=[
            {"shaft_d": 10, "block_h": 30, "base_l": 40, "base_w": 30, "bolt_d": 4},
            {"shaft_d": 15, "block_h": 40, "base_l": 50, "base_w": 40, "bolt_d": 5},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=6, y=6, diameter=4.0),
                BoltHole(x=34, y=6, diameter=4.0),
                BoltHole(x=6, y=24, diameter=4.0),
                BoltHole(x=34, y=24, diameter=4.0),
            ],
            bore_diameter=10.0,
        ),
    ),

    "servo_bracket": PartTemplate(
        id="servo_bracket",
        name_en="Servo Bracket",
        name_cn="舵机支架",
        category="structural",
        subcategory="bracket",
        description="参数化U型舵机支架，适配SG90安装孔",
        tags=["舵机", "支架", "servo", "bracket", "SG90", "structural"],
        parameters=[
            ParamDef("servo_type", "舵机型号", "", "SG90", "SG90", "SG90", 1,
                     param_type="string", choices=["SG90", "MG996R", "DS3218"]),
            ParamDef("plate_l", "底板长", "mm", 40, 20, 80, 1),
            ParamDef("plate_w", "底板宽", "mm", 20, 10, 40, 1),
            ParamDef("plate_t", "板厚", "mm", 2, 1, 6, 0.5),
            ParamDef("flange_h", "法兰高", "mm", 25, 10, 50, 1),
        ],
        fc_script_template=_SERVO_BRACKET_SCRIPT,
        standard_sizes=[
            {"servo_type": "SG90", "plate_l": 40, "plate_w": 20, "plate_t": 2, "flange_h": 25},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=6, y=10, diameter=1.5),
                BoltHole(x=34, y=10, diameter=1.5),
            ],
        ),
    ),

    "nema_mount": PartTemplate(
        id="nema_mount",
        name_en="NEMA Mount Plate",
        name_cn="步进电机安装板",
        category="structural",
        subcategory="bracket",
        description="参数化NEMA步进电机安装板，自动匹配孔位",
        tags=["NEMA", "步进电机", "安装板", "stepper", "mount", "structural"],
        parameters=[
            ParamDef("motor_type", "电机型号", "", "NEMA17", "NEMA14", "NEMA23", 1,
                     param_type="string", choices=["NEMA14", "NEMA17", "NEMA23"]),
            ParamDef("plate_l", "板长", "mm", 50, 25, 100, 1),
            ParamDef("plate_w", "板宽", "mm", 50, 25, 100, 1),
            ParamDef("plate_t", "板厚", "mm", 3, 1, 10, 0.5),
        ],
        fc_script_template=_NEMA_MOUNT_SCRIPT,
        standard_sizes=[
            {"motor_type": "NEMA17", "plate_l": 50, "plate_w": 50, "plate_t": 3},
            {"motor_type": "NEMA23", "plate_l": 70, "plate_w": 70, "plate_t": 5},
            {"motor_type": "NEMA14", "plate_l": 40, "plate_w": 40, "plate_t": 3},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="front",
            holes=[
                BoltHole(x=9.5, y=9.5, diameter=3.4),
                BoltHole(x=40.5, y=9.5, diameter=3.4),
                BoltHole(x=9.5, y=40.5, diameter=3.4),
                BoltHole(x=40.5, y=40.5, diameter=3.4),
            ],
            bore_diameter=22.0,
        ),
    ),

    "standoff_column": PartTemplate(
        id="standoff_column",
        name_en="Standoff Column",
        name_cn="支撑柱",
        category="structural",
        subcategory="bracket",
        description="参数化圆柱支撑柱，含通孔",
        tags=["支撑柱", "standoff", "spacer", "pillar", "structural"],
        parameters=[
            ParamDef("od", "外径", "mm", 6, 3, 20, 0.5),
            ParamDef("length", "长度", "mm", 25, 5, 100, 1),
            ParamDef("hole_d", "通孔径", "mm", 3, 1, 12, 0.5),
        ],
        fc_script_template=_STANDOFF_COLUMN_SCRIPT,
        standard_sizes=[
            {"od": 6, "length": 10, "hole_d": 3},
            {"od": 6, "length": 25, "hole_d": 3},
            {"od": 8, "length": 40, "hole_d": 4},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="top",
            holes=[
                BoltHole(x=0, y=0, diameter=3.0),
            ],
            bore_diameter=3.0,
        ),
    ),

    "cable_chain_mount": PartTemplate(
        id="cable_chain_mount",
        name_en="Cable Chain Mount",
        name_cn="线缆链支架",
        category="structural",
        subcategory="bracket",
        description="参数化L型线缆链固定支架，含线槽",
        tags=["线缆", "支架", "cable", "chain", "drag", "structural"],
        parameters=[
            ParamDef("base_l", "底板长", "mm", 40, 20, 100, 1),
            ParamDef("base_w", "底板宽", "mm", 25, 10, 60, 1),
            ParamDef("slot_w", "线槽宽", "mm", 12, 5, 30, 1),
            ParamDef("slot_h", "线槽高", "mm", 15, 5, 40, 1),
            ParamDef("thickness", "壁厚", "mm", 3, 1, 8, 0.5),
        ],
        fc_script_template=_CABLE_CHAIN_MOUNT_SCRIPT,
        standard_sizes=[
            {"base_l": 40, "base_w": 25, "slot_w": 12, "slot_h": 15, "thickness": 3},
            {"base_l": 60, "base_w": 35, "slot_w": 18, "slot_h": 20, "thickness": 4},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=6, y=12.5, diameter=1.5),
                BoltHole(x=34, y=12.5, diameter=1.5),
            ],
        ),
    ),

    "battery_tray": PartTemplate(
        id="battery_tray",
        name_en="Battery Tray",
        name_cn="电池托盘",
        category="structural",
        subcategory="bracket",
        description="参数化电池托盘，开口盒+绑带槽",
        tags=["电池", "托盘", "battery", "tray", "structural"],
        parameters=[
            ParamDef("length", "长度", "mm", 80, 30, 200, 1),
            ParamDef("width", "宽度", "mm", 50, 20, 100, 1),
            ParamDef("height", "高度", "mm", 25, 10, 60, 1),
            ParamDef("wall_t", "壁厚", "mm", 2, 1, 6, 0.5),
        ],
        fc_script_template=_BATTERY_TRAY_SCRIPT,
        standard_sizes=[
            {"length": 80, "width": 50, "height": 25, "wall_t": 2},
            {"length": 120, "width": 60, "height": 30, "wall_t": 3},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=5, y=5, diameter=2.0),
                BoltHole(x=75, y=5, diameter=2.0),
                BoltHole(x=5, y=45, diameter=2.0),
                BoltHole(x=75, y=45, diameter=2.0),
            ],
        ),
    ),

    "sensor_shelf": PartTemplate(
        id="sensor_shelf",
        name_en="Sensor Shelf",
        name_cn="传感器安装架",
        category="structural",
        subcategory="bracket",
        description="参数化斜面传感器安装架",
        tags=["传感器", "安装架", "sensor", "shelf", "angled", "structural"],
        parameters=[
            ParamDef("base_l", "底板长", "mm", 50, 20, 100, 1),
            ParamDef("base_w", "底板宽", "mm", 30, 15, 60, 1),
            ParamDef("shelf_angle", "斜面角度", "deg", 30, 10, 60, 1),
            ParamDef("thickness", "板厚", "mm", 3, 1, 8, 0.5),
        ],
        fc_script_template=_SENSOR_SHELF_SCRIPT,
        standard_sizes=[
            {"base_l": 50, "base_w": 30, "shelf_angle": 30, "thickness": 3},
            {"base_l": 60, "base_w": 40, "shelf_angle": 45, "thickness": 4},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=8, y=15, diameter=2.0),
                BoltHole(x=42, y=15, diameter=2.0),
            ],
        ),
    ),

    "shaft_coupling_block": PartTemplate(
        id="shaft_coupling_block",
        name_en="Shaft Coupling Block",
        name_cn="轴连接块",
        category="structural",
        subcategory="bracket",
        description="参数化轴连接块，含轴孔和紧定螺钉孔",
        tags=["轴", "连接", "coupling", "shaft", "structural"],
        parameters=[
            ParamDef("block_l", "块长", "mm", 30, 10, 60, 1),
            ParamDef("block_w", "块宽", "mm", 20, 10, 40, 1),
            ParamDef("block_h", "块高", "mm", 20, 10, 40, 1),
            ParamDef("bore_d", "轴孔径", "mm", 6, 2, 20, 0.5),
            ParamDef("set_screw_d", "紧定孔径", "mm", 3, 0, 8, 0.5),
        ],
        fc_script_template=_SHAFT_COUPLING_BLOCK_SCRIPT,
        standard_sizes=[
            {"block_l": 30, "block_w": 20, "block_h": 20, "bore_d": 6, "set_screw_d": 3},
            {"block_l": 40, "block_w": 25, "block_h": 25, "bore_d": 8, "set_screw_d": 4},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="side",
            holes=[
                BoltHole(x=10, y=10, diameter=6.0),
            ],
            bore_diameter=6.0,
        ),
    ),

    "guide_rail_carriage": PartTemplate(
        id="guide_rail_carriage",
        name_en="Guide Rail Carriage",
        name_cn="导轨滑块板",
        category="structural",
        subcategory="bracket",
        description="参数化导轨滑块安装板",
        tags=["导轨", "滑块", "rail", "carriage", "linear", "structural"],
        parameters=[
            ParamDef("rail_type", "导轨型号", "", "MGN12", "MGN9", "MGN15", 1,
                     param_type="string", choices=["MGN9", "MGN12", "MGN15"]),
            ParamDef("plate_l", "板长", "mm", 50, 20, 100, 1),
            ParamDef("plate_w", "板宽", "mm", 30, 15, 60, 1),
            ParamDef("plate_t", "板厚", "mm", 5, 2, 12, 0.5),
        ],
        fc_script_template=_GUIDE_RAIL_CARRIAGE_SCRIPT,
        standard_sizes=[
            {"rail_type": "MGN12", "plate_l": 50, "plate_w": 30, "plate_t": 5},
            {"rail_type": "MGN15", "plate_l": 60, "plate_w": 40, "plate_t": 6},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="top",
            holes=[
                BoltHole(x=7.5, y=7.5, diameter=1.5),
                BoltHole(x=42.5, y=7.5, diameter=1.5),
                BoltHole(x=7.5, y=22.5, diameter=1.5),
                BoltHole(x=42.5, y=22.5, diameter=1.5),
            ],
        ),
    ),

    "pulley_idler_mount": PartTemplate(
        id="pulley_idler_mount",
        name_en="Pulley Idler Mount",
        name_cn="惰轮支架",
        category="structural",
        subcategory="bracket",
        description="参数化L型惰轮支架，含轴承安装孔",
        tags=["惰轮", "支架", "pulley", "idler", "bearing", "structural"],
        parameters=[
            ParamDef("bearing_od", "轴承外径", "mm", 16, 8, 30, 0.5),
            ParamDef("plate_l", "底板长", "mm", 40, 20, 80, 1),
            ParamDef("plate_w", "底板宽", "mm", 25, 10, 50, 1),
            ParamDef("plate_t", "板厚", "mm", 3, 1, 8, 0.5),
        ],
        fc_script_template=_PULLEY_IDLER_MOUNT_SCRIPT,
        standard_sizes=[
            {"bearing_od": 16, "plate_l": 40, "plate_w": 25, "plate_t": 3},
            {"bearing_od": 22, "plate_l": 50, "plate_w": 30, "plate_t": 4},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=6, y=12.5, diameter=1.5),
                BoltHole(x=34, y=12.5, diameter=1.5),
            ],
        ),
    ),

    "encoder_mount": PartTemplate(
        id="encoder_mount",
        name_en="Encoder Mount",
        name_cn="编码器安装架",
        category="structural",
        subcategory="bracket",
        description="参数化编码器安装架，含中心孔和固定孔",
        tags=["编码器", "安装架", "encoder", "mount", "structural"],
        parameters=[
            ParamDef("bore_d", "中心孔径", "mm", 6, 2, 15, 0.5),
            ParamDef("base_l", "底板长", "mm", 30, 15, 60, 1),
            ParamDef("base_w", "底板宽", "mm", 20, 10, 40, 1),
            ParamDef("thickness", "板厚", "mm", 2, 1, 6, 0.5),
        ],
        fc_script_template=_ENCODER_MOUNT_SCRIPT,
        standard_sizes=[
            {"bore_d": 6, "base_l": 30, "base_w": 20, "thickness": 2},
            {"bore_d": 8, "base_l": 35, "base_w": 25, "thickness": 3},
        ],
        part_class="structural",
        scalable=True,
        mounting_interface=MountingInterface(
            interface_type="through_hole",
            contact_face="bottom",
            holes=[
                BoltHole(x=6, y=10, diameter=1.3),
                BoltHole(x=24, y=10, diameter=1.3),
            ],
            bore_diameter=6.0,
        ),
    ),

    # ---- Fastener subcategories: inserts and pins ----

    "heat_set_insert": PartTemplate(
        id="heat_set_insert",
        name_en="Heat-Set Thread Insert",
        name_cn="热熔铜螺母",
        category="fastener",
        subcategory="insert",
        description="热压安装铜螺母，用于3D打印件螺纹加固，带滚花外表面",
        tags=["热熔", "铜螺母", "heat-set", "insert", "brass", "knurled", "3D打印"],
        material_default="brass",
        part_class="fastener", scalable=False,
        parameters=[
            ParamDef("thread_diameter", "内螺纹直径", "mm", 3.0, 2.0, 6.0, 0.5, fixed=True),
            ParamDef("outer_diameter", "外径(滚花)", "mm", 4.6, 3.0, 10.0, 0.1, fixed=True),
            ParamDef("length", "长度", "mm", 5.6, 3.0, 10.0, 0.1, fixed=True),
            ParamDef("knurl_pitch", "滚花间距", "mm", 0.8, 0.4, 1.5, 0.1, fixed=True),
        ],
        fc_script_template=_HEAT_SET_INSERT_SCRIPT,
        standard_sizes=[
            {"thread_diameter": 2.0, "outer_diameter": 3.5, "length": 4.0, "knurl_pitch": 0.5},
            {"thread_diameter": 2.5, "outer_diameter": 4.2, "length": 5.0, "knurl_pitch": 0.6},
            {"thread_diameter": 3.0, "outer_diameter": 4.6, "length": 5.6, "knurl_pitch": 0.7},
            {"thread_diameter": 4.0, "outer_diameter": 6.0, "length": 7.0, "knurl_pitch": 0.8},
            {"thread_diameter": 5.0, "outer_diameter": 7.1, "length": 8.0, "knurl_pitch": 0.9},
            {"thread_diameter": 6.0, "outer_diameter": 8.3, "length": 10.0, "knurl_pitch": 1.0},
        ],
        notes="热压安装：使用电烙铁加热压入3D打印件预留孔。常见品牌：Ruthex, CNC Kitchen。"
              "安装孔径 = 外径 - 0.1~0.2mm（过盈配合）。",
    ),

    "shaft_collar": PartTemplate(
        id="shaft_collar",
        name_en="Shaft Collar",
        name_cn="轴环",
        category="shaft",
        subcategory="collar",
        description="轴用固定环，含紧定螺钉孔，用于轴上零件轴向定位",
        tags=["轴环", "轴固定", "shaft", "collar", "set screw", "定位"],
        material_default="steel",
        part_class="fastener", scalable=False,
        parameters=[
            ParamDef("bore_diameter", "内孔径", "mm", 6.0, 3.0, 12.0, 0.5, fixed=True),
            ParamDef("outer_diameter", "外径", "mm", 12.0, 8.0, 22.0, 0.5, fixed=True),
            ParamDef("width", "宽度", "mm", 6.0, 3.0, 12.0, 0.5, fixed=True),
            ParamDef("set_screw_diameter", "紧定螺钉径", "mm", 3.0, 2.0, 5.0, 0.5, fixed=True),
        ],
        fc_script_template=_SHAFT_COLLAR_SCRIPT,
        standard_sizes=[
            {"bore_diameter": 5.0, "outer_diameter": 10.0, "width": 5.0, "set_screw_diameter": 2.5},
            {"bore_diameter": 6.0, "outer_diameter": 12.0, "width": 6.0, "set_screw_diameter": 3.0},
            {"bore_diameter": 8.0, "outer_diameter": 16.0, "width": 8.0, "set_screw_diameter": 4.0},
            {"bore_diameter": 10.0, "outer_diameter": 18.0, "width": 8.0, "set_screw_diameter": 4.0},
            {"bore_diameter": 12.0, "outer_diameter": 22.0, "width": 10.0, "set_screw_diameter": 5.0},
        ],
        notes="分为一体式和分体式两种。紧定螺钉锁紧轴面，提供轴向定位力。"
              "推荐配合定位螺丝使用，避免损伤轴面。",
    ),

    "t_nut": PartTemplate(
        id="t_nut",
        name_en="T-Nut for Aluminum Extrusion",
        name_cn="T型螺母（铝型材用）",
        category="fastener",
        subcategory="insert",
        description="铝型材T型槽用螺母，T形截面嵌入型材槽内",
        tags=["T型螺母", "铝型材", "T-nut", "extrusion", "2020", "3030", "4040"],
        material_default="steel",
        part_class="fastener", scalable=False,
        parameters=[
            ParamDef("thread_diameter", "螺纹直径", "mm", 3.0, 2.0, 8.0, 0.5, fixed=True),
            ParamDef("plate_width", "T头宽度", "mm", 6.0, 4.0, 12.0, 0.5, fixed=True),
            ParamDef("plate_height", "T头高度", "mm", 3.0, 1.5, 6.0, 0.5, fixed=True),
            ParamDef("flange_width", "法兰宽度", "mm", 8.0, 5.0, 16.0, 0.5, fixed=True),
            ParamDef("flange_height", "法兰高度", "mm", 1.5, 0.5, 3.0, 0.5, fixed=True),
            ParamDef("nut_length", "螺母长度", "mm", 12.0, 6.0, 25.0, 1.0, fixed=True),
        ],
        fc_script_template=_T_NUT_SCRIPT,
        standard_sizes=[
            {"thread_diameter": 3.0, "plate_width": 5.5, "plate_height": 2.5,
             "flange_width": 7.5, "flange_height": 1.5, "nut_length": 10},
            {"thread_diameter": 4.0, "plate_width": 6.0, "plate_height": 3.0,
             "flange_width": 8.0, "flange_height": 1.5, "nut_length": 12},
            {"thread_diameter": 5.0, "plate_width": 8.0, "plate_height": 3.5,
             "flange_width": 11.0, "flange_height": 2.0, "nut_length": 15},
            {"thread_diameter": 6.0, "plate_width": 10.0, "plate_height": 4.0,
             "flange_width": 14.0, "flange_height": 2.5, "nut_length": 18},
        ],
        notes="2020型材槽宽6.2mm，3030型材槽宽8.2mm，4040型材槽宽10.2mm。"
              "滑入式安装，配合螺栓固定其他零件到型材上。",
    ),

    "dowel_pin": PartTemplate(
        id="dowel_pin",
        name_en="Dowel Pin",
        name_cn="定位销",
        category="fastener",
        subcategory="pin",
        description="ISO 8734定位销，两端倒角，用于两零件精确定位",
        tags=["定位销", "销钉", "dowel", "pin", "定位", "ISO8734"],
        material_default="steel",
        part_class="fastener", scalable=False,
        parameters=[
            ParamDef("diameter", "直径", "mm", 5.0, 3.0, 10.0, 0.5, fixed=True),
            ParamDef("length", "长度", "mm", 25.0, 8.0, 50.0, 1.0, fixed=True),
            ParamDef("chamfer", "倒角", "mm", 0.5, 0.2, 1.5, 0.1, fixed=True),
        ],
        fc_script_template=_DOWEL_PIN_SCRIPT,
        standard_sizes=[
            {"diameter": 3.0, "length": 10.0, "chamfer": 0.3},
            {"diameter": 3.0, "length": 16.0, "chamfer": 0.3},
            {"diameter": 3.0, "length": 20.0, "chamfer": 0.3},
            {"diameter": 4.0, "length": 16.0, "chamfer": 0.4},
            {"diameter": 4.0, "length": 20.0, "chamfer": 0.4},
            {"diameter": 4.0, "length": 25.0, "chamfer": 0.4},
            {"diameter": 5.0, "length": 20.0, "chamfer": 0.5},
            {"diameter": 5.0, "length": 25.0, "chamfer": 0.5},
            {"diameter": 5.0, "length": 30.0, "chamfer": 0.5},
            {"diameter": 6.0, "length": 25.0, "chamfer": 0.5},
            {"diameter": 6.0, "length": 30.0, "chamfer": 0.5},
            {"diameter": 6.0, "length": 40.0, "chamfer": 0.5},
            {"diameter": 8.0, "length": 30.0, "chamfer": 0.8},
            {"diameter": 8.0, "length": 40.0, "chamfer": 0.8},
            {"diameter": 10.0, "length": 40.0, "chamfer": 1.0},
        ],
        notes="ISO 8734 m6精度定位销。一端间隙配合(H7)，另一端过盈配合。"
              "安装时轻敲压入过盈端。定位精度可达0.01mm。",
    ),
}
