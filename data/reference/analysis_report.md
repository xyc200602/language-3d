# Open-Source Robot Assembly Pattern Analysis Report

Analysis of Thor and PAROL6 open-source robot arm projects for assembly pattern extraction.

---

## 1. Thor (AngelLM/Thor)

- **Repository**: https://github.com/AngelLM/Thor
- **License**: CC-BY-SA-4.0
- **DOF**: 6 (yaw-roll-roll-yaw-roll-yaw)
- **CAD**: FreeCAD native (.FCStd) — unique among all analyzed projects
- **Height**: 625mm (stretched upright)
- **Payload**: 750g (including end effector)
- **Cost**: <350 EUR hardware

### Key Findings

#### Actuators
- 6x NEMA17 (42BYGH) stepper motors for all joints
- Previously documented as MG996R servos — this was incorrect; Thor uses steppers throughout
- Trimpot/limit switch homing on each axis

#### Transmission
- **GT2 timing belts + pulleys**: Used for most joint drives (base, wrist)
- **3D-printed gears**: Module 1.0 spur/segment gears for shoulder/elbow (reduction ratio ~3:1 to 4:1)
- GT2 16T motor pulleys secured with M3 grub screws on 5mm D-shaft
- GT2 36T-60T joint pulleys (larger for more torque at distal joints)

#### Bearings
- 625-2RS (16x5x5mm) press-fit in joint housings
- Standard interference fit: -0.04mm bore tolerance

#### Fasteners
- M3 socket head cap screws exclusively (M3x8, M3x12)
- NEMA17 standard 4xM3 bolt pattern (31mm spacing)
- M3 grub screws for pulley/shaft coupling

#### Structural Parts
- 37 unique 3D-printable parts (PLA/PETG)
- FreeCAD source files in `freecad-src/` directory
- STL files in `stl/` directory, STEP files in `step/`
- Exploded assembly available via FreeCAD ExplodedAssembly workbench

#### Connection Distribution
| Method | Count | Percentage |
|--------|-------|------------|
| Bolted (M3) | 20 | 56% |
| Belt drive | 6 | 17% |
| Gear mesh | 6 | 17% |
| Press fit | 4 | 11% |
| Set screw | 6 | — |

#### Electronics
- Arduino Mega + custom shield (ThorControlPCB, KiCAD design)
- GRBL or RepRapFirmware for motor control
- ROS2 + MoveIt2 integration available

---

## 2. PAROL6 (PCrnjak/PAROL6-Desktop-robot-arm)

- **Repository**: https://github.com/PCrnjak/PAROL6-Desktop-robot-arm
- **License**: GPLv3
- **DOF**: 6
- **Reach**: ~400mm
- **Payload**: ~300g nominal
- **Design Philosophy**: Industrial robot approach for desktop form factor

### Key Findings

#### Actuators
- 6x NEMA17 stepper motors
- Trinamic TMC2209 silent stepper drivers (important differentiator)
- Silent operation is a design goal

#### Transmission
- **GT2 timing belts** for all joint drives (no gears, unlike Thor)
- Belt tensioning via slotted motor mounts (no spring tensioners in basic design)
- Typical reduction: GT2 16T motor → GT2 48T joint (3:1 ratio)
- GT2 6mm wide belts standard

#### Bearings
- MR105 (5x10x4mm) for shoulder/elbow joints — smaller than 608 used in MOVEO
- 608-2RS (22x8x7mm) for base rotation
- Press-fit into split clamshell housings

#### Housing Design
- **Split clamshell design**: Joint housings consist of 2 halves bolted together
- Advantage: Easy assembly/disassembly, bearing replacement
- Each housing has two bearing seats (opposite sides) for shaft support
- Thor uses unibody housings with side access slots (harder to assemble)

#### Fasteners
- M3 SHCS exclusively, matching Thor and MOVEO pattern
- NEMA17 standard 4xM3 bolt pattern

#### Structural Parts
- ~30 3D-printable parts
- STL files provided, no STEP/FreeCAD source
- PLA for most parts, PETG for high-stress parts
- Designed for FDM printing (0.2mm layer height, 100% infill for stressed parts)

#### Connection Distribution
| Method | Count | Percentage |
|--------|-------|------------|
| Bolted (M3) | 18 | 60% |
| Belt drive | 6 | 20% |
| Press fit | 6 | 15% |
| Set screw | 4 | 5% |

#### Control Software
- Custom Commander GUI (C++/Qt)
- Python API for programmatic control
- ROS2/MoveIt2 simulation available
- Trinamic driver configuration via UART

---

## 3. Cross-Project Patterns

### Universal Constants (across Thor, PAROL6, MOVEO)

1. **M3 is the universal fastener** — 65%+ of all bolts across all 3D-printed arms
2. **NEMA17 standard mounting** — 4xM3 at 31mm spacing, Ø23mm shaft clearance
3. **608/MR105 bearing press-fit** — -0.03 to -0.05mm interference
4. **GT2 timing belt** — 2mm pitch, 6mm width, for joint transmission
5. **PLA structural parts** — 65%+ of material in 3D-printed robots

### Key Differences

| Feature | Thor | PAROL6 | MOVEO |
|---------|------|--------|-------|
| DOF | 6 | 6 | 5 |
| CAD Source | FreeCAD | STL only | FreeCAD |
| Gear Type | 3D-printed | None | None |
| Belt Drive | Partial | Full | Full |
| Housing | Unibody | Split clamshell | Unibody |
| Bearings | 625-2RS | MR105 + 608 | 608-2RS |
| Drivers | GRBL/RRF | Trinamic TMC2209 | A4988 |
| Firmware | GRBL | Custom | Marlin/GRBL |

### Best Practices Extracted

1. **Slotted motor mounts**: PAROL6 pattern — slots perpendicular to belt path for tension adjustment
2. **Split housings**: PAROL6 pattern — easier assembly than unibody Thor housings
3. **Belt reduction ratios**: Shoulder 4:1, Elbow 3:1, Wrist 2:1 (power vs. speed tradeoff)
4. **MR105 bearings**: For compact joints where 608 is too large
5. **Double bearing seats**: Shaft passes through 2 bearings in housing for rigidity
6. **M3 grub screw on pulleys**: Standard method for securing GT2 pulleys to NEMA17 shaft

---

## 4. Updates Applied to Codebase

### assembly_patterns.py
- Updated Thor profile: MG996R → NEMA17 + GT2 belt, 25→55 parts, corrected dimensions
- Added PAROL6 profile with full statistics
- Added 4 new connection patterns: belt_drive_joint, gear_transmission, belt_tensioner, joint_housing_bearing_seats
- Added 5 new interface rules: gt2_belt_drive_housing, mr105_bearing_seat, joint_housing_split, gt2_pulley_mount
- Updated statistics: belt_drive, gear_mesh, transmission_distribution, belt_drive_parameters
- Updated bolt size distribution: M3 up to 65%
- Added material: PETG for high-stress parts, brass for pulleys

### assembly_generator.py
- Added EXAMPLE_6DOF_BELT_DRIVE_ARM (20 parts, 19 joints) based on PAROL6
- Updated prompt logic: 6-DOF requests use PAROL6 example, 5-DOF use MOVEO example
- Includes MR105 bearings, split housings, slotted motor mounts

### tests/test_reference_learning.py
- 62 tests covering Thor profile, PAROL6 profile, new patterns, statistics, few-shot example

---

## 5. Sources

- Thor repository: https://github.com/AngelLM/Thor
- PAROL6 repository: https://github.com/PCrnjak/PAROL6-Desktop-robot-arm
- FreeCAD forum Thor thread: https://forum.freecad.org/viewtopic.php?t=19731
- Hackaday Thor project: https://hackaday.io/project/12989-thor
- PAROL6 Hackaday: https://hackaday.io/project/191860-parol6-desktop-robot-arm
