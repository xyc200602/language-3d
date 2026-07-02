# Formal Evaluation Metrics for Language-3D

## Overview

The Language-3D benchmark uses a **7-phase automated scoring pipeline**
with 41–43 binary checks per case. Each check is PASS, FAIL, or WARN.
The overall score is:

```
Score = PASS / (PASS + FAIL + WARN) × 100%
```

SKIP (missing optional dependency) is excluded from the denominator.
Critical checks (collision, COM stability, MuJoCo physics, grasp) FAIL
the case if they fail — they do not downgrade to WARN.

## Phase Definitions and Metrics

### Phase 1: NL → Assembly Generation (5 checks)
| Check | Metric | Threshold | Critical |
|---|---|---|---|
| vlm_loop_completed | VLM verification status | PASSED | ✓ |
| part_count | Number of generated parts | ≥ 6 | ✓ |
| joint_count | Number of generated joints | ≥ 4 | ✓ |
| connected_tree | All parts reachable via joints | True | ✓ |
| has_arms/has_wheels | Robot-type-specific parts exist | True | |

### Phase 2: Position Solving (4 checks)
| Check | Metric | Threshold | Critical |
|---|---|---|---|
| all_parts_positioned | Parts with valid 3D position | = part_count | |
| no_nan_positions | NaN/Inf in position vectors | 0 | ✓ |
| rotation_data | Parts with non-zero rotation | (advisory) | |
| outliers | Parts > 2000mm from centroid | 0 | |

### Phase 3: Render Quality (2 checks)
| Check | Metric | Threshold |
|---|---|---|
| render_count | Number of rendered views | ≥ 3 |
| render_quality | Average render file size | ≥ 10 KB |

### Phase 4: Engineering Package (10 checks)
| Check | Metric | Threshold | Critical |
|---|---|---|---|
| pkg_exists | Export directory exists | True | ✓ |
| pkg_design_report | design_report.json exists | True | ✓ |
| pkg_bom | bom.md exists | True | ✓ |
| pkg_assembly_guide | assembly_guide.md exists | True | ✓ |
| pkg_urdf | urdf.xml exists | True | ✓ |
| pkg_readme | README.md exists | True | ✓ |
| pkg_freecad_scripts | freecad_scripts/ has ≥ N files | True | ✓ |
| pkg_firmware | firmware/ has ≥ N files | True | ✓ |
| pkg_stl_parts | stl_parts/ has ≥ N files | True | ✓ |
| pkg_subsystems | subsystems/ has ≥ 1 file | True | ✓ |

### Phase 5: Content Validation (9 checks)
| Check | Metric | Threshold | Critical |
|---|---|---|---|
| report_mass | Total mass > 0 | > 0 | |
| report_parts_match | Report parts = assembly parts | equal | |
| verification_status | VLM final status | PASSED | ✓ |
| kinematic_analysis | Solver converged | True | |
| urdf_structure | URDF has links + joints | True | |
| urdf_origins_sane | No absurd joint origins | 0 | ✓ |
| script_complexity | Scripts with features | 100% | |
| stl_triangle_count | Total triangles | > 0 | |
| stl_watertight_ratio | Watertight meshes | 100% | |
| gripper_finger_watertight | Gripper fingers watertight | ALL | ✓ |
| step_completeness | STEP files exported | 100% | ✓ |

### Phase 6: Physical Sanity (5 checks)
| Check | Metric | Threshold | Critical |
|---|---|---|---|
| no_severe_collisions | FCL mesh penetration > 5mm | 0 | ✓ |
| motion_collision_sweep | Joints with sweep collisions | 0 | ✓ |
| com_stability | COM in support polygon | True | ✓ |
| parts_reachable | IK-reachable parts | 100% | |
| workspace_nontrivial | Workspace bbox max edge | ≥ 167mm | ✓ |

### Phase 7: MuJoCo Simulation (4 checks)
| Check | Metric | Threshold | Critical |
|---|---|---|---|
| mujoco_loads | URDF loads into MuJoCo | True | ✓ |
| mujoco_physics_stable | PD-hold angle error | < 1° | ✓ |
| mujoco_joints_actuate | Actuated DOFs | ≥ 4 | ✓ |
| sim_grasp | Cube held against gravity | PASS | ✓ |

## Scoring Formula Justification

- **WARN vs FAIL**: Critical checks (geometry, physics, safety) must pass.
  Non-critical checks (render size, mass estimate) can WARN without failing.
- **SKIP exclusion**: If python-fcl is not installed, collision checks SKIP
  rather than FAIL. This prevents penalizing environments missing optional
  dependencies.
- **Per-phase scoring**: For fine-grained analysis, each phase can be scored
  independently (PhaseScore = phase_pass / phase_total).

## Limitations of Current Metrics

1. **Binary checks, no graded quality**: A part is either "watertight" or
   "not" — no partial credit for "almost watertight."
2. **No aesthetic/functional quality**: The VLM checks structural validity
   but not "does this robot look good" or "can it perform task X."
3. **Fixed thresholds**: The 1° physics stability threshold and 5mm
   collision threshold are engineering judgments, not calibrated.
4. **No human evaluation**: No user study comparing generated robots to
   human-designed ones.
