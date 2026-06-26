"""Connection feature engine — generate engineering features from connection methods.

Given a Part + ConnectionMethod (from Joint.connection), this module generates
FreeCAD operation dicts that create physical connection geometry:

- Bolted: clearance holes, counterbores, bolt/nut/washer models
- Press-fit: interference bore with shoulder
- Snap-fit: cantilever snap hooks and matching slots
- Adhesive: bonding surface with gap
- Welded: weld preparation (bevel/butt)
- Magnetic: magnet pocket

This is the bridge between the abstract ConnectionMethod metadata and actual
CAD geometry — the critical missing link identified in the project bottleneck
analysis.

Pure-function module: no FreeCAD imports, no I/O.  Outputs operation dicts
consumable by ``freecad._build_script()`` and ``part_feature_engine.generate_ops()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..knowledge.mechanics import (
    BoltHole,
    ConnectionMethod,
    Joint,
    Part,
    STANDARD_SCREWS,
)
from ..knowledge.fastener_catalog import (
    BOLT_SPECS,
    NUT_SPECS,
    WASHER_SPECS,
    THREAD_INSERT_SPECS,
    CLEARANCE_HOLE_SPECS,
    TAP_HOLE_SPECS,
    SET_SCREW_SPECS,
    TORQUE_PLA,
    TORQUE_STEEL,
    get_clearance_hole as _get_clearance_hole,
    get_tap_hole as _get_tap_hole,
    get_torque as _get_torque,
    get_bolt_spec as _get_bolt_spec,
    get_nut_spec as _get_nut_spec,
    get_washer_spec as _get_washer_spec,
    get_thread_insert_spec as _get_thread_insert_spec,
    get_set_screw_spec as _get_set_screw_spec,
    get_nut_spec as _get_nut_spec,
    recommend_bolt_length as _recommend_bolt_length,
)
from ..knowledge.tolerance import (
    bearing_seat_diameter,
    press_fit_bore_diameter,
    it_tolerance,
    compute_fit,
)


# ============================================================================
# Data model
# ============================================================================


# Common bearing specs: (inner_diameter, outer_diameter, width)
_BEARING_SPECS: dict[str, tuple[float, float, float]] = {
    "608": (8.0, 22.0, 7.0),
    "623": (3.0, 10.0, 4.0),
    "625": (5.0, 16.0, 5.0),
    "624": (4.0, 13.0, 5.0),
    "626": (6.0, 19.0, 6.0),
    "688": (8.0, 16.0, 5.0),
    "689": (9.0, 17.0, 5.0),
    "MR105": (5.0, 10.0, 4.0),
    "MR115": (5.0, 11.0, 4.0),
}


@dataclass
class ConnectionFeatureResult:
    """Result from connection feature generation."""

    ops: list[dict] = field(default_factory=list)
    """FreeCAD operations to add connection features to the structural part."""

    fastener_ops: list[dict] = field(default_factory=list)
    """FreeCAD operations to create fastener models (bolt/nut/washer)."""

    warnings: list[str] = field(default_factory=list)
    """Any warnings about feature generation (e.g., insufficient space)."""

    features_generated: list[str] = field(default_factory=list)
    """Human-readable list of features that were generated."""


# ============================================================================
# Public query helpers
# ============================================================================


def get_clearance_hole(bolt_size: str, fit: str = "normal") -> float:
    """Return clearance hole diameter for a bolt size. Delegates to fastener_catalog."""
    return _get_clearance_hole(bolt_size, fit)


def get_torque_recommendation(bolt_size: str, material: str = "PLA") -> float:
    """Return recommended tightening torque in N·m. Delegates to fastener_catalog."""
    return _get_torque(bolt_size, material)


def get_bolt_head_dims(bolt_size: str) -> tuple[float, float]:
    """Return (head_diameter, head_height) for a socket head cap screw."""
    spec = _get_bolt_spec(bolt_size)
    return (spec.head_diameter, spec.head_height) if spec else (5.5, 3.0)


def get_nut_dims(bolt_size: str) -> tuple[float, float]:
    """Return (width_across_flats, height) for a standard hex nut."""
    spec = _get_nut_spec(bolt_size)
    return (spec.width_across_flats, spec.height) if spec else (5.5, 2.4)


def get_washer_dims(bolt_size: str) -> tuple[float, float, float]:
    """Return (inner_diameter, outer_diameter, thickness) for a flat washer."""
    spec = _get_washer_spec(bolt_size)
    return (spec.inner_diameter, spec.outer_diameter, spec.thickness) if spec else (3.2, 7.0, 0.5)


def get_thread_insert_dims(bolt_size: str) -> tuple[float, float, float]:
    """Return (outer_diameter, length, installation_hole_diameter) for a heat-set insert."""
    spec = _get_thread_insert_spec(bolt_size)
    return (spec.outer_diameter, spec.length, spec.install_hole_diameter) if spec else (4.6, 5.6, 4.7)


def get_bearing_spec(bearing_name: str) -> tuple[float, float, float] | None:
    """Return (inner_diameter, outer_diameter, width) for a standard bearing."""
    return _BEARING_SPECS.get(bearing_name)


def get_clearance_hole_with_tolerance(
    bolt_size: str,
    fit: str = "normal",
) -> tuple[float, float, float]:
    """Return (nominal, min, max) clearance hole diameter with IT12 tolerance.

    The hole diameter uses ISO 273 clearance + IT12 tolerance band.
    This gives a tolerance-aware hole for 3D printed or machined parts.

    Args:
        bolt_size: e.g. "M3"
        fit: "close", "normal", or "loose"

    Returns:
        (nominal_d, min_d, max_d) in mm.
    """
    nominal = _get_clearance_hole(bolt_size, fit)
    # Use IT12 for clearance holes (generous for 3D printing)
    tol = it_tolerance(nominal, "IT12")
    return (nominal, nominal, nominal + tol)


# ============================================================================
# Connection Feature Engine — main entry point
# ============================================================================


class ConnectionFeatureEngine:
    """Generate engineering features for physical connections between parts.

    Usage::

        engine = ConnectionFeatureEngine()
        result = engine.generate_features(
            structural_part=bracket,
            connection=joint.connection,
            anchor="top",  # face where connection happens
        )
        # result.ops → FreeCAD operation dicts for the structural part
        # result.fastener_ops → FreeCAD operation dicts for fastener models
    """

    def generate_features(
        self,
        structural_part: Part,
        connection: ConnectionMethod,
        anchor: str = "top",
        part_thickness: float | None = None,
        functional_part_id: str | None = None,
        bolt_pattern: list[tuple[tuple[float, float, float], float]] | None = None,
    ) -> ConnectionFeatureResult:
        """Generate connection features for a structural part.

        If *functional_part_id* is provided and a MountingInterface is
        registered for that part, the engine will use the interface data
        (exact hole positions, bore diameters, pocket dimensions) instead
        of heuristic-based layout.  This produces more accurate features
        for known catalog parts (NEMA17, servos, bearings, etc.).

        Args:
            structural_part: The part that receives the connection features
                            (holes, bores, slots, etc.).
            connection: The ConnectionMethod describing how parts are joined.
            anchor: Which face the connection is on
                    ("top"/"bottom"/"left"/"right"/"front"/"back").
            part_thickness: Override thickness (mm). If None, inferred from
                            part dimensions.
            functional_part_id: Optional catalog ID of the functional part
                            being mounted (e.g. "nema17_stepper").
            bolt_pattern: SHARED bolt-hole pattern for this joint (list of
                        ((u, v, 0), diameter) in normalized face coords).
                        When provided, BOTH mating parts use this exact
                        pattern instead of independently auto-laying-out
                        holes from their own dimensions — the fix for the
                        "连不上" misalignment defect.  See
                        ``compute_shared_bolt_pattern``.
        """
        # Stash the shared pattern so the handler (e.g. _generate_bolted_features)
        # uses it instead of calling _auto_layout_bolts fresh per part.
        self._shared_bolt_pattern = bolt_pattern
        # --- Task 76: prefer MountingInterface for known parts ---
        if functional_part_id:
            result = self._generate_from_interface(
                structural_part, connection, anchor, part_thickness,
                functional_part_id,
            )
            if result is not None:
                return result

        d = structural_part.dimensions
        thickness = part_thickness or self._infer_thickness(d, anchor)

        dispatch = {
            "bolted": self._generate_bolted_features,
            "press_fit": self._generate_press_fit_features,
            "snap_fit": self._generate_snap_fit_features,
            "adhesive": self._generate_adhesive_features,
            "welded": self._generate_welded_features,
            "magnetic": self._generate_magnetic_features,
            "dowel_pin": self._generate_dowel_pin_features,
            "set_screw": self._generate_set_screw_features,
        }

        handler = dispatch.get(connection.type)
        if handler is None:
            return ConnectionFeatureResult(
                warnings=[f"Unknown connection type: {connection.type}"],
            )

        return handler(structural_part, connection, anchor, thickness)

    def _generate_from_interface(
        self,
        structural_part: Part,
        connection: ConnectionMethod,
        anchor: str,
        part_thickness: float | None,
        functional_part_id: str,
    ) -> ConnectionFeatureResult | None:
        """Try to generate features using MountingInterface from parts_catalog.

        Returns None if no interface is found, signalling the caller to
        fall back to the heuristic path.
        """
        try:
            from ..knowledge.parts_catalog import (
                get_mounting_interface,
                auto_match_interface,
            )
        except ImportError:
            return None

        mi = get_mounting_interface(functional_part_id)
        if mi is None:
            return None

        result = ConnectionFeatureResult()
        result.ops = auto_match_interface(
            structural_dims=structural_part.dimensions,
            interface=mi,
            anchor=anchor,
        )

        iface_desc = (
            f"MountingInterface({functional_part_id}, "
            f"type={mi.interface_type}, "
            f"holes={len(mi.holes)}, "
            f"bore={mi.bore_diameter}mm)"
        )
        result.features_generated.append(iface_desc)

        # If bolted, also generate fastener models
        if connection.type == "bolted" and mi.holes:
            bolt_size = connection.bolt_size or "M3"
            for i, hole in enumerate(mi.holes):
                result.fastener_ops.append({
                    "type": "fastener_model",
                    "fastener_type": "bolt",
                    "size": bolt_size,
                    "hole_index": i,
                })

        return result

    # ------------------------------------------------------------------
    # Bolted connection
    # ------------------------------------------------------------------

    def _generate_bolted_features(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate bolt holes and fastener models for a bolted connection."""
        hole_type = getattr(conn, 'hole_type', 'through_hole') or 'through_hole'

        # Dispatch to specialized hole-type handlers
        if hole_type == "threaded_hole":
            return self._generate_threaded_hole(part, conn, anchor, thickness)
        elif hole_type == "nut_pocket":
            return self._generate_nut_pocket(part, conn, anchor, thickness)
        elif hole_type == "thread_insert":
            return self._generate_thread_insert_pocket(part, conn, anchor, thickness)

        # Default: through_hole (original logic)
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []
        fastener_ops: list[dict] = []

        bolt_size = conn.bolt_size or "M3"
        # Tolerance-aware clearance hole: nominal + IT12 upper tolerance
        hole_nominal, hole_min, hole_max = get_clearance_hole_with_tolerance(bolt_size)
        hole_d = hole_nominal  # use nominal for geometry (tolerance info available for inspection)
        hole_r = hole_d / 2
        head_d, head_h = get_bolt_head_dims(bolt_size)

        # Determine hole positions
        if conn.bolt_holes:
            # Explicit positions provided
            positions = [(bh.position, bh.diameter) for bh in conn.bolt_holes]
        elif getattr(self, "_shared_bolt_pattern", None):
            # SHARED pattern from the joint interface (both mating parts use
            # the SAME (u,v) fractions → holes align in world space).  This
            # is the fix for "连不上": without it each part ran
            # _auto_layout_bolts from its OWN face size → mismatched margins.
            positions = list(self._shared_bolt_pattern)
        elif conn.bolt_count > 0:
            # Auto-layout: distribute bolt_count holes on the face
            positions = self._auto_layout_bolts(
                conn.bolt_count, d, anchor, hole_d,
            )
        else:
            # Default: 4 corner bolts
            count = 4
            positions = self._auto_layout_bolts(count, d, anchor, hole_d)

        # Generate clearance holes (through holes)
        for i, (pos, dia) in enumerate(positions):
            r = dia / 2
            # Through hole cylinder — oriented + height-clamped for the
            # anchor so side/back/front faces don't fragment the geometry.
            hole_name = f"{part.name}_bolt_hole_{i}"
            ops.append(self._make_hole_cylinder_op(
                hole_name, r, thickness, anchor, d,
            ))
            # Position on the anchor face
            x, y, z = self._position_on_face(pos, anchor, d, thickness)
            ops.append({
                "type": "move",
                "object": hole_name,
                "dx": x, "dy": y, "dz": z - 2,
            })

            # Counterbore for socket head cap screw
            cbore_name = f"{part.name}_cbore_{i}"
            ops.append({
                "type": "make_cylinder",
                "radius": head_d / 2,
                "height": head_h,
                "name": cbore_name,
            })
            # Offset counterbore inward from face surface by head_h/2
            # so it sits at the bolt entry, recessing into the material.
            _inward = {
                "top": (0, 0, -1), "bottom": (0, 0, 1),
                "front": (0, 1, 0), "back": (0, -1, 0),
                "left": (1, 0, 0), "right": (-1, 0, 0),
            }
            nx, ny, nz = _inward.get(anchor, (0, 0, -1))
            cbore_x = x + nx * head_h / 2
            cbore_y = y + ny * head_h / 2
            cbore_z = (z - 2) + nz * head_h / 2
            ops.append({
                "type": "move",
                "object": cbore_name,
                "dx": cbore_x, "dy": cbore_y, "dz": cbore_z,
            })

        # Generate fastener models (bolt + nut + washer per hole).
        # These are returned in result.fastener_ops for the assembly-level
        # pipeline to position at joint interfaces; they are NOT embedded
        # in this part's STL because fasteners span two parts.
        for i, (pos, dia) in enumerate(positions):
            fastener_ops.extend(
                self._generate_fastener_set(bolt_size, thickness, i)
            )

        result.features_generated.append(
            f"{len(positions)}× {bolt_size} clearance holes (Ø{hole_d}mm) "
            f"with Ø{head_d}×{head_h}mm counterbores"
        )
        result.features_generated.append(
            f"{len(positions)}× {bolt_size} bolt+nut+washer sets"
        )
        result.ops = ops
        result.fastener_ops = fastener_ops
        return result

    # ------------------------------------------------------------------
    # Press-fit connection
    # ------------------------------------------------------------------

    def _generate_press_fit_features(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate press-fit bore with interference and shoulder."""
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []

        # Determine bore specs from interference or defaults
        # For bearing seats: bore = bearing OD with H7/js6 tolerance
        # For press-fit: bore = OD - interference with H7/p6 tolerance
        interference = conn.interference_mm or 0.05  # default 0.05mm

        # Try to infer bore diameter from part dimensions or notes
        bore_d = self._infer_bore_diameter(part, conn)

        # Determine if this is a bearing seat (no explicit interference)
        is_bearing_seat = bore_d > 0 and conn.interference_mm == 0
        if bore_d <= 0:
            result.warnings.append(
                "Cannot determine press-fit bore diameter. "
                "Set part.dimensions['bore_diameter'] or connection metadata."
            )
            return result

        # Compute actual bore with tolerance
        if is_bearing_seat and bore_d > 0:
            # Bearing seat: H7 tolerance on bore diameter
            bore_min, bore_nominal, bore_max = bearing_seat_diameter(bore_d)
            actual_bore = bore_nominal
            tol_info = f"H7 ({bore_min:.3f}/{bore_max:.3f})"
        else:
            # Press-fit: use H7/p6 to compute interference bore
            bore_min, bore_nominal, bore_max = press_fit_bore_diameter(bore_d, interference)
            actual_bore = bore_nominal
            tol_info = f"H7/p6 ({bore_min:.3f}/{bore_max:.3f})"
        shoulder_d = bore_d + 2.0  # shoulder 1mm wider than nominal
        shoulder_depth = max(1.0, thickness * 0.1)

        # Deep bore (through)
        bore_name = f"{part.name}_pf_bore"
        ops.append(self._make_hole_cylinder_op(
            bore_name, actual_bore / 2, thickness, anchor, d,
        ))
        bx, by, bz = self._anchor_center(anchor, d, thickness)
        ops.append({
            "type": "move",
            "object": bore_name,
            "dx": bx, "dy": by, "dz": bz - 2,
        })

        # Shoulder (shallow wider bore at entrance)
        sh_name = f"{part.name}_pf_shoulder"
        ops.append({
            "type": "make_cylinder",
            "radius": shoulder_d / 2,
            "height": shoulder_depth,
            "name": sh_name,
        })
        ops.append({
            "type": "move",
            "object": sh_name,
            "dx": bx, "dy": by, "dz": bz,
        })

        result.features_generated.append(
            f"Press-fit bore Ø{actual_bore:.2f}mm (interference {interference}mm, {tol_info}) "
            f"with Ø{shoulder_d}mm × {shoulder_depth}mm shoulder"
        )
        result.ops = ops
        return result

    # ------------------------------------------------------------------
    # Snap-fit connection
    # ------------------------------------------------------------------

    def _generate_snap_fit_features(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate cantilever snap-fit hooks and matching slots."""
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []

        snap_count = conn.snap_count or 2  # default 2 snaps
        # Snap hook dimensions (cantilever beam style)
        hook_length = max(5.0, min(thickness * 0.8, 15.0))
        hook_thickness = max(1.0, thickness * 0.15)
        hook_width = max(3.0, min(d.get("width", 10) * 0.15, 6.0))
        undercut = hook_thickness * 0.4  # snap undercut depth

        # Distribute snaps along the anchor face edge
        face_length = self._face_length(anchor, d)

        for i in range(snap_count):
            if snap_count == 1:
                t = 0.5  # center
            else:
                t = 0.15 + 0.7 * i / (snap_count - 1)

            offset = face_length * t

            # Snap hook (rectangular beam with undercut)
            hook_name = f"{part.name}_snap_{i}"
            ops.append({
                "type": "make_box",
                "length": hook_length,
                "width": hook_width,
                "height": hook_thickness,
                "name": hook_name,
            })

            # Position on anchor face edge
            hx, hy, hz = self._snap_position(
                anchor, d, thickness, offset, hook_length, hook_thickness,
            )
            ops.append({
                "type": "move",
                "object": hook_name,
                "dx": hx, "dy": hy, "dz": hz,
            })

            # Undercut (small box subtracted from hook tip)
            uc_name = f"{part.name}_snap_uc_{i}"
            ops.append({
                "type": "make_box",
                "length": hook_length * 0.3,
                "width": hook_width + 0.5,
                "height": undercut,
                "name": uc_name,
            })
            uc_x, uc_y, uc_z = self._snap_undercut_position(
                anchor, d, thickness, offset, hook_length, hook_thickness,
            )
            ops.append({
                "type": "move",
                "object": uc_name,
                "dx": uc_x, "dy": uc_y, "dz": uc_z,
            })

        result.features_generated.append(
            f"{snap_count}× snap-fit hooks ({hook_length}×{hook_width}×{hook_thickness}mm, "
            f"undercut {undercut:.1f}mm)"
        )
        result.ops = ops
        return result

    # ------------------------------------------------------------------
    # Adhesive connection
    # ------------------------------------------------------------------

    def _generate_adhesive_features(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate adhesive bonding surface features.

        Adds a thin gap (0.1-0.3mm) on the bonding face and optional
        surface grooves to increase bonding area.
        """
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []

        gap = 0.2  # mm, adhesive gap
        groove_depth = 0.3  # mm, surface texture groove depth
        groove_spacing = 3.0  # mm, between grooves
        groove_width = 0.5  # mm

        face_l = self._face_length(anchor, d)
        face_w = self._face_width(anchor, d)
        margin = min(face_l, face_w) * 0.1

        # Generate parallel grooves on bonding surface
        num_grooves = max(1, int((face_w - 2 * margin) / groove_spacing))
        for i in range(num_grooves):
            t = (i + 0.5) / num_grooves
            y_pos = margin + t * (face_w - 2 * margin)

            groove_name = f"{part.name}_groove_{i}"
            ops.append({
                "type": "make_box",
                "length": face_l - 2 * margin,
                "width": groove_width,
                "height": groove_depth,
                "name": groove_name,
            })
            gx, gy, gz = self._groove_position(anchor, d, margin, y_pos, gap)
            ops.append({
                "type": "move",
                "object": groove_name,
                "dx": gx, "dy": gy, "dz": gz,
            })

        bond_area = (face_l - 2 * margin) * (face_w - 2 * margin)
        result.features_generated.append(
            f"Adhesive bonding surface ({conn.adhesive_type or 'epoxy'}), "
            f"gap {gap}mm, {num_grooves} grooves, bond area ≈{bond_area:.0f}mm²"
        )
        result.ops = ops
        return result

    # ------------------------------------------------------------------
    # Welded connection
    # ------------------------------------------------------------------

    def _generate_welded_features(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate weld preparation features (bevel/butt joint prep).

        For butt welds: adds a bevel (V-groove) on the weld edge.
        For fillet welds: no special prep needed (T-joint).
        """
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []

        weld_type = conn.weld_type or "fillet"

        if weld_type == "butt":
            # V-groove bevel on the butt face
            bevel_depth = thickness * 0.6  # 60% penetration
            bevel_width = bevel_depth  # 45° bevel → width = depth
            face_l = self._face_length(anchor, d)

            # Triangular bevel groove (box approximation)
            bevel_name = f"{part.name}_weld_bevel"
            ops.append({
                "type": "make_box",
                "length": face_l,
                "width": bevel_width,
                "height": bevel_depth,
                "name": bevel_name,
            })
            bx, by, bz = self._bevel_position(anchor, d, thickness)
            ops.append({
                "type": "move",
                "object": bevel_name,
                "dx": bx, "dy": by, "dz": bz,
            })
            result.features_generated.append(
                f"Butt weld V-groove ({bevel_depth:.1f}mm deep, "
                f"{bevel_width:.1f}mm wide, 45°)"
            )
        else:
            # Fillet/spot/TIG/MIG: no special prep
            result.features_generated.append(
                f"{weld_type} weld — no joint preparation needed"
            )

        result.ops = ops
        return result

    # ------------------------------------------------------------------
    # Dowel pin alignment
    # ------------------------------------------------------------------

    def _generate_dowel_pin_features(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate dowel pin H7 slip-fit holes and pin models for alignment."""
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []
        fastener_ops: list[dict] = []

        # Determine pin specs from connection metadata or defaults
        pin_diameter = getattr(conn, 'bolt_size', None)
        if pin_diameter:
            # Try to parse pin diameter from metadata
            try:
                pin_d = float(pin_diameter.replace("D", "").split("x")[0])
            except (ValueError, AttributeError):
                pin_d = 5.0
        else:
            pin_d = 5.0

        pin_length = thickness
        hole_d = pin_d + 0.01  # H7 slip-fit
        hole_r = hole_d / 2

        # Default: 2 dowel pins on opposite sides of the face center
        cx, cy, cz = self._anchor_center(anchor, d, thickness)
        face_l = self._face_length(anchor, d)
        offset = min(face_l * 0.2, 15.0)

        pin_positions = []
        if anchor in ("top", "bottom"):
            pin_positions = [
                (cx - offset, cy, cz),
                (cx + offset, cy, cz),
            ]
        elif anchor in ("front", "back"):
            pin_positions = [
                (cx, cy, cz - offset),
                (cx, cy, cz + offset),
            ]
        else:  # left, right
            pin_positions = [
                (cx, cy, cz - offset),
                (cx, cy, cz + offset),
            ]

        for i, (px, py, pz) in enumerate(pin_positions):
            # H7 slip-fit hole
            hole_name = f"{part.name}_dowel_hole_{i}"
            ops.append(self._make_hole_cylinder_op(
                hole_name, hole_r, thickness, anchor, d,
            ))
            ops.append({
                "type": "move",
                "object": hole_name,
                "dx": px, "dy": py, "dz": pz - 2,
            })

            # Dowel pin model (cylinder)
            pin_name = f"dowel_pin_{pin_d:.0f}x{pin_length:.0f}_{i}"
            fastener_ops.append({
                "type": "make_cylinder",
                "radius": pin_d / 2,
                "height": pin_length,
                "name": pin_name,
            })
            fastener_ops.append({
                "type": "move",
                "object": pin_name,
                "dx": px, "dy": py, "dz": pz,
            })

        result.features_generated.append(
            f"{len(pin_positions)}× dowel pin Ø{pin_d:.1f}mm holes "
            f"(H7 slip-fit Ø{hole_d:.2f}mm) for alignment"
        )
        result.ops = ops
        result.fastener_ops = fastener_ops
        return result

    # ------------------------------------------------------------------
    # Magnetic connection
    # ------------------------------------------------------------------

    def _generate_magnetic_features(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate magnet pocket (cylindrical cavity for disc magnet).

        Common magnet sizes: Ø5×2mm, Ø6×2mm, Ø8×3mm, Ø10×3mm.
        Pocket is slightly larger than magnet for press-fit.
        """
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []

        # Default: Ø8×3mm disc magnet
        magnet_od = 8.0
        magnet_height = 3.0
        pocket_od = magnet_od + 0.1  # 0.1mm interference fit
        pocket_depth = magnet_height + 0.5  # 0.5mm below surface

        # Position at center of anchor face
        cx, cy, cz = self._anchor_center(anchor, d, thickness)

        pocket_name = f"{part.name}_magnet_pocket"
        ops.append({
            "type": "make_cylinder",
            "radius": pocket_od / 2,
            "height": pocket_depth,
            "name": pocket_name,
        })
        ops.append({
            "type": "move",
            "object": pocket_name,
            "dx": cx, "dy": cy, "dz": cz,
        })

        result.features_generated.append(
            f"Magnet pocket Ø{pocket_od:.1f}×{pocket_depth:.1f}mm "
            f"(for Ø{magnet_od}×{magnet_height}mm disc magnet)"
        )
        result.ops = ops
        return result

    # ------------------------------------------------------------------
    # Threaded hole (tap drill hole for machine screws)
    # ------------------------------------------------------------------

    def _generate_threaded_hole(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate tapped/threaded holes instead of through holes + counterbore."""
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []

        bolt_size = conn.bolt_size or "M3"
        tap_d = _get_tap_hole(bolt_size)
        if tap_d <= 0:
            tap_d = float(bolt_size.replace("M", "")) - 0.5

        # Determine hole positions
        if conn.bolt_holes:
            positions = [(bh.position, bh.diameter) for bh in conn.bolt_holes]
        elif conn.bolt_count > 0:
            positions = self._auto_layout_bolts(
                conn.bolt_count, d, anchor, tap_d,
            )
        else:
            positions = self._auto_layout_bolts(4, d, anchor, tap_d)

        for i, (pos, dia) in enumerate(positions):
            r = tap_d / 2
            hole_name = f"{part.name}_tap_hole_{i}"
            _tap_op = self._make_hole_cylinder_op(
                hole_name, r, thickness, anchor, d,
            )
            # Preserve threaded-hole metadata used by downstream tooling.
            _tap_op["hole_type"] = "threaded"
            _tap_op["thread_size"] = bolt_size
            ops.append(_tap_op)
            x, y, z = self._position_on_face(pos, anchor, d, thickness)
            ops.append({
                "type": "move",
                "object": hole_name,
                "dx": x, "dy": y, "dz": z - 2,
            })

        result.features_generated.append(
            f"{len(positions)}× {bolt_size} threaded holes "
            f"(tap drill Ø{tap_d}mm, depth {thickness:.1f}mm)"
        )
        result.ops = ops
        return result

    # ------------------------------------------------------------------
    # Nut pocket (through hole + hex pocket for nut)
    # ------------------------------------------------------------------

    def _generate_nut_pocket(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate through hole with hexagonal nut pocket on the back side."""
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []

        bolt_size = conn.bolt_size or "M3"

        # Through hole diameter (clearance)
        hole_nominal, _, _ = get_clearance_hole_with_tolerance(bolt_size)
        hole_r = hole_nominal / 2

        # Nut pocket dimensions from catalog
        nut_spec = _get_nut_spec(bolt_size)
        if nut_spec:
            nut_w = nut_spec.width_across_flats
            nut_h = nut_spec.height
        else:
            nut_w = 5.5
            nut_h = 2.4

        # Hex pocket: approximated as cylinder using width_across_corners
        # (hex inscribed in circle)
        pocket_r = nut_w / 2 * 1.1  # 10% clearance
        pocket_depth = nut_h + 0.5   # slight extra depth

        # Determine hole positions
        if conn.bolt_holes:
            positions = [(bh.position, bh.diameter) for bh in conn.bolt_holes]
        elif conn.bolt_count > 0:
            positions = self._auto_layout_bolts(
                conn.bolt_count, d, anchor, hole_nominal,
            )
        else:
            positions = self._auto_layout_bolts(4, d, anchor, hole_nominal)

        for i, (pos, dia) in enumerate(positions):
            # Through hole
            hole_name = f"{part.name}_bolt_hole_{i}"
            ops.append(self._make_hole_cylinder_op(
                hole_name, hole_r, thickness, anchor, d,
            ))
            x, y, z = self._position_on_face(pos, anchor, d, thickness)
            ops.append({
                "type": "move",
                "object": hole_name,
                "dx": x, "dy": y, "dz": z - 2,
            })

            # Nut pocket (hex approximation on opposite face)
            pocket_name = f"{part.name}_nut_pocket_{i}"
            ops.append({
                "type": "make_cylinder",
                "radius": pocket_r,
                "height": pocket_depth,
                "name": pocket_name,
                "hole_type": "nut_pocket",
                "nut_size": bolt_size,
            })
            ops.append({
                "type": "move",
                "object": pocket_name,
                "dx": x, "dy": y, "dz": z - pocket_depth,
            })

        result.features_generated.append(
            f"{len(positions)}× {bolt_size} through holes (Ø{hole_nominal}mm) "
            f"with Ø{nut_w}mm hex nut pockets ({pocket_depth:.1f}mm deep)"
        )
        result.ops = ops
        return result

    # ------------------------------------------------------------------
    # Thread insert pocket (for heat-set brass inserts)
    # ------------------------------------------------------------------

    def _generate_thread_insert_pocket(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate pilot hole + pocket for heat-set brass thread insert."""
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []

        bolt_size = conn.bolt_size or "M3"
        insert_spec = _get_thread_insert_spec(bolt_size)
        if insert_spec:
            install_d = insert_spec.install_hole_diameter
            insert_len = insert_spec.length
        else:
            install_d = float(bolt_size.replace("M", "")) + 1.5
            insert_len = 5.6

        # Determine hole positions
        if conn.bolt_holes:
            positions = [(bh.position, bh.diameter) for bh in conn.bolt_holes]
        elif conn.bolt_count > 0:
            positions = self._auto_layout_bolts(
                conn.bolt_count, d, anchor, install_d,
            )
        else:
            positions = self._auto_layout_bolts(4, d, anchor, install_d)

        for i, (pos, dia) in enumerate(positions):
            # Insert pocket (cylinder for brass insert)
            pocket_name = f"{part.name}_insert_pocket_{i}"
            ops.append({
                "type": "make_cylinder",
                "radius": install_d / 2,
                "height": insert_len + 1.0,
                "name": pocket_name,
                "hole_type": "thread_insert",
                "insert_size": bolt_size,
            })
            x, y, z = self._position_on_face(pos, anchor, d, thickness)
            ops.append({
                "type": "move",
                "object": pocket_name,
                "dx": x, "dy": y, "dz": z,
            })

        result.features_generated.append(
            f"{len(positions)}× {bolt_size} thread insert pockets "
            f"(Ø{install_d}mm, depth {insert_len + 1.0:.1f}mm)"
        )
        result.ops = ops
        return result

    # ------------------------------------------------------------------
    # Set screw / grub screw connection
    # ------------------------------------------------------------------

    def _generate_set_screw_features(
        self,
        part: Part,
        conn: ConnectionMethod,
        anchor: str,
        thickness: float,
    ) -> ConnectionFeatureResult:
        """Generate radial threaded hole for set screw / grub screw.

        The set screw passes radially through the hub wall to lock onto a shaft.
        """
        result = ConnectionFeatureResult()
        d = part.dimensions
        ops: list[dict] = []
        fastener_ops: list[dict] = []

        size = getattr(conn, 'set_screw_size', None) or conn.bolt_size or "M3"
        tap_d = _get_tap_hole(size)
        if tap_d <= 0:
            tap_d = float(size.replace("M", "")) - 0.5

        # Determine radial hole depth: hub wall thickness
        # If part has outer_diameter and bore_diameter, wall = (OD - bore) / 2
        od = d.get("outer_diameter", d.get("diameter", 0))
        bore = d.get("bore_diameter", d.get("inner_diameter", 0))
        if od > 0 and bore > 0:
            wall_t = (od - bore) / 2
        else:
            wall_t = thickness / 2

        hole_depth = wall_t + 2.0  # penetrate through wall + margin

        # Position: center of anchor face
        cx, cy, cz = self._anchor_center(anchor, d, thickness)

        # Radial threaded hole
        hole_name = f"{part.name}_set_screw_hole"
        ops.append({
            "type": "make_cylinder",
            "radius": tap_d / 2,
            "height": hole_depth,
            "name": hole_name,
            "hole_type": "threaded",
            "thread_size": size,
        })
        # Position from the outer surface inward
        if anchor in ("top", "bottom"):
            offset_x = od / 2 if od > 0 else d.get("length", 20) / 2
            ops.append({
                "type": "move",
                "object": hole_name,
                "dx": cx + offset_x, "dy": cy, "dz": cz,
            })
        else:
            ops.append({
                "type": "move",
                "object": hole_name,
                "dx": cx, "dy": cy, "dz": cz + thickness / 2,
            })

        # Set screw model
        set_spec = _get_set_screw_spec(size)
        thread_d = float(size.replace("M", ""))
        screw_length = wall_t + 1.0
        screw_name = f"set_screw_{size}_0"
        fastener_ops.append({
            "type": "make_cylinder",
            "radius": thread_d / 2,
            "height": screw_length,
            "name": screw_name,
        })

        result.features_generated.append(
            f"1× {size} set screw hole (tap Ø{tap_d}mm, radial, "
            f"wall {wall_t:.1f}mm)"
        )
        result.ops = ops
        result.fastener_ops = fastener_ops
        return result

    # ------------------------------------------------------------------
    # Fastener model generation
    # ------------------------------------------------------------------

    def _generate_fastener_set(
        self,
        bolt_size: str,
        grip_thickness: float,
        index: int,
    ) -> list[dict]:
        """Generate FreeCAD ops for one bolt + nut + washer set.

        The bolt length is chosen to accommodate the grip thickness plus
        washer + nut height.
        """
        ops: list[dict] = []
        head_d, head_h = get_bolt_head_dims(bolt_size)
        nut_w, nut_h = get_nut_dims(bolt_size)
        washer_id, washer_od, washer_t = get_washer_dims(bolt_size)

        # Bolt length: grip + washer + some thread engagement
        thread_d = float(bolt_size.replace("M", ""))
        bolt_length = self._select_bolt_length(grip_thickness + washer_t + nut_h)
        shank_length = bolt_length - head_h

        # 1. Bolt (hex head + shank)
        bolt_name = f"bolt_{bolt_size}_{index}"
        ops.append({
            "type": "make_cylinder",
            "radius": head_d / 2,
            "height": head_h,
            "name": f"{bolt_name}_head",
        })
        ops.append({
            "type": "make_cylinder",
            "radius": thread_d / 2,
            "height": shank_length,
            "name": f"{bolt_name}_shank",
        })
        # Move shank to top of head
        ops.append({
            "type": "move",
            "object": f"{bolt_name}_shank",
            "dx": 0, "dy": 0, "dz": head_h,
        })
        # Fuse head + shank
        ops.append({
            "type": "boolean",
            "operation": "union",
            "object1": f"{bolt_name}_head",
            "object2": f"{bolt_name}_shank",
            "result_name": bolt_name,
        })

        # 2. Washer
        washer_name = f"washer_{bolt_size}_{index}"
        ops.append({
            "type": "cylinder_with_hole",
            "outer_radius": washer_od / 2,
            "inner_radius": washer_id / 2,
            "height": washer_t,
            "name": washer_name,
        })

        # 3. Nut (hexagonal → approximated as cylinder for simplicity)
        nut_name = f"nut_{bolt_size}_{index}"
        ops.append({
            "type": "cylinder_with_hole",
            "outer_radius": nut_w / 2,
            "inner_radius": thread_d / 2,
            "height": nut_h,
            "name": nut_name,
        })

        return ops

    @staticmethod
    def _select_bolt_length(required: float) -> float:
        """Select standard bolt length >= required grip.

        Standard lengths: 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50.
        """
        standard_lengths = [6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50]
        for length in standard_lengths:
            if length >= required:
                return float(length)
        return math.ceil(required / 10) * 10.0

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_thickness(d: dict, anchor: str) -> float:
        """Infer the thickness of the face from part dimensions.

        The thickness is the material depth along the anchor normal — i.e.
        how far the bolt shank travels through the part.  In FreeCAD
        coordinates (X=length, Y=width, Z=height):

        - front/back faces (±Y normal): thickness along Y = ``width``
        - left/right faces (±X normal): thickness along X = ``length``
        - top/bottom faces (±Z normal): thickness along Z = ``height``
        """
        h = d.get("height", d.get("thickness", d.get("length", 10)))
        anchor_map = {
            "top": h, "bottom": h,
            "front": d.get("width", h), "back": d.get("width", h),
            "left": d.get("length", h), "right": d.get("length", h),
        }
        return anchor_map.get(anchor, h)

    # ------------------------------------------------------------------
    # Bolt-hole cylinder orientation (added 2026-06-21, Plan B)
    # ------------------------------------------------------------------
    # ``Part.makeCylinder(r, h)`` always builds a cylinder along the Z axis.
    # For top/bottom anchors this is correct (the bolt enters along Z).
    # But for front/back (bolt enters along Y) and left/right (bolt enters
    # along X) the cylinder axis is wrong by 90°, which means the
    # through-hole cylinder is taller than the part is deep and the boolean
    # cut fragments the part geometry into tens of thousands of triangles
    # (gripper_base went from a clean 28×50×32 box to 90,224 triangles with
    # euler_number=-8).  Every hole-type generator below now routes its
    # cylinder through ``_make_hole_cylinder_op`` which applies the right
    # rotation for the anchor so the cylinder axis aligns with the bolt
    # direction, AND clamps the height so it never exceeds the part depth
    # along that axis plus a small through-penetration margin.
    _ANCHOR_AXIS_ROTATION: dict[str, tuple] = {
        # anchor -> (axis_vector, angle_deg) for rotating the Z-aligned
        # makeCylinder so its axis points along the anchor normal.
        "top":    ((0.0, 0.0, 1.0), 0.0),     # Z axis, no rotation
        "bottom": ((0.0, 0.0, 1.0), 0.0),     # Z axis, no rotation
        "front":  ((1.0, 0.0, 0.0), 90.0),    # rotate Z→Y (around X)
        "back":   ((1.0, 0.0, 0.0), 90.0),    # rotate Z→Y (around X)
        "left":   ((0.0, 1.0, 0.0), 90.0),    # rotate Z→X (around Y)
        "right":  ((0.0, 1.0, 0.0), 90.0),    # rotate Z→X (around Y)
    }

    @staticmethod
    def _axis_depth(d: dict, anchor: str) -> float:
        """Return the part's full extent along the anchor's normal axis.

        Used to clamp hole-cylinder heights so a through-hole protrudes
        only a few mm past the face, not 20+ mm (which fragments the
        boolean result).  Contrast with ``_infer_thickness`` which returns
        the *same* value for through-holes — kept separate so callers
        are explicit about which dimension they mean.
        """
        l = float(d.get("length", 0) or 0)
        w = float(d.get("width", 0) or 0)
        h = float(d.get("height", d.get("thickness", 0)) or 0)
        if anchor in ("top", "bottom"):
            return h
        if anchor in ("front", "back"):
            return w
        if anchor in ("left", "right"):
            return l
        return max(l, w, h)

    @classmethod
    def _make_hole_cylinder_op(
        cls,
        name: str,
        radius: float,
        thickness: float,
        anchor: str,
        d: dict,
        *,
        margin: float = 4.0,
    ) -> dict:
        """Build a through-hole cylinder op oriented for ``anchor``.

        Returns a ``make_cylinder`` op dict with:
        - ``height``: clamped to ``min(thickness + margin, axis_depth + margin)``
          so the cylinder only protrudes ``margin`` past the far face
          (preventing the geometry fragmentation caused by the old
          unbounded ``thickness + 4`` on side anchors).
        - ``rotation``: ``(axis_vector, angle_deg)`` so the cylinder's Z
          axis aligns with the bolt direction for the anchor.  ``None``
          for top/bottom (no rotation needed).

        The rotation is consumed by ``freecad.py``'s ``make_cylinder`` op
        handler, which applies it as ``Placement`` on the FreeCAD cylinder.
        """
        depth = cls._axis_depth(d, anchor)
        # Clamp: never let the cylinder exceed the part depth by more
        # than ``margin`` mm on either side (so it cuts clean through
        # without poking 20mm into empty space and fragmenting the cut).
        height = min(thickness + margin, depth + margin)
        axis, angle = cls._ANCHOR_AXIS_ROTATION.get(anchor, ((0, 0, 1), 0.0))
        op: dict = {
            "type": "make_cylinder",
            "radius": radius,
            "height": height,
            "name": name,
        }
        if angle != 0.0:
            # Rotation applied by freecad.py after makeCylinder.  Passed
            # as a list (JSON-serialisable) so it survives the op dict
            # round-trip into the FreeCAD script generator.
            op["rotation"] = [axis[0], axis[1], axis[2], angle]
        return op

    @staticmethod
    def _face_length(anchor: str, d: dict) -> float:
        """Return the primary length of the anchor face.

        F16: every face previously returned ``d["length"]`` regardless of
        orientation, so left/right faces (which span length×height, not
        length×width) were sized incorrectly and bolt margins were wrong.
        """
        mapping = {
            "top": d.get("length", 20),
            "bottom": d.get("length", 20),
            "left": d.get("width", d.get("height", 20)),
            "right": d.get("width", d.get("height", 20)),
            "front": d.get("length", 20),
            "back": d.get("length", 20),
        }
        return mapping.get(anchor, d.get("length", 20))

    @staticmethod
    def _face_width(anchor: str, d: dict) -> float:
        """Return the secondary width of the anchor face."""
        mapping = {
            "top": d.get("width", 20),
            "bottom": d.get("width", 20),
            "left": d.get("height", 20),
            "right": d.get("height", 20),
            "front": d.get("width", 20),
            "back": d.get("width", 20),
        }
        return mapping.get(anchor, d.get("width", 20))

    @staticmethod
    def _anchor_center(anchor: str, d: dict, thickness: float) -> tuple[float, float, float]:
        """Return the center point of the anchor face.

        Assumes part origin is at one corner (0,0,0) with the body
        extending in +X, +Y, +Z.
        """
        l = d.get("length", 20)
        w = d.get("width", 20)
        h = d.get("height", d.get("thickness", 10))

        centers = {
            "top": (l / 2, w / 2, h),
            "bottom": (l / 2, w / 2, 0),
            # F16: left/right faces are at X=0/X=L but span the full Y
            # extent — the centre Y must be w/2, not 0 (which placed every
            # feature on the front edge of the side face).
            "left": (0, w / 2, h / 2),
            "right": (l, w / 2, h / 2),
            "front": (l / 2, 0, h / 2),
            "back": (l / 2, w, h / 2),
        }
        return centers.get(anchor, (l / 2, w / 2, h / 2))

    def _position_on_face(
        self,
        position: tuple[float, float, float],
        anchor: str,
        d: dict,
        thickness: float,
    ) -> tuple[float, float, float]:
        """Map a relative position on the anchor face to absolute coords.

        position is (u, v, _ignored) where u,v ∈ [0,1] along the face.
        """
        l = d.get("length", 20)
        w = d.get("width", 20)
        h = d.get("height", d.get("thickness", 10))
        u, v, _ = position

        mapping: dict[str, tuple[float, float, float]] = {
            "top": (u * l, v * w, h),
            "bottom": (u * l, v * w, 0),
            "left": (0, u * w, v * h),
            "right": (l, u * w, v * h),
            "front": (u * l, 0, v * h),
            "back": (u * l, w, v * h),
        }
        return mapping.get(anchor, (u * l, v * w, h / 2))

    def _auto_layout_bolts(
        self,
        count: int,
        d: dict,
        anchor: str,
        hole_diameter: float,
    ) -> list[tuple[tuple[float, float, float], float]]:
        """Auto-layout bolt positions on an anchor face.

        Returns list of ((u, v, 0), hole_diameter).
        """
        face_l = self._face_length(anchor, d)
        face_w = self._face_width(anchor, d)
        margin = max(hole_diameter * 1.5, min(face_l, face_w) * 0.1)
        margin_u = margin / face_l
        margin_v = margin / face_w

        if count == 1:
            return [((0.5, 0.5, 0), hole_diameter)]
        elif count == 2:
            return [
                ((margin_u, 0.5, 0), hole_diameter),
                ((1 - margin_u, 0.5, 0), hole_diameter),
            ]
        elif count == 4:
            return [
                ((margin_u, margin_v, 0), hole_diameter),
                ((1 - margin_u, margin_v, 0), hole_diameter),
                ((margin_u, 1 - margin_v, 0), hole_diameter),
                ((1 - margin_u, 1 - margin_v, 0), hole_diameter),
            ]
        else:
            # Distribute evenly on a grid
            cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)
            positions = []
            for r in range(rows):
                for c in range(cols):
                    if len(positions) >= count:
                        break
                    u = margin_u + (1 - 2 * margin_u) * c / max(cols - 1, 1)
                    v = margin_v + (1 - 2 * margin_v) * r / max(rows - 1, 1)
                    positions.append(((u, v, 0), hole_diameter))
            return positions

    @staticmethod
    def compute_shared_bolt_pattern(
        part_a: Part,
        part_b: Part,
        anchor_a: str,
        anchor_b: str,
        conn: ConnectionMethod,
    ) -> list[tuple[tuple[float, float, float], float]]:
        """Compute ONE bolt-hole pattern shared by BOTH mating parts.

        The "连不上" defect: each part ran ``_auto_layout_bolts`` on its OWN
        face dimensions, so parent and child holes landed at different
        world coordinates.  This method derives a single canonical pattern
        from the SMALLER of the two mating faces (so holes fit in both),
        and returns normalized (u, v) fractions that BOTH parts apply on
        their own face — yielding matching hole positions in world space.

        The two anchors are a mated face pair (parent top ↔ child bottom,
        etc.): same physical plane, opposite normal.  Using the smaller
        face for margin keeps the bolt circle inside both parts.

        Args:
            part_a/part_b: the two mating parts (parent and child).
            anchor_a/anchor_b: the respective mated-face anchors.
            conn: the ConnectionMethod (reads bolt_count, bolt_size,
                bolt_holes).

        Returns:
            list of ((u, v, 0), diameter_mm) — the SAME list passed to
            both parts' generate_features via ``bolt_pattern``.
        """
        # Explicit positions always win — both parts use them verbatim.
        if conn.bolt_holes:
            return [(bh.position, bh.diameter) for bh in conn.bolt_holes]
        count = getattr(conn, "bolt_count", 0) or 0
        if count <= 0:
            return []
        bolt_size = getattr(conn, "bolt_size", "M3") or "M3"
        nominal, _min, _max = get_clearance_hole_with_tolerance(bolt_size)
        hole_d = nominal

        # The two faces share one physical plane.  Derive the pattern from
        # the SMALLER face so the margin keeps holes inside both parts.
        # _face_length/_face_width are static helpers; use the min per axis.
        fa_l = ConnectionFeatureEngine._face_length(anchor_a, part_a.dimensions)
        fa_w = ConnectionFeatureEngine._face_width(anchor_a, part_a.dimensions)
        fb_l = ConnectionFeatureEngine._face_length(anchor_b, part_b.dimensions)
        fb_w = ConnectionFeatureEngine._face_width(anchor_b, part_b.dimensions)
        face_l = min(fa_l, fb_l)
        face_w = min(fa_w, fb_w)
        if face_l <= 0 or face_w <= 0:
            face_l = max(face_l, fa_l, fb_l) or 1.0
            face_w = max(face_w, fa_w, fb_w) or 1.0

        margin = max(hole_d * 1.5, min(face_l, face_w) * 0.1)
        margin_u = margin / face_l
        margin_v = margin / face_w

        if count == 1:
            return [((0.5, 0.5, 0), hole_d)]
        if count == 2:
            return [
                ((margin_u, 0.5, 0), hole_d),
                ((1 - margin_u, 0.5, 0), hole_d),
            ]
        if count == 4:
            return [
                ((margin_u, margin_v, 0), hole_d),
                ((1 - margin_u, margin_v, 0), hole_d),
                ((margin_u, 1 - margin_v, 0), hole_d),
                ((1 - margin_u, 1 - margin_v, 0), hole_d),
            ]
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        positions: list[tuple[tuple[float, float, float], float]] = []
        for r in range(rows):
            for c in range(cols):
                if len(positions) >= count:
                    break
                u = margin_u + (1 - 2 * margin_u) * c / max(cols - 1, 1)
                v = margin_v + (1 - 2 * margin_v) * r / max(rows - 1, 1)
                positions.append(((u, v, 0), hole_d))
        return positions

    @staticmethod
    def _infer_bore_diameter(part: Part, conn: ConnectionMethod) -> float:
        """Try to infer bore diameter from part dimensions or connection info."""
        d = part.dimensions
        # Explicit bore diameter
        if "bore_diameter" in d:
            return d["bore_diameter"]
        # Bearing seat: infer from name
        name = part.name.lower()
        for bearing_name, (id_, od, w) in _BEARING_SPECS.items():
            if bearing_name in name:
                return od
        # Press-fit shaft: use shaft diameter from dimensions
        if "shaft_diameter" in d:
            return d["shaft_diameter"]
        if "inner_diameter" in d:
            return d["inner_diameter"]
        return 0.0

    def _snap_position(
        self,
        anchor: str,
        d: dict,
        thickness: float,
        offset: float,
        hook_length: float,
        hook_thickness: float,
    ) -> tuple[float, float, float]:
        """Calculate snap hook position on the anchor face."""
        l = d.get("length", 20)
        w = d.get("width", 20)
        h = d.get("height", d.get("thickness", 10))

        # Position along the edge of the anchor face
        if anchor in ("top", "bottom"):
            return (offset, 0, h if anchor == "top" else hook_thickness)
        elif anchor in ("front", "back"):
            return (offset, w if anchor == "back" else 0, h - hook_thickness)
        else:  # left, right
            return (l if anchor == "right" else hook_length, offset, h - hook_thickness)

    def _snap_undercut_position(
        self,
        anchor: str,
        d: dict,
        thickness: float,
        offset: float,
        hook_length: float,
        hook_thickness: float,
    ) -> tuple[float, float, float]:
        """Calculate snap undercut position (at the tip of the hook)."""
        base = self._snap_position(anchor, d, thickness, offset, hook_length, hook_thickness)
        # Offset to the tip of the hook
        if anchor in ("top", "bottom"):
            return (base[0], base[1], base[2] - hook_thickness * 0.3)
        return (base[0], base[1], base[2])

    def _groove_position(
        self,
        anchor: str,
        d: dict,
        margin: float,
        y_pos: float,
        gap: float,
    ) -> tuple[float, float, float]:
        """Calculate adhesive groove position."""
        l = d.get("length", 20)
        w = d.get("width", 20)
        h = d.get("height", d.get("thickness", 10))

        if anchor == "top":
            return (margin, y_pos, h + gap)
        elif anchor == "bottom":
            return (margin, y_pos, -gap)
        elif anchor == "front":
            return (margin, -gap, y_pos)
        elif anchor == "back":
            return (margin, w + gap, y_pos)
        elif anchor == "left":
            return (-gap, y_pos, margin)
        else:  # right
            return (l + gap, y_pos, margin)

    def _bevel_position(
        self,
        anchor: str,
        d: dict,
        thickness: float,
    ) -> tuple[float, float, float]:
        """Calculate weld bevel position."""
        l = d.get("length", 20)
        w = d.get("width", 20)
        h = d.get("height", d.get("thickness", 10))

        if anchor in ("top", "bottom"):
            return (0, 0, h - thickness * 0.6 if anchor == "top" else 0)
        elif anchor in ("front", "back"):
            return (0, -thickness * 0.3 if anchor == "front" else w, 0)
        else:
            return (-thickness * 0.3 if anchor == "left" else l, 0, 0)


# ============================================================================
# Integration helper — merge connection features into part_feature_engine output
# ============================================================================


def merge_connection_ops(
    base_ops: list[dict],
    connection_ops: list[dict],
    body_name: str,
) -> list[dict]:
    """Merge connection feature ops into a part's base ops list.

    Inserts the connection ops (holes, bores, etc.) before the final
    export step, and adds boolean cuts to subtract them from the body.

    Args:
        base_ops: The part's base FreeCAD ops from part_feature_engine.
        connection_ops: The connection feature ops from ConnectionFeatureEngine.
        body_name: The current body name to cut features from.

    Returns:
        Combined ops list with connection features integrated.
    """
    if not connection_ops:
        return base_ops

    # Find the export step (last operation)
    export_idx = len(base_ops)
    for i, op in enumerate(base_ops):
        if op.get("type") == "export_stl":
            export_idx = i
            break

    # Split: before export / export+after
    before_export = base_ops[:export_idx]
    after_export = base_ops[export_idx:]

    # Extract feature object names from connection_ops
    feature_names = []
    create_ops = []
    for op in connection_ops:
        if op.get("type") in ("make_box", "make_cylinder", "cylinder_with_hole"):
            name = op.get("name", "")
            if name:
                feature_names.append(name)
            create_ops.append(op)
        elif op.get("type") == "move":
            create_ops.append(op)

    # Merge: base_before + connection creates + boolean cuts + export
    merged = list(before_export)
    merged.extend(create_ops)

    # Boolean cut each feature from the body
    for fname in feature_names:
        cut_name = f"{fname}_cut_result"
        merged.append({
            "type": "boolean",
            "operation": "cut",
            "object1": body_name,
            "object2": fname,
            "result_name": cut_name,
        })
        merged.append({"type": "delete_object", "object": body_name})
        merged.append({"type": "delete_object", "object": fname})
        body_name = cut_name

    merged.extend(after_export)
    return merged


# ============================================================================
# Assembly-level feature generation — process all joints
# ============================================================================


def generate_assembly_connection_features(
    assembly_parts: list[Part],
    assembly_joints: list[Joint],
) -> dict[str, ConnectionFeatureResult]:
    """Generate connection features for all joints in an assembly.

    For each joint with a ConnectionMethod, generates features on the
    structural part (the one that gets holes/bores).

    Returns:
        dict mapping part_name → ConnectionFeatureResult
    """
    engine = ConnectionFeatureEngine()
    results: dict[str, ConnectionFeatureResult] = {}
    parts_by_name = {p.name: p for p in assembly_parts}

    for joint in assembly_joints:
        if joint.connection is None:
            continue

        parent_part = parts_by_name.get(joint.parent)
        child_part = parts_by_name.get(joint.child)

        if parent_part is None or child_part is None:
            continue

        # Determine which part is structural (gets features) vs functional
        structural = _pick_structural_part(parent_part, child_part)
        anchor = joint.child_anchor if structural == child_part else joint.parent_anchor

        new_result = engine.generate_features(
            structural_part=structural,
            connection=joint.connection,
            anchor=anchor,
        )

        # For revolute joints with bolted connections, add a central
        # bearing bore so the shaft can pass through.  Bolts hold the
        # housing; the bearing bore allows rotation.  Without this, the
        # joint is physically locked by the solid material between bolts.
        if joint.type == "revolute" and joint.connection.type == "bolted":
            d = structural.dimensions
            thickness = ConnectionFeatureEngine._infer_thickness(d, anchor)
            bore_d = 10.0  # fits MR105ZZ bearing (OD=10mm, ID=5mm)
            bore_name = f"{structural.name}_bearing_bore"
            bx, by, bz = engine._anchor_center(anchor, d, thickness)
            new_result.ops.append(ConnectionFeatureEngine._make_hole_cylinder_op(
                bore_name, bore_d / 2, thickness, anchor, d,
            ))
            new_result.ops.append({
                "type": "move",
                "object": bore_name,
                "dx": bx, "dy": by, "dz": bz - 2,
            })
            new_result.features_generated.append(
                f"Bearing bore O{bore_d:.1f}mm (central shaft passage "
                f"for revolute joint)"
            )

        # F16: a structural part that participates in multiple bolted joints
        # (e.g. a base plate with 4 standoffs + a motor) previously had its
        # features generated only for the FIRST joint — every subsequent
        # joint was silently skipped.  Merge features from all connections.
        if structural.name in results:
            existing = results[structural.name]
            existing.ops.extend(new_result.ops)
            existing.fastener_ops.extend(new_result.fastener_ops)
            existing.warnings.extend(new_result.warnings)
            existing.features_generated.extend(new_result.features_generated)
        else:
            results[structural.name] = new_result

    return results


def _pick_structural_part(part_a: Part, part_b: Part) -> Part:
    """Pick which part receives connection features (the structural one).

    Functional parts (motors, sensors, bearings) have fixed dimensions
    and should NOT receive holes.  Structural parts (brackets, plates,
    links) are custom and should receive matching features.

    Naming heuristic: tokens are matched as whole words (split on
    underscore/dash/space) so "base_plate" is not misclassified by an
    accidental substring.  A structural indicator (mount, bracket,
    housing, plate, standoff, link, ...) always wins — a
    "motor_mount_bracket" is structural because it's the bracket, not
    the motor.
    """
    import re

    # Categories that are typically functional (don't drill holes in them)
    functional_cats = {
        "actuator", "sensor", "electronics", "fastener", "controller",
    }
    # Whole-word tokens that mark a part as structural even if its name
    # also references a functional component.
    structural_indicators = {
        "mount", "bracket", "housing", "plate", "standoff", "link",
        "adapter", "holder", "seat", "frame", "support", "arm",
        "post", "tower", "base", "chassis", "rib", "gusset",
    }
    # Whole-word tokens that mark a part as functional.
    functional_keywords = {
        "motor", "servo", "stepper", "nema", "bearing", "encoder",
        "imu", "lidar", "camera", "arduino", "esp32",
        "driver", "board", "pcb",
    }

    def _is_functional(part: Part) -> bool:
        if part.category in functional_cats:
            # Category wins — unless the name has a structural indicator.
            # E.g. category="actuator" but name="motor_mount_bracket" is
            # almost certainly mis-categorized and the bracket should be
            # drilled, not treated as a motor.
            tokens = set(re.split(r"[_\-\s]+", part.name.lower()))
            if tokens & structural_indicators:
                return False
            return True
        tokens = set(re.split(r"[_\-\s]+", part.name.lower()))
        if tokens & structural_indicators:
            return False
        return bool(tokens & functional_keywords)

    if _is_functional(part_a) and not _is_functional(part_b):
        return part_b
    if _is_functional(part_b) and not _is_functional(part_a):
        return part_a
    # Both structural or both functional: default to parent
    return part_a
