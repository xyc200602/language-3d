"""Tool registration registry for auto-discovery.

Each entry maps a module name to its register function name and optional
extra keyword arguments that the caller can supply.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Declarative tool registry.
# key   = dotted module name relative to lang3d.tools
# value = (register_function_name, {kwarg_name: config_path, ...})
#
# config_path is a dot-separated path into the Agent config / runtime
# objects passed to discover_and_register().  For example:
#   "router"       → the router object
#   "screenshot_dir" → config.agent.screenshot_dir
#
# Special keys:
#   "_factory"  → module has factory functions, not register_*_tools
#   "_priority" → registration order (lower = earlier)

_TOOL_MODULES: list[dict[str, Any]] = [
    # --- Built-in (always available) ---
    {"module": "file_ops", "fn": "register_file_tools"},
    {"module": "bash", "fn": "register_bash_tools"},

    # --- Core optional (screen, vlm, cad, python, gui) ---
    {"module": "screen", "fn": "register_screen_tools",
     "kwargs": {"screenshot_dir": "screenshot_dir"}},
    {"module": "vlm", "fn": "register_vlm_tools",
     "kwargs": {"router": "router", "screenshot_dir": "screenshot_dir"}},
    {"module": "cad_utils", "fn": "register_cad_utils"},
    {"module": "python_exec", "fn": "register_python_tools"},
    {"module": "gui_action", "fn": "register_gui_action_tools",
     "kwargs": {"screenshot_dir": "screenshot_dir"}},
    {"module": "fc_menu", "fn": "register_fc_menu_tools", "special": "fc_menu"},

    # --- CAD backends ---
    {"module": "solidworks", "fn": "register_solidworks_tools"},
    {"module": "freecad", "fn": "register_freecad_tools"},

    # --- Simulation & analysis ---
    {"module": "simulation", "fn": "register_simulation_tools",
     "kwargs": {"router": "router", "screenshot_dir": "screenshot_dir"}},
    {"module": "cfd", "fn": "register_cfd_tools",
     "kwargs": {"router": "router", "screenshot_dir": "screenshot_dir"}},
    {"module": "motion", "fn": "register_motion_tools"},
    {"module": "motion_collision", "fn": "register_motion_collision_tools"},

    # --- Parts & fasteners ---
    {"module": "part_library", "fn": "register_part_library_tools"},
    {"module": "fastener_model", "fn": "register_fastener_tools"},
    {"module": "part_recommend_tool", "fn": "register_part_recommend_tools"},
    {"module": "part_feature_engine", "fn": "register_part_feature_tools"},

    # --- Assembly ---
    {"module": "assembly_solver", "fn": "register_assembly_solver_tools"},
    {"module": "mating_constraint", "fn": "constraint_solve_tool_factory",
     "factory": True},
    {"module": "assembly_matcher", "fn": "assembly_match_tool_factory",
     "factory": True},
    {"module": "tolerance_analysis", "fn": "tolerance_analysis_tool_factory",
     "factory": True},
    {"module": "assembly_vlm", "fn": "AssemblyVLMSolveTool",
     "factory": True, "register_instance": True},
    {"module": "assembly_generator", "fn": "register_assembly_generator_tools"},
    {"module": "assembly_template_tool", "fn": "register_assembly_template_tools"},
    {"module": "assembly_doc", "fn": "register_assembly_doc_tools"},

    # --- Design tools ---
    {"module": "ik_solver", "fn": "register_ik_tools"},
    {"module": "collision", "fn": "register_collision_tools"},
    {"module": "mesh_collision", "fn": "register_mesh_collision_tools"},
    {"module": "workspace", "fn": "register_workspace_tools"},
    {"module": "code_gen", "fn": "register_code_gen_tools"},
    {"module": "bom_gen", "fn": "register_bom_tools"},
    {"module": "scheme_compare", "fn": "register_scheme_tools"},

    # --- Manufacturing ---
    {"module": "slicing", "fn": "register_slicing_tools",
     "kwargs": {"router": "router", "screenshot_dir": "screenshot_dir"}},
    {"module": "print_optimize", "fn": "register_print_optimize_tools"},
    {"module": "quality", "fn": "register_quality_tools"},
    {"module": "production_check", "fn": "register_production_tools"},

    # --- Actuators & power ---
    {"module": "actuator_tools", "fn": "register_actuator_tools"},
    {"module": "power_budget", "fn": "register_power_budget_tools"},
    {"module": "drive_train", "fn": "register_drive_train_tools"},

    # --- Mobile & stability ---
    {"module": "mobile_design", "fn": "register_mobile_design_tools"},
    {"module": "stability", "fn": "register_stability_tools"},

    # --- Export & documentation ---
    {"module": "urdf_export", "fn": "register_urdf_tools"},
    {"module": "cable_routing", "fn": "register_cable_routing_tools"},
    {"module": "export_package", "fn": "register_export_package_tools"},
    {"module": "mass_properties", "fn": "register_mass_properties_tools"},

    # --- Iteration ---
    {"module": "iteration", "fn": "register_iteration_tools"},
]


def discover_and_register(
    registry: Any,
    *,
    router: Any = None,
    screenshot_dir: str = "",
    fc_menu_deps: dict[str, Any] | None = None,
) -> list[str]:
    """Register all available tool modules.

    Returns list of successfully registered module names.
    """
    registered: list[str] = []

    for entry in _TOOL_MODULES:
        module_name = entry["module"]
        fn_name = entry["fn"]
        try:
            mod = importlib.import_module(f".{module_name}", package=__name__)
        except ImportError:
            continue

        try:
            obj = getattr(mod, fn_name, None)
            if obj is None:
                continue

            # Build kwargs from available runtime objects
            kw: dict[str, Any] = {}
            for kw_name, _source in entry.get("kwargs", {}).items():
                if kw_name == "router" and router is not None:
                    kw["router"] = router
                elif kw_name == "screenshot_dir" and screenshot_dir:
                    kw["screenshot_dir"] = screenshot_dir

            # Handle special registration patterns
            if entry.get("special") == "fc_menu":
                # fc_menu needs pre-created tool instances
                if fc_menu_deps:
                    obj(registry, **fc_menu_deps)
                registered.append(module_name)
                continue

            if entry.get("register_instance"):
                # Factory returns a class to instantiate and register
                registry.register(obj())
                registered.append(module_name)
                continue

            if entry.get("factory"):
                # Factory returns (definition, class)
                _def, _cls = obj()
                registry.register(_cls())
                registered.append(module_name)
                continue

            # Standard register function
            if kw:
                obj(registry, **kw)
            else:
                obj(registry)
            registered.append(module_name)

        except Exception as e:
            logger.warning("Failed to register %s tools: %s", module_name, e)

    return registered
