"""Tests for the expert agent roles (architect/solver/cad/verifier/fixer).

Validates the multi-agent Step 1 additions:
- SubAgentRole enum has the new expert roles
- Each role returns a non-trivial system prompt
- get_definitions_for_role scopes tools correctly per role
- DAG from_plan infers the right role from expected_tools

These are unit tests (no LLM calls, no FreeCAD) — they verify the
plumbing, not the end-to-end pipeline.
"""

from __future__ import annotations

import pytest

from lang3d.agent.sub_agent import (
    SubAgent,
    SubAgentRole,
    _ROLE_PROMPTS,
    _EXPERT_ROLES,
)
from lang3d.agent.dag import TaskDAG
from lang3d.agent.state import PlanStep
from lang3d.tools.base import (
    ToolRegistry,
    ROLE_TOOL_CATEGORIES,
    TOOL_CATEGORIES,
)


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------


class TestSubAgentRoleEnum:
    def test_expert_roles_exist(self):
        """The 5 expert pipeline roles must be present in the enum."""
        assert SubAgentRole.ARCHITECT
        assert SubAgentRole.SOLVER
        assert SubAgentRole.CAD_ENGINEER
        assert SubAgentRole.FIXER
        # VERIFICATION already existed pre-2026-06-22
        assert SubAgentRole.VERIFICATION

    def test_expert_roles_set(self):
        """_EXPERT_ROLES must list exactly the expert roles.

        5 core pipeline roles + CHASSIS_ARCHITECT (added for Task-Driven
        Co-Design wheeled-base support)."""
        expected = {
            SubAgentRole.ARCHITECT,
            SubAgentRole.SOLVER,
            SubAgentRole.CAD_ENGINEER,
            SubAgentRole.VERIFICATION,
            SubAgentRole.FIXER,
            SubAgentRole.CHASSIS_ARCHITECT,
        }
        assert _EXPERT_ROLES == expected

    def test_legacy_roles_not_in_expert_set(self):
        """Legacy roles must NOT be in _EXPERT_ROLES (they use step-type filtering)."""
        assert SubAgentRole.MODELING not in _EXPERT_ROLES
        assert SubAgentRole.VISION not in _EXPERT_ROLES
        assert SubAgentRole.GENERAL not in _EXPERT_ROLES


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


class TestRolePrompts:
    @pytest.mark.parametrize("role", [
        SubAgentRole.ARCHITECT,
        SubAgentRole.SOLVER,
        SubAgentRole.CAD_ENGINEER,
        SubAgentRole.VERIFICATION,
        SubAgentRole.FIXER,
    ])
    def test_expert_prompt_exists_and_is_substantial(self, role):
        """Each expert role must have a non-trivial system prompt."""
        prompt = _ROLE_PROMPTS.get(role)
        assert prompt is not None, f"No prompt for {role}"
        assert len(prompt) > 50, f"Prompt for {role} is too short"
        # Must mention the role's domain
        role_kw = {
            SubAgentRole.ARCHITECT: ("装配" , "assembly"),
            SubAgentRole.SOLVER: ("求解", "位置"),
            SubAgentRole.CAD_ENGINEER: ("FreeCAD", "STL"),
            SubAgentRole.VERIFICATION: ("验证", "仲裁"),
            SubAgentRole.FIXER: ("路由", "修复"),
        }
        kws = role_kw[role]
        assert any(kw in prompt for kw in kws), (
            f"Prompt for {role} must mention one of {kws}"
        )

    def test_subagent_returns_role_prompt(self):
        """SubAgent.get_system_prompt must return the role's prompt."""
        for role in [SubAgentRole.ARCHITECT, SubAgentRole.FIXER]:
            agent = SubAgent(role=role)
            prompt = agent.get_system_prompt()
            assert prompt == _ROLE_PROMPTS[role]


# ---------------------------------------------------------------------------
# Tool whitelisting
# ---------------------------------------------------------------------------


class TestRoleToolWhitelist:
    """Verify get_definitions_for_role scopes tools per role."""

    def _build_registry_with_mock_tools(self):
        """Build a registry with mock tools matching real TOOL_CATEGORIES."""
        from lang3d.tools.base import Tool, ToolDefinition

        class MockTool(Tool):
            def __init__(self, name):
                self.name = name
                self.description = f"mock {name}"

            def get_definition(self):
                return ToolDefinition(
                    name=self.name,
                    description=self.description,
                    parameters=[],
                )

            def execute(self, **kwargs):
                return "mock"

        reg = ToolRegistry()
        # Use names that match TOOL_CATEGORIES prefixes + ROLE_EXTRA_TOOLS
        for name in [
            "fc_batch", "fc_open_gui",                          # freecad
            "assembly_solve", "assembly_vlm_solve",             # assembly (category)
            "assembly_template_search", "assembly_generator",   # ROLE_EXTRA_TOOLS for architect
            "cad_verify", "vlm_analyze",                        # vlm
            "read_file",                                        # file_ops
            "part_recommend",                                   # part_library
        ]:
            reg._tools[name] = MockTool(name)
        return reg

    def test_role_tool_categories_defined(self):
        """ROLE_TOOL_CATEGORIES must have entries for all expert roles."""
        for role in ["architect", "solver", "cad", "verifier", "fixer"]:
            assert role in ROLE_TOOL_CATEGORIES, f"Missing role {role}"
            assert ROLE_TOOL_CATEGORIES[role] is not None, (
                f"Role {role} has None (should have a whitelist)"
            )

    def test_general_role_returns_all(self):
        """General role must return ALL tools (None = no filtering)."""
        reg = self._build_registry_with_mock_tools()
        defs = reg.get_definitions_for_role("general")
        assert len(defs) == len(reg._tools)

    def test_unknown_role_returns_all(self):
        """Unknown role string must return all tools (safe default)."""
        reg = self._build_registry_with_mock_tools()
        defs = reg.get_definitions_for_role("nonexistent_role")
        assert len(defs) == len(reg._tools)

    def test_architect_cannot_see_freecad_tools(self):
        """Architect should NOT see fc_* tools (that's CAD's job)."""
        reg = self._build_registry_with_mock_tools()
        defs = reg.get_definitions_for_role("architect")
        names = {d.name for d in defs}
        assert "assembly_template_search" in names, (
            "Architect must see assembly tools"
        )
        assert "fc_batch" not in names, (
            "Architect must NOT see FreeCAD tools"
        )

    def test_cad_cannot_see_assembly_tools(self):
        """CAD Engineer should see fc_* tools but NOT assembly tools."""
        reg = self._build_registry_with_mock_tools()
        defs = reg.get_definitions_for_role("cad")
        names = {d.name for d in defs}
        assert "fc_batch" in names, "CAD must see FreeCAD tools"
        assert "assembly_template_search" not in names, (
            "CAD must NOT see assembly template tools"
        )

    def test_verifier_sees_vlm_but_not_freecad(self):
        """Verifier should see VLM tools but NOT FreeCAD tools."""
        reg = self._build_registry_with_mock_tools()
        defs = reg.get_definitions_for_role("verifier")
        names = {d.name for d in defs}
        assert "cad_verify" in names or "vlm_analyze" in names, (
            "Verifier must see VLM tools"
        )
        assert "fc_batch" not in names, (
            "Verifier must NOT see FreeCAD tools"
        )


# ---------------------------------------------------------------------------
# DAG role inference
# ---------------------------------------------------------------------------


class TestDAGRoleInference:
    """Verify from_plan infers the right expert role from expected_tools."""

    def _make_plan_step(self, step_id, tools, deps=None):
        return PlanStep(
            id=step_id,
            description=f"Step using {tools}",
            expected_tools=tools,
            dependencies=deps or [],
        )

    def test_architect_inferred(self):
        from lang3d.agent.state import Plan
        step = self._make_plan_step("s1", ["assembly_template_search"])
        plan = Plan(goal="test", steps=[step])
        dag = TaskDAG.from_plan(plan)
        assert dag._nodes["s1"].agent_role == "architect"

    def test_solver_inferred(self):
        from lang3d.agent.state import Plan
        step = self._make_plan_step("s1", ["assembly_solver"])
        plan = Plan(goal="test", steps=[step])
        dag = TaskDAG.from_plan(plan)
        assert dag._nodes["s1"].agent_role == "solver"

    def test_cad_inferred(self):
        from lang3d.agent.state import Plan
        step = self._make_plan_step("s1", ["fc_batch", "fc_script"])
        plan = Plan(goal="test", steps=[step])
        dag = TaskDAG.from_plan(plan)
        assert dag._nodes["s1"].agent_role == "cad"

    def test_verifier_inferred(self):
        from lang3d.agent.state import Plan
        step = self._make_plan_step("s1", ["cad_verify"])
        plan = Plan(goal="test", steps=[step])
        dag = TaskDAG.from_plan(plan)
        assert dag._nodes["s1"].agent_role == "verifier"

    def test_fixer_inferred(self):
        from lang3d.agent.state import Plan
        step = self._make_plan_step("s1", ["modify_part"])
        plan = Plan(goal="test", steps=[step])
        dag = TaskDAG.from_plan(plan)
        assert dag._nodes["s1"].agent_role == "fixer"

    def test_general_fallback(self):
        from lang3d.agent.state import Plan
        step = self._make_plan_step("s1", ["read_file"])
        plan = Plan(goal="test", steps=[step])
        dag = TaskDAG.from_plan(plan)
        assert dag._nodes["s1"].agent_role == "general"


# ---------------------------------------------------------------------------
# Pipeline StageAgent integration (the gap these tests close)
# ---------------------------------------------------------------------------
#
# Before this change, AssemblyPipeline's stages (run_architect, run_solver,
# ...) were plain methods with no role identity — the "expert agents" existed
# in name only. StageAgent now gives each stage a SubAgentRole, an expert
# system prompt, and a tool-whitelist. These tests pin that wiring so a
# future refactor can't silently drop the role scoping again.


class TestPipelineStageAgents:
    """Verify each pipeline stage is backed by a real StageAgent."""

    @pytest.fixture
    def pipeline(self):
        from lang3d.agent.pipeline import AssemblyPipeline, PipelineContext
        ctx = PipelineContext(description="4自由度机械臂")
        return AssemblyPipeline(ctx)

    def test_all_five_stages_have_agents(self, pipeline):
        """Every stage must carry a StageAgent with the correct role."""
        from lang3d.agent.pipeline import StageAgent
        agents = [
            pipeline.architect_agent, pipeline.solver_agent,
            pipeline.cad_agent, pipeline.verifier_agent, pipeline.fixer_agent,
        ]
        assert all(isinstance(a, StageAgent) for a in agents)
        roles = [a.role for a in agents]
        assert SubAgentRole.ARCHITECT in roles
        assert SubAgentRole.SOLVER in roles
        assert SubAgentRole.CAD_ENGINEER in roles
        assert SubAgentRole.VERIFICATION in roles
        assert SubAgentRole.FIXER in roles

    @pytest.mark.parametrize("stage_attr,expected_role", [
        ("architect_agent", SubAgentRole.ARCHITECT),
        ("solver_agent", SubAgentRole.SOLVER),
        ("cad_agent", SubAgentRole.CAD_ENGINEER),
        ("verifier_agent", SubAgentRole.VERIFICATION),
        ("fixer_agent", SubAgentRole.FIXER),
    ])
    def test_stage_role_identity(self, pipeline, stage_attr, expected_role):
        agent = getattr(pipeline, stage_attr)
        assert agent.role is expected_role

    def test_each_stage_has_substantial_role_prompt(self, pipeline):
        """The role prompt is what frames the LLM — it must be non-trivial."""
        for agent in [pipeline.architect_agent, pipeline.solver_agent,
                      pipeline.cad_agent, pipeline.verifier_agent,
                      pipeline.fixer_agent]:
            assert len(agent.role_prompt) > 50, (
                f"{agent.stage_name} role prompt too short"
            )

    def test_system_prompt_combines_role_and_domain(self, pipeline):
        """system_prompt(base) must include BOTH the role persona and the
        domain rules — neither should be dropped."""
        sp = pipeline.architect_agent.system_prompt("DOMAIN_RULES_XYZ")
        assert "DOMAIN_RULES_XYZ" in sp  # domain base preserved
        assert "架构师" in sp or "architect" in sp.lower()  # role persona prepended

    def test_system_prompt_without_base_is_just_role(self, pipeline):
        sp = pipeline.solver_agent.system_prompt()
        assert "求解" in sp or "solver" in sp.lower()

    def test_architect_whitelist_excludes_freecad(self, pipeline):
        """Architect must not see FreeCAD modeling tools — that is CAD's job.
        This is the core separation-of-concerns guarantee."""
        wl = pipeline.architect_agent.allowed_tools
        assert wl is not None
        assert "assembly" in wl          # its own category
        assert "actuator" in wl
        # CAD/vision-specific categories must be absent
        assert "freecad" not in wl
        assert "vlm" not in wl
        assert "screen" not in wl

    def test_cad_whitelist_excludes_assembly_tools(self, pipeline):
        """CAD Engineer must not see assembly-design tools — that is the
        Architect's job. Prevents a CAD stage from rewriting topology."""
        wl = pipeline.cad_agent.allowed_tools
        assert wl is not None
        assert "freecad" in wl
        assert "vlm" in wl
        # The architect's assembly_generator must NOT be in CAD's scope
        assert "assembly_generator" not in wl

    def test_verifier_whitelist_is_vision_only(self, pipeline):
        """Verifier judges via VLM + screenshots, not by editing geometry."""
        wl = pipeline.verifier_agent.allowed_tools
        assert wl is not None
        assert "vlm" in wl
        assert "freecad" not in wl
        assert "assembly" not in wl

    def test_fixer_can_route_but_not_model(self, pipeline):
        """Fixer decides routing + applies targeted assembly fixes, but
        cannot do FreeCAD modeling."""
        wl = pipeline.fixer_agent.allowed_tools
        assert wl is not None
        assert "assembly" in wl
        assert "freecad" not in wl
