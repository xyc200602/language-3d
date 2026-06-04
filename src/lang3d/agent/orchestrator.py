"""OrchestratorAgent - multi-agent orchestration with wave-based parallel execution."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from ..knowledge.mechanics import Assembly
from ..models.router import ModelRouter
from ..tools.base import ToolRegistry
from .assembly_verifier import AssemblyVerifier, AssemblyVerificationResult
from .dag import TaskDAG
from .message_bus import AgentMessage, MessageBus
from .shared_registry import SharedToolRegistry
from .state import Plan, PlanStep, StepStatus, HierarchicalPlan, SubSystem
from .sub_agent import SubAgent, SubAgentResult, SubAgentRole
from .reflector import Reflector
from .planner import Planner

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """Orchestrates complex tasks using multiple sub-agents with parallel execution."""

    def __init__(
        self,
        config: Any,
        router: ModelRouter,
        tools: ToolRegistry,
        planner: Planner,
        verifier: AssemblyVerifier | None = None,
        reflector: Reflector | None = None,
        workspace: str | None = None,
        max_parallel: int = 3,
        max_retries: int = 3,
    ) -> None:
        self.config = config
        self.router = router
        self.tools = SharedToolRegistry(tools)
        self.planner = planner
        self.verifier = verifier or AssemblyVerifier()
        self.reflector = reflector or Reflector(router)
        self.workspace = workspace or "."
        self.max_parallel = max_parallel
        self.max_retries = max_retries

        self.message_bus = MessageBus()
        self._active_agents: dict[str, SubAgent] = {}
        self._results: dict[str, SubAgentResult] = {}
        self._dag: TaskDAG | None = None

        # Callbacks for UI updates
        self._on_sub_agent_update: Callable[[str, str, str], None] | None = None
        self._on_dag_update: Callable[[dict], None] | None = None
        self._on_thinking: Callable[[str], None] | None = None
        self._on_tool_call: Callable[[str, dict], None] | None = None
        self._on_tool_result: Callable[[str, str], None] | None = None

    def on_sub_agent_update(
        self, callback: Callable[[str, str, str], None]
    ) -> None:
        """Set sub-agent status callback: (agent_id, status, step_description)."""
        self._on_sub_agent_update = callback

    def on_dag_update(self, callback: Callable[[dict], None]) -> None:
        """Set DAG update callback."""
        self._on_dag_update = callback

    def on_thinking(self, callback: Callable[[str], None]) -> None:
        self._on_thinking = callback

    def on_tool_call(self, callback: Callable[[str, dict], None]) -> None:
        self._on_tool_call = callback

    def on_tool_result(self, callback: Callable[[str, str], None]) -> None:
        self._on_tool_result = callback

    def should_orchestrate(self, task: str, plan: Plan) -> bool:
        """Heuristic to decide if multi-agent orchestration is beneficial.

        Orchestrates when there are >= 4 steps and >= 3 involve independent
        modeling work.
        """
        if len(plan.steps) < 4:
            return False

        modeling_steps = 0
        for step in plan.steps:
            tools_lower = [t.lower() for t in step.expected_tools]
            if any("fc_" in t or "cad" in t for t in tools_lower):
                modeling_steps += 1

        return modeling_steps >= 3

    def run_task(self, task: str) -> str:
        """Main entry point for orchestrated task execution.

        Runs the async orchestration in a new event loop.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing event loop, use nest_asyncio or thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self._run_orchestrated_async(task))
                    return future.result()
            else:
                return loop.run_until_complete(self._run_orchestrated_async(task))
        except RuntimeError:
            return asyncio.run(self._run_orchestrated_async(task))

    async def _run_orchestrated_async(self, task: str) -> str:
        """Execute a task using multi-agent orchestration."""
        if self._on_thinking:
            self._on_thinking("正在分析任务并创建 DAG 计划...")

        # Phase 1: Create plan
        plan = self.planner.create_plan(task)

        # Phase 2: Build DAG
        self._dag = TaskDAG.from_plan(plan)
        if self._on_dag_update:
            self._on_dag_update(self._dag.to_dict())

        if self._on_thinking:
            groups = self._dag.parallel_groups()
            self._on_thinking(
                f"DAG 计划创建完成：{len(plan.steps)} 个步骤，"
                f"{len(groups)} 个波次"
            )

        # Phase 3: Execute waves
        groups = self._dag.parallel_groups()

        for wave_idx, wave in enumerate(groups):
            if self._on_thinking:
                agent_descs = ", ".join(n.step.description[:30] for n in wave)
                self._on_thinking(
                    f"波次 {wave_idx}: 执行 {len(wave)} 个任务 ({agent_descs})"
                )

            await self._execute_wave(wave)

            if self._on_dag_update and self._dag:
                self._on_dag_update(self._dag.to_dict())

        # Phase 4: Assembly verification
        verification_report = ""
        assembly = self._try_get_assembly(task)
        if assembly and self._dag:
            verification_report = self._run_assembly_verification(assembly)

        # Build final result
        completed = sum(
            1 for n in self._dag.nodes.values()
            if n.step.status == StepStatus.COMPLETED
        ) if self._dag else 0
        total = len(self._dag.nodes) if self._dag else 0
        failed = sum(
            1 for n in self._dag.nodes.values()
            if n.step.status == StepStatus.FAILED
        ) if self._dag else 0

        result = (
            f"编排任务完成：{completed}/{total} 步骤成功"
            + (f"，{failed} 步骤失败" if failed else "")
        )
        if verification_report:
            result += f"\n\n{verification_report}"

        return result

    async def _execute_wave(self, nodes: list) -> dict[str, SubAgentResult]:
        """Execute a wave of nodes in parallel with semaphore limiting."""
        semaphore = asyncio.Semaphore(self.max_parallel)
        results: dict[str, SubAgentResult] = {}

        async def _run_node(node) -> SubAgentResult:
            async with semaphore:
                context = self._collect_context(node)
                return await self._run_node_with_retry(node, context)

        tasks = [_run_node(node) for node in nodes]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for node, result in zip(nodes, completed):
            if isinstance(result, Exception):
                error_result = SubAgentResult(
                    agent_id="orchestrator",
                    step_id=node.step.id,
                    success=False,
                    error=str(result),
                )
                results[node.step.id] = error_result
                if self._dag:
                    self._dag.mark_failed(node.step.id)
            else:
                results[node.step.id] = result
                if result.success and self._dag:
                    self._dag.mark_completed(node.step.id)
                elif self._dag:
                    self._dag.mark_failed(node.step.id)

        return results

    async def _run_node_with_retry(
        self,
        node,
        context: dict[str, Any],
    ) -> SubAgentResult:
        """Run a single node with retry logic."""
        for attempt in range(self.max_retries):
            sub_agent = self._spawn_sub_agent(node)

            if self._on_sub_agent_update:
                self._on_sub_agent_update(
                    sub_agent.agent_id, "running", node.step.description
                )

            result = await sub_agent.execute_async(node.step, context)
            self._results[node.step.id] = result
            self._active_agents[sub_agent.agent_id] = sub_agent

            if result.success:
                if self._on_sub_agent_update:
                    self._on_sub_agent_update(
                        sub_agent.agent_id, "completed", node.step.description
                    )
                return result

            # Failure - try reflection and retry
            if attempt < self.max_retries - 1:
                if self._on_thinking:
                    self._on_thinking(
                        f"步骤 '{node.step.description[:40]}' 失败 "
                        f"(尝试 {attempt + 1}/{self.max_retries})，正在反思..."
                    )

                reflection = self.reflector.reflect(
                    Plan(goal="", steps=[node.step]),
                    node.step,
                    result.error,
                    result.tool_history,
                )

                if self._on_thinking:
                    self._on_thinking(f"反思结果：{reflection[:200]}")

                # Reset step for retry
                node.step.status = StepStatus.PENDING
                node.step.attempts = attempt + 1

        if self._on_sub_agent_update:
            self._on_sub_agent_update(
                sub_agent.agent_id, "failed", node.step.description
            )

        return result

    def _spawn_sub_agent(self, node) -> SubAgent:
        """Create a SubAgent for a DAG node."""
        role = SubAgentRole(node.agent_role) if node.agent_role else SubAgentRole.GENERAL
        sub = SubAgent(
            role=role,
            router=self.router,
            tools=self.tools,
            workspace=self.workspace,
        )

        # Wire up callbacks
        if self._on_tool_call:

            def _tc_cb(aid, name, args):
                self.message_bus.publish(
                    AgentMessage(sender=aid, type="tool_call", payload={"name": name, "args": args})
                )
                self._on_tool_call(name, args)

            sub.on_tool_call(_tc_cb)

        if self._on_tool_result:

            def _tr_cb(aid, name, result):
                self.message_bus.publish(
                    AgentMessage(sender=aid, type="tool_result", payload={"name": name, "result": result})
                )
                self._on_tool_result(name, result)

            sub.on_tool_result(_tr_cb)

        if self._on_thinking:

            def _think_cb(aid, text):
                self._on_thinking(f"[{aid}] {text}")

            sub.on_thinking(_think_cb)

        return sub

    def _collect_context(self, node) -> dict[str, Any]:
        """Collect results from completed dependencies."""
        context: dict[str, Any] = {}
        for dep_id in node.dependencies:
            if dep_id in self._results:
                result = self._results[dep_id]
                # Get step description from DAG
                dep_node = self._dag.get_node(dep_id) if self._dag else None
                desc = dep_node.step.description if dep_node else ""
                context[dep_id] = {
                    "description": desc,
                    "result": result.result,
                    "artifacts": result.artifacts,
                    "success": result.success,
                }
        return context

    def _try_get_assembly(self, task: str) -> Assembly | None:
        """Try to find a matching assembly definition for the task."""
        task_lower = task.lower()
        if "机械臂" in task_lower or "robotic arm" in task_lower:
            from ..knowledge.mechanics import ROBOTIC_ARM_ASSEMBLY
            return ROBOTIC_ARM_ASSEMBLY
        return None

    def _run_assembly_verification(self, assembly: Assembly) -> str:
        """Run assembly verification and return the report."""
        if self._on_thinking:
            self._on_thinking("正在进行装配验证...")

        # Collect parts results from sub-agents
        parts_results: dict[str, dict[str, Any]] = {}
        if self._dag:
            for node in self._dag.nodes.values():
                step_desc = node.step.description.lower()
                for part in assembly.parts:
                    if part.name in step_desc or part.name.replace("_", " ") in step_desc:
                        result = self._results.get(node.step.id)
                        if result:
                            parts_results[part.name] = {
                                "artifacts": result.artifacts,
                                "result": result.result,
                                "success": result.success,
                            }

        result = self.verifier.verify_assembly(assembly, self.workspace, parts_results)
        return AssemblyVerifier.generate_assembly_report(result)

    @property
    def active_agents(self) -> dict[str, SubAgent]:
        """Get currently tracked sub-agents."""
        return self._active_agents

    @property
    def dag(self) -> TaskDAG | None:
        return self._dag


# ---------------------------------------------------------------------------
# PhasedOrchestrator — 4-phase execution for complex robot design (Task 44)
# ---------------------------------------------------------------------------

class PhasedOrchestrator(OrchestratorAgent):
    """Orchestrates complex robot designs through 4 sequential phases.

    Phase 1 (layout):       determine spatial layout, interface constraints,
                            mass budget for each subsystem.
    Phase 2 (part_design):  parallel sub-agents design parts within each
                            subsystem, respecting layout constraints.
    Phase 3 (subsystem_verify): per-subsystem verification (geometry,
                            interfaces).
    Phase 4 (integration):  assembly, interference check, stability,
                            kinematics verification.

    Failure recovery:
    - Subsystem failure → retry only that subsystem (up to max_retries).
    - Integration failure → re-plan layout and re-run from Phase 2.
    """

    PHASE_LAYOUT = "layout"
    PHASE_PART_DESIGN = "part_design"
    PHASE_SUBSYSTEM_VERIFY = "subsystem_verify"
    PHASE_INTEGRATION = "integration_verify"

    def __init__(
        self,
        config: Any,
        router: ModelRouter,
        tools: ToolRegistry,
        planner: Planner,
        hierarchical_plan: HierarchicalPlan,
        verifier: AssemblyVerifier | None = None,
        reflector: Reflector | None = None,
        workspace: str | None = None,
        max_parallel: int = 6,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            config=config,
            router=router,
            tools=tools,
            planner=planner,
            verifier=verifier,
            reflector=reflector,
            workspace=workspace,
            max_parallel=max_parallel,
            max_retries=max_retries,
        )
        self.hierarchical_plan = hierarchical_plan
        self.current_phase: str = ""
        # Layout constraints filled during Phase 1
        self._layout_constraints: dict[str, dict[str, Any]] = {}
        # Per-subsystem results collected during Phase 2
        self._subsystem_results: dict[str, list[SubAgentResult]] = {}

    # ── Public entry point ────────────────────────────────────────

    def run_task(self, task: str) -> str:  # type: ignore[override]
        """Run the 4-phase orchestrated execution."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self._run_phased_async(task))
                    return future.result()
            else:
                return loop.run_until_complete(self._run_phased_async(task))
        except RuntimeError:
            return asyncio.run(self._run_phased_async(task))

    # ── Internal phase orchestration ──────────────────────────────

    async def _run_phased_async(self, task: str) -> str:
        """Execute all 4 phases sequentially."""
        plan = self.hierarchical_plan
        total_subsystems = len(plan.subsystems)

        if self._on_thinking:
            self._on_thinking(
                f"开始分阶段编排：{total_subsystems} 个子系统，"
                f"{plan.total_parts()} 个零件"
            )

        # ── Phase 1: Layout ──────────────────────────────────────
        self.current_phase = self.PHASE_LAYOUT
        if self._on_thinking:
            self._on_thinking("Phase 1/4: 子系统空间布局规划...")

        self._plan_layout_phase(plan)

        if self._on_thinking:
            self._on_thinking(
                f"布局完成：{len(self._layout_constraints)} 个子系统约束已确定"
            )

        # ── Phase 2: Part design (parallel per subsystem) ────────
        self.current_phase = self.PHASE_PART_DESIGN
        if self._on_thinking:
            self._on_thinking("Phase 2/4: 子系统并行零件设计...")

        await self._run_part_design_phase(plan)

        # ── Phase 3: Per-subsystem verification ──────────────────
        self.current_phase = self.PHASE_SUBSYSTEM_VERIFY
        if self._on_thinking:
            self._on_thinking("Phase 3/4: 子系统验证...")

        await self._run_subsystem_verify_phase(plan)

        # ── Phase 4: Integration ─────────────────────────────────
        self.current_phase = self.PHASE_INTEGRATION
        if self._on_thinking:
            self._on_thinking("Phase 4/4: 整机集成验证...")

        integration_ok = await self._run_integration_phase(plan)

        if not integration_ok:
            integration_result = "集成验证失败"
        else:
            integration_result = "集成验证通过"

        # Build summary
        completed, total = plan.progress()
        result = (
            f"分阶段编排完成：{completed}/{total} 步骤成功 "
            f"({total_subsystems} 个子系统)\n{integration_result}"
        )

        if self._on_thinking:
            self._on_thinking(result)

        return result

    # ── Phase 1: Layout ────────────────────────────────────────

    def _plan_layout_phase(self, plan: HierarchicalPlan) -> None:
        """Determine spatial layout and interface constraints for each subsystem.

        For each subsystem, derive:
        - spatial_bounds: approximate bounding box [x,y,z,w,d,h]
        - interface_points: where it connects to other subsystems
        - mass_budget: estimated mass allocation
        """
        for ss in plan.subsystems:
            constraints: dict[str, Any] = {
                "subsystem": ss.name,
                "description": ss.description,
                "parts": ss.parts,
                "mirror_of": ss.mirror_of,
                "instance_count": ss.instance_count,
                "interface_points": ss.interface_points,
            }

            # Derive dependencies from system_dependencies
            deps_for_this = [
                {"source": d.source, "reason": d.reason}
                for d in plan.system_dependencies
                if d.target == ss.name
            ]
            if deps_for_this:
                constraints["depends_on"] = deps_for_this

            self._layout_constraints[ss.name] = constraints

    # ── Phase 2: Parallel part design ──────────────────────────

    async def _run_part_design_phase(self, plan: HierarchicalPlan) -> None:
        """Run each subsystem's part design steps in parallel sub-agents.

        Independent subsystems run concurrently. Subsystems that depend on
        others wait for their dependencies to finish first.
        """
        # Build dependency graph: determine execution order
        completed_subsystems: set[str] = set()
        remaining = list(plan.subsystems)

        max_rounds = len(remaining) + 1  # safety bound
        round_num = 0

        while remaining and round_num < max_rounds:
            round_num += 1
            # Find subsystems whose dependencies are all completed
            ready: list[SubSystem] = []
            still_remaining: list[SubSystem] = []

            for ss in remaining:
                deps_for_ss = {
                    d.source
                    for d in plan.system_dependencies
                    if d.target == ss.name
                }
                if deps_for_ss.issubset(completed_subsystems):
                    ready.append(ss)
                else:
                    still_remaining.append(ss)

            if not ready:
                # All remaining have unresolved deps — should not happen
                # with valid input, but break to avoid infinite loop
                logger.warning(
                    "Deadlock in subsystem scheduling: %s",
                    [s.name for s in still_remaining],
                )
                break

            remaining = still_remaining

            # Run ready subsystems in parallel
            await self._run_subsystems_parallel(ready, plan, completed_subsystems)

            for ss in ready:
                completed_subsystems.add(ss.name)

    async def _run_subsystems_parallel(
        self,
        subsystems: list[SubSystem],
        plan: HierarchicalPlan,
        completed_subsystems: set[str],
    ) -> None:
        """Execute multiple subsystems' part-design steps in parallel."""
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _design_subsystem(ss: SubSystem) -> None:
            async with semaphore:
                await self._design_single_subsystem(ss, plan, completed_subsystems)

        tasks = [_design_subsystem(ss) for ss in subsystems]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _design_single_subsystem(
        self,
        ss: SubSystem,
        plan: HierarchicalPlan,
        completed_subsystems: set[str],
    ) -> None:
        """Design all parts for a single subsystem with retry."""
        # Collect context from completed dependent subsystems
        dep_context = self._collect_subsystem_context(ss, plan, completed_subsystems)

        # Get layout constraints for this subsystem
        layout_ctx = self._layout_constraints.get(ss.name, {})

        # Build design context for the executor
        design_ctx: dict[str, Any] = {
            "subsystem": ss.name,
            "parts_to_design": ss.parts,
            "mirror_of": ss.mirror_of,
            "instance_count": ss.instance_count,
        }
        if layout_ctx.get("interface_points"):
            design_ctx["interface_points"] = layout_ctx["interface_points"]
        if dep_context:
            design_ctx["predecessor_results"] = dep_context

        ss.status = StepStatus.IN_PROGRESS
        results: list[SubAgentResult] = []

        for step in ss.steps:
            if step.status == StepStatus.COMPLETED:
                results.append(SubAgentResult(
                    agent_id="skip", step_id=step.id, success=True,
                    result=step.result,
                ))
                continue

            step_result = await self._run_step_with_retry(
                step, design_ctx
            )
            results.append(step_result)

        self._subsystem_results[ss.name] = results

        # Update subsystem status based on step results
        all_ok = all(r.success for r in results)
        ss.status = StepStatus.COMPLETED if all_ok else StepStatus.FAILED

    async def _run_step_with_retry(
        self,
        step: PlanStep,
        design_ctx: dict[str, Any],
    ) -> SubAgentResult:
        """Run a single step with retry logic and design context injection."""
        for attempt in range(self.max_retries):
            sub = SubAgent(
                role=SubAgentRole.MODELING,
                router=self.router,
                tools=self.tools,
                workspace=self.workspace,
            )

            # Wire callbacks
            self._wire_sub_agent_callbacks(sub)

            # Inject design context into the step description
            original_desc = step.description
            context_payload: dict[str, Any] = {"design_context": design_ctx}

            result = await sub.execute_async(step, context_payload)

            self._results[step.id] = result
            self._active_agents[sub.agent_id] = sub

            if result.success:
                return result

            # Failure — reflect and retry
            if attempt < self.max_retries - 1:
                if self._on_thinking:
                    self._on_thinking(
                        f"[{design_ctx.get('subsystem', '?')}] 步骤 "
                        f"'{step.description[:40]}' 失败 "
                        f"(尝试 {attempt + 1}/{self.max_retries})，正在反思..."
                    )

                reflection = self.reflector.reflect(
                    Plan(goal="", steps=[step]),
                    step,
                    result.error,
                    result.tool_history,
                )

                if self._on_thinking:
                    self._on_thinking(f"反思结果：{reflection[:200]}")

                step.status = StepStatus.PENDING
                step.attempts = attempt + 1

        return result

    # ── Phase 3: Per-subsystem verification ─────────────────────

    async def _run_subsystem_verify_phase(self, plan: HierarchicalPlan) -> None:
        """Verify each subsystem's parts independently."""
        for ss in plan.subsystems:
            if ss.status != StepStatus.COMPLETED:
                continue  # skip failed subsystems

            if self._on_thinking:
                self._on_thinking(f"验证子系统 '{ss.name}'...")

            # Create a verification step
            verify_step = PlanStep(
                description=f"验证子系统 '{ss.name}' 的零件：{', '.join(ss.parts)}",
                expected_tools=["cad_verify", "vlm_analyze"],
                verification=f"子系统 {ss.name} 所有零件几何正确",
            )

            ss_results = self._subsystem_results.get(ss.name, [])
            artifacts = []
            for r in ss_results:
                artifacts.extend(r.artifacts)

            design_ctx: dict[str, Any] = {
                "subsystem": ss.name,
                "verification_type": "subsystem",
                "artifacts": artifacts,
            }

            verify_result = await self._run_step_with_retry(
                verify_step, design_ctx
            )

            if not verify_result.success:
                ss.status = StepStatus.FAILED
                if self._on_thinking:
                    self._on_thinking(
                        f"子系统 '{ss.name}' 验证失败: {verify_result.error[:200]}"
                    )

    # ── Phase 4: Integration ───────────────────────────────────

    async def _run_integration_phase(self, plan: HierarchicalPlan) -> bool:
        """Run integration steps: assembly, interference, stability checks."""
        if not plan.integration_steps:
            return True

        # Collect all artifacts from completed subsystems
        all_artifacts: list[str] = []
        for ss in plan.subsystems:
            if ss.status == StepStatus.COMPLETED:
                for r in self._subsystem_results.get(ss.name, []):
                    all_artifacts.extend(r.artifacts)

        design_ctx: dict[str, Any] = {
            "verification_type": "integration",
            "subsystems": [
                {"name": ss.name, "status": ss.status.value, "parts": ss.parts}
                for ss in plan.subsystems
            ],
            "artifacts": all_artifacts,
        }

        for step in plan.integration_steps:
            if self._on_thinking:
                self._on_thinking(f"集成步骤: {step.description[:60]}...")

            result = await self._run_step_with_retry(step, design_ctx)

            if not result.success:
                if self._on_thinking:
                    self._on_thinking(
                        f"集成步骤失败: {result.error[:200]}"
                    )
                return False

        return True

    # ── Helpers ──────────────────────────────────────────────────

    def _collect_subsystem_context(
        self,
        ss: SubSystem,
        plan: HierarchicalPlan,
        completed_subsystems: set[str],
    ) -> dict[str, Any]:
        """Collect results from subsystems this one depends on."""
        context: dict[str, Any] = {}
        for dep in plan.system_dependencies:
            if dep.target != ss.name:
                continue
            if dep.source not in completed_subsystems:
                continue
            source_results = self._subsystem_results.get(dep.source, [])
            artifacts = []
            for r in source_results:
                artifacts.extend(r.artifacts)
            context[dep.source] = {
                "artifacts": artifacts,
                "reason": dep.reason,
            }
        return context

    def _wire_sub_agent_callbacks(self, sub: SubAgent) -> None:
        """Attach message-bus and UI callbacks to a sub-agent."""
        if self._on_tool_call:
            def _tc_cb(aid, name, args):
                self.message_bus.publish(
                    AgentMessage(sender=aid, type="tool_call",
                                 payload={"name": name, "args": args})
                )
                self._on_tool_call(name, args)
            sub.on_tool_call(_tc_cb)

        if self._on_tool_result:
            def _tr_cb(aid, name, result):
                self.message_bus.publish(
                    AgentMessage(sender=aid, type="tool_result",
                                 payload={"name": name, "result": result})
                )
                self._on_tool_result(name, result)
            sub.on_tool_result(_tr_cb)

        if self._on_thinking:
            def _think_cb(aid, text):
                self._on_thinking(f"[{aid}] {text}")
            sub.on_thinking(_think_cb)
