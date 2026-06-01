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
from .state import Plan, PlanStep, StepStatus
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
