"""Task DAG - dependency graph with parallel grouping and topological sort."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from .state import Plan, PlanStep, StepStatus


@dataclass
class DAGNode:
    """A node in the task DAG."""

    step: PlanStep
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    agent_role: str = "general"


class TaskDAG:
    """Directed acyclic graph for task dependencies with parallel grouping."""

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}

    def add_step(
        self,
        step: PlanStep,
        dependencies: list[str] | None = None,
        agent_role: str = "general",
    ) -> DAGNode:
        """Add a step to the DAG."""
        deps = dependencies or []

        # Validate dependencies — log unknown IDs
        known_ids = set(self._nodes.keys())
        unknown = [d for d in deps if d not in known_ids]
        if unknown:
            _logger = logging.getLogger(__name__)
            _logger.warning(
                "Step '%s' references unknown dependencies: %s", step.id, unknown
            )
            deps = [d for d in deps if d in known_ids]

        node = DAGNode(step=step, dependencies=deps, agent_role=agent_role)
        self._nodes[step.id] = node

        # Update dependents of upstream nodes
        for dep_id in deps:
            if dep_id in self._nodes:
                self._nodes[dep_id].dependents.append(step.id)

        return node

    def get_node(self, step_id: str) -> DAGNode | None:
        """Get a DAG node by step ID."""
        return self._nodes.get(step_id)

    def get_ready_steps(self) -> list[DAGNode]:
        """Get all steps whose dependencies are all completed."""
        ready = []
        for node in self._nodes.values():
            if node.step.status != StepStatus.PENDING:
                continue
            if not node.dependencies:
                ready.append(node)
                continue
            all_deps_done = all(
                self._nodes.get(dep_id) is not None
                and self._nodes[dep_id].step.status == StepStatus.COMPLETED
                for dep_id in node.dependencies
            )
            if all_deps_done:
                ready.append(node)
        return ready

    def mark_completed(self, step_id: str) -> list[str]:
        """Mark a step as completed. Returns list of newly unlocked step IDs."""
        node = self._nodes.get(step_id)
        if node is None:
            return []
        node.step.status = StepStatus.COMPLETED

        unlocked = []
        for dependent_id in node.dependents:
            dep_node = self._nodes.get(dependent_id)
            if dep_node is None:
                continue
            if dep_node.step.status != StepStatus.PENDING:
                continue
            all_deps_done = all(
                self._nodes.get(d) is not None
                and self._nodes[d].step.status == StepStatus.COMPLETED
                for d in dep_node.dependencies
            )
            if all_deps_done:
                unlocked.append(dependent_id)
        return unlocked

    def mark_failed(self, step_id: str) -> list[str]:
        """Mark a step as failed and cascade to mark all dependents as SKIPPED.

        Returns list of skipped step IDs.
        """
        node = self._nodes.get(step_id)
        if node is None:
            return []
        node.step.status = StepStatus.FAILED

        skipped: list[str] = []
        to_visit = list(node.dependents)
        visited: set[str] = set()
        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            current_node = self._nodes.get(current_id)
            if current_node is None:
                continue
            current_node.step.status = StepStatus.SKIPPED
            skipped.append(current_id)
            to_visit.extend(current_node.dependents)
        return skipped

    def parallel_groups(self) -> list[list[DAGNode]]:
        """Group nodes into parallel execution waves.

        Returns a list of waves, where each wave is a list of nodes
        that can be executed in parallel.
        """
        if not self._nodes:
            return []

        # Check for cycles
        self._detect_cycles()

        # Topological sort into levels using Kahn's algorithm
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for node in self._nodes.values():
            for dep_id in node.dependencies:
                if dep_id in in_degree:
                    pass  # dep_id is a valid dependency
            # Count actual dependencies present in the DAG
            in_degree[node.step.id] = len(
                [d for d in node.dependencies if d in self._nodes]
            )

        groups: list[list[DAGNode]] = []
        remaining = dict(in_degree)

        while remaining:
            # Find all nodes with in_degree 0
            ready = [
                nid for nid, deg in remaining.items() if deg == 0
            ]
            if not ready:
                # Should not happen if cycle check passed
                break

            group = [self._nodes[nid] for nid in ready]
            groups.append(group)

            # Remove these nodes and update in-degrees
            for nid in ready:
                del remaining[nid]
                node = self._nodes[nid]
                for dep_id in node.dependents:
                    if dep_id in remaining:
                        remaining[dep_id] -= 1

        return groups

    def _detect_cycles(self) -> None:
        """Detect cycles in the DAG. Raises ValueError if found."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self._nodes}

        def dfs(node_id: str) -> bool:
            color[node_id] = GRAY
            node = self._nodes[node_id]
            for dep_id in node.dependencies:
                if dep_id not in color:
                    raise ValueError(
                        f"DAG node '{node_id}' depends on unknown node '{dep_id}'"
                    )
                if color[dep_id] == GRAY:
                    return True  # Cycle detected
                if color[dep_id] == WHITE and dfs(dep_id):
                    return True
            color[node_id] = BLACK
            return False

        for nid in self._nodes:
            if color[nid] == WHITE:
                if dfs(nid):
                    raise ValueError("Cycle detected in task DAG")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the DAG for the web panel."""
        return {
            "nodes": [
                {
                    "id": node.step.id,
                    "description": node.step.description,
                    "status": node.step.status.value,
                    "dependencies": node.dependencies,
                    "dependents": node.dependents,
                    "agent_role": node.agent_role,
                    "assigned_agent": node.step.assigned_agent,
                }
                for node in self._nodes.values()
            ],
            "waves": [
                [
                    {
                        "id": node.step.id,
                        "description": node.step.description,
                        "status": node.step.status.value,
                    }
                    for node in wave
                ]
                for wave in self.parallel_groups()
            ],
        }

    @property
    def nodes(self) -> dict[str, DAGNode]:
        return self._nodes

    @classmethod
    def from_plan(
        cls,
        plan: Plan,
        assembly: Any | None = None,
    ) -> TaskDAG:
        """Build a TaskDAG from a Plan, optionally using Assembly joints for dependencies."""
        dag = cls()

        # Build a name-to-id mapping if assembly is provided
        name_to_step_id: dict[str, str] = {}
        if assembly is not None:
            # Try to match part names to step descriptions
            for step in plan.steps:
                for part in getattr(assembly, "parts", []):
                    if part.name in step.description.lower() or part.name.replace("_", " ") in step.description.lower():
                        name_to_step_id[part.name] = step.id
                        break

        # Determine dependencies
        step_id_set = {s.id for s in plan.steps}

        for step in plan.steps:
            deps: list[str] = []

            # Use explicit dependencies from PlanStep if set
            if step.dependencies:
                deps = [d for d in step.dependencies if d in step_id_set]
            elif assembly is not None:
                # Infer dependencies from assembly joints
                deps = _infer_deps_from_assembly(step, assembly, name_to_step_id, step_id_set)

            # Determine agent role from expected tools
            role = "general"
            step_tools_lower = [t.lower() for t in step.expected_tools]
            if any("fc_" in t or "cad" in t for t in step_tools_lower):
                role = "modeling"
            elif any("vlm" in t or "screen" in t or "verify" in t for t in step_tools_lower):
                role = "vision"
            elif any("gui" in t for t in step_tools_lower):
                role = "gui"

            dag.add_step(step, dependencies=deps, agent_role=role)

        # Validate: detect cycles early
        dag._detect_cycles()

        return dag


def _infer_deps_from_assembly(
    step: PlanStep,
    assembly: Any,
    name_to_step_id: dict[str, str],
    step_id_set: set[str],
) -> list[str]:
    """Infer step dependencies from Assembly joint relationships."""
    deps: list[str] = []
    desc_lower = step.description.lower()

    joints = getattr(assembly, "joints", [])
    for joint in joints:
        parent = getattr(joint, "parent", "")
        child = getattr(joint, "child", "")

        # If this step describes the child part, it depends on the parent part step
        if child and child in desc_lower:
            parent_step_id = name_to_step_id.get(parent)
            if parent_step_id and parent_step_id in step_id_set and parent_step_id != step.id:
                deps.append(parent_step_id)

    return deps
