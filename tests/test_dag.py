"""Tests for TaskDAG dependency graph."""

from __future__ import annotations

import pytest

from lang3d.agent.dag import DAGNode, TaskDAG
from lang3d.agent.state import Plan, PlanStep, StepStatus


def _make_step(desc: str = "", deps: list[str] | None = None) -> PlanStep:
    return PlanStep(description=desc or f"step-{id(object())}", dependencies=deps or [])


class TestDAGConstruction:
    def test_empty_dag(self):
        dag = TaskDAG()
        assert len(dag.nodes) == 0
        assert dag.parallel_groups() == []

    def test_add_single_step(self):
        step = _make_step("task A")
        dag = TaskDAG()
        node = dag.add_step(step)
        assert isinstance(node, DAGNode)
        assert node.step.id == step.id
        assert len(dag.nodes) == 1

    def test_add_step_with_dependencies(self):
        step_a = _make_step("task A")
        step_b = _make_step("task B")
        dag = TaskDAG()
        dag.add_step(step_a)
        node_b = dag.add_step(step_b, dependencies=[step_a.id])
        assert step_a.id in node_b.dependencies
        # step_a should list step_b as a dependent
        assert step_b.id in dag.get_node(step_a.id).dependents

    def test_diamond_dependency(self):
        """A -> B, A -> C, B -> D, C -> D."""
        a, b, c, d = [_make_step(f"task {n}") for n in "ABCD"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c, dependencies=[a.id])
        dag.add_step(d, dependencies=[b.id, c.id])

        assert len(dag.nodes) == 4
        assert len(dag.get_node(d.id).dependencies) == 2


class TestParallelGroups:
    def test_linear_chain(self):
        a, b, c = [_make_step(f"task {n}") for n in "ABC"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c, dependencies=[b.id])

        groups = dag.parallel_groups()
        assert len(groups) == 3
        assert [n.step.id for n in groups[0]] == [a.id]
        assert [n.step.id for n in groups[1]] == [b.id]
        assert [n.step.id for n in groups[2]] == [c.id]

    def test_all_independent(self):
        steps = [_make_step(f"task {i}") for i in range(5)]
        dag = TaskDAG()
        for s in steps:
            dag.add_step(s)

        groups = dag.parallel_groups()
        assert len(groups) == 1
        assert len(groups[0]) == 5

    def test_diamond(self):
        a, b, c, d = [_make_step(f"task {n}") for n in "ABCD"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c, dependencies=[a.id])
        dag.add_step(d, dependencies=[b.id, c.id])

        groups = dag.parallel_groups()
        assert len(groups) == 3
        assert len(groups[0]) == 1  # A
        assert len(groups[1]) == 2  # B, C
        assert len(groups[2]) == 1  # D

    def test_wide_dag(self):
        """Root -> 4 leaves."""
        root = _make_step("root")
        leaves = [_make_step(f"leaf {i}") for i in range(4)]
        dag = TaskDAG()
        dag.add_step(root)
        for leaf in leaves:
            dag.add_step(leaf, dependencies=[root.id])

        groups = dag.parallel_groups()
        assert len(groups) == 2
        assert len(groups[0]) == 1
        assert len(groups[1]) == 4


class TestReadySteps:
    def test_initial_ready(self):
        a, b, c = [_make_step(f"task {n}") for n in "ABC"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c)

        ready = dag.get_ready_steps()
        ready_ids = {n.step.id for n in ready}
        assert a.id in ready_ids
        assert c.id in ready_ids
        assert b.id not in ready_ids

    def test_ready_after_completion(self):
        a, b = [_make_step(f"task {n}") for n in "AB"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])

        assert len(dag.get_ready_steps()) == 1
        unlocked = dag.mark_completed(a.id)
        assert b.id in unlocked
        ready = dag.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].step.id == b.id


class TestMarkCompleted:
    def test_mark_completed_returns_unlocked(self):
        a, b, c = [_make_step(f"task {n}") for n in "ABC"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c, dependencies=[b.id])

        unlocked = dag.mark_completed(a.id)
        assert b.id in unlocked
        assert c.id not in unlocked

        unlocked2 = dag.mark_completed(b.id)
        assert c.id in unlocked2

    def test_mark_completed_updates_status(self):
        step = _make_step("task")
        dag = TaskDAG()
        dag.add_step(step)
        dag.mark_completed(step.id)
        assert step.status == StepStatus.COMPLETED


class TestMarkFailed:
    def test_mark_failed_cascades(self):
        a, b, c = [_make_step(f"task {n}") for n in "ABC"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c, dependencies=[b.id])

        skipped = dag.mark_failed(a.id)
        assert b.id in skipped
        assert c.id in skipped
        assert dag.get_node(b.id).step.status == StepStatus.SKIPPED
        assert dag.get_node(c.id).step.status == StepStatus.SKIPPED

    def test_mark_failed_independent_not_affected(self):
        a, b = [_make_step(f"task {n}") for n in "AB"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b)

        skipped = dag.mark_failed(a.id)
        assert len(skipped) == 0
        assert b.status == StepStatus.PENDING


class TestCycleDetection:
    def test_cycle_raises(self):
        a, b, c = [_make_step(f"task {n}") for n in "ABC"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])
        dag.add_step(c, dependencies=[b.id])
        # Manually create a cycle: a depends on c (c -> b -> a -> c)
        dag.get_node(a.id).dependencies.append(c.id)
        dag.get_node(c.id).dependents.append(a.id)

        with pytest.raises(ValueError, match="Cycle detected"):
            dag.parallel_groups()


class TestSerialization:
    def test_to_dict(self):
        a, b = [_make_step(f"task {n}") for n in "AB"]
        dag = TaskDAG()
        dag.add_step(a)
        dag.add_step(b, dependencies=[a.id])

        d = dag.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["waves"]) == 2
        assert d["nodes"][0]["id"] == a.id

    def test_from_plan(self):
        steps = [
            PlanStep(description="Build base"),
            PlanStep(description="Build arm"),
            PlanStep(description="Assemble"),
        ]
        plan = Plan(goal="Build robot", steps=steps)
        dag = TaskDAG.from_plan(plan)
        assert len(dag.nodes) == 3

    def test_from_plan_with_dependencies(self):
        steps = [
            PlanStep(description="Build base"),
            PlanStep(description="Build arm", dependencies=["step-1"]),
            PlanStep(description="Assemble"),
        ]
        # Use actual step IDs
        steps[1].dependencies = [steps[0].id]

        plan = Plan(goal="Build robot", steps=steps)
        dag = TaskDAG.from_plan(plan)
        groups = dag.parallel_groups()
        assert len(groups) >= 1


class TestPlanStepFields:
    def test_new_fields_default(self):
        step = PlanStep(description="test")
        assert step.dependencies == []
        assert step.assigned_agent == ""

    def test_new_fields_set(self):
        step = PlanStep(
            description="test",
            dependencies=["abc123"],
            assigned_agent="modeler-1",
        )
        assert step.dependencies == ["abc123"]
        assert step.assigned_agent == "modeler-1"

    def test_save_load_roundtrip(self, tmp_path):
        from lang3d.agent.state import AgentState

        state = AgentState(workspace=str(tmp_path))
        state.plan = Plan(
            goal="test",
            steps=[
                PlanStep(
                    description="step 1",
                    dependencies=["other"],
                    assigned_agent="agent-A",
                )
            ],
        )
        save_path = tmp_path / "state.json"
        state.save(save_path)

        loaded = AgentState.load(save_path)
        assert loaded.plan is not None
        step = loaded.plan.steps[0]
        assert step.dependencies == ["other"]
        assert step.assigned_agent == "agent-A"
