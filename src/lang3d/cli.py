"""CLI interface for Language-3D Agent."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .agent.core import Agent
from .agent.state import PlanStep, StepStatus
from .config import load_config

console = Console()
error_console = Console(stderr=True)


def print_banner() -> None:
    banner = Text()
    banner.append("Language-3D Agent", style="bold cyan")
    banner.append(" v0.1.0", style="dim")
    banner.append("\nAutonomous 3D Modeling Assistant", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))
    console.print()


def print_help() -> None:
    help_text = """[bold]Commands:[/bold]
  [cyan]/plan <task>[/cyan]     Plan a task without executing
  [cyan]/run <task>[/cyan]      Run a task with planning
  [cyan]/chat <message>[/cyan]  Chat without tools
  [cyan]/direct <task>[/cyan]   Run without planning
  [cyan]/tools[/cyan]           List available tools
  [cyan]/status[/cyan]          Show current state
  [cyan]/agents[/cyan]          Show active sub-agents
  [cyan]/dag[/cyan]             Show task dependency graph
  [cyan]/help[/cyan]            Show this help
  [cyan]/quit[/cyan]            Exit
"""
    console.print(help_text)


def print_tools(agent: Agent) -> None:
    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for tool in agent.tools._tools.values():
        defn = tool.get_definition()
        table.add_row(defn.name, defn.description)

    console.print(table)


def print_plan(plan: Any) -> None:
    table = Table(title=f"Plan: {plan.goal}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Step", style="white")
    table.add_column("Status", width=12)
    table.add_column("Tools", style="dim")

    for i, step in enumerate(plan.steps, 1):
        status_style = {
            StepStatus.PENDING: "dim",
            StepStatus.IN_PROGRESS: "yellow",
            StepStatus.COMPLETED: "green",
            StepStatus.FAILED: "red",
            StepStatus.SKIPPED: "dim",
        }.get(step.status, "white")

        status_icon = {
            StepStatus.PENDING: "○",
            StepStatus.IN_PROGRESS: "◐",
            StepStatus.COMPLETED: "●",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "○",
        }.get(step.status, "○")

        table.add_row(
            str(i),
            step.description[:60],
            f"[{status_style}]{status_icon} {step.status.value}[/{status_style}]",
            ", ".join(step.expected_tools) if step.expected_tools else "",
        )

    console.print(table)


def print_step_update(step: PlanStep) -> None:
    status_style = "green" if step.status == StepStatus.COMPLETED else "red" if step.status == StepStatus.FAILED else "yellow"
    console.print(f"  [{status_style}]{'●' if step.status == StepStatus.COMPLETED else '◐'} {step.status.value}: {step.description[:80]}[/{status_style}]")


def print_agents(agent: Agent) -> None:
    """Show active sub-agents from the web panel state."""
    try:
        from .web.app import _agent_state
        agents = _agent_state.get("sub_agents", [])
        if not agents:
            console.print("[dim]No active sub-agents[/dim]")
            return
        table = Table(title="Sub-Agents")
        table.add_column("Agent ID", style="cyan")
        table.add_column("Status")
        table.add_column("Step")
        table.add_column("Time", style="dim")
        for a in agents:
            status_style = {
                "running": "yellow",
                "completed": "green",
                "failed": "red",
            }.get(a.get("status", ""), "white")
            table.add_row(
                a.get("agent_id", "?"),
                f"[{status_style}]{a.get('status', '?')}[/{status_style}]",
                (a.get("step", ""))[:50],
                a.get("time", ""),
            )
        console.print(table)
    except ImportError:
        console.print("[dim]Web panel not available[/dim]")


def print_dag(agent: Agent) -> None:
    """Show task DAG from the web panel state."""
    try:
        from .web.app import _agent_state
        dag = _agent_state.get("dag")
        if not dag:
            console.print("[dim]No task DAG available[/dim]")
            return
        waves = dag.get("waves", [])
        for i, wave in enumerate(waves):
            console.print(f"\n[bold cyan]Wave {i}[/bold cyan] ({len(wave)} tasks)")
            for node in wave:
                status_style = {
                    "pending": "dim",
                    "in_progress": "yellow",
                    "running": "yellow",
                    "completed": "green",
                    "failed": "red",
                    "skipped": "dim",
                }.get(node.get("status", "pending"), "white")
                console.print(
                    f"  [{status_style}]\u25CB {node.get('description', '?')[:60]}[/{status_style}]"
                )
            if i < len(waves) - 1:
                console.print("  [dim]\u2193[/dim]")
    except ImportError:
        console.print("[dim]Web panel not available[/dim]")


def run_cli() -> None:
    """Main CLI entry point."""
    print_banner()

    # Load config
    try:
        config = load_config()
    except Exception as e:
        error_console.print(f"[red]Error loading config: {e}[/red]")
        console.print("Tip: Copy .env.example to .env and fill in your API keys")
        return

    # Initialize agent
    try:
        agent = Agent(config)
    except Exception as e:
        error_console.print(f"[red]Error initializing agent: {e}[/red]")
        return

    # Set up callbacks
    def on_tool_call(name: str, args: dict) -> None:
        arg_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
        if len(args) > 3:
            arg_str += ", ..."
        console.print(f"  [dim]Tool:[/dim] [cyan]{name}[/cyan]({arg_str})")

    def on_tool_result(name: str, result: str) -> None:
        # Show truncated result
        preview = result[:200].replace("\n", " ")
        if len(result) > 200:
            preview += "..."
        console.print(f"  [dim]Result:[/dim] {preview}")

    def on_thinking(text: str) -> None:
        if text.strip():
            console.print(f"  [dim italic]{text[:200]}[/dim italic]")

    def on_plan_update(plan: Any) -> None:
        print_plan(plan)

    def on_step_update(step: PlanStep) -> None:
        print_step_update(step)

    agent.on_tool_call(on_tool_call)
    agent.on_tool_result(on_tool_result)
    agent.on_thinking(on_thinking)
    agent.on_plan_update(on_plan_update)
    agent.on_step_update(on_step_update)

    console.print(f"[green]Configured backends:[/green] {', '.join(agent.router.available_backends)}")
    console.print(f"[green]Workspace:[/green] {agent.state.workspace}")
    console.print()

    # Interactive loop
    while True:
        try:
            user_input = console.input("[bold cyan]lang3d>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input == "/quit" or user_input == "/exit":
            console.print("[dim]Goodbye![/dim]")
            break
        elif user_input == "/help":
            print_help()
        elif user_input == "/tools":
            print_tools(agent)
        elif user_input == "/status":
            if agent.state.plan:
                print_plan(agent.state.plan)
            else:
                console.print("[dim]No active plan[/dim]")
        elif user_input == "/agents":
            print_agents(agent)
        elif user_input == "/dag":
            print_dag(agent)
        elif user_input.startswith("/plan "):
            task = user_input[6:].strip()
            console.print(f"[yellow]Planning: {task}[/yellow]")
            try:
                plan = agent.planner.create_plan(task)
                agent.state.plan = plan
                print_plan(plan)
            except Exception as e:
                error_console.print(f"[red]Planning error: {e}[/red]")
        elif user_input.startswith("/run "):
            task = user_input[5:].strip()
            console.print(f"[yellow]Running task: {task}[/yellow]\n")
            try:
                result = agent.run_task(task, use_planning=True)
                console.print(f"\n[green]{result}[/green]")
            except Exception as e:
                error_console.print(f"[red]Error: {e}[/red]")
        elif user_input.startswith("/direct "):
            task = user_input[8:].strip()
            try:
                result = agent.run_task(task, use_planning=False)
                console.print(Markdown(result))
            except Exception as e:
                error_console.print(f"[red]Error: {e}[/red]")
        elif user_input.startswith("/chat "):
            msg = user_input[6:].strip()
            try:
                response = agent.chat(msg)
                console.print(Markdown(response.content))
            except Exception as e:
                error_console.print(f"[red]Error: {e}[/red]")
        else:
            # Default: treat as a task
            console.print(f"[yellow]Running: {user_input}[/yellow]\n")
            try:
                result = agent.run_task(user_input, use_planning=True)
                console.print(f"\n[green]{result}[/green]")
            except Exception as e:
                error_console.print(f"[red]Error: {e}[/red]")

        console.print()


def main() -> None:
    """Entry point for the lang3d command."""
    try:
        run_cli()
    except Exception as e:
        error_console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
