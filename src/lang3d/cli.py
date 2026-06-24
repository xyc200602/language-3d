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
  [cyan]/export <name> [dir][/cyan]  Export engineering package
  [cyan]/iter <folder>[/cyan]   Start iterative edit session on a run folder
  [cyan]/tools[/cyan]           List available tools
  [cyan]/status[/cyan]          Show current state
  [cyan]/agents[/cyan]          Show active sub-agents
  [cyan]/dag[/cyan]             Show task dependency graph
  [cyan]/help[/cyan]            Show this help
  [cyan]/quit[/cyan]            Exit

[bold]Inside /iter session:[/bold]
  [cyan]<text>[/cyan]           Modify assembly (e.g. "把夹爪加长50%")
  [cyan]/undo[/cyan]            Revert last edit
  [cyan]/save[/cyan]            Save in-place (default after each edit anyway)
  [cyan]/save-as <folder>[/cyan]  Save to a new folder
  [cyan]/verify[/cyan]          Run solver + collision check
  [cyan]/render[/cyan]          Re-render assembly to PNGs
  [cyan]/sim[/cyan]             Open MuJoCo viewer (requires mujoco package)
  [cyan]/info[/cyan]            Show assembly info + edit history
  [cyan]/exit-iter[/cyan]       Leave iter session, return to main prompt
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


def run_iter_session(folder: str) -> None:
    """Run an iterative editing session — Claude-Code-style.

    Loads the assembly from *folder*, accepts free-text edit requests,
    applies targeted modifications, and saves in-place after each edit.
    """
    from .interactive import IterativeSession
    from rich.markdown import Markdown

    try:
        session = IterativeSession(folder)
    except Exception as e:
        error_console.print(f"[red]Failed to open session: {e}[/red]")
        return

    console.print(Panel(
        f"[green]Loaded[/green] {session.assembly.name}\n"
        f"  {len(session.assembly.parts)} parts, "
        f"{len(session.assembly.joints)} joints\n"
        f"  folder: {session.folder}",
        title="Iterative Session",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = console.input("[bold magenta]iter>[/bold magenta] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Leaving iter session.[/dim]")
            break

        if not user_input:
            continue

        if user_input in ("/exit-iter", "/quit", "/exit"):
            console.print("[dim]Leaving iter session.[/dim]")
            break
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/info":
            console.print(Panel(session.describe(), border_style="cyan"))
            continue
        if user_input == "/undo":
            if session.undo():
                console.print("[green]Reverted last edit.[/green]")
            else:
                console.print("[dim]Nothing to undo.[/dim]")
            continue
        if user_input == "/save":
            session.save()
            console.print(f"[green]Saved to {session.folder}[/green]")
            continue
        if user_input.startswith("/save-as "):
            new_folder = user_input[9:].strip()
            session.save(new_folder)
            console.print(f"[green]Saved to {session.folder}[/green]")
            continue
        if user_input == "/verify":
            console.print("[yellow]Verifying...[/yellow]")
            report = session.verify()
            for chk in report.get("checks", []):
                icon = "PASS" if chk["ok"] else "FAIL"
                console.print(f"  [{icon}] {chk['name']}: {chk['detail']}")
            continue
        if user_input == "/render":
            console.print("[yellow]Rendering...[/yellow]")
            out = session.render()
            if out:
                console.print(f"[green]Renders: {out}[/green]")
            else:
                console.print("[dim]Renderer unavailable[/dim]")
            continue
        if user_input == "/sim":
            _launch_sim_viewer(session)
            continue

        # Default: treat as a modification request
        try:
            result = session.apply(user_input)
        except Exception as e:
            error_console.print(f"[red]Edit error: {e}[/red]")
            continue

        # Show classified request
        console.print(
            f"  [cyan]scope={result['scope']}[/cyan]  "
            f"[cyan]intent={result['intent']}[/cyan]  "
            f"[cyan]target={result['target'] or '(none)'}[/cyan]  "
            f"[dim]params={result['params']}[/dim]"
        )
        diff = result["diff"]
        for pc in diff["parts_changed"]:
            console.print(
                f"  [green]changed[/green] {pc['name']}: "
                f"{_fmt_dims(pc['dims_before'])} -> "
                f"{_fmt_dims(pc['dims_after'])}"
            )
        for jc in diff["joints_changed"]:
            console.print(
                f"  [green]changed[/green] joint {jc['parent']} -> {jc['child']}: "
                f"offset {jc['offset_before']} -> {jc['offset_after']}"
            )
        for name in diff.get("parts_added", []):
            console.print(f"  [green]added[/green]   {name}")
        for name in diff.get("parts_removed", []):
            console.print(f"  [red]removed[/red] {name}")

        if not result["applied"]:
            console.print(
                "[dim]No mechanical change identified. "
                "Use 'redesign' or describe a specific edit.[/dim]"
            )
            continue

        # Auto-save in-place after each successful edit
        try:
            session.save()
            console.print(f"  [dim]saved -> {session.folder}/assembly.json[/dim]")
        except Exception as e:
            error_console.print(f"[red]Save error: {e}[/red]")


def _fmt_dims(d: dict) -> str:
    """Compact dimensions formatter for CLI output."""
    if not d:
        return "{}"
    return "{" + ", ".join(f"{k}={v:.1f}" for k, v in d.items()) + "}"


def _launch_sim_viewer(session: Any) -> None:
    """Open the MuJoCo viewer on the current assembly's URDF (best effort)."""
    urdf = session.folder / "engineering_package" / "urdf.xml"
    if not urdf.exists():
        error_console.print(
            f"[red]No URDF found at {urdf}.[/red]  Run /export first."
        )
        return
    try:
        import mujoco  # type: ignore[import-not-found]
        import mujoco.viewer  # type: ignore[import-not-found]
        import threading
        import time as _time
    except ImportError:
        error_console.print(
            "[red]mujoco not installed.[/red]  Run: pip install mujoco"
        )
        return

    console.print(f"[yellow]Launching MuJoCo viewer for {urdf.name}...[/yellow]")
    console.print("[dim](close the viewer window to return)[/dim]")

    def _run() -> None:
        try:
            model = mujoco.MjModel.from_xml_path(str(urdf))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Hold the viewer open until the user closes it
                step = 0
                while viewer.is_running() and step < 100000:
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    _time.sleep(0.016)
                    step += 1
        except Exception as e:
            error_console.print(f"[red]Viewer error: {e}[/red]")

    # Run in a thread so the CLI doesn't permanently block if the user
    # closes the terminal instead of the window.
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # Block until the viewer thread finishes (window closed)
    t.join()


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

    # Register the agent with the web panel so that
    # /api/run-task and other endpoints can drive it.
    try:
        from .web.app import set_agent_instance
        set_agent_instance(agent)
    except Exception:
        pass

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
        elif user_input.startswith("/export"):
            parts = user_input[8:].strip().split()
            assembly_name = parts[0] if parts else "complex_robot"
            output_dir = parts[1] if len(parts) > 1 else None
            console.print(f"[yellow]Exporting engineering package for: {assembly_name}[/yellow]\n")
            try:
                kwargs: dict[str, Any] = {"assembly_name": assembly_name}
                if output_dir:
                    kwargs["output_dir"] = output_dir
                result = agent.tools.execute("export_package", **kwargs)
                console.print(Markdown(result))
            except Exception as e:
                error_console.print(f"[red]Error: {e}[/red]")
        elif user_input.startswith("/iter"):
            arg = user_input[5:].strip()
            if not arg:
                error_console.print(
                    "[red]Usage: /iter <run_folder>[/red]  "
                    "(e.g. /iter data/runs/4dof_arm/20260618_120000)"
                )
                continue
            try:
                run_iter_session(arg)
            except Exception as e:
                error_console.print(f"[red]Iter session error: {e}[/red]")
        else:
            # Default: treat as a task
            console.print(f"[yellow]Running: {user_input}[/yellow]\n")
            try:
                result = agent.run_task(user_input, use_planning=True)
                console.print(f"\n[green]{result}[/green]")
            except Exception as e:
                error_console.print(f"[red]Error: {e}[/red]")

        console.print()


def print_argv_help() -> None:
    """Print usage for the top-level subcommands (lang3d <subcommand>)."""
    console.print(Panel(
        "[bold]Usage:[/bold]  [cyan]lang3d[/cyan] [dim]<subcommand>[/dim]\n\n"
        "[bold]Subcommands / 子命令:[/bold]\n"
        "  [cyan](none)[/cyan]             Start the interactive REPL (default)\n"
        "                        启动交互式命令行（默认）\n"
        "  [cyan]web[/cyan] [dim][host] [port][/dim]   Launch the local web dashboard + 3D viewer\n"
        "                        启动本地网页面板与 3D 查看器\n"
        "  [cyan]sim[/cyan] [dim]<run_folder>[/dim]    Open the MuJoCo physics viewer for a run\n"
        "                        打开某次运行产物的 MuJoCo 物理仿真窗口\n"
        "  [cyan]help[/cyan]              Show this help / 显示本帮助\n\n"
        "[dim]Run `lang3d` with no arguments for the full REPL command set.[/dim]",
        title="Language-3D CLI",
        border_style="cyan",
    ))


def _run_web(argv: list[str]) -> None:
    """Launch the local web dashboard (FastAPI + 3D viewer)."""
    host = argv[0] if len(argv) >= 1 and argv[0] else "127.0.0.1"
    port = 8765
    if len(argv) >= 2:
        try:
            port = int(argv[1])
        except ValueError:
            error_console.print(f"[red]Invalid port: {argv[1]}[/red]")
            sys.exit(2)

    console.print(Panel(
        f"[green]Starting web dashboard[/green]\n"
        f"  URL: [cyan]http://{host}:{port}/simulate[/cyan]\n"
        f"  (press Ctrl+C to stop)",
        border_style="cyan",
    ))
    from .web.app import run_server

    # Best-effort: open a browser tab after a short delay so the server is up.
    import threading
    import webbrowser

    url = f"http://{host}:{port}/simulate"

    def _open_browser() -> None:
        import time as _time
        _time.sleep(1.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass  # headless / no default browser — server still runs

    threading.Thread(target=_open_browser, daemon=True).start()
    run_server(host=host, port=port)


def _run_sim(argv: list[str]) -> None:
    """Open the MuJoCo physics viewer for a run folder.

    Reuses ``SimMujocoTool`` (which applies mesh-path rewriting + model
    stabilization) rather than the lighter ``_launch_sim_viewer`` used by
    the ``/iter`` session, so meshes render correctly out of the box.
    """
    if not argv or not argv[0]:
        error_console.print(
            "[red]Usage: lang3d sim <run_folder>[/red]\n"
            "[dim]  e.g. lang3d sim data/runs/4dof_arm/20260624_172515[/dim]"
        )
        # Hint at available runs
        try:
            from pathlib import Path
            runs_root = Path("data/runs")
            if runs_root.exists():
                cases = sorted(p.name for p in runs_root.iterdir() if p.is_dir())
                if cases:
                    console.print(f"[dim]Available cases: {', '.join(cases)}[/dim]")
        except Exception:
            pass
        sys.exit(2)

    folder = argv[0].rstrip("\\/").replace("\\", "/")
    from pathlib import Path

    run_dir = Path(folder)
    urdf = run_dir / "engineering_package" / "urdf.xml"
    if not urdf.exists():
        error_console.print(
            f"[red]No URDF found at {urdf}[/red].\n"
            "[dim]Run the e2e pipeline or /export first to generate the "
            "engineering package.[/dim]"
        )
        sys.exit(2)

    console.print(f"[yellow]Opening MuJoCo viewer for:[/yellow] {urdf.name}")
    console.print("[dim](close the viewer window to exit)[/dim]")
    try:
        from .tools.sim_mujoco import SimMujocoTool

        SimMujocoTool().execute(
            urdf_path=str(urdf),
            mode="validate",
            interactive=True,
        )
    except ImportError:
        error_console.print(
            "[red]mujoco not installed.[/red]  Run: pip install mujoco"
        )
        sys.exit(2)
    except Exception as e:
        error_console.print(f"[red]Viewer error: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Entry point for the lang3d command.

    Supports subcommand dispatch on ``sys.argv[1]`` (``web``, ``sim``,
    ``help``); with no recognized subcommand it launches the interactive
    REPL (backward compatible).
    """
    sub = sys.argv[1] if len(sys.argv) > 1 else ""
    rest = sys.argv[2:] if len(sys.argv) > 2 else []

    if sub in ("web",):
        _run_web(rest)
        return
    if sub in ("sim",):
        _run_sim(rest)
        return
    if sub in ("help", "-h", "--help"):
        print_argv_help()
        return
    if sub.startswith("-"):
        # Unknown flag — show help rather than treating it as a task.
        print_argv_help()
        sys.exit(2)

    try:
        run_cli()
    except Exception as e:
        error_console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
