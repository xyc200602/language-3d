"""Unit tests for the top-level CLI subcommand dispatch.

Covers ``lang3d web`` / ``lang3d sim`` / ``lang3d help`` argv handling in
``src/lang3d/cli.py:main``.  External servers (uvicorn) and the MuJoCo
viewer are mocked so these run as fast pure-Python unit tests.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from lang3d import cli


def _set_argv(*args: str) -> None:
    """Replace sys.argv for a subcommand invocation."""
    sys.argv = ["lang3d", *args]


def test_help_prints_and_returns(capsys: pytest.CaptureFixture[str]) -> None:
    """`lang3d help` prints usage and does not start the REPL."""
    _set_argv("help")
    cli.main()
    out = capsys.readouterr().out
    assert "Subcommands" in out or "子命令" in out
    assert "web" in out and "sim" in out


def test_help_via_flags(capsys: pytest.CaptureFixture[str]) -> None:
    """`lang3d --help` / `lang3d -h` behave like `lang3d help`."""
    for flag in ("--help", "-h"):
        _set_argv(flag)
        cli.main()
        out = capsys.readouterr().out
        assert "web" in out


def test_unknown_flag_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    """An unrecognized flag should not fall through to the REPL."""
    _set_argv("--bogus")
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "web" in out


def test_web_dispatches_to_run_server() -> None:
    """`lang3d web` calls web.app.run_server with default host/port."""
    _set_argv("web")
    with patch("lang3d.web.app.run_server") as mock_run, \
         patch("webbrowser.open"), \
         patch("threading.Thread"):
        cli.main()
    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs.get("host") == "127.0.0.1"
    assert kwargs.get("port") == 8765


def test_web_accepts_custom_host_and_port() -> None:
    """`lang3d web 0.0.0.0 9000` passes through host and port."""
    _set_argv("web", "0.0.0.0", "9000")
    with patch("lang3d.web.app.run_server") as mock_run, \
         patch("webbrowser.open"), \
         patch("threading.Thread"):
        cli.main()
    _, kwargs = mock_run.call_args
    assert kwargs.get("host") == "0.0.0.0"
    assert kwargs.get("port") == 9000


def test_web_rejects_bad_port() -> None:
    """A non-integer port exits with code 2."""
    _set_argv("web", "127.0.0.1", "not-a-port")
    with patch("lang3d.web.app.run_server") as mock_run, \
         patch("threading.Thread"):
        with pytest.raises(SystemExit) as exc:
            cli.main()
    assert exc.value.code == 2
    mock_run.assert_not_called()


def test_sim_without_arg_exits_with_usage() -> None:
    """`lang3d sim` with no folder prints usage and exits 2."""
    _set_argv("sim")
    with pytest.raises(SystemExit) as exc:
        cli._run_sim([])
    assert exc.value.code == 2


def test_sim_missing_urdf_exits(tmp_path) -> None:
    """A folder without an engineering_package/urdf.xml exits cleanly."""
    _set_argv("sim", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        cli._run_sim([str(tmp_path)])
    assert exc.value.code == 2


def test_sim_dispatches_to_tool(tmp_path) -> None:
    """`lang3d sim <folder>` calls SimMujocoTool.execute on the URDF."""
    pkg = tmp_path / "engineering_package"
    pkg.mkdir()
    (pkg / "urdf.xml").write_text("<robot/>")
    _set_argv("sim", str(tmp_path))
    with patch("lang3d.tools.sim_mujoco.SimMujocoTool") as mock_tool_cls:
        cli.main()
    mock_tool_cls.assert_called_once()
    mock_tool_cls.return_value.execute.assert_called_once()
    call_kwargs = mock_tool_cls.return_value.execute.call_args.kwargs
    assert call_kwargs["urdf_path"].endswith("urdf.xml")
    assert call_kwargs["interactive"] is True


def test_no_argv_falls_through_to_repl() -> None:
    """With no subcommand, main() launches the interactive REPL."""
    _set_argv()
    with patch("lang3d.cli.run_cli") as mock_run:
        cli.main()
    mock_run.assert_called_once()
