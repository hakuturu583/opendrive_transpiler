"""The command line, which is how most people will actually use this.

`--target` decides what gets produced; `-o` says where it goes. The combinations
are few but each has a different default, so they are pinned here rather than
left to be discovered.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from opendrive_transpiler.cli import EXIT_ERRORS, EXIT_OK, EXIT_USAGE, main

from ..conftest import needs_emit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def script(tmp_path: Path) -> Path:
    """A convertible input, copied so outputs land in the temp directory."""
    target = tmp_path / "chain.py"
    shutil.copy(FIXTURES / "chain.py", target)
    return target


# --------------------------------------------------------------------------
# The Python target
# --------------------------------------------------------------------------


def test_the_default_target_writes_the_script_to_stdout(script: Path, capsys):
    assert main([str(script)]) == EXIT_OK
    assert "from scenariogeneration import xodr" in capsys.readouterr().out


def test_an_output_path_takes_the_script(script: Path, tmp_path: Path, capsys):
    out = tmp_path / "generated.py"
    assert main([str(script), "-o", str(out)]) == EXIT_OK
    assert "def build()" in out.read_text()
    assert capsys.readouterr().out == "", "nothing should reach stdout once -o is given"


def test_the_output_directory_is_created(script: Path, tmp_path: Path):
    out = tmp_path / "nested" / "deeper" / "generated.py"
    assert main([str(script), "-o", str(out)]) == EXIT_OK
    assert out.exists()


# --------------------------------------------------------------------------
# The OpenDRIVE target
# --------------------------------------------------------------------------


@needs_emit
def test_target_xodr_writes_beside_the_input(script: Path, capsys):
    """`--target xodr` with no -o still has to produce a file somewhere obvious."""
    assert main([str(script), "--target", "xodr"]) == EXIT_OK
    written = script.with_suffix(".xodr")
    assert written.exists()
    assert written.read_text().lstrip().startswith("<?xml")
    assert capsys.readouterr().out == "", "the script is not the target, so not on stdout"


@needs_emit
def test_target_xodr_honours_the_output_path(script: Path, tmp_path: Path):
    out = tmp_path / "map.xodr"
    assert main([str(script), "--target", "xodr", "-o", str(out)]) == EXIT_OK
    assert out.exists()
    assert not script.with_suffix(".xodr").exists()


@needs_emit
def test_target_both_writes_the_script_and_the_xodr(script: Path, tmp_path: Path):
    """-o names the script; the .xodr cannot share that path, so it sits by the input."""
    out = tmp_path / "generated.py"
    assert main([str(script), "--target", "both", "-o", str(out)]) == EXIT_OK
    assert "def build()" in out.read_text()
    assert script.with_suffix(".xodr").exists()


@needs_emit
def test_an_explicit_xodr_path_asks_for_one(script: Path, tmp_path: Path, capsys):
    """`--xodr` names a path, and naming it is itself the request."""
    out = tmp_path / "explicit.xodr"
    assert main([str(script), "--xodr", str(out)]) == EXIT_OK
    assert out.exists()
    # The default target is still py, so the script goes to stdout as usual.
    assert "def build()" in capsys.readouterr().out


@needs_emit
def test_an_explicit_xodr_path_wins_over_the_output_path(script: Path, tmp_path: Path):
    named = tmp_path / "named.xodr"
    assert (
        main(
            [str(script), "--target", "xodr", "-o", str(tmp_path / "o.xodr"), "--xodr", str(named)]
        )
        == EXIT_OK
    )
    assert named.exists()


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


def test_a_missing_input_is_a_usage_error(tmp_path: Path):
    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / "nope.py")])
    assert excinfo.value.code == EXIT_USAGE


def test_an_unconvertible_script_reports_errors(tmp_path: Path):
    script = tmp_path / "loads.py"
    shutil.copy(FIXTURES / "loads_from_file.py", script)
    assert main([str(script)]) == EXIT_ERRORS


def test_an_unknown_option_value_is_a_usage_error(script: Path):
    """Not exit 2 -- that means the conversion found errors, which is different."""
    with pytest.raises(SystemExit) as excinfo:
        main([str(script), "--target", "dxf"])
    assert excinfo.value.code == EXIT_USAGE
    assert EXIT_USAGE != EXIT_ERRORS


def test_quiet_suppresses_the_summary(script: Path, tmp_path: Path, capsys):
    out = tmp_path / "generated.py"
    assert main([str(script), "-o", str(out), "-q"]) == EXIT_OK
    assert capsys.readouterr().err == ""
