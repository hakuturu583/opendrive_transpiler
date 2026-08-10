"""The safety property, asserted mechanically.

The claim this package makes is precise: *the input script is never executed*.
`frontend/` may only parse and interpret. The one legitimate `exec` in the
package is in `runner.py`, and it runs the source this package itself generated.

A grep test is crude, but it is exactly the right shape here -- the property is
"this token does not appear in this directory", and that is what CI should check.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "opendrive_transpiler"
FRONTEND = SRC / "frontend"

FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__", "open", "input"}
FORBIDDEN_IMPORTS = {"importlib", "subprocess", "os", "socket", "shutil", "pickle"}


def python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def test_the_frontend_contains_no_dynamic_execution():
    offenders: list[str] = []
    for path in python_files(FRONTEND):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_CALLS
            ):
                offenders.append(f"{path.name}:{node.lineno}: {node.func.id}()")
    assert not offenders, "dynamic execution reached the frontend: " + "; ".join(offenders)


def test_the_frontend_imports_nothing_that_touches_the_outside_world():
    offenders: list[str] = []
    for path in python_files(FRONTEND):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in FORBIDDEN_IMPORTS:
                    offenders.append(f"{path.name}:{node.lineno}: {name}")
    assert not offenders, "frontend reached outside the process: " + "; ".join(offenders)


def test_only_the_runner_may_exec():
    execers = set()
    for path in python_files(SRC):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "exec"
            ):
                execers.add(path.name)
    assert execers <= {"runner.py"}, f"unexpected exec() in {sorted(execers - {'runner.py'})}"


def test_the_core_never_imports_its_optional_backend():
    """Only `runner.py` may import scenariogeneration; the rest must install bare."""
    offenders: list[str] = []
    for path in python_files(SRC):
        if path.name == "runner.py":
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                module = node.names[0].name
            elif isinstance(node, ast.ImportFrom):
                module = node.module
            if module and module.split(".")[0] in {"scenariogeneration", "numpy", "lanelet2"}:
                offenders.append(f"{path.name}:{node.lineno}: {module}")
    assert not offenders, "the zero-dependency core grew a dependency: " + "; ".join(offenders)


def test_transpiling_a_hostile_script_does_not_run_it(tmp_path: pytest.TempPathFactory):
    """The strongest form of the test: a script whose side effect would be visible."""
    from opendrive_transpiler import TranspileOptions, transpile_source

    marker = Path(str(tmp_path)) / "should_not_exist"
    source = (
        "import os\n"
        f"open({str(marker)!r}, 'w').write('executed')\n"
        f"os.system('touch {marker}')\n"
        "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
        "ll = Lanelet(getId(), "
        "LineString3d(getId(), [Point3d(getId(), 0.0, 1.0, 0.0), "
        "Point3d(getId(), 9.0, 1.0, 0.0)]),\n"
        "             LineString3d(getId(), [Point3d(getId(), 0.0, -1.0, 0.0), "
        "Point3d(getId(), 9.0, -1.0, 0.0)]))\n"
        "ll.attributes['subtype'] = 'road'\n"
    )
    result = transpile_source(source, "hostile.py", options=TranspileOptions(strict=False))

    assert not marker.exists(), "the input script's side effect actually happened"
    # And the convertible part of the map still came through.
    assert result.stats.roads == 1
