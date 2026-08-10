"""Running a generated script to produce a `.xodr`.

This is the one module in the package that calls `exec`, and the distinction is
worth being explicit about:

    The *input* lanelet2 script is never executed. It is parsed and interpreted
    symbolically by `frontend/`, which contains no exec/eval/compile at all.

    The script executed here is the one *this package just generated*, from a
    template we control, containing only `scenariogeneration.xodr` calls.

It is also the only module that imports `scenariogeneration`, which is why the
rest of the package installs and runs with no dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class EmitDependencyMissing(RuntimeError):
    """Raised when the generated script cannot run because its dependency is absent."""


def _require_scenariogeneration() -> Any:
    try:
        from scenariogeneration import xodr
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise EmitDependencyMissing(
            "writing a .xodr needs scenariogeneration; install it with\n"
            '    pip install "opendrive-transpiler[emit]"\n'
            "The generated Python script itself does not require it to be produced."
        ) from exc
    return xodr


def build_model(code: str) -> Any:
    """Execute a generated script and return the `xodr.OpenDrive` it builds."""
    _require_scenariogeneration()
    namespace: dict[str, Any] = {"__name__": "opendrive_transpiler.generated"}
    # This is our own generated source, not the user's input script.
    exec(compile(code, "<generated>", "exec"), namespace)
    builder = namespace.get("build")
    if builder is None:
        raise RuntimeError("generated script defines no build() function")
    return builder()


def run_generated(code: str, out: str | Path) -> Path:
    """Execute a generated script and write its OpenDRIVE output to `out`."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(code)
    model.write_xml(str(out))
    return out
