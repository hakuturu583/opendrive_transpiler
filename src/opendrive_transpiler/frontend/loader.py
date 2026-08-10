"""Reading and parsing the input script.

This module is the only place the input text enters the process, and it does
exactly one thing with it: `ast.parse`. There is no `exec`, no `eval`, no
`compile` to a code object, and no import of the script as a module. The input
is data from here to the end of the pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ..diagnostics import E_ENCODING, E_READ, E_SYNTAX, DiagnosticBag, SourceSpan


def read_source(path: str | Path, bag: DiagnosticBag) -> str:
    """Read a script from disk, reporting failures as diagnostics."""
    path = Path(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        bag.error(E_ENCODING, f"{path}: not valid UTF-8 ({exc.reason})", SourceSpan(str(path)))
        return ""
    except OSError as exc:
        bag.error(E_READ, f"{path}: {exc.strerror or exc}", SourceSpan(str(path)))
        return ""


def parse_source(source: str, filename: str, bag: DiagnosticBag) -> ast.Module | None:
    """Parse to an AST. The script is never executed -- only inspected."""
    bag.set_source(source)
    try:
        return ast.parse(source, filename=filename, mode="exec")
    except SyntaxError as exc:
        span = SourceSpan(filename, exc.lineno or 0, max((exc.offset or 1) - 1, 0))
        bag.error(E_SYNTAX, exc.msg, span)
        return None
