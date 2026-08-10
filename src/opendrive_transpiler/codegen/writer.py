"""A small source writer: indentation, comments, and exact float literals.

Floats go through `repr(float(x))`, which round-trips exactly in Python. That
matters more than it looks: the whole point of the `<line>`-per-segment strategy
is that coordinates survive unchanged, and a formatter that printed `0.1` as
`0.1000000000000000055` -- or worse, rounded it -- would give that away in the
generated file.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def literal(value: object) -> str:
    """Render a value as a Python literal that reads well and round-trips."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"cannot emit non-finite float: {value}")
        # `repr` of a float always round-trips; normalise -0.0 so that two runs
        # over equivalent input produce byte-identical output.
        return repr(value + 0.0)
    if isinstance(value, int):
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    raise TypeError(f"no literal form for {type(value).__name__}")


class SourceWriter:
    """Accumulates lines of Python source at a tracked indentation level."""

    INDENT = "    "

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._level = 0

    # -- structure ---------------------------------------------------------
    def indent(self) -> None:
        self._level += 1

    def dedent(self) -> None:
        self._level = max(0, self._level - 1)

    class _Block:
        def __init__(self, writer: SourceWriter) -> None:
            self._writer = writer

        def __enter__(self) -> SourceWriter:
            self._writer.indent()
            return self._writer

        def __exit__(self, *_exc: object) -> None:
            self._writer.dedent()

    def block(self) -> _Block:
        return SourceWriter._Block(self)

    # -- emission ----------------------------------------------------------
    def line(self, text: str = "") -> None:
        if not text:
            self._lines.append("")
        else:
            self._lines.append(self.INDENT * self._level + text)

    def lines(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.line(text)

    def comment(self, text: str) -> None:
        for piece in text.splitlines() or [""]:
            self.line(f"# {piece}" if piece else "#")

    def rule(self, title: str, width: int = 74) -> None:
        """A banner comment; the fill makes road boundaries scannable."""
        prefix = f"# ---- {title} "
        self.line(prefix + "-" * max(width - len(prefix) - self._level * 4, 3))

    def docstring(self, text: str) -> None:
        self.line('"""' + text.split("\n", 1)[0])
        rest = text.split("\n", 1)[1] if "\n" in text else ""
        for piece in rest.splitlines():
            self.line(piece)
        self.line('"""')

    def blank(self, count: int = 1) -> None:
        """Ensure exactly `count` blank lines here.

        Idempotent rather than additive, so callers can ask for separation
        without having to know what the previous one already emitted -- and so
        `blank(2)` reliably gives the two lines PEP 8 wants before a top-level def.
        """
        existing = 0
        while existing < len(self._lines) and self._lines[-1 - existing] == "":
            existing += 1
        for _ in range(count - existing):
            self._lines.append("")

    def render(self) -> str:
        text = "\n".join(self._lines).rstrip("\n")
        return text + "\n"
