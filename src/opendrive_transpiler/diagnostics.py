"""Diagnostics: the single channel through which every stage reports.

A silently half-converted HD map is worse than a failed conversion -- it looks
plausible while missing roads. So every stage that drops, guesses at, or
approximates something records a coded diagnostic here, and the CLI aborts on the
first error unless the caller opts into best-effort mode.

Codes are namespaced by stage so that a code alone identifies where to look:

    E1xx  parse / source loading
    E2xx  unsupported Python construct
    E3xx  import or lanelet2 API misuse
    E4xx  value that cannot be resolved statically
    W5xx  topology
    W6xx  control flow
    W7xx  geometry
    W8xx  attribute mapping
    I9xx  informational (feature recognised but deliberately not converted)

The I9xx codes are deliberately paired with unchecked rows in the README support
matrix, so that what the tool reports and what the README promises stay in sync.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field


class Severity(enum.IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name.lower()


# --------------------------------------------------------------------------
# Codes
# --------------------------------------------------------------------------
# Kept as plain module constants rather than an enum so that they interpolate
# into messages without ceremony and so that new codes are one line to add.

# E1xx -- parse / loading
E_SYNTAX = "LL2ODR-E101"
E_READ = "LL2ODR-E102"
E_ENCODING = "LL2ODR-E103"

# E2xx -- unsupported construct
E_UNSUPPORTED_STMT = "LL2ODR-E201"
E_UNSUPPORTED_EXPR = "LL2ODR-E202"
E_UNSUPPORTED_TARGET = "LL2ODR-E203"

# E3xx -- import / API
E_UNKNOWN_ATTRIBUTE = "LL2ODR-E301"
E_BAD_ARITY = "LL2ODR-E302"
E_NOT_INSTANTIABLE = "LL2ODR-E303"
I_QUERY_IGNORED = "LL2ODR-I304"
I_LOCAL_IMPORT = "LL2ODR-I305"
E_LOCAL_IMPORT_CYCLE = "LL2ODR-W306"

# E4xx -- non-static values
E_NOT_STATIC = "LL2ODR-E401"
E_LOAD_FROM_FILE = "LL2ODR-E402"
E_NAME_UNDEFINED = "LL2ODR-E403"

# W5xx -- topology
W_BOUNDS_SWAPPED = "LL2ODR-W501"
W_DEGENERATE_LANELET = "LL2ODR-W502"
W_UNEQUAL_BOUND_ENDS = "LL2ODR-W503"
W_NO_LANELETS = "LL2ODR-W504"
W_BOUNDS_DISAGREE = "LL2ODR-W505"
W_PIVOT_REFERENCE = "LL2ODR-W506"
W_STACK_NOT_SHARED = "LL2ODR-W507"
W_CONTRAFLOW_RIGHT = "LL2ODR-W508"
W_UNJOINED_MERGE = "LL2ODR-W509"

# W6xx -- control flow
W_UNKNOWN_CONDITION = "LL2ODR-W601"
W_UNKNOWN_ITERABLE = "LL2ODR-W602"
E_ITERATION_LIMIT = "LL2ODR-E603"
E_UNBOUNDED_WHILE = "LL2ODR-E604"
E_RECURSION_LIMIT = "LL2ODR-E605"
E_STATEMENT_BUDGET = "LL2ODR-E606"
E_UNCAUGHT_RAISE = "LL2ODR-W607"

# W7xx -- geometry
W_ZERO_LENGTH_SEGMENT = "LL2ODR-W701"
W_SHORT_ROAD = "LL2ODR-W702"
W_NEGATIVE_WIDTH = "LL2ODR-W703"

# W8xx -- attribute mapping
W_EMPTY_SUBTYPE = "LL2ODR-W801"
W_UNKNOWN_SUBTYPE = "LL2ODR-W802"
W_BAD_SPEED_LIMIT = "LL2ODR-W803"
W_UNKNOWN_ROADMARK = "LL2ODR-W804"
W_NON_STRING_ATTRIBUTE = "LL2ODR-W805"

# I9xx -- recognised but not converted (mirrors README "not yet supported")
I_JUNCTION_SKIPPED = "LL2ODR-I901"
I_REGELEM_SKIPPED = "LL2ODR-I902"
I_AREA_SKIPPED = "LL2ODR-I903"
I_POLYGON_SKIPPED = "LL2ODR-I904"
I_PRIORITY_SKIPPED = "LL2ODR-I905"
I_TWO_WAY = "LL2ODR-I907"
I_GEO_REFERENCE = "LL2ODR-I908"
I_PROJECTION_LOCALISED = "LL2ODR-I909"
I_CROSSWALK_OBJECT = "LL2ODR-I910"


@dataclass(frozen=True)
class SourceSpan:
    """A location in the *input* script, 1-indexed line, 0-indexed column."""

    filename: str = "<unknown>"
    line: int = 0
    column: int = 0

    def __str__(self) -> str:
        if not self.line:
            return self.filename
        return f"{self.filename}:{self.line}:{self.column}"

    @classmethod
    def from_node(cls, filename: str, node: object) -> SourceSpan:
        """Build a span from any AST node (which carries lineno/col_offset)."""
        return cls(
            filename=filename,
            line=getattr(node, "lineno", 0) or 0,
            column=getattr(node, "col_offset", 0) or 0,
        )


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: Severity = Severity.WARNING
    span: SourceSpan = field(default_factory=SourceSpan)

    def format(self, source_line: str | None = None) -> str:
        """Render as a compiler-style message, optionally with a caret line."""
        # Later stages work on the IR and have no source location to point at;
        # a bare "<unknown>:" prefix would be noise, so it is left off.
        located = self.span.line or self.span.filename != SourceSpan().filename
        head = (
            f"{self.span}: {self.severity}: {self.code}: {self.message}"
            if located
            else f"{self.severity}: {self.code}: {self.message}"
        )
        if source_line is None:
            return head
        stripped = source_line.rstrip("\n")
        caret = " " * self.span.column + "^"
        return f"{head}\n  {stripped}\n  {caret}"

    def __str__(self) -> str:  # pragma: no cover - delegates
        return self.format()


class TranspileError(Exception):
    """Raised when a diagnostic aborts the run (strict mode, or unrecoverable)."""

    def __init__(self, diagnostic: Diagnostic, source_line: str | None = None) -> None:
        self.diagnostic = diagnostic
        self.source_line = source_line
        super().__init__(diagnostic.format(source_line))


class DiagnosticBag:
    """Collects diagnostics and decides whether one aborts the run.

    `strict` controls only whether *errors* abort immediately. Warnings never
    abort; they are always collected and summarised at the end.
    """

    def __init__(self, *, strict: bool = True, source_lines: list[str] | None = None) -> None:
        self.strict = strict
        self._items: list[Diagnostic] = []
        self._source_lines = source_lines or []

    # -- recording ---------------------------------------------------------
    def add(self, diagnostic: Diagnostic) -> Diagnostic:
        self._items.append(diagnostic)
        if diagnostic.severity is Severity.ERROR and self.strict:
            raise TranspileError(diagnostic, self._line_for(diagnostic.span))
        return diagnostic

    def error(self, code: str, message: str, span: SourceSpan | None = None) -> Diagnostic:
        return self.add(Diagnostic(code, message, Severity.ERROR, span or SourceSpan()))

    def warn(self, code: str, message: str, span: SourceSpan | None = None) -> Diagnostic:
        return self.add(Diagnostic(code, message, Severity.WARNING, span or SourceSpan()))

    def info(self, code: str, message: str, span: SourceSpan | None = None) -> Diagnostic:
        return self.add(Diagnostic(code, message, Severity.INFO, span or SourceSpan()))

    # -- querying ----------------------------------------------------------
    def _line_for(self, span: SourceSpan) -> str | None:
        if 1 <= span.line <= len(self._source_lines):
            return self._source_lines[span.line - 1]
        return None

    def set_source(self, source: str) -> None:
        self._source_lines = source.splitlines()

    @property
    def source_lines(self) -> list[str]:
        """The lines diagnostics quote from, so a caller can restore them.

        Reading a second file retargets them; whoever did that has to put the
        original back or later carets point into the wrong source.
        """
        return self._source_lines

    @source_lines.setter
    def source_lines(self, lines: list[str]) -> None:
        self._source_lines = list(lines)

    @property
    def items(self) -> list[Diagnostic]:
        return list(self._items)

    def of(self, severity: Severity) -> list[Diagnostic]:
        return [d for d in self._items if d.severity is severity]

    @property
    def errors(self) -> list[Diagnostic]:
        return self.of(Severity.ERROR)

    @property
    def warnings(self) -> list[Diagnostic]:
        return self.of(Severity.WARNING)

    @property
    def ok(self) -> bool:
        return not self.errors

    def counts(self) -> dict[str, int]:
        return {
            "error": len(self.of(Severity.ERROR)),
            "warning": len(self.of(Severity.WARNING)),
            "info": len(self.of(Severity.INFO)),
        }

    def summary(self) -> str:
        counts = self.counts()
        parts = [f"{n} {name}{'s' if n != 1 else ''}" for name, n in counts.items() if n]
        return ", ".join(parts) if parts else "no diagnostics"

    def render(self, *, with_source: bool = True) -> str:
        return "\n".join(
            d.format(self._line_for(d.span) if with_source else None) for d in self._items
        )

    def to_json_obj(self) -> list[dict[str, object]]:
        return [
            {
                "code": d.code,
                "severity": str(d.severity),
                "message": d.message,
                "file": d.span.filename,
                "line": d.span.line,
                "column": d.span.column,
            }
            for d in self._items
        ]

    def extend(self, items: Iterable[Diagnostic]) -> None:
        for item in items:
            self.add(item)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)
