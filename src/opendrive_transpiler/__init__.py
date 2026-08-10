"""opendrive_transpiler -- lanelet2 map scripts to OpenDRIVE.

Takes a Python script written against the lanelet2 / simple_lanelet2 API, parses
its AST, executes it *symbolically* (never for real), and emits a Python script
that builds the equivalent OpenDRIVE network with `scenariogeneration.xodr`.

    from opendrive_transpiler import transpile
    result = transpile("my_map.py")
    print(result.code)

Because the conversion is static, neither lanelet2 nor a Rust toolchain has to be
installed to convert a map. The core has no dependencies at all; only *running*
the generated script needs `scenariogeneration` (the `[emit]` extra).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from .config import TranspileOptions
from .diagnostics import Diagnostic, DiagnosticBag, Severity, TranspileError
from .odr.model import OdrModel, TranspileStats

__version__ = "0.1.0"

__all__ = [
    "Diagnostic",
    "DiagnosticBag",
    "OdrModel",
    "Severity",
    "TranspileError",
    "TranspileOptions",
    "TranspileResult",
    "TranspileStats",
    "__version__",
    "transpile",
    "transpile_source",
    "transpile_to_xodr",
]


@dataclass
class TranspileResult:
    """The generated script, plus everything the run has to say about it."""

    code: str = ""
    diagnostics: list[Diagnostic] = field(default_factory=list)
    stats: TranspileStats = field(default_factory=TranspileStats)
    model: OdrModel | None = None
    source_name: str = "<string>"

    @property
    def ok(self) -> bool:
        return not any(d.severity is Severity.ERROR for d in self.diagnostics)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is Severity.WARNING]

    def __str__(self) -> str:
        return self.code


def transpile_source(
    code: str,
    filename: str = "<string>",
    *,
    options: TranspileOptions | None = None,
) -> TranspileResult:
    """Transpile lanelet2 script *text* into a scenariogeneration script."""
    from .codegen.scenariogeneration import emit_source
    from .frontend.interp import execute
    from .frontend.loader import parse_source
    from .ir.model import build_ir
    from .mapping.build import build_model

    options = options or TranspileOptions()
    options.validate()

    bag = DiagnosticBag(strict=options.strict)
    result = TranspileResult(source_name=filename)

    try:
        module = parse_source(code, filename, bag)
        if module is None:
            result.diagnostics = bag.items
            return result

        registry = execute(module, filename, bag, options)
        stem = Path(filename).stem or "map"
        ir = build_ir(registry, bag, options, source_name=options.name or stem)
        model, stats = build_model(ir, bag, options)
        result.model = model
        result.stats = stats
        result.code = emit_source(
            model, stats, bag, options, source_name=filename, source_text=code
        )
    except TranspileError:
        # Strict mode: the diagnostic that aborted is already in the bag, and it
        # carries the source location. Callers read it from `result.errors`.
        pass

    result.diagnostics = bag.items
    return result


def transpile(
    src: str | Path,
    *,
    options: TranspileOptions | None = None,
) -> TranspileResult:
    """Transpile a lanelet2 script file."""
    from .frontend.loader import read_source

    options = options or TranspileOptions()
    path = Path(src)
    bag = DiagnosticBag(strict=options.strict)
    try:
        text = read_source(path, bag)
    except TranspileError:
        return TranspileResult(diagnostics=bag.items, source_name=str(path))
    if not text:
        return TranspileResult(diagnostics=bag.items, source_name=str(path))
    return transpile_source(text, str(path), options=options)


def transpile_to_xodr(
    src: str | Path,
    out: str | Path,
    *,
    options: TranspileOptions | None = None,
) -> Path:
    """Transpile and immediately run the result, writing a `.xodr`.

    Requires the `[emit]` extra (`scenariogeneration`). Note the asymmetry that
    matters: the *input* script is never executed, only the script this package
    generated.
    """
    from .runner import run_generated

    result = transpile(src, options=options)
    if not result.ok:
        raise TranspileError(result.errors[0])
    return run_generated(result.code, out)


def source_digest(text: str) -> str:
    """SHA-256 of an input script, as recorded in generated file headers."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
