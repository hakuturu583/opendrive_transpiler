"""Command-line interface.

Exit codes are meant to be usable in a pipeline:

    0  clean
    1  converted, but with warnings (only under --strict)
    2  errors; nothing usable was produced
    3  usage error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import TranspileOptions, TranspileResult, __version__, transpile
from .diagnostics import Severity

EXIT_OK = 0
EXIT_WARNINGS = 1
EXIT_ERRORS = 2
EXIT_USAGE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opendrive-transpile",
        description=(
            "Transpile a lanelet2 map-building Python script into a "
            "scenariogeneration script that produces the equivalent OpenDRIVE."
        ),
        epilog=(
            "The input script is parsed and interpreted symbolically -- it is never "
            "executed, and lanelet2 does not need to be installed."
        ),
    )
    parser.add_argument("input", type=Path, help="lanelet2 script to transpile")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="where to write the generated Python (default: stdout; '-' for stdout)",
    )
    parser.add_argument(
        "--xodr",
        type=Path,
        default=None,
        help="also run the generated script and write this .xodr (needs the [emit] extra)",
    )
    parser.add_argument("--name", default=None, help="OpenDRIVE map name (default: input stem)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    behaviour = parser.add_argument_group("behaviour")
    mode = behaviour.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=True,
        help="stop at the first error (default)",
    )
    mode.add_argument(
        "--best-effort",
        dest="strict",
        action="store_false",
        help="continue past recoverable errors and convert what is convertible",
    )
    behaviour.add_argument(
        "--on-unknown-branch",
        choices=["then", "else", "error"],
        default="then",
        help="which branch to take when an if-condition cannot be resolved statically",
    )
    behaviour.add_argument("--max-iterations", type=int, default=100_000)

    geometry = parser.add_argument_group("geometry")
    geometry.add_argument(
        "--reference-line",
        choices=["left-bound", "centerline"],
        default="left-bound",
        help=(
            "what the planView follows: 'left-bound' (the exact outer-left "
            "boundary, default) or 'centerline' (lanes on both sides)"
        ),
    )
    geometry.add_argument(
        "--fit",
        choices=["line", "arc", "parampoly3"],
        default="line",
        help=(
            "planView fitting: 'line' (exact, one <line> per segment, default), "
            "'arc' (circular arcs) or 'parampoly3' (C1-continuous cubics)"
        ),
    )
    geometry.add_argument("--point-tolerance", type=float, default=1e-3, metavar="M")
    geometry.add_argument("--chord-tolerance", type=float, default=1e-4, metavar="M")
    geometry.add_argument("--heading-tolerance", type=float, default=1e-6, metavar="RAD")
    geometry.add_argument("--width-sample-step", type=float, default=5.0, metavar="M")
    geometry.add_argument(
        "--cubic-profiles",
        action="store_true",
        help="fit one cubic per width/elevation profile where it fits within tolerance",
    )

    topology = parser.add_argument_group("topology")
    topology.add_argument(
        "--no-junctions",
        dest="junctions",
        action="store_false",
        default=True,
        help="do not build <junction> elements; leave branching roads unconnected",
    )

    header = parser.add_argument_group("header")
    header.add_argument("--geo-reference", default=None, help="override the PROJ string")
    header.add_argument(
        "--no-geo-reference",
        dest="emit_geo_reference",
        action="store_false",
        default=True,
        help="omit <geoReference> entirely",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--diagnostics", choices=["text", "json"], default="text")
    output.add_argument("-q", "--quiet", action="store_true", help="only report errors")
    output.add_argument("-v", "--verbose", action="store_true", help="include informational notes")

    return parser


def options_from(args: argparse.Namespace) -> TranspileOptions:
    return TranspileOptions(
        strict=args.strict,
        on_unknown_branch=args.on_unknown_branch,
        max_iterations=args.max_iterations,
        point_tolerance=args.point_tolerance,
        reference_line=args.reference_line,
        fit=args.fit,
        heading_tolerance=args.heading_tolerance,
        chord_tolerance=args.chord_tolerance,
        width_sample_step=args.width_sample_step,
        cubic_profiles=args.cubic_profiles,
        junctions=args.junctions,
        name=args.name,
        geo_reference=args.geo_reference,
        emit_geo_reference=args.emit_geo_reference,
    )


def report(result: TranspileResult, args: argparse.Namespace, stream) -> None:
    if args.diagnostics == "json":
        payload = {
            "source": result.source_name,
            "ok": result.ok,
            "stats": vars(result.stats),
            "diagnostics": [
                {
                    "code": d.code,
                    "severity": str(d.severity),
                    "message": d.message,
                    "file": d.span.filename,
                    "line": d.span.line,
                    "column": d.span.column,
                }
                for d in result.diagnostics
            ],
        }
        print(json.dumps(payload, indent=2), file=stream)
        return

    floor = Severity.ERROR if args.quiet else (Severity.INFO if args.verbose else Severity.WARNING)
    for diagnostic in result.diagnostics:
        if diagnostic.severity >= floor:
            print(diagnostic.format(), file=stream)

    if not args.quiet:
        print(f"{result.source_name}: {result.stats.describe()}", file=stream)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"input file not found: {args.input}")
        return EXIT_USAGE

    try:
        result = transpile(args.input, options=options_from(args))
    except ValueError as exc:  # invalid option combination
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    report(result, args, sys.stderr)

    if not result.ok:
        return EXIT_ERRORS

    destination = args.output or "-"
    if destination == "-":
        sys.stdout.write(result.code)
    else:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.code, encoding="utf-8")
        if not args.quiet:
            print(f"wrote {path}", file=sys.stderr)

    if args.xodr is not None:
        from .runner import EmitDependencyMissing, run_generated

        try:
            written = run_generated(result.code, args.xodr)
        except EmitDependencyMissing as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_ERRORS
        if not args.quiet:
            print(f"wrote {written}", file=sys.stderr)

    if args.strict and result.warnings:
        return EXIT_WARNINGS
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
