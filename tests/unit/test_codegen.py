"""Code generation: literals, structure, and the guarantee that output parses."""

from __future__ import annotations

import ast
import math

import pytest

from opendrive_transpiler import TranspileOptions, transpile_source
from opendrive_transpiler.codegen.writer import SourceWriter, literal

SIMPLE = (
    "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
    "left = LineString3d(getId(), [Point3d(getId(), 0.0, 1.0, 0.0), "
    "Point3d(getId(), 10.0, 1.0, 0.0)])\n"
    "right = LineString3d(getId(), [Point3d(getId(), 0.0, -1.0, 0.0), "
    "Point3d(getId(), 10.0, -1.0, 0.0)])\n"
    "ll = Lanelet(getId(), left, right)\n"
    "ll.attributes['subtype'] = 'road'\n"
)


def generate(source: str = SIMPLE, **kwargs) -> str:
    result = transpile_source(source, "sample.py", options=TranspileOptions(**kwargs))
    assert result.ok, [d.message for d in result.errors]
    return result.code


# --------------------------------------------------------------------------
# Literals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 1.0, 0.1, 1e-9, 1e17, -3.25, math.pi, 1 / 3])
def test_float_literals_round_trip_exactly(value: float):
    """The whole exactness claim rests on this."""
    assert ast.literal_eval(literal(value)) == value


def test_negative_zero_is_normalised():
    """Otherwise two equivalent inputs could produce different bytes."""
    assert literal(-0.0) == literal(0.0)


def test_non_finite_floats_are_refused_rather_than_emitted():
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            literal(value)


@pytest.mark.parametrize(
    "value,expected", [(True, "True"), (False, "False"), (None, "None"), (7, "7")]
)
def test_scalar_literals(value: object, expected: str):
    assert literal(value) == expected


def test_string_literals_are_escaped():
    assert ast.literal_eval(literal('it\'s a "line"\n')) == 'it\'s a "line"\n'


def test_unsupported_types_are_refused():
    with pytest.raises(TypeError):
        literal(object())


# --------------------------------------------------------------------------
# The writer
# --------------------------------------------------------------------------


def test_blank_is_idempotent_and_reaches_the_requested_count():
    writer = SourceWriter()
    writer.line("a")
    writer.blank()
    writer.blank()
    writer.line("b")
    assert writer.render() == "a\n\nb\n"

    writer = SourceWriter()
    writer.line("a")
    writer.blank(2)
    writer.line("b")
    assert writer.render() == "a\n\n\nb\n"


def test_blocks_indent_and_dedent():
    writer = SourceWriter()
    writer.line("def f():")
    with writer.block():
        writer.line("return 1")
    writer.line("x = f()")
    assert writer.render() == "def f():\n    return 1\nx = f()\n"


# --------------------------------------------------------------------------
# Generated script structure
# --------------------------------------------------------------------------


def test_generated_source_is_valid_python():
    ast.parse(generate())


def test_generated_source_defines_build_and_a_main_guard():
    tree = ast.parse(generate())
    functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert "build" in functions
    assert any(isinstance(node, ast.If) for node in tree.body)


def test_generated_source_imports_only_scenariogeneration():
    tree = ast.parse(generate())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 1
    assert imports[0].module == "scenariogeneration"


def test_generated_source_uses_fixed_geometry():
    """Mixing add_geometry and add_fixed_geometry raises; we must use only the latter."""
    code = generate()
    assert "add_fixed_geometry(" in code
    assert "add_geometry(" not in code.replace("add_fixed_geometry(", "")


def test_generated_source_adjusts_only_to_derive_links():
    code = generate()
    assert "odr.adjust_roads_and_lanes()" in code
    assert "adjust_startpoints" not in code


def test_generated_source_records_provenance():
    code = generate()
    assert "sha256:" in code
    assert "lanelet #" in code
    assert "subtype='road'" in code


def test_generated_header_lists_every_notice():
    """A reader must be able to tell a complete network from a partial one."""
    source = SIMPLE.replace("ll.attributes['subtype'] = 'road'\n", "")
    result = transpile_source(source, "sample.py", options=TranspileOptions(strict=False))
    assert "LL2ODR-W801" in result.code


def test_the_first_width_record_sits_at_the_section_origin():
    """OpenDRIVE requires it, and the Lane constructor *is* that record."""
    for line in generate().splitlines():
        if "xodr.Lane(lane_type" in line:
            assert "soffset=0.0" in line


def test_regenerating_the_same_input_gives_the_same_bytes():
    assert generate() == generate()


def test_an_empty_map_still_produces_a_valid_script():
    result = transpile_source(
        "from lanelet2.core import Point3d\np = Point3d(1, 0.0, 0.0, 0.0)\n",
        "empty.py",
        options=TranspileOptions(strict=False),
    )
    ast.parse(result.code)
    assert result.stats.roads == 0


# --------------------------------------------------------------------------
# Options that are planned but not implemented
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [("reference_line", "centerline"), ("fit", "arc"), ("fit", "parampoly3")],
)
def test_planned_options_are_refused_rather_than_silently_ignored(field: str, value: str):
    """Accepting a flag and doing something else is the failure mode to avoid."""
    options = TranspileOptions(**{field: value})
    with pytest.raises(ValueError, match="not implemented yet"):
        options.validate()


@pytest.mark.parametrize("field,value", [("reference_line", "middle"), ("fit", "bezier")])
def test_nonsense_options_are_refused(field: str, value: str):
    with pytest.raises(ValueError, match="invalid"):
        TranspileOptions(**{field: value}).validate()
