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


def test_generated_source_does_not_adjust_anything():
    """Every link is written down, so nothing is left for the backend to infer.

    `adjust_roads_and_lanes()` would re-derive lane links from geometry, and it
    refuses when two connected roads carry different lane counts -- which is what a
    lane widening is. The script must not depend on it.
    """
    code = generate()
    # The comment names the call to explain its absence, so look for the call
    # itself: a statement, not a mention.
    calls = [line.strip() for line in code.splitlines() if not line.strip().startswith("#")]
    assert not [line for line in calls if "adjust_roads_and_lanes" in line]
    assert not [line for line in calls if "adjust_startpoints" in line]


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
# Option validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_line", "left-bound"),
        ("reference_line", "centerline"),
        ("fit", "line"),
        ("fit", "arc"),
        ("fit", "parampoly3"),
    ],
)
def test_implemented_options_are_accepted(field: str, value: str):
    TranspileOptions(**{field: value}).validate()


@pytest.mark.parametrize("field,value", [("reference_line", "middle"), ("fit", "bezier")])
def test_nonsense_options_are_refused(field: str, value: str):
    with pytest.raises(ValueError, match="invalid"):
        TranspileOptions(**{field: value}).validate()


def test_a_planned_option_value_would_be_refused_not_ignored():
    """The guard itself, exercised through a temporarily-planned value.

    Every value is implemented today, so this pins the mechanism rather than a
    particular option: adding a new planned value must refuse it, not quietly
    fall back to the default.
    """
    options = TranspileOptions()
    options.PLANNED_FITS = ("spiral",)
    options.fit = "spiral"
    with pytest.raises(ValueError, match="not implemented yet"):
        options.validate()


# --------------------------------------------------------------------------
# Alternative geometry strategies
# --------------------------------------------------------------------------


CURVE = (
    "import math\n"
    "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
    "inner, outer = [], []\n"
    "for step in range(13):\n"
    "    angle = math.radians(90.0 * step / 12)\n"
    "    inner.append(Point3d(getId(), 38.0 * math.cos(angle), 38.0 * math.sin(angle), 0.0))\n"
    "    outer.append(Point3d(getId(), 42.0 * math.cos(angle), 42.0 * math.sin(angle), 0.0))\n"
    "ll = Lanelet(getId(), LineString3d(getId(), inner), LineString3d(getId(), outer))\n"
    "ll.attributes['subtype'] = 'road'\n"
)


def test_arc_fitting_emits_arcs_for_a_curve():
    code = generate(CURVE, fit="arc")
    assert "xodr.Arc(" in code
    assert code.count("add_fixed_geometry") < 12  # collapsed, not one per segment


def test_parampoly3_fitting_emits_cubics():
    code = generate(CURVE, fit="parampoly3")
    assert "xodr.ParamPoly3(" in code


def test_line_fitting_stays_the_default_and_emits_only_lines():
    code = generate(CURVE)
    assert "xodr.Line(" in code
    assert "xodr.Arc(" not in code and "xodr.ParamPoly3(" not in code


TWO_LANE = (
    "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
    "top = [Point3d(getId(), i * 20.0, 3.5, 0.0) for i in range(3)]\n"
    "mid = [Point3d(getId(), i * 20.0, 0.0, 0.0) for i in range(3)]\n"
    "bot = [Point3d(getId(), i * 20.0, -3.5, 0.0) for i in range(3)]\n"
    "shared = LineString3d(getId(), mid)\n"
    "a = Lanelet(getId(), LineString3d(getId(), top), shared)\n"
    "b = Lanelet(getId(), shared, LineString3d(getId(), bot))\n"
    "a.attributes['subtype'] = 'road'\n"
    "b.attributes['subtype'] = 'road'\n"
)


def test_centerline_reference_puts_lanes_on_both_sides():
    """With the reference down the middle, the two lanes straddle it as +1/-1."""
    code = generate(TWO_LANE, reference_line="centerline")
    assert "add_left_lane(" in code
    assert "add_right_lane(" in code


UNEVEN_LANES = TWO_LANE.replace("i * 20.0, 3.5", "i * 20.0, 6.0").replace(
    "i * 20.0, -3.5", "i * 20.0, -2.0"
)


def test_a_symmetric_cross_section_needs_no_lane_offset():
    """The centreline lands on the shared boundary, so lane 0 is already there."""
    assert "add_laneoffset(" not in generate(TWO_LANE, reference_line="centerline")


def test_an_uneven_cross_section_records_where_lane_zero_sits():
    """A 6 m lane beside a 2 m one puts the centreline off the shared boundary.

    Without a laneOffset the wide lane would straddle lane 0, which is invalid;
    the offset moves lane 0 back onto the boundary that actually divides them.
    """
    code = generate(UNEVEN_LANES, reference_line="centerline")
    assert "add_laneoffset(" in code
    assert "add_left_lane(" in code and "add_right_lane(" in code


def test_a_boundary_reference_never_needs_a_lane_offset():
    assert "add_laneoffset(" not in generate(UNEVEN_LANES)


def test_left_bound_reference_keeps_every_lane_on_one_side():
    code = generate(CURVE)
    assert "add_left_lane(" not in code
    assert "add_right_lane(" in code
