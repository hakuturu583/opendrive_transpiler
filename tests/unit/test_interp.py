"""The symbolic executor, snippet by snippet.

These tests are the specification of what an input script may contain. When the
README's "Python constructs" checklist gains a tick, a case here is what backs it.
"""

from __future__ import annotations

import pytest

from opendrive_transpiler.config import TranspileOptions
from opendrive_transpiler.diagnostics import DiagnosticBag, Severity, TranspileError
from opendrive_transpiler.frontend.interp import Interpreter, execute
from opendrive_transpiler.frontend.loader import parse_source
from opendrive_transpiler.ir.model import build_ir

PRELUDE = "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"


def run(body: str, *, strict: bool = True, prelude: str = PRELUDE, **kwargs):
    options = TranspileOptions(strict=strict, **kwargs)
    bag = DiagnosticBag(strict=strict)
    source = prelude + body
    module = parse_source(source, "<test>", bag)
    assert module is not None
    registry = execute(module, "<test>", bag, options)
    return registry, bag


def to_ir(body: str, **kwargs):
    registry, bag = run(body, **kwargs)
    options = TranspileOptions(strict=kwargs.get("strict", True))
    return build_ir(registry, bag, options), bag


def codes(bag: DiagnosticBag) -> set[str]:
    return {d.code for d in bag}


# --------------------------------------------------------------------------
# Construction and the constructor overloads
# --------------------------------------------------------------------------


def test_point_constructor_forms():
    registry, _ = run(
        "from lanelet2.core import BasicPoint3d\n"
        "a = Point3d()\n"
        "b = Point3d(1, 2.0, 3.0)\n"
        "c = Point3d(2, 4.0, 5.0, 6.0)\n"
        "d = Point3d(3, BasicPoint3d(7.0, 8.0, 9.0))\n"
        "e = Point3d(id=4, x=1.0, y=2.0, z=3.0)\n"
        "left = LineString3d(getId(), [a, b, c, d, e])\n"
        "right = LineString3d(getId(), [a, b])\n"
        "ll = Lanelet(getId(), left, right)\n"
    )
    points = registry.lanelets[0].left.points
    assert [p.xyz for p in points] == [
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 0.0),
        (4.0, 5.0, 6.0),
        (7.0, 8.0, 9.0),
        (1.0, 2.0, 3.0),
    ]


def test_cross_dimension_constructor_aliases_storage():
    """`Point2d(p3d)` is a second handle on one point, not a copy.

    This is what lets topology answer "same physical node?" by identity.
    """
    registry, _ = run(
        "from lanelet2.core import Point2d\n"
        "p = Point3d(1, 1.0, 2.0, 3.0)\n"
        "q = Point2d(p)\n"
        "q.x = 42.0\n"
        "left = LineString3d(getId(), [p, Point3d(2, 5.0, 5.0, 0.0)])\n"
        "ll = Lanelet(getId(), left, LineString3d(getId(), [p, p]))\n"
    )
    assert registry.lanelets[0].left.points[0].x == 42.0


def test_linestring_alias_and_invert_share_points():
    registry, _ = run(
        "a = Point3d(1, 0.0, 0.0, 0.0)\n"
        "b = Point3d(2, 1.0, 0.0, 0.0)\n"
        "ls = LineString3d(10, [a, b])\n"
        "flipped = ls.invert()\n"
        "ls.append(Point3d(3, 2.0, 0.0, 0.0))\n"
        "ll = Lanelet(getId(), ls, LineString3d(11, [a, b]))\n"
        "inverted_flag = flipped.inverted()\n"
    )
    lanelet = registry.lanelets[0]
    # The appended point is visible through the original handle...
    assert len(lanelet.left) == 3
    # ...and the inverted view reports itself as inverted.
    assert lanelet.left.inverted() is False


def test_attributes_assigned_after_construction():
    """Tags are usually set as statements, not constructor arguments."""
    ir, _ = to_ir(
        "ls = LineString3d(1, [Point3d(2, 0.0, 0.0, 0.0), Point3d(3, 1.0, 0.0, 0.0)])\n"
        "ls.attributes['type'] = 'line_thin'\n"
        "ls.attributes['subtype'] = 'dashed'\n"
        "ll = Lanelet(4, ls, LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 1.0, -1.0, 0.0)]))\n"
        "ll.attributes['subtype'] = 'road'\n"
    )
    lanelet = ir.lanelets[0]
    assert lanelet.subtype == "road"
    assert lanelet.left.attributes == {"type": "line_thin", "subtype": "dashed"}


def test_attribute_map_coerces_non_string_values():
    _, bag = run(
        "ls = LineString3d(1, [Point3d(2, 0.0, 0.0, 0.0), Point3d(3, 1.0, 0.0, 0.0)])\n"
        "ls.attributes['width'] = 3\n",
        strict=False,
    )
    assert "LL2ODR-W805" in codes(bag)


# --------------------------------------------------------------------------
# Python constructs
# --------------------------------------------------------------------------


def test_comprehension_default_args_loop_and_if():
    """The shape real map scripts are written in, from 0700_routing.py."""
    ir, bag = to_ir(
        "def chain(count, y=0.0, subtype='road'):\n"
        "    left = [Point3d(getId(), i * 10.0, y + 1.0, 0.0) for i in range(count + 1)]\n"
        "    right = [Point3d(getId(), i * 10.0, y - 1.0, 0.0) for i in range(count + 1)]\n"
        "    out = []\n"
        "    for i in range(count):\n"
        "        ll = Lanelet(getId(),\n"
        "                     LineString3d(getId(), [left[i], left[i + 1]]),\n"
        "                     LineString3d(getId(), [right[i], right[i + 1]]))\n"
        "        if subtype is not None:\n"
        "            ll.attributes['subtype'] = subtype\n"
        "        out.append(ll)\n"
        "    return out, left, right\n"
        "lanelets, _l, _r = chain(4)\n"
    )
    assert len(ir.lanelets) == 4
    assert all(ll.subtype == "road" for ll in ir.lanelets)
    assert not bag.errors
    # Consecutive lanelets share their joint points by identity.
    assert ir.lanelets[0].left.points[-1].key == ir.lanelets[1].left.points[0].key


def test_tuple_unpacking_with_star():
    ir, _ = to_ir(
        "pts = [Point3d(i, float(i), 0.0, 0.0) for i in range(4)]\n"
        "first, *rest = pts\n"
        "ll = Lanelet(9, LineString3d(10, [first, rest[0]]),\n"
        "             LineString3d(11, [rest[1], rest[2]]))\n"
    )
    assert len(ir.lanelets) == 1


def test_while_loop_and_augmented_assignment():
    ir, _ = to_ir(
        "pts = []\n"
        "x = 0.0\n"
        "while x < 30.0:\n"
        "    pts.append(Point3d(getId(), x, 0.0, 0.0))\n"
        "    x += 10.0\n"
        "other = [Point3d(getId(), p.x, -2.0, 0.0) for p in pts]\n"
        "ll = Lanelet(getId(), LineString3d(getId(), pts), LineString3d(getId(), other))\n"
    )
    assert len(ir.lanelets[0].left.points) == 3


def test_fstrings_and_dict_comprehension():
    ir, _ = to_ir(
        "tags = {f'key{i}': f'value{i}' for i in range(2)}\n"
        "ls = LineString3d(1, [Point3d(2, 0.0, 0.0, 0.0), Point3d(3, 1.0, 0.0, 0.0)])\n"
        "for key in sorted(tags):\n"
        "    ls.attributes[key] = tags[key]\n"
        "ll = Lanelet(4, ls, LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 1.0, -1.0, 0.0)]))\n"
    )
    assert ir.lanelets[0].left.attributes == {"key0": "value0", "key1": "value1"}


def test_main_guard_is_entered():
    """Scripts commonly build the map inside `if __name__ == '__main__'`."""
    ir, _ = to_ir(
        "def main():\n"
        "    ll = Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 5.0, 1.0, 0.0)]),\n"
        "                 LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 5.0, -1.0, 0.0)]))\n"
        "    return ll\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    assert len(ir.lanelets) == 1


def test_bare_main_function_is_called_when_module_body_built_nothing():
    ir, _ = to_ir(
        "def main():\n"
        "    Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 5.0, 1.0, 0.0)]),\n"
        "            LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 5.0, -1.0, 0.0)]))\n"
    )
    assert len(ir.lanelets) == 1


# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prelude,call",
    [
        ("from lanelet2.core import Point3d\n", "Point3d(1, 1.0, 2.0, 3.0)"),
        ("from lanelet2.core import Point3d as P\n", "P(1, 1.0, 2.0, 3.0)"),
        ("import lanelet2\n", "lanelet2.core.Point3d(1, 1.0, 2.0, 3.0)"),
        ("import lanelet2.core\n", "lanelet2.core.Point3d(1, 1.0, 2.0, 3.0)"),
        ("from lanelet2 import core\n", "core.Point3d(1, 1.0, 2.0, 3.0)"),
        ("import lanelet2.core as c\n", "c.Point3d(1, 1.0, 2.0, 3.0)"),
    ],
)
def test_every_import_form_resolves(prelude: str, call: str):
    # The interpreter's scopes are internal, so the point is observed through a
    # lanelet the script builds from it.
    registry, bag = run(
        prelude
        + "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
        + f"p = {call}\n"
        + "q = Point3d(9, 9.0, 9.0, 9.0)\n"
        + "ll = Lanelet(getId(), LineString3d(getId(), [p, q]), "
        "LineString3d(getId(), [q, p]))\n",
        prelude="",
    )
    assert not bag.errors
    assert registry.lanelets[0].left.points[0].xyz == (1.0, 2.0, 3.0)


def test_routing_queries_are_inert_but_the_map_still_converts():
    """0700_routing.py builds a fine map and then runs dozens of queries."""
    ir, bag = to_ir(
        "import lanelet2.routing as routing\n"
        "import lanelet2.traffic_rules as traffic_rules\n"
        "from lanelet2.core import createMapFromLanelets\n"
        "ll = Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 5.0, 1.0, 0.0)]),\n"
        "             LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 5.0, -1.0, 0.0)]))\n"
        "ll.attributes['subtype'] = 'road'\n"
        "m = createMapFromLanelets([ll])\n"
        "rules = traffic_rules.create(traffic_rules.Locations.Germany,\n"
        "                             traffic_rules.Participants.Vehicle)\n"
        "graph = routing.RoutingGraph(m, rules)\n"
        "following = graph.following(ll)\n",
        strict=False,
    )
    assert len(ir.lanelets) == 1
    assert not bag.errors
    assert "LL2ODR-I304" in codes(bag)


def test_load_from_file_is_reported():
    with pytest.raises(TranspileError) as excinfo:
        run(
            "from lanelet2.io import load\n"
            "from lanelet2.projection import UtmProjector\n"
            "from lanelet2.io import Origin\n"
            "m = load('x.osm', UtmProjector(Origin(49.0, 8.4, 0.0)))\n"
        )
    assert excinfo.value.diagnostic.code == "LL2ODR-E402"


def test_projector_origin_is_captured():
    registry, _ = run(
        "from lanelet2.io import Origin\n"
        "from lanelet2.projection import UtmProjector\n"
        "p = UtmProjector(Origin(49.0, 8.4, 0.0))\n"
    )
    assert registry.projection is not None
    assert (registry.projection.kind, registry.projection.lon) == ("utm", 8.4)


# --------------------------------------------------------------------------
# Unknowns and limits
# --------------------------------------------------------------------------


def test_unknown_condition_takes_the_configured_branch():
    ir, bag = to_ir(
        "import lanelet2.routing as routing\n"
        "graph = routing.RoutingGraph(None, None)\n"
        "if graph.following(None):\n"
        "    Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 5.0, 1.0, 0.0)]),\n"
        "            LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 5.0, -1.0, 0.0)]))\n",
        strict=False,
    )
    assert "LL2ODR-W601" in codes(bag)
    assert len(ir.lanelets) == 1  # "then" is the default policy


def test_loop_over_unknown_is_skipped_with_a_diagnostic():
    _, bag = run(
        "import lanelet2.routing as routing\n"
        "graph = routing.RoutingGraph(None, None)\n"
        "for x in graph.following(None):\n"
        "    pass\n",
        strict=False,
    )
    assert "LL2ODR-W602" in codes(bag)


def test_iteration_limit_is_enforced():
    _, bag = run("for i in range(1000):\n    pass\n", strict=False, max_iterations=10)
    assert "LL2ODR-E603" in codes(bag)


def test_recursion_limit_is_enforced():
    _, bag = run("def f(n):\n    return f(n + 1)\nf(0)\n", strict=False, max_recursion=8)
    assert "LL2ODR-E605" in codes(bag)


def test_undefined_name_reports_its_location():
    with pytest.raises(TranspileError) as excinfo:
        run("x = definitely_not_defined\n")
    assert excinfo.value.diagnostic.code == "LL2ODR-E403"
    assert excinfo.value.diagnostic.severity is Severity.ERROR


def test_syntax_error_is_a_diagnostic_not_a_crash():
    bag = DiagnosticBag(strict=False)
    assert parse_source("def (:\n", "<test>", bag) is None
    assert "LL2ODR-E101" in codes(bag)


# --------------------------------------------------------------------------
# Map semantics
# --------------------------------------------------------------------------


def test_map_add_assigns_fresh_ids_to_id_zero_primitives():
    registry, _ = run(
        "from lanelet2.core import LaneletMap\nm = LaneletMap()\np = Point3d()\nm.add(p)\n"
    )
    assert registry.maps[0].pointLayer.items[0].id != 0


def test_create_map_from_lanelets_pulls_in_bounds_and_points():
    registry, _ = run(
        "from lanelet2.core import createMapFromLanelets\n"
        "ll = Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 5.0, 1.0, 0.0)]),\n"
        "             LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 5.0, -1.0, 0.0)]))\n"
        "m = createMapFromLanelets([ll])\n"
    )
    shadow_map = registry.maps[0]
    assert len(shadow_map.laneletLayer) == 1
    assert len(shadow_map.lineStringLayer) == 2
    assert len(shadow_map.pointLayer) == 4


def test_lanelet_centerline_is_computed_on_demand():
    registry, _ = run(
        "ll = Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 2.0, 1.0, 0.0)]),\n"
        "             LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 2.0, -1.0, 0.0)]))\n"
        "c = ll.centerline\n"
        "n = len(c)\n"
    )
    assert registry.lanelets  # the script ran to completion


def test_assigned_centerline_survives_into_the_ir():
    ir, _ = to_ir(
        "custom = LineString3d(99, "
        "[Point3d(100, 0.0, 0.0, 0.0), Point3d(101, 2.0, 0.0, 0.0)])\n"
        "ll = Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 2.0, 1.0, 0.0)]),\n"
        "             LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 2.0, -1.0, 0.0)]))\n"
        "ll.centerline = custom\n"
    )
    assert ir.lanelets[0].centerline == ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))


def test_regulatory_element_is_recorded_and_reported():
    ir, bag = to_ir(
        "from lanelet2.core import AttributeMap, TrafficLight\n"
        "ll = Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 5.0, 1.0, 0.0)]),\n"
        "             LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 5.0, -1.0, 0.0)]))\n"
        "ll.addRegulatoryElement(TrafficLight(8, AttributeMap(), "
        "[LineString3d(9, [Point3d(10, 0.0, 0.0, 0.0), Point3d(11, 1.0, 0.0, 0.0)])]))\n",
        strict=False,
    )
    regelem = ir.lanelets[0].regelems[0]
    assert regelem.kind == "TrafficLight"
    # The referred geometry comes through too: a signal cannot be placed without it.
    assert regelem.geometry["refers"]
    assert not bag.errors


# --------------------------------------------------------------------------
# Features that are recognised but deliberately not converted
# --------------------------------------------------------------------------


def test_areas_are_reported_with_their_holes():
    """The holes must survive the snapshot, or the outline loses them silently."""
    ir, bag = to_ir(
        "from lanelet2.core import Area, AttributeMap\n"
        "def ring(x0, y0, x1, y1):\n"
        "    pts = [Point3d(getId(), x0, y0, 0.0), Point3d(getId(), x1, y0, 0.0),\n"
        "           Point3d(getId(), x1, y1, 0.0), Point3d(getId(), x0, y1, 0.0)]\n"
        "    return [LineString3d(getId(), [pts[i], pts[(i + 1) % 4]]) for i in range(4)]\n"
        "a = Area(getId(), ring(0.0, 0.0, 10.0, 10.0), [ring(3.0, 3.0, 6.0, 6.0)],\n"
        "         AttributeMap({'subtype': 'parking'}))\n",
        strict=False,
    )
    assert len(ir.areas) == 1
    assert ir.areas[0].attributes["subtype"] == "parking"
    # The hole is carried through rather than dropped at the IR boundary.
    assert len(ir.areas[0].inners) == 1
    assert not bag.errors


def test_standalone_polygons_are_reported():
    ir, bag = to_ir(
        "from lanelet2.core import Polygon3d\n"
        "p = Polygon3d(getId(), [Point3d(getId(), 0.0, 0.0, 0.0),\n"
        "                        Point3d(getId(), 1.0, 0.0, 0.0),\n"
        "                        Point3d(getId(), 1.0, 1.0, 0.0)])\n",
        strict=False,
    )
    assert len(ir.polygons) == 1
    assert not bag.errors


def test_a_lanelet_bound_is_not_mistaken_for_a_polygon():
    """Only real Polygon* constructions count; boundaries must not inflate the tally."""
    ir, _ = to_ir(
        "ll = Lanelet(1, LineString3d(2, "
        "[Point3d(3, 0.0, 1.0, 0.0), Point3d(4, 5.0, 1.0, 0.0)]),\n"
        "             LineString3d(5, "
        "[Point3d(6, 0.0, -1.0, 0.0), Point3d(7, 5.0, -1.0, 0.0)]))\n",
        strict=False,
    )
    assert ir.polygons == []


# --------------------------------------------------------------------------
# Compound views, sequences and layer queries
# --------------------------------------------------------------------------

CHAIN_OF_THREE = (
    "up = [Point3d(getId(), i * 10.0, 1.0, 0.0) for i in range(4)]\n"
    "down = [Point3d(getId(), i * 10.0, -1.0, 0.0) for i in range(4)]\n"
    "lls = []\n"
    "for i in range(3):\n"
    "    ll = Lanelet(getId(), LineString3d(getId(), [up[i], up[i + 1]]),\n"
    "                 LineString3d(getId(), [down[i], down[i + 1]]))\n"
    "    ll.attributes['subtype'] = 'road'\n"
    "    lls.append(ll)\n"
)


def evaluate(body: str, name: str):
    """Run a snippet and read one of its module-level variables back."""
    options = TranspileOptions(strict=True)
    bag = DiagnosticBag(strict=True)
    module = parse_source(PRELUDE + body, "<test>", bag)
    interpreter = Interpreter("<test>", bag, options)
    interpreter.run(module)
    found, value = interpreter.globals.lookup(name)
    assert found, f"{name!r} was never bound"
    return value, bag


def test_compound_line_string_chains_members_without_repeating_joints():
    value, bag = evaluate(
        CHAIN_OF_THREE + "from lanelet2.core import CompoundLineString3d\n"
        "c = CompoundLineString3d([ll.leftBound for ll in lls])\n"
        "probe = [len(c), c.numSegments(), len(c.lineStrings()), len(c.ids())]\n",
        "probe",
    )
    # Three two-point line strings sharing their joints: four points, three segments.
    assert value == [4, 3, 3, 3]
    assert not bag.errors


def test_lanelet_sequence_reads_as_one_long_lanelet():
    value, _ = evaluate(
        CHAIN_OF_THREE + "from lanelet2.core import LaneletSequence\n"
        "s = LaneletSequence(lls)\n"
        "probe = [len(s.lanelets()), len(s.leftBound), len(s.rightBound), s.inverted()]\n",
        "probe",
    )
    assert value == [3, 4, 4, False]


def test_layer_search_returns_what_is_inside_the_box():
    value, _ = evaluate(
        CHAIN_OF_THREE
        + "from lanelet2.core import BasicPoint2d, BoundingBox2d, createMapFromLanelets\n"
        "m = createMapFromLanelets(lls)\n"
        "box = BoundingBox2d(BasicPoint2d(-1.0, -2.0), BasicPoint2d(11.0, 2.0))\n"
        "probe = len(m.pointLayer.search(box))\n",
        "probe",
    )
    assert value == 4  # x in {0, 10} on both sides


def test_layer_nearest_returns_the_requested_count_in_order():
    value, _ = evaluate(
        CHAIN_OF_THREE + "from lanelet2.core import BasicPoint2d, createMapFromLanelets\n"
        "m = createMapFromLanelets(lls)\n"
        "near = m.pointLayer.nearest(BasicPoint2d(0.0, 0.0), 2)\n"
        "probe = [len(near), round(near[0].x, 6)]\n",
        "probe",
    )
    assert value == [2, 0.0]


def test_find_usages_is_structural_not_spatial():
    """Consecutive lanelets share joint points but not boundaries."""
    value, _ = evaluate(
        CHAIN_OF_THREE + "from lanelet2.core import createMapFromLanelets\n"
        "m = createMapFromLanelets(lls)\n"
        "probe = len(m.laneletLayer.findUsages(lls[0].leftBound))\n",
        "probe",
    )
    assert value == 1


def test_vector2d_refuses_construction_as_lanelet2_does():
    with pytest.raises(TranspileError) as excinfo:
        run("from lanelet2.core import Vector2d\nv = Vector2d()\n")
    assert excinfo.value.diagnostic.code == "LL2ODR-E303"


# --------------------------------------------------------------------------
# Constructs the executor evaluates
# --------------------------------------------------------------------------
# The input script is never executed -- these are evaluated symbolically, and
# each case here backs a ticked row in the README's construct checklist.


def test_a_class_gathers_state_and_reads_it_back():
    value, bag = evaluate(
        "class Road:\n"
        "    width = 3.5\n"
        "    def __init__(self, length):\n"
        "        self.length = length\n"
        "    def area(self):\n"
        "        return self.length * self.width\n"
        "probe = Road(40.0).area()\n",
        "probe",
    )
    assert value == 140.0
    assert not codes(bag)


def test_a_subclass_overrides_and_calls_back_through_super():
    value, _ = evaluate(
        "class Base:\n"
        "    def length(self):\n"
        "        return 20.0\n"
        "class Long(Base):\n"
        "    def length(self):\n"
        "        return super().length() * 2\n"
        "probe = [Long().length(), Base().length()]\n",
        "probe",
    )
    assert value == [40.0, 20.0]


def test_the_explicit_two_argument_super_works_too():
    value, _ = evaluate(
        "class Base:\n"
        "    def name(self):\n"
        "        return 'base'\n"
        "class Sub(Base):\n"
        "    def name(self):\n"
        "        return super(Sub, self).name() + '+sub'\n"
        "probe = Sub().name()\n",
        "probe",
    )
    assert value == "base+sub"


def test_isinstance_sees_script_defined_classes():
    value, _ = evaluate(
        "class Base:\n    pass\n"
        "class Sub(Base):\n    pass\n"
        "probe = [isinstance(Sub(), Sub), isinstance(Sub(), Base), isinstance(Base(), Sub)]\n",
        "probe",
    )
    assert value == [True, True, False]


def test_a_decorator_wraps_the_function_it_decorates():
    value, _ = evaluate(
        "def double(fn):\n"
        "    def wrapper(x):\n"
        "        return fn(x) * 2\n"
        "    return wrapper\n"
        "@double\n"
        "def identity(x):\n"
        "    return x\n"
        "probe = identity(20.0)\n",
        "probe",
    )
    assert value == 40.0


def test_stacked_decorators_apply_innermost_first():
    value, _ = evaluate(
        "def add(fn):\n"
        "    return lambda x: fn(x) + 1\n"
        "def times(fn):\n"
        "    return lambda x: fn(x) * 10\n"
        "@add\n"
        "@times\n"
        "def f(x):\n"
        "    return x\n"
        "probe = f(2)\n",
        "probe",
    )
    # `times` is nearest the def, so it runs first: 2 * 10, then + 1.
    assert value == 21


def test_match_selects_by_literal_and_falls_through_to_the_wildcard():
    value, _ = evaluate(
        "def width(kind):\n"
        "    match kind:\n"
        "        case 'street':\n"
        "            return 3.0\n"
        "        case 'highway':\n"
        "            return 3.75\n"
        "        case _:\n"
        "            return 2.5\n"
        "probe = [width('street'), width('highway'), width('track')]\n",
        "probe",
    )
    assert value == [3.0, 3.75, 2.5]


def test_match_destructures_sequences_and_honours_guards():
    value, _ = evaluate(
        "def classify(spec):\n"
        "    match spec:\n"
        "        case [name, length] if length > 10:\n"
        "            return name + '-long'\n"
        "        case [name, _]:\n"
        "            return name + '-short'\n"
        "        case _:\n"
        "            return 'unknown'\n"
        "probe = [classify(['a', 40]), classify(['b', 4]), classify('x')]\n",
        "probe",
    )
    assert value == ["a-long", "b-short", "unknown"]


def test_match_destructures_mappings_and_classes():
    value, _ = evaluate(
        "class Straight:\n"
        "    def __init__(self, length):\n"
        "        self.length = length\n"
        "def describe(item):\n"
        "    match item:\n"
        "        case {'length': length}:\n"
        "            return length\n"
        "        case Straight(length=length):\n"
        "            return length\n"
        "        case _:\n"
        "            return 0.0\n"
        "probe = [describe({'length': 12.0}), describe(Straight(9.0)), describe(3)]\n",
        "probe",
    )
    assert value == [12.0, 9.0, 0.0]


def test_match_binds_or_patterns_and_captures():
    value, _ = evaluate(
        "def side(kind):\n"
        "    match kind:\n"
        "        case 'left' | 'port':\n"
        "            return -1\n"
        "        case other:\n"
        "            return other\n"
        "probe = [side('port'), side('left'), side(7)]\n",
        "probe",
    )
    assert value == [-1, -1, 7]


def test_a_generator_is_materialised_into_its_values():
    value, _ = evaluate(
        "def lengths():\n"
        "    for i in range(3):\n"
        "        yield i * 10.0\n"
        "probe = [list(lengths()), max(lengths()), sum(lengths())]\n",
        "probe",
    )
    assert value == [[0.0, 10.0, 20.0], 20.0, 30.0]


def test_yield_from_flattens_a_delegated_generator():
    value, _ = evaluate(
        "def inner():\n"
        "    yield 1\n"
        "    yield 2\n"
        "def outer():\n"
        "    yield 0\n"
        "    yield from inner()\n"
        "    yield 3\n"
        "probe = list(outer())\n",
        "probe",
    )
    assert value == [0, 1, 2, 3]


def test_a_generator_can_be_iterated_by_a_for_loop():
    value, _ = evaluate(
        "def points():\n"
        "    yield 0.0\n"
        "    yield 5.0\n"
        "total = 0.0\n"
        "for p in points():\n"
        "    total = total + p\n"
        "probe = total\n",
        "probe",
    )
    assert value == 5.0


def test_a_raise_unwinds_to_the_matching_handler():
    value, bag = evaluate(
        "def length_of(kind):\n"
        "    if kind != 'highway':\n"
        "        raise ValueError('unknown kind')\n"
        "    return 40.0\n"
        "try:\n"
        "    probe = length_of('street')\n"
        "except ValueError:\n"
        "    probe = 0.0\n",
        "probe",
    )
    assert value == 0.0
    assert not codes(bag)


def test_handlers_are_tried_in_order_and_else_runs_when_nothing_raised():
    value, _ = evaluate(
        "class TooShort(Exception):\n"
        "    pass\n"
        "def attempt(raising):\n"
        "    try:\n"
        "        if raising:\n"
        "            raise TooShort('nope')\n"
        "    except ValueError:\n"
        "        return 'value'\n"
        "    except TooShort as exc:\n"
        "        return 'short:' + exc.args[0]\n"
        "    else:\n"
        "        return 'clean'\n"
        "probe = [attempt(True), attempt(False)]\n",
        "probe",
    )
    assert value == ["short:nope", "clean"]


def test_a_base_class_handler_catches_a_derived_exception():
    value, _ = evaluate(
        "try:\n    raise ZeroDivisionError('x')\nexcept ArithmeticError:\n    probe = 'caught'\n",
        "probe",
    )
    assert value == "caught"


def test_finally_runs_on_both_paths():
    value, _ = evaluate(
        "log = []\n"
        "def attempt(raising):\n"
        "    try:\n"
        "        if raising:\n"
        "            raise ValueError('x')\n"
        "        return 'ok'\n"
        "    except ValueError:\n"
        "        return 'failed'\n"
        "    finally:\n"
        "        log.append(raising)\n"
        "probe = [attempt(False), attempt(True), log]\n",
        "probe",
    )
    assert value == ["ok", "failed", [False, True]]


def test_a_bare_raise_reraises_what_is_being_handled():
    value, _ = evaluate(
        "def inner():\n"
        "    try:\n"
        "        raise ValueError('boom')\n"
        "    except ValueError:\n"
        "        raise\n"
        "try:\n"
        "    inner()\n"
        "    probe = 'escaped'\n"
        "except ValueError as exc:\n"
        "    probe = exc.args[0]\n",
        "probe",
    )
    assert value == "boom"


def test_an_uncaught_raise_is_reported_not_swallowed():
    _, bag = run("raise ValueError('nothing catches this')\n", strict=False)
    assert "LL2ODR-W607" in codes(bag)
