"""The symbolic executor, snippet by snippet.

These tests are the specification of what an input script may contain. When the
README's "Python constructs" checklist gains a tick, a case here is what backs it.
"""

from __future__ import annotations

import pytest

from opendrive_transpiler.config import TranspileOptions
from opendrive_transpiler.diagnostics import DiagnosticBag, Severity, TranspileError
from opendrive_transpiler.frontend.interp import execute
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


def test_class_definitions_are_refused_clearly():
    with pytest.raises(TranspileError) as excinfo:
        run("class Foo:\n    pass\n")
    assert excinfo.value.diagnostic.code == "LL2ODR-E201"
    assert excinfo.value.diagnostic.span.line > 0


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
