"""`--fit=arc`, and how far an arc is allowed to leave the polyline.

Arc fitting reads the input vertices as *samples* of a curve, so the arc it
recovers is meant to pass between them rather than along their chords. That
reading is what makes the option useful, and unbounded it is also what made it
wrong: three points lie exactly on their own circumcircle whatever the arc does
between them, so a two-segment run passed however far it strayed.

On the Lanelet2 Karlsruhe example that produced a circle of radius 21.8 m swept
171 degrees out of three points -- 65 m of arc across a 43 m gap, bulging 19 m
off the road, with both endpoints still sitting on the polyline so nothing
noticed. 85 of 251 records were over 100 mm out.

The bound is now the sagitta between consecutive samples, and the worst record on
that map is 0.0995 m.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from opendrive_transpiler.geometry.fit import ARC_SAGITTA, arc_geometries, build_plan_view
from opendrive_transpiler.geometry.polyline import total_length


def circle(radius: float, degrees: float, step: float, *, at=(0.0, 0.0)) -> list:
    count = int(degrees / step)
    return [
        (
            at[0] + radius * math.cos(math.radians(step * i)),
            at[1] + radius * math.sin(math.radians(step * i)),
            0.0,
        )
        for i in range(count + 1)
    ]


def stray(records, polyline, samples: int = 40) -> float:
    """Worst distance from the emitted curve to the polyline it claims to describe."""
    worst = 0.0
    for record in records:
        for k in range(samples + 1):
            ds = record.length * k / samples
            if record.kind == "arc" and abs(record.params["curvature"]) > 1e-12:
                curvature = record.params["curvature"]
                theta = record.hdg + curvature * ds
                x = record.x + (math.sin(theta) - math.sin(record.hdg)) / curvature
                y = record.y - (math.cos(theta) - math.cos(record.hdg)) / curvature
            else:
                x = record.x + ds * math.cos(record.hdg)
                y = record.y + ds * math.sin(record.hdg)
            worst = max(worst, _distance((x, y), polyline))
    return worst


def _distance(point, polyline) -> float:
    best = math.inf
    for a, b in pairwise(polyline):
        dx, dy = b[0] - a[0], b[1] - a[1]
        span = dx * dx + dy * dy
        t = (
            0.0
            if span == 0
            else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / span))
        )
        best = min(best, math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy)))
    return best


# --------------------------------------------------------------------------
# What it still does
# --------------------------------------------------------------------------


def test_a_densely_sampled_arc_collapses_to_one_record():
    points = circle(50.0, 90.0, 3.0)
    records = arc_geometries(points)
    assert len(records) == 1
    assert records[0].kind == "arc"
    assert records[0].params["curvature"] == pytest.approx(0.02, rel=1e-6)
    assert records[0].length == pytest.approx(50.0 * math.pi / 2, rel=1e-6)


def test_a_coarsely_sampled_arc_is_still_fitted():
    """7.5-degree samples leave their own chords by 81 mm, and that is the point.

    Held to `chord_tolerance` this would fall back to lines, which would make the
    option useless on any real map: nothing samples a curve finely enough to sit
    within a tenth of a millimetre of its own chords.
    """
    points = circle(38.0, 90.0, 7.5)
    records = arc_geometries(points)
    assert any(r.kind == "arc" for r in records)


def test_a_straight_run_is_still_a_line():
    points = [(float(i) * 10.0, 0.0, 0.0) for i in range(8)]
    records = arc_geometries(points)
    assert [r.kind for r in records] == ["line"]


def test_the_planview_still_covers_the_whole_polyline():
    points = circle(50.0, 90.0, 3.0)
    records = arc_geometries(points)
    assert sum(r.length for r in records) == pytest.approx(total_length(points), rel=0.01), (
        "an arc is longer than the chords it replaces, but not by much"
    )


# --------------------------------------------------------------------------
# What it may no longer do
# --------------------------------------------------------------------------


def test_an_arc_may_not_bulge_further_than_the_sagitta_bound():
    points = circle(38.0, 90.0, 7.5)
    records = arc_geometries(points)
    assert stray(records, points) <= ARC_SAGITTA + 1e-9


def test_a_near_reversal_no_longer_invents_most_of_a_circle():
    """The Karlsruhe shape: a tiny segment beside a long one.

    Both endpoints sit on the polyline, so a circumcircle through the three
    points is a perfect fit *at the samples* -- and sweeps 171 degrees between
    them.
    """
    points = [(74.801, 284.905, 0.0), (74.444, 283.050, 0.0), (116.188, 271.635, 0.0)]
    records = arc_geometries(points)
    assert all(r.kind == "line" for r in records), "no arc is evidenced by these three points"
    assert stray(records, points) == pytest.approx(0.0, abs=1e-9)


def test_the_bound_holds_however_pathological_the_sampling():
    """A swept angle just under a full turn used to be admissible."""
    points = [
        (0.0, 0.0, 0.0),
        (0.1, 1.0, 0.0),
        (0.2, 0.0, 0.0),
    ]
    records = arc_geometries(points)
    assert stray(records, points) <= ARC_SAGITTA + 1e-9


def test_a_caller_who_widens_the_chord_tolerance_gets_the_slack():
    """`--chord-tolerance` above the sagitta bound is an explicit ask for more."""
    points = circle(38.0, 90.0, 30.0)
    tight = arc_geometries(points)
    loose = arc_geometries(points, chord_tolerance=5.0)
    assert stray(tight, points) <= ARC_SAGITTA + 1e-9
    assert stray(loose, points) > ARC_SAGITTA, "the wider bound was actually used"


def test_the_bound_can_be_set_outright():
    points = circle(38.0, 90.0, 15.0)
    assert stray(arc_geometries(points, sagitta=0.01), points) <= 0.01 + 1e-9


def test_build_plan_view_applies_the_bound_too():
    """The bound has to reach the path the transpiler actually calls."""
    points = [(74.801, 284.905, 0.0), (74.444, 283.050, 0.0), (116.188, 271.635, 0.0)]
    records, simplified = build_plan_view(points, fit="arc")
    assert stray(records, simplified) <= ARC_SAGITTA + 1e-9
