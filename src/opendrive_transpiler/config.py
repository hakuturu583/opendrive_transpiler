"""Tunables for a transpile run.

Defaults are chosen so that the corpus of hand-written lanelet2 scripts converts
without any flags at all; every knob exists because some real input needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TranspileOptions:
    # -- diagnostics -------------------------------------------------------
    strict: bool = True
    """Abort on the first error. Off => recoverable errors become warnings."""

    on_unknown_branch: str = "then"
    """What to do with an `if` whose condition cannot be resolved statically.

    "then" takes the body, "else" takes the orelse, "error" refuses. Forking into
    multiple candidate maps is deliberately not offered: there would be no way to
    choose between the results.
    """

    # -- execution limits --------------------------------------------------
    max_iterations: int = 100_000
    """Cap on iterations of any single loop."""

    max_statements: int = 5_000_000
    """Cap on total statements executed across the whole script."""

    max_recursion: int = 64
    """Cap on user-function call depth."""

    # -- topology ----------------------------------------------------------
    point_tolerance: float = 1e-3
    """Metres. Two points this close are treated as the same physical node."""

    # -- geometry ----------------------------------------------------------
    reference_line: str = "left-bound"
    """What the planView follows.

    "left-bound" follows the leftmost lanelet's outer-left boundary, which is
    real input geometry and so is reproduced exactly; every lane then sits to its
    right as -1, -2, ….

    "centerline" follows the computed centre of the cross-section, so lanes fall
    on both sides as +1/-1, and `<laneOffset>` records where lane 0 actually
    sits. Exact only where the centreline happens to pass through vertices.

    "auto" follows the left bound except on roads where doing so would cut a
    lanelet short, and lanelet2's centerline there. A lane is only as long as the
    reference line, and lanelet2 does not require a lanelet's two bounds to span
    the same stretch of it -- on the Lanelet2 Karlsruhe example, following the
    bound leaves 330 m of lanelet unrepresented, a third of one lanelet in the
    worst case, and "auto" recovers 187 m of that.

    It is not the default because the recovery is not free: on the roads it
    switches, the lanelet's own left bound lands a median 1.1 m from where the
    file puts the lane edge, against 0.000 m for a road that keeps its bound.

    That cost is a coverage problem rather than a limit of the lane model, which
    an earlier version of this note had backwards. A lane edge sits at `C + t * n`,
    perpendicular to the reference line, and wherever the boundary lies under that
    normal the placement is *exact*. What "auto" runs into is that a lanelet which
    outruns its bound has stretches with no left bound at all -- 30% of stations on
    that map -- and there is nothing to measure. So it trades a missing stretch of
    road for a lane edge placed by guesswork over that stretch.

    Choose by what the output is for: "auto" for routing and coverage, the
    default for anything measuring against the source geometry. Splitting a
    laneSection where a boundary starts and ends is what would shrink the
    residual, and is not implemented yet.
    """

    fit: str = "line"
    """planView fitting.

    "line" reproduces the input exactly, one `<line>` per segment, at the cost of
    a heading discontinuity at each vertex. "arc" greedily fits circular arcs,
    and "parampoly3" fits C1-continuous cubics; both trade up to
    `chord_tolerance` of positional error for curvature continuity.
    """

    cubic_profiles: bool = False
    """Fit one cubic across a width/elevation profile when it fits within
    tolerance, instead of emitting piecewise-linear records."""

    heading_tolerance: float = 1e-6
    """Radians. Consecutive segments within this heading delta may merge."""

    chord_tolerance: float = 1e-4
    """Metres. Max deviation allowed when merging segments or fitting curves."""

    width_sample_step: float = 5.0
    """Metres. Extra width/elevation samples are inserted beyond this spacing."""

    width_tolerance: float = 1e-4
    """Metres. Width variation below this collapses to a single constant record."""

    min_road_length: float = 1e-6
    """Metres. Roads shorter than this are dropped with a diagnostic."""

    # -- output ------------------------------------------------------------
    name: str | None = None
    """OpenDRIVE map name. Defaults to the input file's stem."""

    geo_reference: str | None = None
    """Override the <geoReference> string. None => derive from the projector."""

    emit_geo_reference: bool = True

    junctions: bool = True
    """Emit `<junction>` elements at branch points, with the branch lanelets as
    connecting roads. Off means branches simply end their road, unconnected."""

    signals: bool = True
    """Emit `<signal>` for traffic lights, signs and speed limits."""

    objects: bool = True
    """Emit `<object>` outlines for areas, polygons, barriers and crosswalks."""

    # -- roadmark conventions (not carried by lanelet2 attributes) ---------
    thin_mark_width: float = 0.12
    thick_mark_width: float = 0.30
    dash_length: float = 3.0
    dash_space: float = 6.0

    barrier_height: float = 0.75
    """Height of a `guard_rail` / `fence` / `wall` object, in metres.

    lanelet2 boundaries carry no height, so this is a stated convention like the
    mark widths above rather than anything read from the map. It reaches the
    emitted object, so a reader can see what was assumed."""

    # -- codegen -----------------------------------------------------------
    revision: tuple[str, str] = field(default=("1", "5"))
    """OpenDRIVE revMajor/revMinor written into the header."""

    # Values the options accept today, as opposed to values that are planned.
    # Accepting a planned value and quietly doing something else is the one
    # failure mode this whole package is built to avoid.
    IMPLEMENTED_REFERENCE_LINES = ("left-bound", "centerline", "auto")
    PLANNED_REFERENCE_LINES = ()
    IMPLEMENTED_FITS = ("line", "arc", "parampoly3")
    PLANNED_FITS = ()

    def validate(self) -> None:
        if self.on_unknown_branch not in {"then", "else", "error"}:
            raise ValueError(f"invalid on_unknown_branch: {self.on_unknown_branch!r}")

        if self.reference_line in self.PLANNED_REFERENCE_LINES:
            raise ValueError(
                f"reference_line={self.reference_line!r} is not implemented yet; "
                f"use one of {self.IMPLEMENTED_REFERENCE_LINES}"
            )
        if self.reference_line not in self.IMPLEMENTED_REFERENCE_LINES:
            raise ValueError(f"invalid reference_line: {self.reference_line!r}")

        if self.fit in self.PLANNED_FITS:
            raise ValueError(
                f"fit={self.fit!r} is not implemented yet; use one of {self.IMPLEMENTED_FITS}"
            )
        if self.fit not in self.IMPLEMENTED_FITS:
            raise ValueError(f"invalid fit: {self.fit!r}")
        for name in ("point_tolerance", "chord_tolerance", "width_sample_step"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
