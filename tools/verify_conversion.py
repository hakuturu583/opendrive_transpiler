"""Check an emitted `.xodr` against the `.osm` it was built from.

Not a test, and deliberately not part of the package: it needs `pyproj` and a real
surveyed map, neither of which the transpiler depends on. It is the tool that
found the defects fixed in #8, #9, #10 and #13, kept here so those numbers can be
reproduced rather than taken on trust.

    pip install pyproj
    python tools/osm_to_script.py map.osm map.py
    transpile_lanelet2 map.py --target xodr -o map.xodr
    python tools/verify_conversion.py map.osm map.xodr

**It shares no geometry or projection code with the transpiler.** The `.osm` is
parsed with ElementTree, projected with pyproj using the `.xodr`'s own
`<geoReference>`, and the `.xodr` plan view is evaluated here from the OpenDRIVE
spec. That separation is the whole point: a checker built on the code under test
cancels its own bugs out and reports a clean bill of health.

Three questions, and the third is the one a router cares about:

`geometry`
    Each emitted lane's two boundaries against the two bounds of the lanelet it
    says it is. The inner boundary of the innermost lane *is* the reference line,
    so it should match exactly; boundaries further out are reconstructed by adding
    fitted widths and drift.

`direction`
    Whether a lane's bounds fit better with left and right swapped, which is the
    signature of a lane whose id implies the wrong direction of travel.

`connectivity`
    Every lanelet succession in the `.osm` -- shared start and end boundary nodes,
    the rule `geometry::follows` uses -- against every lane-to-lane connection the
    `.xodr` states, in both directions. A link the file asserts with no succession
    behind it is worse than a missing one: it reads as connectivity that is not
    there.

Provenance comes from the `<userData code="lanelet2_id">` records the transpiler
emits per lane. That is the only thing taken from the conversion, it is
bookkeeping rather than measurement, and a wrong claim shows up as a geometry
failure rather than being absorbed.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import pairwise

try:
    from pyproj import CRS, Transformer
except ImportError:  # pragma: no cover - a tool, not a test
    raise SystemExit("this needs pyproj: pip install pyproj") from None

STEP = 0.5
"""Metres between samples along a reference line."""


# --------------------------------------------------------------------------
# The .osm side
# --------------------------------------------------------------------------


def tags(element) -> dict[str, str]:
    return {t.get("k"): t.get("v") for t in element.findall("tag")}


class Osm:
    def __init__(self, path: str) -> None:
        root = ET.parse(path).getroot()
        self.nodes = {
            n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in root.findall("node")
        }
        self.ways = {
            w.get("id"): [nd.get("ref") for nd in w.findall("nd")] for w in root.findall("way")
        }
        self.lanelets: dict[str, tuple[str, str, dict[str, str]]] = {}
        for relation in root.findall("relation"):
            attributes = tags(relation)
            if attributes.get("type") != "lanelet":
                continue
            roles = {
                m.get("role"): m.get("ref")
                for m in relation.findall("member")
                if m.get("type") == "way"
            }
            left, right = roles.get("left"), roles.get("right")
            if left in self.ways and right in self.ways:
                self.lanelets[relation.get("id")] = (left, right, attributes)

    def project(self, geo_reference: str) -> dict[str, tuple[float, float]]:
        """lat/lon -> the plane the .xodr says its coordinates live in."""
        transformer = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_proj4(geo_reference), always_xy=True
        )
        return {nid: transformer.transform(lon, lat) for nid, (lat, lon) in self.nodes.items()}


def signed_side(polyline, point) -> float:
    """>0 if the point lies left of the polyline, <0 right (nearest segment)."""
    best, best_distance = 0.0, float("inf")
    for a, b in pairwise(polyline):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = dx * dx + dy * dy
        t = (
            0.0
            if length == 0
            else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length))
        )
        px, py = a[0] + t * dx, a[1] + t * dy
        distance = (point[0] - px) ** 2 + (point[1] - py) ** 2
        if distance < best_distance:
            best_distance = distance
            best = dx * (point[1] - a[1]) - dy * (point[0] - a[0])
    return best


def align(left_refs, right_refs, xy):
    """The orientation every lanelet2 loader applies on load.

    A way's stored node order is not the lanelet's direction of travel -- two
    lanelets share one bound and traverse it opposite ways -- so each bound is
    re-oriented against the other. Only the order changes, never which way is
    left, so this matters for succession and not for shape.
    """
    if len(left_refs) <= 1 or len(right_refs) <= 1:
        return left_refs, right_refs

    def middle(refs):
        points = [xy[r] for r in refs if r in xy]
        return points[len(points) // 2] if points else None

    point = middle(right_refs)
    if point is not None and not signed_side([xy[r] for r in left_refs if r in xy], point) < 0.0:
        left_refs = list(reversed(left_refs))
    point = middle(left_refs)
    if point is not None and not signed_side([xy[r] for r in right_refs if r in xy], point) > 0.0:
        right_refs = list(reversed(right_refs))
    return left_refs, right_refs


# --------------------------------------------------------------------------
# The .xodr side: the plan view, evaluated from the spec
# --------------------------------------------------------------------------


def poly(coefficients, ds: float) -> float:
    a, b, c, d = coefficients
    return a + b * ds + c * ds * ds + d * ds * ds * ds


class Geometry:
    def __init__(self, element) -> None:
        self.s = float(element.get("s"))
        self.x = float(element.get("x"))
        self.y = float(element.get("y"))
        self.hdg = float(element.get("hdg"))
        self.length = float(element.get("length"))
        child = next(iter(element))
        self.kind = child.tag
        self.p = {k: float(v) for k, v in child.attrib.items() if k != "pRange"}
        self.p_range = child.get("pRange", "arcLength")

    def at(self, ds: float) -> tuple[float, float, float]:
        if self.kind == "line":
            return (self.x + ds * math.cos(self.hdg), self.y + ds * math.sin(self.hdg), self.hdg)
        if self.kind == "arc":
            k = self.p["curvature"]
            if k == 0.0:
                return (
                    self.x + ds * math.cos(self.hdg),
                    self.y + ds * math.sin(self.hdg),
                    self.hdg,
                )
            heading = self.hdg + k * ds
            return (
                self.x + (math.sin(heading) - math.sin(self.hdg)) / k,
                self.y - (math.cos(heading) - math.cos(self.hdg)) / k,
                heading,
            )
        if self.kind == "paramPoly3":
            p = ds if self.p_range == "arcLength" else (ds / self.length if self.length else 0.0)
            u = poly((self.p["aU"], self.p["bU"], self.p["cU"], self.p["dU"]), p)
            v = poly((self.p["aV"], self.p["bV"], self.p["cV"], self.p["dV"]), p)
            du = self.p["bU"] + 2 * self.p["cU"] * p + 3 * self.p["dU"] * p * p
            dv = self.p["bV"] + 2 * self.p["cV"] * p + 3 * self.p["dV"] * p * p
            cos_h, sin_h = math.cos(self.hdg), math.sin(self.hdg)
            return (
                self.x + u * cos_h - v * sin_h,
                self.y + u * sin_h + v * cos_h,
                self.hdg + math.atan2(dv, du),
            )
        if self.kind == "poly3":
            v = poly((self.p["a"], self.p["b"], self.p["c"], self.p["d"]), ds)
            dv = self.p["b"] + 2 * self.p["c"] * ds + 3 * self.p["d"] * ds * ds
            cos_h, sin_h = math.cos(self.hdg), math.sin(self.hdg)
            return (
                self.x + ds * cos_h - v * sin_h,
                self.y + ds * sin_h + v * cos_h,
                self.hdg + math.atan2(dv, 1.0),
            )
        raise NotImplementedError(f"plan-view geometry {self.kind!r} is not evaluated here")


def _lane(element) -> dict:
    link = element.find("link")
    user = {d.get("code"): d.get("value") for d in element.findall("userData")}

    def linked(tag: str):
        if link is None or link.find(tag) is None:
            return None
        return int(link.find(tag).get("id"))

    return {
        "id": int(element.get("id")),
        "type": element.get("type"),
        "lanelet2_id": user.get("lanelet2_id"),
        "subtype": user.get("lanelet2_subtype"),
        "predecessor": linked("predecessor"),
        "successor": linked("successor"),
        "widths": sorted(
            (
                float(w.get("sOffset")),
                (float(w.get("a")), float(w.get("b")), float(w.get("c")), float(w.get("d"))),
            )
            for w in element.findall("width")
        ),
    }


class Road:
    def __init__(self, element) -> None:
        self.id = element.get("id")
        self.name = element.get("name") or ""
        self.length = float(element.get("length"))
        self.junction = element.get("junction", "-1")
        self.geometries = [Geometry(g) for g in element.find("planView").findall("geometry")]
        self._samples: list[tuple[float, tuple[float, float]]] | None = None

        lanes = element.find("lanes")
        self.offsets = sorted(
            (
                float(o.get("s")),
                (float(o.get("a")), float(o.get("b")), float(o.get("c")), float(o.get("d"))),
            )
            for o in lanes.findall("laneOffset")
        )
        self.sections: list[dict] = []
        for section in lanes.findall("laneSection"):
            entry: dict = {"s": float(section.get("s")), "left": [], "right": []}
            for side in ("left", "right"):
                group = section.find(side)
                if group is not None:
                    entry[side] = [_lane(lane) for lane in group.findall("lane")]
            entry["left"].sort(key=lambda item: item["id"])
            entry["right"].sort(key=lambda item: -item["id"])
            self.sections.append(entry)
        self.sections.sort(key=lambda item: item["s"])
        for index, section in enumerate(self.sections):
            section["end"] = (
                self.sections[index + 1]["s"] if index + 1 < len(self.sections) else self.length
            )

        link = element.find("link")
        self.predecessor = self.successor = None
        if link is not None:
            for tag in ("predecessor", "successor"):
                node = link.find(tag)
                if node is not None:
                    setattr(
                        self,
                        tag,
                        (node.get("elementType"), node.get("elementId"), node.get("contactPoint")),
                    )

    # -- geometry ----------------------------------------------------------
    def reference_at(self, s: float) -> tuple[float, float, float]:
        geometry = self.geometries[0]
        for candidate in self.geometries:
            if candidate.s <= s + 1e-9:
                geometry = candidate
        return geometry.at(max(0.0, min(s - geometry.s, geometry.length)))

    def offset_at(self, s: float) -> float:
        if not self.offsets:
            return 0.0
        chosen = self.offsets[0]
        for candidate in self.offsets:
            if candidate[0] <= s + 1e-9:
                chosen = candidate
        return poly(chosen[1], s - chosen[0])

    def station_of(self, point, step: float = 0.25) -> float:
        """Roughly where along the reference line a point projects to.

        Used only to decide whether a boundary point lies within a lane's own
        stretch of road, so a sampled nearest point is precise enough.
        """
        if self._samples is None:
            count = max(2, int(self.length / step) + 1)
            self._samples = [
                (s, self.reference_at(s)[:2])
                for s in (self.length * i / (count - 1) for i in range(count))
            ]
        return min(self._samples, key=lambda item: math.dist(point, item[1]))[0]

    @staticmethod
    def width_at(lane: dict, section_s: float, s: float) -> float:
        ds = s - section_s
        chosen = None
        for offset, coefficients in lane["widths"]:
            if offset <= ds + 1e-9:
                chosen = (offset, coefficients)
        return 0.0 if chosen is None else poly(chosen[1], ds - chosen[0])

    def stations(self, section: dict, step: float = STEP) -> list[float]:
        """Sample stations including every breakpoint in the section.

        Sampling a piecewise curve on a blind grid rounds off its corners, which
        reads as geometry error when it is only the sampling -- it cost a wrong
        conclusion once. Every geometry start and every width sOffset is therefore
        a station in its own right.
        """
        start, end = section["s"], section["end"]
        marks = {start, end}
        for geometry in self.geometries:
            if start < geometry.s < end:
                marks.add(geometry.s)
        for side in ("left", "right"):
            for lane in section[side]:
                for offset, _coefficients in lane["widths"]:
                    if start < start + offset < end:
                        marks.add(start + offset)
        ordered = sorted(marks)
        out: list[float] = []
        for a, b in pairwise(ordered):
            count = max(1, math.ceil((b - a) / step))
            out.extend(a + (b - a) * i / count for i in range(count))
        out.append(ordered[-1])
        return out

    def lane_boundaries(self, section: dict, side: str, index: int, step: float = STEP):
        """The (inner, outer) boundary polylines of one lane.

        `inner` is the boundary nearer the reference line. A right lane travels
        along `+s` and its inner boundary is on its left; a left lane travels
        against `+s` and its inner boundary is *also* on its left. So in both
        cases inner is the lanelet's left bound.
        """
        sign = 1.0 if side == "left" else -1.0
        inner: list[tuple[float, float]] = []
        outer: list[tuple[float, float]] = []
        for s in self.stations(section, step):
            x, y, hdg = self.reference_at(s)
            normal = (-math.sin(hdg), math.cos(hdg))
            t = self.offset_at(s)
            for position, lane in enumerate(section[side]):
                width = self.width_at(lane, section["s"], s)
                if position == index:
                    inner.append((x + t * normal[0], y + t * normal[1]))
                    t += sign * width
                    outer.append((x + t * normal[0], y + t * normal[1]))
                    break
                t += sign * width
        return inner, outer

    def end_touching(self, junction_id: str) -> int:
        """Which laneSection of this road meets a junction.

        Joining both ends instead would invent connections on any road with more
        than one section, which is most of them.
        """
        if self.successor is not None and self.successor[1] == junction_id:
            return len(self.sections) - 1
        if self.predecessor is not None and self.predecessor[1] == junction_id:
            return 0
        return len(self.sections) - 1


class Xodr:
    def __init__(self, path: str) -> None:
        root = ET.parse(path).getroot()
        node = root.find("header/geoReference")
        self.geo_reference = None if node is None else node.text.strip()
        self.roads = [Road(r) for r in root.findall("road")]
        self.by_id = {road.id: road for road in self.roads}
        self.junctions: dict[str, list[dict]] = {}
        for junction in root.findall("junction"):
            self.junctions[junction.get("id")] = [
                {
                    "incoming": c.get("incomingRoad"),
                    "connecting": c.get("connectingRoad"),
                    "contact": c.get("contactPoint"),
                    "lane_links": [
                        (int(link.get("from")), int(link.get("to")))
                        for link in c.findall("laneLink")
                    ],
                }
                for c in junction.findall("connection")
            ]

    def lane_of_lanelet(self) -> dict[str, tuple[Road, int, str, int]]:
        """lanelet2 id -> (road, section index, side, position within side)."""
        out: dict[str, tuple[Road, int, str, int]] = {}
        for road in self.roads:
            for index, section in enumerate(road.sections):
                for side in ("left", "right"):
                    for position, lane in enumerate(section[side]):
                        if lane["lanelet2_id"]:
                            out[lane["lanelet2_id"]] = (road, index, side, position)
        return out


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def point_to_polyline(point, polyline) -> float:
    if len(polyline) == 1:
        return math.hypot(point[0] - polyline[0][0], point[1] - polyline[0][1])
    best = float("inf")
    for a, b in pairwise(polyline):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = dx * dx + dy * dy
        t = (
            0.0
            if length == 0
            else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length))
        )
        best = min(best, math.hypot(point[0] - (a[0] + t * dx), point[1] - (a[1] + t * dy)))
    return best


def hausdorff(a, b) -> float:
    if not a or not b:
        return float("inf")
    return max(
        max(point_to_polyline(p, b) for p in a),
        max(point_to_polyline(p, a) for p in b),
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def length_of(polyline) -> float:
    return math.fsum(math.dist(a, b) for a, b in pairwise(polyline))


def report(label: str, rows: list[tuple], tolerance: float, show: int, note: str = "") -> None:
    values = [row[0] for row in rows]
    if not values:
        print(f"\n{label}: nothing to compare")
        return
    over = [row for row in rows if row[0] > tolerance]
    print(f"\n{label}  ({len(rows)} lanes){note}")
    print(
        f"  median {percentile(values, 0.5):.4f}   p90 {percentile(values, 0.90):.4f}"
        f"   p99 {percentile(values, 0.99):.4f}   max {max(values):.4f} m"
    )
    print(f"  over {tolerance} m: {len(over)} / {len(rows)}")
    for row in sorted(over, key=lambda r: -r[0])[:show]:
        print(f"    lanelet {row[1]:>22}  road {row[2]:>4} {row[3]:<5}  {row[0]:.3f} m")


def check_geometry(osm: Osm, xodr: Xodr, xy, lane_of, tolerance: float, show: int) -> None:
    print("\n== geometry: emitted lane boundaries vs the lanelet's own bounds ==")

    truth = {}
    for lanelet, (left, right, _attributes) in osm.lanelets.items():
        left_refs, right_refs = align(list(osm.ways[left]), list(osm.ways[right]), xy)
        truth[lanelet] = (
            [xy[r] for r in left_refs if r in xy],
            [xy[r] for r in right_refs if r in xy],
        )

    inner_rows, outer_rows, windowed, swapped, uneven = [], [], [], [], []
    for lanelet, (road, index, side, position) in sorted(lane_of.items()):
        if lanelet not in truth:
            continue
        section = road.sections[index]
        inner, outer = road.lane_boundaries(section, side, position)
        left_truth, right_truth = truth[lanelet]
        if len(left_truth) < 2 or len(right_truth) < 2:
            continue

        inner_rows.append((hausdorff(inner, left_truth), lanelet, road.id, side))
        outer_rows.append((hausdorff(outer, right_truth), lanelet, road.id, side))

        # Restricted to the part of the right bound inside this lane's own stretch
        # of road. Real maps carry bounds several times the length of their
        # partner, and an emitted lane spans the whole road; counting the leftover
        # as conversion error says more about the input than the conversion.
        window = [
            p
            for p in right_truth
            if section["s"] - 1.0 <= road.station_of(p) <= section["end"] + 1.0
        ]
        if window:
            windowed.append(
                (max(point_to_polyline(p, outer) for p in window), lanelet, road.id, side)
            )

        ratio = length_of(right_truth) / (length_of(left_truth) or 1.0)
        if not 0.8 <= ratio <= 1.25:
            uneven.append((lanelet, road.id, ratio))

        direct = max(hausdorff(inner, left_truth), hausdorff(outer, right_truth))
        flipped = max(hausdorff(inner, right_truth), hausdorff(outer, left_truth))
        if flipped < direct - 0.05:
            swapped.append((direct, flipped, lanelet, road.id))

    report("left bound (the innermost lane's is the reference line)", inner_rows, tolerance, show)
    report(
        "right bound (reference + fitted width), points inside the lane's own stretch",
        windowed,
        tolerance,
        show,
    )
    report(
        "right bound, two-sided",
        outer_rows,
        tolerance,
        show,
        "  -- also counts the emitted lane running the full road length",
    )

    if uneven:
        print(f"\n  lanelets whose two bounds cover different stretches: {len(uneven)}")
        for lanelet, road_id, ratio in sorted(uneven, key=lambda r: -abs(math.log(r[2])))[:show]:
            print(f"    lanelet {lanelet:>22} road {road_id:>4}: right/left length {ratio:.2f}")

    print("\n== direction ==")
    if not swapped:
        print("  no lane fits better with left and right swapped")
    else:
        print(f"  !! {len(swapped)} lane(s) fit better swapped, which means an inverted direction:")
        for direct, flipped, lanelet, road_id in sorted(swapped, reverse=True)[:show]:
            print(f"    lanelet {lanelet:>22} road {road_id:>4}: {direct:.3f} m -> {flipped:.3f} m")


def check_connectivity(osm: Osm, xodr: Xodr, xy, lane_of, show: int) -> None:
    print("\n== lane connectivity ==")

    ends = {}
    for lanelet, (left, right, _attributes) in osm.lanelets.items():
        left_refs, right_refs = align(list(osm.ways[left]), list(osm.ways[right]), xy)
        if len(left_refs) >= 2 and len(right_refs) >= 2:
            ends[lanelet] = ((left_refs[0], right_refs[0]), (left_refs[-1], right_refs[-1]))

    by_front = defaultdict(list)
    for lanelet, (front, _back) in ends.items():
        by_front[front].append(lanelet)
    successions = [
        (a, b) for a, (_front, back) in ends.items() for b in by_front.get(back, ()) if a != b
    ]

    # Every lane-to-lane connection the file states. Only what it states: matching
    # equal lane ids as a fallback would invent the very thing being measured.
    connected = defaultdict(set)

    def join(a, b) -> None:
        connected[a].add(b)
        connected[b].add(a)

    for road in xodr.roads:
        for index, section in enumerate(road.sections[:-1]):
            following = road.sections[index + 1]
            for side in ("left", "right"):
                for lane in section[side]:
                    if lane["successor"] is not None:
                        join((road.id, index, lane["id"]), (road.id, index + 1, lane["successor"]))
                for lane in following[side]:
                    if lane["predecessor"] is not None:
                        join(
                            (road.id, index, lane["predecessor"]), (road.id, index + 1, lane["id"])
                        )

    for junction_id, connections in xodr.junctions.items():
        for connection in connections:
            incoming, connecting = connection["incoming"], connection["connecting"]
            if incoming not in xodr.by_id or connecting not in xodr.by_id:
                continue
            near = xodr.by_id[incoming].end_touching(junction_id)
            far = len(xodr.by_id[connecting].sections) - 1 if connection["contact"] == "end" else 0
            for source, target in connection["lane_links"]:
                join((incoming, near, source), (connecting, far, target))

    for road in xodr.roads:
        for other, own, tag in (
            (road.predecessor, 0, "predecessor"),
            (road.successor, len(road.sections) - 1, "successor"),
        ):
            if other is None or other[0] != "road" or other[1] not in xodr.by_id:
                continue
            neighbour = xodr.by_id[other[1]]
            far = len(neighbour.sections) - 1 if other[2] == "end" else 0
            for side in ("left", "right"):
                for lane in road.sections[own][side]:
                    # At a road's outer end the lane's own <link> names the lane it
                    # continues into on the neighbouring road, which need not carry
                    # the same id -- matching ids would miss exactly the cases where
                    # the lane count changes across the join.
                    if lane[tag] is not None:
                        join((road.id, own, lane["id"]), (neighbour.id, far, lane[tag]))

    def address(lanelet: str):
        if lanelet not in lane_of:
            return None
        road, index, side, position = lane_of[lanelet]
        return (road.id, index, road.sections[index][side][position]["id"])

    verdicts: Counter[str] = Counter()
    missing: list[tuple] = []
    for a, b in successions:
        first, second = address(a), address(b)
        if first is None or second is None:
            absent = [
                osm.lanelets[x][2].get("subtype", "?")
                for x, found in ((a, first), (b, second))
                if found is None
            ]
            verdicts[f"an end never became a lane ({'/'.join(sorted(set(absent)))})"] += 1
        elif second in connected.get(first, ()):
            verdicts["stated in the .xodr"] += 1
        elif first[0] == second[0]:
            verdicts["same road, but no lane link"] += 1
            missing.append((a, b, first, second))
        else:
            verdicts["different roads, no link at all"] += 1
            missing.append((a, b, first, second))

    print(f".osm successions (shared start and end nodes): {len(successions)}")
    for label, count in verdicts.most_common():
        print(f"  {count:>5}  {label}")
    for a, b, first, second in missing[:show]:
        print(f"    {a} -> {b}   {first} -> {second}")

    lanelet_at = {address(lanelet): lanelet for lanelet in lane_of if address(lanelet)}
    real = {(a, b) for a, b in successions} | {(b, a) for a, b in successions}
    invented = [
        (lanelet_at[first], lanelet_at[second], first, second)
        for first, others in connected.items()
        for second in others
        if first in lanelet_at
        and second in lanelet_at
        and (lanelet_at[first], lanelet_at[second]) not in real
    ]
    print(f"\n  links the .xodr states with no succession behind them: {len(invented)}")
    for a, b, first, second in invented[:show]:
        print(f"    {a} <-> {b}   {first} <-> {second}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("osm", help="the source lanelet2 .osm map")
    parser.add_argument("xodr", help="the .xodr the transpiler produced from it")
    parser.add_argument(
        "--tolerance", type=float, default=0.1, help="metres before a lane counts as deviating"
    )
    parser.add_argument("--show", type=int, default=8, help="worst cases to list per section")
    args = parser.parse_args()

    osm = Osm(args.osm)
    xodr = Xodr(args.xodr)
    if xodr.geo_reference is None:
        raise SystemExit("the .xodr has no <geoReference>, so the two cannot be compared")
    xy = osm.project(xodr.geo_reference)
    lane_of = xodr.lane_of_lanelet()

    print(f"geoReference : {xodr.geo_reference}")
    print(f".osm         : {len(osm.nodes)} nodes, {len(osm.lanelets)} lanelets")
    print(f".xodr        : {len(xodr.roads)} roads, {len(xodr.junctions)} junctions")

    if not lane_of:
        raise SystemExit(
            "no lane carries a <userData code='lanelet2_id'> record, so nothing can be "
            "matched up; this needs an .xodr from opendrive-transpiler 0.2 or later"
        )
    print(f"lanelets claimed by an emitted lane: {len(lane_of)} / {len(osm.lanelets)}")
    unclaimed = sorted(set(osm.lanelets) - set(lane_of))
    if unclaimed:
        subtypes: Counter[str] = Counter(
            osm.lanelets[lanelet][2].get("subtype", "?") for lanelet in unclaimed
        )
        print(f"  unclaimed by subtype: {dict(subtypes)}")

    check_geometry(osm, xodr, xy, lane_of, args.tolerance, args.show)
    check_connectivity(osm, xodr, xy, lane_of, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
