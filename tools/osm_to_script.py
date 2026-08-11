"""Turn a lanelet2 .osm into the simple_lanelet2 script that would build it.

The transpiler converts maps a *script* builds, so this is the bridge that lets a
real surveyed map reach it. Coordinates are projected exactly the way
UtmProjector(origin, useOffset=True) does, so the emitted script is what someone
would have written by hand.
"""

import pathlib
import sys
import xml.etree.ElementTree as ET
from itertools import pairwise

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from opendrive_transpiler.mapping.proj import utm_forward, utm_zone

src, out = sys.argv[1], sys.argv[2]
root = ET.parse(src).getroot()


def tags(el):
    return {t.get("k"): t.get("v") for t in el.findall("tag")}


nodes, ways, rels = {}, {}, []
for n in root.findall("node"):
    nodes[n.get("id")] = (float(n.get("lat")), float(n.get("lon")), tags(n))
for w in root.findall("way"):
    ways[w.get("id")] = ([nd.get("ref") for nd in w.findall("nd")], tags(w))
for r in root.findall("relation"):
    rels.append(
        (
            r.get("id"),
            [(m.get("type"), m.get("ref"), m.get("role")) for m in r.findall("member")],
            tags(r),
        )
    )

lat0 = sum(v[0] for v in nodes.values()) / len(nodes)
lon0 = sum(v[1] for v in nodes.values()) / len(nodes)
zone = utm_zone(lon0)
e0, n0 = utm_forward(lat0, lon0, zone)

L = [
    "from lanelet2.core import (",
    "    Area, Lanelet, LineString3d, Point3d, RightOfWay, SpeedLimit,",
    "    TrafficLight, TrafficSign, createMapFromLanelets,",
    ")",
    "from lanelet2.io import Origin",
    "from lanelet2.projection import UtmProjector",
    "",
    f"projector = UtmProjector(Origin({lat0!r}, {lon0!r}))",
    "",
]

for nid, (lat, lon, t) in nodes.items():
    e, n = utm_forward(lat, lon, zone)
    z = float(t.get("ele", 0.0))
    L.append(f"n{nid} = Point3d({nid}, {e - e0!r}, {n - n0!r}, {z!r})")
L.append("")

used = set()
for _rid, members, _t in rels:
    for mtype, ref, _role in members:
        if mtype == "way":
            used.add(ref)

for wid in sorted(used):
    if wid not in ways:
        continue
    refs, t = ways[wid]
    pts = ", ".join(f"n{r}" for r in refs if r in nodes)
    L.append(f"w{wid} = LineString3d({wid}, [{pts}])")
    for k, v in t.items():
        L.append(f"w{wid}.attributes[{k!r}] = {v!r}")
L.append("")


def xy(ref):
    lat, lon, _t = nodes[ref]
    e, n = utm_forward(lat, lon, zone)
    return (e - e0, n - n0)


def middle(refs):
    pts = [xy(r) for r in refs if r in nodes]
    return pts[len(pts) // 2] if pts else None


def signed_distance_2d(refs, point):
    """Which side of a polyline a point lies on: >0 left, <0 right."""
    pts = [xy(r) for r in refs if r in nodes]
    if len(pts) < 2:
        return 0.0
    best, best_d = 0.0, float("inf")
    for a, b in pairwise(pts):
        dx, dy = b[0] - a[0], b[1] - a[1]
        seg = dx * dx + dy * dy
        t = (
            0.0
            if seg == 0
            else max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / seg))
        )
        px, py = a[0] + t * dx, a[1] + t * dy
        d = (point[0] - px) ** 2 + (point[1] - py) ** 2
        if d < best_d:
            best_d = d
            best = dx * (point[1] - a[1]) - dy * (point[0] - a[0])
    return best


def align(lrefs, rrefs):
    """The orientation step every real lanelet2 loader performs on load.

    A way's stored node order is not authoritative for the lanelet that uses it --
    two lanelets share one boundary and traverse it opposite ways -- so the loader
    re-orients each bound against the other. Ported from
    simple_lanelet2 crates/ll2-io/src/load.rs:align.
    """
    if len(lrefs) <= 1 or len(rrefs) <= 1:
        return lrefs, rrefs
    m = middle(rrefs)
    if m is not None and not (signed_distance_2d(lrefs, m) < 0.0):
        lrefs = list(reversed(lrefs))
    m = middle(lrefs)
    if m is not None and not (signed_distance_2d(rrefs, m) > 0.0):
        rrefs = list(reversed(rrefs))
    return lrefs, rrefs


lanelets, made = [], set()
for rid, members, t in rels:
    if t.get("type") != "lanelet":
        continue
    left = next((r for mt, r, role in members if role == "left" and mt == "way"), None)
    right = next((r for mt, r, role in members if role == "right" and mt == "way"), None)
    if left not in ways or right not in ways:
        continue
    lrefs, rrefs = align(list(ways[left][0]), list(ways[right][0]))
    for side, wid, refs in (("l", left, lrefs), ("r", right, rrefs)):
        pts = ", ".join(f"n{r}" for r in refs if r in nodes)
        L.append(f"b{rid}{side} = LineString3d({wid}, [{pts}])")
        for k, v in ways[wid][1].items():
            L.append(f"b{rid}{side}.attributes[{k!r}] = {v!r}")
    L.append(f"ll{rid} = Lanelet({rid}, b{rid}l, b{rid}r)")
    for k, v in t.items():
        L.append(f"ll{rid}.attributes[{k!r}] = {v!r}")
    lanelets.append(f"ll{rid}")
    made.add(rid)

L.append("")
for rid, members, t in rels:
    if t.get("type") != "multipolygon":
        continue
    outer = [f"w{r}" for mt, r, role in members if role == "outer" and mt == "way" and r in ways]
    inner = [f"w{r}" for mt, r, role in members if role == "inner" and mt == "way" and r in ways]
    if not outer:
        continue
    L.append(
        f"a{rid} = Area({rid}, [{', '.join(outer)}], [[{', '.join(inner)}]])"
        if inner
        else f"a{rid} = Area({rid}, [{', '.join(outer)}])"
    )
    for k, v in t.items():
        L.append(f"a{rid}.attributes[{k!r}] = {v!r}")

L.append("")
L.append(f"lanelet_map = createMapFromLanelets([{', '.join(lanelets)}])")
pathlib.Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
print(
    f"wrote {out}: {len(L)} lines, {len(nodes)} points, "
    f"{len(used)} line strings, {len(lanelets)} lanelets"
)
