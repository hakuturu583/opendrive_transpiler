"""Shadow objects: stand-ins for lanelet2 primitives during symbolic execution.

These are not a reimplementation of lanelet2. They model exactly the behaviour a
transpiler has to observe, and one piece of that behaviour is subtle enough to
drive the whole design:

    lanelet2's cross-dimension constructors *alias* rather than copy.
    `Point2d(p3d)` is a second handle onto the same storage; writing `x` through
    one is visible from the other. The same holds for `LineString3d(ls2d)` and
    for `invert()`, which returns a reversed *view*, not a reversed copy.

Downstream, topology inference asks "are these the same physical node?" -- and
storage identity is the strongest available answer, stronger than comparing
coordinates. So every shadow is a thin, cheap wrapper (`eq=False`, so identity
comparison) around a shared mutable `*Storage` object, and the wrapper carries
only the view-dependent bits: dimension, constness, inversion.

Ids are tracked, but only as provenance for generated comments. `getId()` is a
side-effecting global counter in the real library, so no static analysis can
reproduce its values; nothing downstream is allowed to depend on them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

INVAL_ID = 0


class Unknown:
    """A value that could not be resolved statically.

    Deliberately viral: arithmetic and attribute access on an Unknown yield
    Unknown rather than raising, so that a script which computes something we
    cannot follow still executes to the end and still yields whatever map it
    built along the way.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Unknown({self.reason!r})" if self.reason else "Unknown()"

    def __bool__(self) -> bool:
        # Truthiness is genuinely unknown; the interpreter checks for Unknown
        # explicitly before ever needing this, so reaching here is a bug.
        raise TypeError("truthiness of Unknown must be resolved by the interpreter")

    # Any operation on an unknown stays unknown.
    def _propagate(self, *_: object) -> Unknown:
        return UNKNOWN

    __add__ = __radd__ = __sub__ = __rsub__ = _propagate
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _propagate
    __floordiv__ = __mod__ = __pow__ = __neg__ = __pos__ = _propagate

    def __getattr__(self, _name: str) -> Unknown:
        return UNKNOWN

    def __call__(self, *_a: object, **_k: object) -> Unknown:
        return UNKNOWN

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("cannot iterate an Unknown")


UNKNOWN = Unknown()


def is_unknown(value: object) -> bool:
    return isinstance(value, Unknown)


# --------------------------------------------------------------------------
# Attributes
# --------------------------------------------------------------------------


class AttributeMap(dict):
    """`lanelet2.core.AttributeMap` -- a str->str dict.

    lanelet2 stores every semantic (lane subtype, marking style, speed limit) as
    a string tag, so this is the single most load-bearing container in the input.
    Non-string values are coerced rather than rejected: the real library would
    raise, but a coercion plus a diagnostic keeps the geometry convertible, and
    geometry is what we are here for.
    """

    def __init__(self, initial: Any = None) -> None:
        super().__init__()
        self.coercions: list[tuple[str, Any]] = []
        if isinstance(initial, dict):
            for key, value in initial.items():
                self[key] = value
        elif isinstance(initial, AttributeMap):  # pragma: no cover - dict covers it
            self.update(initial)

    def __setitem__(self, key: Any, value: Any) -> None:
        key = str(key)
        if is_unknown(value):
            self.coercions.append((key, value))
            super().__setitem__(key, "")
            return
        if not isinstance(value, str):
            self.coercions.append((key, value))
            value = _stringify(value)
        super().__setitem__(key, value)

    def keys_list(self) -> list[str]:
        return list(self.keys())


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# --------------------------------------------------------------------------
# Storage -- the shared, mutable state behind aliased handles
# --------------------------------------------------------------------------


@dataclass(eq=False)
class PointStorage:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    id: int = INVAL_ID
    attributes: AttributeMap = field(default_factory=AttributeMap)


@dataclass(eq=False)
class LineStringStorage:
    points: list[ShadowPoint] = field(default_factory=list)
    id: int = INVAL_ID
    attributes: AttributeMap = field(default_factory=AttributeMap)


# --------------------------------------------------------------------------
# Shadows
# --------------------------------------------------------------------------


@dataclass(eq=False)
class ShadowPoint:
    storage: PointStorage
    dim: int = 3
    const: bool = False

    # -- lanelet2 surface --------------------------------------------------
    @property
    def id(self) -> int:
        return self.storage.id

    @id.setter
    def id(self, value: int) -> None:
        self.storage.id = int(value)

    @property
    def attributes(self) -> AttributeMap:
        return self.storage.attributes

    @property
    def x(self) -> float:
        return self.storage.x

    @x.setter
    def x(self, value: float) -> None:
        self.storage.x = float(value)

    @property
    def y(self) -> float:
        return self.storage.y

    @y.setter
    def y(self, value: float) -> None:
        self.storage.y = float(value)

    @property
    def z(self) -> float:
        # A Point2d is a *view* on 3d storage, so z still exists underneath; the
        # 2d handle simply does not expose it. Reads through this internal
        # property always see the real value.
        return self.storage.z

    @z.setter
    def z(self, value: float) -> None:
        self.storage.z = float(value)

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.storage.x, self.storage.y, self.storage.z)

    @property
    def xy(self) -> tuple[float, float]:
        return (self.storage.x, self.storage.y)

    def alias(self, dim: int, const: bool = False) -> ShadowPoint:
        """A second handle onto the same storage -- what `Point2d(p3d)` returns."""
        return ShadowPoint(storage=self.storage, dim=dim, const=const)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Point{self.dim}d(#{self.id}, {self.x}, {self.y}, {self.z})"


@dataclass(eq=False)
class BasicPoint:
    """`BasicPoint2d`/`BasicPoint3d` -- a plain coordinate triple, no id, no tags."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    dim: int = 3

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def _combine(self, other: Any, op: str) -> BasicPoint | Unknown:
        if isinstance(other, BasicPoint):
            ox, oy, oz = other.x, other.y, other.z
        elif isinstance(other, (int, float)):
            ox = oy = oz = float(other)
        else:
            return UNKNOWN
        if op == "+":
            return BasicPoint(self.x + ox, self.y + oy, self.z + oz, self.dim)
        if op == "-":
            return BasicPoint(self.x - ox, self.y - oy, self.z - oz, self.dim)
        if op == "*":
            return BasicPoint(self.x * ox, self.y * oy, self.z * oz, self.dim)
        if op == "/":
            if 0.0 in (ox, oy, oz):
                return UNKNOWN
            return BasicPoint(self.x / ox, self.y / oy, self.z / oz, self.dim)
        return UNKNOWN

    def __add__(self, other: Any) -> Any:
        return self._combine(other, "+")

    def __sub__(self, other: Any) -> Any:
        return self._combine(other, "-")

    def __mul__(self, other: Any) -> Any:
        return self._combine(other, "*")

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Any:
        return self._combine(other, "/")


@dataclass(eq=False)
class GPSPoint:
    lat: float = 0.0
    lon: float = 0.0
    ele: float = 0.0

    @property
    def alt(self) -> float:
        """lanelet2 exposes the elevation under both names."""
        return self.ele


@dataclass(eq=False)
class ShadowLineString:
    """A view onto `LineStringStorage`.

    `inverted` is a property of the *view*, not the storage: `ls.invert()` hands
    back a second handle onto the same points that iterates them backwards.
    """

    storage: LineStringStorage
    dim: int = 3
    const: bool = False
    hybrid: bool = False
    polygon: bool = False
    inverted_view: bool = False

    @property
    def id(self) -> int:
        return self.storage.id

    @id.setter
    def id(self, value: int) -> None:
        self.storage.id = int(value)

    @property
    def attributes(self) -> AttributeMap:
        return self.storage.attributes

    @property
    def points(self) -> list[ShadowPoint]:
        """Points in *view* order."""
        pts = self.storage.points
        return list(reversed(pts)) if self.inverted_view else list(pts)

    def __len__(self) -> int:
        return len(self.storage.points)

    def __iter__(self) -> Iterator[ShadowPoint]:
        return iter(self.points)

    def __getitem__(self, index: Any) -> Any:
        return self.points[index]

    def append(self, point: ShadowPoint) -> None:
        if self.inverted_view:
            self.storage.points.insert(0, point)
        else:
            self.storage.points.append(point)

    def invert(self) -> ShadowLineString:
        return ShadowLineString(
            storage=self.storage,
            dim=self.dim,
            const=self.const,
            hybrid=self.hybrid,
            polygon=self.polygon,
            inverted_view=not self.inverted_view,
        )

    def inverted(self) -> bool:
        return self.inverted_view

    def alias(self, dim: int, const: bool = False, hybrid: bool = False) -> ShadowLineString:
        return ShadowLineString(
            storage=self.storage,
            dim=dim,
            const=const,
            hybrid=hybrid,
            polygon=self.polygon,
            inverted_view=self.inverted_view,
        )

    def coords(self) -> list[tuple[float, float, float]]:
        return [p.xyz for p in self.points]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        kind = "Polygon" if self.polygon else "LineString"
        return f"{kind}{self.dim}d(#{self.id}, {len(self)} pts)"


@dataclass(eq=False)
class ShadowLanelet:
    id: int = INVAL_ID
    left: ShadowLineString | None = None
    right: ShadowLineString | None = None
    attributes: AttributeMap = field(default_factory=AttributeMap)
    regelems: list[Any] = field(default_factory=list)
    centerline_override: ShadowLineString | None = None
    inverted_view: bool = False

    @property
    def leftBound(self) -> ShadowLineString | None:
        return self.left

    @property
    def rightBound(self) -> ShadowLineString | None:
        return self.right

    @property
    def regulatoryElements(self) -> list[Any]:
        return self.regelems

    def invert(self) -> ShadowLanelet:
        """Swap the bounds *and* reverse each, matching lanelet2."""
        inv = ShadowLanelet(
            id=self.id,
            left=self.right.invert() if self.right is not None else None,
            right=self.left.invert() if self.left is not None else None,
            attributes=self.attributes,
            regelems=self.regelems,
            inverted_view=not self.inverted_view,
        )
        if self.centerline_override is not None:
            inv.centerline_override = self.centerline_override.invert()
        return inv

    def inverted(self) -> bool:
        return self.inverted_view

    def addRegulatoryElement(self, value: Any) -> None:
        self.regelems.append(value)

    def removeRegulatoryElement(self, value: Any) -> bool:
        for i, existing in enumerate(self.regelems):
            if existing is value:
                del self.regelems[i]
                return True
        return False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Lanelet(#{self.id}, subtype={self.attributes.get('subtype', '')!r})"


@dataclass(eq=False)
class ShadowArea:
    id: int = INVAL_ID
    outer: list[ShadowLineString] = field(default_factory=list)
    inners: list[list[ShadowLineString]] = field(default_factory=list)
    attributes: AttributeMap = field(default_factory=AttributeMap)
    regelems: list[Any] = field(default_factory=list)

    @property
    def outerBound(self) -> list[ShadowLineString]:
        return self.outer

    @property
    def innerBounds(self) -> list[list[ShadowLineString]]:
        return self.inners

    @property
    def regulatoryElements(self) -> list[Any]:
        return self.regelems

    def addRegulatoryElement(self, value: Any) -> None:
        self.regelems.append(value)

    def removeRegulatoryElement(self, value: Any) -> bool:
        for i, existing in enumerate(self.regelems):
            if existing is value:
                del self.regelems[i]
                return True
        return False


@dataclass(eq=False)
class ShadowRegulatoryElement:
    """Any regulatory element, typed only by `kind`.

    The typed lanelet2 classes are conveniences over one role->parameters map, so
    modelling them uniformly costs nothing and covers types we have not enumerated.
    """

    kind: str = "RegulatoryElement"
    id: int = INVAL_ID
    attributes: AttributeMap = field(default_factory=AttributeMap)
    parameters: dict[str, list[Any]] = field(default_factory=dict)

    @property
    def roles(self) -> list[str]:
        return list(self.parameters)

    def role(self, name: str) -> list[Any]:
        return self.parameters.get(name, [])

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{self.kind}(#{self.id})"


@dataclass(eq=False)
class ShadowLaneletWithStopLine:
    lanelet: Any = None
    stop_line: Any = None


@dataclass(eq=False)
class ShadowLayer:
    """One of a LaneletMap's six layers.

    Backed by a list rather than a dict because ids are unreliable here (see the
    module docstring); membership is by object identity.
    """

    name: str
    items: list[Any] = field(default_factory=list)

    def add(self, value: Any) -> None:
        if not any(existing is value for existing in self.items):
            self.items.append(value)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.items)

    def __contains__(self, value: Any) -> bool:
        if isinstance(value, int):
            return any(getattr(i, "id", None) == value for i in self.items)
        return any(existing is value for existing in self.items)

    def get(self, ident: int) -> Any:
        for item in self.items:
            if getattr(item, "id", None) == ident:
                return item
        return UNKNOWN

    def exists(self, ident: int) -> bool:
        return any(getattr(i, "id", None) == ident for i in self.items)

    def __getitem__(self, ident: int) -> Any:
        return self.get(ident)


LAYER_NAMES = (
    "pointLayer",
    "lineStringLayer",
    "polygonLayer",
    "laneletLayer",
    "areaLayer",
    "regulatoryElementLayer",
)


@dataclass(eq=False)
class ShadowMap:
    """`LaneletMap` / `LaneletSubmap`.

    `add()` mirrors the real thing: it pulls in everything the primitive
    references and hands a fresh id to anything still carrying id 0.
    """

    submap: bool = False
    layers: dict[str, ShadowLayer] = field(default_factory=dict)
    _id_source: Any = None

    def __post_init__(self) -> None:
        if not self.layers:
            self.layers = {name: ShadowLayer(name) for name in LAYER_NAMES}

    # -- layer properties --------------------------------------------------
    @property
    def pointLayer(self) -> ShadowLayer:
        return self.layers["pointLayer"]

    @property
    def lineStringLayer(self) -> ShadowLayer:
        return self.layers["lineStringLayer"]

    @property
    def polygonLayer(self) -> ShadowLayer:
        return self.layers["polygonLayer"]

    @property
    def laneletLayer(self) -> ShadowLayer:
        return self.layers["laneletLayer"]

    @property
    def areaLayer(self) -> ShadowLayer:
        return self.layers["areaLayer"]

    @property
    def regulatoryElementLayer(self) -> ShadowLayer:
        return self.layers["regulatoryElementLayer"]

    @property
    def lanelets(self) -> list[ShadowLanelet]:
        return list(self.layers["laneletLayer"].items)

    # -- mutation ----------------------------------------------------------
    def add(self, value: Any, assign_id: Any = None) -> None:
        assign = assign_id or self._id_source
        self._add(value, assign, set())

    def _add(self, value: Any, assign: Any, seen: set[int]) -> None:
        if value is None or is_unknown(value) or id(value) in seen:
            return
        seen.add(id(value))

        def fresh(obj: Any) -> None:
            # Submaps do not own ids; only a full map hands them out.
            if assign is not None and not self.submap and getattr(obj, "id", None) == INVAL_ID:
                obj.id = assign()

        if isinstance(value, ShadowPoint):
            fresh(value)
            self.layers["pointLayer"].add(value)
        elif isinstance(value, ShadowLineString):
            fresh(value)
            layer = "polygonLayer" if value.polygon else "lineStringLayer"
            self.layers[layer].add(value)
            for point in value.storage.points:
                self._add(point, assign, seen)
        elif isinstance(value, ShadowLanelet):
            fresh(value)
            self.layers["laneletLayer"].add(value)
            for bound in (value.left, value.right):
                self._add(bound, assign, seen)
            for regelem in value.regelems:
                self._add(regelem, assign, seen)
        elif isinstance(value, ShadowArea):
            fresh(value)
            self.layers["areaLayer"].add(value)
            for ring in [value.outer, *value.inners]:
                for bound in ring:
                    self._add(bound, assign, seen)
            for regelem in value.regelems:
                self._add(regelem, assign, seen)
        elif isinstance(value, ShadowRegulatoryElement):
            fresh(value)
            self.layers["regulatoryElementLayer"].add(value)
            for members in value.parameters.values():
                for member in members:
                    self._add(member, assign, seen)

    def laneletMap(self) -> ShadowMap:
        return self


@dataclass(eq=False)
class ProjectionInfo:
    """A captured projector, kept so the origin can reach `<geoReference>`."""

    kind: str
    lat: float = 0.0
    lon: float = 0.0
    alt: float = 0.0
    use_offset: bool = True
    mgrs_code: str = ""
    """Set by `setMGRSCode`, which names a grid square instead of an origin."""


@dataclass(eq=False)
class Origin:
    position: GPSPoint = field(default_factory=GPSPoint)


@dataclass(eq=False)
class OpaqueValue:
    """A recognised lanelet2 object we deliberately do not model (a routing graph,
    traffic rules, a matcher). Distinct from Unknown so diagnostics can say which."""

    kind: str

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{self.kind}>"


@dataclass(eq=False)
class BoundingBox:
    """`BoundingBox2d`/`BoundingBox3d` -- a min/max corner pair."""

    min: BasicPoint = field(default_factory=BasicPoint)
    max: BasicPoint = field(default_factory=BasicPoint)
    dim: int = 2

    def contains(self, point: tuple[float, float, float]) -> bool:
        if not (self.min.x <= point[0] <= self.max.x):
            return False
        if not (self.min.y <= point[1] <= self.max.y):
            return False
        return not (self.dim == 3 and not (self.min.z <= point[2] <= self.max.z))


@dataclass(eq=False)
class ShadowCompound:
    """`CompoundLineString*` / `CompoundPolygon*` -- several line strings read as one.

    A view, not a copy: the members keep their own identity, which is what
    `lineStrings()` hands back and what topology would still match on.
    """

    members: list[ShadowLineString] = field(default_factory=list)
    dim: int = 3
    hybrid: bool = False
    polygon: bool = False
    inverted_view: bool = False

    @property
    def points(self) -> list[ShadowPoint]:
        """Members chained end to start, with the shared joints collapsed."""
        out: list[ShadowPoint] = []
        members = list(reversed(self.members)) if self.inverted_view else self.members
        for member in members:
            for point in member.points:
                if not out or out[-1].storage is not point.storage:
                    out.append(point)
        return out

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[ShadowPoint]:
        return iter(self.points)

    def __getitem__(self, index: Any) -> Any:
        return self.points[index]

    def ids(self) -> list[int]:
        return [member.id for member in self.members]

    def lineStrings(self) -> list[ShadowLineString]:
        return list(self.members)

    def numSegments(self) -> int:
        return max(len(self.points) - 1, 0)

    def invert(self) -> ShadowCompound:
        return ShadowCompound(
            members=self.members,
            dim=self.dim,
            hybrid=self.hybrid,
            polygon=self.polygon,
            inverted_view=not self.inverted_view,
        )

    def inverted(self) -> bool:
        return self.inverted_view


@dataclass(eq=False)
class ShadowLaneletSequence:
    """`LaneletSequence` -- a run of lanelets read as one long lanelet."""

    members: list[ShadowLanelet] = field(default_factory=list)
    inverted_view: bool = False

    def _ordered(self) -> list[ShadowLanelet]:
        return list(reversed(self.members)) if self.inverted_view else self.members

    def _chain(self, side: str) -> ShadowLineString:
        points: list[ShadowPoint] = []
        for lanelet in self._ordered():
            bound = getattr(lanelet, side)
            if bound is None:
                continue
            for point in bound.points:
                if not points or points[-1].storage is not point.storage:
                    points.append(point)
        return ShadowLineString(LineStringStorage(points=points))

    @property
    def leftBound(self) -> ShadowLineString:
        return self._chain("left")

    @property
    def rightBound(self) -> ShadowLineString:
        return self._chain("right")

    def lanelets(self) -> list[ShadowLanelet]:
        return self._ordered()

    def invert(self) -> ShadowLaneletSequence:
        return ShadowLaneletSequence(self.members, not self.inverted_view)

    def inverted(self) -> bool:
        return self.inverted_view

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self) -> Iterator[ShadowLanelet]:
        return iter(self._ordered())
