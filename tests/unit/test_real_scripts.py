"""Shapes that real generator scripts turned out to have.

Every case here comes from a script someone actually wrote and tried to convert,
rather than from imagining what one might contain. Each was reported as an issue
against 0.1.0, and each converted to nothing -- or to nothing at all -- before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opendrive_transpiler import TranspileOptions, transpile

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def convert(path: Path, **kwargs):
    options = TranspileOptions(strict=False, name=path.stem, **kwargs)
    return transpile(path, options=options)


def codes(result) -> set[str]:
    return {d.code for d in result.diagnostics}


def write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


LANELET = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
left = LineString3d(getId(), [Point3d(getId(), 0, 3, 0), Point3d(getId(), 20, 3, 0)])
right = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 20, 0, 0)])
ll = Lanelet(getId(), left, right); ll.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([ll])
"""

HELPER = """
from lanelet2.core import Lanelet, LineString3d, Point3d, getId
def mk(y0, y1, length=20.0):
    left = LineString3d(getId(), [Point3d(getId(), 0, y1, 0), Point3d(getId(), length, y1, 0)])
    right = LineString3d(getId(), [Point3d(getId(), 0, y0, 0), Point3d(getId(), length, y0, 0)])
    ll = Lanelet(getId(), left, right); ll.attributes["subtype"] = "road"
    return ll
"""


# --------------------------------------------------------------------------
# Module dunders
# --------------------------------------------------------------------------


def test_a_script_may_use_dunder_file(tmp_path: Path):
    """It only locates the output, which the conversion does not care about."""
    script = write(
        tmp_path,
        "uses_file.py",
        "import pathlib\n" + LANELET + "_ = pathlib.Path(__file__).parent\n",
    )
    result = convert(script)
    assert "LL2ODR-E403" not in codes(result)
    assert result.stats.roads == 1


def test_dunder_file_is_the_input_path(tmp_path: Path):
    script = write(tmp_path, "named.py", LANELET + "probe = __file__\n")
    assert convert(script).ok


def test_dunder_name_still_runs_the_main_guard(tmp_path: Path):
    """It is "__main__" on purpose -- a guarded build has to actually build."""
    script = write(
        tmp_path,
        "guarded.py",
        "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
        "def main():\n"
        "    left = LineString3d(getId(), [Point3d(getId(), 0, 3, 0),"
        " Point3d(getId(), 20, 3, 0)])\n"
        "    right = LineString3d(getId(), [Point3d(getId(), 0, 0, 0),"
        " Point3d(getId(), 20, 0, 0)])\n"
        "    ll = Lanelet(getId(), left, right)\n"
        "    ll.attributes['subtype'] = 'road'\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
    )
    assert convert(script).stats.roads == 1


# --------------------------------------------------------------------------
# Local modules
# --------------------------------------------------------------------------


def test_a_sibling_module_is_resolved(tmp_path: Path):
    """The single biggest blocker: helpers in a file next to the script."""
    write(tmp_path, "_helper_mod.py", HELPER)
    script = write(
        tmp_path,
        "main.py",
        "from lanelet2.core import createMapFromLanelets\n"
        "from _helper_mod import mk\n"
        "lanelet_map = createMapFromLanelets([mk(0.0, 3.0)])\n",
    )
    result = convert(script)
    assert result.stats.lanelets_converted == 1
    assert "LL2ODR-W504" not in codes(result)


def test_resolving_one_is_recorded_as_provenance(tmp_path: Path):
    """The converted map depends on that second file; the header should say so."""
    write(tmp_path, "_helper_mod.py", HELPER)
    script = write(
        tmp_path,
        "main.py",
        "from lanelet2.core import createMapFromLanelets\n"
        "from _helper_mod import mk\n"
        "lanelet_map = createMapFromLanelets([mk(0.0, 3.0)])\n",
    )
    result = convert(script)
    assert "LL2ODR-I305" in codes(result)
    assert "_helper_mod.py" in result.code


@pytest.mark.parametrize(
    "importer",
    [
        "import _helper_mod\nfrom lanelet2.core import createMapFromLanelets\n"
        "lanelet_map = createMapFromLanelets([_helper_mod.mk(0.0, 3.0)])\n",
        "from _helper_mod import mk\nfrom lanelet2.core import createMapFromLanelets\n"
        "lanelet_map = createMapFromLanelets([mk(0.0, 3.0)])\n",
        "from _helper_mod import *\nfrom lanelet2.core import createMapFromLanelets\n"
        "lanelet_map = createMapFromLanelets([mk(0.0, 3.0)])\n",
        "import _helper_mod as h\nfrom lanelet2.core import createMapFromLanelets\n"
        "lanelet_map = createMapFromLanelets([h.mk(0.0, 3.0)])\n",
    ],
    ids=["import", "from-import", "star", "aliased"],
)
def test_every_import_form_resolves(tmp_path: Path, importer: str):
    write(tmp_path, "_helper_mod.py", HELPER)
    script = write(tmp_path, "main.py", importer)
    assert convert(script).stats.lanelets_converted == 1


def test_a_package_and_its_submodule_resolve(tmp_path: Path):
    write(tmp_path, "pkg/builders.py", HELPER)
    write(tmp_path, "pkg/__init__.py", "from .builders import mk\n")
    script = write(
        tmp_path,
        "main.py",
        "from pkg import mk\nfrom pkg.builders import mk as also\n"
        "from lanelet2.core import createMapFromLanelets\n"
        "lanelet_map = createMapFromLanelets([mk(0.0, 3.0), also(6.0, 9.0)])\n",
    )
    assert convert(script).stats.lanelets_converted == 2


def test_a_helper_sees_its_own_module_globals(tmp_path: Path):
    """Not the entry script's -- a function's globals are where it was defined."""
    write(
        tmp_path,
        "_widths.py",
        "WIDTH = 4.0\n"
        + HELPER.replace(
            "def mk(y0, y1, length=20.0):",
            "def mk(y0, y1=None, length=20.0):\n    y1 = y0 + WIDTH if y1 is None else y1",
        ),
    )
    script = write(
        tmp_path,
        "main.py",
        "WIDTH = 999.0\n"
        "from _widths import mk\n"
        "from lanelet2.core import createMapFromLanelets\n"
        "lanelet_map = createMapFromLanelets([mk(0.0)])\n",
    )
    result = convert(script)
    lane = result.model.roads[0].lane_sections[0].lanes[0]
    assert lane.constant_width == pytest.approx(4.0), "the helper's WIDTH, not the script's"


def test_a_circular_import_is_reported_rather_than_hanging(tmp_path: Path):
    write(tmp_path, "_a.py", "import _b\nX = 1\n")
    write(tmp_path, "_b.py", "import _a\nY = 2\n")
    script = write(tmp_path, "main.py", "import _a\n" + LANELET)
    result = convert(script)
    assert "LL2ODR-W306" in codes(result)
    assert result.stats.roads == 1, "the rest of the script still converts"


def test_a_name_the_module_does_not_define_is_reported(tmp_path: Path):
    write(tmp_path, "_helper_mod.py", HELPER)
    script = write(tmp_path, "main.py", "from _helper_mod import nope\n" + LANELET)
    assert "LL2ODR-E301" in codes(convert(script))


def test_a_third_party_import_is_still_left_alone(tmp_path: Path):
    """Only the script's own directory is searched; the world is not interpreted."""
    script = write(tmp_path, "main.py", "import numpy\nimport requests\n" + LANELET)
    result = convert(script)
    assert "LL2ODR-I305" not in codes(result)
    assert result.stats.roads == 1


# --------------------------------------------------------------------------
# Corner-pivot turns
# --------------------------------------------------------------------------

PIVOT = """
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
corner = Point3d(getId(), 0, 0, 0)
inner = LineString3d(getId(), [corner, corner])
outer = LineString3d(getId(), [Point3d(getId(), 0, -3, 0), Point3d(getId(), 2.1, -2.1, 0),
                               Point3d(getId(), 3, 0, 0)])
turn = Lanelet(getId(), inner, outer); turn.attributes["subtype"] = "road"
turn.attributes["turn_direction"] = "left"
lanelet_map = createMapFromLanelets([turn])
"""


def test_a_corner_pivot_turn_converts(tmp_path: Path):
    """Its inner bound is a single point, which lanelet2 accepts and draws."""
    result = convert(write(tmp_path, "pivot.py", PIVOT))
    assert result.stats.lanelets_converted == 1
    assert result.stats.lanelets_skipped == 0

    road = result.model.roads[0]
    assert road.length > 4.0, "the reference follows the outer arc"
    lane = road.lane_sections[0].lanes[0]
    # The pie slice between the arc and the pivot is about the corner radius wide.
    assert 2.0 < lane.widths[0].a < 3.5


def test_the_pivot_reference_is_reported(tmp_path: Path):
    """It costs a direction convention, so it cannot pass without saying so."""
    result = convert(write(tmp_path, "pivot.py", PIVOT))
    assert "LL2ODR-W506" in codes(result)


def test_a_lanelet_with_no_extent_at_all_is_reported_not_dropped(tmp_path: Path):
    """Both bounds a single point: nothing to follow, and it has to say so."""
    result = convert(
        write(
            tmp_path,
            "degenerate.py",
            "from lanelet2.core import Lanelet, LineString3d, Point3d, getId\n"
            "a = Point3d(getId(), 0, 0, 0)\n"
            "b = Point3d(getId(), 0, 3, 0)\n"
            "ll = Lanelet(getId(), LineString3d(getId(), [a, a]),"
            " LineString3d(getId(), [b, b]))\n"
            "ll.attributes['subtype'] = 'road'\n",
        )
    )
    assert result.model.roads == []
    assert "LL2ODR-W502" in codes(result)


# --------------------------------------------------------------------------
# MGRS
# --------------------------------------------------------------------------

MGRS = """
from autoware_lanelet2_extension_python.projection import MGRSProjector
from lanelet2.core import Lanelet, LineString3d, Point3d, createMapFromLanelets, getId
proj = MGRSProjector()
proj.setMGRSCode("54SUE")
left = LineString3d(getId(), [Point3d(getId(), 0, 3, 0), Point3d(getId(), 20, 3, 0)])
right = LineString3d(getId(), [Point3d(getId(), 0, 0, 0), Point3d(getId(), 20, 0, 0)])
ll = Lanelet(getId(), left, right); ll.attributes["subtype"] = "road"
lanelet_map = createMapFromLanelets([ll])
"""


def test_set_mgrs_code_is_modelled(tmp_path: Path):
    """Autoware's usual pattern: a grid square instead of an origin."""
    result = convert(write(tmp_path, "mgrs.py", MGRS))
    assert "LL2ODR-E301" not in codes(result)


def test_the_grid_square_reaches_the_geo_reference(tmp_path: Path):
    result = convert(write(tmp_path, "mgrs.py", MGRS))
    reference = result.model.geo_reference
    assert reference is not None
    # 54SUE is zone 54, whose central meridian is 141.
    assert "+lon_0=141.0" in reference
    # The square's south-west corner: easting 300 km, northing 3900 km.
    assert "+x_0=200000.0" in reference
    assert "+y_0=-3900000.0" in reference


def test_a_malformed_grid_square_is_refused_rather_than_guessed(tmp_path: Path):
    result = convert(write(tmp_path, "mgrs.py", MGRS.replace('"54SUE"', '"not-a-code"')))
    assert result.model.geo_reference is None
    assert "LL2ODR-I908" in codes(result)
