"""Golden-file tests over the fixture corpus.

Byte comparison, deliberately. The generated script is a deliverable a human
reads, so a change to its comments or layout is a change worth reviewing, not
something to normalise away. Regenerate with:

    pytest tests/integration/test_golden.py --update-golden
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from opendrive_transpiler import TranspileOptions, transpile

TESTS = Path(__file__).resolve().parents[1]
GOLDEN = TESTS / "golden"


def options_for(path: Path) -> TranspileOptions:
    # Fixtures include deliberately imperfect maps, so warnings must not abort.
    return TranspileOptions(strict=False, name=path.stem)


def test_golden_matches(fixture_path: Path, update_golden: bool):
    result = transpile(fixture_path, options=options_for(fixture_path))
    expected_path = GOLDEN / f"{fixture_path.stem}.expected.py"

    if not result.code:
        # A fixture that cannot convert records its diagnostics instead.
        expected_path = GOLDEN / f"{fixture_path.stem}.expected.txt"
        actual = "\n".join(f"{d.code}: {d.message}" for d in result.diagnostics) + "\n"
    else:
        actual = result.code
        ast.parse(actual, filename=str(expected_path))

    # The absolute fixture path varies by checkout, so it is normalised out.
    actual = actual.replace(str(fixture_path), f"tests/fixtures/{fixture_path.name}")

    if update_golden:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(actual, encoding="utf-8")
        pytest.skip(f"updated {expected_path.name}")

    assert expected_path.exists(), f"no golden for {fixture_path.name}; run pytest --update-golden"
    assert actual == expected_path.read_text(encoding="utf-8")


def test_every_generated_golden_is_valid_python():
    for path in sorted(GOLDEN.glob("*.expected.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_transpiling_is_deterministic(fixture_path: Path):
    """Two runs over one input must agree byte for byte."""
    first = transpile(fixture_path, options=options_for(fixture_path))
    second = transpile(fixture_path, options=options_for(fixture_path))
    assert first.code == second.code
    assert [d.code for d in first.diagnostics] == [d.code for d in second.diagnostics]
