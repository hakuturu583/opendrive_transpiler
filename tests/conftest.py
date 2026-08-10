"""Shared test fixtures and the golden-file update switch."""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS = Path(__file__).parent
FIXTURES = TESTS / "fixtures"
GOLDEN = TESTS / "golden"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="rewrite tests/golden/*.expected.py from the current output",
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))


def fixture_paths() -> list[Path]:
    return sorted(FIXTURES.glob("*.py"))


def fixture_ids() -> list[str]:
    return [path.stem for path in fixture_paths()]


@pytest.fixture(params=fixture_paths(), ids=fixture_ids())
def fixture_path(request: pytest.FixtureRequest) -> Path:
    return request.param


def has_scenariogeneration() -> bool:
    try:
        import scenariogeneration  # noqa: F401
    except ImportError:
        return False
    return True


needs_emit = pytest.mark.skipif(
    not has_scenariogeneration(),
    reason="requires the [emit] extra (scenariogeneration)",
)
