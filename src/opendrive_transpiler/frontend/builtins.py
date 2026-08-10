"""The builtins and stdlib surface the symbolic executor exposes.

Everything here is a real Python callable, applied to already-evaluated values.
That is safe because the interpreter never hands it an unevaluated AST node and
never resolves a name the input script did not get from this table -- so there is
no path from input source text to an arbitrary call.

The list is deliberately small: it covers what map-building scripts actually use
(ranges, lengths, enumerate/zip, min/max, trig) and nothing that touches the
filesystem, the network, imports, or the interpreter itself.
"""

from __future__ import annotations

import math
from typing import Any

from .shadow import UNKNOWN, ShadowLineString, is_unknown


def _guard(fn: Any) -> Any:
    """Wrap a builtin so an Unknown argument yields Unknown instead of raising."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if any(is_unknown(a) for a in args) or any(is_unknown(v) for v in kwargs.values()):
            return UNKNOWN
        try:
            return fn(*args, **kwargs)
        except (TypeError, ValueError, ZeroDivisionError, IndexError, KeyError, AttributeError):
            return UNKNOWN

    return wrapper


def _len(value: Any) -> Any:
    if isinstance(value, ShadowLineString):
        return len(value)
    return len(value)


def _print(*_args: Any, **_kwargs: Any) -> None:
    """Swallowed: the input script's own output is not our output."""
    return None


def _isinstance(value: Any, _types: Any) -> Any:
    # Shadow types are not the real lanelet2 types, so an honest answer is not
    # available. Unknown lets the interpreter route the branch through its
    # unresolved-condition policy rather than silently taking a wrong one.
    return UNKNOWN


SAFE_BUILTINS: dict[str, Any] = {
    "abs": _guard(abs),
    "all": _guard(all),
    "any": _guard(any),
    "bool": _guard(bool),
    "dict": _guard(dict),
    "divmod": _guard(divmod),
    "enumerate": _guard(lambda it, start=0: list(enumerate(it, start))),
    "float": _guard(float),
    "int": _guard(int),
    "isinstance": _isinstance,
    "len": _guard(_len),
    "list": _guard(list),
    "max": _guard(max),
    "min": _guard(min),
    "print": _print,
    "range": _guard(lambda *a: list(range(*a))),
    "repr": _guard(repr),
    "reversed": _guard(lambda it: list(reversed(list(it)))),
    "round": _guard(round),
    "set": _guard(set),
    "sorted": _guard(lambda it, **kw: sorted(it, **kw)),
    "str": _guard(str),
    "sum": _guard(sum),
    "tuple": _guard(tuple),
    "zip": _guard(lambda *its: list(zip(*its, strict=False))),
}

# Modules an input script may legitimately import for coordinate arithmetic.
SAFE_MODULES: dict[str, dict[str, Any]] = {
    "math": {
        name: getattr(math, name)
        for name in (
            "acos",
            "asin",
            "atan",
            "atan2",
            "ceil",
            "copysign",
            "cos",
            "cosh",
            "degrees",
            "dist",
            "e",
            "exp",
            "fabs",
            "floor",
            "fmod",
            "hypot",
            "inf",
            "isclose",
            "isfinite",
            "isinf",
            "isnan",
            "log",
            "log10",
            "log2",
            "nan",
            "pi",
            "pow",
            "radians",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "tau",
            "trunc",
        )
    },
}

# Methods callable on plain Python values. Restricting by name keeps dunder and
# introspection attributes (`__class__`, `__globals__`, ...) out of reach.
SAFE_METHODS: dict[type, frozenset[str]] = {
    list: frozenset(
        {
            "append",
            "extend",
            "insert",
            "pop",
            "remove",
            "reverse",
            "sort",
            "index",
            "count",
            "copy",
            "clear",
        }
    ),
    dict: frozenset(
        {"get", "items", "keys", "values", "setdefault", "update", "pop", "copy", "clear"}
    ),
    set: frozenset({"add", "discard", "remove", "update", "union", "intersection", "difference"}),
    str: frozenset(
        {
            "format",
            "join",
            "split",
            "rsplit",
            "strip",
            "lstrip",
            "rstrip",
            "replace",
            "lower",
            "upper",
            "startswith",
            "endswith",
            "zfill",
            "count",
            "find",
            "index",
            "title",
            "capitalize",
            "casefold",
            "removeprefix",
            "removesuffix",
            "encode",
            "isdigit",
            "isalpha",
        }
    ),
    tuple: frozenset({"index", "count"}),
    float: frozenset({"is_integer", "hex"}),
    int: frozenset({"bit_length", "to_bytes"}),
}


def safe_method(owner: Any, name: str) -> Any:
    """Return a bound method if `name` is on the allow-list for `owner`'s type."""
    for kind, allowed in SAFE_METHODS.items():
        if isinstance(owner, kind) and name in allowed:
            return getattr(owner, name)
    return None
