"""The symbolic executor.

Walks the input script's AST and *interprets* the lanelet2 calls, producing the
map the script would have built. It does this without running the script and
without lanelet2 installed: every lanelet2 name resolves to a shadow constructor
from the registry, and every other name resolves to a literal, a user-defined
function, an allow-listed builtin, or `Unknown`.

Three design points are worth stating up front, because they explain most of the
code below:

* **Unknown is not an error.** Scripts routinely compute things we cannot follow
  (a routing query, a random seed, a value read from a file). Those become
  `Unknown` and propagate; execution continues so that the parts of the map that
  *are* statically determined still get built.

* **An unresolvable condition does not fork.** Taking both branches would yield a
  set of candidate maps with no principled way to choose between them. One branch
  plus a loud diagnostic is more useful, and the choice is configurable.

* **Every loop is bounded.** We own the interpreter loop, so iteration counts, a
  global statement budget and recursion depth are all enforced directly rather
  than left to the host interpreter to blow up on.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import TranspileOptions
from ..diagnostics import (
    E_BAD_ARITY,
    E_ITERATION_LIMIT,
    E_LOCAL_IMPORT_CYCLE,
    E_NAME_UNDEFINED,
    E_RECURSION_LIMIT,
    E_STATEMENT_BUDGET,
    E_UNBOUNDED_WHILE,
    E_UNCAUGHT_RAISE,
    E_UNKNOWN_ATTRIBUTE,
    E_UNSUPPORTED_EXPR,
    E_UNSUPPORTED_STMT,
    E_UNSUPPORTED_TARGET,
    I_LOCAL_IMPORT,
    W_NON_STRING_ATTRIBUTE,
    W_UNKNOWN_CONDITION,
    W_UNKNOWN_ITERABLE,
    DiagnosticBag,
    SourceSpan,
)
from ..ir.centerline import compute_centerline
from .builtins import SAFE_BUILTINS, SAFE_MODULES, exception_types, safe_method
from .imports import (
    MODULE_CONSTANTS,
    QUERY_MODULES,
    Args,
    ModuleRef,
    Registry,
)
from .shadow import (
    UNKNOWN,
    AttributeMap,
    BasicPoint,
    BoundingBox,
    GPSPoint,
    LineStringStorage,
    OpaqueValue,
    Origin,
    ProjectionInfo,
    ShadowArea,
    ShadowCompound,
    ShadowLanelet,
    ShadowLaneletSequence,
    ShadowLaneletWithStopLine,
    ShadowLayer,
    ShadowLineString,
    ShadowMap,
    ShadowPoint,
    ShadowRegulatoryElement,
    is_unknown,
)

# --------------------------------------------------------------------------
# Control-flow signals
# --------------------------------------------------------------------------


class _Return(Exception):
    def __init__(self, value: Any) -> None:
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _Abort(Exception):
    """A limit was hit; unwind to the top without pretending to have a result."""


class _Raise(Exception):
    """The input script raised. Carries the script's own exception value."""

    def __init__(self, value: Any) -> None:
        self.value = value


# --------------------------------------------------------------------------
# Callables
# --------------------------------------------------------------------------


@dataclass
class NativeCtor:
    """A lanelet2 constructor or free function from the registry."""

    dotted: str
    fn: Any


@dataclass
class OpaqueCallable:
    """A call into a module we deliberately do not model (routing, rules, ...)."""

    dotted: str


@dataclass
class Function:
    """A function defined in the input script."""

    node: ast.FunctionDef | ast.AsyncFunctionDef
    closure: Env
    defaults: list[Any] = field(default_factory=list)
    kw_defaults: dict[str, Any] = field(default_factory=dict)
    owner: Any = None
    """The class whose body defined this function, if any -- what `super()` skips past."""

    @property
    def name(self) -> str:
        return self.node.name


@dataclass
class Lambda:
    node: ast.Lambda
    closure: Env
    defaults: list[Any] = field(default_factory=list)


@dataclass
class BoundShadowMethod:
    owner: Any
    name: str


@dataclass(eq=False)
class PyClass:
    """A class defined in the input script.

    Deliberately minimal: a name, its bases, and the namespace its body produced.
    That is enough for the only thing map scripts do with classes -- gather some
    state in `__init__` and read it back through methods.
    """

    name: str
    bases: list[Any] = field(default_factory=list)
    namespace: dict[str, Any] = field(default_factory=dict)

    def lookup(self, name: str) -> tuple[bool, Any]:
        """Depth-first attribute lookup through the class and its bases."""
        if name in self.namespace:
            return True, self.namespace[name]
        for base in self.bases:
            if isinstance(base, PyClass):
                found, value = base.lookup(name)
                if found:
                    return True, value
        return False, None

    def is_subclass_of(self, other: Any) -> bool:
        if self is other:
            return True
        return any(isinstance(base, PyClass) and base.is_subclass_of(other) for base in self.bases)

    def derives_from(self, builtin: str) -> bool:
        """Whether a built-in exception of this name is anywhere in the bases."""
        for base in self.bases:
            if isinstance(base, ExceptionType):
                if base.name == builtin or builtin in base.bases:
                    return True
            elif isinstance(base, PyClass) and base.derives_from(builtin):
                return True
        return False


@dataclass(eq=False)
class PyInstance:
    cls: PyClass
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundMethod:
    instance: Any
    function: Any


@dataclass
class LocalModule:
    """A module from the input's own directory, interpreted rather than imported.

    Map-building scripts routinely factor their node and lanelet factories into a
    `helpers.py` alongside. Resolving those the same symbolic way as the script
    itself keeps the guarantee intact -- nothing is executed, the helper is parsed
    and interpreted exactly like the input.
    """

    name: str
    path: str
    namespace: dict[str, Any]


@dataclass
class Super:
    """What `super()` evaluates to: an instance seen through its bases.

    Attribute lookup starts *after* `owner` in the base list, which is what makes
    an override able to call the implementation it overrides.
    """

    instance: Any
    owner: Any

    def lookup(self, name: str) -> tuple[bool, Any]:
        bases = getattr(self.owner, "bases", ())
        for base in bases:
            if isinstance(base, PyClass):
                found, value = base.lookup(name)
                if found:
                    return True, value
        return False, None


@dataclass(eq=False)
class ExceptionType:
    """A built-in exception class, as a name plus the names it derives from."""

    name: str
    bases: tuple[str, ...] = ()

    def catches(self, raised: ExceptionType) -> bool:
        """Whether an `except` clause naming this type catches `raised`.

        The direction matters: `except ArithmeticError` catches a
        `ZeroDivisionError` because *the raised type* derives from this one.
        """
        return self is raised or self.name == raised.name or self.name in raised.bases


@dataclass(eq=False)
class ExceptionInstance:
    type: Any
    args: tuple[Any, ...] = ()

    @property
    def name(self) -> str:
        return getattr(self.type, "name", "Exception")


# --------------------------------------------------------------------------
# Scopes
# --------------------------------------------------------------------------


class Env:
    """A lexical scope. `globals_env` is the module scope, for `global`."""

    __slots__ = ("global_names", "globals_env", "parent", "vars")

    def __init__(self, parent: Env | None = None, globals_env: Env | None = None) -> None:
        self.vars: dict[str, Any] = {}
        self.parent = parent
        self.globals_env = globals_env or (parent.globals_env if parent else self)
        self.global_names: set[str] = set()

    def lookup(self, name: str) -> tuple[bool, Any]:
        if name in self.global_names:
            return (name in self.globals_env.vars, self.globals_env.vars.get(name))
        scope: Env | None = self
        while scope is not None:
            if name in scope.vars:
                return True, scope.vars[name]
            scope = scope.parent
        return False, None

    def assign(self, name: str, value: Any) -> None:
        if name in self.global_names:
            self.globals_env.vars[name] = value
        else:
            self.vars[name] = value

    def declare_global(self, names: Iterable[str]) -> None:
        self.global_names.update(names)


# --------------------------------------------------------------------------
# Shadow attribute access
# --------------------------------------------------------------------------
# Explicit allow-lists rather than bare getattr: this is what keeps dunder and
# introspection attributes out of reach of the input script.

_SHADOW_ATTRS: dict[type, frozenset[str]] = {
    ShadowPoint: frozenset({"id", "attributes", "x", "y", "z", "basicPoint"}),
    BasicPoint: frozenset({"x", "y", "z"}),
    GPSPoint: frozenset({"lat", "lon", "ele", "alt"}),
    Origin: frozenset({"position"}),
    ShadowLineString: frozenset({"id", "attributes", "append", "invert", "inverted"}),
    ShadowLanelet: frozenset(
        {
            "id",
            "attributes",
            "leftBound",
            "rightBound",
            "centerline",
            "regulatoryElements",
            "invert",
            "inverted",
            "addRegulatoryElement",
            "removeRegulatoryElement",
            "resetCache",
            "polygon2d",
            "polygon3d",
            "trafficLights",
            "trafficSigns",
            "speedLimits",
            "rightOfWay",
            "allWayStop",
        }
    ),
    ShadowCompound: frozenset({"ids", "lineStrings", "numSegments", "invert", "inverted"}),
    ShadowLaneletSequence: frozenset(
        {
            "lanelets",
            "leftBound",
            "rightBound",
            "centerline",
            "invert",
            "inverted",
            "polygon2d",
            "polygon3d",
        }
    ),
    BoundingBox: frozenset({"min", "max"}),
    ShadowArea: frozenset(
        {
            "id",
            "attributes",
            "outerBound",
            "innerBounds",
            "regulatoryElements",
            "addRegulatoryElement",
            "removeRegulatoryElement",
            "outerBoundPolygon",
            "innerBoundPolygons",
        }
    ),
    ShadowRegulatoryElement: frozenset(
        {
            "id",
            "attributes",
            "parameters",
            "roles",
            "find",
            "trafficLights",
            "trafficSigns",
            "stopLine",
            "refLines",
            "cancelLines",
            "lanelets",
            "stopLines",
            "rightOfWayLanelets",
            "yieldLanelets",
            "cancellingTrafficSigns",
            "type",
            "cancelTypes",
            "addTrafficLight",
            "removeTrafficLight",
            "removeStopLine",
        }
    ),
    ShadowLaneletWithStopLine: frozenset({"lanelet", "stopLine"}),
    ShadowMap: frozenset(
        {
            "add",
            "laneletMap",
            "pointLayer",
            "lineStringLayer",
            "polygonLayer",
            "laneletLayer",
            "areaLayer",
            "regulatoryElementLayer",
        }
    ),
    ShadowLayer: frozenset({"exists", "get", "search", "nearest", "uniqueId", "findUsages"}),
    ProjectionInfo: frozenset({"forward", "reverse", "origin", "setMGRSCode"}),
}

# Regulatory-element accessors, spelled as lanelet2 spells them, mapped to the
# OSM role they read. lanelet2's typed classes are conveniences over one
# role->members map, so this table is the whole implementation.
_REGELEM_ROLE_ACCESSORS: dict[str, str] = {
    "trafficLights": "refers",
    "trafficSigns": "refers",
    "lanelets": "refers",
    "rightOfWayLanelets": "right_of_way",
    "yieldLanelets": "yield",
    "cancellingTrafficSigns": "cancels",
    "refLines": "ref_line",
    "stopLines": "ref_line",
    "cancelLines": "cancel_line",
}


class Interpreter:
    """Executes an AST symbolically, producing shadow lanelet2 objects."""

    def __init__(
        self,
        filename: str,
        bag: DiagnosticBag,
        options: TranspileOptions,
        registry: Registry | None = None,
    ) -> None:
        self.filename = filename
        self.bag = bag
        self.options = options
        self.registry = registry or Registry(bag)
        self.globals = Env()
        self.aliases: dict[str, Any] = {}
        self._statements = 0
        self._depth = 0
        self._yields: list[list[Any]] = []
        self._handling: list[Any] = []
        self._exceptions = exception_types()
        self._modules: dict[str, LocalModule] = {}
        self._loading: list[str] = []

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def define_module_dunders(self, module: ast.Module) -> None:
        """Seed the globals CPython would have seeded before the first statement.

        Scripts reach for these to locate their own output file or to guard the
        build, neither of which the conversion cares about -- but an undefined
        name is an error, so omitting them turned ordinary I/O boilerplate into a
        failed run.

        `__name__` is `"__main__"` deliberately: a script that guards its build
        behind that check should have the build run.
        """
        docstring = ast.get_docstring(module, clean=False)
        for name, value in (
            ("__name__", "__main__"),
            ("__file__", self.filename),
            ("__doc__", docstring),
            ("__package__", ""),
            ("__spec__", None),
            ("__loader__", None),
        ):
            self.globals.assign(name, value)

    def run(self, module: ast.Module) -> Registry:
        self.define_module_dunders(module)
        try:
            self.exec_block(module.body, self.globals)
            # A script that guards its build behind `if __name__ == "__main__"`
            # (or just defines `main()`) has not run it yet. Call it once.
            self._run_main_if_present()
        except _Abort:
            pass
        except _Return:
            pass
        except _Raise as raised:
            # Uncaught, exactly as it would be when run: the script stops there,
            # and whatever it had already built is still worth converting.
            name = getattr(raised.value, "name", type(raised.value).__name__)
            self.bag.warn(
                E_UNCAUGHT_RAISE,
                f"the script raised {name} and did not catch it; converting the map "
                "as it stood at that point",
            )
        return self.registry

    def _run_main_if_present(self) -> None:
        found, main = self.globals.lookup("main")
        if not found or not isinstance(main, Function):
            return
        if self.registry.lanelets or self.registry.maps:
            return  # the module body already built something
        if main.node.args.args and not main.defaults:
            return  # needs arguments we cannot supply
        self.call(main, Args(), SourceSpan(self.filename, main.node.lineno, 0))

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    def span(self, node: ast.AST) -> SourceSpan:
        return SourceSpan.from_node(self.filename, node)

    def _tick(self, node: ast.AST) -> None:
        self._statements += 1
        if self._statements > self.options.max_statements:
            self.bag.error(
                E_STATEMENT_BUDGET,
                f"statement budget of {self.options.max_statements} exhausted",
                self.span(node),
            )
            raise _Abort

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------
    def exec_block(self, body: list[ast.stmt], env: Env) -> None:
        for stmt in body:
            self.exec_stmt(stmt, env)

    def exec_stmt(self, node: ast.stmt, env: Env) -> None:
        self._tick(node)

        if isinstance(node, ast.Expr):
            self.eval(node.value, env)

        elif isinstance(node, ast.Assign):
            value = self.eval(node.value, env)
            for target in node.targets:
                self.assign(target, value, env)

        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.assign(node.target, self.eval(node.value, env), env)

        elif isinstance(node, ast.AugAssign):
            current = self.eval_load_of_target(node.target, env)
            operand = self.eval(node.value, env)
            self.assign(node.target, self.binop(node.op, current, operand, node), env)

        elif isinstance(node, ast.If):
            truth = self.truthiness(self.eval(node.test, env), node, "if")
            self.exec_block(node.body if truth else node.orelse, env)

        elif isinstance(node, ast.For):
            self.exec_for(node, env)

        elif isinstance(node, ast.While):
            self.exec_while(node, env)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            env.assign(node.name, self.make_function(node, env))

        elif isinstance(node, ast.Return):
            raise _Return(self.eval(node.value, env) if node.value is not None else None)

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self.exec_import(node, env)

        elif isinstance(node, ast.Break):
            raise _Break

        elif isinstance(node, ast.Continue):
            raise _Continue

        elif isinstance(node, (ast.Pass, ast.Global, ast.Nonlocal)):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                env.declare_global(node.names)

        elif isinstance(node, ast.Assert):
            # Assertions describe the script's own invariants, not the map.
            pass

        elif isinstance(node, ast.Delete):
            for target in node.targets:
                self.delete(target, env)

        elif isinstance(node, ast.With):
            # The context manager itself cannot affect the map; the body can.
            for item in node.items:
                self.eval(item.context_expr, env)
            self.exec_block(node.body, env)

        elif isinstance(node, ast.Try):
            self.exec_try(node, env)

        elif isinstance(node, ast.Raise):
            self.exec_raise(node, env)

        elif isinstance(node, ast.ClassDef):
            env.assign(node.name, self.make_class(node, env))

        elif isinstance(node, ast.Match):
            self.exec_match(node, env)

        else:
            self.bag.error(
                E_UNSUPPORTED_STMT,
                f"unsupported statement: {type(node).__name__}",
                self.span(node),
            )

    def exec_for(self, node: ast.For, env: Env) -> None:
        iterable = self.eval(node.iter, env)
        items = self.iterate(iterable, node)
        if items is None:
            return
        completed = True
        for index, item in enumerate(items):
            if index >= self.options.max_iterations:
                self.bag.error(
                    E_ITERATION_LIMIT,
                    f"loop exceeded {self.options.max_iterations} iterations",
                    self.span(node),
                )
                raise _Abort
            self.assign(node.target, item, env)
            try:
                self.exec_block(node.body, env)
            except _Break:
                completed = False
                break
            except _Continue:
                continue
        if completed:
            self.exec_block(node.orelse, env)

    def exec_while(self, node: ast.While, env: Env) -> None:
        count = 0
        while True:
            condition = self.eval(node.test, env)
            if is_unknown(condition):
                self.bag.error(
                    E_UNBOUNDED_WHILE,
                    "while-loop condition cannot be resolved statically",
                    self.span(node),
                )
                return
            if not self._python_truth(condition):
                self.exec_block(node.orelse, env)
                return
            count += 1
            if count > self.options.max_iterations:
                self.bag.error(
                    E_ITERATION_LIMIT,
                    f"while-loop exceeded {self.options.max_iterations} iterations",
                    self.span(node),
                )
                raise _Abort
            try:
                self.exec_block(node.body, env)
            except _Break:
                return
            except _Continue:
                continue

    def exec_raise(self, node: ast.Raise, env: Env) -> None:
        if node.exc is None:
            # A bare `raise` re-raises whatever is currently being handled.
            if self._handling:
                raise _Raise(self._handling[-1])
            raise _Raise(ExceptionInstance(ExceptionType("RuntimeError", ("Exception",))))
        raise _Raise(self.as_exception(self.eval(node.exc, env)))

    @staticmethod
    def as_exception(value: Any) -> Any:
        """Normalise what was raised so a handler can match it."""
        # `raise ValueError` names the class; `raise ValueError(...)` builds one.
        if isinstance(value, ExceptionType):
            return ExceptionInstance(value)
        return value

    @staticmethod
    def _raises_as(raised: Any, candidate: Any) -> bool:
        if isinstance(candidate, ExceptionType):
            if isinstance(raised, ExceptionInstance) and isinstance(raised.type, ExceptionType):
                return candidate.catches(raised.type)
            # A script-defined exception caught by a built-in base:
            # `except Exception` over `class TooShort(Exception)`.
            if isinstance(raised, PyInstance):
                return raised.cls.derives_from(candidate.name)
        if isinstance(candidate, PyClass) and isinstance(raised, PyInstance):
            return raised.cls.is_subclass_of(candidate)
        return False

    def handler_matches(self, handler: ast.ExceptHandler, raised: Any, env: Env) -> bool:
        if handler.type is None:
            return True  # bare `except:`
        expected = self.eval(handler.type, env)
        candidates = expected if isinstance(expected, tuple) else (expected,)
        return any(self._raises_as(raised, candidate) for candidate in candidates)

    def exec_try(self, node: ast.Try, env: Env) -> None:
        """Real exception semantics: raise, match a handler, unwind.

        Only what the *script* raised is caught. The interpreter's own control
        signals -- return, break, a budget abort -- pass straight through, or a
        stray `except Exception` in the input would swallow them.
        """
        try:
            self.exec_block(node.body, env)
        except _Raise as raised:
            for handler in node.handlers:
                if not self.handler_matches(handler, raised.value, env):
                    continue
                if handler.name:
                    env.assign(handler.name, raised.value)
                self._handling.append(raised.value)
                try:
                    self.exec_block(handler.body, env)
                finally:
                    self._handling.pop()
                    if handler.name:
                        env.vars.pop(handler.name, None)
                break
            else:
                raise
        else:
            self.exec_block(node.orelse, env)
        finally:
            self.exec_block(node.finalbody, env)

    # ------------------------------------------------------------------
    # Structural pattern matching
    # ------------------------------------------------------------------
    def exec_match(self, node: ast.Match, env: Env) -> None:
        subject = self.eval(node.subject, env)
        for case in node.cases:
            captured: dict[str, Any] = {}
            if not self.pattern_matches(case.pattern, subject, captured, env):
                continue
            # Captures bind before the guard runs, because the guard may use them.
            for name, value in captured.items():
                env.assign(name, value)
            if case.guard is not None and not self.truthiness(
                self.eval(case.guard, env), node, "match guard"
            ):
                continue
            self.exec_block(case.body, env)
            return

    def pattern_matches(
        self, pattern: ast.pattern, subject: Any, captured: dict[str, Any], env: Env
    ) -> bool:
        if isinstance(pattern, ast.MatchValue):
            expected = self.eval(pattern.value, env)
            if is_unknown(expected) or is_unknown(subject):
                return False
            try:
                return bool(subject == expected)
            except (TypeError, ValueError):
                return False

        if isinstance(pattern, ast.MatchSingleton):
            return subject is pattern.value

        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None and not self.pattern_matches(
                pattern.pattern, subject, captured, env
            ):
                return False
            if pattern.name:
                captured[pattern.name] = subject
            return True  # `case _` and `case name` always match

        if isinstance(pattern, ast.MatchOr):
            return any(
                self.pattern_matches(alternative, subject, captured, env)
                for alternative in pattern.patterns
            )

        if isinstance(pattern, ast.MatchSequence):
            return self._match_sequence(pattern, subject, captured, env)

        if isinstance(pattern, ast.MatchMapping):
            return self._match_mapping(pattern, subject, captured, env)

        if isinstance(pattern, ast.MatchClass):
            return self._match_class(pattern, subject, captured, env)

        self.bag.warn(
            E_UNSUPPORTED_STMT,
            f"unsupported match pattern: {type(pattern).__name__}; case skipped",
            self.span(pattern),
        )
        return False

    def _match_sequence(
        self, pattern: ast.MatchSequence, subject: Any, captured: dict[str, Any], env: Env
    ) -> bool:
        # A string is a sequence to Python but never matches a sequence pattern.
        if isinstance(subject, (str, bytes)) or not isinstance(subject, (list, tuple)):
            return False

        stars = [i for i, p in enumerate(pattern.patterns) if isinstance(p, ast.MatchStar)]
        if not stars:
            if len(subject) != len(pattern.patterns):
                return False
            return all(
                self.pattern_matches(p, item, captured, env)
                for p, item in zip(pattern.patterns, subject, strict=True)
            )

        pivot = stars[0]
        before = pattern.patterns[:pivot]
        after = pattern.patterns[pivot + 1 :]
        if len(subject) < len(before) + len(after):
            return False

        head = subject[: len(before)]
        tail = subject[len(subject) - len(after) :] if after else []
        middle = subject[len(before) : len(subject) - len(after)]

        if not all(
            self.pattern_matches(p, item, captured, env)
            for p, item in zip(before, head, strict=True)
        ):
            return False
        if not all(
            self.pattern_matches(p, item, captured, env)
            for p, item in zip(after, tail, strict=True)
        ):
            return False

        star = pattern.patterns[pivot]
        if isinstance(star, ast.MatchStar) and star.name:
            captured[star.name] = list(middle)
        return True

    def _match_mapping(
        self, pattern: ast.MatchMapping, subject: Any, captured: dict[str, Any], env: Env
    ) -> bool:
        if not isinstance(subject, dict):
            return False
        consumed = set()
        for key_node, value_pattern in zip(pattern.keys, pattern.patterns, strict=True):
            key = self.eval(key_node, env)
            if is_unknown(key) or key not in subject:
                return False
            if not self.pattern_matches(value_pattern, subject[key], captured, env):
                return False
            consumed.add(key)
        if pattern.rest:
            captured[pattern.rest] = {k: v for k, v in subject.items() if k not in consumed}
        return True

    def _match_class(
        self, pattern: ast.MatchClass, subject: Any, captured: dict[str, Any], env: Env
    ) -> bool:
        expected = self.eval(pattern.cls, env)
        if not self._is_instance_of(subject, expected):
            return False

        # Positional sub-patterns need __match_args__, which nothing here defines.
        if pattern.patterns:
            self.bag.warn(
                E_UNSUPPORTED_STMT,
                "positional class patterns need __match_args__, which this "
                "interpreter does not model; case skipped",
                self.span(pattern),
            )
            return False

        for name, sub_pattern in zip(pattern.kwd_attrs, pattern.kwd_patterns, strict=True):
            value = self.getattr_value(subject, name, pattern)
            if is_unknown(value) or not self.pattern_matches(sub_pattern, value, captured, env):
                return False
        return True

    @staticmethod
    def _is_instance_of(subject: Any, expected: Any) -> bool:
        if isinstance(expected, PyClass):
            return isinstance(subject, PyInstance) and subject.cls.is_subclass_of(expected)
        if isinstance(expected, ExceptionType):
            return isinstance(subject, ExceptionInstance) and Interpreter._raises_as(
                subject, expected
            )
        return False

    def bind_from_local(
        self, module: LocalModule, node: ast.AST, alias: ast.alias, env: Env
    ) -> None:
        """`from helpers import mk` -- or `import *`, which map scripts do use."""
        if alias.name == "*":
            for name, value in module.namespace.items():
                if not name.startswith("_"):
                    env.assign(name, value)
            return

        if alias.name in module.namespace:
            env.assign(alias.asname or alias.name, module.namespace[alias.name])
            return

        # A submodule of a package, rather than a name the module defined.
        nested = self.local_module(f"{module.name}.{alias.name}", node)
        if nested is not None:
            env.assign(alias.asname or alias.name, nested)
            return

        self.bag.warn(
            E_UNKNOWN_ATTRIBUTE,
            f"{module.name} defines no {alias.name!r}",
            self.span(node),
        )
        env.assign(alias.asname or alias.name, UNKNOWN)

    def local_module(self, dotted: str, node: ast.AST, level: int = 0) -> LocalModule | None:
        """Find and interpret a module sitting next to the current file.

        Only the importing file's own directory is searched, which is the same
        place Python puts first on `sys.path` when a script is run directly.
        Anything further afield is a third-party package, and those stay
        unresolved by design -- interpreting them would mean interpreting the
        world.

        `level` is the leading-dot count of a relative import: `.sibling` is
        level 1 and means this directory, `..cousin` is level 2 and means the one
        above, and so on.
        """
        if not dotted or self.filename in ("<string>", "<unknown>", "<test>"):
            return None

        root = Path(self.filename).resolve().parent
        for _ in range(max(level - 1, 0)):
            root = root.parent
        relative = dotted.replace(".", "/")
        for candidate in (root / f"{relative}.py", root / relative / "__init__.py"):
            if candidate.is_file():
                break
        else:
            return None

        key = str(candidate)
        if key in self._modules:
            return self._modules[key]
        if key in self._loading:
            # A cycle. Python would hand back a half-built module here; saying so
            # and moving on is more useful than pretending it resolved.
            self.bag.warn(
                E_LOCAL_IMPORT_CYCLE,
                f"circular import of {dotted!r} while it is still being read; "
                "names from it resolve to Unknown",
                self.span(node),
            )
            return None

        return self.interpret_module(dotted, candidate, node)

    def interpret_module(self, dotted: str, path: Path, node: ast.AST) -> LocalModule | None:
        """Parse and symbolically execute a sibling module in its own namespace."""
        from .loader import parse_source, read_source

        key = str(path)
        source = read_source(path, self.bag)
        if not source:
            return None

        # `parse_source` retargets the bag's source lines for caret rendering, and
        # the caller's diagnostics still need the original ones afterwards.
        outer_source, outer_filename = self.bag.source_lines, self.filename
        module_ast = parse_source(source, key, self.bag)
        if module_ast is None:
            self.bag.source_lines = outer_source
            return None

        namespace = Env()
        self._loading.append(key)
        self.filename = key
        try:
            saved_globals, self.globals = self.globals, namespace
            try:
                self.exec_block(module_ast.body, namespace)
            finally:
                self.globals = saved_globals
        except (_Return, _Abort):
            pass
        except _Raise as raised:
            self.bag.warn(
                E_UNCAUGHT_RAISE,
                f"{dotted} raised {getattr(raised.value, 'name', 'an exception')} while "
                "being read; what it defined up to that point is still used",
            )
        finally:
            self._loading.pop()
            self.filename = outer_filename
            self.bag.source_lines = outer_source

        module = LocalModule(name=dotted, path=key, namespace=dict(namespace.vars))
        self._modules[key] = module
        self.bag.info(
            I_LOCAL_IMPORT,
            f"resolved {dotted!r} from {path.name} beside the input and interpreted it; "
            "the converted map depends on that file too",
        )
        return module

    def exec_import(self, node: ast.Import | ast.ImportFrom, env: Env) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                dotted = alias.name
                bound = alias.asname or dotted.split(".")[0]
                if self.registry.is_known_prefix(dotted):
                    # `import lanelet2.core` binds `lanelet2` unless aliased.
                    target = dotted if alias.asname else dotted.split(".")[0]
                    env.assign(bound, ModuleRef(target))
                elif dotted in SAFE_MODULES:
                    env.assign(bound, ModuleRef(dotted))
                elif (local := self.local_module(dotted, node)) is not None:
                    env.assign(bound, local)
                else:
                    env.assign(bound, UNKNOWN)
            return

        module = node.module or ""
        level = getattr(node, "level", 0) or 0
        for alias in node.names:
            bound = alias.asname or alias.name
            dotted = f"{module}.{alias.name}" if module else alias.name
            if self.registry.is_module(dotted):
                env.assign(bound, ModuleRef(dotted))
            elif self.registry.resolve(dotted) is not None:
                env.assign(bound, NativeCtor(dotted, self.registry.resolve(dotted)))
            elif dotted in MODULE_CONSTANTS:
                env.assign(bound, MODULE_CONSTANTS[dotted])
            elif module in SAFE_MODULES and alias.name in SAFE_MODULES[module]:
                env.assign(bound, SAFE_MODULES[module][alias.name])
            elif any(dotted.startswith(q) for q in QUERY_MODULES):
                env.assign(bound, OpaqueCallable(dotted))
            elif (local := self.local_module(module, node, level)) is not None:
                self.bind_from_local(local, node, alias, env)
            elif (nested := self.local_module(dotted, node, level)) is not None:
                # `from package import submodule`, where the name is a module.
                env.assign(bound, nested)
            else:
                env.assign(bound, UNKNOWN)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------
    def assign(self, target: ast.expr, value: Any, env: Env) -> None:
        if isinstance(target, ast.Name):
            env.assign(target.id, value)

        elif isinstance(target, (ast.Tuple, ast.List)):
            self.assign_sequence(target, value, env)

        elif isinstance(target, ast.Subscript):
            owner = self.eval(target.value, env)
            key = self.eval_slice(target.slice, env)
            self.setitem(owner, key, value, target)

        elif isinstance(target, ast.Attribute):
            owner = self.eval(target.value, env)
            self.setattr_shadow(owner, target.attr, value, target)

        elif isinstance(target, ast.Starred):
            self.assign(target.value, value, env)

        else:
            self.bag.error(
                E_UNSUPPORTED_TARGET,
                f"unsupported assignment target: {type(target).__name__}",
                self.span(target),
            )

    def assign_sequence(self, target: ast.Tuple | ast.List, value: Any, env: Env) -> None:
        elements = list(target.elts)
        starred = [i for i, e in enumerate(elements) if isinstance(e, ast.Starred)]

        if is_unknown(value):
            for element in elements:
                self.assign(element, UNKNOWN, env)
            return
        try:
            items = list(value)
        except TypeError:
            for element in elements:
                self.assign(element, UNKNOWN, env)
            return

        if not starred:
            if len(items) != len(elements):
                self.bag.warn(
                    W_UNKNOWN_ITERABLE,
                    f"unpacking {len(items)} values into {len(elements)} targets",
                    self.span(target),
                )
            for element, item in zip(elements, items, strict=False):
                self.assign(element, item, env)
            for element in elements[len(items) :]:
                self.assign(element, UNKNOWN, env)
            return

        pivot = starred[0]
        before, after = elements[:pivot], elements[pivot + 1 :]
        tail = len(items) - len(after)
        for element, item in zip(before, items[: len(before)], strict=False):
            self.assign(element, item, env)
        self.assign(elements[pivot], items[len(before) : max(tail, len(before))], env)
        for element, item in zip(after, items[max(tail, len(before)) :], strict=False):
            self.assign(element, item, env)

    def setitem(self, owner: Any, key: Any, value: Any, node: ast.AST) -> None:
        if is_unknown(owner) or is_unknown(key):
            return
        if isinstance(owner, AttributeMap):
            before = len(owner.coercions)
            owner[key] = value
            if len(owner.coercions) > before:
                coerced_key, coerced_value = owner.coercions[-1]
                self.bag.warn(
                    W_NON_STRING_ATTRIBUTE,
                    f"attribute {coerced_key!r} was given a non-string value "
                    f"({type(coerced_value).__name__}); coerced to a string",
                    self.span(node),
                )
            return
        if isinstance(owner, ShadowLineString):
            points = owner.storage.points
            index = key if not owner.inverted_view else len(points) - 1 - int(key)
            with suppress(IndexError, TypeError):
                points[index] = value
            return
        with suppress(TypeError, IndexError, KeyError):
            owner[key] = value

    def setattr_shadow(self, owner: Any, name: str, value: Any, node: ast.AST) -> None:
        if is_unknown(owner):
            return
        if isinstance(owner, PyInstance):
            owner.fields[name] = value
            return
        if isinstance(owner, PyClass):
            owner.namespace[name] = value
            return
        if isinstance(owner, ShadowLanelet) and name == "centerline":
            # A user-assigned centerline outranks the computed one, and survives
            # later changes to the bounds -- so it is stored, not recomputed.
            owner.centerline_override = value if isinstance(value, ShadowLineString) else None
            return
        if isinstance(owner, ShadowPoint) and name in {"x", "y", "z", "id"}:
            setattr(owner, name, value)
            return
        if isinstance(owner, (ShadowLineString, ShadowLanelet, ShadowArea)) and name == "id":
            owner.id = value
            return
        if isinstance(owner, Origin) and name == "position":
            owner.position = value
            return
        self.bag.warn(
            E_UNKNOWN_ATTRIBUTE,
            f"assignment to unsupported attribute {name!r} on {type(owner).__name__}",
            self.span(node),
        )

    def delete(self, target: ast.expr, env: Env) -> None:
        if isinstance(target, ast.Subscript):
            owner = self.eval(target.value, env)
            key = self.eval_slice(target.slice, env)
            if isinstance(owner, ShadowLineString):
                points = owner.storage.points
                index = key if not owner.inverted_view else len(points) - 1 - int(key)
                with suppress(IndexError, TypeError):
                    del points[index]
                return
            with suppress(TypeError, IndexError, KeyError):
                del owner[key]
        elif isinstance(target, ast.Name):
            env.vars.pop(target.id, None)

    def eval_load_of_target(self, target: ast.expr, env: Env) -> Any:
        """Read the current value of an augmented-assignment target."""
        if isinstance(target, ast.Name):
            found, value = env.lookup(target.id)
            return value if found else UNKNOWN
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            return self.eval(target, env)
        return UNKNOWN

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    def eval(self, node: ast.expr, env: Env) -> Any:
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            found, value = env.lookup(node.id)
            if found:
                return value
            if node.id in SAFE_BUILTINS:
                return SAFE_BUILTINS[node.id]
            if node.id in self._exceptions:
                return self._exceptions[node.id]
            self.bag.error(E_NAME_UNDEFINED, f"undefined name {node.id!r}", self.span(node))
            return UNKNOWN

        if isinstance(node, ast.Attribute):
            return self.eval_attribute(node, env)

        if isinstance(node, ast.Call):
            return self.eval_call(node, env)

        if isinstance(node, ast.BinOp):
            return self.binop(node.op, self.eval(node.left, env), self.eval(node.right, env), node)

        if isinstance(node, ast.UnaryOp):
            return self.unaryop(node, env)

        if isinstance(node, ast.BoolOp):
            return self.boolop(node, env)

        if isinstance(node, ast.Compare):
            return self.compare(node, env)

        if isinstance(node, ast.IfExp):
            truth = self.truthiness(self.eval(node.test, env), node, "conditional expression")
            return self.eval(node.body if truth else node.orelse, env)

        if isinstance(node, ast.List):
            return self.eval_elements(node.elts, env)

        if isinstance(node, ast.Tuple):
            return tuple(self.eval_elements(node.elts, env))

        if isinstance(node, ast.Set):
            values = self.eval_elements(node.elts, env)
            try:
                return set(values)
            except TypeError:
                return UNKNOWN

        if isinstance(node, ast.Dict):
            return self.eval_dict(node, env)

        if isinstance(node, ast.Subscript):
            return self.eval_subscript(node, env)

        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            return self.eval_comprehension(node, env)

        if isinstance(node, ast.JoinedStr):
            return self.eval_fstring(node, env)

        if isinstance(node, ast.FormattedValue):
            return self.format_value(node, env)

        if isinstance(node, ast.Lambda):
            return Lambda(node, env, [self.eval(d, env) for d in node.args.defaults])

        if isinstance(node, ast.Starred):
            return self.eval(node.value, env)

        if isinstance(node, ast.Slice):
            return self.eval_slice(node, env)

        if isinstance(node, ast.Yield):
            if self._yields:
                self._yields[-1].append(
                    self.eval(node.value, env) if node.value is not None else None
                )
            return None

        if isinstance(node, ast.YieldFrom):
            source = self.eval(node.value, env)
            items = self.iterate(source, node)
            if self._yields and items:
                self._yields[-1].extend(items)
            return None

        if isinstance(node, ast.NamedExpr):
            value = self.eval(node.value, env)
            self.assign(node.target, value, env)
            return value

        self.bag.error(
            E_UNSUPPORTED_EXPR, f"unsupported expression: {type(node).__name__}", self.span(node)
        )
        return UNKNOWN

    def eval_elements(self, elements: list[ast.expr], env: Env) -> list[Any]:
        out: list[Any] = []
        for element in elements:
            if isinstance(element, ast.Starred):
                inner = self.eval(element.value, env)
                if is_unknown(inner):
                    out.append(UNKNOWN)
                    continue
                try:
                    out.extend(inner)
                except TypeError:
                    out.append(UNKNOWN)
            else:
                out.append(self.eval(element, env))
        return out

    def eval_dict(self, node: ast.Dict, env: Env) -> Any:
        result: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=False):
            if key_node is None:  # `{**other}`
                other = self.eval(value_node, env)
                if isinstance(other, dict):
                    result.update(other)
                continue
            key = self.eval(key_node, env)
            if is_unknown(key):
                continue
            try:
                result[key] = self.eval(value_node, env)
            except TypeError:
                continue
        return result

    def eval_subscript(self, node: ast.Subscript, env: Env) -> Any:
        owner = self.eval(node.value, env)
        key = self.eval_slice(node.slice, env)
        if is_unknown(owner) or is_unknown(key):
            return UNKNOWN
        try:
            return owner[key]
        except (TypeError, IndexError, KeyError):
            return UNKNOWN

    def eval_slice(self, node: ast.expr, env: Env) -> Any:
        if isinstance(node, ast.Slice):
            lower = self.eval(node.lower, env) if node.lower else None
            upper = self.eval(node.upper, env) if node.upper else None
            step = self.eval(node.step, env) if node.step else None
            if any(is_unknown(v) for v in (lower, upper, step)):
                return UNKNOWN
            return slice(lower, upper, step)
        return self.eval(node, env)

    def eval_fstring(self, node: ast.JoinedStr, env: Env) -> Any:
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant):
                parts.append(str(piece.value))
            else:
                value = self.eval(piece, env)
                if is_unknown(value):
                    return UNKNOWN
                parts.append(str(value))
        return "".join(parts)

    def format_value(self, node: ast.FormattedValue, env: Env) -> Any:
        value = self.eval(node.value, env)
        if is_unknown(value):
            return UNKNOWN
        spec = ""
        if node.format_spec is not None:
            spec_value = self.eval(node.format_spec, env)
            if is_unknown(spec_value):
                return UNKNOWN
            spec = str(spec_value)
        if node.conversion == 114:  # !r
            value = repr(value)
        elif node.conversion == 115:  # !s
            value = str(value)
        try:
            return format(value, spec)
        except (TypeError, ValueError):
            return UNKNOWN

    def eval_comprehension(self, node: ast.expr, env: Env) -> Any:
        results: list[Any] = []
        pairs: list[tuple[Any, Any]] = []

        def recurse(index: int, scope: Env) -> None:
            if index == len(node.generators):
                if isinstance(node, ast.DictComp):
                    pairs.append((self.eval(node.key, scope), self.eval(node.value, scope)))
                else:
                    results.append(self.eval(node.elt, scope))
                return
            generator = node.generators[index]
            iterable = self.eval(generator.iter, scope)
            items = self.iterate(iterable, node)
            if items is None:
                return
            for count, item in enumerate(items):
                if count >= self.options.max_iterations:
                    self.bag.error(
                        E_ITERATION_LIMIT,
                        f"comprehension exceeded {self.options.max_iterations} iterations",
                        self.span(node),
                    )
                    raise _Abort
                inner = Env(scope, self.globals)
                self.assign(generator.target, item, inner)
                if all(
                    self.truthiness(self.eval(cond, inner), node, "comprehension condition")
                    for cond in generator.ifs
                ):
                    recurse(index + 1, inner)

        # Comprehensions have their own scope; writes inside must not leak out.
        recurse(0, Env(env, self.globals))

        if isinstance(node, ast.DictComp):
            return {k: v for k, v in pairs if not is_unknown(k)}
        if isinstance(node, ast.SetComp):
            try:
                return set(results)
            except TypeError:
                return UNKNOWN
        return results

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def binop(self, op: ast.operator, left: Any, right: Any, node: ast.AST) -> Any:
        if is_unknown(left) or is_unknown(right):
            return UNKNOWN
        try:
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                return left**right
            if isinstance(op, ast.BitAnd):
                return left & right
            if isinstance(op, ast.BitOr):
                return left | right
            if isinstance(op, ast.BitXor):
                return left ^ right
            if isinstance(op, ast.LShift):
                return left << right
            if isinstance(op, ast.RShift):
                return left >> right
            if isinstance(op, ast.MatMult):
                return left @ right
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return UNKNOWN
        self.bag.error(
            E_UNSUPPORTED_EXPR, f"unsupported operator: {type(op).__name__}", self.span(node)
        )
        return UNKNOWN

    def unaryop(self, node: ast.UnaryOp, env: Env) -> Any:
        operand = self.eval(node.operand, env)
        if isinstance(node.op, ast.Not):
            if self._unresolved(operand):
                return UNKNOWN
            return not self._python_truth(operand)
        if is_unknown(operand):
            return UNKNOWN
        try:
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Invert):
                return ~operand
        except TypeError:
            return UNKNOWN
        return UNKNOWN

    def boolop(self, node: ast.BoolOp, env: Env) -> Any:
        result: Any = None
        for index, value_node in enumerate(node.values):
            value = self.eval(value_node, env)
            if self._unresolved(value):
                return UNKNOWN
            truth = self._python_truth(value)
            if isinstance(node.op, ast.And):
                if not truth:
                    return value
            elif truth:
                return value
            result = value
            if index == len(node.values) - 1:
                return value
        return result

    def compare(self, node: ast.Compare, env: Env) -> Any:
        left = self.eval(node.left, env)
        for op, right_node in zip(node.ops, node.comparators, strict=False):
            right = self.eval(right_node, env)
            # `is`/`is not` are answerable even against Unknown, and scripts use
            # `x is not None` as a real branch, so handle them before bailing out.
            if isinstance(op, ast.Is):
                outcome = left is right
            elif isinstance(op, ast.IsNot):
                outcome = left is not right
            elif self._unresolved(left) or self._unresolved(right):
                return UNKNOWN
            else:
                try:
                    if isinstance(op, ast.Eq):
                        outcome = left == right
                    elif isinstance(op, ast.NotEq):
                        outcome = left != right
                    elif isinstance(op, ast.Lt):
                        outcome = left < right
                    elif isinstance(op, ast.LtE):
                        outcome = left <= right
                    elif isinstance(op, ast.Gt):
                        outcome = left > right
                    elif isinstance(op, ast.GtE):
                        outcome = left >= right
                    elif isinstance(op, ast.In):
                        outcome = left in right
                    elif isinstance(op, ast.NotIn):
                        outcome = left not in right
                    else:
                        return UNKNOWN
                except TypeError:
                    return UNKNOWN
            if not outcome:
                return False
            left = right
        return True

    # ------------------------------------------------------------------
    # Truthiness and iteration
    # ------------------------------------------------------------------
    @staticmethod
    def _python_truth(value: Any) -> bool:
        if isinstance(value, ShadowLineString):
            return len(value) > 0
        try:
            return bool(value)
        except (TypeError, ValueError):
            return True

    @staticmethod
    def _unresolved(value: Any) -> bool:
        """Whether a value's truthiness is genuinely unknowable here.

        `OpaqueValue` counts: it stands for a lanelet2 object we deliberately do
        not model (a routing graph's answer, a traffic-rules verdict), so testing
        it is exactly as unresolvable as testing an `Unknown`.
        """
        return is_unknown(value) or isinstance(value, OpaqueValue)

    def truthiness(self, value: Any, node: ast.AST, what: str, *, quiet: bool = False) -> bool:
        """Resolve a condition to a single branch.

        Forking on an unresolvable condition would produce several candidate maps
        with no way to pick one, so the policy is to choose a branch and say so.
        """
        if not self._unresolved(value):
            return self._python_truth(value)
        if quiet:
            return False
        policy = self.options.on_unknown_branch
        if policy == "error":
            self.bag.error(
                W_UNKNOWN_CONDITION,
                f"{what} cannot be resolved statically",
                self.span(node),
            )
            return False
        taken = policy == "then"
        self.bag.warn(
            W_UNKNOWN_CONDITION,
            f"{what} cannot be resolved statically; assuming {str(taken).lower()}",
            self.span(node),
        )
        return taken

    def iterate(self, iterable: Any, node: ast.AST) -> list[Any] | None:
        if self._unresolved(iterable):
            self.bag.warn(
                W_UNKNOWN_ITERABLE,
                "loop over a value that cannot be resolved statically; body skipped",
                self.span(node),
            )
            return None
        if isinstance(iterable, (ShadowLineString, ShadowCompound)):
            return iterable.points
        if isinstance(iterable, ShadowLaneletSequence):
            return iterable.lanelets()
        if isinstance(iterable, (ShadowLayer, ShadowMap)):
            return list(iterable) if isinstance(iterable, ShadowLayer) else []
        try:
            return list(iterable)
        except TypeError:
            self.bag.warn(
                W_UNKNOWN_ITERABLE,
                f"value of type {type(iterable).__name__} is not iterable; loop skipped",
                self.span(node),
            )
            return None

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------
    def eval_attribute(self, node: ast.Attribute, env: Env) -> Any:
        owner = self.eval(node.value, env)
        return self.getattr_value(owner, node.attr, node)

    def getattr_value(self, owner: Any, name: str, node: ast.AST) -> Any:
        if is_unknown(owner):
            return UNKNOWN

        if isinstance(owner, ModuleRef):
            return self.resolve_module_member(owner, name, node)

        if isinstance(owner, PyInstance):
            if name in owner.fields:
                return owner.fields[name]
            found, value = owner.cls.lookup(name)
            if found:
                # A function found on the class is a method: bind it.
                return BoundMethod(owner, value) if isinstance(value, Function) else value
            self.bag.warn(
                E_UNKNOWN_ATTRIBUTE,
                f"{owner.cls.name} instance has no attribute {name!r}",
                self.span(node),
            )
            return UNKNOWN

        if isinstance(owner, LocalModule):
            if name in owner.namespace:
                return owner.namespace[name]
            nested = self.local_module(f"{owner.name}.{name}", node)
            if nested is not None:
                return nested
            self.bag.warn(
                E_UNKNOWN_ATTRIBUTE,
                f"{owner.name} defines no {name!r}",
                self.span(node),
            )
            return UNKNOWN

        if isinstance(owner, Super):
            found, value = owner.lookup(name)
            if not found:
                self.bag.warn(
                    E_UNKNOWN_ATTRIBUTE,
                    f"no base of {getattr(owner.owner, 'name', '?')} defines {name!r}",
                    self.span(node),
                )
                return UNKNOWN
            return BoundMethod(owner.instance, value) if isinstance(value, Function) else value

        if isinstance(owner, PyClass):
            found, value = owner.lookup(name)
            return value if found else UNKNOWN

        if isinstance(owner, ExceptionInstance):
            if name == "args":
                return list(owner.args)
            return UNKNOWN

        if isinstance(owner, OpaqueValue):
            return OpaqueCallable(f"{owner.kind}.{name}")

        # Shadow objects: allow-listed names only.
        for kind, allowed in _SHADOW_ATTRS.items():
            if isinstance(owner, kind):
                if name not in allowed:
                    self.bag.warn(
                        E_UNKNOWN_ATTRIBUTE,
                        f"{type(owner).__name__} has no attribute {name!r} that this "
                        "transpiler models",
                        self.span(node),
                    )
                    return UNKNOWN
                return self.shadow_attribute(owner, name, node)

        if isinstance(owner, dict) and not isinstance(owner, AttributeMap):
            method = safe_method(owner, name)
            return method if method is not None else UNKNOWN

        method = safe_method(owner, name)
        if method is not None:
            return method

        if isinstance(owner, AttributeMap):
            inner = safe_method(dict(owner), name)
            if inner is not None:
                return getattr(owner, name, UNKNOWN)

        return UNKNOWN

    def resolve_module_member(self, owner: ModuleRef, name: str, node: ast.AST) -> Any:
        dotted = f"{owner.dotted}.{name}"

        if self.registry.is_module(dotted):
            return ModuleRef(dotted)
        ctor = self.registry.resolve(dotted)
        if ctor is not None:
            return NativeCtor(dotted, ctor)
        if dotted in MODULE_CONSTANTS:
            return MODULE_CONSTANTS[dotted]
        if owner.dotted in SAFE_MODULES and name in SAFE_MODULES[owner.dotted]:
            return SAFE_MODULES[owner.dotted][name]
        if any(dotted.startswith(q) for q in QUERY_MODULES):
            return OpaqueCallable(dotted)
        if self.registry.is_known_prefix(dotted):
            # A real lanelet2 name we have not modelled: inert, not fatal, so a
            # script that merely touches it still yields its map.
            return OpaqueCallable(dotted)
        return UNKNOWN

    def shadow_attribute(self, owner: Any, name: str, node: ast.AST) -> Any:
        # Properties that read as values.
        if isinstance(owner, ShadowLanelet) and name == "centerline":
            if owner.centerline_override is not None:
                return owner.centerline_override
            return compute_centerline(owner.left, owner.right)

        if isinstance(owner, ShadowRegulatoryElement):
            if name in _REGELEM_ROLE_ACCESSORS:
                return BoundShadowMethod(owner, name)
            if name == "stopLine":
                members = owner.role("ref_line")
                return members[0] if members else None
            if name in {"roles", "parameters", "id", "attributes"}:
                return getattr(owner, name)
            return BoundShadowMethod(owner, name)

        if isinstance(owner, ShadowMap) and name in {"add", "laneletMap"}:
            return BoundShadowMethod(owner, name)

        if isinstance(owner, ShadowLayer):
            return BoundShadowMethod(owner, name)

        if isinstance(owner, (ShadowCompound, ShadowLaneletSequence)):
            if name in {"leftBound", "rightBound"}:
                return getattr(owner, name)
            if name == "centerline":
                return compute_centerline(owner.leftBound, owner.rightBound)
            return BoundShadowMethod(owner, name)

        if isinstance(owner, BoundingBox):
            return getattr(owner, name, UNKNOWN)

        if isinstance(owner, ProjectionInfo):
            return BoundShadowMethod(owner, name)

        if isinstance(owner, (ShadowCompound, ShadowLaneletSequence)):
            if name == "invert":
                return owner.invert()
            if name == "inverted":
                return owner.inverted()
            if name in {"ids", "lineStrings", "numSegments", "lanelets"}:
                return getattr(owner, name)()
            if name in {"polygon2d", "polygon3d"} and isinstance(owner, ShadowLaneletSequence):
                left = owner.leftBound.points
                right = list(reversed(owner.rightBound.points))
                storage = LineStringStorage(points=[*left, *right])
                return ShadowLineString(storage, dim=3 if name == "polygon3d" else 2, polygon=True)
            return UNKNOWN

        if isinstance(owner, ShadowPoint) and name == "basicPoint":
            return BoundShadowMethod(owner, name)

        if isinstance(owner, (ShadowLineString, ShadowLanelet, ShadowArea)):
            value = getattr(owner, name, UNKNOWN)
            if callable(value) and not isinstance(value, (list, dict)):
                return BoundShadowMethod(owner, name)
            return value

        return getattr(owner, name, UNKNOWN)

    # ------------------------------------------------------------------
    # Calls
    # ------------------------------------------------------------------
    def eval_call(self, node: ast.Call, env: Env) -> Any:
        func = self.eval(node.func, env)
        args = self.eval_args(node, env)
        return self.call(func, args, self.span(node))

    def eval_args(self, node: ast.Call, env: Env) -> Args:
        args = Args()
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                spread = self.eval(arg.value, env)
                if is_unknown(spread):
                    args.positional.append(UNKNOWN)
                    continue
                try:
                    args.positional.extend(spread)
                except TypeError:
                    args.positional.append(UNKNOWN)
            else:
                args.positional.append(self.eval(arg, env))
        for keyword in node.keywords:
            value = self.eval(keyword.value, env)
            if keyword.arg is None:  # `**kwargs`
                if isinstance(value, dict):
                    args.keyword.update({str(k): v for k, v in value.items()})
                continue
            args.keyword[keyword.arg] = value
        return args

    def call(self, func: Any, args: Args, span: SourceSpan) -> Any:
        if is_unknown(func):
            return UNKNOWN

        if isinstance(func, NativeCtor):
            return func.fn(args, span)

        if isinstance(func, OpaqueCallable):
            return self.registry.opaque_call(func.dotted, span)

        if isinstance(func, BoundShadowMethod):
            return self.call_shadow_method(func, args, span)

        if func is SAFE_BUILTINS["isinstance"] and len(args.positional) == 2:
            return self._isinstance(args.positional[0], args.positional[1])

        if isinstance(func, PyClass):
            return self.instantiate(func, args, span)

        if isinstance(func, BoundMethod):
            bound = Args([func.instance, *args.positional], dict(args.keyword))
            return self.call(func.function, bound, span)

        if isinstance(func, ExceptionType):
            return ExceptionInstance(func, tuple(args.positional))

        if isinstance(func, Function):
            return self.call_user_function(func, args, span)

        if isinstance(func, Lambda):
            return self.call_lambda(func, args, span)

        if callable(func):
            try:
                return func(*args.positional, **args.keyword)
            except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError):
                return UNKNOWN

        return UNKNOWN

    @staticmethod
    def _is_generator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Whether this function's *own* body yields (nested ones do not count)."""
        stack: list[ast.AST] = list(node.body)
        while stack:
            current = stack.pop()
            if isinstance(current, (ast.Yield, ast.YieldFrom)):
                return True
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue  # a nested function's yields belong to it, not to us
            stack.extend(ast.iter_child_nodes(current))
        return False

    @staticmethod
    def bind_super(func: Function, args: Args, env: Env) -> None:
        """Make `super` a local name inside a method.

        CPython does this with a compiler-inserted `__class__` cell; the effect is
        the same and the mechanism is simpler here. The explicit two-argument form
        is honoured too, since a script that writes `super(Base, self)` means it.
        """
        instance = args.positional[0] if args.positional else None

        def make_super(*given: Any) -> Super:
            if len(given) == 2:
                return Super(given[1], given[0])
            return Super(instance, func.owner)

        env.assign("super", make_super)

    def call_user_function(self, func: Function, args: Args, span: SourceSpan) -> Any:
        self._depth += 1
        if self._depth > self.options.max_recursion:
            self._depth -= 1
            self.bag.error(
                E_RECURSION_LIMIT,
                f"call depth exceeded {self.options.max_recursion} in {func.name}()",
                span,
            )
            raise _Abort
        try:
            env = Env(func.closure, func.closure.globals_env)
            self.bind_arguments(func.node.args, func.defaults, func.kw_defaults, args, env, span)
            if func.owner is not None:
                self.bind_super(func, args, env)
            generator = self._is_generator(func.node)
            if generator:
                # Generators are materialised eagerly into a list. Every loop in
                # this interpreter is bounded anyway, so laziness buys nothing,
                # and a list is what every consumer here does with one. The one
                # thing it cannot model is a value *sent* into a yield, which
                # `yield` therefore evaluates to None.
                self._yields.append([])
            try:
                self.exec_block(func.node.body, env)
            except _Return as ret:
                if generator:
                    return self._yields.pop()
                return ret.value
            return self._yields.pop() if generator else None
        finally:
            self._depth -= 1

    def _isinstance(self, value: Any, expected: Any) -> Any:
        """`isinstance` is answerable for the types this interpreter owns.

        For lanelet2 shadows it still is not: they are not the real classes, so
        the honest answer stays Unknown and the branch goes through the
        unresolved-condition policy.
        """
        candidates = expected if isinstance(expected, tuple) else (expected,)

        def hit(candidate: Any) -> bool:
            if isinstance(candidate, PyClass) and isinstance(value, PyInstance):
                return value.cls.is_subclass_of(candidate)
            if isinstance(candidate, ExceptionType) and isinstance(value, ExceptionInstance):
                return isinstance(value.type, ExceptionType) and candidate.matches(value.type)
            return False

        if any(hit(candidate) for candidate in candidates):
            return True
        if all(isinstance(c, (PyClass, ExceptionType)) for c in candidates):
            return False
        return UNKNOWN

    def instantiate(self, cls: PyClass, args: Args, span: SourceSpan) -> Any:
        instance = PyInstance(cls=cls)
        found, initialiser = cls.lookup("__init__")
        if found:
            self.call(BoundMethod(instance, initialiser), args, span)
        elif cls.derives_from("BaseException"):
            # What `BaseException.__init__` does: a script-defined exception with
            # no `__init__` of its own still remembers its arguments, so a handler
            # can read `exc.args`.
            instance.fields["args"] = list(args.positional)
        return instance

    def call_lambda(self, func: Lambda, args: Args, span: SourceSpan) -> Any:
        env = Env(func.closure, self.globals)
        self.bind_arguments(func.node.args, func.defaults, {}, args, env, span)
        return self.eval(func.node.body, env)

    def bind_arguments(
        self,
        spec: ast.arguments,
        defaults: list[Any],
        kw_defaults: dict[str, Any],
        args: Args,
        env: Env,
        span: SourceSpan,
    ) -> None:
        names = [a.arg for a in spec.posonlyargs] + [a.arg for a in spec.args]
        supplied = list(args.positional)

        if len(supplied) > len(names) and spec.vararg is None:
            self.bag.warn(
                E_BAD_ARITY,
                f"{len(supplied)} positional arguments for {len(names)} parameters; extras ignored",
                span,
            )

        # Defaults fill the tail of the positional list.
        offset = len(names) - len(defaults)
        for index, name in enumerate(names):
            if index < len(supplied):
                env.assign(name, supplied[index])
            elif name in args.keyword:
                env.assign(name, args.keyword[name])
            elif index >= offset and defaults:
                env.assign(name, defaults[index - offset])
            else:
                env.assign(name, UNKNOWN)

        if spec.vararg is not None:
            env.assign(spec.vararg.arg, supplied[len(names) :])

        for arg in spec.kwonlyargs:
            if arg.arg in args.keyword:
                env.assign(arg.arg, args.keyword[arg.arg])
            else:
                env.assign(arg.arg, kw_defaults.get(arg.arg, UNKNOWN))

        if spec.kwarg is not None:
            consumed = set(names) | {a.arg for a in spec.kwonlyargs}
            env.assign(spec.kwarg.arg, {k: v for k, v in args.keyword.items() if k not in consumed})

    def make_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, env: Env) -> Any:
        # Defaults are evaluated once, at definition time -- including mutable
        # ones. That is Python's behaviour, and scripts occasionally rely on it.
        defaults = [self.eval(d, env) for d in node.args.defaults]
        kw_defaults: dict[str, Any] = {}
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
            if default is not None:
                kw_defaults[arg.arg] = self.eval(default, env)
        return self.apply_decorators(
            Function(node, env, defaults, kw_defaults), node.decorator_list, env
        )

    def apply_decorators(self, value: Any, decorators: list[ast.expr], env: Env) -> Any:
        """Apply decorators innermost-first, as Python does.

        A decorator can replace what it wraps outright, so applying it is the
        only way the resulting map matches what the script meant.
        """
        for decorator in reversed(decorators):
            function = self.eval(decorator, env)
            value = self.call(function, Args([value]), self.span(decorator))
        return value

    def make_class(self, node: ast.ClassDef, env: Env) -> Any:
        """Execute a class body and capture what it defined.

        Deliberately minimal -- no metaclasses, no descriptors, no MRO
        linearisation -- because the only thing map scripts do with a class is
        gather state in `__init__` and read it back through methods.
        """
        body_env = Env(env, self.globals)
        self.exec_block(node.body, body_env)
        bases = [self.eval(base, env) for base in node.bases]
        cls = PyClass(name=node.name, bases=bases, namespace=dict(body_env.vars))
        # Methods remember the class they were written in, so `super()` inside one
        # knows where in the base list to resume the search.
        for value in cls.namespace.values():
            if isinstance(value, Function) and value.owner is None:
                value.owner = cls
        return self.apply_decorators(cls, node.decorator_list, env)

    # ------------------------------------------------------------------
    # Shadow methods
    # ------------------------------------------------------------------
    def call_shadow_method(self, bound: BoundShadowMethod, args: Args, span: SourceSpan) -> Any:
        owner, name = bound.owner, bound.name
        positional = args.positional

        if isinstance(owner, ShadowRegulatoryElement):
            if name in _REGELEM_ROLE_ACCESSORS:
                return owner.role(_REGELEM_ROLE_ACCESSORS[name])
            if name == "find":
                target = positional[0] if positional else None
                for members in owner.parameters.values():
                    for member in members:
                        if getattr(member, "id", None) == target:
                            return member
                return None
            if name == "type":
                return owner.attributes.get("subtype", "")
            if name.startswith("add") or name.startswith("remove"):
                return None
            return UNKNOWN

        if isinstance(owner, ShadowLineString):
            if name == "append" and positional:
                if isinstance(positional[0], ShadowPoint):
                    owner.append(positional[0])
                return None
            if name == "invert":
                return owner.invert()
            if name == "inverted":
                return owner.inverted()
            return UNKNOWN

        if isinstance(owner, ShadowLanelet):
            return self._lanelet_method(owner, name, positional)

        if isinstance(owner, ShadowArea):
            if name == "addRegulatoryElement" and positional:
                owner.addRegulatoryElement(positional[0])
                return None
            if name == "removeRegulatoryElement" and positional:
                return owner.removeRegulatoryElement(positional[0])
            if name == "outerBoundPolygon":
                return self._ring_polygon(owner.outer)
            if name == "innerBoundPolygons":
                return [self._ring_polygon(ring) for ring in owner.inners]
            return UNKNOWN

        if isinstance(owner, (ShadowCompound, ShadowLaneletSequence)):
            if name == "invert":
                return owner.invert()
            if name == "inverted":
                return owner.inverted()
            if name in {"ids", "lineStrings", "numSegments", "lanelets"}:
                return getattr(owner, name)()
            if name in {"polygon2d", "polygon3d"} and isinstance(owner, ShadowLaneletSequence):
                left = owner.leftBound.points
                right = list(reversed(owner.rightBound.points))
                storage = LineStringStorage(points=[*left, *right])
                return ShadowLineString(storage, dim=3 if name == "polygon3d" else 2, polygon=True)
            return UNKNOWN

        if isinstance(owner, ShadowPoint) and name == "basicPoint":
            return BasicPoint(owner.x, owner.y, owner.z, owner.dim)

        if isinstance(owner, ShadowMap):
            if name == "add" and positional:
                owner.add(positional[0])
                return None
            if name == "laneletMap":
                return owner.laneletMap()
            return UNKNOWN

        if isinstance(owner, ShadowLayer):
            if name == "exists" and positional:
                return owner.exists(positional[0])
            if name == "get" and positional:
                return owner.get(positional[0])
            if name == "uniqueId":
                return self.registry.get_id()
            if name == "search" and positional:
                return _layer_search(owner, positional[0])
            if name == "nearest" and positional:
                count = int(positional[1]) if len(positional) > 1 else 1
                return _layer_nearest(owner, positional[0], count)
            if name == "findUsages" and positional:
                return _layer_find_usages(owner, positional[0])
            return UNKNOWN

        if isinstance(owner, ProjectionInfo):
            if name == "origin":
                return Origin(GPSPoint(owner.lat, owner.lon, owner.alt))
            if name == "setMGRSCode" and positional:
                # Autoware georeferences by naming a 100 km grid square rather
                # than an origin; the square is what the coordinates are relative
                # to, so it has to reach <geoReference>.
                owner.mgrs_code = str(positional[0])
                return None
            # forward/reverse need a real projection; the map is already in
            # metres, so nothing downstream depends on the answer.
            return UNKNOWN

        return UNKNOWN

    def _lanelet_method(self, owner: ShadowLanelet, name: str, positional: list[Any]) -> Any:
        if name == "invert":
            return owner.invert()
        if name == "inverted":
            return owner.inverted()
        if name == "addRegulatoryElement" and positional:
            owner.addRegulatoryElement(positional[0])
            return None
        if name == "removeRegulatoryElement" and positional:
            return owner.removeRegulatoryElement(positional[0])
        if name == "resetCache":
            return None
        if name in {"polygon2d", "polygon3d"}:
            left = owner.left.points if owner.left else []
            right = list(reversed(owner.right.points)) if owner.right else []
            storage = LineStringStorage(points=[*left, *right])
            return ShadowLineString(storage, dim=3 if name == "polygon3d" else 2, polygon=True)
        if name in {"trafficLights", "trafficSigns", "speedLimits", "rightOfWay", "allWayStop"}:
            wanted = {
                "trafficLights": {"TrafficLight", "AutowareTrafficLight"},
                "trafficSigns": {"TrafficSign", "SpeedLimit"},
                "speedLimits": {"SpeedLimit"},
                "rightOfWay": {"RightOfWay"},
                "allWayStop": {"AllWayStop"},
            }[name]
            return [r for r in owner.regelems if getattr(r, "kind", None) in wanted]
        return UNKNOWN

    @staticmethod
    def _ring_polygon(ring: list[ShadowLineString]) -> ShadowLineString:
        points: list[ShadowPoint] = []
        for bound in ring:
            for point in bound.points:
                if not points or points[-1] is not point:
                    points.append(point)
        return ShadowLineString(LineStringStorage(points=points), dim=3, polygon=True)


def _primitive_points(value: Any) -> list[ShadowPoint]:
    if isinstance(value, ShadowPoint):
        return [value]
    if isinstance(value, (ShadowLineString, ShadowCompound)):
        return list(value.points)
    if isinstance(value, ShadowLanelet):
        return [
            *(value.left.points if value.left else []),
            *(value.right.points if value.right else []),
        ]
    if isinstance(value, ShadowArea):
        return [p for bound in value.outer for p in bound.points]
    return []


def _layer_search(layer: ShadowLayer, box: Any) -> list[Any]:
    """Everything in the layer with a point inside the bounding box."""
    if not isinstance(box, BoundingBox):
        return []
    return [
        item
        for item in layer.items
        if any(box.contains(point.xyz) for point in _primitive_points(item))
    ]


def _layer_nearest(layer: ShadowLayer, target: Any, count: int) -> list[Any]:
    if isinstance(target, ShadowPoint):
        anchor = target.xy
    elif isinstance(target, BasicPoint):
        anchor = (target.x, target.y)
    else:
        return []

    def distance(item: Any) -> float:
        points = _primitive_points(item)
        if not points:
            return float("inf")
        return min((p.x - anchor[0]) ** 2 + (p.y - anchor[1]) ** 2 for p in points)

    return sorted(layer.items, key=distance)[: max(count, 0)]


def _layer_find_usages(layer: ShadowLayer, value: Any) -> list[Any]:
    """Layer members that *use* `value`, by storage identity.

    Usage is structural, not spatial: a lanelet uses a line string when that line
    string is one of its bounds, not merely when the two happen to share a point.
    Two consecutive lanelets share their joint points, so a looser test would
    report the neighbour as a user of a boundary it has never seen.
    """
    out: list[Any] = []

    if isinstance(value, ShadowLineString):
        target = value.storage

        def uses_bound(item: Any) -> bool:
            if isinstance(item, ShadowLanelet):
                return any(
                    bound is not None and bound.storage is target
                    for bound in (item.left, item.right)
                )
            if isinstance(item, ShadowArea):
                return any(
                    bound.storage is target for ring in [item.outer, *item.inners] for bound in ring
                )
            return False

        return [item for item in layer.items if uses_bound(item)]

    if isinstance(value, ShadowPoint):
        target = value.storage
        for item in layer.items:
            if isinstance(item, ShadowLineString) and any(
                point.storage is target for point in item.storage.points
            ):
                out.append(item)
        return out

    return out


def execute(
    module: ast.Module,
    filename: str,
    bag: DiagnosticBag,
    options: TranspileOptions,
) -> Registry:
    """Run a parsed script symbolically and return everything it built."""
    return Interpreter(filename, bag, options).run(module)
