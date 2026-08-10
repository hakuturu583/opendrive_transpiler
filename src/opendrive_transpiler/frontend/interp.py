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
from typing import Any

from ..config import TranspileOptions
from ..diagnostics import (
    E_BAD_ARITY,
    E_ITERATION_LIMIT,
    E_NAME_UNDEFINED,
    E_RECURSION_LIMIT,
    E_STATEMENT_BUDGET,
    E_UNBOUNDED_WHILE,
    E_UNKNOWN_ATTRIBUTE,
    E_UNSUPPORTED_EXPR,
    E_UNSUPPORTED_STMT,
    E_UNSUPPORTED_TARGET,
    W_NON_STRING_ATTRIBUTE,
    W_UNKNOWN_CONDITION,
    W_UNKNOWN_ITERABLE,
    DiagnosticBag,
    SourceSpan,
)
from ..ir.centerline import compute_centerline
from .builtins import SAFE_BUILTINS, SAFE_MODULES, safe_method
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
    ProjectionInfo: frozenset({"forward", "reverse", "origin"}),
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

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self, module: ast.Module) -> Registry:
        try:
            self.exec_block(module.body, self.globals)
            # A script that guards its build behind `if __name__ == "__main__"`
            # (or just defines `main()`) has not run it yet. Call it once.
            self._run_main_if_present()
        except _Abort:
            pass
        except _Return:
            pass
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
            # A script that raises is signalling a case it does not support; the
            # map built so far is still what we convert.
            raise _Abort

        elif isinstance(node, ast.ClassDef):
            self.bag.error(
                E_UNSUPPORTED_STMT,
                "class definitions are not supported; the map-building code must be "
                "at module level or in plain functions",
                self.span(node),
            )

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

    def exec_try(self, node: ast.Try, env: Env) -> None:
        try:
            self.exec_block(node.body, env)
        except (_Return, _Break, _Continue, _Abort):
            raise
        except Exception:
            for handler in node.handlers:
                if handler.name:
                    env.assign(handler.name, UNKNOWN)
                self.exec_block(handler.body, env)
                break
        else:
            self.exec_block(node.orelse, env)
        finally:
            self.exec_block(node.finalbody, env)

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
                else:
                    env.assign(bound, UNKNOWN)
            return

        module = node.module or ""
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
            if node.id == "__name__":
                # Scripts commonly guard the build with `if __name__ == "__main__"`.
                # Reporting "__main__" runs that block, which is what we want.
                return "__main__"
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
            env = Env(func.closure, self.globals)
            self.bind_arguments(func.node.args, func.defaults, func.kw_defaults, args, env, span)
            try:
                self.exec_block(func.node.body, env)
            except _Return as ret:
                return ret.value
            return None
        finally:
            self._depth -= 1

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

    def make_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, env: Env) -> Function:
        if node.decorator_list:
            # A decorator can replace the function outright, so ignoring one
            # silently could change the map without anything saying so.
            self.bag.warn(
                E_UNSUPPORTED_STMT,
                f"decorators on {node.name}() are not applied; the undecorated function is used",
                self.span(node),
            )
        # Defaults are evaluated once, at definition time -- including mutable
        # ones. That is Python's behaviour, and scripts occasionally rely on it.
        defaults = [self.eval(d, env) for d in node.args.defaults]
        kw_defaults: dict[str, Any] = {}
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=False):
            if default is not None:
                kw_defaults[arg.arg] = self.eval(default, env)
        return Function(node, env, defaults, kw_defaults)

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
