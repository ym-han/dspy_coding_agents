"""Property-based tests for adapter parsing, rendering, and schema helpers.

Runs quickly (low example counts) but now covers a broader grammar:
- primitives, Pydantic models (static + generated), dict[str, T], list[T], T | None,
  and simple non-None unions for rendering invariants.
- schema build/parse round-trips for all parse-supported annotations.
"""

from __future__ import annotations

import inspect
import types
from functools import reduce
from typing import Any, Literal, Union, get_args, get_origin

from hypothesis import assume, given, settings, strategies as st
from pydantic import BaseModel

from codex_dspy.agent import _build_output_schema, _parse_output_value
from codex_dspy.adapter import CodexAdapter, _is_optional_type, _render_type_str, _ts_type

# Keep property runs fast
TEST_SETTINGS = settings(max_examples=75, deadline=None)


class SmallModel(BaseModel):
    number: int
    text: str | None = None


class FlagModel(BaseModel):
    flag: bool
    note: str | None = None


def _model_dict_strategy() -> st.SearchStrategy[dict[str, Any]]:
    return st.builds(
        lambda n, t: {"number": n, "text": t},
        n=st.integers(-100, 100),
        t=st.one_of(st.none(), st.text(max_size=16)),
    )


def _flag_model_dict_strategy() -> st.SearchStrategy[dict[str, Any]]:
    return st.builds(
        lambda f, n: {"flag": f, "note": n},
        f=st.booleans(),
        n=st.one_of(st.none(), st.text(max_size=16)),
    )


BASE_ANNOTATIONS = [str, int, float, bool, SmallModel, FlagModel]


def _generated_model_strategy() -> st.SearchStrategy[type[BaseModel]]:
    """Generate small pydantic models (1-2 fields) over primitives/bool/optional str."""

    primitive_types = st.sampled_from([str, int, bool])

    @st.composite
    def _model(draw):
        field_count = draw(st.integers(min_value=1, max_value=2))
        fields = {}
        for i in range(field_count):
            base = draw(primitive_types)
            optional = draw(st.booleans())
            ann = base | None if optional else base
            fields[f"f{i}"] = (ann, ...)
        # create_model gives unique class per draw
        from pydantic import create_model

        return create_model("GenModel", **fields)  # type: ignore[arg-type]

    return _model()


def _annotation_strategy(include_unions: bool = True, include_dict: bool = True, max_union: int = 2) -> st.SearchStrategy[Any]:
    """Generate annotations: primitives, models, list, optional, dict[str, T], unions."""

    literal_values = st.one_of(
        st.text(max_size=4),
        st.integers(-2, 2),
        st.booleans(),
    )

    base = st.one_of(
        st.sampled_from(BASE_ANNOTATIONS),
        _generated_model_strategy(),
        literal_values.map(lambda v: Literal[v]),
    )

    def expand(children: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
        parts = [
            children.map(lambda ann: list[ann]),  # list[T]
            children.map(lambda ann: ann | None),  # Optional[T]
        ]
        if include_dict:
            parts.append(
                st.tuples(st.just(str), children).map(lambda t: dict[t[0], t[1]])  # type: ignore[index]
            )
        if include_unions:
            # unions of size 2..max_union
            union = st.lists(children, min_size=2, max_size=max_union).map(tuple)
            parts.append(union.map(_dedupe_union))
        return st.one_of(*parts)

    return st.recursive(base, expand, max_leaves=14)


def _annotation_without_optional_strategy() -> st.SearchStrategy[Any]:
    return _annotation_strategy(max_union=5).filter(lambda ann: not _is_optional_type(ann))


def _value_strategy_for_annotation(annotation: Any) -> st.SearchStrategy[Any]:
    origin = get_origin(annotation)

    if annotation is str:
        return st.text(max_size=8)
    if annotation is int:
        return st.integers(-50, 50)
    if annotation is float:
        return st.floats(-50, 50, allow_nan=False, allow_infinity=False)
    if annotation is bool:
        return st.booleans()

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        fields = annotation.model_fields
        if fields:
            items = {}
            for name, field in fields.items():
                strat = _value_strategy_for_annotation(field.annotation)
                if not field.is_required():
                    strat = st.one_of(st.none(), strat)
                items[name] = strat
            return st.fixed_dictionaries(items)
        return st.just({})

    if origin is list:
        inner = get_args(annotation)[0]
        strat = st.lists(_value_strategy_for_annotation(inner), max_size=4)
        # If inner is not optional, avoid generating None elements
        inner_origin = get_origin(inner)
        inner_has_none = inner_origin in (Union, types.UnionType) and type(None) in get_args(inner)
        if not inner_has_none and inner is not type(None):
            strat = strat.filter(lambda xs: all(x is not None for x in xs))
        return strat

    if origin is Literal:
        allowed = get_args(annotation)
        return st.sampled_from(allowed)

    if origin is dict:
        _, val_type = get_args(annotation)
        strat = st.dictionaries(
            keys=st.text(min_size=0, max_size=8),
            values=_value_strategy_for_annotation(val_type),
            max_size=4,
        )
        val_origin = get_origin(val_type)
        val_has_none = val_origin in (Union, types.UnionType) and type(None) in get_args(val_type)
        if not val_has_none and val_type is not type(None):
            strat = strat.filter(lambda d: all(v is not None for v in d.values()))
        return strat

    if origin is types.UnionType or origin is getattr(types, "UnionType", object):
        args = get_args(annotation)
        if type(None) in args:
            non_none = [a for a in args if a is not type(None)]
            # choose None or one branch
            return st.one_of(
                st.none(),
                st.sampled_from(non_none).flatmap(_value_strategy_for_annotation),
            )
        # Non-optional union: pick one branch and generate a value for it
        return st.sampled_from(args).flatmap(_value_strategy_for_annotation)

    return st.just(None)


@st.composite
def annotation_and_value(draw) -> tuple[Any, Any]:
    ann = draw(_annotation_strategy(include_unions=False))  # parse-supported
    val = draw(_value_strategy_for_annotation(ann))
    return ann, val


@st.composite
def two_annotations(draw) -> tuple[Any, Any]:
    return draw(_annotation_strategy()), draw(_annotation_strategy())


def _assert_matches_annotation(parsed: Any, annotation: Any) -> None:
    origin = get_origin(annotation)

    if annotation is str:
        assert isinstance(parsed, str)
        return
    if annotation is int:
        assert isinstance(parsed, int)
        return
    if annotation is float:
        assert isinstance(parsed, float)
        return
    if annotation is bool:
        assert isinstance(parsed, bool)
        return
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        assert isinstance(parsed, annotation)
        return

    if origin is Literal:
        allowed = get_args(annotation)
        assert parsed in allowed
        return

    if origin is list:
        inner = get_args(annotation)[0]
        assert isinstance(parsed, list)
        for item in parsed:
            _assert_matches_annotation(item, inner)
        return

    if origin is dict:
        val_type = get_args(annotation)[1]
        assert isinstance(parsed, dict)
        for v in parsed.values():
            _assert_matches_annotation(v, val_type)
        return

    if origin is types.UnionType or origin is getattr(types, "UnionType", object):
        if parsed is None:
            return
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        # If optional, use the single non-None branch; otherwise accept any branch match
        if len(args) == 2 and type(None) in args and len(non_none) == 1:
            _assert_matches_annotation(parsed, non_none[0])
        else:
            assert any(_matches_annotation(parsed, a) for a in args)
        return

    return


def _matches_annotation(parsed: Any, annotation: Any) -> bool:
    try:
        _assert_matches_annotation(parsed, annotation)
        return True
    except AssertionError:
        return False


def _dedupe_union(members: tuple[Any, ...]) -> Any:
    """Deduplicate while preserving order; collapse to single type if only one remains."""

    def _equivalent(a: Any, b: Any) -> bool:
        if a is b:
            return True
        if inspect.isclass(a) and inspect.isclass(b):
            if issubclass(a, BaseModel) and issubclass(b, BaseModel):
                return a.__name__ == b.__name__
        return False

    unique: list[Any] = []
    for m in members:
        if not any(_equivalent(m, u) for u in unique):
            unique.append(m)
    if len(unique) == 1:
        return unique[0]
    return reduce(lambda a, b: a | b, unique)


# --- Parsing invariants (existing coverage) ---


@TEST_SETTINGS
@given(st.lists(_model_dict_strategy(), max_size=5))
def test_list_of_models_validated(dicts: list[dict[str, Any]]):
    result = _parse_output_value(dicts, list[SmallModel])
    assert len(result) == len(dicts)
    assert all(isinstance(item, SmallModel) for item in result)


@TEST_SETTINGS
@given(st.lists(_model_dict_strategy(), max_size=5))
def test_optional_list_of_models_validated_when_present(dicts: list[dict[str, Any]]):
    result = _parse_output_value(dicts, list[SmallModel] | None)
    assert result is not None
    assert len(result) == len(dicts)
    assert all(isinstance(item, SmallModel) for item in result)


def test_optional_list_of_models_allows_none():
    assert _parse_output_value(None, list[SmallModel] | None) is None


@TEST_SETTINGS
@given(st.lists(st.one_of(_model_dict_strategy(), st.none()), max_size=6))
def test_list_of_optional_models_preserves_nones(items: list[dict[str, Any] | None]):
    result = _parse_output_value(items, list[SmallModel | None])
    assert len(result) == len(items)
    for source, parsed in zip(items, result):
        if source is None:
            assert parsed is None
        else:
            assert isinstance(parsed, SmallModel)


@TEST_SETTINGS
@given(_model_dict_strategy())
def test_optional_model_validates_dict(value: dict[str, Any]):
    result = _parse_output_value(value, SmallModel | None)
    assert isinstance(result, SmallModel)


def test_optional_model_allows_none():
    assert _parse_output_value(None, SmallModel | None) is None


@TEST_SETTINGS
@given(st.lists(st.text(max_size=8), max_size=6))
def test_primitive_list_passthrough(values: list[str]):
    assert _parse_output_value(values, list[str]) == values


PRIMITIVE_TYPES = st.sampled_from([str, int, float, bool])


@TEST_SETTINGS
@given(PRIMITIVE_TYPES)
def test_is_optional_type_positive(base_type: type):
    assert _is_optional_type(base_type | None) is True


@TEST_SETTINGS
@given(PRIMITIVE_TYPES, PRIMITIVE_TYPES)
def test_is_optional_type_negative(t1: type, t2: type):
    annotation = t1 | t2
    assert _is_optional_type(annotation) is False


# --- Rendering invariants ---


@TEST_SETTINGS
@given(_annotation_without_optional_strategy())
def test_ts_type_optional_monotonic(annotation: Any):
    base = _ts_type(annotation)
    optional = _ts_type(annotation | None)
    assert "null" in optional
    base_set = set(part.strip() for part in base.split("|"))
    opt_set = set(part.strip() for part in optional.split("|"))
    assert base_set.issubset(opt_set)


@TEST_SETTINGS
@given(_annotation_without_optional_strategy())
def test_render_type_str_optional_monotonic(annotation: Any):
    base = _render_type_str(annotation)
    optional = _render_type_str(annotation | None)
    assert "null" in optional
    simplified_base = base.replace(" or null", "")
    assert simplified_base in optional or base in optional


# --- Schema/build/parse coverage ---


class _MockFieldInfo:
    def __init__(self, annotation: Any, description: str | None = None):
        self.annotation = annotation
        self.description = description


class _MockSignature:
    def __init__(self, output_fields: dict[str, tuple[Any, str | None]]):
        self.input_fields = {}
        self.instructions = ""
        self.output_fields = {
            name: _MockFieldInfo(ann, desc) for name, (ann, desc) in output_fields.items()
        }


@TEST_SETTINGS
@given(annotation_and_value())
def test_schema_build_and_parse_round_trip(pair: tuple[Any, Any]):
    annotation, value = pair
    sig = _MockSignature({"result": (annotation, None)})

    schema = _build_output_schema(sig)
    assert "properties" in schema and "result" in schema["properties"]

    parsed = _parse_output_value(value, annotation)
    _assert_matches_annotation(parsed, annotation)


@TEST_SETTINGS
@given(_annotation_strategy(max_union=5), st.data())
def test_parse_union_multi_branch(annotation: Any, data):
    # Only test unions with >=2 branches
    origin = get_origin(annotation)
    assume(origin is types.UnionType or origin is Union)
    args = get_args(annotation)
    assume(len(args) >= 2)

    # Generate a value matching one branch
    branch = data.draw(st.sampled_from(args))
    if branch is type(None):
        value = None
    else:
        value = data.draw(_value_strategy_for_annotation(branch))

    parsed = _parse_output_value(value, annotation)
    if branch is type(None):
        assert parsed is None
    else:
        assert any(_matches_annotation(parsed, a) for a in args if a is not type(None))


@TEST_SETTINGS
@given(two_annotations())
def test_format_turn2_json_handles_multiple_outputs(pair: tuple[Any, Any]):
    ann1, ann2 = pair
    sig = _MockSignature(
        {
            "first": (ann1, "first output"),
            "second": (ann2, "second output"),
        }
    )

    adapter = CodexAdapter()
    out = adapter.format_turn2_json(sig)
    assert "first" in out and "second" in out
    assert "{" in out and "}" in out
