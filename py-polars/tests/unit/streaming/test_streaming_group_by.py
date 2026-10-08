from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.exceptions import DuplicateError
from polars.testing import assert_frame_equal
from tests.unit.conftest import INTEGER_DTYPES
from tests.unit.streaming.conftest import assert_engines_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.mark.slow
def test_streaming_group_by_sorted_fast_path_nulls_10273() -> None:
    df = pl.Series(
        name="x",
        values=(
            *(i for i in range(4) for _ in range(100)),
            *(None for _ in range(100)),
        ),
    ).to_frame()

    assert (
        df.set_sorted("x")
        .lazy()
        .group_by("x")
        .agg(pl.len())
        .collect(engine="streaming")
        .sort("x")
    ).to_dict(as_series=False) == {
        "x": [None, 0, 1, 2, 3],
        "len": [100, 100, 100, 100, 100],
    }


def test_streaming_group_by_types() -> None:
    df = pl.DataFrame(
        {
            "person_id": [1, 1],
            "year": [1995, 1995],
            "person_name": ["bob", "foo"],
            "bool": [True, False],
            "date": [date(2022, 1, 1), date(2022, 1, 1)],
        }
    )

    for by in ["person_id", "year", "date", ["person_id", "year"]]:
        out = (
            (
                df.lazy()
                .group_by(by)
                .agg(
                    [
                        pl.col("person_name").first().alias("str_first"),
                        pl.col("person_name").last().alias("str_last"),
                        pl.col("bool").first().alias("bool_first"),
                        pl.col("bool").last().alias("bool_last"),
                        pl.col("bool").mean().alias("bool_mean"),
                        pl.col("bool").sum().alias("bool_sum"),
                        # pl.col("date").sum().alias("date_sum"),
                        # Date streaming mean/median has been temporarily disabled
                        # pl.col("date").mean().alias("date_mean"),
                        pl.col("date").first().alias("date_first"),
                        pl.col("date").last().alias("date_last"),
                        pl.col("date").min().alias("date_min"),
                        pl.col("date").max().alias("date_max"),
                    ]
                )
            )
            .select(pl.all().exclude(by))
            .collect(engine="streaming")
        )
        assert out.schema == {
            "str_first": pl.String,
            "str_last": pl.String,
            "bool_first": pl.Boolean,
            "bool_last": pl.Boolean,
            "bool_mean": pl.Float64,
            "bool_sum": pl.get_index_type(),
            # "date_sum": pl.Date,
            # "date_mean": pl.Date,
            "date_first": pl.Date,
            "date_last": pl.Date,
            "date_min": pl.Date,
            "date_max": pl.Date,
        }

        assert out.to_dict(as_series=False) == {
            "str_first": ["bob"],
            "str_last": ["foo"],
            "bool_first": [True],
            "bool_last": [False],
            "bool_mean": [0.5],
            "bool_sum": [1],
            # "date_sum": [None],
            # Date streaming mean/median has been temporarily disabled
            # "date_mean": [date(2022, 1, 1)],
            "date_first": [date(2022, 1, 1)],
            "date_last": [date(2022, 1, 1)],
            "date_min": [date(2022, 1, 1)],
            "date_max": [date(2022, 1, 1)],
        }

    with pytest.raises(DuplicateError):
        (
            df.lazy()
            .group_by("person_id")
            .agg(
                [
                    pl.col("person_name").first().alias("str_first"),
                    pl.col("person_name").last().alias("str_last"),
                    pl.col("person_name").mean().alias("str_mean"),
                    pl.col("bool").first().alias("bool_first"),
                    pl.col("bool").last().alias("bool_first"),
                ]
            )
            .select(pl.all().exclude("person_id"))
            .collect(engine="streaming")
        )


def test_streaming_group_by_min_max() -> None:
    df = pl.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6],
            "year": [1995, 1995, 1995, 2, 2, 2],
        }
    )
    out = (
        df.lazy()
        .group_by("year")
        .agg([pl.min("person_id").alias("min"), pl.max("person_id").alias("max")])
        .collect()
        .sort("year")
    )
    assert out["min"].to_list() == [4, 1]
    assert out["max"].to_list() == [6, 3]


def test_streaming_non_streaming_gb() -> None:
    n = 100
    df = pl.DataFrame({"a": np.random.randint(0, 20, n)})
    q = df.lazy().group_by("a").agg(pl.len()).sort("a")
    assert_frame_equal(q.collect(engine="streaming"), q.collect())

    q = df.lazy().with_columns(pl.col("a").cast(pl.String))
    q = q.group_by("a").agg(pl.len()).sort("a")
    assert_frame_equal(q.collect(engine="streaming"), q.collect())
    q = df.lazy().with_columns(pl.col("a").alias("b"))
    q = q.group_by(["a", "b"]).agg(pl.len(), pl.col("a").sum().alias("sum_a")).sort("a")
    assert_frame_equal(q.collect(engine="streaming"), q.collect())


def test_streaming_group_by_sorted_fast_path() -> None:
    a = np.random.randint(0, 20, 80)
    df = pl.DataFrame(
        {
            # test on int8 as that also tests proper conversions
            "a": pl.Series(np.sort(a), dtype=pl.Int8)
        }
    ).with_row_index()

    df_sorted = df.with_columns(pl.col("a").set_sorted())

    for streaming in [True, False]:
        results = []
        for df_ in [df, df_sorted]:
            out = (
                df_.lazy()
                .group_by("a")
                .agg(
                    [
                        pl.first("a").alias("first"),
                        pl.last("a").alias("last"),
                        pl.sum("a").alias("sum"),
                        pl.mean("a").alias("mean"),
                        pl.count("a").alias("count"),
                        pl.min("a").alias("min"),
                        pl.max("a").alias("max"),
                    ]
                )
                .sort("a")
                .collect(engine="streaming" if streaming else "in-memory")
            )
            results.append(out)

        assert_frame_equal(results[0], results[1])


def test_streaming_group_by_slice_out_of_bounds_28554() -> None:
    q = (
        pl.LazyFrame({"k": [1], "a": [1]})
        .group_by("k")
        .agg(s=pl.col("a").sum())
        .slice(2, 1)
    )

    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


# Small hot tables, morsels and waves give small inputs cold rows, evictions,
# and several rounds of a few frames each. Unless the partition count is forced,
# every thread gets a partition.
SMALL_ENV = {
    "POLARS_GROUP_BY_MIN_ROWS_PER_PARTITION": "1",
    "POLARS_GROUP_BY_WAVE_BYTES": "4096",
    "POLARS_HOT_TABLE_SIZE": "4",
    "POLARS_MAX_HOT_TABLE_SIZE": "16",
    "POLARS_IDEAL_MORSEL_SIZE": "32",
}

# The memory manager only sees allocations once a thread's drift exceeds the
# drift threshold, so small frames never trigger a spill unless it is zero.
SPILL_ENV = {
    "POLARS_OOC_MEMORY_BUDGET_MB": "0",
    "POLARS_OOC_SPILL_MIN_BYTES": "1",
    "POLARS_OOC_DRIFT_THRESHOLD": "0",
}


def _set_env(plmonkeypatch: PlMonkeyPatch, env: dict[str, str]) -> None:
    for name, value in env.items():
        plmonkeypatch.setenv(name, value)


def _set_spill_env(plmonkeypatch: PlMonkeyPatch, tmp_path: Path) -> None:
    _set_env(plmonkeypatch, SPILL_ENV)
    plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(tmp_path))


def _num_written_out_partitions() -> int:
    # More partitions than threads, and not a multiple of them.
    return 4 * pl.thread_pool_size() + 3


@pytest.fixture(
    params=[
        "direct",
        "written_out",
        pytest.param("spilled", marks=[pytest.mark.write_disk, pytest.mark.slow]),
    ]
)
def group_by_mode(
    request: pytest.FixtureRequest, plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> str:
    mode = str(request.param)
    if mode == "old":
        # The default group-by node, with hot tables smaller than the inputs.
        plmonkeypatch.setenv("POLARS_HOT_TABLE_SIZE", "16")
        plmonkeypatch.setenv("POLARS_MAX_HOT_TABLE_SIZE", "16")
        return mode
    plmonkeypatch.setenv("POLARS_NEW_GROUPBY", "1")
    _set_env(plmonkeypatch, SMALL_ENV)
    if mode != "direct":
        num_partitions = _num_written_out_partitions()
        plmonkeypatch.setenv("POLARS_GROUP_BY_NUM_PARTITIONS", str(num_partitions))
    if mode == "spilled":
        _set_spill_env(plmonkeypatch, tmp_path)
    return mode


# Order-sensitive aggregations keep every key hot, so nothing is stored or spilled.
HOT_ONLY = pytest.mark.parametrize(
    "group_by_mode", ["direct", "written_out"], indirect=True
)

# The default group-by node and the new one's modes without spilling.
HOT_ONLY_AND_OLD = pytest.mark.parametrize(
    "group_by_mode", ["old", "direct", "written_out"], indirect=True
)


@pytest.fixture(scope="module")
def random_integers() -> pl.Series:
    np.random.seed(1)
    return pl.Series("a", np.random.randint(0, 10, 100), dtype=pl.Int64)


@HOT_ONLY_AND_OLD
def test_streaming_group_by_ooc_q1(
    random_integers: pl.Series, group_by_mode: str
) -> None:
    lf = random_integers.to_frame().lazy()
    result = (
        lf.group_by("a")
        .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
        .sort("a")
        .collect(engine="streaming")
    )

    expected = pl.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_first": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_last": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    assert_frame_equal(result, expected)


@HOT_ONLY_AND_OLD
def test_streaming_group_by_ooc_q2(
    random_integers: pl.Series, group_by_mode: str
) -> None:
    lf = random_integers.cast(str).to_frame().lazy()
    result = (
        lf.group_by("a")
        .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
        .sort("a")
        .collect(engine="streaming")
    )

    expected = pl.DataFrame(
        {
            "a": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "a_first": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "a_last": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        }
    )
    assert_frame_equal(result, expected)


@HOT_ONLY_AND_OLD
def test_streaming_group_by_ooc_q3(
    random_integers: pl.Series, group_by_mode: str
) -> None:
    lf = pl.LazyFrame({"a": random_integers, "b": random_integers})
    result = (
        lf.group_by("a", "b")
        .agg(pl.first("a").alias("a_first"), pl.last("a").alias("a_last"))
        .sort("a")
        .collect(engine="streaming")
    )

    expected = pl.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_first": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "a_last": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )
    assert_frame_equal(result, expected)


def test_streaming_group_by_struct_key() -> None:
    df = pl.DataFrame(
        {"A": [1, 2, 3, 2], "B": ["google", "ms", "apple", "ms"], "C": [2, 3, 4, 3]}
    )
    df1 = df.lazy().with_columns(pl.struct(["A", "C"]).alias("tuples"))
    assert df1.group_by("tuples").agg(pl.len(), pl.col("B").first()).sort("B").collect(
        engine="streaming"
    ).to_dict(as_series=False) == {
        "tuples": [{"A": 3, "C": 4}, {"A": 1, "C": 2}, {"A": 2, "C": 3}],
        "len": [1, 1, 2],
        "B": ["apple", "google", "ms"],
    }


@pytest.mark.slow
def test_streaming_group_by_all_numeric_types_stability_8570() -> None:
    m = 1000
    n = 1000

    rng = np.random.default_rng(seed=0)
    dfa = pl.DataFrame({"x": pl.arange(start=0, end=n, eager=True)})
    dfb = pl.DataFrame(
        {
            "y": rng.integers(low=0, high=10, size=m),
            "z": rng.integers(low=0, high=2, size=m),
        }
    )
    dfc = dfa.join(dfb, how="cross")

    for keys in [["x", "y"], "z"]:
        for dtype in [*INTEGER_DTYPES, pl.Boolean]:
            # the alias checks if the schema is correctly handled
            dfd = (
                dfc.lazy()
                .with_columns(pl.col("z").cast(dtype))
                .group_by(keys)
                .agg(pl.col("z").sum().alias("z_sum"))
                .collect(engine="streaming")
            )
            assert dfd["z_sum"].sum() == dfc["z"].sum()


def test_streaming_group_by_categorical_aggregate() -> None:
    out = (
        pl.LazyFrame(
            {
                "a": pl.Series(
                    ["a", "a", "b", "b", "c", "c", None, None], dtype=pl.Categorical
                ),
                "b": pl.Series(
                    pl.date_range(
                        date(2023, 4, 28),
                        date(2023, 5, 5),
                        eager=True,
                    ).to_list(),
                    dtype=pl.Date,
                ),
            }
        )
        .group_by(["a", "b"])
        .agg([pl.col("a").first().alias("sum")])
        .collect(engine="streaming")
    )

    assert out.sort("b").to_dict(as_series=False) == {
        "a": ["a", "a", "b", "b", "c", "c", None, None],
        "b": [
            date(2023, 4, 28),
            date(2023, 4, 29),
            date(2023, 4, 30),
            date(2023, 5, 1),
            date(2023, 5, 2),
            date(2023, 5, 3),
            date(2023, 5, 4),
            date(2023, 5, 5),
        ],
        "sum": ["a", "a", "b", "b", "c", "c", None, None],
    }


def test_streaming_group_by_list_9758() -> None:
    payload = {"a": [[1, 2]]}
    assert (
        pl.LazyFrame(payload)
        .group_by("a")
        .first()
        .collect(engine="streaming")
        .to_dict(as_series=False)
        == payload
    )


def test_group_by_min_max_string_type() -> None:
    table = pl.from_dict({"a": [1, 1, 2, 2, 2], "b": ["a", "b", "c", "d", None]})

    expected = {"a": [1, 2], "min": ["a", "c"], "max": ["b", "d"]}

    for streaming in [True, False]:
        assert (
            table.lazy()
            .group_by("a")
            .agg([pl.min("b").alias("min"), pl.max("b").alias("max")])
            .collect(engine="streaming" if streaming else "in-memory")
            .sort("a")
            .to_dict(as_series=False)
            == expected
        )


@pytest.mark.parametrize("literal", [True, "foo", 1])
def test_streaming_group_by_literal(literal: Any) -> None:
    df = pl.LazyFrame({"a": range(20)})

    assert df.group_by(pl.lit(literal)).agg(
        [
            pl.col("a").count().alias("a_count"),
            pl.col("a").sum().alias("a_sum"),
        ]
    ).collect(engine="streaming").to_dict(as_series=False) == {
        "literal": [literal],
        "a_count": [20],
        "a_sum": [190],
    }


@pytest.mark.parametrize("streaming", [True, False])
def test_group_by_multiple_keys_one_literal(streaming: bool) -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})

    expected = {"a": [1, 2], "literal": [1, 1], "b": [5, 6]}
    assert (
        df.lazy()
        .group_by("a", pl.lit(1))
        .agg(pl.col("b").max())
        .sort(["a", "b"])
        .collect(engine="streaming" if streaming else "in-memory")
        .to_dict(as_series=False)
        == expected
    )


def test_streaming_group_null_count() -> None:
    df = pl.DataFrame({"g": [1] * 6, "a": ["yes", None] * 3}).lazy()
    assert df.group_by("g").agg(pl.col("a").count()).collect(
        engine="streaming"
    ).to_dict(as_series=False) == {"g": [1], "a": [3]}


def test_streaming_group_by_binary_15116() -> None:
    assert (
        pl.LazyFrame(
            {
                "str": [
                    "A",
                    "A",
                    "BB",
                    "BB",
                    "CCCC",
                    "CCCC",
                    "DDDDDDDD",
                    "DDDDDDDD",
                    "EEEEEEEEEEEEEEEE",
                    "A",
                ]
            }
        )
        .select([pl.col("str").cast(pl.Binary)])
        .group_by(["str"])
        .agg([pl.len().alias("count")])
    ).sort("str").collect(engine="streaming").to_dict(as_series=False) == {
        "str": [b"A", b"BB", b"CCCC", b"DDDDDDDD", b"EEEEEEEEEEEEEEEE"],
        "count": [3, 2, 2, 2, 1],
    }


def test_streaming_group_by_convert_15380(partition_limit: int) -> None:
    assert (
        pl.DataFrame({"a": [1] * partition_limit}).group_by(b="a").len()["len"].item()
        == partition_limit
    )


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("n_rows_limit_offset", [-1, +3])
def test_streaming_group_by_boolean_mean_15610(
    n_rows_limit_offset: int, streaming: bool, partition_limit: int
) -> None:
    n_rows = partition_limit + n_rows_limit_offset

    # Also test non-streaming because it sometimes dispatched to streaming agg.
    expect = pl.DataFrame({"a": [False, True], "c": [0.0, 0.5]})

    n_repeats = n_rows // 3
    assert n_repeats > 0

    out = (
        pl.select(
            a=pl.repeat([True, False, True], n_repeats).explode(empty_as_null=True),
            b=pl.repeat([True, False, False], n_repeats).explode(empty_as_null=True),
        )
        .lazy()
        .group_by("a")
        .agg(c=pl.mean("b"))
        .sort("a")
        .collect(engine="streaming" if streaming else "in-memory")
    )

    assert_frame_equal(out, expect)


def test_streaming_group_by_all_null_21593() -> None:
    df = pl.DataFrame(
        {
            "col_1": ["A", "B", "C", "D"],
            "col_2": ["test", None, None, None],
        }
    )

    out = df.lazy().group_by(pl.all()).min().collect(engine="streaming")
    assert_frame_equal(df, out, check_row_order=False)


def test_streaming_group_by_nested_agg_fallback() -> None:
    n = 1001
    df = pl.DataFrame(
        {
            "id": range(n),
            "key": ["aaa" if i < n // 3 else "bbb" for i in range(n)],
        }
    )

    # nested agg `col.len().sum()` maps to `Sum(Count(col))` in IR, which the streaming
    # engine cannot (currently) lower; fallback should have used in-memory engine but
    # was re-entering the streaming engine, causing infinite recursion and SIGSEGV.
    res = (
        df.lazy()
        .group_by("key")
        .agg(pl.col("id").len().sum())
        .sort("key")
        .collect(engine="streaming")
    )
    expected = {("aaa", n // 3), ("bbb", n - n // 3)}
    assert expected == set(res.rows())


@pytest.mark.parametrize("n_groups", [3, 5000])
def test_streaming_group_by_shared_agg_subexpression(n_groups: int) -> None:
    n = 20_000
    df = pl.DataFrame(
        {
            "g": [i % n_groups for i in range(n)],
            "a": [float(i) for i in range(n)],
            "b": [None if i % 11 == 0 else (i % 7) / 10 for i in range(n)],
            "c": [(i % 5) / 10 for i in range(n)],
        }
    )
    shared = pl.col("a") * (1 - pl.col("b"))
    q = (
        df.lazy()
        .group_by("g")
        .agg(
            shared.sum().alias("s1"),
            (shared * (1 + pl.col("c"))).sum().alias("s2"),
            (shared * (1 + pl.col("c"))).max().alias("m2"),
            pl.col("b").mean(),
        )
    )
    assert_frame_equal(
        q.collect(engine="streaming"),
        q.collect(engine="in-memory"),
        check_row_order=False,
    )


@pytest.mark.parametrize("key_dtype", [pl.Int64, pl.String])
def test_streaming_group_by_hot_table_growth(
    key_dtype: pl.DataType,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "1000")
    plmonkeypatch.setenv("POLARS_HOT_TABLE_SIZE", "2")
    plmonkeypatch.setenv("POLARS_MAX_HOT_TABLE_SIZE", "256")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    # Frequent keys, more than the initial hot table holds, mixed with keys that
    # occur only once.
    n = 200_000
    rng = np.random.default_rng(0)
    heavy = rng.integers(0, 30, n)
    unique = np.arange(1_000_000, 1_000_000 + n)
    key = np.where(rng.random(n) < 0.7, heavy, unique)
    df = pl.DataFrame({"g": key, "v": np.arange(n)}).with_columns(
        pl.col("g").cast(key_dtype)
    )
    q = (
        df.lazy()
        .group_by("g")
        .agg(pl.col("v").sum().alias("sum"), pl.col("v").first().alias("first"))
    )

    capfd.readouterr()
    out = q.collect(engine="streaming")
    assert "[group-by]: hot table" in capfd.readouterr().err
    assert_frame_equal(out, q.collect(engine="in-memory"), check_row_order=False)


@HOT_ONLY_AND_OLD
@pytest.mark.parametrize(
    "keys",
    [
        [pl.col("a")],
        [pl.col("s")],
        [pl.col("a"), pl.col("b")],
        [pl.struct("a", "s")],
    ],
)
def test_streaming_group_by_sorted_runs(
    group_by_mode: str, keys: list[pl.Expr]
) -> None:
    # Runs of equal keys, with far more keys than the hot table holds.
    n = 2_000
    a = np.repeat(np.arange(n // 4), 4)
    df = pl.DataFrame({"a": a, "v": np.arange(n)}).with_columns(
        pl.when(pl.col("a") % 10 != 0).then(pl.col("a")).alias("a"),
        b=pl.col("a") % 7,
        s=pl.col("a").cast(pl.String),
    )
    q = (
        df.lazy()
        .group_by(keys)
        .agg(pl.col("v").sum(), pl.col("v").min().alias("min"), pl.len())
    )
    assert_engines_equal(q)


@pytest.mark.parametrize("n_groups", [3, 100])
def test_streaming_group_by_null_on_empty_after_valid_morsels(
    n_groups: int, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_IDEAL_MORSEL_SIZE", "50")

    # Morsels without nulls, then a group that only has nulls.
    n = 1_000
    df = pl.DataFrame(
        {
            "g": [i % n_groups for i in range(n)] + [n_groups] * 10,
            "v": [*range(n), *([None] * 10)],
        }
    )
    v = pl.col("v")
    q = (
        df.lazy()
        .group_by("g")
        .agg(
            s=pl.when(v.count() > 0).then(v.sum()),
            mn=v.min(),
            mx=v.max(),
        )
    )
    out = q.collect(engine="streaming")
    assert out.filter(pl.col("g") == n_groups).select("s", "mn", "mx").row(0) == (
        None,
        None,
        None,
    )
    assert_frame_equal(out, q.collect(engine="in-memory"), check_row_order=False)


def _payload_columns(n: int) -> list[pl.Series]:
    # Short strings are stored inline, long ones in buffers.
    words = [f"p{j % 101}" if j % 2 else f"payload-string-{j % 101}" for j in range(n)]
    return [
        pl.Series("v", [None if j % 11 == 0 else j for j in range(n)], dtype=pl.Int64),
        pl.Series(
            "s",
            [None if j % 13 == 0 else w for j, w in enumerate(words)],
            dtype=pl.String,
        ),
    ]


def _aggs(*, order_sensitive: bool = False) -> list[pl.Expr]:
    v = pl.col("v")
    aggs = [pl.len(), v.sum(), v.min().alias("v_min"), pl.col("s").max()]
    if order_sensitive:
        aggs += [
            v.first().alias("v_first"),
            v.last().alias("v_last"),
            pl.col("s").first().alias("s_first"),
        ]
    return aggs


def _mixed_keys(n: int, low: int, high: int, seed: int) -> list[int | None]:
    """Keys that arrive one at a time in the first half and in runs of two after."""
    rng = np.random.default_rng(seed)
    num_runs = n // 4
    singles = rng.integers(low, high, n - 2 * num_runs)
    runs = np.repeat(rng.integers(low, high, num_runs), 2)
    keys = [int(v) for v in np.concatenate([singles, runs])]
    return [None if j % 37 == 0 else v for j, v in enumerate(keys)]


KEY_ROWS = 600


def _key_series() -> dict[str, pl.Series]:
    n = KEY_ROWS
    ints = _mixed_keys(n, -40, 40, seed=1234)

    floats: list[float | None] = []
    for j, v in enumerate(ints):
        if v is None:
            floats.append(None)
        elif j % 53 == 0:
            floats.append(float("nan"))
        elif j % 101 == 0:
            floats.append(float("inf") if j % 202 == 0 else float("-inf"))
        elif j % 71 == 0:
            floats.append(-0.0)
        else:
            floats.append(v / 8)

    words = ["", "a", "zz", "Ångström", "日本語", "b" * 300, "aa", "ab"]
    strings = [None if v is None else f"{words[abs(v) % len(words)]}{v}" for v in ints]
    cats = [str(v) for v in range(-40, 40)]
    cat_values = [None if v is None else str(v) for v in ints]
    decimals = [None if v is None else Decimal(v) / Decimal(1000) for v in ints]

    # Single keys of each physical width, string keys, and row-encoded keys.
    return {
        "i8": pl.Series("k", ints, dtype=pl.Int8),
        "i64": pl.Series("k", ints, dtype=pl.Int64),
        "i128": pl.Series(
            "k", [None if v is None else v * 10**30 for v in ints], dtype=pl.Int128
        ),
        "f16": pl.Series("k", floats, dtype=pl.Float16),
        "f64": pl.Series("k", floats, dtype=pl.Float64),
        "datetime": pl.Series(
            "k", [None if v is None else v * 3_600_000_000 for v in ints]
        ).cast(pl.Datetime("us")),
        "decimal": pl.Series("k", decimals, dtype=pl.Decimal(18, 3)),
        "enum": pl.Series("k", cat_values, dtype=pl.Enum(cats)),
        "categorical": pl.Series("k", cat_values, dtype=pl.Categorical),
        "string": pl.Series("k", strings, dtype=pl.String),
        "binary": pl.Series(
            "k",
            [None if s is None else s.encode() for s in strings],
            dtype=pl.Binary,
        ),
        "boolean": pl.Series(
            "k", [None if v is None else v % 2 == 0 for v in ints], dtype=pl.Boolean
        ),
        "struct": pl.Series(
            "k",
            [
                None if v is None else {"a": None if v % 5 == 0 else v % 3, "b": str(v)}
                for v in ints
            ],
            dtype=pl.Struct({"a": pl.Int32, "b": pl.String}),
        ),
        "null": pl.Series("k", [None] * n, dtype=pl.Null),
        "all_null": pl.Series("k", [None] * n, dtype=pl.Int64),
    }


KEY_SERIES = _key_series()
KEY_FRAME = pl.DataFrame(
    [*(s.alias(name) for name, s in KEY_SERIES.items()), *_payload_columns(KEY_ROWS)]
)


@pytest.mark.parametrize("key_dtype", list(KEY_SERIES))
def test_streaming_group_by_key_dtypes(group_by_mode: str, key_dtype: str) -> None:
    assert_engines_equal(KEY_FRAME.lazy().group_by(key_dtype).agg(_aggs()))


@pytest.mark.parametrize(
    "keys",
    [
        # Key rows with columns of every width.
        ["f64", "boolean", "i8", "categorical"],
        ["i128", "f16", "string"],
        ["binary", pl.col("string").str.reverse().alias("reversed")],
        # Row-encoded keys.
        ["i64", "struct"],
        [pl.struct("i64", "string").alias("pair")],
    ],
)
def test_streaming_group_by_multiple_keys(
    group_by_mode: str, keys: list[str | pl.Expr]
) -> None:
    assert_engines_equal(KEY_FRAME.lazy().group_by(keys).agg(_aggs()))


SHAPE_ROWS = 2_000


def _shape_keys() -> dict[str, pl.Series]:
    n = SHAPE_ROWS
    rng = np.random.default_rng(42)
    idx = np.arange(n)
    shapes = {
        # The same keys in several bursts, in runs of two.
        "bursts": np.concatenate(
            [np.repeat(rng.permutation(n // 8), 2) for _ in range(4)]
        ),
        "unique": rng.permutation(n),
        # A few frequent keys and many rare ones, so cold rows are gathered.
        "skewed": np.where(
            rng.random(n) < 0.7, rng.integers(0, 5, n), rng.integers(5, 100_000, n)
        ),
        # One hot key, so that most morsels have at least 75% cold rows.
        "mostly_cold": np.where(idx % 8 == 0, 0, n + idx),
    }
    keys = {name: pl.Series("k", k, dtype=pl.Int64) for name, k in shapes.items()}
    keys["all_null"] = pl.Series("k", [None] * n, dtype=pl.Int64)
    return keys


SHAPE_PAYLOAD = _payload_columns(SHAPE_ROWS)
SHAPE_FRAMES = {
    name: pl.DataFrame([keys, *SHAPE_PAYLOAD]) for name, keys in _shape_keys().items()
}


@pytest.mark.parametrize("shape", list(SHAPE_FRAMES))
def test_streaming_group_by_data_shapes(group_by_mode: str, shape: str) -> None:
    assert_engines_equal(SHAPE_FRAMES[shape].lazy().group_by("k").agg(_aggs()))


@HOT_ONLY
@pytest.mark.parametrize("shape", list(SHAPE_FRAMES))
def test_streaming_group_by_maintain_order(group_by_mode: str, shape: str) -> None:
    lf = (
        SHAPE_FRAMES[shape]
        .lazy()
        .group_by("k", maintain_order=True)
        .agg(_aggs(order_sensitive=True))
    )
    assert_engines_equal(lf, check_row_order=True)


@HOT_ONLY
@pytest.mark.parametrize("keys", [["a"], ["a_str"], ["a", "b"], ["ab"]])
def test_streaming_group_by_first_last_many_keys(
    group_by_mode: str, keys: list[str]
) -> None:
    # Every missed key is made hot, so with many more keys than hot slots keys
    # are evicted again and again, often several times in one batch of evictions.
    n = 3_000
    rng = np.random.default_rng(3)
    df = (
        pl.DataFrame(
            [
                pl.Series("a", rng.integers(0, 200, n), dtype=pl.Int64),
                *_payload_columns(n),
            ]
        )
        .with_columns(a_str=pl.format("key-{}", "a"), b=pl.col("a") % 7)
        .with_columns(ab=pl.struct("a", "b"))
    )
    v = pl.col("v")
    s = pl.col("s")
    lf = (
        df.lazy()
        .group_by(keys)
        .agg(
            v.first().alias("v_first"),
            v.last().alias("v_last"),
            v.first(ignore_nulls=True).alias("v_first_non_null"),
            v.last(ignore_nulls=True).alias("v_last_non_null"),
            s.first().alias("s_first"),
            s.last(ignore_nulls=True).alias("s_last_non_null"),
            v.sum(),
        )
    )
    assert_engines_equal(lf)


@pytest.mark.parametrize(
    "keys",
    [
        [pl.lit("x"), pl.col("k")],
        [pl.col("k"), pl.lit(None, dtype=pl.Int32)],
    ],
)
def test_streaming_group_by_literal_keys(
    group_by_mode: str, keys: list[pl.Expr]
) -> None:
    assert_engines_equal(SHAPE_FRAMES["skewed"].lazy().group_by(keys).agg(_aggs()))


def _aggregation_frame() -> pl.DataFrame:
    n = 1_000
    rng = np.random.default_rng(7)
    raw = [int(v) for v in rng.integers(-50, 50, n)]
    floats = [v / 4 for v in raw]
    with_nans = [float("nan") if j % 29 == 0 else f for j, f in enumerate(floats)]
    df = pl.DataFrame(
        {
            "k": pl.Series(_mixed_keys(n, 0, 100, seed=7), dtype=pl.Int64),
            "i": pl.Series(
                [None if j % 11 == 0 else v for j, v in enumerate(raw)],
                dtype=pl.Int64,
            ),
            "x": pl.Series(
                [None if j % 13 == 0 else f for j, f in enumerate(floats)],
                dtype=pl.Float64,
            ),
            "f": pl.Series(
                [None if j % 13 == 0 else f for j, f in enumerate(with_nans)],
                dtype=pl.Float64,
            ),
            "b": pl.Series(
                [None if j % 7 == 0 else v % 3 == 0 for j, v in enumerate(raw)],
                dtype=pl.Boolean,
            ),
            "s": pl.Series(
                [None if j % 17 == 0 else f"s{v}" for j, v in enumerate(raw)],
                dtype=pl.String,
            ),
            "w": pl.Series(rng.permutation(n), dtype=pl.Int64),
        }
    )
    # The group of key 0 only has null values.
    return df.with_columns(
        [
            pl.when(pl.col("k").ne_missing(0)).then(pl.col(c)).alias(c)
            for c in ["i", "x", "f", "b", "s"]
        ]
    )


AGG_FRAME = _aggregation_frame()

# The engines differ on groups with fewer than two complete pairs: the streaming
# engine gives null where the in-memory one gives 0.0 (cov) or NaN (corr).
_TWO_PAIRS = (pl.col("i").is_not_null() & pl.col("x").is_not_null()).sum() >= 2

AGGREGATIONS = [
    pl.col("i").sum().alias("i_sum"),
    pl.col("x").sum().alias("x_sum"),
    pl.col("b").sum().alias("b_sum"),
    pl.col("i").mean().alias("i_mean"),
    pl.col("x").mean().alias("x_mean"),
    pl.col("i").min().alias("i_min"),
    pl.col("x").max().alias("x_max"),
    pl.col("f").min().alias("f_min"),
    pl.col("f").max().alias("f_max"),
    pl.col("f").nan_min().alias("f_nan_min"),
    pl.col("f").nan_max().alias("f_nan_max"),
    pl.col("s").min().alias("s_min"),
    pl.col("s").max().alias("s_max"),
    pl.col("b").min().alias("b_min"),
    pl.col("b").max().alias("b_max"),
    pl.col("x").var().alias("x_var"),
    pl.col("i").var(ddof=0).alias("i_var"),
    pl.col("x").std().alias("x_std"),
    pl.col("i").std(ddof=0).alias("i_std"),
    pl.len(),
    pl.col("i").count().alias("i_count"),
    pl.col("s").len().alias("s_len"),
    pl.col("x").null_count().alias("x_null_count"),
    pl.col("i").n_unique().alias("i_n_unique"),
    pl.col("s").n_unique().alias("s_n_unique"),
    pl.col("b").any().alias("b_any"),
    pl.col("b").all().alias("b_all"),
    pl.col("b").any(ignore_nulls=False).alias("b_any_with_nulls"),
    pl.col("b").all(ignore_nulls=False).alias("b_all_with_nulls"),
    pl.col("s").has_nulls().alias("s_has_nulls"),
    pl.col("i").is_empty(ignore_nulls=True).alias("i_is_empty"),
    pl.col("i").bitwise_and().alias("i_and"),
    pl.col("i").bitwise_or().alias("i_or"),
    pl.col("i").bitwise_xor().alias("i_xor"),
    pl.col("s").min_by("w").alias("s_min_by"),
    pl.col("i").max_by("w").alias("i_max_by"),
    pl.col("x").max_by(pl.struct("w")).alias("x_max_by_struct"),
    pl.when(_TWO_PAIRS).then(pl.cov("i", "x")).alias("cov"),
    pl.when(_TWO_PAIRS).then(pl.corr("i", "x")).alias("corr"),
    pl.col("x").skew().alias("x_skew"),
    pl.col("x").kurtosis().alias("x_kurtosis"),
    (pl.col("w") + 1).entropy().alias("w_entropy"),
    pl.col("i").implode(maintain_order=False).list.sort().alias("i_list"),
    pl.col("s").implode(maintain_order=False).list.sort().alias("s_list"),
    pl.col("x").approx_quantile(0.5).alias("x_median"),
    pl.col("i").approx_quantile([0.1, 0.9]).alias("i_quantiles"),
]


def test_streaming_group_by_aggregations(group_by_mode: str) -> None:
    assert_engines_equal(AGG_FRAME.lazy().group_by("k").agg(AGGREGATIONS))


def test_streaming_group_by_approx_n_unique(
    group_by_mode: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    lf = (
        AGG_FRAME.lazy()
        .group_by("k")
        .agg(
            pl.col("i").approx_n_unique().alias("i_approx"),
            pl.col("s").approx_n_unique().alias("s_approx"),
        )
    )
    out = lf.collect(engine="streaming")

    # The in-memory engine estimates differently, so compare against the
    # streaming engine with its default settings.
    plmonkeypatch.undo()
    assert_frame_equal(out, lf.collect(engine="streaming"), check_row_order=False)


def test_streaming_group_by_item(group_by_mode: str) -> None:
    v = pl.col("v").fill_null(-1)
    lf = (
        SHAPE_FRAMES["unique"]
        .lazy()
        .group_by("k")
        .agg(
            v.item().alias("v_item"),
            v.filter(v % 4 == 0).item(allow_empty=True).alias("v_item_div_4"),
        )
    )
    assert_engines_equal(lf)


@pytest.mark.xfail(
    reason="the streaming engine raises on the item of a single null value",
    raises=pl.exceptions.ComputeError,
)
def test_streaming_group_by_item_of_null() -> None:
    lf = pl.LazyFrame({"k": [1, 2], "v": [None, 3]})
    assert_engines_equal(lf.group_by("k").agg(pl.col("v").item()))


def test_streaming_group_by_multiple_inputs(group_by_mode: str) -> None:
    v = pl.col("v")
    lf = (
        SHAPE_FRAMES["skewed"]
        .lazy()
        .group_by("k")
        .agg(
            v.filter(v % 3 == 0).sum().alias("v_sum_div_3"),
            v.filter(pl.col("s").is_not_null()).max().alias("v_max_with_s"),
            v.drop_nulls().mean().alias("v_mean"),
            v.n_unique().alias("v_n_unique"),
            # Inputs that receive no rows.
            v.filter(v < 0).sum().alias("v_sum_negative"),
            pl.col("s").filter(v < 0).len().alias("s_len_negative"),
            v.sum(),
        )
    )
    assert_engines_equal(lf)


@pytest.mark.parametrize("shape", ["unique", "mostly_cold"])
def test_streaming_group_by_without_payload(group_by_mode: str, shape: str) -> None:
    gb = SHAPE_FRAMES[shape].lazy().group_by("k")
    v = pl.col("v")
    assert_engines_equal(gb.agg(pl.len()))
    # Frames without columns: with no aggregations, and for the input that only
    # has the keys next to a filtered input.
    assert_engines_equal(gb.agg())
    assert_engines_equal(gb.agg(v.filter(v > 0).sum()))


@pytest.mark.parametrize("height", [0, 1, 3])
def test_streaming_group_by_tiny_inputs(group_by_mode: str, height: int) -> None:
    df = pl.DataFrame(
        {
            "k": pl.Series([3, None, 3][:height], dtype=pl.Int64),
            "v": pl.Series([1, 2, None][:height], dtype=pl.Int64),
            "s": pl.Series(["a", None, "b"][:height], dtype=pl.String),
        }
    )
    v = pl.col("v")
    lf = (
        df.lazy()
        .group_by("k")
        .agg(
            *_aggs(),
            v.mean().alias("v_mean"),
            v.var().alias("v_var"),
            v.n_unique().alias("v_n_unique"),
            v.filter(v > 1).sum().alias("v_sum_above_1"),
        )
    )
    assert_engines_equal(lf)


def _fused_aggs() -> list[pl.Expr]:
    v = pl.col("v")
    return [
        (v * 2).sum().alias("v_double_sum"),
        (v * 3).max().alias("v_triple_max"),
        (v - 1).min().alias("v_dec_min"),
        (v.cast(pl.Float64) / 2).mean().alias("v_half_mean"),
        pl.col("s").str.len_bytes().sum().alias("s_bytes"),
        v.sum(),
    ]


@pytest.mark.parametrize("shape", ["skewed", "mostly_cold"])
def test_streaming_group_by_fused_aggregations(group_by_mode: str, shape: str) -> None:
    assert_engines_equal(SHAPE_FRAMES[shape].lazy().group_by("k").agg(_fused_aggs()))


@pytest.mark.parametrize("shape", ["skewed", "mostly_cold"])
def test_streaming_group_by_scalar_payload(group_by_mode: str, shape: str) -> None:
    lf = (
        SHAPE_FRAMES[shape]
        .lazy()
        .with_columns(
            one=pl.lit(1),
            tag=pl.lit("tag"),
            null_struct=pl.lit(None, dtype=pl.Struct({"a": pl.Int64})),
        )
        .group_by("k")
        .agg(
            pl.col("one").sum(),
            pl.col("one").count().alias("one_count"),
            pl.col("tag").max(),
            pl.col("null_struct").null_count().alias("null_struct_null_count"),
            pl.col("v").sum(),
        )
    )
    assert_engines_equal(lf)


def test_streaming_group_by_selective_filter(group_by_mode: str) -> None:
    n = 10_000
    rng = np.random.default_rng(11)
    df = pl.DataFrame(
        [
            pl.Series("k", rng.integers(0, 150, n), dtype=pl.Int64),
            pl.Series("x", rng.integers(0, 1_000, n), dtype=pl.Int64),
            *_payload_columns(n),
        ]
    )
    # Few rows pass the filter, so the group-by receives many small morsels.
    lf = df.lazy().filter(pl.col("x") % 50 == 0).group_by("k").agg(_aggs())
    assert_engines_equal(lf)


@pytest.mark.parametrize("n", [1, 100])
def test_streaming_group_by_head(group_by_mode: str, n: int) -> None:
    lf = SHAPE_FRAMES["unique"].lazy().group_by("k").agg(pl.len(), pl.col("v").sum())
    expected = lf.collect(engine="in-memory")
    out = lf.head(n).collect(engine="streaming")
    assert out.height == n
    assert set(out.rows()) <= set(expected.rows())


def _assert_engines_raise(lf: pl.LazyFrame, match: str) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        lf.collect(engine="streaming")
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        lf.collect(engine="in-memory")


def test_streaming_group_by_error_after_input(group_by_mode: str) -> None:
    # Numbers on the hot rows, and a few non-numbers in the last rows, whose keys
    # are cold, so the error comes after all input is received.
    df = SHAPE_FRAMES["mostly_cold"]
    n = df.height
    t = pl.Series("t", [str(j) if j < n - 3 else "x" for j in range(n)])
    gb = df.with_columns(t).lazy().group_by("k")
    parsed = pl.col("t").str.to_integer().sum().alias("t_sum")
    _assert_engines_raise(gb.agg(pl.col("t").max(), parsed), "strict integer parsing")
    # Key 0 has many rows.
    _assert_engines_raise(gb.agg(pl.col("v").fill_null(0).item()), "'item'")


GROUP_BY_LINE = re.compile(
    r"^\[group-by\]: (direct|written-out) path, (\d+) partitions, (\d+) frames in "
    r"(\d+) waves, (\d+) pre-aggregates, (\d+) estimated groups$",
    flags=re.MULTILINE,
)


def _group_by_lines(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("[group-by]")]


def _collect_verbose(lf: pl.LazyFrame, capfd: pytest.CaptureFixture[str]) -> str:
    """Compares the engines and returns what the streaming engine printed."""
    capfd.readouterr()
    out = lf.collect(engine="streaming")
    err = capfd.readouterr().err
    assert_frame_equal(out, lf.collect(engine="in-memory"), check_row_order=False)
    return err


def _path_line(err: str) -> tuple[str, int, int, int]:
    """The path, partitions, frames and waves of the only group-by."""
    lines = GROUP_BY_LINE.findall(err)
    assert len(lines) == 1, _group_by_lines(err)
    path, num_partitions, num_frames, num_waves, _, _ = lines[0]
    return path, int(num_partitions), int(num_frames), int(num_waves)


def _assert_group_by_spilled(err: str) -> None:
    spills = re.findall(
        r"^spill_stats\(group-by\): .* spill\(succ=([\d.]+)%, n=(\d+)\)",
        err,
        flags=re.MULTILINE,
    )
    assert any(float(succ) > 0 for succ, _ in spills), spills


def test_streaming_group_by_verbose_path(
    group_by_mode: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setenv("POLARS_OOC_LOG_METRICS", "1")
    lf = SHAPE_FRAMES["unique"].lazy().group_by("k").agg(pl.len(), pl.col("v").sum())
    err = _collect_verbose(lf, capfd)

    path, num_partitions, num_frames, num_waves = _path_line(err)
    if group_by_mode == "direct":
        assert (path, num_partitions) == ("direct", pl.thread_pool_size())
    else:
        assert (path, num_partitions) == ("written-out", _num_written_out_partitions())
    # Several rounds, of several frames each.
    assert 1 < num_waves < num_frames
    if group_by_mode == "spilled":
        _assert_group_by_spilled(err)


@pytest.mark.write_disk
@pytest.mark.slow
def test_streaming_group_by_direct_path_spilled(
    plmonkeypatch: PlMonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_NEW_GROUPBY", "1")
    _set_env(plmonkeypatch, SMALL_ENV)
    _set_spill_env(plmonkeypatch, tmp_path)
    # Without memory only a forced partition count keeps the direct path.
    plmonkeypatch.setenv("POLARS_GROUP_BY_NUM_PARTITIONS", str(pl.thread_pool_size()))
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setenv("POLARS_OOC_LOG_METRICS", "1")
    lf = SHAPE_FRAMES["mostly_cold"].lazy().group_by("k").agg(_fused_aggs())
    err = _collect_verbose(lf, capfd)

    path, num_partitions, _, _ = _path_line(err)
    assert (path, num_partitions) == ("direct", pl.thread_pool_size())
    _assert_group_by_spilled(err)


def test_streaming_group_by_phase_ends_during_output(
    group_by_mode: str,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    # The join stops sampling an input once it has 100 rows, which ends the
    # phase while the group-bys feeding it are sending their output.
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "100")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    n = 10_000
    rng = np.random.default_rng(5)
    left = pl.DataFrame({"k": rng.integers(0, 3_000, n), "v": rng.integers(0, 100, n)})
    right = pl.DataFrame({"k": rng.integers(0, 3_000, n), "w": rng.integers(0, 100, n)})
    lf = (
        left.lazy()
        .group_by("k")
        .agg(pl.col("v").sum(), pl.len())
        .join(right.lazy().group_by("k").agg(pl.col("w").max()), on="k")
    )
    err = _collect_verbose(lf, capfd)

    if group_by_mode != "direct":
        resumed = re.findall(
            r"^\[group-by\]: resuming (\d+) partly built or sent partitions$",
            err,
            flags=re.MULTILINE,
        )
        assert resumed, _group_by_lines(err)
        assert all(int(k) > 0 for k in resumed)
