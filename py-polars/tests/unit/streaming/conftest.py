from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.fixture
def io_files_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files"


def assert_engines_equal(lf: pl.LazyFrame, *, check_row_order: bool = False) -> None:
    assert_frame_equal(
        lf.collect(engine="streaming"),
        lf.collect(engine="in-memory"),
        check_row_order=check_row_order,
    )
