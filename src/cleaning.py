"""Clean the Digikala review frame.

The steps follow ``notebooks/01-data-cleaning.ipynb``: keep
``id``, ``body``, ``product_id``, ``recommendation_status``, and
``rate``; drop a missing body or recommendation; keep rates in
``[0, 5]``; normalize whitespace; drop empty bodies; keep one row
per ``id``; drop any remaining full-row duplicates. Rows that share
a body and nothing else are counted and kept.

This module reads and writes with Polars. Pass a DataFrame in.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from dotenv import find_dotenv, load_dotenv

PROJECT_COLUMNS: tuple[str, ...] = (
    "id",
    "body",
    "product_id",
    "recommendation_status",
    "rate",
)

# Paths are relative to the repository root, matching the notebook's
# ../data/... reads and writes from notebooks/.
DEFAULT_INPUT_CSV = "data/raw/digikala-comments.csv"
DEFAULT_OUTPUT_CSV = "data/cleaned/digikala-comments-cleaned.csv"
INPUT_CSV_ENV = "REVIEWS_INPUT_CSV"
OUTPUT_CSV_ENV = "REVIEWS_OUTPUT_CSV"

# Same missing-value tokens as notebooks/01-data-cleaning.ipynb.
NA_VALUES: list[str] = [
    "",
    "nan",
    "NaN",
    "NAN",
    "null",
    "NULL",
    "None",
    "none",
    "-",
]


@dataclass(frozen=True)
class CleaningSummary:
    """Row counts for one call of :func:`clean_reviews`.

    Drop counts are rows removed at that step, not the size of a
    ``keep=False`` duplicate group. ``repeated_bodies_remaining`` is
    ``body.duplicated().sum()`` after the drops: extra copies only.
    """

    rows_before: int
    rows_after: int
    dropped_missing_body: int
    dropped_missing_recommendation: int
    dropped_out_of_range_rate: int
    dropped_empty_body: int
    dropped_duplicate_ids: int
    dropped_duplicate_rows: int
    repeated_bodies_remaining: int


def select_project_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Return a frame with ``id``, ``body``, ``product_id``, ``recommendation_status``, and ``rate``."""
    return df.select(list(PROJECT_COLUMNS))


def drop_missing_bodies(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop rows whose ``body`` is missing. Return the frame and the count."""
    kept = df.drop_nulls(subset=["body"])
    return kept, df.height - kept.height


def drop_missing_recommendations(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop rows whose ``recommendation_status`` is missing."""
    kept = df.drop_nulls(subset=["recommendation_status"])
    return kept, df.height - kept.height


def filter_rates(
    df: pl.DataFrame,
    *,
    minimum: float = 0.0,
    maximum: float = 5.0,
) -> tuple[pl.DataFrame, int]:
    """Keep rates inside ``[minimum, maximum]``, including both ends.

    Rate 0 and fractional values inside the interval stay. A rate such
    as 2500 is dropped. Missing rates fail the comparison and are dropped.
    """
    kept = df.filter((pl.col("rate") >= minimum) & (pl.col("rate") <= maximum))
    return kept, df.height - kept.height


def normalize_bodies(df: pl.DataFrame) -> pl.DataFrame:
    """Replace CR, LF, and tab with a space, collapse whitespace, and strip.

    Returns a new frame. The caller's frame is left unchanged.
    """
    return df.with_columns(
        pl.col("body")
        .str.replace_all(r"[\r\n\t]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def drop_empty_bodies(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop rows whose ``body`` is an empty string after normalization."""
    kept = df.filter(pl.col("body") != "")
    return kept, df.height - kept.height


def drop_duplicate_ids(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop surplus rows that repeat an ``id``. Keep the first row of each id.

    The count is only the surplus. It is not the number of rows that sit
    in a duplicate group.
    """
    kept = df.unique(subset=["id"], keep="first", maintain_order=True)
    return kept, df.height - kept.height


def drop_duplicate_rows(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop extra copies of a fully identical row. Keep the first copy."""
    kept = df.unique(keep="first", maintain_order=True)
    return kept, df.height - kept.height


def count_repeated_bodies(df: pl.DataFrame) -> int:
    """Count extra copies of a ``body``. Do not drop those rows.

    Short texts such as «عالی» are real reviews of different products.
    The count is the extra copies, not every row that shares a body.
    """
    return df.height - df.get_column("body").n_unique()


def clean_reviews(df: pl.DataFrame) -> tuple[pl.DataFrame, CleaningSummary]:
    """Apply the notebook cleaning steps and return the frame plus a summary.

    Order: project columns, missing bodies, missing recommendations,
    rates in ``[0, 5]``, body whitespace, empty bodies, duplicate ids,
    full-row duplicates. Repeated bodies are counted in the summary and kept.

    ``rows_before`` is the height of ``df`` before the column cut.
    The input frame is not modified.
    """
    rows_before = df.height
    frame = select_project_columns(df)
    frame, dropped_missing_body = drop_missing_bodies(frame)
    frame, dropped_missing_recommendation = drop_missing_recommendations(frame)
    frame, dropped_out_of_range_rate = filter_rates(frame)
    frame = normalize_bodies(frame)
    frame, dropped_empty_body = drop_empty_bodies(frame)
    frame, dropped_duplicate_ids = drop_duplicate_ids(frame)
    frame, dropped_duplicate_rows = drop_duplicate_rows(frame)
    repeated_bodies_remaining = count_repeated_bodies(frame)
    summary = CleaningSummary(
        rows_before=rows_before,
        rows_after=frame.height,
        dropped_missing_body=dropped_missing_body,
        dropped_missing_recommendation=dropped_missing_recommendation,
        dropped_out_of_range_rate=dropped_out_of_range_rate,
        dropped_empty_body=dropped_empty_body,
        dropped_duplicate_ids=dropped_duplicate_ids,
        dropped_duplicate_rows=dropped_duplicate_rows,
        repeated_bodies_remaining=repeated_bodies_remaining,
    )
    return frame, summary


def load_project_env() -> None:
    """Load ``.env`` from the working directory or a parent directory.

    Existing environment variables stay in place. A value set in the
    shell or on the command line is not replaced by the file.
    """
    env_file = find_dotenv(usecwd=True)
    load_dotenv(env_file or None)


def clean_reviews_file(input_csv: str | Path, output_csv: str | Path) -> CleaningSummary:
    """Read a comments CSV, clean it, and write the result.

    ``input_csv`` is read with the notebook's missing-value tokens.
    Parent directories of ``output_csv`` are created when missing.
    """
    frame = pl.read_csv(
        input_csv,
        null_values=NA_VALUES,
        schema_overrides={
            "id": pl.Int64,
            "body": pl.String,
            "product_id": pl.Int64,
            "recommendation_status": pl.String,
            "rate": pl.Float64,
        },
    )
    cleaned, summary = clean_reviews(frame)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_csv(output_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Parser for the cleaning CLI.

    Call :func:`load_project_env` first so ``.env`` can fill the defaults.
    An explicit ``--input`` or ``--output`` replaces both the file and
    the hardcoded path.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src cleaning",
        description="Clean Digikala review comments and write a CSV.",
    )
    parser.add_argument(
        "--input",
        default=os.getenv(INPUT_CSV_ENV, DEFAULT_INPUT_CSV),
        help=(
            f"Input comments CSV. Defaults to ${INPUT_CSV_ENV} "
            f"or {DEFAULT_INPUT_CSV}."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.getenv(OUTPUT_CSV_ENV, DEFAULT_OUTPUT_CSV),
        help=(
            f"Cleaned CSV to write. Defaults to ${OUTPUT_CSV_ENV} "
            f"or {DEFAULT_OUTPUT_CSV}."
        ),
    )
    return parser


def format_count(count: int, total: int) -> str:
    """Format a row count with its share of ``total``, as ``12 (34.56%)``.

    ``total`` is the number of rows loaded from the CSV. An empty file
    prints ``0.00%`` instead of dividing by zero.
    """
    share = 0.0 if total == 0 else 100.0 * count / total
    return f"{count} ({share:.2f}%)"


def main(argv: list[str] | None = None) -> None:
    """Run the cleaning CLI. Paths come from arguments, then ``.env``, then the defaults.

    The load is printed before the CSV is read, then again with the
    loaded row count. Every later count includes its percent of that
    loaded total.
    """
    load_project_env()
    args = build_parser().parse_args(argv)
    print(f"loading: {args.input}", flush=True)
    summary = clean_reviews_file(args.input, args.output)
    loaded = summary.rows_before
    print(f"loaded: {format_count(loaded, loaded)}")
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"rows before: {format_count(summary.rows_before, loaded)}")
    print(f"dropped missing body: {format_count(summary.dropped_missing_body, loaded)}")
    print(
        "dropped missing recommendation: "
        f"{format_count(summary.dropped_missing_recommendation, loaded)}"
    )
    print(
        "dropped out-of-range rate: "
        f"{format_count(summary.dropped_out_of_range_rate, loaded)}"
    )
    print(f"dropped empty body: {format_count(summary.dropped_empty_body, loaded)}")
    print(f"dropped duplicate ids: {format_count(summary.dropped_duplicate_ids, loaded)}")
    print(f"dropped duplicate rows: {format_count(summary.dropped_duplicate_rows, loaded)}")
    print(
        "repeated bodies remaining: "
        f"{format_count(summary.repeated_bodies_remaining, loaded)}"
    )
    print(f"rows after: {format_count(summary.rows_after, loaded)}")


if __name__ == "__main__":
    main()
