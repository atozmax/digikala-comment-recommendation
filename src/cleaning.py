"""Clean the four-column Digikala review frame.

The steps follow ``notebooks/01-descriptive-analysis.ipynb``: keep
``id``, ``body``, ``product_id``, and ``rate``; drop missing bodies;
keep rates in ``[0, 5]``; normalize whitespace; drop empty bodies;
keep one row per ``id``; drop any remaining full-row duplicates.
Rows that share a body and nothing else are counted and kept.

This module does not read the comments CSV. Pass a DataFrame in.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

PROJECT_COLUMNS: tuple[str, ...] = ("id", "body", "product_id", "rate")

# Paths are relative to the repository root, matching the notebook's
# ../data/... reads and writes from notebooks/.
DEFAULT_INPUT_CSV = "data/raw/digikala-comments.csv"
DEFAULT_OUTPUT_CSV = "data/cleaned/digikala-comments-cleaned.csv"
INPUT_CSV_ENV = "REVIEWS_INPUT_CSV"
OUTPUT_CSV_ENV = "REVIEWS_OUTPUT_CSV"

# Same missing-value tokens as the descriptive-analysis notebook.
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
    dropped_out_of_range_rate: int
    dropped_empty_body: int
    dropped_duplicate_ids: int
    dropped_duplicate_rows: int
    repeated_bodies_remaining: int


def select_project_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with only ``id``, ``body``, ``product_id``, and ``rate``."""
    return df.loc[:, list(PROJECT_COLUMNS)].copy()


def drop_missing_bodies(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows whose ``body`` is missing. Return the frame and the count."""
    kept = df.dropna(subset=["body"]).reset_index(drop=True)
    return kept, len(df) - len(kept)


def filter_rates(
    df: pd.DataFrame,
    *,
    minimum: float = 0.0,
    maximum: float = 5.0,
) -> tuple[pd.DataFrame, int]:
    """Keep rates inside ``[minimum, maximum]``, including both ends.

    Rate 0 and fractional values inside the interval stay. A rate such
    as 2500 is dropped. Missing rates fail the comparison and are dropped.
    """
    mask = (df["rate"] >= minimum) & (df["rate"] <= maximum)
    kept = df.loc[mask].copy()
    return kept, len(df) - len(kept)


def normalize_bodies(df: pd.DataFrame) -> pd.DataFrame:
    """Replace CR, LF, and tab with a space, collapse whitespace, and strip.

    Returns a copy. The caller's frame is left unchanged.
    """
    out = df.copy()
    out["body"] = (
        out["body"]
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return out


def drop_empty_bodies(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows whose ``body`` is an empty string after normalization."""
    kept = df.loc[df["body"].ne("")].copy()
    return kept, len(df) - len(kept)


def drop_duplicate_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop surplus rows that repeat an ``id``. Keep the first row of each id.

    The count is only the surplus (``id.duplicated().sum()``). It is not
    the number of rows that sit in a duplicate group.
    """
    dropped = int(df["id"].duplicated().sum())
    kept = df.drop_duplicates(subset=["id"], keep="first")
    return kept, dropped


def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop extra copies of a fully identical row. Keep the first copy."""
    dropped = int(df.duplicated().sum())
    kept = df.drop_duplicates(keep="first")
    return kept, dropped


def count_repeated_bodies(df: pd.DataFrame) -> int:
    """Count extra copies of a ``body``. Do not drop those rows.

    Short texts such as «عالی» are real reviews of different products.
    The count is ``keep='first'`` extras, not every row that shares a body.
    """
    return int(df["body"].duplicated().sum())


def clean_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningSummary]:
    """Apply the notebook cleaning steps and return the frame plus a summary.

    Order: project columns, missing bodies, rates in ``[0, 5]``, body
    whitespace, empty bodies, duplicate ids, full-row duplicates.
    Repeated bodies are counted in the summary and kept.

    ``rows_before`` is the length of ``df`` before the column cut.
    The input frame is not modified.
    """
    rows_before = len(df)
    frame = select_project_columns(df)
    frame, dropped_missing_body = drop_missing_bodies(frame)
    frame, dropped_out_of_range_rate = filter_rates(frame)
    frame = normalize_bodies(frame)
    frame, dropped_empty_body = drop_empty_bodies(frame)
    frame, dropped_duplicate_ids = drop_duplicate_ids(frame)
    frame, dropped_duplicate_rows = drop_duplicate_rows(frame)
    repeated_bodies_remaining = count_repeated_bodies(frame)
    frame = frame.reset_index(drop=True)
    summary = CleaningSummary(
        rows_before=rows_before,
        rows_after=len(frame),
        dropped_missing_body=dropped_missing_body,
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
    frame = pd.read_csv(input_csv, na_values=NA_VALUES)
    cleaned, summary = clean_reviews(frame)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Parser for the cleaning CLI.

    Call :func:`load_project_env` first so ``.env`` can fill the defaults.
    An explicit ``--input`` or ``--output`` replaces both the file and
    the hardcoded path.
    """
    parser = argparse.ArgumentParser(
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


def main(argv: list[str] | None = None) -> None:
    """Run the cleaning CLI. Paths come from arguments, then ``.env``, then the defaults."""
    load_project_env()
    args = build_parser().parse_args(argv)
    summary = clean_reviews_file(args.input, args.output)
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"rows before: {summary.rows_before}")
    print(f"dropped missing body: {summary.dropped_missing_body}")
    print(f"dropped out-of-range rate: {summary.dropped_out_of_range_rate}")
    print(f"dropped empty body: {summary.dropped_empty_body}")
    print(f"dropped duplicate ids: {summary.dropped_duplicate_ids}")
    print(f"dropped duplicate rows: {summary.dropped_duplicate_rows}")
    print(f"repeated bodies remaining: {summary.repeated_bodies_remaining}")
    print(f"rows after: {summary.rows_after}")


if __name__ == "__main__":
    main()
