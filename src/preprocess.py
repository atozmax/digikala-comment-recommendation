"""Rewrite a cleaned review body into the surface form the tokenizer expects.

The input is the cleaned CSV, not the raw comments file. The written
file keeps ``id``, ``body``, ``product_id``, ``recommendation_status``,
and ``rate``.
Cleaning has already dropped empty bodies, kept rates in ``[0, 5]``,
and collapsed line breaks. This pass does the character work measured
in ``notebooks/02-descriptive-analysis.ipynb``:

- NFKC, then Arabic letters onto Persian letters
- ASCII and Arabic-Indic digits onto Persian digits
- one interior half-space; drop a repeated or edge half-space
- drop diacritics, tatweel, and invisible marks other than that half-space
- replace a URL or email with a space
- cap a run of three identical characters at two
- collapse whitespace again

Emoji stay. Latin letters stay. Stemming and lemmatization stay out:
a Hazm lemma such as ``رفت#رو`` is not the surface word in the
tokenizer vocabulary, and it erases polarity.
"""

from __future__ import annotations

import argparse
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.cleaning import (
    NA_VALUES,
    format_count,
    load_project_env,
)

SAVED_COLUMNS: tuple[str, ...] = (
    "id",
    "body",
    "product_id",
    "recommendation_status",
    "rate",
)

DEFAULT_INPUT_CSV = "data/cleaned/digikala-comments-cleaned.csv"
DEFAULT_OUTPUT_CSV = "data/processed/digikala-comments-preprocessed.csv"
INPUT_CSV_ENV = "PREPROCESS_INPUT_CSV"
OUTPUT_CSV_ENV = "PREPROCESS_OUTPUT_CSV"

_ARABIC_LETTERS = str.maketrans(
    {
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ة": "ه",
        "ۀ": "ه",
        "ھ": "ه",
        "ؤ": "و",
    }
)

_DIGITS = str.maketrans(
    {chr(ord("0") + i): chr(0x06F0 + i) for i in range(10)}
    | {chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)}
)

_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_INVISIBLE = re.compile(r"[\u200B\u200D\uFEFF\u200E\u200F\u202A-\u202E\u2066-\u2069]")
_REPEATED_ZWNJ = re.compile(r"\u200C{2,}")
_EDGE_ZWNJ = re.compile(r"(?<!\S)\u200C|\u200C(?!\S)")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CHAR_RUN = re.compile(r"(.)\1{2,}")
_LINE_BREAKS = re.compile(r"[\r\n\t]+")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessSummary:
    """Row counts for one call of :func:`preprocess_reviews`."""

    rows_before: int
    rows_after: int
    rows_changed: int
    dropped_empty_body: int


def preprocess_text(text: str) -> str:
    """Return one review body after the character pass. Emoji and Latin stay."""
    if not isinstance(text, str):
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.translate(_ARABIC_LETTERS)
    out = _DIACRITICS.sub("", out)
    out = _INVISIBLE.sub("", out)
    out = _REPEATED_ZWNJ.sub("\u200C", out)
    out = _EDGE_ZWNJ.sub("", out)
    out = _URL.sub(" ", out)
    out = _EMAIL.sub(" ", out)
    out = out.translate(_DIGITS)
    out = _CHAR_RUN.sub(r"\1\1", out)
    out = _LINE_BREAKS.sub(" ", out)
    out = _SPACES.sub(" ", out).strip()
    out = _EDGE_ZWNJ.sub("", out)
    return out.strip()


def _rewrite_batch(batch: pl.Series) -> pl.Series:
    return pl.Series(
        batch.name,
        [preprocess_text(value) for value in batch.to_list()],
        dtype=pl.String,
    )


def preprocess_reviews(df: pl.DataFrame) -> tuple[pl.DataFrame, PreprocessSummary]:
    """Rewrite ``body`` on a cleaned frame and drop a body that became empty.

    The frame must already have ``id``, ``body``, ``product_id``,
    ``recommendation_status``, and ``rate``. Other columns are ignored.
    The input frame is not modified.
    ``rows_changed`` counts bodies whose text differs after the pass,
    including rows that then become empty and are dropped.
    """
    frame = df.select(list(SAVED_COLUMNS)).with_columns(pl.col("body").fill_null(""))
    original = frame.get_column("body")
    rewritten = frame.select(
        pl.col("body").map_batches(_rewrite_batch, return_dtype=pl.String)
    ).get_column("body")
    changed = int((rewritten != original).sum())
    kept = frame.with_columns(rewritten.alias("body")).filter(pl.col("body") != "")
    summary = PreprocessSummary(
        rows_before=df.height,
        rows_after=kept.height,
        rows_changed=changed,
        dropped_empty_body=frame.height - kept.height,
    )
    return kept, summary


def preprocess_reviews_file(input_csv: str | Path, output_csv: str | Path) -> PreprocessSummary:
    """Read a cleaned comments CSV, preprocess ``body``, and write the result."""
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
    rewritten, summary = preprocess_reviews(frame)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rewritten.select(
        "id",
        "body",
        "product_id",
        "recommendation_status",
        "rate",
    ).write_csv(output_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Parser for the preprocess CLI. Call :func:`load_project_env` first."""
    parser = argparse.ArgumentParser(
        prog="python -m src preprocess",
        description="Preprocess cleaned Digikala review bodies and write a CSV.",
    )
    parser.add_argument(
        "--input",
        default=os.getenv(INPUT_CSV_ENV, DEFAULT_INPUT_CSV),
        help=(
            f"Cleaned comments CSV. Defaults to ${INPUT_CSV_ENV} "
            f"or {DEFAULT_INPUT_CSV}."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.getenv(OUTPUT_CSV_ENV, DEFAULT_OUTPUT_CSV),
        help=(
            f"Preprocessed CSV to write. Defaults to ${OUTPUT_CSV_ENV} "
            f"or {DEFAULT_OUTPUT_CSV}."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the preprocess CLI. Paths come from arguments, then ``.env``, then the defaults."""
    load_project_env()
    args = build_parser().parse_args(argv)
    print(f"loading: {args.input}", flush=True)
    summary = preprocess_reviews_file(args.input, args.output)
    loaded = summary.rows_before
    print(f"loaded: {format_count(loaded, loaded)}")
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"rows changed: {format_count(summary.rows_changed, loaded)}")
    print(f"dropped empty body: {format_count(summary.dropped_empty_body, loaded)}")
    print(f"rows after: {format_count(summary.rows_after, loaded)}")


if __name__ == "__main__":
    main()
