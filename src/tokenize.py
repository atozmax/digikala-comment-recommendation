"""Tokenize a preprocessed review body with the ParsBERT tokenizer.

Run :mod:`src.preprocess` first. This module does not rewrite letters
or digits. It loads ``HooshvareLab/bert-fa-base-uncased`` unless another
checkpoint name is passed, truncates at 512 tokens, and writes one
parquet row per review: ``id``, ``product_id``, ``recommendation_status``,
``rate``, ``input_ids``, and ``truncated``. ``product_id`` stays so a
later split can keep every review of one product in the same part.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.cleaning import NA_VALUES, format_count, load_project_env

DEFAULT_MODEL = "HooshvareLab/bert-fa-base-uncased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 2048
DEFAULT_INPUT_CSV = "data/processed/digikala-comments-preprocessed.csv"
DEFAULT_OUTPUT_PARQUET = "data/processed/digikala-comments-tokenized.parquet"
INPUT_CSV_ENV = "TOKENIZE_INPUT_CSV"
OUTPUT_PARQUET_ENV = "TOKENIZE_OUTPUT_PARQUET"
MODEL_ENV = "TOKENIZER_MODEL"
MAX_LENGTH_ENV = "TOKENIZER_MAX_LENGTH"

SCHEMA = pa.schema(
    [
        ("id", pa.int64()),
        ("product_id", pa.int64()),
        ("recommendation_status", pa.string()),
        ("rate", pa.float64()),
        ("input_ids", pa.list_(pa.int32())),
        ("truncated", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class TokenizeSummary:
    """Row counts for one call of :func:`tokenize_reviews_file`."""

    rows: int
    truncated: int
    max_length: int


def load_tokenizer(model_name: str):
    """Load the Hugging Face tokenizer that ships with ``model_name``."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def encode_bodies(texts: list[str], tokenizer, max_length: int) -> tuple[list[list[int]], list[bool]]:
    """Encode bodies without padding. A row is truncated when it exceeds ``max_length``.

    The tokenizer adds its own special tokens. Ids are then cut to
    ``max_length``. Emoji and Latin letters are already in the text;
    this function does not strip them.
    """
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    input_ids: list[list[int]] = []
    truncated: list[bool] = []
    for ids in encoded["input_ids"]:
        is_long = len(ids) > max_length
        input_ids.append(ids[:max_length])
        truncated.append(is_long)
    return input_ids, truncated


def tokenize_reviews_file(
    input_csv: str | Path,
    output_parquet: str | Path,
    tokenizer,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TokenizeSummary:
    """Read a preprocessed CSV in batches and write token ids to parquet."""
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    n_truncated = 0
    writer: pq.ParquetWriter | None = None
    batches = (
        pl.scan_csv(
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
        .select("id", "product_id", "recommendation_status", "rate", "body")
        .collect_batches(chunk_size=batch_size)
    )
    try:
        for chunk in batches:
            bodies = chunk.get_column("body").fill_null("").to_list()
            input_ids, truncated = encode_bodies(bodies, tokenizer, max_length)
            table = pa.Table.from_pydict(
                {
                    "id": chunk.get_column("id").to_list(),
                    "product_id": chunk.get_column("product_id").to_list(),
                    "recommendation_status": chunk.get_column("recommendation_status").to_list(),
                    "rate": chunk.get_column("rate").to_list(),
                    "input_ids": input_ids,
                    "truncated": truncated,
                },
                schema=SCHEMA,
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, SCHEMA)
            writer.write_table(table)
            rows += len(chunk)
            n_truncated += sum(truncated)
            print(f"\rtokenized: {rows}", end="", flush=True)
    finally:
        if writer is not None:
            writer.close()
        elif rows == 0:
            pq.write_table(
                pa.table(
                    {
                        "id": [],
                        "product_id": [],
                        "recommendation_status": [],
                        "rate": [],
                        "input_ids": [],
                        "truncated": [],
                    },
                    schema=SCHEMA,
                ),
                output_path,
            )
    if rows:
        print()
    return TokenizeSummary(rows=rows, truncated=n_truncated, max_length=max_length)


def build_parser() -> argparse.ArgumentParser:
    """Parser for the tokenize CLI. Call :func:`load_project_env` first."""
    parser = argparse.ArgumentParser(
        prog="python -m src tokenize",
        description="Tokenize preprocessed Digikala review bodies and write parquet.",
    )
    parser.add_argument(
        "--input",
        default=os.getenv(INPUT_CSV_ENV, DEFAULT_INPUT_CSV),
        help=(
            f"Preprocessed comments CSV. Defaults to ${INPUT_CSV_ENV} "
            f"or {DEFAULT_INPUT_CSV}."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.getenv(OUTPUT_PARQUET_ENV, DEFAULT_OUTPUT_PARQUET),
        help=(
            f"Token parquet to write. Defaults to ${OUTPUT_PARQUET_ENV} "
            f"or {DEFAULT_OUTPUT_PARQUET}."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv(MODEL_ENV, DEFAULT_MODEL),
        help=f"Tokenizer checkpoint. Defaults to ${MODEL_ENV} or {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv(MAX_LENGTH_ENV, str(DEFAULT_MAX_LENGTH))),
        help=f"Maximum tokens, including special tokens. Defaults to {DEFAULT_MAX_LENGTH}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Reviews encoded at once. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the tokenize CLI. Paths come from arguments, then ``.env``, then the defaults."""
    load_project_env()
    args = build_parser().parse_args(argv)
    print(f"loading: {args.input}", flush=True)
    print(f"tokenizer: {args.model}")
    tokenizer = load_tokenizer(args.model)
    summary = tokenize_reviews_file(
        args.input,
        args.output,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(f"output: {args.output}")
    print(f"max length: {summary.max_length}")
    print(f"rows: {format_count(summary.rows, summary.rows)}")
    print(f"truncated: {format_count(summary.truncated, summary.rows)}")


if __name__ == "__main__":
    main()
