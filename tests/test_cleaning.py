"""Checks for src.cleaning on a handmade frame. Does not read the comments CSV."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import (  # noqa: E402
    PROJECT_COLUMNS,
    clean_reviews,
    drop_duplicate_rows,
    main,
)


def handmade_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "title": "extra",
                "body": "عالی",
                "product_id": 10,
                "rate": 5.0,
                "likes": 3,
            },
            {
                "id": 2,
                "title": "extra",
                "body": None,
                "product_id": 11,
                "rate": 4.0,
                "likes": 0,
            },
            {
                "id": 3,
                "title": "extra",
                "body": "کوچک و با کیفیت",
                "product_id": 5095301,
                "rate": 2500.0,
                "likes": 1,
            },
            {
                "id": 4,
                "title": "extra",
                "body": "بد",
                "product_id": 12,
                "rate": 0.0,
                "likes": 0,
            },
            {
                "id": 5,
                "title": "extra",
                "body": "  خوب\r\nبود\t ",
                "product_id": 13,
                "rate": 3.5,
                "likes": 0,
            },
            {
                "id": 6,
                "title": "extra",
                "body": "اول",
                "product_id": 14,
                "rate": 4.0,
                "likes": 0,
            },
            {
                "id": 6,
                "title": "extra",
                "body": "دوم",
                "product_id": 15,
                "rate": 1.0,
                "likes": 0,
            },
            {
                "id": 7,
                "title": "extra",
                "body": "تکرار",
                "product_id": 16,
                "rate": 2.0,
                "likes": 9,
            },
            {
                "id": 7,
                "title": "extra",
                "body": "تکرار",
                "product_id": 16,
                "rate": 2.0,
                "likes": 8,
            },
            {
                "id": 8,
                "title": "extra",
                "body": "عالی",
                "product_id": 17,
                "rate": 5.0,
                "likes": 0,
            },
            {
                "id": 9,
                "title": "extra",
                "body": "نسبی",
                "product_id": 18,
                "rate": 4.5,
                "likes": 0,
            },
            {
                "id": 10,
                "title": "extra",
                "body": " \n\t ",
                "product_id": 19,
                "rate": 1.0,
                "likes": 0,
            },
            {
                "id": 11,
                "title": "extra",
                "body": "منفی",
                "product_id": 20,
                "rate": -0.1,
                "likes": 0,
            },
        ]
    )


class CleanReviewsTest(unittest.TestCase):
    def test_pipeline_matches_the_notebook_rules(self) -> None:
        source = handmade_frame()
        original = source.copy(deep=True)
        cleaned, summary = clean_reviews(source)

        pd.testing.assert_frame_equal(source, original)
        self.assertEqual(list(cleaned.columns), list(PROJECT_COLUMNS))
        self.assertEqual(summary.rows_before, 13)
        self.assertEqual(summary.dropped_missing_body, 1)
        self.assertEqual(summary.dropped_out_of_range_rate, 2)
        self.assertEqual(summary.dropped_empty_body, 1)
        self.assertEqual(summary.dropped_duplicate_ids, 2)
        self.assertEqual(summary.dropped_duplicate_rows, 0)
        self.assertEqual(summary.repeated_bodies_remaining, 1)
        self.assertEqual(summary.rows_after, 7)
        self.assertEqual(len(cleaned), 7)

        kept = cleaned.set_index("id")
        self.assertEqual(set(kept.index), {1, 4, 5, 6, 7, 8, 9})
        self.assertEqual(kept.loc[4, "rate"], 0.0)
        self.assertEqual(kept.loc[5, "body"], "خوب بود")
        self.assertEqual(kept.loc[5, "rate"], 3.5)
        self.assertEqual(kept.loc[6, "body"], "اول")
        self.assertEqual(kept.loc[7, "body"], "تکرار")
        self.assertEqual(kept.loc[9, "rate"], 4.5)
        self.assertEqual(list(cleaned.loc[cleaned["body"] == "عالی", "id"]), [1, 8])
        self.assertTrue(((cleaned["rate"] >= 0) & (cleaned["rate"] <= 5)).all())

    def test_full_row_duplicate_drop_keeps_one_copy(self) -> None:
        frame = pd.DataFrame(
            [
                {"id": 7, "body": "تکرار", "product_id": 16, "rate": 2.0},
                {"id": 7, "body": "تکرار", "product_id": 16, "rate": 2.0},
                {"id": 8, "body": "عالی", "product_id": 17, "rate": 5.0},
            ]
        )
        kept, dropped = drop_duplicate_rows(frame)
        self.assertEqual(dropped, 1)
        self.assertEqual(len(kept), 2)
        self.assertEqual(list(kept["id"]), [7, 8])


class CleanReviewsCliTest(unittest.TestCase):
    def _write_sample(self, directory: Path) -> Path:
        source = directory / "comments.csv"
        handmade_frame().to_csv(source, index=False)
        return source

    def test_cli_arguments_choose_the_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = self._write_sample(folder)
            target = folder / "out" / "cleaned.csv"
            main(["--input", str(source), "--output", str(target)])
            cleaned = pd.read_csv(target)
            self.assertEqual(len(cleaned), 7)
            self.assertEqual(list(cleaned.columns), list(PROJECT_COLUMNS))
            self.assertEqual(cleaned.loc[cleaned["id"] == 5, "body"].item(), "خوب بود")

    def test_env_file_values_are_the_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = self._write_sample(folder)
            target = folder / "from-env.csv"
            previous = {
                "REVIEWS_INPUT_CSV": os.environ.get("REVIEWS_INPUT_CSV"),
                "REVIEWS_OUTPUT_CSV": os.environ.get("REVIEWS_OUTPUT_CSV"),
            }
            os.environ["REVIEWS_INPUT_CSV"] = str(source)
            os.environ["REVIEWS_OUTPUT_CSV"] = str(target)
            try:
                main([])
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            cleaned = pd.read_csv(target)
            self.assertEqual(len(cleaned), 7)


if __name__ == "__main__":
    unittest.main()
