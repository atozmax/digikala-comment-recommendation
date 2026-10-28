"""Checks for the character pass and the tokenizer wrapper. Does not read the comments CSV."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocess import preprocess_reviews, preprocess_text  # noqa: E402
from src.tokenize import encode_bodies  # noqa: E402


class PreprocessTextTest(unittest.TestCase):
    def test_arabic_letters_and_digits_become_persian(self) -> None:
        self.assertEqual(preprocess_text("كيفيّة ١٢٣ و 45"), "کیفیه ۱۲۳ و ۴۵")
        self.assertEqual(preprocess_text("آبي"), "آبی")
        self.assertIn("😀", preprocess_text("عالی 😀"))

    def test_half_space_and_whitespace(self) -> None:
        self.assertEqual(preprocess_text("می\u200c\u200cشود"), "می‌شود")
        self.assertEqual(preprocess_text("\u200cمی شود\u200c"), "می شود")
        self.assertEqual(preprocess_text("خط\nبعد\tتمام"), "خط بعد تمام")

    def test_url_and_character_run(self) -> None:
        self.assertEqual(preprocess_text("ببین http://example.com/a تمام"), "ببین تمام")
        self.assertEqual(preprocess_text("عاااالی!!!!"), "عاالی!!")

    def test_frame_drops_a_body_that_becomes_empty(self) -> None:
        frame = pl.DataFrame(
            [
                {"id": 1, "body": "ك", "product_id": 10, "recommendation_status": "recommended", "rate": 5.0},
                {"id": 2, "body": "http://example.com", "product_id": 11, "recommendation_status": "not_recommended", "rate": 1.0},
            ]
        )
        kept, summary = preprocess_reviews(frame)
        self.assertEqual(kept.get_column("body").to_list(), ["ک"])
        self.assertEqual(summary.rows_before, 2)
        self.assertEqual(summary.dropped_empty_body, 1)
        self.assertEqual(summary.rows_after, 1)
        self.assertEqual(summary.rows_changed, 2)


class _FakeTokenizer:
    def __call__(self, texts, add_special_tokens, truncation, padding):
        del add_special_tokens, truncation, padding
        ids = []
        for text in texts:
            ids.append([1, *([7] * len(text)), 2])
        return {"input_ids": ids}


class EncodeBodiesTest(unittest.TestCase):
    def test_long_body_is_cut_at_max_length(self) -> None:
        ids, truncated = encode_bodies(["abcd", "ab"], _FakeTokenizer(), max_length=4)
        self.assertEqual(ids, [[1, 7, 7, 7], [1, 7, 7, 2]])
        self.assertEqual(truncated, [True, False])


class PreprocessCliTest(unittest.TestCase):
    def test_cli_writes_the_preprocessed_csv(self) -> None:
        from src.preprocess import main

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "cleaned.csv"
            target = folder / "out.csv"
            pl.DataFrame(
                [{"id": 1, "body": "كي 5", "product_id": 9, "recommendation_status": "recommended", "rate": 4.0}]
            ).write_csv(source)
            main(["--input", str(source), "--output", str(target)])
            written = pl.read_csv(target)
            self.assertEqual(written.get_column("body")[0], "کی ۵")


if __name__ == "__main__":
    unittest.main()
