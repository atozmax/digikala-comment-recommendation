"""Rating regression helpers. Does not load the comments file or a model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.text_v2 import apply_digits, apply_emoji, apply_letters  # noqa: E402
from src.experiments.tookabert_rate_regression import (  # noqa: E402
    accumulate_score,
    add_score,
    empty_score,
    prepare_body,
    rating_band,
    regression_metrics,
)


class BandTest(unittest.TestCase):
    def test_edges_match_the_report(self) -> None:
        self.assertEqual(rating_band(0), "low")
        self.assertEqual(rating_band(2.49), "low")
        self.assertEqual(rating_band(2.5), "medium")
        self.assertEqual(rating_band(3.9), "medium")
        self.assertEqual(rating_band(4), "high")
        self.assertEqual(rating_band(5), "high")


class PipelineTest(unittest.TestCase):
    def test_prepare_body_is_the_three_stages_in_order(self) -> None:
        text = "عاااالی 128 iPhone 😀"
        self.assertEqual(prepare_body(text), apply_emoji(apply_digits(apply_letters(text))))
        self.assertNotIn("😀", prepare_body(text))


class MetricsTest(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        totals = empty_score()
        accumulate_score(totals, 5.0, 5.0)
        accumulate_score(totals, 1.0, 1.0)
        metrics = regression_metrics(totals)
        self.assertEqual(metrics["rows"], 2)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)

    def test_mae_rmse_and_clip(self) -> None:
        totals = empty_score()
        accumulate_score(totals, 5.0, 6.0)
        accumulate_score(totals, 0.0, -1.0)
        metrics = regression_metrics(totals)
        self.assertEqual(metrics["mae"], 1.0)
        self.assertEqual(metrics["rmse"], 1.0)
        self.assertEqual(metrics["clipped_mae"], 0.0)
        self.assertEqual(metrics["high_rows"], 1)
        self.assertEqual(metrics["low_rows"], 1)
        self.assertEqual(metrics["medium_rows"], 0)

    def test_shards_add(self) -> None:
        left = empty_score()
        right = empty_score()
        accumulate_score(left, 4.0, 3.0)
        accumulate_score(right, 2.0, 2.5)
        metrics = regression_metrics(add_score(left, right))
        self.assertEqual(metrics["rows"], 2)
        self.assertAlmostEqual(metrics["mae"], 0.75)

    def test_constant_ratings_have_zero_r2(self) -> None:
        totals = empty_score()
        accumulate_score(totals, 4.0, 3.0)
        accumulate_score(totals, 4.0, 5.0)
        self.assertEqual(regression_metrics(totals)["r2"], 0.0)


if __name__ == "__main__":
    unittest.main()
