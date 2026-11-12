"""Character rules for the server experiment. Does not read the comments CSV."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.text_v2 import (  # noqa: E402
    apply_digits,
    apply_emoji,
    apply_letters,
    confusion_metrics,
)


class LetterStageTest(unittest.TestCase):
    def test_arabic_letters_keep_the_surface_word(self) -> None:
        self.assertEqual(apply_letters("كيفيّة"), "کیفیه")
        self.assertEqual(apply_letters("کتاب‌ها"), "کتاب‌ها")

    def test_half_space_url_and_run(self) -> None:
        self.assertEqual(apply_letters("می\u200c\u200cشود"), "می‌شود")
        self.assertEqual(apply_letters("\u200cمی شود\u200c"), "می شود")
        self.assertEqual(apply_letters("ببین http://example.com/a تمام"), "ببین تمام")
        self.assertEqual(apply_letters("عاااالی"), "عاالی")

    def test_latin_and_emoji_stay_until_later_stages(self) -> None:
        self.assertEqual(apply_letters("عالی iPhone 😀"), "عالی iPhone 😀")


class DigitAndEmojiTest(unittest.TestCase):
    def test_pure_numbers_go_and_model_codes_stay(self) -> None:
        folded = apply_digits(apply_letters("قیمت 128 و A54 و ۱۲۸GB"))
        self.assertEqual(folded, "قیمت و A۵۴ و ۱۲۸GB")

    def test_emoji_only_becomes_empty_and_english_stays(self) -> None:
        self.assertEqual(apply_emoji("عالی 😀 iPhone"), "عالی iPhone")
        self.assertEqual(apply_emoji("😀"), "")


class ConfusionMetricsTest(unittest.TestCase):
    def test_perfect_split_records_macro_and_micro(self) -> None:
        matrix = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        report = confusion_metrics(matrix, ("recommended", "not_recommended", "no_idea"))
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["f1_macro"], 1.0)
        self.assertEqual(report["f1_micro"], 1.0)
        self.assertEqual(report["recommended_support"], 2)
        self.assertEqual(report["no_idea_support"], 1)
