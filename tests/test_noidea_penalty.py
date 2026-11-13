"""The no_idea penalty uses only true no_idea rows."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.tookabert_noidea_penalty import no_idea_wrong_mass  # noqa: E402


class NoIdeaPenaltyTest(unittest.TestCase):
    def test_other_labels_do_not_add_penalty(self) -> None:
        probabilities = [[0.1, 0.2, 0.7], [0.0, 0.0, 1.0]]
        self.assertEqual(no_idea_wrong_mass(probabilities, [0, 2]), 0.0)

    def test_mean_over_true_no_idea_rows(self) -> None:
        probabilities = [[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.9, 0.1, 0.0]]
        self.assertAlmostEqual(no_idea_wrong_mass(probabilities, [2, 2, 0]), 0.65)

    def test_empty_selection_is_zero(self) -> None:
        self.assertEqual(no_idea_wrong_mass([[0.5, 0.5, 0.0]], [0]), 0.0)


if __name__ == "__main__":
    unittest.main()
