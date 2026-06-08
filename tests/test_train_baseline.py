from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ufc_predictor.preprocessing import flipped_matchup_rows  # noqa: E402


class TrainBaselineTests(unittest.TestCase):
    def test_flipped_matchup_rows_swap_fighters_and_reverse_differences(self) -> None:
        row = pd.DataFrame(
            [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Bravo",
                    "fighter_a_wins": 1,
                    "a_wins": 3.0,
                    "b_wins": 1.0,
                    "wins_diff": 2.0,
                    "a_stance": "Orthodox",
                    "b_stance": "Southpaw",
                    "stance_matchup": "Orthodox_vs_Southpaw",
                }
            ]
        )

        flipped = flipped_matchup_rows(row).iloc[0]

        self.assertEqual(flipped["fighter_a"], "Bravo")
        self.assertEqual(flipped["fighter_b"], "Alpha")
        self.assertEqual(flipped["fighter_a_wins"], 0)
        self.assertEqual(flipped["a_wins"], 1.0)
        self.assertEqual(flipped["b_wins"], 3.0)
        self.assertEqual(flipped["wins_diff"], -2.0)
        self.assertEqual(flipped["stance_matchup"], "Southpaw_vs_Orthodox")


if __name__ == "__main__":
    unittest.main()
