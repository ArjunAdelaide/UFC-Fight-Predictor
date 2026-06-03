from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_engineering import build_prefight_dataset  # noqa: E402


def fight_row(
    *,
    event_date: str,
    winner: str,
    loser: str,
    winner_sig_landed: int,
    loser_sig_landed: int,
    method: str = "Decision - Unanimous",
) -> dict[str, object]:
    return {
        "event_name": f"{winner} vs {loser}",
        "event_date": event_date,
        "fighter_1": winner,
        "fighter_2": loser,
        "method": method,
        "round_num": 3,
        "time": "5:00",
        "weight_class": "Lightweight",
        "f1_Height_cm": 180,
        "f2_Height_cm": 178,
        "f1_Reach_cm": 185,
        "f2_Reach_cm": 180,
        "f1_Weight_kg": 70,
        "f2_Weight_kg": 70,
        "f1_DOB": "1990-01-01",
        "f2_DOB": "1991-01-01",
        "f1_Stance": "Orthodox",
        "f2_Stance": "Southpaw",
        "f1_Sig_str_landed": winner_sig_landed,
        "f1_Sig_str_attempted": winner_sig_landed * 2,
        "f2_Sig_str_landed": loser_sig_landed,
        "f2_Sig_str_attempted": loser_sig_landed * 2,
        "f1_Td_landed": 1,
        "f1_Td_attempted": 2,
        "f2_Td_landed": 0,
        "f2_Td_attempted": 1,
        "f1_KD": 0,
        "f2_KD": 0,
        "f1_Ctrl": "1:00",
        "f2_Ctrl": "0:30",
    }


def write_fights_csv(rows: list[dict[str, object]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


class FeatureEngineeringTests(unittest.TestCase):
    def test_fighter_a_orientation_matches_target(self) -> None:
        rows = [
            fight_row(
                event_date=f"2020-01-{day:02d}",
                winner=f"Winner {day}",
                loser=f"Loser {day}",
                winner_sig_landed=20 + day,
                loser_sig_landed=10 + day,
            )
            for day in range(1, 21)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "fights.csv"
            write_fights_csv(rows, csv_path)

            features = build_prefight_dataset(csv_path, random_seed=42)

        self.assertEqual(set(features["fighter_a_wins"]), {0, 1})

        for _, row in features.iterrows():
            if row["fighter_a_wins"] == 1:
                self.assertEqual(row["fighter_a"], row["winner"])
            else:
                self.assertEqual(row["fighter_b"], row["winner"])

    def test_current_fight_stats_do_not_leak_into_prefight_row(self) -> None:
        rows = [
            fight_row(
                event_date="2020-01-01",
                winner="Alpha",
                loser="Bravo",
                winner_sig_landed=100,
                loser_sig_landed=10,
            ),
            fight_row(
                event_date="2021-01-01",
                winner="Alpha",
                loser="Charlie",
                winner_sig_landed=300,
                loser_sig_landed=20,
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "fights.csv"
            write_fights_csv(rows, csv_path)

            features = build_prefight_dataset(csv_path, random_seed=7)

        first_fight = features.loc[features["source_row"] == 0].iloc[0]
        self.assertEqual(first_fight["a_fights"], 0)
        self.assertEqual(first_fight["b_fights"], 0)
        self.assertEqual(first_fight["sig_landed_per_fight_diff"], 0)

        second_fight = features.loc[features["source_row"] == 1].iloc[0]

        if second_fight["fighter_a"] == "Alpha":
            alpha_prefix = "a"
            opponent_prefix = "b"
        else:
            alpha_prefix = "b"
            opponent_prefix = "a"

        self.assertEqual(second_fight[f"{alpha_prefix}_fights"], 1)
        self.assertEqual(second_fight[f"{alpha_prefix}_wins"], 1)
        self.assertEqual(second_fight[f"{alpha_prefix}_sig_landed_per_fight"], 100)
        self.assertNotEqual(second_fight[f"{alpha_prefix}_sig_landed_per_fight"], 300)

        self.assertEqual(second_fight[f"{opponent_prefix}_fights"], 0)
        self.assertEqual(second_fight[f"{opponent_prefix}_sig_landed_per_fight"], 0)

    def test_dirty_methods_are_excluded_by_default(self) -> None:
        rows = [
            fight_row(
                event_date="2020-01-01",
                winner="Alpha",
                loser="Bravo",
                winner_sig_landed=50,
                loser_sig_landed=20,
            ),
            fight_row(
                event_date="2020-02-01",
                winner="Charlie",
                loser="Delta",
                winner_sig_landed=40,
                loser_sig_landed=30,
                method="Overturned",
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "fights.csv"
            write_fights_csv(rows, csv_path)

            features = build_prefight_dataset(csv_path)

        self.assertEqual(len(features), 1)
        self.assertEqual(features.iloc[0]["winner"], "Alpha")


if __name__ == "__main__":
    unittest.main()
