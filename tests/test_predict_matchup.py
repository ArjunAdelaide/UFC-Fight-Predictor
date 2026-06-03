from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import predict_matchup  # noqa: E402
from feature_engineering import FighterBio, FighterState  # noqa: E402


class PredictMatchupTests(unittest.TestCase):
    def test_unknown_fighter_name_suggests_close_matches(self) -> None:
        with self.assertRaises(ValueError) as error:
            predict_matchup.resolve_fighter_name(
                "Islam Makachev",
                ["Islam Makhachev", "Ilia Topuria"],
            )

        message = str(error.exception)
        self.assertIn("Did you mean", message)
        self.assertIn("Islam Makhachev", message)

    def test_cached_profiles_load_without_rebuild(self) -> None:
        states = {"Alpha": FighterState(wins=1, fights=1)}
        bios = {"Alpha": FighterBio()}

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "profiles.pkl"
            with cache_path.open("wb") as cache_file:
                pickle.dump({"states": states, "bios": bios}, cache_file)

            with patch.object(predict_matchup, "PROFILE_CACHE_OUTPUT", cache_path):
                loaded_states, loaded_bios, source = predict_matchup.load_fighter_profiles(
                    Path("unused.csv")
                )

        self.assertEqual(loaded_states["Alpha"].wins, 1)
        self.assertIn("Alpha", loaded_bios)
        self.assertIn("loaded from", source)


if __name__ == "__main__":
    unittest.main()
