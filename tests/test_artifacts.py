from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ufc_predictor.artifacts import ModelArtifact, load_model_artifact, save_model_artifact  # noqa: E402


class ArtifactTests(unittest.TestCase):
    def test_model_artifact_round_trip_preserves_arrays_and_metadata(self) -> None:
        artifact = ModelArtifact(
            weights=np.array([0.1, 0.2, -0.3]),
            feature_names=["intercept", "elo_diff", "win_pct_diff"],
            means=pd.Series([1500.0, 0.5], index=["elo_diff", "win_pct_diff"]),
            stds=pd.Series([100.0, 0.2], index=["elo_diff", "win_pct_diff"]),
            metadata={"train_start": "2020-01-01", "split_date": "2024-01-01"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.npz"
            save_model_artifact(path, artifact)
            loaded = load_model_artifact(path)

        np.testing.assert_allclose(loaded.weights, artifact.weights)
        self.assertEqual(loaded.feature_names, artifact.feature_names)
        self.assertEqual(loaded.metadata["artifact_version"], 2)
        self.assertEqual(loaded.metadata["model_type"], "numpy_logistic_regression")
        self.assertEqual(loaded.metadata["split_date"], "2024-01-01")
        pd.testing.assert_series_equal(loaded.means, artifact.means)
        pd.testing.assert_series_equal(loaded.stds, artifact.stds)


if __name__ == "__main__":
    unittest.main()
