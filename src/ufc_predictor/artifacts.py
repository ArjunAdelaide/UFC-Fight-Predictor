from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufc_predictor.domain import FighterBio, FighterState


MODEL_ARTIFACT_VERSION = 2
PROFILE_CACHE_VERSION = 1


@dataclass(frozen=True)
class ModelArtifact:
    weights: np.ndarray
    feature_names: list[str]
    means: pd.Series
    stds: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def encoded_feature_names(self) -> list[str]:
        return self.feature_names[1:]


def save_model_artifact(
    path: Path,
    artifact: ModelArtifact,
) -> None:
    metadata = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "model_type": "numpy_logistic_regression",
        "created_at_utc": datetime.now(UTC).isoformat(),
        **artifact.metadata,
    }
    np.savez(
        path,
        weights=artifact.weights,
        feature_names=np.array(artifact.feature_names, dtype=object),
        means=artifact.means.to_numpy(dtype=float),
        stds=artifact.stds.to_numpy(dtype=float),
        metadata_json=np.array(json.dumps(metadata), dtype=object),
        train_start=str(metadata.get("train_start", "")),
        train_end=str(metadata.get("train_end", "")),
        test_start=str(metadata.get("test_start", "")),
        test_end=str(metadata.get("test_end", "")),
    )


def load_model_artifact(path: Path) -> ModelArtifact:
    model_data = np.load(path, allow_pickle=True)
    feature_names = model_data["feature_names"].tolist()
    expected_columns = feature_names[1:]
    means = pd.Series(model_data["means"], index=expected_columns)
    stds = pd.Series(model_data["stds"], index=expected_columns).replace(0, 1.0)

    if "metadata_json" in model_data:
        metadata = json.loads(str(model_data["metadata_json"].item()))
    else:
        def optional_string(key: str) -> str:
            return str(model_data[key]) if key in model_data else ""

        metadata = {
            "artifact_version": 1,
            "model_type": "numpy_logistic_regression",
            "train_start": optional_string("train_start"),
            "train_end": optional_string("train_end"),
            "test_start": optional_string("test_start"),
            "test_end": optional_string("test_end"),
        }

    return ModelArtifact(
        weights=model_data["weights"],
        feature_names=feature_names,
        means=means,
        stds=stds,
        metadata=metadata,
    )


@dataclass(frozen=True)
class ProfileCache:
    dataset_path: str
    states: dict[str, FighterState]
    bios: dict[str, FighterBio]
    metadata: dict[str, Any] = field(default_factory=dict)


def save_profile_cache(
    path: Path,
    *,
    dataset_path: Path,
    states: dict[str, FighterState],
    bios: dict[str, FighterBio],
) -> None:
    payload = {
        "artifact_version": PROFILE_CACHE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "states": states,
        "bios": bios,
    }
    with path.open("wb") as cache_file:
        pickle.dump(payload, cache_file)


def load_profile_cache(path: Path) -> ProfileCache:
    with path.open("rb") as cache_file:
        payload = pickle.load(cache_file)
    return ProfileCache(
        dataset_path=str(payload.get("dataset_path", "")),
        states=payload["states"],
        bios=payload["bios"],
        metadata={
            "artifact_version": payload.get("artifact_version", 0),
            "created_at_utc": payload.get("created_at_utc", ""),
        },
    )
