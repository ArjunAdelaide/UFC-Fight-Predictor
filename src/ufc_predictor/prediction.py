from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_predictor.artifacts import ModelArtifact, load_model_artifact, load_profile_cache
from ufc_predictor.config import DATASET_PATH, DEFAULT_PATHS, ProjectPaths
from ufc_predictor.domain import FighterBio, FighterState
from ufc_predictor.features import build_current_fighter_data, build_matchup_row
from ufc_predictor.model import sigmoid
from ufc_predictor.preprocessing import CATEGORICAL_COLUMNS, model_feature_columns


@dataclass(frozen=True)
class PredictionResult:
    fighter_a: str
    fighter_b: str
    probability_a: float
    probability_b: float
    fight_date: pd.Timestamp
    weight_class: str
    profile_source: str
    a_profile: dict[str, float]
    b_profile: dict[str, float]


def resolve_fighter_name(name: str, known_names: list[str]) -> str:
    exact_lookup = {fighter.lower(): fighter for fighter in known_names}
    normalized = name.strip().lower()
    if normalized in exact_lookup:
        return exact_lookup[normalized]

    matches = get_close_matches(name, known_names, n=5, cutoff=0.65)
    if not matches:
        raise ValueError(f"No fighter found for '{name}'.")

    match_list = "\n".join(f"  - {match}" for match in matches)
    raise ValueError(
        f"No exact fighter found for '{name}'. Did you mean one of these?\n{match_list}"
    )


def is_dummy_column(column: str) -> bool:
    return any(column.startswith(f"{categorical}_") for categorical in CATEGORICAL_COLUMNS)


def prepare_single_matchup_matrix(row: pd.DataFrame, model_artifact: ModelArtifact) -> np.ndarray:
    expected_columns = model_artifact.encoded_feature_names
    means = model_artifact.means
    stds = model_artifact.stds.replace(0, 1.0)

    values = {}
    for column in expected_columns:
        values[column] = 0.0 if is_dummy_column(column) else float(means[column])

    feature_columns = model_feature_columns(row)
    candidate = row[feature_columns].iloc[0]

    for column, value in candidate.items():
        if column in CATEGORICAL_COLUMNS:
            category_value = "Unknown" if pd.isna(value) else str(value)
            dummy_name = f"{column}_{category_value}"
            if dummy_name in values:
                values[dummy_name] = 1.0
        elif column in values and pd.notna(value):
            values[column] = float(value)

    ordered = pd.Series(values, index=expected_columns)
    scaled = (ordered - means) / stds
    return np.array([[1.0, *scaled.to_numpy(dtype=float)]])


def load_model(paths: ProjectPaths = DEFAULT_PATHS) -> ModelArtifact:
    if not paths.model_output.exists():
        raise FileNotFoundError(
            f"Model file not found: {paths.model_output}\n"
            "Run `python3 src/train_baseline.py` first."
        )
    return load_model_artifact(paths.model_output)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def load_fighter_profiles(
    dataset_path: Path,
    *,
    paths: ProjectPaths = DEFAULT_PATHS,
    rebuild: bool = False,
) -> tuple[dict[str, FighterState], dict[str, FighterBio], str]:
    if rebuild or not paths.profile_cache_output.exists():
        states, bios = build_current_fighter_data(dataset_path)
        return states, bios, "rebuilt from dataset"

    cache = load_profile_cache(paths.profile_cache_output)
    return cache.states, cache.bios, f"loaded from {display_path(paths.profile_cache_output)}"


def predict_matchup(
    fighter_a: str,
    fighter_b: str,
    *,
    fight_date: pd.Timestamp,
    states: dict[str, FighterState],
    bios: dict[str, FighterBio],
    model_artifact: ModelArtifact,
    weight_class: str | None = None,
    profile_source: str = "",
) -> PredictionResult:
    matchup = build_matchup_row(
        fighter_a,
        fighter_b,
        fight_date=fight_date,
        states=states,
        bios=bios,
        weight_class=weight_class,
    )
    matrix = prepare_single_matchup_matrix(matchup, model_artifact)
    probability_a = float(sigmoid(matrix @ model_artifact.weights)[0])
    probability_b = 1.0 - probability_a
    a_profile = states[fighter_a].profile(fight_date)
    b_profile = states[fighter_b].profile(fight_date)

    return PredictionResult(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        probability_a=probability_a,
        probability_b=probability_b,
        fight_date=fight_date,
        weight_class=str(matchup.loc[0, "weight_class"]),
        profile_source=profile_source,
        a_profile=a_profile,
        b_profile=b_profile,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the probability that Fighter A beats Fighter B."
    )
    parser.add_argument("fighter_a", help="Name of the first fighter.")
    parser.add_argument("fighter_b", help="Name of the second fighter.")
    parser.add_argument(
        "--date",
        default=None,
        help="Fight date as YYYY-MM-DD. Defaults to today.",
    )
    parser.add_argument(
        "--weight-class",
        default=None,
        help="Optional UFC weight class, e.g. Lightweight or Welterweight.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATASET_PATH,
        help="Path to the cleaned UFC dataset CSV. Used when rebuilding fighter profiles.",
    )
    parser.add_argument(
        "--rebuild-profiles",
        action="store_true",
        help="Rebuild fighter profiles from the dataset instead of using the saved cache.",
    )
    return parser.parse_args()


def print_prediction(result: PredictionResult) -> None:
    print("\nPrediction")
    print(f"  {result.fighter_a}: {result.probability_a:.1%}")
    print(f"  {result.fighter_b}: {result.probability_b:.1%}")
    print(f"  fight date: {result.fight_date.date()}")
    print(f"  weight class: {result.weight_class}")
    print(f"  profiles: {result.profile_source}")

    print("\nQuick comparison")
    print(
        f"  Elo: {result.fighter_a} {result.a_profile['elo']:.0f} vs "
        f"{result.fighter_b} {result.b_profile['elo']:.0f}"
    )
    print(
        "  Record in dataset: "
        f"{result.fighter_a} {int(result.a_profile['wins'])}-{int(result.a_profile['losses'])} vs "
        f"{result.fighter_b} {int(result.b_profile['wins'])}-{int(result.b_profile['losses'])}"
    )
    print(
        "  Recent win pct: "
        f"{result.fighter_a} {result.a_profile['recent_win_pct']:.2f} vs "
        f"{result.fighter_b} {result.b_profile['recent_win_pct']:.2f}"
    )

    print("\nNote")
    print("  This is a baseline model, not betting advice. UFC outcomes are noisy.")


def main() -> None:
    args = parse_args()
    fight_date = (
        pd.Timestamp(args.date)
        if args.date
        else pd.Timestamp.now(tz="Australia/Adelaide").tz_localize(None).normalize()
    )

    states, bios, profile_source = load_fighter_profiles(
        args.data.expanduser(),
        rebuild=args.rebuild_profiles,
    )
    known_names = sorted(states)
    fighter_a = resolve_fighter_name(args.fighter_a, known_names)
    fighter_b = resolve_fighter_name(args.fighter_b, known_names)

    result = predict_matchup(
        fighter_a,
        fighter_b,
        fight_date=fight_date,
        states=states,
        bios=bios,
        model_artifact=load_model(),
        weight_class=args.weight_class,
        profile_source=profile_source,
    )
    print_prediction(result)


if __name__ == "__main__":
    main()
