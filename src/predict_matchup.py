from __future__ import annotations

import argparse
import pickle
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    DATASET_PATH,
    build_current_fighter_data,
    build_matchup_row,
)
from train_baseline import (
    CATEGORICAL_COLUMNS,
    MODEL_OUTPUT,
    PROFILE_CACHE_OUTPUT,
    model_feature_columns,
    sigmoid,
)


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


def prepare_single_matchup_matrix(row: pd.DataFrame, model_data: np.lib.npyio.NpzFile) -> np.ndarray:
    feature_names = model_data["feature_names"].tolist()
    expected_columns = feature_names[1:]
    means = pd.Series(model_data["means"], index=expected_columns)
    stds = pd.Series(model_data["stds"], index=expected_columns).replace(0, 1.0)

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


def load_model() -> np.lib.npyio.NpzFile:
    if not MODEL_OUTPUT.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_OUTPUT}\n"
            "Run `python3 src/train_baseline.py` first."
        )
    return np.load(MODEL_OUTPUT, allow_pickle=True)


def display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def load_fighter_profiles(
    dataset_path: Path,
    *,
    rebuild: bool = False,
) -> tuple[dict, dict, str]:
    if rebuild or not PROFILE_CACHE_OUTPUT.exists():
        states, bios = build_current_fighter_data(dataset_path)
        return states, bios, "rebuilt from dataset"

    with PROFILE_CACHE_OUTPUT.open("rb") as cache_file:
        payload = pickle.load(cache_file)
    return payload["states"], payload["bios"], f"loaded from {display_path(PROFILE_CACHE_OUTPUT)}"


def main() -> None:
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
    args = parser.parse_args()

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

    model_data = load_model()
    matchup = build_matchup_row(
        fighter_a,
        fighter_b,
        fight_date=fight_date,
        states=states,
        bios=bios,
        weight_class=args.weight_class,
    )
    matrix = prepare_single_matchup_matrix(matchup, model_data)
    probability_a = float(sigmoid(matrix @ model_data["weights"])[0])
    probability_b = 1.0 - probability_a

    a_profile = states[fighter_a].profile(fight_date)
    b_profile = states[fighter_b].profile(fight_date)

    print("\nPrediction")
    print(f"  {fighter_a}: {probability_a:.1%}")
    print(f"  {fighter_b}: {probability_b:.1%}")
    print(f"  fight date: {fight_date.date()}")
    print(f"  weight class: {matchup.loc[0, 'weight_class']}")
    print(f"  profiles: {profile_source}")

    print("\nQuick comparison")
    print(f"  Elo: {fighter_a} {a_profile['elo']:.0f} vs {fighter_b} {b_profile['elo']:.0f}")
    print(
        "  Record in dataset: "
        f"{fighter_a} {int(a_profile['wins'])}-{int(a_profile['losses'])} vs "
        f"{fighter_b} {int(b_profile['wins'])}-{int(b_profile['losses'])}"
    )
    print(
        "  Recent win pct: "
        f"{fighter_a} {a_profile['recent_win_pct']:.2f} vs "
        f"{fighter_b} {b_profile['recent_win_pct']:.2f}"
    )

    print("\nNote")
    print("  This is a baseline model, not betting advice. UFC outcomes are noisy.")


if __name__ == "__main__":
    main()
