from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufc_predictor.config import DATASET_PATH
from ufc_predictor.data import (
    age_years,
    extract_fight_stats,
    fighter_names,
    load_fights,
    method_bucket,
    safe_float,
)
from ufc_predictor.domain import FightStats, FighterBio, FighterState, update_elo


MEASUREMENT_KEYS = ["age", "height_cm", "reach_cm", "weight_kg", "reach_to_height"]


def apply_fight_result(
    *,
    winner_state: FighterState,
    loser_state: FighterState,
    fight_date: pd.Timestamp,
    stats: FightStats,
) -> None:
    winner_state.update(
        won=True,
        method=stats.method,
        fight_date=fight_date,
        sig_landed=stats.f1_sig_landed,
        sig_attempted=stats.f1_sig_attempted,
        sig_absorbed=stats.f2_sig_landed,
        td_landed=stats.f1_td_landed,
        td_attempted=stats.f1_td_attempted,
        td_absorbed=stats.f2_td_landed,
        kd_for=stats.f1_kd,
        kd_against=stats.f2_kd,
        ctrl_for_seconds=stats.f1_ctrl_seconds,
        ctrl_against_seconds=stats.f2_ctrl_seconds,
        duration_seconds=stats.duration_seconds,
    )
    loser_state.update(
        won=False,
        method=stats.method,
        fight_date=fight_date,
        sig_landed=stats.f2_sig_landed,
        sig_attempted=stats.f2_sig_attempted,
        sig_absorbed=stats.f1_sig_landed,
        td_landed=stats.f2_td_landed,
        td_attempted=stats.f2_td_attempted,
        td_absorbed=stats.f1_td_landed,
        kd_for=stats.f2_kd,
        kd_against=stats.f1_kd,
        ctrl_for_seconds=stats.f2_ctrl_seconds,
        ctrl_against_seconds=stats.f1_ctrl_seconds,
        duration_seconds=stats.duration_seconds,
    )
    update_elo(winner_state, loser_state, method=stats.method)


def fighter_measurements(row: pd.Series, prefix: str, fight_date: pd.Timestamp) -> dict[str, Any]:
    height = safe_float(row.get(f"{prefix}_Height_cm"))
    reach = safe_float(row.get(f"{prefix}_Reach_cm"))
    weight = safe_float(row.get(f"{prefix}_Weight_kg"))
    return {
        "age": age_years(row.get(f"{prefix}_DOB"), fight_date),
        "height_cm": height,
        "reach_cm": reach,
        "weight_kg": weight,
        "reach_to_height": reach / height if height and not np.isnan(height) else np.nan,
        "stance": row.get(f"{prefix}_Stance") if pd.notna(row.get(f"{prefix}_Stance")) else "Unknown",
    }


def bio_measurements(bio: FighterBio, fight_date: pd.Timestamp) -> dict[str, Any]:
    height = bio.height_cm
    reach = bio.reach_cm
    weight = bio.weight_kg
    return {
        "age": age_years(bio.dob, fight_date),
        "height_cm": height,
        "reach_cm": reach,
        "weight_kg": weight,
        "reach_to_height": reach / height if height and not np.isnan(height) else np.nan,
        "stance": bio.stance or "Unknown",
    }


def update_bio(
    bios: dict[str, FighterBio],
    *,
    name: str,
    row: pd.Series,
    prefix: str,
    fight_date: pd.Timestamp,
) -> None:
    bio = bios.setdefault(name, FighterBio())
    bio.height_cm = safe_float(row.get(f"{prefix}_Height_cm"), bio.height_cm)
    bio.weight_kg = safe_float(row.get(f"{prefix}_Weight_kg"), bio.weight_kg)
    bio.reach_cm = safe_float(row.get(f"{prefix}_Reach_cm"), bio.reach_cm)
    if pd.notna(row.get(f"{prefix}_Stance")):
        bio.stance = str(row.get(f"{prefix}_Stance"))
    dob = pd.to_datetime(row.get(f"{prefix}_DOB"), errors="coerce")
    if pd.notna(dob):
        bio.dob = dob
    if pd.notna(row.get("weight_class")):
        bio.latest_weight_class = str(row.get("weight_class"))
    bio.latest_fight_date = fight_date


def add_pair_features(
    output: dict[str, Any],
    a_measurements: dict[str, Any],
    b_measurements: dict[str, Any],
    a_profile: dict[str, float],
    b_profile: dict[str, float],
) -> None:
    for key, value in a_measurements.items():
        if key != "stance":
            output[f"a_{key}"] = value
    for key, value in b_measurements.items():
        if key != "stance":
            output[f"b_{key}"] = value

    for key in MEASUREMENT_KEYS:
        output[f"{key}_diff"] = a_measurements[key] - b_measurements[key]

    for key, value in a_profile.items():
        output[f"a_{key}"] = value
    for key, value in b_profile.items():
        output[f"b_{key}"] = value
    for key in a_profile:
        output[f"{key}_diff"] = a_profile[key] - b_profile[key]

    output["a_stance"] = a_measurements["stance"]
    output["b_stance"] = b_measurements["stance"]
    output["stance_matchup"] = f"{a_measurements['stance']}_vs_{b_measurements['stance']}"


def build_prefight_dataset(
    csv_path: Path = DATASET_PATH,
    *,
    random_seed: int = 42,
    include_dirty_methods: bool = False,
) -> pd.DataFrame:
    raw = load_fights(csv_path, include_dirty_methods=include_dirty_methods)
    states: defaultdict[str, FighterState] = defaultdict(FighterState)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []

    for _, fight in raw.iterrows():
        fight_date = fight["event_date"]
        names = fighter_names(fight)
        if names is None:
            continue
        winner_name, loser_name = names

        winner_state = states[winner_name]
        loser_state = states[loser_name]

        winner_profile = winner_state.profile(fight_date)
        loser_profile = loser_state.profile(fight_date)
        winner_measurements = fighter_measurements(fight, "f1", fight_date)
        loser_measurements = fighter_measurements(fight, "f2", fight_date)

        keep_original_order = bool(rng.integers(0, 2))
        if keep_original_order:
            fighter_a = winner_name
            fighter_b = loser_name
            a_profile, b_profile = winner_profile, loser_profile
            a_measurements, b_measurements = winner_measurements, loser_measurements
            target = 1
        else:
            fighter_a = loser_name
            fighter_b = winner_name
            a_profile, b_profile = loser_profile, winner_profile
            a_measurements, b_measurements = loser_measurements, winner_measurements
            target = 0

        output: dict[str, Any] = {
            "event_date": fight_date,
            "event_name": fight["event_name"],
            "source_row": int(fight["source_row"]),
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner": winner_name,
            "weight_class": fight["weight_class"],
            "method": fight["method"],
            "method_bucket": method_bucket(fight["method"]),
            "fighter_a_wins": target,
        }
        add_pair_features(output, a_measurements, b_measurements, a_profile, b_profile)
        rows.append(output)

        # Leakage guard: post-fight stats update histories only after the row is built.
        apply_fight_result(
            winner_state=winner_state,
            loser_state=loser_state,
            fight_date=fight_date,
            stats=extract_fight_stats(fight),
        )

    return pd.DataFrame(rows)


def build_current_fighter_data(
    csv_path: Path = DATASET_PATH,
    *,
    include_dirty_methods: bool = False,
) -> tuple[dict[str, FighterState], dict[str, FighterBio]]:
    raw = load_fights(csv_path, include_dirty_methods=include_dirty_methods)
    states: defaultdict[str, FighterState] = defaultdict(FighterState)
    bios: dict[str, FighterBio] = {}

    for _, fight in raw.iterrows():
        fight_date = fight["event_date"]
        names = fighter_names(fight)
        if names is None:
            continue
        winner_name, loser_name = names
        update_bio(bios, name=winner_name, row=fight, prefix="f1", fight_date=fight_date)
        update_bio(bios, name=loser_name, row=fight, prefix="f2", fight_date=fight_date)

        winner_state = states[winner_name]
        loser_state = states[loser_name]
        apply_fight_result(
            winner_state=winner_state,
            loser_state=loser_state,
            fight_date=fight_date,
            stats=extract_fight_stats(fight),
        )

    return dict(states), bios


def build_matchup_row(
    fighter_a: str,
    fighter_b: str,
    *,
    fight_date: pd.Timestamp,
    states: dict[str, FighterState],
    bios: dict[str, FighterBio],
    weight_class: str | None = None,
) -> pd.DataFrame:
    if fighter_a not in states or fighter_a not in bios:
        raise KeyError(f"Unknown fighter: {fighter_a}")
    if fighter_b not in states or fighter_b not in bios:
        raise KeyError(f"Unknown fighter: {fighter_b}")

    a_profile = states[fighter_a].profile(fight_date)
    b_profile = states[fighter_b].profile(fight_date)
    a_measurements = bio_measurements(bios[fighter_a], fight_date)
    b_measurements = bio_measurements(bios[fighter_b], fight_date)
    inferred_weight_class = weight_class or bios[fighter_a].latest_weight_class

    output: dict[str, Any] = {
        "event_date": fight_date,
        "event_name": "Future matchup",
        "source_row": -1,
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "winner": "",
        "weight_class": inferred_weight_class,
        "method": "",
        "method_bucket": "",
        "fighter_a_wins": np.nan,
    }
    add_pair_features(output, a_measurements, b_measurements, a_profile, b_profile)
    return pd.DataFrame([output])
