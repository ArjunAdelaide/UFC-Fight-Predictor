from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "clean_ufc_dataset.csv"
DATASET_PATH = Path(os.environ.get("UFC_DATASET_PATH", DEFAULT_DATASET_PATH)).expanduser()

EXCLUDED_METHODS = {"Overturned", "Other"}
MEASUREMENT_KEYS = ["age", "height_cm", "reach_cm", "weight_kg", "reach_to_height"]


@dataclass(frozen=True)
class FightStats:
    method: str
    duration_seconds: float
    f1_sig_landed: float
    f1_sig_attempted: float
    f2_sig_landed: float
    f2_sig_attempted: float
    f1_td_landed: float
    f1_td_attempted: float
    f2_td_landed: float
    f2_td_attempted: float
    f1_kd: float
    f2_kd: float
    f1_ctrl_seconds: float
    f2_ctrl_seconds: float


@dataclass
class FighterBio:
    height_cm: float = np.nan
    weight_kg: float = np.nan
    reach_cm: float = np.nan
    stance: str = "Unknown"
    dob: pd.Timestamp | None = None
    latest_weight_class: str = "Unknown"
    latest_fight_date: pd.Timestamp | None = None


def parse_control_time(value: Any) -> float:
    """Convert UFCStats control time strings such as MM:SS to seconds."""
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text == "--":
        return 0.0
    try:
        minutes, seconds = text.split(":")
        return float(int(minutes) * 60 + int(seconds))
    except ValueError:
        return 0.0


def parse_finish_time(row: pd.Series) -> float:
    """Approximate total fight duration in seconds."""
    round_num = row.get("round_num")
    time_text = row.get("time")
    if pd.isna(round_num) or pd.isna(time_text):
        return np.nan
    try:
        current_round = int(round_num)
        minutes, seconds = str(time_text).split(":")
        elapsed_this_round = int(minutes) * 60 + int(seconds)
        return float((current_round - 1) * 300 + elapsed_this_round)
    except (TypeError, ValueError):
        return np.nan


def method_bucket(method: Any) -> str:
    if pd.isna(method):
        return "unknown"
    text = str(method)
    if "KO" in text or "TKO" in text:
        return "ko_tko"
    if "Submission" in text:
        return "submission"
    if "Decision" in text:
        return "decision"
    if "DQ" in text:
        return "dq"
    return "other"


def safe_float(value: Any, default: float = np.nan) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def age_years(dob: Any, fight_date: pd.Timestamp) -> float:
    if pd.isna(dob) or pd.isna(fight_date):
        return np.nan
    dob_ts = pd.to_datetime(dob, errors="coerce")
    if pd.isna(dob_ts):
        return np.nan
    return float((fight_date - dob_ts).days / 365.25)


@dataclass
class FighterState:
    fights: int = 0
    wins: int = 0
    losses: int = 0
    finish_wins: int = 0
    ko_tko_wins: int = 0
    submission_wins: int = 0
    decision_wins: int = 0
    finish_losses: int = 0
    ko_tko_losses: int = 0
    submission_losses: int = 0
    decision_losses: int = 0
    sig_landed: float = 0.0
    sig_attempted: float = 0.0
    sig_absorbed: float = 0.0
    td_landed: float = 0.0
    td_attempted: float = 0.0
    td_absorbed: float = 0.0
    kd_for: float = 0.0
    kd_against: float = 0.0
    ctrl_for_seconds: float = 0.0
    ctrl_against_seconds: float = 0.0
    total_duration_seconds: float = 0.0
    duration_count: int = 0
    streak: int = 0
    longest_win_streak: int = 0
    last_fight_date: pd.Timestamp | None = None
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=3))
    recent_sig_diffs: deque[float] = field(default_factory=lambda: deque(maxlen=3))
    elo: float = 1500.0
    method_adjusted_elo: float = 1500.0

    def profile(self, fight_date: pd.Timestamp) -> dict[str, float]:
        fights = self.fights
        wins = self.wins
        sig_diff = self.sig_landed - self.sig_absorbed
        td_diff = self.td_landed - self.td_absorbed
        ctrl_diff = self.ctrl_for_seconds - self.ctrl_against_seconds

        days_since_last = np.nan
        if self.last_fight_date is not None:
            days_since_last = float((fight_date - self.last_fight_date).days)

        return {
            "fights": float(fights),
            "wins": float(wins),
            "losses": float(self.losses),
            "win_pct": wins / fights if fights else 0.5,
            "finish_win_rate": self.finish_wins / wins if wins else 0.0,
            "ko_tko_win_rate": self.ko_tko_wins / wins if wins else 0.0,
            "submission_win_rate": self.submission_wins / wins if wins else 0.0,
            "decision_win_rate": self.decision_wins / wins if wins else 0.0,
            "finish_loss_rate": self.finish_losses / self.losses if self.losses else 0.0,
            "sig_landed_per_fight": self.sig_landed / fights if fights else 0.0,
            "sig_attempted_per_fight": self.sig_attempted / fights if fights else 0.0,
            "sig_absorbed_per_fight": self.sig_absorbed / fights if fights else 0.0,
            "sig_accuracy": self.sig_landed / self.sig_attempted if self.sig_attempted else 0.0,
            "sig_diff_per_fight": sig_diff / fights if fights else 0.0,
            "td_landed_per_fight": self.td_landed / fights if fights else 0.0,
            "td_attempted_per_fight": self.td_attempted / fights if fights else 0.0,
            "td_absorbed_per_fight": self.td_absorbed / fights if fights else 0.0,
            "td_accuracy": self.td_landed / self.td_attempted if self.td_attempted else 0.0,
            "td_diff_per_fight": td_diff / fights if fights else 0.0,
            "kd_for_per_fight": self.kd_for / fights if fights else 0.0,
            "kd_against_per_fight": self.kd_against / fights if fights else 0.0,
            "ctrl_for_seconds_per_fight": self.ctrl_for_seconds / fights if fights else 0.0,
            "ctrl_against_seconds_per_fight": self.ctrl_against_seconds / fights if fights else 0.0,
            "ctrl_diff_seconds_per_fight": ctrl_diff / fights if fights else 0.0,
            "avg_duration_seconds": (
                self.total_duration_seconds / self.duration_count
                if self.duration_count
                else 0.0
            ),
            "streak": float(self.streak),
            "longest_win_streak": float(self.longest_win_streak),
            "recent_win_pct": (
                float(sum(self.recent_results) / len(self.recent_results))
                if self.recent_results
                else 0.5
            ),
            "recent_sig_diff": (
                float(sum(self.recent_sig_diffs) / len(self.recent_sig_diffs))
                if self.recent_sig_diffs
                else 0.0
            ),
            "days_since_last_fight": days_since_last,
            "is_debut": 1.0 if fights == 0 else 0.0,
            "elo": self.elo,
            "method_adjusted_elo": self.method_adjusted_elo,
        }

    def update(
        self,
        *,
        won: bool,
        method: str,
        fight_date: pd.Timestamp,
        sig_landed: float,
        sig_attempted: float,
        sig_absorbed: float,
        td_landed: float,
        td_attempted: float,
        td_absorbed: float,
        kd_for: float,
        kd_against: float,
        ctrl_for_seconds: float,
        ctrl_against_seconds: float,
        duration_seconds: float,
    ) -> None:
        self.fights += 1
        self.wins += int(won)
        self.losses += int(not won)

        if won:
            if method in {"ko_tko", "submission"}:
                self.finish_wins += 1
            if method == "ko_tko":
                self.ko_tko_wins += 1
            elif method == "submission":
                self.submission_wins += 1
            elif method == "decision":
                self.decision_wins += 1
            self.streak = max(1, self.streak + 1)
            self.longest_win_streak = max(self.longest_win_streak, self.streak)
        else:
            if method in {"ko_tko", "submission"}:
                self.finish_losses += 1
            if method == "ko_tko":
                self.ko_tko_losses += 1
            elif method == "submission":
                self.submission_losses += 1
            elif method == "decision":
                self.decision_losses += 1
            self.streak = min(-1, self.streak - 1)

        self.sig_landed += sig_landed
        self.sig_attempted += sig_attempted
        self.sig_absorbed += sig_absorbed
        self.td_landed += td_landed
        self.td_attempted += td_attempted
        self.td_absorbed += td_absorbed
        self.kd_for += kd_for
        self.kd_against += kd_against
        self.ctrl_for_seconds += ctrl_for_seconds
        self.ctrl_against_seconds += ctrl_against_seconds

        if not np.isnan(duration_seconds):
            self.total_duration_seconds += duration_seconds
            self.duration_count += 1

        self.recent_results.append(1 if won else 0)
        self.recent_sig_diffs.append(sig_landed - sig_absorbed)
        self.last_fight_date = fight_date


def elo_delta(winner_rating: float, loser_rating: float, k_factor: float) -> tuple[float, float]:
    expected_winner = 1.0 / (1.0 + 10.0 ** ((loser_rating - winner_rating) / 400.0))
    expected_loser = 1.0 - expected_winner
    return k_factor * (1.0 - expected_winner), k_factor * (0.0 - expected_loser)


def method_elo_multiplier(method: str) -> float:
    if method in {"ko_tko", "submission"}:
        return 1.25
    if method == "decision":
        return 1.0
    return 0.9


def update_elo(
    winner: FighterState,
    loser: FighterState,
    *,
    method: str,
    k_factor: float = 32.0,
) -> None:
    winner_delta, loser_delta = elo_delta(winner.elo, loser.elo, k_factor)
    winner.elo += winner_delta
    loser.elo += loser_delta

    adjusted_k = k_factor * method_elo_multiplier(method)
    winner_delta, loser_delta = elo_delta(
        winner.method_adjusted_elo,
        loser.method_adjusted_elo,
        adjusted_k,
    )
    winner.method_adjusted_elo += winner_delta
    loser.method_adjusted_elo += loser_delta


def load_fights(csv_path: Path, *, include_dirty_methods: bool = False) -> pd.DataFrame:
    fights = pd.read_csv(csv_path, parse_dates=["event_date"])
    fights = fights.reset_index(names="source_row")
    fights = fights.sort_values(["event_date", "source_row"], ascending=[True, True])
    if not include_dirty_methods:
        fights = fights[~fights["method"].isin(EXCLUDED_METHODS)].copy()
    return fights


def fighter_names(fight: pd.Series) -> tuple[str, str] | None:
    winner_name = fight["fighter_1"]
    loser_name = fight["fighter_2"]
    if pd.isna(winner_name) or pd.isna(loser_name):
        return None
    return str(winner_name), str(loser_name)


def extract_fight_stats(fight: pd.Series) -> FightStats:
    return FightStats(
        method=method_bucket(fight["method"]),
        duration_seconds=parse_finish_time(fight),
        f1_sig_landed=safe_float(fight.get("f1_Sig_str_landed"), 0.0),
        f1_sig_attempted=safe_float(fight.get("f1_Sig_str_attempted"), 0.0),
        f2_sig_landed=safe_float(fight.get("f2_Sig_str_landed"), 0.0),
        f2_sig_attempted=safe_float(fight.get("f2_Sig_str_attempted"), 0.0),
        f1_td_landed=safe_float(fight.get("f1_Td_landed"), 0.0),
        f1_td_attempted=safe_float(fight.get("f1_Td_attempted"), 0.0),
        f2_td_landed=safe_float(fight.get("f2_Td_landed"), 0.0),
        f2_td_attempted=safe_float(fight.get("f2_Td_attempted"), 0.0),
        f1_kd=safe_float(fight.get("f1_KD"), 0.0),
        f2_kd=safe_float(fight.get("f2_KD"), 0.0),
        f1_ctrl_seconds=parse_control_time(fight.get("f1_Ctrl")),
        f2_ctrl_seconds=parse_control_time(fight.get("f2_Ctrl")),
    )


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
