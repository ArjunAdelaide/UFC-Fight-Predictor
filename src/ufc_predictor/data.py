from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ufc_predictor.domain import FightStats


EXCLUDED_METHODS = {"Overturned", "Other"}


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
