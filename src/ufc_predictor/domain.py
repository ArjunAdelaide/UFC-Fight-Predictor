from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


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
