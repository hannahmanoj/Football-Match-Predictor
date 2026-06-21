from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "match_model.joblib"

TEAM_COLUMNS = [
    "team",
    "group",
    "elo",
    "fifa_rank",
    "form_points",
    "attack_rating",
    "defense_rating",
]

MODEL_FEATURE_COLUMNS = [
    "elo_diff",
    "elo_abs_diff",
    "is_close_elo_match",
    "home_elo",
    "away_elo",
    "form_diff",
    "form_abs_diff",
    "is_close_form_match",
    "home_form_points",
    "away_form_points",
    "attack_diff",
    "attack_abs_diff",
    "is_close_attack_match",
    "defense_diff",
    "defense_abs_diff",
    "is_close_defense_match",
    "combined_attack",
    "combined_defense",
    "low_total_attack",
    "high_total_defense",
    "home_attack_rating",
    "away_attack_rating",
    "home_defense_rating",
    "away_defense_rating",
    "home_goal_diff_recent",
    "away_goal_diff_recent",
    "goal_diff_recent_diff",
    "goal_diff_recent_abs_diff",
    "similar_recent_goal_difference",
    "home_win_rate_recent",
    "away_win_rate_recent",
    "win_rate_diff",
    "home_draw_rate_recent",
    "away_draw_rate_recent",
    "draw_rate_diff",
    "similar_draw_rate",
    "drawish_profile",
    "neutral",
    "home_advantage",
    "tournament_importance",
    "is_world_cup",
    "is_qualifier",
    "is_friendly",
]


@dataclass(frozen=True)
class MatchPrediction:
    team_a: str
    team_b: str
    team_a_win: float
    draw: float
    team_b_win: float
    explanation: pd.DataFrame


def load_teams(path: str | Path) -> pd.DataFrame:
    teams = pd.read_csv(path)
    missing = sorted(set(TEAM_COLUMNS) - set(teams.columns))
    if missing:
        raise ValueError(f"Team data is missing required columns: {missing}")
    extra_columns = [column for column in teams.columns if column not in TEAM_COLUMNS]
    return teams[TEAM_COLUMNS + extra_columns].copy()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def _team_row(teams: pd.DataFrame, team: str) -> pd.Series:
    rows = teams.loc[teams["team"] == team]
    if rows.empty:
        raise ValueError(f"Unknown team: {team}")
    return rows.iloc[0]


@lru_cache(maxsize=1)
def _load_model_artifact() -> dict:
    return joblib.load(MODEL_PATH)


def _fallback_prediction(teams: pd.DataFrame, team_a: str, team_b: str) -> MatchPrediction:
    a = _team_row(teams, team_a)
    b = _team_row(teams, team_b)

    factors = pd.DataFrame(
        [
            {
                "factor": "Elo rating",
                "team_a_value": a["elo"],
                "team_b_value": b["elo"],
                "impact": (a["elo"] - b["elo"]) / 420,
            },
            {
                "factor": "Ranking",
                "team_a_value": a["fifa_rank"],
                "team_b_value": b["fifa_rank"],
                "impact": (b["fifa_rank"] - a["fifa_rank"]) / 65,
            },
            {
                "factor": "Recent form",
                "team_a_value": a["form_points"],
                "team_b_value": b["form_points"],
                "impact": (a["form_points"] - b["form_points"]) / 12,
            },
            {
                "factor": "Attack vs opponent defense",
                "team_a_value": a["attack_rating"],
                "team_b_value": b["defense_rating"],
                "impact": ((a["attack_rating"] - b["defense_rating"]) - (b["attack_rating"] - a["defense_rating"])) / 30,
            },
            {
                "factor": "Defensive strength",
                "team_a_value": a["defense_rating"],
                "team_b_value": b["defense_rating"],
                "impact": (a["defense_rating"] - b["defense_rating"]) / 45,
            },
        ]
    )

    strength_gap = float(factors["impact"].sum())
    draw_logit = 0.45 - min(abs(strength_gap), 1.8) * 0.38
    logits = np.array([strength_gap, draw_logit, -strength_gap])
    probs = _softmax(logits)

    explanation = factors.assign(
        favors=np.where(factors["impact"] > 0, team_a, np.where(factors["impact"] < 0, team_b, "Even")),
        abs_impact=factors["impact"].abs(),
    ).sort_values("abs_impact", ascending=False)

    return MatchPrediction(
        team_a=team_a,
        team_b=team_b,
        team_a_win=float(probs[0]),
        draw=float(probs[1]),
        team_b_win=float(probs[2]),
        explanation=explanation.drop(columns=["abs_impact"]),
    )


def _value(row: pd.Series, column: str, default: float = 0.0) -> float:
    return float(row[column]) if column in row and pd.notna(row[column]) else default


def _model_features(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    elo_diff = _value(a, "elo") - _value(b, "elo")
    form_diff = _value(a, "form_points") - _value(b, "form_points")
    attack_diff = _value(a, "attack_rating") - _value(b, "defense_rating")
    defense_diff = _value(a, "defense_rating") - _value(b, "attack_rating")
    goal_diff_recent_diff = _value(a, "goal_difference_recent") - _value(b, "goal_difference_recent")
    win_rate_diff = _value(a, "win_rate_recent") - _value(b, "win_rate_recent")
    draw_rate_diff = _value(a, "draw_rate_recent") - _value(b, "draw_rate_recent")
    combined_attack = _value(a, "attack_rating") + _value(b, "attack_rating")
    combined_defense = _value(a, "defense_rating") + _value(b, "defense_rating")

    features = {
        "elo_diff": elo_diff,
        "elo_abs_diff": abs(elo_diff),
        "is_close_elo_match": int(abs(elo_diff) < 75),
        "home_elo": _value(a, "elo"),
        "away_elo": _value(b, "elo"),
        "form_diff": form_diff,
        "form_abs_diff": abs(form_diff),
        "is_close_form_match": int(abs(form_diff) <= 2),
        "home_form_points": _value(a, "form_points"),
        "away_form_points": _value(b, "form_points"),
        "attack_diff": attack_diff,
        "attack_abs_diff": abs(attack_diff),
        "is_close_attack_match": int(abs(attack_diff) < 8),
        "defense_diff": defense_diff,
        "defense_abs_diff": abs(defense_diff),
        "is_close_defense_match": int(abs(defense_diff) < 8),
        "combined_attack": combined_attack,
        "combined_defense": combined_defense,
        "low_total_attack": int(combined_attack < 150),
        "high_total_defense": int(combined_defense > 160),
        "home_attack_rating": _value(a, "attack_rating"),
        "away_attack_rating": _value(b, "attack_rating"),
        "home_defense_rating": _value(a, "defense_rating"),
        "away_defense_rating": _value(b, "defense_rating"),
        "home_goal_diff_recent": _value(a, "goal_difference_recent"),
        "away_goal_diff_recent": _value(b, "goal_difference_recent"),
        "goal_diff_recent_diff": goal_diff_recent_diff,
        "goal_diff_recent_abs_diff": abs(goal_diff_recent_diff),
        "similar_recent_goal_difference": int(abs(goal_diff_recent_diff) < 0.5),
        "home_win_rate_recent": _value(a, "win_rate_recent"),
        "away_win_rate_recent": _value(b, "win_rate_recent"),
        "win_rate_diff": win_rate_diff,
        "home_draw_rate_recent": _value(a, "draw_rate_recent"),
        "away_draw_rate_recent": _value(b, "draw_rate_recent"),
        "draw_rate_diff": draw_rate_diff,
        "similar_draw_rate": int(abs(draw_rate_diff) < 0.2),
        "drawish_profile": int(
            abs(elo_diff) < 100
            and abs(goal_diff_recent_diff) < 0.75
            and combined_defense > 150
        ),
        "neutral": 1,
        "home_advantage": 0,
        "tournament_importance": 60,
        "is_world_cup": 1,
        "is_qualifier": 0,
        "is_friendly": 0,
    }
    return pd.DataFrame([features], columns=MODEL_FEATURE_COLUMNS)


def _model_explanation(a: pd.Series, b: pd.Series, team_a: str, team_b: str) -> pd.DataFrame:
    explanation = pd.DataFrame(
        [
            {
                "factor": "Elo rating",
                "team_a_value": a["elo"],
                "team_b_value": b["elo"],
                "impact": (a["elo"] - b["elo"]) / 420,
            },
            {
                "factor": "Recent form",
                "team_a_value": a["form_points"],
                "team_b_value": b["form_points"],
                "impact": (a["form_points"] - b["form_points"]) / 12,
            },
            {
                "factor": "Attack vs defense",
                "team_a_value": a["attack_rating"],
                "team_b_value": b["defense_rating"],
                "impact": (a["attack_rating"] - b["defense_rating"]) / 30,
            },
            {
                "factor": "Defense vs attack",
                "team_a_value": a["defense_rating"],
                "team_b_value": b["attack_rating"],
                "impact": (a["defense_rating"] - b["attack_rating"]) / 30,
            },
        ]
    )
    return (
        explanation.assign(
            favors=np.where(explanation["impact"] > 0, team_a, np.where(explanation["impact"] < 0, team_b, "Even")),
            abs_impact=explanation["impact"].abs(),
        )
        .sort_values("abs_impact", ascending=False)
        .drop(columns=["abs_impact"])
    )


def _predict_with_model(teams: pd.DataFrame, team_a: str, team_b: str) -> MatchPrediction:
    a = _team_row(teams, team_a)
    b = _team_row(teams, team_b)
    artifact = _load_model_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    features = _model_features(a, b)[feature_columns]
    probabilities = dict(zip(model.classes_, model.predict_proba(features)[0]))

    return MatchPrediction(
        team_a=team_a,
        team_b=team_b,
        team_a_win=float(probabilities.get("home_win", 0)),
        draw=float(probabilities.get("draw", 0)),
        team_b_win=float(probabilities.get("away_win", 0)),
        explanation=_model_explanation(a, b, team_a, team_b),
    )


def predict_match(teams: pd.DataFrame, team_a: str, team_b: str) -> MatchPrediction:
    if MODEL_PATH.exists():
        try:
            return _predict_with_model(teams, team_a, team_b)
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            return _fallback_prediction(teams, team_a, team_b)
    return _fallback_prediction(teams, team_a, team_b)
