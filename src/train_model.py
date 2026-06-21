from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_team_ratings import (
    actual_score,
    expected_score,
    goal_difference_multiplier,
    load_completed_matches,
    match_importance,
    scale_attack,
    scale_defense,
)


RAW_MATCHES_PATH = Path("data/international_matches.csv")
MODEL_PATH = Path("models/match_model.joblib")
REPORT_PATH = Path("models/model_report.txt")
METRICS_PATH = Path("models/model_metrics.json")
CALIBRATION_PATH = Path("models/calibration_curve.csv")
FEATURE_IMPORTANCE_PATH = Path("models/feature_importance.csv")

MIN_MATCH_DATE = "1990-01-01"
TEST_START_DATE = "2022-01-01"
RECENT_FORM_MATCHES = 5
RECENT_STRENGTH_MATCHES = 10
FEATURE_IMPORTANCE_SAMPLE_SIZE = 1500

FEATURE_COLUMNS = [
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

OUTCOME_ORDER = ["away_win", "draw", "home_win"]


def recent_points(history: list[tuple[int, int]], window: int = RECENT_FORM_MATCHES) -> int:
    recent = history[-window:]
    points = 0
    for goals_for, goals_against in recent:
        if goals_for > goals_against:
            points += 3
        elif goals_for == goals_against:
            points += 1
    return points


def recent_attack(history: list[tuple[int, int]], window: int = RECENT_STRENGTH_MATCHES) -> float:
    recent = history[-window:]
    if not recent:
        return 70.0
    goals_for = np.mean([goals_for for goals_for, _ in recent])
    return round(scale_attack(goals_for), 1)


def recent_defense(history: list[tuple[int, int]], window: int = RECENT_STRENGTH_MATCHES) -> float:
    recent = history[-window:]
    if not recent:
        return 70.0
    goals_against = np.mean([goals_against for _, goals_against in recent])
    return round(scale_defense(goals_against), 1)


def recent_goal_difference(history: list[tuple[int, int]], window: int = RECENT_STRENGTH_MATCHES) -> float:
    recent = history[-window:]
    if not recent:
        return 0.0
    return round(np.mean([goals_for - goals_against for goals_for, goals_against in recent]), 2)


def recent_win_rate(history: list[tuple[int, int]], window: int = RECENT_STRENGTH_MATCHES) -> float:
    recent = history[-window:]
    if not recent:
        return 0.0
    return round(np.mean([goals_for > goals_against for goals_for, goals_against in recent]), 3)


def recent_draw_rate(history: list[tuple[int, int]], window: int = RECENT_STRENGTH_MATCHES) -> float:
    recent = history[-window:]
    if not recent:
        return 0.0
    return round(np.mean([goals_for == goals_against for goals_for, goals_against in recent]), 3)


def result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if home_score == away_score:
        return "draw"
    return "away_win"


def one_hot_targets(targets: pd.Series, classes: np.ndarray) -> np.ndarray:
    class_index = {label: index for index, label in enumerate(classes)}
    encoded = np.zeros((len(targets), len(classes)))
    for row_index, target in enumerate(targets):
        encoded[row_index, class_index[target]] = 1
    return encoded


def brier_score_multiclass(targets: pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> float:
    encoded = one_hot_targets(targets, classes)
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))


def ranked_probability_score(targets: pd.Series, probabilities: np.ndarray, classes: np.ndarray) -> float:
    ordered_indices = [list(classes).index(label) for label in OUTCOME_ORDER if label in classes]
    ordered_probabilities = probabilities[:, ordered_indices]
    ordered_targets = one_hot_targets(targets, np.array(OUTCOME_ORDER))[:, : len(ordered_indices)]
    cumulative_probabilities = np.cumsum(ordered_probabilities, axis=1)
    cumulative_targets = np.cumsum(ordered_targets, axis=1)
    return float(np.mean(np.sum((cumulative_probabilities[:, :-1] - cumulative_targets[:, :-1]) ** 2, axis=1) / (len(ordered_indices) - 1)))


def calibration_curve_data(
    targets: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    encoded = one_hot_targets(targets, classes)
    rows = pd.DataFrame(
        {
            "predicted_probability": probabilities.ravel(),
            "actual": encoded.ravel(),
        }
    )
    rows["bin"] = pd.cut(rows["predicted_probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    calibration = (
        rows.groupby("bin", observed=False)
        .agg(
            mean_predicted_probability=("predicted_probability", "mean"),
            actual_rate=("actual", "mean"),
            count=("actual", "size"),
        )
        .dropna()
        .reset_index(drop=True)
    )
    total = calibration["count"].sum()
    ece = float(
        (
            calibration["count"]
            / total
            * (calibration["actual_rate"] - calibration["mean_predicted_probability"]).abs()
        ).sum()
    )
    return calibration, ece


def feature_importance_data(model, test: pd.DataFrame) -> pd.DataFrame:
    sample_size = min(FEATURE_IMPORTANCE_SAMPLE_SIZE, len(test))
    sample = test.sample(n=sample_size, random_state=42)
    importance = permutation_importance(
        model,
        sample[FEATURE_COLUMNS],
        sample["target"],
        scoring="neg_log_loss",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def tournament_flags(tournament: str) -> dict[str, int]:
    value = tournament.lower()
    return {
        "is_world_cup": int(value == "fifa world cup"),
        "is_qualifier": int("qualification" in value),
        "is_friendly": int("friendly" in value),
    }


def update_elo(
    ratings: defaultdict[str, float],
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool,
) -> None:
    home_rating = ratings[home_team]
    away_rating = ratings[away_team]
    home_advantage = 65 if not neutral else 0
    expected_home = expected_score(home_rating, away_rating, home_advantage)
    score_home = actual_score(home_score, away_score)
    change = (
        match_importance(tournament)
        * goal_difference_multiplier(home_score - away_score)
        * (score_home - expected_home)
    )
    ratings[home_team] += change
    ratings[away_team] -= change


def build_training_rows(matches: pd.DataFrame) -> pd.DataFrame:
    ratings = defaultdict(lambda: 1500.0)
    histories: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    rows = []

    for match in matches.itertuples(index=False):
        home_history = histories[match.home_team]
        away_history = histories[match.away_team]

        home_form = recent_points(home_history)
        away_form = recent_points(away_history)
        home_attack = recent_attack(home_history)
        away_attack = recent_attack(away_history)
        home_defense = recent_defense(home_history)
        away_defense = recent_defense(away_history)
        home_goal_diff = recent_goal_difference(home_history)
        away_goal_diff = recent_goal_difference(away_history)
        home_win_rate = recent_win_rate(home_history)
        away_win_rate = recent_win_rate(away_history)
        home_draw_rate = recent_draw_rate(home_history)
        away_draw_rate = recent_draw_rate(away_history)
        home_advantage = 65 if not match.neutral else 0
        flags = tournament_flags(match.tournament)
        elo_diff = ratings[match.home_team] + home_advantage - ratings[match.away_team]
        form_diff = home_form - away_form
        attack_diff = home_attack - away_defense
        defense_diff = home_defense - away_attack
        combined_attack = home_attack + away_attack
        combined_defense = home_defense + away_defense
        goal_diff_recent_diff = home_goal_diff - away_goal_diff
        draw_rate_diff = home_draw_rate - away_draw_rate

        if match.date >= pd.Timestamp(MIN_MATCH_DATE):
            rows.append(
                {
                    "date": match.date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "target": result_label(match.home_score, match.away_score),
                    "elo_diff": elo_diff,
                    "elo_abs_diff": abs(elo_diff),
                    "is_close_elo_match": int(abs(elo_diff) < 75),
                    "home_elo": ratings[match.home_team],
                    "away_elo": ratings[match.away_team],
                    "form_diff": form_diff,
                    "form_abs_diff": abs(form_diff),
                    "is_close_form_match": int(abs(form_diff) <= 2),
                    "home_form_points": home_form,
                    "away_form_points": away_form,
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
                    "home_attack_rating": home_attack,
                    "away_attack_rating": away_attack,
                    "home_defense_rating": home_defense,
                    "away_defense_rating": away_defense,
                    "home_goal_diff_recent": home_goal_diff,
                    "away_goal_diff_recent": away_goal_diff,
                    "goal_diff_recent_diff": goal_diff_recent_diff,
                    "goal_diff_recent_abs_diff": abs(goal_diff_recent_diff),
                    "similar_recent_goal_difference": int(abs(goal_diff_recent_diff) < 0.5),
                    "home_win_rate_recent": home_win_rate,
                    "away_win_rate_recent": away_win_rate,
                    "win_rate_diff": home_win_rate - away_win_rate,
                    "home_draw_rate_recent": home_draw_rate,
                    "away_draw_rate_recent": away_draw_rate,
                    "draw_rate_diff": draw_rate_diff,
                    "similar_draw_rate": int(abs(draw_rate_diff) < 0.2),
                    "drawish_profile": int(
                        abs(elo_diff) < 100
                        and abs(goal_diff_recent_diff) < 0.75
                        and combined_defense > 150
                    ),
                    "neutral": int(match.neutral),
                    "home_advantage": home_advantage,
                    "tournament_importance": match_importance(match.tournament),
                    **flags,
                }
            )

        update_elo(
            ratings,
            match.home_team,
            match.away_team,
            match.home_score,
            match.away_score,
            match.tournament,
            match.neutral,
        )
        histories[match.home_team].append((match.home_score, match.away_score))
        histories[match.away_team].append((match.away_score, match.home_score))

    return pd.DataFrame(rows)


def main() -> None:
    matches = load_completed_matches(RAW_MATCHES_PATH)
    data = build_training_rows(matches)

    train = data[data["date"] < TEST_START_DATE]
    test = data[data["date"] >= TEST_START_DATE]

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=220,
            max_leaf_nodes=24,
            l2_regularization=0.2,
            random_state=42,
        ),
        "BalancedHistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=220,
            max_leaf_nodes=24,
            l2_regularization=0.2,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []
    fitted_models = {}
    for name, candidate in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            candidate.fit(train[FEATURE_COLUMNS], train["target"])
            predictions = candidate.predict(test[FEATURE_COLUMNS])
            probabilities = candidate.predict_proba(test[FEATURE_COLUMNS])
            accuracy = accuracy_score(test["target"], predictions)
            loss = log_loss(test["target"], probabilities, labels=candidate.classes_)
            brier = brier_score_multiclass(test["target"], probabilities, candidate.classes_)
            rps = ranked_probability_score(test["target"], probabilities, candidate.classes_)
            _, ece = calibration_curve_data(test["target"], probabilities, candidate.classes_)
            class_metrics = classification_report(test["target"], predictions, output_dict=True)
        results.append(
            {
                "model": name,
                "accuracy": accuracy,
                "log_loss": loss,
                "brier_score": brier,
                "rps": rps,
                "ece": ece,
                "draw_precision": class_metrics["draw"]["precision"],
                "draw_recall": class_metrics["draw"]["recall"],
            }
        )
        fitted_models[name] = candidate

    comparison = pd.DataFrame(results).sort_values(["log_loss", "accuracy"], ascending=[True, False])
    best_name = str(comparison.iloc[0]["model"])
    model = fitted_models[best_name]
    if isinstance(model, RandomForestClassifier):
        model.n_jobs = 1

    predictions = model.predict(test[FEATURE_COLUMNS])
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])
    accuracy = accuracy_score(test["target"], predictions)
    loss = log_loss(test["target"], probabilities, labels=model.classes_)
    brier = brier_score_multiclass(test["target"], probabilities, model.classes_)
    rps = ranked_probability_score(test["target"], probabilities, model.classes_)
    calibration, ece = calibration_curve_data(test["target"], probabilities, model.classes_)
    feature_importance = feature_importance_data(model, test)
    report = classification_report(test["target"], predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(CALIBRATION_PATH, index=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "training_rows": len(train),
                "test_rows": len(test),
                "test_start_date": TEST_START_DATE,
                "accuracy": accuracy,
                "log_loss": loss,
                "brier_score": brier,
                "ranked_probability_score": rps,
                "expected_calibration_error": ece,
                "feature_importance_rows": len(feature_importance),
                "model_comparison": comparison.to_dict(orient="records"),
            },
            indent=2,
        )
    )
    joblib.dump(
        {
            "model": model,
            "model_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "classes": model.classes_.tolist(),
            "trained_on_rows": len(train),
            "tested_on_rows": len(test),
            "test_start_date": TEST_START_DATE,
            "accuracy": accuracy,
            "log_loss": loss,
            "brier_score": brier,
            "ranked_probability_score": rps,
            "expected_calibration_error": ece,
        },
        MODEL_PATH,
    )

    REPORT_PATH.write_text(
        "\n".join(
            [
                "International Match Result Model",
                f"Selected model: {best_name}",
                f"Training rows: {len(train):,}",
                f"Test rows: {len(test):,}",
                f"Test start date: {TEST_START_DATE}",
                f"Accuracy: {accuracy:.3f}",
                f"Log loss: {loss:.3f}",
                f"Brier score: {brier:.3f}",
                f"Ranked probability score: {rps:.3f}",
                f"Expected calibration error: {ece:.3%}",
                "",
                "Top feature importances:",
                feature_importance.head(15).to_string(
                    index=False,
                    formatters={
                        "importance_mean": "{:.5f}".format,
                        "importance_std": "{:.5f}".format,
                    },
                ),
                "",
                "Model comparison:",
                comparison.to_string(
                    index=False,
                    formatters={
                        "accuracy": "{:.3f}".format,
                        "log_loss": "{:.3f}".format,
                        "brier_score": "{:.3f}".format,
                        "rps": "{:.3f}".format,
                        "ece": "{:.3%}".format,
                        "draw_precision": "{:.3f}".format,
                        "draw_recall": "{:.3f}".format,
                    },
                ),
                "",
                report,
            ]
        )
    )

    print(f"Built {len(data):,} training examples from completed matches.")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "accuracy": "{:.3f}".format,
                "log_loss": "{:.3f}".format,
                "brier_score": "{:.3f}".format,
                "rps": "{:.3f}".format,
                "ece": "{:.3%}".format,
                "draw_precision": "{:.3f}".format,
                "draw_recall": "{:.3f}".format,
            },
        )
    )
    print(f"Selected model: {best_name}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Log loss: {loss:.3f}")
    print(f"Brier score: {brier:.3f}")
    print(f"Ranked probability score: {rps:.3f}")
    print(f"Expected calibration error: {ece:.3%}")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved calibration curve to {CALIBRATION_PATH}")
    print(f"Saved feature importance to {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
