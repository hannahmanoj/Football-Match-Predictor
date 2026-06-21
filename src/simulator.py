from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from .predictor import predict_match
except ImportError:
    from predictor import predict_match


def group_fixtures(teams: pd.DataFrame) -> list[tuple[str, str, str]]:
    fixtures = []
    for group, group_df in teams.groupby("group", sort=True):
        names = group_df["team"].tolist()
        for i, team_a in enumerate(names):
            for team_b in names[i + 1 :]:
                fixtures.append((group, team_a, team_b))
    return fixtures


def cached_prediction(
    teams: pd.DataFrame,
    team_a: str,
    team_b: str,
    prediction_cache: dict[tuple[str, str], object],
):
    key = (team_a, team_b)
    if key not in prediction_cache:
        prediction_cache[key] = predict_match(teams, team_a, team_b)
    return prediction_cache[key]


def simulate_match(
    teams: pd.DataFrame,
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
    prediction_cache: dict[tuple[str, str], object] | None = None,
) -> tuple[str, int, int]:
    if prediction_cache is None:
        prediction_cache = {}

    prediction = cached_prediction(teams, team_a, team_b, prediction_cache)
    outcome = rng.choice(
        ["a", "draw", "b"],
        p=[prediction.team_a_win, prediction.draw, prediction.team_b_win],
    )

    if outcome == "a":
        return team_a, int(rng.integers(1, 4)), int(rng.integers(0, 2))
    if outcome == "b":
        return team_b, int(rng.integers(0, 2)), int(rng.integers(1, 4))
    goals = int(rng.integers(0, 3))
    return "Draw", goals, goals


def simulate_group_stage(
    teams: pd.DataFrame,
    rng: np.random.Generator,
    prediction_cache: dict[tuple[str, str], object] | None = None,
) -> pd.DataFrame:
    if prediction_cache is None:
        prediction_cache = {}

    table = {
        team: {
            "team": team,
            "group": row["group"],
            "points": 0,
            "goal_difference": 0,
            "goals_for": 0,
        }
        for _, row in teams.iterrows()
        for team in [row["team"]]
    }

    for _, team_a, team_b in group_fixtures(teams):
        winner, goals_a, goals_b = simulate_match(teams, team_a, team_b, rng, prediction_cache)
        table[team_a]["goals_for"] += goals_a
        table[team_b]["goals_for"] += goals_b
        table[team_a]["goal_difference"] += goals_a - goals_b
        table[team_b]["goal_difference"] += goals_b - goals_a

        if winner == "Draw":
            table[team_a]["points"] += 1
            table[team_b]["points"] += 1
        elif winner == team_a:
            table[team_a]["points"] += 3
        else:
            table[team_b]["points"] += 3

    standings = pd.DataFrame(table.values())
    return standings.sort_values(
        ["group", "points", "goal_difference", "goals_for"],
        ascending=[True, False, False, False],
    )


def qualified_teams(standings: pd.DataFrame) -> pd.DataFrame:
    qualified_rows = []
    third_place = []

    for _, group_df in standings.groupby("group", sort=True):
        ordered = group_df.sort_values(
            ["points", "goal_difference", "goals_for"],
            ascending=False,
        ).reset_index(drop=True)

        for position, row in ordered.head(2).iterrows():
            qualified_rows.append({**row.to_dict(), "group_position": position + 1})

        third_row = ordered.iloc[2].to_dict()
        third_place.append({**third_row, "group_position": 3})

    best_thirds = (
        pd.DataFrame(third_place)
        .sort_values(["points", "goal_difference", "goals_for"], ascending=False)
        .head(8)
    )
    qualified_rows.extend(best_thirds.to_dict("records"))

    qualifiers = pd.DataFrame(qualified_rows)
    return qualifiers.sort_values(
        ["group_position", "points", "goal_difference", "goals_for"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def group_qualifiers(standings: pd.DataFrame) -> list[str]:
    return qualified_teams(standings)["team"].tolist()


def build_round_of_32(qualifiers: pd.DataFrame) -> list[str]:
    seeded = qualifiers.sort_values(
        ["group_position", "points", "goal_difference", "goals_for"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    top_seeds = seeded.head(16).to_dict("records")
    lower_seeds = seeded.tail(16).sort_values(
        ["group_position", "points", "goal_difference", "goals_for"],
        ascending=[False, True, True, True],
    ).to_dict("records")

    bracket = []
    available = lower_seeds.copy()
    for seed in top_seeds:
        opponent_index = next(
            (index for index, candidate in enumerate(available) if candidate["group"] != seed["group"]),
            0,
        )
        opponent = available.pop(opponent_index)
        bracket.extend([seed["team"], opponent["team"]])

    return bracket


def simulate_knockout(
    teams: pd.DataFrame,
    qualifiers: pd.DataFrame | list[str],
    rng: np.random.Generator,
    prediction_cache: dict[tuple[str, str], object] | None = None,
) -> dict[str, list[str] | str]:
    if prediction_cache is None:
        prediction_cache = {}

    if isinstance(qualifiers, pd.DataFrame):
        current_round = build_round_of_32(qualifiers)
    else:
        current_round = qualifiers.copy()
    stages = {"round_of_32": current_round.copy()}

    stage_names = ["round_of_16", "quarterfinal", "semifinal", "final", "champion"]
    for stage_name in stage_names:
        winners = []
        for i in range(0, len(current_round), 2):
            team_a = current_round[i]
            team_b = current_round[i + 1]
            winner, _, _ = simulate_match(teams, team_a, team_b, rng, prediction_cache)
            if winner == "Draw":
                prediction = cached_prediction(teams, team_a, team_b, prediction_cache)
                winner = team_a if prediction.team_a_win >= prediction.team_b_win else team_b
            winners.append(winner)

        if stage_name == "champion":
            stages[stage_name] = winners[0]
        else:
            stages[stage_name] = winners.copy()
        current_round = winners

    return stages


def monte_carlo_tournament(teams: pd.DataFrame, simulations: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = defaultdict(lambda: defaultdict(int))
    prediction_cache = {}

    for _ in range(simulations):
        standings = simulate_group_stage(teams, rng, prediction_cache)
        qualifiers = qualified_teams(standings)
        stages = simulate_knockout(teams, qualifiers, rng, prediction_cache)

        for team in qualifiers["team"]:
            counts[team]["round_of_32"] += 1
        for stage in ["round_of_16", "quarterfinal", "semifinal", "final"]:
            for team in stages[stage]:
                counts[team][stage] += 1
        counts[stages["champion"]]["champion"] += 1

    rows = []
    for team in teams["team"]:
        row = {"team": team}
        for stage in ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final", "champion"]:
            row[stage] = counts[team][stage] / simulations
        rows.append(row)

    return pd.DataFrame(rows).sort_values("champion", ascending=False)
