from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


RAW_MATCHES_PATH = Path("data/international_matches.csv")
TARGET_TEAMS_PATH = Path("data/sample_team_ratings.csv")
OUTPUT_PATH = Path("data/team_ratings.csv")
ALL_OUTPUT_PATH = Path("data/all_team_ratings.csv")

RECENT_MATCHES_FOR_FORM = 5
RECENT_MATCHES_FOR_STRENGTH = 10


def match_importance(tournament: str) -> float:
    tournament = tournament.lower()
    if "fifa world cup" == tournament:
        return 60
    if "world cup qualification" in tournament:
        return 40
    if any(name in tournament for name in ["uefa euro", "copa américa", "african cup", "asian cup", "gold cup", "nations league"]):
        return 35
    if "friendly" in tournament:
        return 20
    return 30


def goal_difference_multiplier(goal_difference: float) -> float:
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8


def expected_score(team_rating: float, opponent_rating: float, home_advantage: float = 0) -> float:
    rating_gap = team_rating + home_advantage - opponent_rating
    return 1 / (1 + 10 ** (-rating_gap / 400))


def actual_score(goals_for: float, goals_against: float) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def load_completed_matches(path: Path) -> pd.DataFrame:
    matches = pd.read_csv(path)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    return matches.sort_values("date")


def calculate_elo_ratings(matches: pd.DataFrame) -> pd.Series:
    ratings = defaultdict(lambda: 1500.0)

    for match in matches.itertuples(index=False):
        home_rating = ratings[match.home_team]
        away_rating = ratings[match.away_team]

        home_advantage = 65 if not match.neutral else 0
        expected_home = expected_score(home_rating, away_rating, home_advantage)
        score_home = actual_score(match.home_score, match.away_score)

        k = match_importance(match.tournament)
        multiplier = goal_difference_multiplier(match.home_score - match.away_score)
        rating_change = k * multiplier * (score_home - expected_home)

        ratings[match.home_team] += rating_change
        ratings[match.away_team] -= rating_change

    return pd.Series(ratings, name="elo").round().astype(int)


def build_team_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    home_rows = matches.rename(
        columns={
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "goals_for",
            "away_score": "goals_against",
        }
    )
    away_rows = matches.rename(
        columns={
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "goals_for",
            "home_score": "goals_against",
        }
    )

    team_matches = pd.concat(
        [
            home_rows[["date", "team", "opponent", "goals_for", "goals_against"]],
            away_rows[["date", "team", "opponent", "goals_for", "goals_against"]],
        ],
        ignore_index=True,
    ).sort_values(["team", "date"])

    team_matches["points"] = team_matches.apply(
        lambda row: 3
        if row["goals_for"] > row["goals_against"]
        else 1
        if row["goals_for"] == row["goals_against"]
        else 0,
        axis=1,
    )
    return team_matches


def scale_attack(goals_for_per_match: float) -> float:
    return min(99, max(50, 55 + goals_for_per_match * 18))


def scale_defense(goals_against_per_match: float) -> float:
    return min(99, max(50, 95 - goals_against_per_match * 18))


def build_recent_strengths(team_matches: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for team, matches in team_matches.groupby("team"):
        recent_form = matches.tail(RECENT_MATCHES_FOR_FORM)
        recent_strength = matches.tail(RECENT_MATCHES_FOR_STRENGTH)

        goals_for = recent_strength["goals_for"].mean()
        goals_against = recent_strength["goals_against"].mean()
        wins = (recent_strength["goals_for"] > recent_strength["goals_against"]).mean()
        draws = (recent_strength["goals_for"] == recent_strength["goals_against"]).mean()

        rows.append(
            {
                "team": team,
                "form_points": int(recent_form["points"].sum()),
                "goals_scored_per_match": round(goals_for, 2),
                "goals_conceded_per_match": round(goals_against, 2),
                "goal_difference_recent": round(goals_for - goals_against, 2),
                "win_rate_recent": round(wins, 3),
                "draw_rate_recent": round(draws, 3),
                "attack_rating": round(scale_attack(goals_for), 1),
                "defense_rating": round(scale_defense(goals_against), 1),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    matches = load_completed_matches(RAW_MATCHES_PATH)
    targets = pd.read_csv(TARGET_TEAMS_PATH)[["team", "group"]]

    elo = calculate_elo_ratings(matches).rename_axis("team").reset_index()
    strengths = build_recent_strengths(build_team_match_rows(matches))

    all_ratings = strengths.merge(elo, on="team", how="left")
    all_ratings["fifa_rank"] = all_ratings["elo"].rank(ascending=False, method="min").astype(int)
    all_ratings["group"] = "TBD"
    all_ratings = all_ratings[
        [
            "team",
            "group",
            "elo",
            "fifa_rank",
            "form_points",
            "attack_rating",
            "defense_rating",
            "goals_scored_per_match",
            "goals_conceded_per_match",
            "goal_difference_recent",
            "win_rate_recent",
            "draw_rate_recent",
        ]
    ].sort_values("team")

    ratings = targets.merge(all_ratings.drop(columns=["group"]), on="team", how="left")
    missing = ratings.loc[ratings["elo"].isna(), "team"].tolist()
    if missing:
        raise ValueError(f"These target teams were not found in match history: {missing}")

    ratings = ratings[
        [
            "team",
            "group",
            "elo",
            "fifa_rank",
            "form_points",
            "attack_rating",
            "defense_rating",
            "goals_scored_per_match",
            "goals_conceded_per_match",
            "goal_difference_recent",
            "win_rate_recent",
            "draw_rate_recent",
        ]
    ].sort_values(["group", "team"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(OUTPUT_PATH, index=False)
    all_ratings.to_csv(ALL_OUTPUT_PATH, index=False)

    print(f"Read {len(matches):,} completed international matches.")
    print(f"Saved {len(ratings):,} team ratings to {OUTPUT_PATH}.")
    print(f"Saved {len(all_ratings):,} all-team ratings to {ALL_OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
