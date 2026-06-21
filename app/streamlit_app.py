from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import base64
from datetime import date
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from predictor import load_teams, predict_match
import simulator

simulator = importlib.reload(simulator)

build_round_of_32 = simulator.build_round_of_32
monte_carlo_tournament = simulator.monte_carlo_tournament
qualified_teams = simulator.qualified_teams
simulate_group_stage = simulator.simulate_group_stage


GENERATED_DATA_PATH = ROOT / "data" / "team_ratings.csv"
ALL_RATINGS_PATH = ROOT / "data" / "all_team_ratings.csv"
SAMPLE_DATA_PATH = ROOT / "data" / "sample_team_ratings.csv"
MODEL_REPORT_PATH = ROOT / "models" / "model_report.txt"
MODEL_METRICS_PATH = ROOT / "models" / "model_metrics.json"
CALIBRATION_PATH = ROOT / "models" / "calibration_curve.csv"
FEATURE_IMPORTANCE_PATH = ROOT / "models" / "feature_importance.csv"
HERO_IMAGE_PATH = ROOT / "app" / "assets" / "football.jpg"
MATCHES_PATH = ROOT / "data" / "international_matches.csv"


st.set_page_config(
    page_title="World Cup 2026 Simulator",
    page_icon="⚽",
    layout="wide",
)

def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


hero_background = image_data_uri(HERO_IMAGE_PATH) if HERO_IMAGE_PATH.exists() else ""

st.markdown(
    """
    <style>
        :root {
            --ink: #17201d;
            --muted: #5d6b65;
            --line: #dfe7e2;
            --surface: #ffffff;
            --surface-soft: #f6f9f7;
            --green: #167a55;
            --teal: #157b84;
            --red: #b43d4a;
            --gold: #b78020;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f7faf8 0%, #eef5f1 42%, #f8faf8 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        div[data-testid="stTabs"] button {
            border-radius: 6px 6px 0 0;
            padding: 0.7rem 1rem;
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--green);
            border-bottom-color: var(--green);
        }

        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            box-shadow: 0 6px 18px rgba(21, 52, 39, 0.06);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-weight: 650;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 22px rgba(21, 52, 39, 0.05);
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--line);
            background: var(--surface);
            color: var(--ink);
            font-weight: 800;
            white-space: nowrap;
            box-shadow: 0 6px 16px rgba(21, 52, 39, 0.06);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            border-color: var(--green);
            box-shadow: 0 10px 24px rgba(21, 52, 39, 0.12);
        }

        .app-header {
            min-height: 390px;
            background-image:
                url("__HERO_BACKGROUND__");
            background-size: cover;
            background-position: center;
            border-radius: 8px;
            padding: clamp(1.3rem, 4vw, 3rem);
            box-shadow: 0 18px 42px rgba(21, 52, 39, 0.18);
            margin-bottom: 1.1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }

        .app-header .app-title {
            font-size: clamp(2rem, 3.35vw, 3.05rem);
            line-height: 1.04;
            font-weight: 800;
            margin: 0;
            max-width: none;
            color: #ffffff !important;
            white-space: nowrap;
            text-shadow: 0 3px 22px rgba(0, 0, 0, 0.78);
            text-align: center;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1.02rem;
            margin: 0;
            max-width: 760px;
        }

        .hero-copy {
            max-width: 980px;
            background: transparent;
            border: 0;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
            width: 100%;
            display: flex;
            justify-content: center;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.95rem 0 1.1rem 0;
        }

        .summary-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.58);
            border-radius: 8px;
            padding: 0.85rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            backdrop-filter: blur(10px);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .summary-card:hover,
        .match-card:hover,
        .team-card:hover,
        .prob-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(21, 52, 39, 0.12);
        }

        .app-header .summary-grid {
            max-width: 760px;
            margin-bottom: 0;
        }

        .summary-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }

        .summary-value {
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 800;
        }

        .prob-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.65rem;
            box-shadow: 0 6px 18px rgba(21, 52, 39, 0.05);
        }

        .prob-row {
            display: grid;
            grid-template-columns: minmax(120px, 1.2fr) 4fr 70px;
            align-items: center;
            gap: 0.8rem;
        }

        .prob-label {
            font-weight: 750;
            color: var(--ink);
            overflow-wrap: anywhere;
        }

        .prob-track {
            height: 12px;
            background: #e6eee9;
            border-radius: 999px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
        }

        .prob-value {
            font-weight: 800;
            text-align: right;
            color: var(--ink);
        }

        .today-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0 1.1rem 0;
            box-shadow: 0 10px 26px rgba(21, 52, 39, 0.07);
        }

        .today-header {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: baseline;
            margin-bottom: 0.8rem;
        }

        .today-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--ink);
        }

        .today-date {
            color: var(--muted);
            font-weight: 650;
        }

        .match-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .match-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.9rem;
            background: var(--surface-soft);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .match-meta {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 650;
            margin-bottom: 0.45rem;
        }

        .match-teams {
            color: var(--ink);
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .match-pick {
            color: var(--green);
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .mini-probs {
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 650;
        }

        .team-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.9rem 0 1rem 0;
        }

        .team-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 8px 22px rgba(21, 52, 39, 0.05);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .team-name {
            color: var(--ink);
            font-size: 1.1rem;
            font-weight: 850;
            margin-bottom: 0.65rem;
        }

        .flag {
            display: inline-block;
            margin-right: 0.35rem;
            filter: saturate(1.05);
        }

        .stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.5rem;
        }

        .stat-box {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 0.55rem;
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.7rem;
            font-weight: 750;
            text-transform: uppercase;
        }

        .stat-value {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 850;
        }

        .confidence-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 0.35rem 0.65rem;
            margin: 0.2rem 0 0.75rem 0;
            color: #ffffff;
            font-weight: 800;
            background: var(--teal);
        }

        @media (max-width: 760px) {
            .summary-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .app-header {
                min-height: 470px;
                background-position: center right;
            }

            .app-header .app-title {
                font-size: 2rem;
                white-space: normal;
            }

            .prob-row {
                grid-template-columns: 1fr;
                gap: 0.45rem;
            }

            .prob-value {
                text-align: left;
            }

            .today-header {
                display: block;
            }

            .match-grid {
                grid-template-columns: 1fr;
            }

            .team-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """.replace("__HERO_BACKGROUND__", hero_background),
    unsafe_allow_html=True,
)


@st.cache_data
def get_teams() -> pd.DataFrame:
    data_path = GENERATED_DATA_PATH if GENERATED_DATA_PATH.exists() else SAMPLE_DATA_PATH
    return load_teams(data_path)


@st.cache_data
def get_prediction_teams() -> pd.DataFrame:
    data_path = ALL_RATINGS_PATH if ALL_RATINGS_PATH.exists() else GENERATED_DATA_PATH
    if not data_path.exists():
        data_path = SAMPLE_DATA_PATH
    return load_teams(data_path)


@st.cache_data
def run_simulations(simulations: int, seed: int) -> pd.DataFrame:
    return monte_carlo_tournament(get_teams(), simulations=simulations, seed=seed)


@st.cache_data
def get_model_metrics() -> dict:
    if not MODEL_METRICS_PATH.exists():
        return {}
    return json.loads(MODEL_METRICS_PATH.read_text())


@st.cache_data
def get_calibration_curve() -> pd.DataFrame:
    if not CALIBRATION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CALIBRATION_PATH)


@st.cache_data
def get_model_report() -> str:
    if not MODEL_REPORT_PATH.exists():
        return ""
    return MODEL_REPORT_PATH.read_text()


@st.cache_data
def get_feature_importance() -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data
def get_international_matches() -> pd.DataFrame:
    if not MATCHES_PATH.exists():
        return pd.DataFrame()
    matches = pd.read_csv(MATCHES_PATH)
    matches["date"] = pd.to_datetime(matches["date"]).dt.date
    return matches


def metric_grid(items: list[tuple[str, str]]) -> None:
    st.markdown(metric_grid_html(items), unsafe_allow_html=True)


def metric_grid_html(items: list[tuple[str, str]]) -> str:
    cards = "".join(
        f'<div class="summary-card">'
        f'<div class="summary-label">{label}</div>'
        f'<div class="summary-value">{value}</div>'
        f'</div>'
        for label, value in items
    )
    return f'<div class="summary-grid">{cards}</div>'


def probability_panel(rows: list[tuple[str, float, str]]) -> None:
    html_rows = []
    for label, probability, color in rows:
        percent = probability * 100
        html_rows.append(
            f'<div class="prob-card">'
            f'<div class="prob-row">'
            f'<div class="prob-label">{label}</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width: {percent:.1f}%; background: {color};"></div>'
            f'</div>'
            f'<div class="prob-value">{percent:.1f}%</div>'
            f'</div>'
            f'</div>'
        )
    st.markdown("".join(html_rows), unsafe_allow_html=True)


TEAM_COUNTRY_CODES = {
    "Algeria": "DZ",
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Bolivia": "BO",
    "Brazil": "BR",
    "Cameroon": "CM",
    "Canada": "CA",
    "Cape Verde": "CV",
    "Chile": "CL",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Denmark": "DK",
    "DR Congo": "CD",
    "Ecuador": "EC",
    "Egypt": "EG",
    "England": "GB",
    "France": "FR",
    "Germany": "DE",
    "Ghana": "GH",
    "Honduras": "HN",
    "Iran": "IR",
    "Iraq": "IQ",
    "Italy": "IT",
    "Jamaica": "JM",
    "Japan": "JP",
    "Mexico": "MX",
    "Morocco": "MA",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Nigeria": "NG",
    "Panama": "PA",
    "Poland": "PL",
    "Portugal": "PT",
    "Qatar": "QA",
    "Saudi Arabia": "SA",
    "Senegal": "SN",
    "Serbia": "RS",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Switzerland": "CH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United States": "US",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
}


def flag_emoji(team: str) -> str:
    country_code = TEAM_COUNTRY_CODES.get(team)
    if not country_code:
        return ""
    return "".join(chr(127397 + ord(letter)) for letter in country_code.upper())


def team_label(team: str) -> str:
    flag = flag_emoji(team)
    return f"{flag} {team}" if flag else team


def team_html(team: str) -> str:
    flag = flag_emoji(team)
    if not flag:
        return escape(str(team))
    return f'<span class="flag">{flag}</span>{escape(str(team))}'


def confidence_text(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "Strong lean", "var(--green)"
    if probability >= 0.5:
        return "Moderate lean", "var(--teal)"
    if probability >= 0.4:
        return "Open match", "var(--gold)"
    return "Very tight", "var(--red)"


def team_card(row: pd.Series) -> str:
    stats = [
        ("Elo", f"{row['elo']:.0f}"),
        ("Rank", f"{row['fifa_rank']:.0f}"),
        ("Form", f"{row['form_points']:.0f}"),
        ("Attack", f"{row['attack_rating']:.1f}"),
        ("Defense", f"{row['defense_rating']:.1f}"),
        ("GD", f"{row.get('goal_difference_recent', 0):+.2f}"),
    ]
    stat_html = "".join(
        f'<div class="stat-box"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'
        for label, value in stats
    )
    return (
        f'<div class="team-card">'
        f'<div class="team-name">{team_html(str(row["team"]))}</div>'
        f'<div class="stat-grid">{stat_html}</div>'
        f'</div>'
    )


def team_snapshot(teams: pd.DataFrame, team_a: str, team_b: str) -> None:
    a = teams.loc[teams["team"] == team_a].iloc[0]
    b = teams.loc[teams["team"] == team_b].iloc[0]
    st.markdown(f'<div class="team-grid">{team_card(a)}{team_card(b)}</div>', unsafe_allow_html=True)


def swap_selected_teams() -> None:
    team_a = st.session_state.get("team_a_select")
    team_b = st.session_state.get("team_b_select")
    st.session_state["team_a_select"] = team_b
    st.session_state["team_b_select"] = team_a


def todays_match_rows(prediction_teams: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, date | None]:
    if matches.empty:
        return pd.DataFrame(), None

    world_cup = matches[
        (matches["tournament"] == "FIFA World Cup")
        & (matches["home_score"].isna())
        & (matches["away_score"].isna())
    ].copy()
    if world_cup.empty:
        return pd.DataFrame(), None

    today = date.today()
    todays_matches = world_cup[world_cup["date"] == today]
    match_date = today

    if todays_matches.empty:
        future_matches = world_cup[world_cup["date"] > today].sort_values("date")
        if future_matches.empty:
            return pd.DataFrame(), None
        match_date = future_matches.iloc[0]["date"]
        todays_matches = future_matches[future_matches["date"] == match_date]

    rows = []
    available_teams = set(prediction_teams["team"])
    for match in todays_matches.itertuples(index=False):
        home_team = match.home_team
        away_team = match.away_team
        kickoff = getattr(match, "time", "TBD") if hasattr(match, "time") else "TBD"
        city = getattr(match, "city", "")
        country = getattr(match, "country", "")

        if home_team in available_teams and away_team in available_teams:
            prediction = predict_match(prediction_teams, home_team, away_team)
            probabilities = {
                home_team: prediction.team_a_win,
                "Draw": prediction.draw,
                away_team: prediction.team_b_win,
            }
            pick, pick_probability = max(probabilities.items(), key=lambda item: item[1])
            prediction_text = f"{team_label(pick) if pick != 'Draw' else pick} {pick_probability:.1%}"
            probability_text = (
                f"{team_label(home_team)} {prediction.team_a_win:.0%} | "
                f"Draw {prediction.draw:.0%} | "
                f"{team_label(away_team)} {prediction.team_b_win:.0%}"
            )
        else:
            missing = sorted({home_team, away_team} - available_teams)
            prediction_text = "Prediction unavailable"
            probability_text = f"Missing ratings: {', '.join(missing)}"

        rows.append(
            {
                "kickoff": kickoff,
                "home_team": home_team,
                "away_team": away_team,
                "location": ", ".join(part for part in [city, country] if part),
                "prediction": prediction_text,
                "probabilities": probability_text,
            }
        )

    return pd.DataFrame(rows), match_date


def todays_matches_panel(prediction_teams: pd.DataFrame) -> None:
    rows, match_date = todays_match_rows(prediction_teams, get_international_matches())
    if rows.empty or match_date is None:
        return

    cards = []
    for row in rows.itertuples(index=False):
        cards.append(
            f'<div class="match-card">'
            f'<div class="match-meta">{escape(str(row.kickoff))} · {escape(str(row.location))}</div>'
            f'<div class="match-teams">{team_html(str(row.home_team))} vs {team_html(str(row.away_team))}</div>'
            f'<div class="match-pick">{escape(str(row.prediction))}</div>'
            f'<div class="mini-probs">{escape(str(row.probabilities))}</div>'
            f'</div>'
        )

    panel_title = "Today's Matches" if match_date == date.today() else "Next Matchday"
    st.markdown(
        f'<section class="today-panel">'
        f'<div class="today-header">'
        f'<div class="today-title">{panel_title}</div>'
        f'<div class="today-date">{match_date.strftime("%B %d, %Y")}</div>'
        f'</div>'
        f'<div class="match-grid">{"".join(cards)}</div>'
        f'</section>',
        unsafe_allow_html=True,
    )

    options = [f"{team_label(row.home_team)} vs {team_label(row.away_team)}" for row in rows.itertuples(index=False)]
    selected = st.selectbox("Inspect fixture", options, key="today_fixture")
    selected_row = rows.iloc[options.index(selected)]
    home_team = selected_row["home_team"]
    away_team = selected_row["away_team"]
    if home_team in set(prediction_teams["team"]) and away_team in set(prediction_teams["team"]):
        prediction = predict_match(prediction_teams, home_team, away_team)
        probability_panel(
            [
                (f"{team_label(home_team)} win", prediction.team_a_win, "var(--green)"),
                ("Draw", prediction.draw, "var(--gold)"),
                (f"{team_label(away_team)} win", prediction.team_b_win, "var(--red)"),
            ]
        )


teams = get_teams()
prediction_teams = get_prediction_teams()
team_names = teams["team"].tolist()

metrics = get_model_metrics()
selected_model = metrics.get("selected_model", "Formula fallback")
accuracy_label = f"{metrics['accuracy']:.3f}" if metrics else "N/A"

st.markdown(
    '<section class="app-header">'
    '<div class="hero-copy">'
    '<div class="app-title">World Cup 2026 Match & Tournament Simulator</div>'
    '</div>'
    '</section>',
    unsafe_allow_html=True,
)

st.caption("Machine-learning match probabilities, calibrated model diagnostics, and seeded Monte Carlo tournament paths.")
metric_grid(
    [
        ("Teams", f"{len(teams):,}"),
        ("Groups", f"{teams['group'].nunique():,}"),
        ("Model", selected_model),
        ("Accuracy", accuracy_label),
    ]
)

todays_matches_panel(prediction_teams)

tab_match, tab_tournament, tab_performance, tab_data = st.tabs(
    ["Match Predictor", "Tournament Simulator", "Model Performance", "Team Data"]
)

with tab_match:
    st.subheader("Match Predictor")
    if "team_a_select" not in st.session_state:
        st.session_state["team_a_select"] = team_names[0]
    if "team_b_select" not in st.session_state:
        st.session_state["team_b_select"] = team_names[1 if len(team_names) > 1 else 0]

    left, middle, right = st.columns([1, 0.24, 1])
    with left:
        team_a = st.selectbox("Team A", team_names, key="team_a_select", format_func=team_label)
    with middle:
        st.write("")
        st.button("Swap", use_container_width=True, on_click=swap_selected_teams)
    with right:
        team_b = st.selectbox("Team B", team_names, key="team_b_select", format_func=team_label)

    if team_a == team_b:
        st.warning("Choose two different teams.")
    else:
        team_snapshot(teams, team_a, team_b)
        prediction = predict_match(teams, team_a, team_b)
        outcomes = {
            f"{team_label(team_a)} win": prediction.team_a_win,
            "Draw": prediction.draw,
            f"{team_label(team_b)} win": prediction.team_b_win,
        }
        favorite, favorite_probability = max(outcomes.items(), key=lambda item: item[1])
        confidence, confidence_color = confidence_text(favorite_probability)
        probs = pd.DataFrame(
            {
                "Outcome": [f"{team_a} win", "Draw", f"{team_b} win"],
                "Probability": [prediction.team_a_win, prediction.draw, prediction.team_b_win],
            }
        )

        st.subheader("Prediction")
        st.markdown(
            f'<span class="confidence-pill" style="background: {confidence_color};">{confidence}: {favorite} ({favorite_probability:.1%})</span>',
            unsafe_allow_html=True,
        )
        probability_panel(
            [
                (f"{team_label(team_a)} win", prediction.team_a_win, "var(--green)"),
                ("Draw", prediction.draw, "var(--gold)"),
                (f"{team_label(team_b)} win", prediction.team_b_win, "var(--red)"),
            ]
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team_label(team_a)} win", f"{prediction.team_a_win:.1%}")
        c2.metric("Draw", f"{prediction.draw:.1%}")
        c3.metric(f"{team_label(team_b)} win", f"{prediction.team_b_win:.1%}")

        st.subheader("Why the model thinks this")
        explanation = prediction.explanation.copy()
        explanation["impact"] = explanation["impact"].map(lambda value: f"{value:+.2f}")
        st.dataframe(explanation, use_container_width=True, hide_index=True)

with tab_tournament:
    st.subheader("Monte Carlo Tournament Simulation")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    simulations = col_a.slider("Simulations", min_value=100, max_value=2000, value=100, step=100)
    seed = col_b.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)
    stage_to_chart = col_c.selectbox(
        "Chart stage",
        ["champion", "final", "semifinal", "quarterfinal", "round_of_16", "round_of_32"],
        format_func=lambda value: value.replace("_", " ").title(),
    )

    results = run_simulations(simulations, seed)

    leader = results.iloc[0]
    metric_grid(
        [
            ("Top Champion", team_label(str(leader["team"]))),
            ("Title Chance", f"{leader['champion']:.1%}"),
            ("Simulations", f"{simulations:,}"),
            ("Seed", f"{seed:,}"),
        ]
    )

    st.dataframe(
        results.assign(
            round_of_32=lambda df: df["round_of_32"].map("{:.1%}".format),
            round_of_16=lambda df: df["round_of_16"].map("{:.1%}".format),
            quarterfinal=lambda df: df["quarterfinal"].map("{:.1%}".format),
            semifinal=lambda df: df["semifinal"].map("{:.1%}".format),
            final=lambda df: df["final"].map("{:.1%}".format),
            champion=lambda df: df["champion"].map("{:.1%}".format),
        ),
        use_container_width=True,
        hide_index=True,
    )

    top_n = st.slider("Teams shown in chart", min_value=6, max_value=24, value=12, step=2)
    top_chances = results.sort_values(stage_to_chart, ascending=False).head(top_n)[["team", stage_to_chart]].copy()
    top_chances["team"] = top_chances["team"].map(team_label)
    st.subheader(stage_to_chart.replace("_", " ").title())
    st.bar_chart(top_chances, x="team", y=stage_to_chart, height=360)

    with st.expander("Example simulated group table"):
        example_standings = simulate_group_stage(teams, np.random.default_rng(seed))
        st.dataframe(example_standings, use_container_width=True, hide_index=True)

    with st.expander("Example Round of 32 bracket"):
        example_qualifiers = qualified_teams(example_standings)
        bracket = build_round_of_32(example_qualifiers)
        pairings = pd.DataFrame(
            [
                {"match": index // 2 + 1, "team_a": team_label(bracket[index]), "team_b": team_label(bracket[index + 1])}
                for index in range(0, len(bracket), 2)
            ]
        )
        st.dataframe(pairings, use_container_width=True, hide_index=True)

with tab_performance:
    metrics = get_model_metrics()
    calibration = get_calibration_curve()
    feature_importance = get_feature_importance()
    report = get_model_report()

    if not metrics:
        st.warning("Train the model with `python3 src/train_model.py` to generate performance metrics.")
    else:
        st.subheader("Selected Model")
        metric_grid(
            [
                ("Model", metrics["selected_model"]),
                ("Accuracy", f"{metrics['accuracy']:.3f}"),
                ("Log Loss", f"{metrics['log_loss']:.3f}"),
                ("Brier", f"{metrics['brier_score']:.3f}"),
            ]
        )
        metric_grid(
            [
                ("RPS", f"{metrics['ranked_probability_score']:.3f}"),
                ("ECE", f"{metrics['expected_calibration_error']:.2%}"),
                ("Training Rows", f"{metrics['training_rows']:,}"),
                ("Test Rows", f"{metrics['test_rows']:,}"),
            ]
        )

        comparison = pd.DataFrame(metrics["model_comparison"])
        st.subheader("Model Comparison")
        st.dataframe(
            comparison.assign(
                accuracy=lambda df: df["accuracy"].map("{:.3f}".format),
                log_loss=lambda df: df["log_loss"].map("{:.3f}".format),
                brier_score=lambda df: df["brier_score"].map("{:.3f}".format),
                rps=lambda df: df["rps"].map("{:.3f}".format),
                ece=lambda df: df["ece"].map("{:.2%}".format),
                draw_precision=lambda df: df["draw_precision"].map("{:.3f}".format),
                draw_recall=lambda df: df["draw_recall"].map("{:.3f}".format),
            ),
            use_container_width=True,
            hide_index=True,
        )

        if not calibration.empty:
            st.subheader("Calibration Curve")
            chart_data = calibration[
                ["mean_predicted_probability", "actual_rate"]
            ].rename(
                columns={
                    "mean_predicted_probability": "Predicted",
                    "actual_rate": "Actual",
                }
            )
            st.line_chart(chart_data, height=320)
            st.dataframe(
                calibration.assign(
                    mean_predicted_probability=lambda df: df["mean_predicted_probability"].map("{:.2%}".format),
                    actual_rate=lambda df: df["actual_rate"].map("{:.2%}".format),
                ),
                use_container_width=True,
                hide_index=True,
            )

        if not feature_importance.empty:
            st.subheader("Feature Importance")
            top_features = feature_importance.head(15)
            st.bar_chart(top_features, x="feature", y="importance_mean", height=360)
            st.dataframe(
                top_features.assign(
                    importance_mean=lambda df: df["importance_mean"].map("{:.5f}".format),
                    importance_std=lambda df: df["importance_std"].map("{:.5f}".format),
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Full model report"):
            st.code(report)

with tab_data:
    st.subheader("Editable Team Ratings")
    st.dataframe(teams, use_container_width=True, hide_index=True)
    st.markdown(
        "Run `python3 src/build_team_ratings.py` after updating `data/international_matches.csv`. "
        "Edit `data/sample_team_ratings.csv` only when you want to change the tournament teams or groups."
    )
