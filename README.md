# World Cup 2026 Match & Tournament Simulator

A Streamlit football analytics app that predicts individual match outcomes and runs Monte Carlo tournament simulations.

## What It Does

- Predicts win/draw/loss probabilities for any two teams.
- Explains predictions using rating, ranking, form, attack, and defense factors.
- Simulates group-stage matches.
- Advances teams using the 2026-style format: top two in each group plus the best third-place teams.
- Builds a seeded Round of 32 bracket from group-stage performance instead of randomly shuffling qualifiers.
- Runs Monte Carlo simulations to estimate each team's chance of reaching each tournament stage.
- Shows champion probabilities in an interactive Streamlit dashboard.

## Project Structure

```text
app/
  streamlit_app.py
data/
  sample_team_ratings.csv
src/
  predictor.py
  simulator.py
EPL_match_predictor.ipynb
matches.csv
requirements.txt
```

## Run The App

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app/streamlit_app.py
```

## Update The Data

Download the latest international match results:

```bash
python3 src/download_international_data.py
```

Build team ratings from completed matches:

```bash
python3 src/build_team_ratings.py
```

This creates `data/team_ratings.csv`, which the Streamlit app uses automatically. The generated ratings include an Elo-style rating, an Elo-derived ranking, recent form points, attack rating, defense rating, goals scored per match, and goals conceded per match.

Train the match prediction model:

```bash
python3 src/train_model.py
```

This creates `models/match_model.joblib` and `models/model_report.txt`. When the model file exists, `src/predictor.py` uses the selected trained model probabilities. If the model file is missing, it falls back to the simpler rating formula.

The training script also writes:

- `models/model_metrics.json` for dashboard metrics.
- `models/calibration_curve.csv` for the reliability chart.
- `models/feature_importance.csv` for model explainability.
- Brier score, ranked probability score, and expected calibration error in the model report.

## How To Make It Portfolio Worthy

Replace `data/sample_team_ratings.csv` with a real dataset containing:

- Team name
- Group
- Elo rating
- FIFA ranking
- Recent form points
- Attack rating
- Defense rating

Then improve the prediction engine in `src/predictor.py` by replacing the current interpretable rating formula with a trained scikit-learn model. Keep the same output shape: team A win probability, draw probability, team B win probability, and explanation data.

Good next upgrades:

- Train on historical international matches.
- Backtest on the 2014, 2018, and 2022 World Cups.
- Add real fixtures and live result updates.
- Add SHAP or permutation importance for stronger model explanations.
- Save trained models in a `models/` folder with `joblib`.
