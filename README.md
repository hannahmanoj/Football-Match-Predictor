# ⚽ PL Match Predictor using scikit-learn
A Premier League Match Predictor that utilizes machine learning (Random Forest Classifier) to predict match results based on past performance and match conditions.
- Prepares Data: Converts categorical features (venue, opponent, match time, etc.) into numerical values.
- Trains Model: Uses a Random Forest Classifier on matches before 2022-01-01.
- Makes Predictions: Predicts match outcomes for games after 2022-01-01.
- Calculates Accuracy: Compares predictions with actual results using precision score.
- Includes Rolling Averages: Uses the last 3 games' performance to improve predictions.
- Merges Home & Away Predictions: Provides a combined match analysis.
