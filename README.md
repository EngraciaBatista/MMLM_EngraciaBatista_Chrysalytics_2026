# MMLM_EngraciaBatista_Chrysalytics_2026

A machine learning solution for the [2026 NCAA March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview) Kaggle competition. The goal is to predict the outcome of every possible Men's and Women's NCAA tournament matchup, evaluated by Brier score (lower is better).

## Approach

This solution uses a **combined Men's + Women's margin regression model** trained on NCAA tournament data from 2003 to 2026. Rather than predicting win/loss directly, the model predicts the point differential between teams, which is then converted to a win probability using a quintic spline calibrator fitted on out-of-fold predictions.

Key design decisions:

- **LightGBM regressor** predicting point margin, converted to win probability via quintic spline calibration
- **Combined M+W model** — Men's and Women's data trained together (with a `men_women` flag), doubling the calibration data available to the spline
- **11-fold leave-one-season-out cross-validation** (seasons 2012–2023, skipping 2020) for robust OOF calibration
- **Ensemble** of 5 fold models averaged at inference time to reduce variance
- **Rich feature set** including SOS-adjusted efficiency ratings, Elo with margin-of-victory, Bradley-Terry strength ratings, margin quality (Ridge regression), and box score differentials (rebounds, blocks, fouls)
- **Post-processing**: 10% confidence boost for predictions below 0.85, clipped to [0.018, 0.982]

## Project Structure

```
MMLM_EngraciaBatista_Chrysalytics_2026/
├── configs/
│   └── model_config.yaml        # All model and pipeline configuration
├── data/                        # Kaggle competition data (not tracked in git)
├── outputs/
│   ├── models/                  # Trained models, team features, CV results
│   └── submissions/             # Final submission CSV
├── src/
│   ├── features/                # Feature engineering (season stats, matchup features)
│   ├── models/                  # Power rating and Elo model implementations
│   └── pipeline/                # Runnable pipeline scripts
└── README.md
```

## How to Run

All commands must be run from the project root directory. Run them in order — each step depends on the outputs of the previous one.

**Step 0 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 1 — Build features**

Computes team season stats, Elo ratings, Bradley-Terry strength, SOS-adjusted efficiency, and matchup features for both Men's and Women's leagues. Outputs to `outputs/models/`.
```bash
python -m src.pipeline.build_features --config configs/model_config.yaml
```

**Step 2 — Train the power rating model**

Trains an auxiliary LightGBM model on regular season data to generate power ratings used as features downstream.
```bash
python -m src.pipeline.train_power_model --config configs/model_config.yaml
```

> Note: this step may produce a harmless sklearn warning about feature names — it does not affect results.

**Step 3 — Train the tournament model**

Trains the main combined M+W model with leave-one-season-out CV, fits the quintic spline calibrator on OOF predictions, and saves fold models for ensembling.
```bash
python -m src.pipeline.train_tournament_model --config configs/model_config.yaml --league combined
```

Optionally, add `--tune` to run Optuna hyperparameter search (100 trials, slower):
```bash
python -m src.pipeline.train_tournament_model --config configs/model_config.yaml --league combined --tune
```

**Step 4 — Generate the submission**

Runs the ensemble of fold models on all possible 2026 tournament matchups and outputs the final submission file.
```bash
python -m src.pipeline.predict_submission --config configs/model_config.yaml
```

The submission file will be written to `outputs/submissions/FinalSubmissionStage.csv` (132,133 rows).

## CV Results

Cross-validation Brier scores by season (11-fold leave-one-season-out):

| Season | Brier |
|--------|-------|
| 2012   | 0.1556 |
| 2013   | 0.1758 |
| 2014   | 0.1581 |
| 2015   | 0.1419 |
| 2016   | 0.1725 |
| 2017   | 0.1489 |
| 2018   | 0.1810 |
| 2019   | 0.1449 |
| 2021   | 0.1821 |
| 2022   | 0.1812 |
| 2023   | 0.1907 |
| **Mean** | **0.1666** |
