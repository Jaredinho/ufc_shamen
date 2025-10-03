# UFC Matchup Web App

A minimal Flask frontend that uses the existing XGBoost pipeline to score future UFC matchups and surface the top feature contributions.

## Prerequisites

1. Ensure the training script has produced `modeling/artifacts/win_model.joblib`:
   ```bash
   python modeling/train_win_model.py --test-fraction 0.2
   ```
2. Install the extra runtime dependency (Flask):
   ```bash
   pip install flask
   ```

## Run locally

```bash
python webapp/app.py
```

Then open http://127.0.0.1:5000 in a browser. Enter two fighter names (matching the scraped dataset) plus optional fight date, weight class, and the number of key factors to display. The app returns win probabilities, log-odds contributions, and high-level fighter snapshots.

## Notes & Next Steps

- To export explanations for each query, continue using the CLI flags in `modeling/train_win_model.py`.
- Consider containerising the service or adding auth before exposing it publicly.
- For richer UX, you can preload upcoming cards and allow multi-matchup comparisons.
