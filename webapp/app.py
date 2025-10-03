from __future__ import annotations

import datetime as dt
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
import json

from modeling.train_win_model import (
    build_matchup_dataset,
    build_future_matchup,
    build_future_matchup_with_recency,
    load_fighters_data_only,
    load_fighters_with_recency_only,
    get_feature_columns,
    get_expected_feature_columns,
    get_prediction_contributions,
    format_contribution_summary,
)


def _clean_for_json(obj):
    """Recursively clean NaN values from nested objects for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.floating, float)) and np.isnan(obj):
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


app = Flask(__name__, template_folder="templates", static_folder="static")

DATA_DIR = PROJECT_ROOT / "scrape_ufc_stats"
MODEL_PATH = PROJECT_ROOT / "modeling" / "artifacts" / "win_model.joblib"

# Load data and model once at startup - now with pre-computed recency features
print("Loading UFC data with recency features (pre-computed during model training)...")
FIGHTERS_HISTORY = load_fighters_with_recency_only(DATA_DIR)
FIGHTERS_RAW = load_fighters_data_only(DATA_DIR) # Keep for name suggestions
print("Loading trained model (now includes recency features)...")
PIPELINE = joblib.load(MODEL_PATH)
print("Setting up feature columns...")
FEATURE_COLUMNS = get_expected_feature_columns()
print("Webapp ready with recency-enhanced model!")



def _normalize_name(value: str) -> str:
    """Collapse whitespace and title-case like the stored fighter names."""
    if not value:
        return ""
    return " ".join(str(value).split())


FIGHTER_NAMES = tuple(
    sorted(
        {
            _normalize_name(name)
            for name in FIGHTERS_RAW["fighter_name"].dropna().astype(str)
            if _normalize_name(name)
        }
    )
)
FIGHTER_NAME_INDEX = [(name, name.casefold()) for name in FIGHTER_NAMES]


def _suggest_fighter_names(query: str, limit: int = 10) -> List[str]:
    """Return up to limit fighter name suggestions for a query string."""
    if not FIGHTER_NAMES:
        return []
    limit = max(1, min(limit, len(FIGHTER_NAMES)))
    normalized = _normalize_name(query)
    if not normalized:
        return list(FIGHTER_NAMES[:limit])
    lowered = normalized.casefold()
    suggestions: List[str] = []
    seen = set()
    # Prioritise substring matches for quick wins
    for name, folded in FIGHTER_NAME_INDEX:
        if lowered in folded:
            suggestions.append(name)
            seen.add(name)
            if len(suggestions) >= limit:
                return suggestions
    # Fall back to fuzzy matches for typos
    for candidate in get_close_matches(normalized, FIGHTER_NAMES, n=limit * 2, cutoff=0.6):
        if candidate not in seen:
            suggestions.append(candidate)
            seen.add(candidate)
            if len(suggestions) >= limit:
                return suggestions
    # Pad with remaining names to keep datalist populated
    for name in FIGHTER_NAMES:
        if name not in seen:
            suggestions.append(name)
            if len(suggestions) >= limit:
                break
    return suggestions[:limit]


def _ensure_feature_columns(matchup_dataset: pd.DataFrame) -> pd.DataFrame:
    """Align matchup dataset with the feature columns used during training."""
    aligned = matchup_dataset.copy()
    for col in FEATURE_COLUMNS:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned.reindex(columns=FEATURE_COLUMNS)


def _build_snapshot(row: pd.Series) -> Dict[str, Any]:
    """Return a lightweight summary for UI display."""
    return {
        "fighter_name": None if pd.isna(row.get("fighter_name")) else str(row.get("fighter_name")),
        "weight_class": None if pd.isna(row.get("weight_class")) else str(row.get("weight_class")),
        "stance": None if pd.isna(row.get("fighter_stance")) else str(row.get("fighter_stance")),
        "height_inches": None if pd.isna(row.get("height_inches")) else float(row.get("height_inches")),
        "reach_inches": None if pd.isna(row.get("reach_inches")) else float(row.get("reach_inches")),
        "weight_lbs": None if pd.isna(row.get("weight_lbs")) else float(row.get("weight_lbs")),
        "age_years": None if pd.isna(row.get("age_years")) else float(row.get("age_years")),
        "prev_fight_count": None if pd.isna(row.get("prev_fight_count")) else int(row.get("prev_fight_count")),
        "prev_win_rate": None if pd.isna(row.get("prev_win_rate")) else float(row.get("prev_win_rate")),
        "days_since_last_fight": None
        if pd.isna(row.get("days_since_last_fight"))
        else int(row.get("days_since_last_fight")),
        "prev_sig_strikes_landed_avg": None
        if pd.isna(row.get("prev_sig_strikes_landed_avg"))
        else float(row.get("prev_sig_strikes_landed_avg")),
        "prev_sig_strikes_accuracy_avg": None
        if pd.isna(row.get("prev_sig_strikes_accuracy_avg"))
        else float(row.get("prev_sig_strikes_accuracy_avg")),
        "prev_takedowns_landed_avg": None
        if pd.isna(row.get("prev_takedowns_landed_avg"))
        else float(row.get("prev_takedowns_landed_avg")),
        "prev_takedown_accuracy_avg": None
        if pd.isna(row.get("prev_takedown_accuracy_avg"))
        else float(row.get("prev_takedown_accuracy_avg")),
        "prev_control_seconds_avg": None
        if pd.isna(row.get("prev_control_seconds_avg"))
        else float(row.get("prev_control_seconds_avg")),
        "prev_submission_attempts_avg": None
        if pd.isna(row.get("prev_submission_attempts_avg"))
        else float(row.get("prev_submission_attempts_avg")),
    }


def _format_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Produce user-friendly strings for the frontend."""
    formatted: Dict[str, Any] = {"fighter_name": snapshot["fighter_name"]}
    formatted["weight_class"] = snapshot.get("weight_class") or "Unknown"
    formatted["stance"] = snapshot.get("stance") or "Unknown"

    def _inch_str(value: float | None) -> str:
        if value is None:
            return "N/A"
        feet = int(value // 12)
        inches = int(round(value - feet * 12))
        if inches == 12:
            feet += 1
            inches = 0
        return f"{feet}' {inches}\""

    formatted["height"] = _inch_str(snapshot.get("height_inches"))
    formatted["reach"] = "N/A" if snapshot.get("reach_inches") is None else f"{snapshot['reach_inches']:.0f}\""
    formatted["weight"] = "N/A" if snapshot.get("weight_lbs") is None else f"{snapshot['weight_lbs']:.0f} lbs"
    formatted["age"] = "N/A" if snapshot.get("age_years") is None else f"{snapshot['age_years']:.1f} yrs"
    formatted["total_fights"] = snapshot.get("prev_fight_count", "N/A")
    formatted["win_rate"] = (
        "N/A" if snapshot.get("prev_win_rate") is None else f"{snapshot['prev_win_rate'] * 100:.1f}%"
    )
    formatted["days_since_last_fight"] = (
        "N/A" if snapshot.get("days_since_last_fight") is None else f"{snapshot['days_since_last_fight']} days"
    )
    formatted["sig_strikes_avg"] = (
        "N/A" if snapshot.get("prev_sig_strikes_landed_avg") is None else f"{snapshot['prev_sig_strikes_landed_avg']:.2f}"
    )
    formatted["sig_strikes_acc"] = (
        "N/A" if snapshot.get("prev_sig_strikes_accuracy_avg") is None else f"{snapshot['prev_sig_strikes_accuracy_avg'] * 100:.1f}%"
    )
    formatted["takedowns_avg"] = (
        "N/A" if snapshot.get("prev_takedowns_landed_avg") is None else f"{snapshot['prev_takedowns_landed_avg']:.2f}"
    )
    formatted["takedown_acc"] = (
        "N/A" if snapshot.get("prev_takedown_accuracy_avg") is None else f"{snapshot['prev_takedown_accuracy_avg'] * 100:.1f}%"
    )
    formatted["control_time_avg"] = (
        "N/A" if snapshot.get("prev_control_seconds_avg") is None else f"{snapshot['prev_control_seconds_avg']:.1f} s"
    )
    formatted["sub_attempts_avg"] = (
        "N/A" if snapshot.get("prev_submission_attempts_avg") is None else f"{snapshot['prev_submission_attempts_avg']:.2f}"
    )
    return formatted


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/suggest_fighters")
def suggest_fighters() -> Any:
    query = request.args.get("q", default="") or ""
    limit = request.args.get("limit", default=10, type=int)
    if limit is None or limit <= 0:
        limit = 10
    limit = max(1, min(limit, 25))
    suggestions = _suggest_fighter_names(query, limit)
    return jsonify({"fighters": suggestions})


@app.post("/api/predict")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    fighter_a = (payload.get("fighter_a") or "").strip()
    fighter_b = (payload.get("fighter_b") or "").strip()
    if not fighter_a or not fighter_b:
        return jsonify({"error": "fighter_a and fighter_b are required"}), 400

    fight_date_str = payload.get("fight_date")
    if fight_date_str:
        try:
            fight_date = pd.to_datetime(fight_date_str).normalize()
        except Exception:  # pylint: disable=broad-except
            return jsonify({"error": "fight_date must be YYYY-MM-DD"}), 400
    else:
        fight_date = pd.Timestamp(dt.date.today())

    weight_class = payload.get("weight_class")
    top_contribs = payload.get("top_contribs", 8)
    if not isinstance(top_contribs, int) or top_contribs <= 0:
        return jsonify({"error": "top_contribs must be a positive integer"}), 400

    try:
        # Use pre-computed recency features for fast predictions
        matchup_dataset, future_rows = build_future_matchup_with_recency(
            FIGHTERS_HISTORY.copy(), fighter_a, fighter_b, fight_date, weight_class
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    features = _ensure_feature_columns(matchup_dataset)
    proba = float(PIPELINE.predict_proba(features)[0, 1])

    contribs = get_prediction_contributions(PIPELINE, features)
    top_series = format_contribution_summary(contribs, top_contribs)
    contributions = [
        {"feature": name, "log_odds_contribution": float(value)}
        for name, value in top_series.items()
    ]

    snapshots_raw = future_rows.sort_values("fighter_role")
    canonical_names = {
        row["fighter_role"]: row["fighter_name"]
        for _, row in snapshots_raw.iterrows()
    }
    display_a = canonical_names.get("A", fighter_a)
    display_b = canonical_names.get("B", fighter_b)
    snapshots = [
        _format_snapshot(_build_snapshot(row))
        for _, row in snapshots_raw.iterrows()
    ]

    response_data = {
        "fight_date": fight_date.date().isoformat(),
        "fighter_a": display_a,
        "fighter_b": display_b,
        "weight_class": matchup_dataset["weight_class"].iloc[0] or "Unknown",
        "probabilities": {
            display_a: proba,
            display_b: 1 - proba,
        },
        "contributions": contributions,
        "snapshots": snapshots,
    }
    
    # Clean NaN values before JSON serialization
    clean_response = _clean_for_json(response_data)
    return jsonify(clean_response)


if __name__ == "__main__":
    app.run(debug=True)
