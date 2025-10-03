"""Build a time-aware UFC matchup dataset, train an XGBoost win/loss model, and run custom predictions."""
import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import xgboost as xgb


STAT_COLUMNS = [
    "knockdowns",
    "sig_strikes_landed",
    "sig_strikes_attempted",
    "total_strikes_landed",
    "total_strikes_attempted",
    "takedowns_landed",
    "takedowns_attempted",
    "submission_attempts",
    "reversals",
    "control_seconds",
    "distance_sig_strikes_landed",
    "distance_sig_strikes_attempted",
    "clinch_sig_strikes_landed",
    "clinch_sig_strikes_attempted",
    "ground_sig_strikes_landed",
    "ground_sig_strikes_attempted",
]

ACCURACY_COLUMNS = [
    "sig_strikes_accuracy",
    "takedown_accuracy",
    "distance_sig_strikes_accuracy",
    "clinch_sig_strikes_accuracy",
    "ground_sig_strikes_accuracy",
]

FIGHTER_HISTORY_COLUMNS = [
    "fight_id",
    "fight_date",
    "fighter_name",
    "opponent_name",
    "fighter_role",
    "won",
    "weight_class",
    "height_inches",
    "reach_inches",
    "weight_lbs",
    "fighter_stance",
    "age_years",
    "prev_fight_count",
    "prev_win_rate",
    "prev_knockdowns_avg",
    "prev_sig_strikes_landed_avg",
    "prev_sig_strikes_attempted_avg",
    "prev_total_strikes_landed_avg",
    "prev_total_strikes_attempted_avg",
    "prev_takedowns_landed_avg",
    "prev_takedowns_attempted_avg",
    "prev_submission_attempts_avg",
    "prev_reversals_avg",
    "prev_control_seconds_avg",
    "prev_distance_sig_strikes_landed_avg",
    "prev_distance_sig_strikes_attempted_avg",
    "prev_clinch_sig_strikes_landed_avg",
    "prev_clinch_sig_strikes_attempted_avg",
    "prev_ground_sig_strikes_landed_avg",
    "prev_ground_sig_strikes_attempted_avg",
    "prev_sig_strikes_accuracy_avg",
    "prev_takedown_accuracy_avg",
    "prev_distance_sig_strikes_accuracy_avg",
    "prev_clinch_sig_strikes_accuracy_avg",
    "prev_ground_sig_strikes_accuracy_avg",
    "days_since_last_fight",
]

NUMERIC_BASE_COLUMNS = [
    "height_inches",
    "reach_inches",
    "weight_lbs",
    "age_years",
    "prev_fight_count",
    "prev_win_rate",
    "prev_knockdowns_avg",
    "prev_sig_strikes_landed_avg",
    "prev_sig_strikes_attempted_avg",
    "prev_total_strikes_landed_avg",
    "prev_total_strikes_attempted_avg",
    "prev_takedowns_landed_avg",
    "prev_takedowns_attempted_avg",
    "prev_submission_attempts_avg",
    "prev_reversals_avg",
    "prev_control_seconds_avg",
    "prev_distance_sig_strikes_landed_avg",
    "prev_distance_sig_strikes_attempted_avg",
    "prev_clinch_sig_strikes_landed_avg",
    "prev_clinch_sig_strikes_attempted_avg",
    "prev_ground_sig_strikes_landed_avg",
    "prev_ground_sig_strikes_attempted_avg",
    "prev_sig_strikes_accuracy_avg",
    "prev_takedown_accuracy_avg",
    "prev_distance_sig_strikes_accuracy_avg",
    "prev_clinch_sig_strikes_accuracy_avg",
    "prev_ground_sig_strikes_accuracy_avg",
    "days_since_last_fight",
]

CATEGORICAL_FEATURES = ["weight_class", "fighter_stance", "opponent_stance"]

IDENTIFIER_COLUMNS = {"fight_id", "fight_date", "fighter_A", "fighter_B", "target"}

# Optimal recency parameters from hyperparameter tuning
OPTIMAL_TAU_DAYS = 180.0  # 6-month exponential decay
OPTIMAL_LAST_N = 7        # Consider last 7 fights

# Mapping for recency feature computation
RECENCY_SOURCE_COLUMNS = {
    "win_rate": "won",
    "sig_strikes_landed_avg": "sig_strikes_landed",
    "sig_strikes_accuracy_avg": "sig_strikes_accuracy",
    "takedowns_landed_avg": "takedowns_landed",
    "takedown_accuracy_avg": "takedown_accuracy",
    "control_seconds_avg": "control_seconds",
    "submission_attempts_avg": "submission_attempts",
}


def clean_text(value):
    if pd.isna(value):
        return np.nan
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_height(value):
    if not isinstance(value, str):
        return np.nan
    value = value.strip()
    if not value or value == "--":
        return np.nan
    match = re.match(r"(?P<feet>\d+)'[\s]*(?P<inches>\d+)", value)
    if not match:
        return np.nan
    feet = int(match.group("feet"))
    inches = int(match.group("inches"))
    return feet * 12 + inches


def parse_inches(value):
    if not isinstance(value, str):
        return np.nan
    value = value.strip()
    if not value or value == "--":
        return np.nan
    match = re.search(r"(\d+)", value)
    if not match:
        return np.nan
    return float(match.group(1))


def parse_weight(value):
    if not isinstance(value, str):
        return np.nan
    value = value.strip()
    if not value or value == "--":
        return np.nan
    match = re.search(r"(\d+)", value)
    if not match:
        return np.nan
    return float(match.group(1))


def parse_control_time(value):
    if pd.isna(value):
        return 0.0
    value = str(value).strip()
    if not value or value in {"--", "---"}:
        return 0.0
    if ":" not in value:
        return 0.0
    try:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return 0.0


def extract_landed_attempted(df: pd.DataFrame, source_col: str, landed_col: str, attempted_col: str) -> pd.DataFrame:
    extracted = df[source_col].str.extract(r"(?P<landed>\d+)\s*of\s*(?P<attempted>\d+)", expand=True)
    df[landed_col] = pd.to_numeric(extracted["landed"], errors="coerce").fillna(0)
    df[attempted_col] = pd.to_numeric(extracted["attempted"], errors="coerce").fillna(0)
    return df


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator.divide(denom)


def compute_age_years(dob: pd.Timestamp, fight_date: pd.Timestamp) -> float:
    if pd.isna(dob) or pd.isna(fight_date):
        return np.nan
    return (fight_date - dob).days / 365.25


def _weighted_average(values: pd.Series, weights: np.ndarray) -> float:
    """Compute weighted average with exponential decay weights."""
    import math
    weight_sum = float(weights.sum())
    if weight_sum <= 0 or np.isnan(weight_sum):
        return math.nan
    return float(np.dot(values.to_numpy(dtype=float), weights) / weight_sum)


def compute_recency_features(
    fighters_raw: pd.DataFrame,
    tau_days: float = OPTIMAL_TAU_DAYS,
    last_n: int = OPTIMAL_LAST_N,
) -> pd.DataFrame:
    """Return per-fighter recency metrics for each fight using optimal parameters."""
    import math
    
    relevant_cols = {
        "fight_id",
        "fighter_name", 
        "fight_date",
        "won",
    }
    relevant_cols.update(RECENCY_SOURCE_COLUMNS.values())
    data = fighters_raw.loc[:, sorted(relevant_cols)].copy()
    data.sort_values(["fighter_name", "fight_date", "fight_id"], inplace=True)

    records = []
    for _, group in data.groupby("fighter_name", sort=False):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            previous = group.iloc[:idx]
            record = {
                "fight_id": row["fight_id"],
                "fighter_name": row["fighter_name"],
            }
            
            if previous.empty:
                for key in RECENCY_SOURCE_COLUMNS:
                    record[f"recent_decay_{key}"] = math.nan
                    record[f"recent_lastN_{key}"] = math.nan
                record["recent_decay_weight_sum"] = 0.0
                record["recent_lastN_fight_count"] = 0
                records.append(record)
                continue

            # Exponential decay features
            deltas = (row["fight_date"] - previous["fight_date"]).dt.days.to_numpy(dtype=float)
            weights = np.exp(-deltas / tau_days)
            record["recent_decay_weight_sum"] = float(weights.sum())
            for target, source in RECENCY_SOURCE_COLUMNS.items():
                record[f"recent_decay_{target}"] = _weighted_average(previous[source], weights)
            
            # Last-N features
            subset = previous.tail(last_n)
            record["recent_lastN_fight_count"] = int(len(subset))
            if subset.empty:
                for target in RECENCY_SOURCE_COLUMNS:
                    record[f"recent_lastN_{target}"] = math.nan
            else:
                for target, source in RECENCY_SOURCE_COLUMNS.items():
                    record[f"recent_lastN_{target}"] = float(subset[source].mean())

            records.append(record)

    return pd.DataFrame.from_records(records)


def merge_recency_features(
    dataset: pd.DataFrame,
    recency_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge recency features into the main dataset."""
    if recency_df.empty:
        return dataset
        
    value_cols = [col for col in recency_df.columns if col not in {"fight_id", "fighter_name"}]

    # Merge for fighter A
    recency_a = recency_df.rename(columns={"fighter_name": "fighter_key", **{col: f"{col}_A" for col in value_cols}})
    dataset = dataset.merge(
        recency_a,
        left_on=["fight_id", "fighter_A"],
        right_on=["fight_id", "fighter_key"],
        how="left",
    ).drop(columns=["fighter_key"])

    # Merge for fighter B
    recency_b = recency_df.rename(columns={"fighter_name": "fighter_key", **{col: f"{col}_B" for col in value_cols}})
    dataset = dataset.merge(
        recency_b,
        left_on=["fight_id", "fighter_B"],
        right_on=["fight_id", "fighter_key"],
        how="left",
    ).drop(columns=["fighter_key"])

    # Create difference features
    new_cols = []
    for col in value_cols:
        col_a = f"{col}_A"
        col_b = f"{col}_B"
        dataset[col_a] = pd.to_numeric(dataset[col_a], errors="coerce")
        dataset[col_b] = pd.to_numeric(dataset[col_b], errors="coerce")
        diff_col = f"{col}_diff"
        dataset[diff_col] = dataset[col_a] - dataset[col_b]
        new_cols.extend([col_a, col_b, diff_col])

    # Drop columns that are all NaN
    drop_candidates = [col for col in new_cols if dataset[col].isna().all()]
    if drop_candidates:
        dataset = dataset.drop(columns=drop_candidates)

    return dataset


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return slug.strip("-") or "fighter"


def get_feature_columns(dataset: pd.DataFrame) -> List[str]:
    return [col for col in dataset.columns if col not in IDENTIFIER_COLUMNS]


def get_expected_feature_columns() -> List[str]:
    """Get the expected feature columns without building a full dataset.
    
    This is useful for webapp startup to avoid expensive computation.
    """
    feature_cols = []
    
    # Add base columns with A/B/diff suffixes
    for col in NUMERIC_BASE_COLUMNS:
        feature_cols.extend([f"{col}_A", f"{col}_B", f"{col}_diff"])
    
    # Add categorical features
    feature_cols.extend(CATEGORICAL_FEATURES)
    
    # Add recency features
    recency_features = [
        "recency_win_rate_A", "recency_win_rate_B", "recency_win_rate_diff",
        "recency_sig_strikes_landed_avg_A", "recency_sig_strikes_landed_avg_B", "recency_sig_strikes_landed_avg_diff",
        "recency_sig_strikes_absorbed_avg_A", "recency_sig_strikes_absorbed_avg_B", "recency_sig_strikes_absorbed_avg_diff",
        "recency_sig_strikes_accuracy_avg_A", "recency_sig_strikes_accuracy_avg_B", "recency_sig_strikes_accuracy_avg_diff",
        "recency_takedowns_landed_avg_A", "recency_takedowns_landed_avg_B", "recency_takedowns_landed_avg_diff",
        "recency_takedowns_absorbed_avg_A", "recency_takedowns_absorbed_avg_B", "recency_takedowns_absorbed_avg_diff",
        "recency_takedown_accuracy_avg_A", "recency_takedown_accuracy_avg_B", "recency_takedown_accuracy_avg_diff",
        "recency_takedown_defense_avg_A", "recency_takedown_defense_avg_B", "recency_takedown_defense_avg_diff",
        "recency_control_seconds_avg_A", "recency_control_seconds_avg_B", "recency_control_seconds_avg_diff",
        "recency_submission_attempts_avg_A", "recency_submission_attempts_avg_B", "recency_submission_attempts_avg_diff"
    ]
    feature_cols.extend(recency_features)
    
    return feature_cols


def format_height_inches(value) -> str:
    if pd.isna(value):
        return "N/A"
    total_inches = float(value)
    feet = int(total_inches // 12)
    inches = int(round(total_inches - feet * 12))
    if inches == 12:
        feet += 1
        inches = 0
    return f"{feet}' {inches}{chr(34)}"


def format_reach_inches(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.0f}{chr(34)}"


def format_weight_lbs(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.0f} lbs"


def format_age_years(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.1f} yrs"


def format_percent(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def format_float(value, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def format_count(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{int(value)}"


def format_days(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{int(value)} days"


def format_optional_string(value, fallback: str = "Unknown") -> str:
    if pd.isna(value) or value == "":
        return fallback
    return str(value)


def prepare_results(data_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(data_dir / "ufc_fight_results.csv")
    results["EVENT"] = results["EVENT"].map(clean_text)
    results["BOUT"] = results["BOUT"].map(clean_text)
    results["WEIGHTCLASS"] = results["WEIGHTCLASS"].map(clean_text)
    results["OUTCOME"] = results["OUTCOME"].map(clean_text)
    results["URL"] = results["URL"].map(lambda x: x.strip() if isinstance(x, str) else x)
    results = results[results["OUTCOME"].isin(["W/L", "L/W"])].copy()
    fighters = results["BOUT"].str.split(r"\s+vs\.\s+", n=1, expand=True)
    results["fighter_a"] = fighters[0].map(clean_text)
    results["fighter_b"] = fighters[1].map(clean_text)
    results.dropna(subset=["fighter_a", "fighter_b"], inplace=True)
    events = pd.read_csv(data_dir / "ufc_event_details.csv")
    events["EVENT"] = events["EVENT"].map(clean_text)
    events["DATE"] = pd.to_datetime(events["DATE"], errors="coerce")
    results = results.merge(events[["EVENT", "DATE"]], on="EVENT", how="left")
    results.rename(columns={"DATE": "fight_date", "WEIGHTCLASS": "weight_class"}, inplace=True)
    results["fight_id"] = results["URL"]
    results["fight_date"] = pd.to_datetime(results["fight_date"], errors="coerce")
    results.dropna(subset=["fight_id", "fight_date"], inplace=True)
    return results


def prepare_fight_stats(data_dir: Path, results: pd.DataFrame) -> pd.DataFrame:
    stats = pd.read_csv(data_dir / "ufc_fight_stats.csv")
    stats["EVENT"] = stats["EVENT"].map(clean_text)
    stats["BOUT"] = stats["BOUT"].map(clean_text)
    stats["FIGHTER"] = stats["FIGHTER"].map(clean_text)
    stats = stats.merge(results[["EVENT", "BOUT", "fight_id"]], on=["EVENT", "BOUT"], how="left")
    stats.dropna(subset=["fight_id"], inplace=True)
    stats["KD"] = pd.to_numeric(stats["KD"], errors="coerce").fillna(0)
    stats = extract_landed_attempted(stats, "SIG.STR.", "sig_strikes_landed", "sig_strikes_attempted")
    stats = extract_landed_attempted(stats, "TOTAL STR.", "total_strikes_landed", "total_strikes_attempted")
    stats = extract_landed_attempted(stats, "TD", "takedowns_landed", "takedowns_attempted")
    stats = extract_landed_attempted(stats, "HEAD", "head_sig_strikes_landed", "head_sig_strikes_attempted")
    stats = extract_landed_attempted(stats, "BODY", "body_sig_strikes_landed", "body_sig_strikes_attempted")
    stats = extract_landed_attempted(stats, "LEG", "leg_sig_strikes_landed", "leg_sig_strikes_attempted")
    stats = extract_landed_attempted(stats, "DISTANCE", "distance_sig_strikes_landed", "distance_sig_strikes_attempted")
    stats = extract_landed_attempted(stats, "CLINCH", "clinch_sig_strikes_landed", "clinch_sig_strikes_attempted")
    stats = extract_landed_attempted(stats, "GROUND", "ground_sig_strikes_landed", "ground_sig_strikes_attempted")
    stats["control_seconds"] = stats["CTRL"].map(parse_control_time)
    stats["submission_attempts"] = pd.to_numeric(stats["SUB.ATT"], errors="coerce").fillna(0)
    stats["reversals"] = pd.to_numeric(stats["REV."], errors="coerce").fillna(0)
    stats.rename(columns={"KD": "knockdowns"}, inplace=True)
    agg_cols = {
        "knockdowns": "sum",
        "sig_strikes_landed": "sum",
        "sig_strikes_attempted": "sum",
        "total_strikes_landed": "sum",
        "total_strikes_attempted": "sum",
        "takedowns_landed": "sum",
        "takedowns_attempted": "sum",
        "submission_attempts": "sum",
        "reversals": "sum",
        "control_seconds": "sum",
        "distance_sig_strikes_landed": "sum",
        "distance_sig_strikes_attempted": "sum",
        "clinch_sig_strikes_landed": "sum",
        "clinch_sig_strikes_attempted": "sum",
        "ground_sig_strikes_landed": "sum",
        "ground_sig_strikes_attempted": "sum",
    }
    grouped = stats.groupby(["fight_id", "FIGHTER"], as_index=False).agg(agg_cols)
    grouped.rename(columns={"FIGHTER": "fighter_name"}, inplace=True)
    grouped["sig_strikes_accuracy"] = safe_div(grouped["sig_strikes_landed"], grouped["sig_strikes_attempted"])
    grouped["takedown_accuracy"] = safe_div(grouped["takedowns_landed"], grouped["takedowns_attempted"])
    grouped["distance_sig_strikes_accuracy"] = safe_div(grouped["distance_sig_strikes_landed"], grouped["distance_sig_strikes_attempted"])
    grouped["clinch_sig_strikes_accuracy"] = safe_div(grouped["clinch_sig_strikes_landed"], grouped["clinch_sig_strikes_attempted"])
    grouped["ground_sig_strikes_accuracy"] = safe_div(grouped["ground_sig_strikes_landed"], grouped["ground_sig_strikes_attempted"])
    return grouped


def prepare_fighter_tott(data_dir: Path) -> pd.DataFrame:
    tott = pd.read_csv(data_dir / "ufc_fighter_tott.csv")
    tott["FIGHTER"] = tott["FIGHTER"].map(clean_text)
    tott["height_inches"] = tott["HEIGHT"].map(parse_height)
    tott["reach_inches"] = tott["REACH"].map(parse_inches)
    tott["weight_lbs"] = tott["WEIGHT"].map(parse_weight)
    tott["stance"] = tott["STANCE"].map(clean_text)
    tott["dob"] = pd.to_datetime(tott["DOB"], errors="coerce")
    tott = tott[["FIGHTER", "height_inches", "reach_inches", "weight_lbs", "stance", "dob"]]
    tott = tott.drop_duplicates(subset="FIGHTER", keep="last")
    tott.rename(columns={"FIGHTER": "fighter_name"}, inplace=True)
    return tott


def build_fighter_rows(results: pd.DataFrame, stats: pd.DataFrame, fighter_tott: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["fight_id", "fight_date", "weight_class", "fighter_a", "fighter_b", "OUTCOME"]
    base = results[base_cols].copy()
    fighters_a = base.copy()
    fighters_a["fighter_name"] = fighters_a["fighter_a"]
    fighters_a["opponent_name"] = fighters_a["fighter_b"]
    fighters_a["fighter_role"] = "A"
    fighters_a["won"] = (fighters_a["OUTCOME"] == "W/L").astype(int)
    fighters_b = base.copy()
    fighters_b["fighter_name"] = fighters_b["fighter_b"]
    fighters_b["opponent_name"] = fighters_b["fighter_a"]
    fighters_b["fighter_role"] = "B"
    fighters_b["won"] = (fighters_b["OUTCOME"] == "L/W").astype(int)
    fighters = pd.concat([fighters_a, fighters_b], ignore_index=True)
    fighters = fighters.merge(stats, on=["fight_id", "fighter_name"], how="left")
    fighters = fighters.merge(fighter_tott, on="fighter_name", how="left")
    fighters["age_years"] = (fighters["fight_date"] - fighters["dob"]).dt.days / 365.25
    return fighters


def add_history_features(fighters: pd.DataFrame) -> pd.DataFrame:
    fighters = fighters.sort_values(["fighter_name", "fight_date", "fight_id"]).reset_index(drop=True)
    fighters[STAT_COLUMNS] = fighters[STAT_COLUMNS].fillna(0)
    fighters[ACCURACY_COLUMNS] = fighters[ACCURACY_COLUMNS].fillna(0)
    group = fighters.groupby("fighter_name", sort=False)
    fighters["prev_fight_count"] = group.cumcount()
    fighters["prev_wins"] = group["won"].cumsum() - fighters["won"]
    fighters["prev_win_rate"] = safe_div(fighters["prev_wins"].astype(float), fighters["prev_fight_count"].astype(float))
    fighters["previous_fight_date"] = group["fight_date"].shift()
    fighters["days_since_last_fight"] = (fighters["fight_date"] - fighters["previous_fight_date"]).dt.days
    fighters.drop(columns=["previous_fight_date"], inplace=True)
    for col in STAT_COLUMNS + ACCURACY_COLUMNS:
        cumulative = group[col].cumsum() - fighters[col]
        fighters[f"prev_{col}_avg"] = safe_div(cumulative.astype(float), fighters["prev_fight_count"].astype(float))
    fighters.drop(columns=["prev_wins"], inplace=True)
    fighters.rename(columns={"stance": "fighter_stance"}, inplace=True)
    fighters.drop(columns=STAT_COLUMNS + ACCURACY_COLUMNS, inplace=True)
    fighters.replace([np.inf, -np.inf], np.nan, inplace=True)
    return fighters


def assemble_matchups(fighters: pd.DataFrame, fight_ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    subset = fighters[FIGHTER_HISTORY_COLUMNS].copy()
    if fight_ids is not None:
        subset = subset[subset["fight_id"].isin(fight_ids)]
    fighters_a = subset[subset["fighter_role"] == "A"]
    fighters_b = subset[subset["fighter_role"] == "B"]
    merged = fighters_a.merge(fighters_b, on="fight_id", suffixes=("_A", "_B"))
    merged["fight_date"] = merged["fight_date_A"]
    merged["fighter_A"] = merged["fighter_name_A"]
    merged["fighter_B"] = merged["fighter_name_B"]
    merged["weight_class"] = merged["weight_class_A"]
    merged["fighter_stance"] = merged["fighter_stance_A"]
    merged["opponent_stance"] = merged["fighter_stance_B"]
    merged["target"] = merged["won_A"]
    for col in NUMERIC_BASE_COLUMNS:
        merged[f"{col}_diff"] = merged[f"{col}_A"] - merged[f"{col}_B"]
    final_cols = [
        "fight_id",
        "fight_date",
        "fighter_A",
        "fighter_B",
        "weight_class",
        "fighter_stance",
        "opponent_stance",
        "target",
    ]
    for col in NUMERIC_BASE_COLUMNS:
        final_cols.extend([f"{col}_A", f"{col}_B", f"{col}_diff"])
    dataset = merged[final_cols].sort_values("fight_date").reset_index(drop=True)
    return dataset


def load_fighters_data_only(data_dir: Path) -> pd.DataFrame:
    """Load only the raw fighter data without computing expensive recency features.
    
    This is used by the web app for fast startup and fighter name suggestions.
    Recency features are only computed when needed for specific predictions.
    """
    results = prepare_results(data_dir)
    stats = prepare_fight_stats(data_dir, results)
    fighter_tott = prepare_fighter_tott(data_dir)
    fighters_raw = build_fighter_rows(results, stats, fighter_tott)
    return fighters_raw


def load_fighters_with_recency_only(data_dir: Path) -> pd.DataFrame:
    """Load fighter data with recency features but without assembling full matchup dataset.
    
    Use this for webapp when you need recency features for predictions but want to avoid
    the expensive matchup assembly step during startup.
    """
    results = prepare_results(data_dir)
    stats = prepare_fight_stats(data_dir, results)
    fighter_tott = prepare_fighter_tott(data_dir)
    fighters_raw = build_fighter_rows(results, stats, fighter_tott)
    fighters_history = add_history_features(fighters_raw.copy())
    
    # Add optimal recency features (tau=180 days, last_n=7 fights)
    print("Computing recency features with optimal parameters (tau=180 days, last_n=7)...")
    recency_df = compute_recency_features(fighters_raw)
    fighters_with_recency = merge_recency_features(fighters_history, recency_df)
    print(f"Added recency features. Fighter data shape: {fighters_with_recency.shape}")
    
    return fighters_with_recency


def build_matchup_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build complete dataset with recency features for training/evaluation."""
    results = prepare_results(data_dir)
    stats = prepare_fight_stats(data_dir, results)
    fighter_tott = prepare_fighter_tott(data_dir)
    fighters_raw = build_fighter_rows(results, stats, fighter_tott)
    fighters_history = add_history_features(fighters_raw.copy())
    dataset = assemble_matchups(fighters_history)
    
    # Add optimal recency features (tau=180 days, last_n=7 fights)
    print("Computing recency features with optimal parameters (tau=180 days, last_n=7)...")
    recency_df = compute_recency_features(fighters_raw)
    dataset = merge_recency_features(dataset, recency_df)
    print(f"Added recency features. Dataset shape: {dataset.shape}")
    
    return dataset, fighters_history, fighters_raw


def find_latest_fighter_row(fighters: pd.DataFrame, fighter_name: str) -> pd.Series:
    # Clean the input fighter name to match how names are stored in the data
    cleaned_fighter_name = clean_text(fighter_name)
    mask = fighters["fighter_name"].str.casefold() == cleaned_fighter_name.casefold()
    subset = fighters.loc[mask]
    if subset.empty:
        raise ValueError(f"Fighter '{fighter_name}' not found in historical data.")
    subset = subset.sort_values("fight_date")
    return subset.iloc[-1]


def determine_weight_class(proposed: Optional[str], latest_a: pd.Series, latest_b: pd.Series) -> Optional[str]:
    if proposed:
        return proposed
    candidate_a = latest_a.get("weight_class")
    candidate_b = latest_b.get("weight_class")
    if pd.isna(candidate_a) and pd.isna(candidate_b):
        return None
    if pd.isna(candidate_a):
        return candidate_b
    if pd.isna(candidate_b) or candidate_a == candidate_b:
        return candidate_a
    print(f"Warning: recent weight classes differ ({candidate_a} vs {candidate_b}); defaulting to {candidate_a}.")
    return candidate_a


def build_future_matchup(
    fighters_raw: pd.DataFrame,
    fighter_a: str,
    fighter_b: str,
    fight_date: pd.Timestamp,
    weight_class: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fight_id = f"future-{fight_date.strftime('%Y%m%d')}-{slugify(fighter_a)}-vs-{slugify(fighter_b)}"
    latest_a = find_latest_fighter_row(fighters_raw, fighter_a)
    latest_b = find_latest_fighter_row(fighters_raw, fighter_b)
    resolved_weight_class = determine_weight_class(weight_class, latest_a, latest_b)
    new_rows = []
    canonical_a = latest_a.get("fighter_name")
    if pd.isna(canonical_a):
        canonical_a = clean_text(fighter_a)
    canonical_b = latest_b.get("fighter_name")
    if pd.isna(canonical_b):
        canonical_b = clean_text(fighter_b)
    canonical_names = {
        "A": canonical_a,
        "B": canonical_b,
    }
    for role, source in [("A", latest_a), ("B", latest_b)]:
        opponent_role = "B" if role == "A" else "A"
        row = source.copy()
        row["fight_id"] = fight_id
        row["fight_date"] = fight_date
        row["weight_class"] = resolved_weight_class
        row["fighter_a"] = canonical_names["A"]
        row["fighter_b"] = canonical_names["B"]
        row["OUTCOME"] = "TBD"
        row["fighter_name"] = canonical_names[role]
        row["opponent_name"] = canonical_names[opponent_role]
        row["fighter_role"] = role
        row["won"] = 0
        for col in STAT_COLUMNS:
            row[col] = 0.0
        for col in ACCURACY_COLUMNS:
            row[col] = 0.0
        row["age_years"] = compute_age_years(row.get("dob"), fight_date)
        new_rows.append(row)
    augmented = pd.concat([fighters_raw, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    fighters_with_history = add_history_features(augmented)
    fighters_with_history.loc[fighters_with_history["fight_id"] == fight_id, "won"] = np.nan
    matchup_dataset = assemble_matchups(fighters_with_history, fight_ids=[fight_id])
    future_fighters = fighters_with_history[fighters_with_history["fight_id"] == fight_id].copy()
    return matchup_dataset, future_fighters


def build_future_matchup_with_recency(
    fighters_with_recency: pd.DataFrame,
    fighter_a: str,
    fighter_b: str,
    fight_date: pd.Timestamp,
    weight_class: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build future matchup using pre-computed recency features for performance.
    This avoids recomputing recency features from scratch.
    """
    fight_id = f"future-{fight_date.strftime('%Y%m%d')}-{slugify(fighter_a)}-vs-{slugify(fighter_b)}"
    latest_a = find_latest_fighter_row(fighters_with_recency, fighter_a)
    latest_b = find_latest_fighter_row(fighters_with_recency, fighter_b)
    resolved_weight_class = determine_weight_class(weight_class, latest_a, latest_b)
    new_rows = []
    canonical_a = latest_a.get("fighter_name")
    if pd.isna(canonical_a):
        canonical_a = clean_text(fighter_a)
    canonical_b = latest_b.get("fighter_name")
    if pd.isna(canonical_b):
        canonical_b = clean_text(fighter_b)
    canonical_names = {
        "A": canonical_a,
        "B": canonical_b,
    }
    for role, source in [("A", latest_a), ("B", latest_b)]:
        opponent_role = "B" if role == "A" else "A"
        row = source.copy()
        row["fight_id"] = fight_id
        row["fight_date"] = fight_date
        row["weight_class"] = resolved_weight_class
        row["fighter_a"] = canonical_names["A"]
        row["fighter_b"] = canonical_names["B"]
        row["OUTCOME"] = "TBD"
        row["fighter_name"] = canonical_names[role]
        row["opponent_name"] = canonical_names[opponent_role]
        row["fighter_role"] = role
        row["won"] = 0
        for col in STAT_COLUMNS:
            row[col] = 0.0
        for col in ACCURACY_COLUMNS:
            row[col] = 0.0
        row["age_years"] = compute_age_years(row.get("dob"), fight_date)
        new_rows.append(row)
    
    # Create DataFrame with new matchup rows
    future_fighters_df = pd.DataFrame(new_rows)
    future_fighters_df.loc[future_fighters_df["fight_id"] == fight_id, "won"] = np.nan
    matchup_dataset = assemble_matchups(future_fighters_df, fight_ids=[fight_id])
    
    return matchup_dataset, future_fighters_df


def print_fighter_profile(row: pd.Series) -> None:
    print(f"  {row['fighter_name']}")
    print(f"    Weight class: {format_optional_string(row.get('weight_class'))}")
    print(f"    Stance: {format_optional_string(row.get('fighter_stance'))}")
    print(
        f"    Height: {format_height_inches(row.get('height_inches'))} | Reach: {format_reach_inches(row.get('reach_inches'))} | Weight: {format_weight_lbs(row.get('weight_lbs'))}"
    )
    print(f"    Age on fight night: {format_age_years(row.get('age_years'))}")
    print(f"    Total fights recorded: {format_count(row.get('prev_fight_count'))}")
    print(f"    Win rate: {format_percent(row.get('prev_win_rate'))}")
    print(f"    Days since last fight: {format_days(row.get('days_since_last_fight'))}")
    print(
        f"    Avg sig strikes landed: {format_float(row.get('prev_sig_strikes_landed_avg'))} | accuracy: {format_percent(row.get('prev_sig_strikes_accuracy_avg'))}"
    )
    print(
        f"    Avg takedowns landed: {format_float(row.get('prev_takedowns_landed_avg'))} | accuracy: {format_percent(row.get('prev_takedown_accuracy_avg'))}"
    )
    print(
        f"    Avg control seconds: {format_float(row.get('prev_control_seconds_avg'))} | sub attempts: {format_float(row.get('prev_submission_attempts_avg'))}"
    )



def get_prediction_contributions(pipeline: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Return log-odds contributions for the supplied features (one row)."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(features)
    feature_names = preprocessor.get_feature_names_out()
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(transformed, feature_names=list(feature_names))
    contribs = booster.predict(dmatrix, pred_contribs=True)
    contrib_names = list(feature_names) + ["bias"]
    return pd.Series(contribs[0], index=contrib_names)


def format_contribution_summary(contribs: pd.Series, top_n: int) -> pd.Series:
    ordered = contribs.abs().sort_values(ascending=False).index
    return contribs.loc[ordered[:max(1, top_n)]]


def build_contribution_log_dataframe(
    fight_date: pd.Timestamp,
    fighter_a: str,
    fighter_b: str,
    proba_a: float,
    contribs: pd.Series,
    top_n: int,
) -> pd.DataFrame:
    ordered = contribs.abs().sort_values(ascending=False).index
    selected = contribs.loc[ordered[:max(1, top_n)]].reset_index()
    selected.columns = ["feature", "log_odds_contribution"]
    selected["abs_log_odds_contribution"] = selected["log_odds_contribution"].abs()
    selected.insert(0, "fight_date", fight_date.date())
    selected.insert(1, "fighter_a", fighter_a)
    selected.insert(2, "fighter_b", fighter_b)
    selected.insert(3, "prob_fighter_a", proba_a)
    selected.insert(4, "prob_fighter_b", 1 - proba_a)
    return selected

def build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=42,
        verbosity=1,  # Show training progress
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, label: str) -> None:
    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    ll = log_loss(y, probas)
    print(f"{label} accuracy: {acc:.3f}")
    print(f"{label} F1: {f1:.3f}")
    print(f"{label} log loss: {ll:.3f}")
    print(f"{label} confusion matrix:")
    print(confusion_matrix(y, preds))
    print(f"{label} classification report:")
    print(classification_report(y, preds, digits=3))


def run_training(dataset: pd.DataFrame, test_fraction: float, model_path: Optional[Path] = None) -> Pipeline:
    dataset = dataset.dropna(subset=["fight_date"])
    dataset = dataset.sort_values("fight_date").reset_index(drop=True)
    split_idx = int(len(dataset) * (1 - test_fraction))
    if split_idx <= 0 or split_idx >= len(dataset):
        raise ValueError("Not enough data for the chosen test fraction.")
    train = dataset.iloc[:split_idx]
    test = dataset.iloc[split_idx:]
    feature_cols = get_feature_columns(dataset)
    numeric_features = [col for col in feature_cols if col.endswith(("_A", "_B", "_diff"))]
    categorical_features = CATEGORICAL_FEATURES
    pipeline = build_pipeline(numeric_features, categorical_features)
    X_train = train[feature_cols]
    y_train = train["target"]
    X_test = test[feature_cols]
    y_test = test["target"]
    print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")
    pipeline.fit(X_train, y_train)
    evaluate_model(pipeline, X_train, y_train, "Train")
    evaluate_model(pipeline, X_test, y_test, "Test")
    if model_path:
        model_path = model_path.resolve()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        print(f"Saved trained pipeline to {model_path}")
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    print("Top 20 feature importances:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    recent = test[["fight_date", "fighter_A", "fighter_B"]].copy()
    recent["predicted_prob"] = pipeline.predict_proba(X_test)[:, 1]
    print("Sample predictions (holdout):")
    print(recent.tail(10).to_string(index=False))
    return pipeline


def main() -> None:
    default_data_dir = (Path(__file__).resolve().parent.parent / "scrape_ufc_stats").resolve()
    default_artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    parser = argparse.ArgumentParser(description="Train a UFC win/loss XGBoost model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing the UFC CSV exports.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of chronologically latest fights to reserve for testing.",
    )
    parser.add_argument(
        "--export-dataset",
        type=Path,
        help="Optional path to write the engineered matchup dataset as CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_artifacts_dir / "win_model.joblib",
        help="Where to save the trained model pipeline (joblib file).",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Skip writing the trained model to disk.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and load an existing model for prediction.",
    )
    parser.add_argument(
        "--fighter-a",
        type=str,
        help="Name of Fighter A for prediction (must match historical data).",
    )
    parser.add_argument(
        "--fighter-b",
        type=str,
        help="Name of Fighter B for prediction (must match historical data).",
    )
    parser.add_argument(
        "--fight-date",
        type=str,
        help="Future fight date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--weight-class",
        type=str,
        help="Override weight class label for the custom prediction.",
    )
    parser.add_argument(
        "--top-contribs",
        type=int,
        default=12,
        help="Number of top absolute feature contributions to display for a custom prediction.",
    )
    parser.add_argument(
        "--explain-log",
        type=Path,
        help="Optional CSV file to append feature contribution breakdowns for custom predictions.",
    )
    parser.add_argument(
        "--explain-log-top-n",
        type=int,
        help="Number of feature contributions to store in the explain log (defaults to --top-contribs).",
    )
    args = parser.parse_args()
    model_path = None if args.no_save_model else args.model_path
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    dataset, _, fighters_raw = build_matchup_dataset(args.data_dir)
    print(
        f"Built dataset with {len(dataset)} rows covering fights from {dataset['fight_date'].min().date()} to {dataset['fight_date'].max().date()}."
    )
    if args.export_dataset:
        dataset.to_csv(args.export_dataset, index=False)
        print(f"Exported dataset to {args.export_dataset}")
    pipeline: Optional[Pipeline] = None
    if not args.predict_only:
        pipeline = run_training(dataset, args.test_fraction, model_path)
    if (args.predict_only or args.fighter_a or args.fighter_b) and pipeline is None:
        if model_path is None:
            raise ValueError("Model path is required to load a trained pipeline.")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run training or provide the correct path."
            )
        pipeline = joblib.load(model_path)
        print(f"Loaded trained pipeline from {model_path}")
    if args.fighter_a and args.fighter_b:
        fight_date = pd.Timestamp.today().normalize()
        if args.fight_date:
            fight_date = pd.to_datetime(args.fight_date).normalize()
        matchup_dataset, future_rows = build_future_matchup(
            fighters_raw.copy(), args.fighter_a.strip(), args.fighter_b.strip(), fight_date, args.weight_class
        )
        feature_cols = get_feature_columns(dataset)
        for col in feature_cols:
            if col not in matchup_dataset.columns:
                matchup_dataset[col] = np.nan
        features = matchup_dataset.reindex(columns=feature_cols)
        proba = pipeline.predict_proba(features)[:, 1][0]
        print("\nCustom matchup prediction")
        print(f"  Event date: {fight_date.date()}")
        print(f"  Fighters: {args.fighter_a} (A) vs {args.fighter_b} (B)")
        weight_label = matchup_dataset["weight_class"].iloc[0]
        print(f"  Weight class: {weight_label if weight_label else 'Unknown'}")
        print(f"  Win probability ({args.fighter_a}): {proba:.3f}")
        print(f"  Win probability ({args.fighter_b}): {1 - proba:.3f}")
        contribs = get_prediction_contributions(pipeline, features)
        top_contribs = format_contribution_summary(contribs, args.top_contribs)
        print(f"\nTop {len(top_contribs)} feature contributions (log-odds impact for {args.fighter_a}):")
        for name, value in top_contribs.items():
            print(f"  {name}: {value:+.4f}")
        if args.explain_log:
            log_path = args.explain_log.resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_top_n = args.explain_log_top_n or args.top_contribs
            log_df = build_contribution_log_dataframe(
                fight_date, args.fighter_a, args.fighter_b, proba, contribs, log_top_n
            )
            log_df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
            print(f"Saved explanation details to {log_path}")
        print("\nFighter snapshots prior to matchup:")
        for _, row in future_rows.sort_values("fighter_role").iterrows():
            print_fighter_profile(row)
    elif args.predict_only:
        print("No fighters supplied for prediction. Provide --fighter-a and --fighter-b to score a matchup.")


if __name__ == "__main__":
    main()
