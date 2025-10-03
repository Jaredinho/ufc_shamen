"""Hyperparameter search for recency-weighted fighter history features."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import accuracy_score, f1_score, log_loss

from modeling.train_win_model import (
    CATEGORICAL_FEATURES,
    build_matchup_dataset,
    build_pipeline,
    get_feature_columns,
)

RECENCY_SOURCE_COLUMNS: Dict[str, str] = {
    "win_rate": "won",
    "sig_strikes_landed_avg": "sig_strikes_landed",
    "sig_strikes_accuracy_avg": "sig_strikes_accuracy",
    "takedowns_landed_avg": "takedowns_landed",
    "takedown_accuracy_avg": "takedown_accuracy",
    "control_seconds_avg": "control_seconds",
    "submission_attempts_avg": "submission_attempts",
}


def parse_float_list(raw: str) -> List[Optional[float]]:
    items: List[Optional[float]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "na", "null"}:
            items.append(None)
        else:
            value = float(token)
            if value <= 0:
                items.append(None)
            else:
                items.append(value)
    return items or [None]


def parse_int_list(raw: str) -> List[Optional[int]]:
    items: List[Optional[int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() in {"none", "na", "null", "0"}:
            items.append(None)
        else:
            value = int(token)
            items.append(value if value > 0 else None)
    return items or [None]


def _weighted_average(values: pd.Series, weights: np.ndarray) -> float:
    weight_sum = float(weights.sum())
    if weight_sum <= 0 or np.isnan(weight_sum):
        return math.nan
    return float(np.dot(values.to_numpy(dtype=float), weights) / weight_sum)


def compute_recency_features(
    fighters_raw: pd.DataFrame,
    tau_days: Optional[float],
    last_n: Optional[int],
) -> pd.DataFrame:
    """Return per-fighter recency metrics for each fight."""
    relevant_cols = {
        "fight_id",
        "fighter_name",
        "fight_date",
        "won",
    }
    relevant_cols.update(RECENCY_SOURCE_COLUMNS.values())
    data = fighters_raw.loc[:, sorted(relevant_cols)].copy()
    data.sort_values(["fighter_name", "fight_date", "fight_id"], inplace=True)

    records: List[Dict[str, float]] = []
    for _, group in data.groupby("fighter_name", sort=False):
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            previous = group.iloc[:idx]
            record: Dict[str, float] = {
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

            if tau_days is not None:
                deltas = (row["fight_date"] - previous["fight_date"]).dt.days.to_numpy(dtype=float)
                weights = np.exp(-deltas / tau_days)
                record["recent_decay_weight_sum"] = float(weights.sum())
                for target, source in RECENCY_SOURCE_COLUMNS.items():
                    record[f"recent_decay_{target}"] = _weighted_average(previous[source], weights)
            else:
                for target in RECENCY_SOURCE_COLUMNS:
                    record[f"recent_decay_{target}"] = math.nan
                record["recent_decay_weight_sum"] = 0.0

            if last_n is not None:
                subset = previous.tail(last_n)
                record["recent_lastN_fight_count"] = int(len(subset))
                if subset.empty:
                    for target in RECENCY_SOURCE_COLUMNS:
                        record[f"recent_lastN_{target}"] = math.nan
                else:
                    for target, source in RECENCY_SOURCE_COLUMNS.items():
                        record[f"recent_lastN_{target}"] = float(subset[source].mean())
            else:
                for target in RECENCY_SOURCE_COLUMNS:
                    record[f"recent_lastN_{target}"] = math.nan
                record["recent_lastN_fight_count"] = 0

            records.append(record)

    return pd.DataFrame.from_records(records)


def merge_recency_features(
    dataset: pd.DataFrame,
    recency_df: pd.DataFrame,
) -> pd.DataFrame:
    if recency_df.empty:
        return dataset
    value_cols = [col for col in recency_df.columns if col not in {"fight_id", "fighter_name"}]

    recency_a = recency_df.rename(columns={"fighter_name": "fighter_key", **{col: f"{col}_A" for col in value_cols}})
    dataset = dataset.merge(
        recency_a,
        left_on=["fight_id", "fighter_A"],
        right_on=["fight_id", "fighter_key"],
        how="left",
    ).drop(columns=["fighter_key"])

    recency_b = recency_df.rename(columns={"fighter_name": "fighter_key", **{col: f"{col}_B" for col in value_cols}})
    dataset = dataset.merge(
        recency_b,
        left_on=["fight_id", "fighter_B"],
        right_on=["fight_id", "fighter_key"],
        how="left",
    ).drop(columns=["fighter_key"])

    new_cols: List[str] = []
    for col in value_cols:
        col_a = f"{col}_A"
        col_b = f"{col}_B"
        dataset[col_a] = pd.to_numeric(dataset[col_a], errors="coerce")
        dataset[col_b] = pd.to_numeric(dataset[col_b], errors="coerce")
        diff_col = f"{col}_diff"
        dataset[diff_col] = dataset[col_a] - dataset[col_b]
        new_cols.extend([col_a, col_b, diff_col])

    drop_candidates = [col for col in new_cols if dataset[col].isna().all()]
    if drop_candidates:
        dataset = dataset.drop(columns=drop_candidates)

    return dataset


def train_and_score(dataset: pd.DataFrame, test_fraction: float) -> Dict[str, float]:
    dataset = dataset.dropna(subset=["fight_date"]).sort_values("fight_date").reset_index(drop=True)
    split_idx = int(len(dataset) * (1 - test_fraction))
    if split_idx <= 0 or split_idx >= len(dataset):
        raise ValueError("test_fraction leaves no data for train/test split")

    train = dataset.iloc[:split_idx]
    test = dataset.iloc[split_idx:]

    feature_cols = get_feature_columns(dataset)
    numeric_features = [col for col in feature_cols if col.endswith(("_A", "_B", "_diff"))]
    pipeline = build_pipeline(numeric_features, CATEGORICAL_FEATURES)

    X_train = train[feature_cols]
    y_train = train["target"]
    pipeline.fit(X_train, y_train)

    X_test = test[feature_cols]
    y_test = test["target"]
    preds = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "log_loss": float(log_loss(y_test, probas)),
    }


def iter_parameter_grid(taus: Sequence[Optional[float]], windows: Sequence[Optional[int]]) -> Iterable[Tuple[Optional[float], Optional[int]]]:
    for tau in taus:
        for window in windows:
            yield tau, window


def main() -> None:
    parser = argparse.ArgumentParser(description="Search recency parameters for UFC matchup model.")
    parser.add_argument("--data-dir", type=Path, default=Path("scrape_ufc_stats"), help="Directory with scraped UFC CSV files.")
    parser.add_argument(
        "--taus",
        type=str,
        default="180,270,365,540",
        help="Comma-separated exponential decay windows in days (use 'none' to skip).",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="3,5,7",
        help="Comma-separated last-N fight windows (use 'none' to skip).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of most recent fights reserved for testing.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include a baseline run with no additional recency features.",
    )
    args = parser.parse_args()

    taus = parse_float_list(args.taus)
    windows = parse_int_list(args.windows)

    dataset_base, _, fighters_raw = build_matchup_dataset(args.data_dir)

    grid: List[Tuple[Optional[float], Optional[int]]] = list(iter_parameter_grid(taus, windows))
    if args.baseline and (None, None) not in grid:
        grid = [(None, None)] + grid

    results: List[Dict[str, float]] = []
    for tau_days, last_n in grid:
        dataset = dataset_base.copy(deep=True)
        label_tau = "None" if tau_days is None else f"{tau_days:.0f}"
        label_last = "None" if last_n is None else str(last_n)
        print(f"Evaluating tau={label_tau}, last_n={label_last}...")

        if tau_days is not None or last_n is not None:
            recency_df = compute_recency_features(fighters_raw, tau_days, last_n)
            dataset = merge_recency_features(dataset, recency_df)

        metrics = train_and_score(dataset, args.test_fraction)
        metrics.update({"tau_days": tau_days if tau_days is not None else float("nan"), "last_n": last_n if last_n is not None else float("nan")})
        results.append(metrics)
        print(
            f"  -> log_loss={metrics['log_loss']:.4f} | accuracy={metrics['accuracy']:.3f} | F1={metrics['f1']:.3f}"
        )

    if not results:
        print("No parameter combinations evaluated.")
        return

    results_df = pd.DataFrame(results)
    results_df.sort_values("log_loss", inplace=True)
    display_df = results_df.copy()
    display_df["tau_label"] = display_df["tau_days"].apply(lambda v: "None" if pd.isna(v) else f"{v:.0f}")
    display_df["last_n_label"] = display_df["last_n"].apply(lambda v: "None" if pd.isna(v) else f"{int(v)}")
    columns = ["tau_label", "last_n_label", "log_loss", "accuracy", "f1"]
    print()
    print("Top configurations (sorted by log loss):")
    print(display_df.loc[:, columns].head().to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best = results_df.iloc[0]
    best_tau = None if pd.isna(best["tau_days"]) else best["tau_days"]
    best_window = None if pd.isna(best["last_n"]) else int(best["last_n"])
    tau_text = "None" if best_tau is None else f"{best_tau:.0f}"
    win_text = "None" if best_window is None else str(best_window)
    summary_line = (
        f"Best combo -> tau_days={tau_text}, last_n={win_text}, "
        f"log_loss={best['log_loss']:.4f}, accuracy={best['accuracy']:.3f}, F1={best['f1']:.3f}"
    )
    print()
    print(summary_line)

if __name__ == "__main__":
    main()
