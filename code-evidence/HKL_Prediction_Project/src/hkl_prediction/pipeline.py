from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from hkl_prediction.features import (
    build_preprocessor,
    clean_race_data,
    feature_matrix,
    make_race_key,
    metadata,
)


DEFAULT_HISTORICAL = Path("DataSets/Prediction/historical_races.csv")
DEFAULT_UPCOMING = Path("DataSets/Prediction/new_upcoming_races.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class PipelinePaths:
    historical: Path = DEFAULT_HISTORICAL
    upcoming: Path = DEFAULT_UPCOMING
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.csv"

    @property
    def predictions_csv_path(self) -> Path:
        return self.output_dir / "predictions.csv"

    @property
    def predictions_xlsx_path(self) -> Path:
        return self.output_dir / "predictions.xlsx"

    @property
    def meta_path(self) -> Path:
        return self.model_dir / "meta.json"

    @property
    def thresholds_path(self) -> Path:
        return self.model_dir / "thresholds.json"

    @property
    def evaluation_report_path(self) -> Path:
        return self.output_dir / "evaluation_report.md"


def run_pipeline(paths: PipelinePaths, sample_size: int | None = None, random_state: int = 42) -> None:
    trained_models, thresholds = train_models(paths, sample_size=sample_size, random_state=random_state)
    predict_upcoming(paths, models=trained_models, thresholds=thresholds)


def train_models(
    paths: PipelinePaths,
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], dict[str, float]]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    historical = pd.read_csv(paths.historical, low_memory=False)
    data = clean_race_data(historical, require_target=True)
    data = data.dropna(subset=["Date", "Race"])
    data = data.sort_values(["Date", "Race"]).reset_index(drop=True)

    if sample_size and sample_size > 0 and sample_size < len(data):
        data = data.tail(sample_size).reset_index(drop=True)

    train_df, valid_df = time_aware_split(data, validation_fraction=0.2)

    X_train = feature_matrix(train_df)
    y_train = train_df["HitTop4"].astype(int)
    X_valid = feature_matrix(valid_df)
    y_valid = valid_df["HitTop4"].astype(int)

    meta = metadata()
    models = build_models(random_state=random_state)
    thresholds: dict[str, float] = {}
    metrics_rows: list[dict] = []
    trained: dict[str, Pipeline] = {}

    for name, estimator in models.items():
        model = Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(
                        meta.numeric_features,
                        meta.low_card_features,
                        meta.high_card_features,
                    ),
                ),
                ("model", estimator),
            ]
        )
        model.fit(X_train, y_train)
        probabilities = predict_positive_probability(model, X_valid)
        threshold, threshold_f1 = tune_threshold(probabilities, y_valid.to_numpy())
        thresholds[name] = threshold
        metrics_rows.append(metric_row(name, probabilities, y_valid.to_numpy(), threshold, threshold_f1))
        joblib.dump(model, paths.model_dir / f"{name}.joblib")
        trained[name] = model

    ensemble_probabilities = average_probabilities([predict_positive_probability(m, X_valid) for m in trained.values()])
    threshold, threshold_f1 = tune_threshold(ensemble_probabilities, y_valid.to_numpy())
    thresholds["ensemble_soft"] = threshold
    metrics_rows.append(
        metric_row("ensemble_soft", ensemble_probabilities, y_valid.to_numpy(), threshold, threshold_f1)
    )

    validation_positive_rate = float(y_valid.mean())
    majority_class = int(validation_positive_rate >= 0.5)
    save_json(paths.thresholds_path, thresholds)
    save_json(
        paths.meta_path,
        {
            **meta.to_dict(),
            "target_definition": "FinishPosition <= 4",
            "validation_strategy": "Chronological holdout of the final 20% of races",
            "historical_rows": int(len(data)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(valid_df)),
            "training_date_start": format_date(train_df["Date"].min()),
            "training_date_end": format_date(train_df["Date"].max()),
            "validation_date_start": format_date(valid_df["Date"].min()),
            "validation_date_end": format_date(valid_df["Date"].max()),
            "training_positive_rows": int(y_train.sum()),
            "validation_positive_rows": int(y_valid.sum()),
            "training_positive_rate": float(y_train.mean()),
            "validation_positive_rate": validation_positive_rate,
            "majority_class": majority_class,
            "majority_baseline_accuracy": max(
                validation_positive_rate,
                1.0 - validation_positive_rate,
            ),
            "random_state": random_state,
            "sample_size": sample_size,
            "generated_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        },
    )
    pd.DataFrame(metrics_rows).to_csv(paths.metrics_path, index=False)
    from hkl_prediction.reporting import generate_markdown_report

    generate_markdown_report(
        metrics_path=paths.metrics_path,
        meta_path=paths.meta_path,
        report_path=paths.evaluation_report_path,
    )
    return trained, thresholds


def predict_upcoming(
    paths: PipelinePaths,
    models: dict[str, Pipeline] | None = None,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = load_models(paths.model_dir)
    if thresholds is None:
        thresholds = load_json(paths.thresholds_path)

    upcoming = pd.read_csv(paths.upcoming, low_memory=False)
    data = clean_race_data(upcoming, require_target=False)
    X_pred = feature_matrix(data)

    output = data[["Date", "Race", "HorseName", "Jockey", "Trainer", "WinOdds"]].copy()
    output["Date"] = output["Date"].dt.strftime("%Y-%m-%d")

    probability_columns = []
    for name, model in models.items():
        probabilities = predict_positive_probability(model, X_pred)
        probability_col = f"{name}_proba"
        rank_col = f"{name}_rank"
        threshold_col = f"{name}_above_threshold"
        top4_col = f"{name}_rank_top4"
        output[probability_col] = probabilities
        output[rank_col] = rank_within_race(data, probabilities)
        output[threshold_col] = probabilities >= thresholds.get(name, 0.5)
        output[top4_col] = output[rank_col] <= 4
        probability_columns.append(probability_col)

    if probability_columns:
        output["ensemble_soft_proba"] = output[probability_columns].mean(axis=1)
        output["ensemble_soft_rank"] = rank_within_race(data, output["ensemble_soft_proba"].to_numpy())
        output["ensemble_soft_above_threshold"] = output["ensemble_soft_proba"] >= thresholds.get(
            "ensemble_soft", 0.5
        )
        output["ensemble_soft_rank_top4"] = output["ensemble_soft_rank"] <= 4

    sort_cols = ["Date", "Race", "ensemble_soft_rank" if "ensemble_soft_rank" in output else probability_columns[0]]
    output = output.sort_values(sort_cols).reset_index(drop=True)
    output.to_csv(paths.predictions_csv_path, index=False)
    output.to_excel(paths.predictions_xlsx_path, index=False)
    return output


def build_models(random_state: int) -> dict[str, object]:
    return {
        "log_reg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=1,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=160,
            l2_regularization=0.01,
            random_state=random_state,
        ),
    }


def time_aware_split(
    data: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    race_keys = make_race_key(data)
    ordered_races = race_keys.drop_duplicates().tolist()
    if len(ordered_races) < 2:
        raise ValueError("At least two races are required for a train/validation split.")

    split_index = min(
        len(ordered_races) - 1,
        max(1, int(len(ordered_races) * (1.0 - validation_fraction))),
    )
    train_races = set(ordered_races[:split_index])
    train_df = data.loc[race_keys.isin(train_races)].copy()
    valid_df = data.loc[~race_keys.isin(train_races)].copy()

    if train_df["HitTop4"].nunique() < 2 or valid_df["HitTop4"].nunique() < 2:
        shuffled_races = pd.Series(ordered_races).sample(frac=1.0, random_state=42).tolist()
        train_races = set(shuffled_races[:split_index])
        train_df = data.loc[race_keys.isin(train_races)].copy()
        valid_df = data.loc[~race_keys.isin(train_races)].copy()

    return train_df, valid_df


def predict_positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def tune_threshold(probabilities: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        labels = probabilities >= threshold
        score = f1_score(y_true, labels, zero_division=0)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def metric_row(
    name: str,
    probabilities: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    threshold_f1: float,
) -> dict:
    labels = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    try:
        roc_auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        roc_auc = np.nan
    return {
        "model": name,
        "threshold": threshold,
        "threshold_f1": threshold_f1,
        "roc_auc": roc_auc,
        "accuracy": accuracy_score(y_true, labels),
        "balanced_accuracy": balanced_accuracy_score(y_true, labels),
        "precision": precision_score(y_true, labels, zero_division=0),
        "recall": recall_score(y_true, labels, zero_division=0),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "predicted_positive_rate": float(np.mean(labels)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "validation_rows": int(len(y_true)),
    }


def average_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    return np.vstack(probabilities).mean(axis=0)


def rank_within_race(df: pd.DataFrame, probabilities: np.ndarray) -> pd.Series:
    temp = pd.DataFrame(
        {
            "RaceKey": make_race_key(df),
            "Probability": probabilities,
        },
        index=df.index,
    )
    return temp.groupby("RaceKey")["Probability"].rank(method="first", ascending=False).astype(int)


def load_models(model_dir: Path) -> dict[str, Pipeline]:
    model_paths = sorted(model_dir.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No saved models found in {model_dir}. Run training first.")
    return {path.stem: joblib.load(path) for path in model_paths}


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def format_date(value: object) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")
