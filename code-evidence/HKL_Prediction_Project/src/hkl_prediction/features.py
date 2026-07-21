from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMNS = ["Date", "Race", "HorseName"]

BASE_FEATURE_COLUMNS = [
    "Jockey",
    "Trainer",
    "JockeyWeight",
    "HorseWeight",
    "LaneNo",
    "WinOdds",
    "Location",
    "Going",
    "Course",
    "TotalBet",
    "ClassNo",
    "Distance",
    "Age",
    "HorseRating",
]

NUMERIC_COLUMNS = [
    "JockeyWeight",
    "HorseWeight",
    "LaneNo",
    "WinOdds",
    "TotalBet",
    "ClassNo",
    "Distance",
    "Age",
    "HorseRating",
]

LOW_CARD_COLUMNS = ["Location", "Going", "Course"]
HIGH_CARD_COLUMNS = ["Jockey", "Trainer"]

ENGINEERED_NUMERIC_COLUMNS = [
    "FieldSize",
    "OddsRankInRace",
    "RatingRankInRace",
    "DrawRelativeToField",
    "WeightDelta",
    "RatingToWeight",
    "ImpliedWinProbability",
]

MODEL_FEATURE_COLUMNS = (
    NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS + LOW_CARD_COLUMNS + HIGH_CARD_COLUMNS
)


@dataclass(frozen=True)
class FeatureMetadata:
    model_features: list[str]
    numeric_features: list[str]
    low_card_features: list[str]
    high_card_features: list[str]
    target: str = "HitTop4"

    def to_dict(self) -> dict:
        return {
            "model_features": self.model_features,
            "numeric_features": self.numeric_features,
            "low_card_features": self.low_card_features,
            "high_card_features": self.high_card_features,
            "target": self.target,
        }


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode high-cardinality categories by their training-set frequency."""

    def __init__(self) -> None:
        self.frequencies_: dict[str, dict[str, float]] = {}
        self.columns_: list[str] = []

    def fit(self, X, y=None):  # noqa: D401
        frame = _as_frame(X)
        self.columns_ = list(frame.columns)
        self.frequencies_ = {}
        for column in self.columns_:
            values = frame[column].fillna("__MISSING__").astype(str)
            self.frequencies_[column] = values.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        frame = _as_frame(X)
        encoded = pd.DataFrame(index=frame.index)
        for column in self.columns_:
            values = frame[column].fillna("__MISSING__").astype(str)
            encoded[column] = values.map(self.frequencies_.get(column, {})).fillna(0.0)
        return encoded.to_numpy(dtype=float)


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X)


def one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip() for column in df.columns}
    df = df.rename(columns=renamed)
    compatibility = {
        "Win Odds": "WinOdds",
        "Finish Time": "FinishTime",
        "Act. Wt.": "JockeyWeight",
        "Declar. Horse Wt.": "HorseWeight",
        "Dr.": "LaneNo",
        "Pla.": "FinishPosition",
        "Horse": "HorseName",
        "date": "Date",
        "race": "Race",
        "location": "Location",
        "going": "Going",
        "course": "Course",
        "bet": "TotalBet",
        "class": "ClassNo",
    }
    return df.rename(columns={k: v for k, v in compatibility.items() if k in df.columns})


def parse_dates_safe(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.strip()
    parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        fallback = pd.to_datetime(raw[missing], errors="coerce", dayfirst=False)
        parsed.loc[missing] = fallback
    return parsed


def clean_race_data(df: pd.DataFrame, require_target: bool) -> pd.DataFrame:
    df = normalise_columns(df.copy())

    for column in ID_COLUMNS + BASE_FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["Date"] = parse_dates_safe(df["Date"])
    df["Race"] = pd.to_numeric(df["Race"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in LOW_CARD_COLUMNS + HIGH_CARD_COLUMNS + ["HorseName"]:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
        df[column] = df[column].replace("", "Unknown")

    if require_target:
        if "FinishPosition" not in df.columns:
            raise ValueError("Historical data must contain FinishPosition.")
        df["FinishPosition"] = pd.to_numeric(df["FinishPosition"], errors="coerce")
        df = df.dropna(subset=["FinishPosition"])
        df["HitTop4"] = (df["FinishPosition"] <= 4).astype(int)

    return add_engineered_features(df)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    race_key = make_race_key(df)

    df["FieldSize"] = df.groupby(race_key)["HorseName"].transform("count")
    df["OddsRankInRace"] = df.groupby(race_key)["WinOdds"].rank(method="average", ascending=True)
    df["RatingRankInRace"] = df.groupby(race_key)["HorseRating"].rank(
        method="average", ascending=False
    )
    df["DrawRelativeToField"] = df["LaneNo"] / df["FieldSize"].replace(0, np.nan)
    df["WeightDelta"] = df["HorseWeight"] - df["JockeyWeight"]
    df["RatingToWeight"] = df["HorseRating"] / df["JockeyWeight"].replace(0, np.nan)
    df["ImpliedWinProbability"] = 1.0 / df["WinOdds"].replace(0, np.nan)

    return df


def make_race_key(df: pd.DataFrame) -> pd.Series:
    date_part = df["Date"].dt.strftime("%Y-%m-%d").fillna("unknown-date")
    race_part = df["Race"].fillna(-1).astype(int).astype(str)
    return date_part + "_R" + race_part


def metadata() -> FeatureMetadata:
    return FeatureMetadata(
        model_features=list(MODEL_FEATURE_COLUMNS),
        numeric_features=list(NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS),
        low_card_features=list(LOW_CARD_COLUMNS),
        high_card_features=list(HIGH_CARD_COLUMNS),
    )


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    meta = metadata()
    for column in meta.model_features:
        if column not in df.columns:
            df[column] = np.nan
    return df[meta.model_features].copy()


def build_preprocessor(
    numeric_features: Iterable[str],
    low_card_features: Iterable[str],
    high_card_features: Iterable[str],
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    low_card_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", one_hot_encoder()),
        ]
    )
    high_card_pipeline = Pipeline(
        steps=[
            ("freq", FrequencyEncoder()),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_features)),
            ("low_card", low_card_pipeline, list(low_card_features)),
            ("high_card", high_card_pipeline, list(high_card_features)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

