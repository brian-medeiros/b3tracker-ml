from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from b3tracker_ml.config import CLASS_ORDER, CLASS_TO_ID


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def temporal_split(df: pd.DataFrame, validation_size: float, test_size: float) -> SplitData:
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) < 10:
        raise ValueError("Poucas datas para separacao temporal.")

    test_start_idx = int(len(dates) * (1 - test_size))
    validation_start_idx = int(len(dates) * (1 - test_size - validation_size))
    validation_start = dates[validation_start_idx]
    test_start = dates[test_start_idx]

    train = df[df["date"] < validation_start].copy()
    validation = df[(df["date"] >= validation_start) & (df["date"] < test_start)].copy()
    test = df[df["date"] >= test_start].copy()
    return SplitData(train=train, validation=validation, test=test)


def build_model(model_name: str, random_state: int):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        if model_name != "random_forest":
            raise ImportError(
                "scikit-learn nao esta instalado. Use random_forest para fallback ou instale requirements.txt"
            ) from exc
        from b3tracker_ml.fallback_model import LightweightRandomForestClassifier

        return LightweightRandomForestClassifier(random_state=random_state)

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=7,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost nao esta instalado. Use --model random_forest ou instale requirements.txt"
            ) from exc
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError("Modelo invalido. Use random_forest ou xgboost.")

    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def apply_risk_threshold(
    probabilities: np.ndarray,
    buy_threshold: float,
    sell_threshold: float,
    min_margin: float,
) -> np.ndarray:
    """Convert probabilities into risk-aware alerts.

    A compra or venda is emitted only when the probability and directional margin
    are high enough. Otherwise, the decision remains neutral.
    """
    neutral_id = CLASS_TO_ID["NEUTRO"]
    buy_id = CLASS_TO_ID["COMPRA"]
    sell_id = CLASS_TO_ID["VENDA"]
    output = np.full(probabilities.shape[0], neutral_id, dtype=int)

    buy_prob = probabilities[:, buy_id]
    sell_prob = probabilities[:, sell_id]
    buy_mask = (buy_prob >= buy_threshold) & ((buy_prob - sell_prob) >= min_margin)
    sell_mask = (sell_prob >= sell_threshold) & ((sell_prob - buy_prob) >= min_margin)
    output[buy_mask] = buy_id
    output[sell_mask] = sell_id
    return output


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["target"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    percent = counts / counts.sum()
    return pd.DataFrame({"classe": counts.index, "quantidade": counts.values, "percentual": percent.values})
