from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from b3tracker_ml.config import CLASS_ORDER, CLASS_TO_ID
from b3tracker_ml.modeling import apply_risk_threshold


@dataclass(frozen=True)
class AlertDecision:
    ticker: str
    date: str
    decision: str
    probability_neutral: float
    probability_buy: float
    probability_sell: float


def predict_latest_alerts(
    model,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    buy_threshold: float,
    sell_threshold: float,
    min_margin: float,
) -> list[AlertDecision]:
    latest = dataset.sort_values("date").groupby("ticker").tail(1).copy()
    probabilities = model.predict_proba(latest[feature_columns])
    predicted_ids = apply_risk_threshold(probabilities, buy_threshold, sell_threshold, min_margin)

    decisions: list[AlertDecision] = []
    for row, probs, pred_id in zip(latest.itertuples(index=False), probabilities, predicted_ids):
        decisions.append(
            AlertDecision(
                ticker=row.ticker,
                date=str(row.date.date()),
                decision=CLASS_ORDER[int(pred_id)],
                probability_neutral=float(probs[CLASS_TO_ID["NEUTRO"]]),
                probability_buy=float(probs[CLASS_TO_ID["COMPRA"]]),
                probability_sell=float(probs[CLASS_TO_ID["VENDA"]]),
            )
        )
    return decisions
