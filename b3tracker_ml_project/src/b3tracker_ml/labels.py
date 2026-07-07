from __future__ import annotations

import numpy as np
import pandas as pd

from b3tracker_ml.config import CLASS_TO_ID


def add_future_labels(
    df: pd.DataFrame,
    horizon: int,
    min_abs_return: float,
    volatility_multiplier: float,
) -> pd.DataFrame:
    labeled = df.sort_values(["ticker", "date"]).copy()
    future_price = labeled.groupby("ticker")["adjusted_close"].shift(-horizon)
    labeled["future_return"] = future_price / labeled["adjusted_close"] - 1
    horizon_vol = labeled["vol_20d_daily"] * np.sqrt(horizon)
    labeled["dynamic_threshold"] = np.maximum(min_abs_return, volatility_multiplier * horizon_vol)

    labeled["target"] = "NEUTRO"
    labeled.loc[labeled["future_return"] >= labeled["dynamic_threshold"], "target"] = "COMPRA"
    labeled.loc[labeled["future_return"] <= -labeled["dynamic_threshold"], "target"] = "VENDA"
    labeled["target_id"] = labeled["target"].map(CLASS_TO_ID)
    return labeled.dropna(subset=["future_return", "dynamic_threshold", "target_id"])
