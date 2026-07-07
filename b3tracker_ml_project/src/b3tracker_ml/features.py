from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _consecutive_count(condition: pd.Series) -> pd.Series:
    groups = condition.ne(condition.shift()).cumsum()
    return condition.groupby(groups).cumcount().add(1).where(condition, 0)


def _feature_block(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    close = group["adjusted_close"].astype(float)
    high = group["high"].astype(float)
    low = group["low"].astype(float)
    volume = group["volume"].astype(float)
    log_return = np.log(close / close.shift(1))

    group["retorno_1d"] = close.pct_change(1)
    group["retorno_3d"] = close.pct_change(3)
    group["retorno_5d"] = close.pct_change(5)
    group["retorno_10d"] = close.pct_change(10)
    group["retorno_20d"] = close.pct_change(20)
    group["log_return_1d"] = log_return

    for window in [5, 10, 20, 50, 200]:
        sma = close.rolling(window).mean()
        group[f"mm_{window}"] = sma
        group[f"dist_mm_{window}"] = close / sma - 1

    group["mm_5_acima_mm_20"] = (group["mm_5"] > group["mm_20"]).astype(int)
    group["mm_20_acima_mm_50"] = (group["mm_20"] > group["mm_50"]).astype(int)
    group["dist_mm5_mm20"] = group["mm_5"] / group["mm_20"] - 1
    group["dist_mm20_mm50"] = group["mm_20"] / group["mm_50"] - 1

    group["rsi_7"] = _rsi(close, 7)
    group["rsi_14"] = _rsi(close, 14)
    group["rsi_14_menor_30"] = (group["rsi_14"] < 30).astype(int)
    group["rsi_14_maior_70"] = (group["rsi_14"] > 70).astype(int)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    group["macd"] = ema_12 - ema_26
    group["macd_signal"] = group["macd"].ewm(span=9, adjust=False).mean()
    group["macd_hist"] = group["macd"] - group["macd_signal"]
    group["macd_acima_signal"] = (group["macd"] > group["macd_signal"]).astype(int)

    for window in [5, 10, 20, 60]:
        group[f"vol_{window}d_daily"] = log_return.rolling(window).std()
        group[f"vol_{window}d_annual"] = group[f"vol_{window}d_daily"] * np.sqrt(252)

    group["vol_ratio_5_20"] = group["vol_5d_daily"] / group["vol_20d_daily"]
    group["amplitude_diaria"] = (high - low) / close

    bollinger_mean = close.rolling(20).mean()
    bollinger_std = close.rolling(20).std()
    upper = bollinger_mean + 2 * bollinger_std
    lower = bollinger_mean - 2 * bollinger_std
    group["bollinger_position"] = (close - lower) / (upper - lower)
    group["bollinger_width"] = (upper - lower) / bollinger_mean

    group["volume_medio_5d"] = volume.rolling(5).mean()
    group["volume_medio_20d"] = volume.rolling(20).mean()
    group["volume_relativo"] = volume / group["volume_medio_20d"]
    group["volume_zscore_20d"] = (volume - group["volume_medio_20d"]) / volume.rolling(20).std()
    group["retorno_x_volume"] = group["retorno_1d"] * group["volume_relativo"]

    group["dias_consecutivos_alta"] = _consecutive_count(close.diff() > 0)
    group["dias_consecutivos_queda"] = _consecutive_count(close.diff() < 0)
    max_20d = close.rolling(20).max()
    min_20d = close.rolling(20).min()
    group["dist_max_20d"] = close / max_20d - 1
    group["dist_min_20d"] = close / min_20d - 1
    group["drawdown_20d"] = close / max_20d - 1

    group["dia_da_semana"] = group["date"].dt.dayofweek
    group["mes"] = group["date"].dt.month
    group["inicio_mes"] = (group["date"].dt.day <= 5).astype(int)
    group["fim_mes"] = (group["date"].dt.day >= 25).astype(int)
    return group


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.groupby("ticker", group_keys=False).apply(_feature_block, include_groups=True)
    return featured.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_market_context(df: pd.DataFrame, market_ticker: str) -> pd.DataFrame:
    market = (
        df[df["ticker"] == market_ticker][
            ["date", "retorno_1d", "retorno_5d", "vol_20d_daily", "log_return_1d"]
        ]
        .rename(
            columns={
                "retorno_1d": "retorno_ibov_1d",
                "retorno_5d": "retorno_ibov_5d",
                "vol_20d_daily": "vol_ibov_20d",
                "log_return_1d": "market_log_return_1d",
            }
        )
        .copy()
    )
    merged = df.merge(market, on="date", how="left")

    def beta(group: pd.DataFrame) -> pd.Series:
        cov = group["log_return_1d"].rolling(60).cov(group["market_log_return_1d"])
        var = group["market_log_return_1d"].rolling(60).var()
        return cov / var

    merged["beta_60d"] = merged.groupby("ticker", group_keys=False).apply(beta)
    return merged.drop(columns=["market_log_return_1d"])


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "future_return",
        "dynamic_threshold",
        "target",
        "target_id",
    }
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns
    return [col for col in numeric_cols if col not in excluded]
