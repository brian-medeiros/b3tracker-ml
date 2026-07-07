from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_yfinance_frame(downloaded: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if isinstance(downloaded.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in downloaded.columns.get_level_values(0):
                continue
            ticker_df = downloaded[ticker].copy()
            ticker_df["ticker"] = ticker
            rows.append(ticker_df)
    else:
        ticker_df = downloaded.copy()
        ticker_df["ticker"] = tickers[0]
        rows.append(ticker_df)

    if not rows:
        raise ValueError("Nenhuma cotacao foi retornada pela fonte de dados.")

    df = pd.concat(rows).reset_index()
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]
    keep = ["ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"]
    df = df[keep].dropna(subset=["date", "close"]).sort_values(["ticker", "date"])
    return df


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
    output_path: Path,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "A dependencia yfinance nao esta instalada. Rode: pip install -r requirements.txt"
        ) from exc

    downloaded = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    df = _normalize_yfinance_frame(downloaded, tickers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def generate_demo_market_data(
    tickers: list[str],
    start_date: str,
    periods: int,
    output_path: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate deterministic OHLCV data for offline demonstration.

    The demo series is not used as evidence of market performance. It exists so the
    pipeline can be executed in environments without internet access.
    """
    rng = np.random.default_rng(random_state)
    dates = pd.bdate_range(start=start_date, periods=periods)
    rows: list[dict[str, float | str | pd.Timestamp]] = []

    market_shock = rng.normal(0.00035, 0.011, size=len(dates))
    cycle = 0.006 * np.sin(np.linspace(0, 8 * math.pi, len(dates)))

    for idx, ticker in enumerate(tickers):
        beta = 0.75 + 0.12 * idx
        idiosyncratic = rng.normal(0, 0.009 + 0.0015 * idx, size=len(dates))
        jump = np.zeros(len(dates))
        jump_points = rng.choice(np.arange(30, len(dates) - 30), size=8, replace=False)
        jump[jump_points] = rng.normal(0, 0.035, size=len(jump_points))
        returns = beta * market_shock + cycle + idiosyncratic + jump

        price = 25 + 8 * idx
        close = price * np.exp(np.cumsum(returns))
        open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.018, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.018, len(dates)))
        volume_base = 2_000_000 + 450_000 * idx
        volume = volume_base * (1 + 12 * np.abs(returns) + rng.uniform(0, 0.8, len(dates)))

        for date, o, h, l, c, v in zip(dates, open_, high, low, close, volume):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "adjusted_close": c,
                    "volume": int(v),
                }
            )

    df = pd.DataFrame(rows).sort_values(["ticker", "date"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_market_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    expected = {"ticker", "date", "open", "high", "low", "close", "adjusted_close", "volume"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Arquivo de dados sem colunas obrigatorias: {sorted(missing)}")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
