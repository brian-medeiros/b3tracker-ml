from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"


@dataclass(frozen=True)
class ProjectConfig:
    tickers: list[str] = field(
        default_factory=lambda: [
            "PETR4.SA",
            "VALE3.SA",
            "ITUB4.SA",
            "BBDC4.SA",
            "BBAS3.SA",
            "BOVA11.SA",
        ]
    )
    market_ticker: str = "BOVA11.SA"
    start_date: str = "2020-01-01"
    end_date: str | None = None
    prediction_horizon: int = 5
    min_abs_return: float = 0.02
    volatility_multiplier: float = 1.0
    random_state: int = 42
    validation_size: float = 0.15
    test_size: float = 0.15
    buy_threshold: float = 0.60
    sell_threshold: float = 0.60
    min_margin: float = 0.15


CLASS_ORDER = ["NEUTRO", "COMPRA", "VENDA"]
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASS_ORDER)}
ID_TO_CLASS = {idx: label for label, idx in CLASS_TO_ID.items()}


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, FIGURES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
