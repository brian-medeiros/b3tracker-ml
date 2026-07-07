from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from b3tracker_ml.config import (  # noqa: E402
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ProjectConfig,
    ensure_directories,
)
from b3tracker_ml.data import download_market_data, generate_demo_market_data, load_market_data  # noqa: E402
from b3tracker_ml.evaluation import evaluate_predictions, save_metrics, threshold_sweep  # noqa: E402
from b3tracker_ml.features import add_market_context, add_technical_features, get_feature_columns  # noqa: E402
from b3tracker_ml.inference import predict_latest_alerts  # noqa: E402
from b3tracker_ml.labels import add_future_labels  # noqa: E402
from b3tracker_ml.modeling import apply_risk_threshold, build_model, class_distribution, temporal_split  # noqa: E402
from b3tracker_ml.plotting import (  # noqa: E402
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline ML do B3Tracker.")
    parser.add_argument("--source", choices=["yfinance", "csv", "demo"], default="demo")
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--min-abs-return", type=float, default=0.02)
    parser.add_argument("--volatility-multiplier", type=float, default=1.0)
    parser.add_argument("--buy-threshold", type=float, default=0.60)
    parser.add_argument("--sell-threshold", type=float, default=0.60)
    parser.add_argument("--min-margin", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    config = ProjectConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        prediction_horizon=args.horizon,
        min_abs_return=args.min_abs_return,
        volatility_multiplier=args.volatility_multiplier,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        min_margin=args.min_margin,
    )

    raw_path = RAW_DIR / "cotacoes_b3.csv"
    if args.source == "yfinance":
        raw = download_market_data(config.tickers, config.start_date, config.end_date, raw_path)
    elif args.source == "csv":
        if args.csv_path is None:
            raise ValueError("Informe --csv-path quando usar --source csv.")
        raw = load_market_data(args.csv_path)
    else:
        raw = generate_demo_market_data(
            config.tickers,
            config.start_date,
            periods=1250,
            output_path=raw_path,
            random_state=config.random_state,
        )

    featured = add_technical_features(raw)
    featured = add_market_context(featured, config.market_ticker)
    labeled = add_future_labels(
        featured,
        horizon=config.prediction_horizon,
        min_abs_return=config.min_abs_return,
        volatility_multiplier=config.volatility_multiplier,
    )
    labeled = labeled.replace([float("inf"), float("-inf")], pd.NA).dropna()

    dataset_path = PROCESSED_DIR / "dataset_modelagem.csv"
    labeled.to_csv(dataset_path, index=False)

    feature_columns = get_feature_columns(labeled)
    with (PROCESSED_DIR / "feature_columns.json").open("w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2, ensure_ascii=False)

    split = temporal_split(labeled, config.validation_size, config.test_size)
    distribution = class_distribution(labeled)
    distribution.to_csv(PROCESSED_DIR / "class_distribution.csv", index=False)
    plot_class_distribution(distribution, FIGURES_DIR / "class_distribution.png")

    model = build_model(args.model, config.random_state)
    x_train = split.train[feature_columns]
    y_train = split.train["target_id"].astype(int)
    model.fit(x_train, y_train)

    model_path = MODELS_DIR / f"{args.model}_b3tracker.joblib"
    try:
        import joblib

        joblib.dump({"model": model, "feature_columns": feature_columns, "config": config}, model_path)
    except ImportError:
        import pickle

        model_path = model_path.with_suffix(".pkl")
        with model_path.open("wb") as file:
            pickle.dump({"model": model, "feature_columns": feature_columns, "config": config}, file)

    for name, frame in [("validation", split.validation), ("test", split.test)]:
        probabilities = model.predict_proba(frame[feature_columns])
        y_pred_raw = model.predict(frame[feature_columns])
        y_pred_threshold = apply_risk_threshold(
            probabilities,
            buy_threshold=config.buy_threshold,
            sell_threshold=config.sell_threshold,
            min_margin=config.min_margin,
        )
        y_true = frame["target_id"].astype(int).to_numpy()

        raw_metrics, raw_cm = evaluate_predictions(y_true, y_pred_raw)
        threshold_metrics, threshold_cm = evaluate_predictions(y_true, y_pred_threshold)
        save_metrics(raw_metrics, raw_cm, PROCESSED_DIR, f"{name}_raw")
        save_metrics(threshold_metrics, threshold_cm, PROCESSED_DIR, f"{name}_risk_threshold")
        plot_confusion_matrix(raw_cm, FIGURES_DIR / f"{name}_raw_confusion_matrix.png", f"{name}: matriz sem limiar")
        plot_confusion_matrix(
            threshold_cm,
            FIGURES_DIR / f"{name}_risk_threshold_confusion_matrix.png",
            f"{name}: matriz com limiar de risco",
        )

    validation_probabilities = model.predict_proba(split.validation[feature_columns])
    sweep = threshold_sweep(
        validation_probabilities,
        split.validation["target_id"].astype(int).to_numpy(),
        apply_threshold_fn=apply_risk_threshold,
        buy_values=[0.50, 0.55, 0.60, 0.65, 0.70],
        sell_values=[0.50, 0.55, 0.60, 0.65, 0.70],
        min_margin=config.min_margin,
    )
    sweep.to_csv(PROCESSED_DIR / "threshold_sweep_validation.csv", index=False)

    importance = plot_feature_importance(model, feature_columns, FIGURES_DIR / "feature_importance.png")
    importance.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)

    alerts = predict_latest_alerts(
        model,
        labeled,
        feature_columns,
        buy_threshold=config.buy_threshold,
        sell_threshold=config.sell_threshold,
        min_margin=config.min_margin,
    )
    alerts_df = pd.DataFrame([alert.__dict__ for alert in alerts])
    alerts_df.to_csv(PROCESSED_DIR / "latest_alerts.csv", index=False)

    summary = {
        "source": args.source,
        "model": args.model,
        "rows": int(len(labeled)),
        "features": int(len(feature_columns)),
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
    }
    with (PROCESSED_DIR / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
