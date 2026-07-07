from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from b3tracker_ml.config import CLASS_ORDER


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, pd.DataFrame]:
    try:
        from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
    except ImportError:
        labels = list(range(len(CLASS_ORDER)))
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for true_id, pred_id in zip(y_true, y_pred):
            cm[int(true_id), int(pred_id)] += 1

        recalls: list[float] = []
        f1s: list[float] = []
        supports: list[int] = []
        by_class: dict[str, dict[str, float]] = {}
        for idx, label in enumerate(CLASS_ORDER):
            tp = cm[idx, idx]
            fp = cm[:, idx].sum() - tp
            fn = cm[idx, :].sum() - tp
            support = int(cm[idx, :].sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            by_class[label] = {"precision": precision, "recall": recall, "f1": f1}
            recalls.append(float(recall))
            f1s.append(float(f1))
            supports.append(support)

        support_arr = np.asarray(supports, dtype=float)
        weighted_f1 = float(np.average(f1s, weights=support_arr)) if support_arr.sum() else 0.0
        metrics = {
            "balanced_accuracy": float(np.mean(recalls)),
            "precision_compra": float(by_class["COMPRA"]["precision"]),
            "recall_compra": float(by_class["COMPRA"]["recall"]),
            "f1_compra": float(by_class["COMPRA"]["f1"]),
            "precision_venda": float(by_class["VENDA"]["precision"]),
            "recall_venda": float(by_class["VENDA"]["recall"]),
            "f1_venda": float(by_class["VENDA"]["f1"]),
            "macro_f1": float(np.mean(f1s)),
            "weighted_f1": weighted_f1,
        }
        cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
        return metrics, cm_df

    labels = list(range(len(CLASS_ORDER)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_compra": float(report["COMPRA"]["precision"]),
        "recall_compra": float(report["COMPRA"]["recall"]),
        "f1_compra": float(report["COMPRA"]["f1-score"]),
        "precision_venda": float(report["VENDA"]["precision"]),
        "recall_venda": float(report["VENDA"]["recall"]),
        "f1_venda": float(report["VENDA"]["f1-score"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
    return metrics, cm_df


def save_metrics(metrics: dict, confusion_matrix: pd.DataFrame, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{prefix}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    confusion_matrix.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")


def threshold_sweep(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    apply_threshold_fn,
    buy_values: list[float],
    sell_values: list[float],
    min_margin: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for buy_threshold in buy_values:
        for sell_threshold in sell_values:
            y_pred = apply_threshold_fn(probabilities, buy_threshold, sell_threshold, min_margin)
            metrics, _ = evaluate_predictions(y_true, y_pred)
            rows.append(
                {
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "min_margin": min_margin,
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["precision_compra", "precision_venda", "macro_f1"], ascending=False)
