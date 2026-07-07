from __future__ import annotations

from pathlib import Path

import pandas as pd

from b3tracker_ml.config import CLASS_ORDER


def plot_class_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        distribution.to_csv(output_path.with_suffix(".csv"), index=False)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=distribution, x="classe", y="quantidade", order=CLASS_ORDER, color="#2F6B9A")
    plt.title("Distribuicao das classes")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(confusion_matrix: pd.DataFrame, output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        confusion_matrix.to_csv(output_path.with_suffix(".csv"))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_feature_importance(model, feature_columns: list[str], output_path: Path, top_n: int = 20) -> pd.DataFrame:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        estimator = model.named_steps["model"]
        importance = pd.DataFrame(
            {"feature": feature_columns, "importance": estimator.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance.to_csv(output_path.with_suffix(".csv"), index=False)
        return importance

    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Modelo nao possui atributo feature_importances_.")
    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": estimator.feature_importances_}
    ).sort_values("importance", ascending=False)
    top = importance.head(top_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top, y="feature", x="importance", color="#3A7D44")
    plt.title(f"Top {top_n} features por importancia")
    plt.xlabel("Importancia")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return importance
