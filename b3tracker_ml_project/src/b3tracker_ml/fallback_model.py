from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class _Node:
    proba: np.ndarray
    feature_idx: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


class LightweightRandomForestClassifier:
    """Small Random-Forest-like classifier used only when sklearn is unavailable.

    It supports the subset of the sklearn API used by this project. The official
    reproducible run should use scikit-learn or XGBoost from requirements.txt.
    """

    def __init__(
        self,
        n_estimators: int = 80,
        max_depth: int = 5,
        min_samples_leaf: int = 10,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_classes: int = 3,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_classes = n_classes
        self.trees_: list[_Node] = []
        self.medians_: np.ndarray | None = None
        self.feature_importances_: np.ndarray | None = None
        self.named_steps = {"model": self}

    def _to_numpy(self, x) -> np.ndarray:
        if isinstance(x, (pd.DataFrame, pd.Series)):
            arr = x.to_numpy(dtype=float)
        else:
            arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _impute_fit(self, x: np.ndarray) -> np.ndarray:
        self.medians_ = np.nanmedian(x, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        return self._impute_transform(x)

    def _impute_transform(self, x: np.ndarray) -> np.ndarray:
        if self.medians_ is None:
            raise ValueError("Modelo ainda nao ajustado.")
        x = x.copy()
        rows, cols = np.where(~np.isfinite(x))
        x[rows, cols] = self.medians_[cols]
        return x

    def _class_proba(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.n_classes).astype(float)
        counts += 1.0
        return counts / counts.sum()

    def _gini(self, y: np.ndarray) -> float:
        proba = self._class_proba(y)
        return 1.0 - float(np.sum(proba**2))

    def _candidate_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        size = max(1, int(np.sqrt(n_features))) if self.max_features == "sqrt" else n_features
        return rng.choice(n_features, size=size, replace=False)

    def _build_tree(
        self,
        x: np.ndarray,
        y: np.ndarray,
        depth: int,
        rng: np.random.Generator,
        importance: np.ndarray,
    ) -> _Node:
        node = _Node(proba=self._class_proba(y))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or len(np.unique(y)) == 1:
            return node

        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left = None

        for feature_idx in self._candidate_features(x.shape[1], rng):
            values = x[:, feature_idx]
            if np.unique(values).size < 4:
                continue
            quantiles = rng.choice(np.linspace(0.15, 0.85, 9), size=4, replace=False)
            thresholds = np.unique(np.quantile(values, quantiles))
            for threshold in thresholds:
                left = values <= threshold
                n_left = int(left.sum())
                n_right = int((~left).sum())
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                gain = parent_gini
                gain -= (n_left / len(y)) * self._gini(y[left])
                gain -= (n_right / len(y)) * self._gini(y[~left])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = float(threshold)
                    best_left = left

        if best_feature is None or best_left is None:
            return node

        importance[best_feature] += best_gain * len(y)
        node.feature_idx = int(best_feature)
        node.threshold = float(best_threshold)
        node.left = self._build_tree(x[best_left], y[best_left], depth + 1, rng, importance)
        node.right = self._build_tree(x[~best_left], y[~best_left], depth + 1, rng, importance)
        return node

    def fit(self, x, y):
        rng = np.random.default_rng(self.random_state)
        x_arr = self._impute_fit(self._to_numpy(x))
        y_arr = np.asarray(y, dtype=int)
        n_samples, n_features = x_arr.shape
        importance = np.zeros(n_features, dtype=float)
        self.trees_ = []

        class_indices = [np.where(y_arr == class_id)[0] for class_id in range(self.n_classes)]
        non_empty = [idx for idx in class_indices if len(idx) > 0]
        per_class = max(1, n_samples // max(1, len(non_empty)))

        for _ in range(self.n_estimators):
            sampled_parts = [rng.choice(indices, size=per_class, replace=True) for indices in non_empty]
            sample_idx = np.concatenate(sampled_parts)
            rng.shuffle(sample_idx)
            tree = self._build_tree(x_arr[sample_idx], y_arr[sample_idx], 0, rng, importance)
            self.trees_.append(tree)

        total = importance.sum()
        self.feature_importances_ = importance / total if total > 0 else np.ones(n_features) / n_features
        return self

    def _predict_one_tree(self, tree: _Node, row: np.ndarray) -> np.ndarray:
        node = tree
        while node.feature_idx is not None and node.threshold is not None:
            if row[node.feature_idx] <= node.threshold:
                node = node.left if node.left is not None else node
            else:
                node = node.right if node.right is not None else node
        return node.proba

    def predict_proba(self, x) -> np.ndarray:
        x_arr = self._impute_transform(self._to_numpy(x))
        predictions = np.zeros((x_arr.shape[0], self.n_classes), dtype=float)
        for tree in self.trees_:
            predictions += np.vstack([self._predict_one_tree(tree, row) for row in x_arr])
        predictions /= max(1, len(self.trees_))
        predictions /= predictions.sum(axis=1, keepdims=True)
        return predictions

    def predict(self, x) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)
