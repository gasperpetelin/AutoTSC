"""Utility functions for AutoTSC."""

import numpy as np
from aeon.datasets import load_classification
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.extmath import softmax
from threadpoolctl import threadpool_limits


class RidgeClassifierCVDecisionProba(RidgeClassifierCV):
    """RidgeClassifierCV with softmax probabilities and a thread budget.

    ``n_jobs`` is a BLAS thread count, not process parallelism: sklearn's
    RidgeClassifierCV has no ``n_jobs`` of its own, and with ``cv=None`` it takes the
    leave-one-out GCV path, which is pure BLAS -- so the thread pool is the only lever.
    ``-1`` lifts the limit entirely.

    The default of 1 is what the stack fits with, since the outer loop already
    parallelises over folds and models and a wide inner ridge would oversubscribe.
    """

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        store_cv_results=False,
        n_jobs=1,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring,
            cv=cv,
            class_weight=class_weight,
            store_cv_results=store_cv_results,
        )
        self.n_jobs = n_jobs

    def fit(self, X, y):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return super().fit(X, y)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return softmax(scores)


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "This feature requires PyTorch. Install with:\n"
            "  pip install 'tscglue[torch]'\n"
            "  uv pip install 'tscglue[cpu]'\n"
            "  uv pip install 'tscglue[cu124]'"
        ) from exc


def load_dataset(dataset_name):
    """Load and normalize a dataset."""
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    return X_train, y_train, X_test, y_test


def get_folds(X, y, n_splits=10, random_state=None, stratify=True):
    folds = []
    if stratify:
        try:
            # Try stratified split (classification)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_idx, val_idx in skf.split(X, y):
                folds.append((train_idx.tolist(), val_idx.tolist()))
            return folds
        except ValueError:
            # Fall back to regular KFold
            print(f"StratifiedKFold failed, falling back to regular KFold with n_splits={n_splits}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(X):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds
