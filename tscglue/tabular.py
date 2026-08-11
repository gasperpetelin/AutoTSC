"""Tabular classifiers used as the head of the time-series pipelines."""

# TODO import tabular models into this

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.utils.extmath import softmax
from threadpoolctl import threadpool_limits


class RidgeClassifierCVDecisionProba(RidgeClassifierCV):
    """RidgeClassifierCV with softmax probabilities and ``n_jobs`` as a BLAS thread budget."""

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
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            scores = self.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return softmax(scores)


class RidgeClassifierCVIndicator(RidgeClassifierCV):
    """RidgeClassifierCV whose probabilities are a one-hot indicator of the predicted class."""

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

    def predict_proba(self, X):
        dists = np.zeros((X.shape[0], len(self.classes_)))
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            preds = self.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1
        return dists

    def fit(self, X, y):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return super().fit(X, y)
