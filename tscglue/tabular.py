"""Tabular classifiers used as the head of the time-series pipelines."""

# TODO import tabular models into this

from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
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


class AutoSelectKBestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        classifier=None,
        k=None,
        k_min=6000,
        k_max=35000,
        midpoint=300,
        steepness=0.010,
        score_func=f_classif,
        variance_filter=True,
        n_jobs=1,
    ):
        self.classifier = classifier
        self.k = k
        self.k_min = k_min
        self.k_max = k_max
        self.midpoint = midpoint
        self.steepness = steepness
        self.score_func = score_func
        self.variance_filter = variance_filter
        self.n_jobs = n_jobs

    def _optimal_k(self, n_train: int) -> int:
        return int(
            self.k_min
            + (self.k_max - self.k_min)
            / (1.0 + np.exp(-self.steepness * (n_train - self.midpoint)))
        )

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self._predict(X)

    def predict_proba(self, X):
        return self._predict_proba(X)

    # internal helpers
    def _fit(self, X, y):
        k = self.k if self.k is not None else self._optimal_k(X.shape[0])
        k = min(k, X.shape[1])

        if self.classifier is None:
            clf = RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10), n_jobs=self.n_jobs)
        else:
            # An explicit classifier carries its own thread budget; n_jobs is not forced on it.
            clf = clone(self.classifier)

        var = VarianceThreshold() if self.variance_filter else None
        select = SelectKBest(score_func=self.score_func, k=k)

        # Suppress f_classif "Features are constant" warnings: after CV fold splitting
        # and scaling, some features have near-zero variance that passes VarianceThreshold
        # but f_classif still treats as constant. These features are harmless (never selected).
        import warnings

        # Steps are fitted one at a time rather than through Pipeline.fit purely so each
        # can be timed; the order and the inputs are what Pipeline.fit would do. The
        # already-fitted steps are then assembled into the Pipeline predict/predict_proba
        # use, so nothing downstream changes.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = perf_counter()
            Xt = var.fit_transform(X) if var is not None else X
            t_var = perf_counter() - t0

            t0 = perf_counter()
            Xt = select.fit_transform(Xt, y)
            t_select = perf_counter() - t0

            t0 = perf_counter()
            clf.fit(Xt, y)
            t_clf = perf_counter() - t0

        steps = [("var", var)] if var is not None else []
        self.classifier_ = Pipeline([*steps, ("select", select), ("clf", clf)])
        self.fit_times_ = {
            "variance_s": t_var,
            "select_s": t_select,
            "clf_s": t_clf,
            "total_s": t_var + t_select + t_clf,
        }

        # sklearn convention: expose classes_
        inner = self.classifier_.named_steps["clf"]
        if hasattr(inner, "classes_"):
            self.classes_ = inner.classes_

        return self

    def _predict(self, X):
        return self.classifier_.predict(X)

    def _predict_proba(self, X):
        inner = self.classifier_.named_steps["clf"]
        if not hasattr(inner, "predict_proba"):
            raise AttributeError("Underlying classifier does not support predict_proba().")
        return self.classifier_.predict_proba(X)
