"""Standalone feature-pipeline baseline built from TSCGlue's representations.

A plain aeon classifier: extract features with one or more of the transformers
TSCGlue already uses, concatenate the blocks, and fit a single sklearn head. No
fold training, no stacking.

- ``MRHydraET``  multirocket + hydra -> ExtraTrees  rocket features, forest probabilities

It is the probabilistic fallback the stacked classifiers in ``models.py`` fit
alongside the main pipeline. ``TSCFeatureBaseline`` stays generic over feature
sets and heads (``et``, ``logistic``, ``ridge``) so further baselines can be
added as thin subclasses.
"""

import numpy as np
from aeon.classification.base import BaseClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from tscglue.models import get_feature_transformer
from tscglue.tabular import (
    AutoSelectKBestClassifier,
    RareClassSafeLogisticCV,
    SparseScaler,
)


class TSCFeatureBaseline(BaseClassifier):
    """Concatenated-feature pipeline classifier over TSCGlue's transformers.

    Parameters
    ----------
    features : tuple of str
        Feature transformer names understood by ``get_feature_transformer``
        (e.g. "quant", "multirocket", "hydra", "rdst", "rstsf-random", "weasel").
    head : str
        "et" (ExtraTrees), "logistic" (RareClassSafeLogisticCV) or
        "ridge" (AutoSelectKBestClassifier with RidgeClassifierCVDecisionProba).
    """

    _tags = {"capability:multivariate": True}

    PROBA_EPS = 1e-4

    def __init__(
        self,
        features=("quant",),
        head="et",
        n_estimators=500,
        random_state=None,
        n_jobs=1,
        verbose=0,
    ):
        super().__init__()
        self.features = features
        self.head = head
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ----------------- components -----------------

    def _make_head(self, seed: int):
        if self.head == "et":
            return ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                criterion="entropy",
                max_features="sqrt",
                random_state=seed,
                n_jobs=self.n_jobs,
            )
        if self.head == "logistic":
            return RareClassSafeLogisticCV()
        if self.head == "ridge":
            return AutoSelectKBestClassifier()
        raise ValueError(f"Unknown head {self.head!r}; expected 'et', 'logistic' or 'ridge'")

    def _block_scaler(self, feature_name: str):
        # Tree heads take raw features; linear heads get the same per-block
        # scaling the stacked pipeline uses (SparseScaler for hydra counts).
        if self.head == "et":
            return None
        if feature_name == "hydra":
            return SparseScaler()
        return StandardScaler()

    def _feature_input(self, feature_name: str, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float64) if feature_name == "rdst" else X

    # ----------------- aeon API -----------------

    def _fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.transformers_ = {}
        self.scalers_ = {}
        blocks = []
        for name in self.features:
            seed = int(rng.integers(0, 2**31 - 1))
            transformer = get_feature_transformer(name, seed=seed, n_jobs=self.n_jobs)
            Xt = np.asarray(
                transformer.fit_transform(self._feature_input(name, X)), dtype=np.float32
            )
            self.transformers_[name] = transformer
            scaler = self._block_scaler(name)
            if scaler is not None:
                Xt = np.asarray(scaler.fit_transform(Xt), dtype=np.float32)
                self.scalers_[name] = scaler
            if self.verbose:
                print(f"[{type(self).__name__}] {name}: {Xt.shape[1]} features")
            blocks.append(Xt)
        Xt_all = np.hstack(blocks)
        self.head_ = self._make_head(int(rng.integers(0, 2**31 - 1)))
        self.head_.fit(Xt_all, y)
        return self

    def _transform_blocks(self, X) -> np.ndarray:
        blocks = []
        for name in self.features:
            Xt = np.asarray(
                self.transformers_[name].transform(self._feature_input(name, X)),
                dtype=np.float32,
            )
            if name in self.scalers_:
                Xt = np.asarray(self.scalers_[name].transform(Xt), dtype=np.float32)
            blocks.append(Xt)
        return np.hstack(blocks)

    def _predict_proba(self, X):
        proba = self.head_.predict_proba(self._transform_blocks(X))
        head_classes = np.asarray(self.head_.classes_)
        if not np.array_equal(head_classes, self.classes_):
            order = [int(np.where(head_classes == c)[0][0]) for c in self.classes_]
            proba = proba[:, order]
        proba = np.clip(proba, self.PROBA_EPS, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def _predict(self, X):
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]


class MRHydraET(TSCFeatureBaseline):
    """multirocket + hydra -> ExtraTrees. Rocket features with forest probabilities instead of ridge."""

    def __init__(self, n_estimators=500, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("multirocket", "hydra"), head="et", n_estimators=n_estimators,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )
