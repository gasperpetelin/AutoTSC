import multiprocessing
import os
import threading
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.dictionary_based._weasel_v2 import WEASELTransformerV2
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, chi2
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from tscglue import utils
from tscglue.tabular import (
    AutoSelectKBestClassifier,
    NoScaler,
    RidgeClassifierCVDecisionProba,
    RidgeClassifierCVIndicator,
    SparseScaler,
)
from tscglue.utils import (
    _noop,
    _run_in_subprocess,
    log,
    read_array,
    read_model,
    save_array,
    save_model,
)


class RareClassSafeLogisticCV(BaseEstimator, ClassifierMixin):
    """LogisticRegressionCV that won't crash on rare classes.

    LogisticRegressionCV runs an internal stratified CV to choose C. When a class
    has a single member in the data it is handed, one fold's training split omits
    that class and the multinomial coefficient paths become ragged, so the
    internal ``np.reshape`` raises "inhomogeneous shape". This wrapper drops the
    internal CV (fixed-C ``LogisticRegression``) only in that singleton case and
    is otherwise bit-identical to the original ``LogisticRegressionCV``.
    """

    def __init__(self, Cs=10, fixed_C=1.0, solver="lbfgs", max_iter=1000, class_weight=None):
        self.Cs = Cs
        self.fixed_C = fixed_C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

        min_count = int(np.unique(y, return_counts=True)[1].min())
        if min_count < 2:
            # A singleton class would make the internal CV crash; skip it.
            self.estimator_ = LogisticRegression(
                C=self.fixed_C,
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
            )
        else:
            # Identical to the original stacker: default cv=5, multinomial.
            self.estimator_ = LogisticRegressionCV(
                Cs=self.Cs,
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
            )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class ThreadBudgetMLPClassifier(MLPClassifier):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        *,
        max_iter=200,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.n_jobs = n_jobs

    def fit(self, X, y):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return super().fit(X, y)

    def predict_proba(self, X):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return super().predict_proba(X)


# name -> (feature scalers, head factory). Scalers are stored as classes so each call
# builds its own unfitted instance; every head takes **_ and names only what it uses.
_MODELS_V6 = {
    "multirockethydra-ridgecv": (
        {"hydra": SparseScaler, "multirocket": StandardScaler},
        lambda **_: RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 10)),
    ),
    "multirockethydra-p-ridgecv": (
        {"hydra": SparseScaler, "multirocket": StandardScaler},
        lambda **_: RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)),
    ),
    "quant-etc": (
        {"quant": NoScaler},
        lambda seed=None, n_jobs=1, **_: ExtraTreesClassifier(
            n_estimators=200,
            max_features=0.1,
            criterion="entropy",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    ),
    "rdst-ridgecv": (
        {"rdst": StandardScaler},
        lambda **_: RidgeClassifierCVIndicator(alphas=np.logspace(-4, 4, 20)),
    ),
    "rdst-p-ridgecv": (
        {"rdst": StandardScaler},
        lambda **_: RidgeClassifierCVDecisionProba(alphas=np.logspace(-4, 4, 20)),
    ),
    "probability-ridgecv": (
        {"probabilities": StandardScaler},
        lambda **_: RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 20)),
    ),
    "probability-ridgecv-balanced": (
        {"probabilities": StandardScaler},
        lambda **_: RidgeClassifierCVIndicator(
            alphas=np.logspace(-3, 3, 20), class_weight="balanced"
        ),
    ),
    "probability-logisticcv": (
        {"probabilities": StandardScaler},
        lambda **_: RareClassSafeLogisticCV(Cs=np.logspace(-3, 3, 20)),
    ),
    "probability-logisticcv-balanced": (
        {"probabilities": StandardScaler},
        lambda **_: RareClassSafeLogisticCV(Cs=np.logspace(-3, 3, 20), class_weight="balanced"),
    ),
    "probability-et": (
        {"probabilities": NoScaler},
        lambda seed=None, n_jobs=1, **_: ExtraTreesClassifier(
            n_estimators=1000,
            random_state=seed,
            n_jobs=n_jobs,
        ),
    ),
    "probability-et-balanced": (
        {"probabilities": NoScaler},
        lambda seed=None, n_jobs=1, **_: ExtraTreesClassifier(
            n_estimators=1000,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    ),
    "probability-rf": (
        {"probabilities": NoScaler},
        lambda seed=None, n_jobs=1, **_: RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=n_jobs
        ),
    ),
    "probability-rf-balanced": (
        {"probabilities": NoScaler},
        lambda seed=None, n_jobs=1, **_: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=n_jobs
        ),
    ),
    "probability-nn": (
        {"probabilities": StandardScaler},
        lambda seed=None, **_: MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=500, random_state=seed
        ),
    ),
    "multirockethydra-bestk-p-ridgecv": (
        {"hydra": SparseScaler, "multirocket": StandardScaler},
        lambda prepruned_features=False, **_: AutoSelectKBestClassifier(
            variance_filter=not prepruned_features
        ),
    ),
    "weasel-bestk-p-ridgecv": (
        {"weasel": NoScaler},
        lambda prepruned_features=False, **_: AutoSelectKBestClassifier(
            k=30000, score_func=chi2, variance_filter=not prepruned_features
        ),
    ),
    "fm-p-ridgecv": (
        {"mantis": StandardScaler, "chronos2": StandardScaler},
        lambda **_: RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)),
    ),
    "rstsf-random-etc": (
        {"rstsf-random": NoScaler},
        lambda seed=None, n_jobs=1, **_: ExtraTreesClassifier(
            n_estimators=200,
            criterion="entropy",
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        ),
    ),
    "multirockethydra-etc": (
        {"hydra": NoScaler, "multirocket": NoScaler},
        lambda seed=None, n_jobs=1, **_: ExtraTreesClassifier(
            n_estimators=200,
            criterion="entropy",
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    ),
    "rdst-etc": (
        {"rdst": NoScaler},
        lambda seed=None, n_jobs=1, **_: ExtraTreesClassifier(
            n_estimators=200,
            criterion="entropy",
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    ),
    "fm-etc": (
        {"mantis": NoScaler, "chronos2": NoScaler},
        lambda seed=None, n_jobs=1, **_: ExtraTreesClassifier(
            n_estimators=200,
            criterion="entropy",
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    ),
    "weasel-etc": (
        {"weasel": NoScaler},
        lambda seed=None, n_jobs=1, **_: ExtraTreesClassifier(
            n_estimators=200,
            criterion="entropy",
            max_features="sqrt",
            random_state=seed,
            n_jobs=n_jobs,
        ),
    ),
    "quant-p-ridgecv": (
        {"quant": StandardScaler},
        lambda **_: RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)),
    ),
    "rstsf-random-p-ridgecv": (
        {"rstsf-random": StandardScaler},
        lambda **_: RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)),
    ),
}


def get_model_v6(name, seed=None, n_jobs=1, model_dir=None, prepruned_features=False, **kwargs):
    """Returns (DictMultiScaler, classifier) for feature and stacking models."""
    if name not in _MODELS_V6:
        raise ValueError(f"Unknown model name: {name}")
    scalers, head = _MODELS_V6[name]
    scaler = DictMultiScaler(scalers={feat: cls() for feat, cls in scalers.items()})
    clf = head(
        seed=seed,
        n_jobs=n_jobs,
        model_dir=model_dir,
        prepruned_features=prepruned_features,
        **kwargs,
    )
    return scaler, clf


class RDSTFloat64(RandomDilatedShapeletTransform):
    """RDST wrapper that casts input to float64 (numba requires it)."""

    def _fit(self, X, y=None):
        return super()._fit(np.asarray(X, dtype=np.float64), y)


class WEASELTransformerV2Unsupervised(WEASELTransformerV2):
    """WEASELTransformerV2 usable in this project's unsupervised feature pipeline.

    Upstream always does ``y.copy()`` in fit_transform, but feature_selection="none"
    means y is otherwise unused, so a placeholder y stands in for the real labels
    that the shared feature-fitting call sites don't have access to.
    """

    def __init__(
        self,
        min_window=4,
        norm_options=(False,),
        word_lengths=(7, 8),
        use_first_differences=(True, False),
        max_feature_count=30_000,
        random_state=None,
        n_jobs=4,
    ):
        super().__init__(
            min_window=min_window,
            norm_options=norm_options,
            word_lengths=word_lengths,
            use_first_differences=use_first_differences,
            feature_selection="none",
            max_feature_count=max_feature_count,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit_transform(self, X, y=None):
        if y is None:
            y = np.zeros(X.shape[0], dtype=int)

        self.transformers_ = []
        Xt = []
        for channel in range(X.shape[1]):
            transformer = WEASELTransformerV2(
                min_window=self.min_window,
                norm_options=self.norm_options,
                word_lengths=self.word_lengths,
                use_first_differences=self.use_first_differences,
                feature_selection="none",
                max_feature_count=self.max_feature_count,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            Xt.append(transformer.fit_transform(X[:, channel : channel + 1], y))
            self.transformers_.append(transformer)
        return np.hstack(Xt)

    def transform(self, X, y=None):
        if X.shape[1] != len(self.transformers_):
            raise ValueError("X must have the same number of channels as the training data")
        return np.hstack(
            [
                transformer.transform(X[:, channel : channel + 1])
                for channel, transformer in enumerate(self.transformers_)
            ]
        )

    def _transform(self, X, y=None):
        return super()._transform(np.asarray(X, dtype=np.float64), y)


def get_feature_transformer(feature_type: str, seed: int, n_jobs: int = 1, device: str = "cpu"):
    match feature_type:
        case "multirocket":
            return MultiRocket(n_jobs=n_jobs, random_state=seed)
        case "rdst":
            return RDSTFloat64(n_jobs=n_jobs, random_state=seed)
        case "quant":
            return QUANTTransformer()
        case "hydra":
            if device != "cpu":
                # Local import, as for the foundation models below: torch is only a
                # hard dependency on the device path. HydraTransformerDevice takes no
                # n_jobs -- the device path has no CPU work worth budgeting.
                from tscglue.features_gpu import HydraTransformerDevice

                return HydraTransformerDevice(random_state=seed, device=device)
            return HydraTransformer(n_jobs=n_jobs, random_state=seed)
        case "mantis":
            from tscglue.models_tsfm import MantisEmbedding

            return MantisEmbedding(device=device)
        case "chronos2":
            from tscglue.models_tsfm import Chronos2Embedding

            return Chronos2Embedding(device=device)
        case "tsfresh":
            from aeon.transformations.collection.feature_based import TSFresh

            return TSFresh(default_fc_parameters="efficient", n_jobs=n_jobs)
        case "rstsf-random":
            from tscglue.interval_models import RSTSFRandomTransformer

            return RSTSFRandomTransformer(n_jobs=n_jobs, random_state=seed)
        case "drcif":
            from tscglue.drcif_features import DrCIFExtractor

            return DrCIFExtractor(random_state=seed, n_jobs=n_jobs)
        case "weasel":
            return WEASELTransformerV2Unsupervised(random_state=seed, n_jobs=n_jobs)
        case _:
            raise ValueError(f"Unknown feature transformer type: {feature_type}")


def _transform_in_subprocess(
    feature_id, X_path, model_dir, output_dir, dtype=np.float64, verbose=0
):
    X = np.load(X_path, allow_pickle=True)
    transformer = read_model(f"transformer_{feature_id}", model_dir)
    t0 = perf_counter()
    Xt = transformer.transform(X)
    if verbose >= 3:
        print(f"[subprocess] transform {feature_id}: {perf_counter() - t0:.4f}s")
    if dtype is not None:
        Xt = np.asarray(Xt, dtype=dtype)
    save_array(Xt, f"Xt_{feature_id}", output_dir)


def _wrap_variance_filter(transformer):
    return Pipeline([("features", transformer), ("var", VarianceThreshold())])


def _fit_transform_in_subprocess(
    feature_name,
    feature_seed,
    n_jobs,
    X_path,
    model_dir,
    output_dir,
    feature_id,
    dtype=np.float64,
    verbose=0,
    device="cpu",
    variance_filter=False,
):
    X = np.load(X_path, allow_pickle=True)
    transformer = get_feature_transformer(
        feature_name, seed=feature_seed, n_jobs=n_jobs, device=device
    )
    if variance_filter:
        transformer = _wrap_variance_filter(transformer)
    t0 = perf_counter()
    Xt = transformer.fit_transform(X)
    if verbose >= 3:
        print(f"[subprocess] fit_transform {feature_id}: {perf_counter() - t0:.4f}s")
    save_model(transformer, f"transformer_{feature_id}", model_dir)
    if dtype is not None:
        Xt = np.asarray(Xt, dtype=dtype)
    save_array(Xt, f"Xt_{feature_id}", output_dir)


def _transform_inline(feature_id, X_path, model_dir, output_dir, dtype=np.float64):
    X = np.load(X_path, allow_pickle=True)
    transformer = read_model(f"transformer_{feature_id}", model_dir)
    Xt = transformer.transform(X)
    if dtype is not None:
        Xt = np.asarray(Xt, dtype=dtype)
    save_array(Xt, f"Xt_{feature_id}", output_dir)


def _fit_transform_inline(
    feature_name,
    feature_seed,
    n_jobs,
    X_path,
    model_dir,
    output_dir,
    feature_id,
    dtype=np.float64,
    device="cpu",
    variance_filter=False,
):
    X = np.load(X_path, allow_pickle=True)
    transformer = get_feature_transformer(
        feature_name, seed=feature_seed, n_jobs=n_jobs, device=device
    )
    if variance_filter:
        transformer = _wrap_variance_filter(transformer)
    Xt = transformer.fit_transform(X)
    save_model(transformer, f"transformer_{feature_id}", model_dir)
    if dtype is not None:
        Xt = np.asarray(Xt, dtype=dtype)
    save_array(Xt, f"Xt_{feature_id}", output_dir)


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    feature_seed: int | None = None
    use_subprocess: bool = True
    support_gpu: bool = False

    def get_feature_id(self):
        return (
            f"{self.feature_name}_s_{self.feature_seed}"
            if self.feature_seed is not None
            else self.feature_name
        )


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_seed: int
    level: int
    features: tuple[FeatureSpec, ...]
    fold_seeds: tuple[int, ...]

    def get_model_id(self):
        return f"{self.model_name}_s_{self.model_seed}"

    @property
    def n_repetitions(self) -> int:
        return len(self.fold_seeds)


def _load_feature_dict_v10(directory, feature_specs):
    """Load feature arrays using read_array with (feat_type, repetition) specs."""
    feature_dict = {}
    for feat_spec in feature_specs:
        feat_id = feat_spec.get_feature_id()
        feature_dict[feat_spec.feature_name] = read_array(f"Xt_{feat_id}", directory)
    return feature_dict


def _predict_one_model_v10(model_id, model_name, directory, feature_specs, model_dir, fold):
    """Prediction function - loads model from disk, loads data via read_array."""
    feature_dict = _load_feature_dict_v10(directory, feature_specs)

    scaler, clf = read_model(model_id, model_dir, None, fold)
    start_predict = perf_counter()

    proba = clf.predict_proba(scaler.transform(feature_dict))

    predict_dur = perf_counter() - start_predict
    return (proba, clf.classes_, predict_dur, model_id)


def _train_one_model_v10(
    fold_number,
    model_id,
    model_name,
    train_idx,
    val_idx,
    model_seed,
    directory,
    feature_specs,
    model_dir,
    model_kwargs=None,
):
    """Training function - loads data via read_array, saves model to disk."""
    y = read_array("y", directory)
    feature_dict = _load_feature_dict_v10(directory, feature_specs)

    scaler, clf = get_model_v6(model_name, seed=model_seed, model_dir=model_dir, **(model_kwargs or {}))
    start_train = perf_counter()

    clf.fit(scaler.fit_transform(feature_dict, idx=train_idx), y[train_idx])
    proba = clf.predict_proba(scaler.transform(feature_dict, idx=val_idx))
    _, model_size = save_model((scaler, clf), model_id, model_dir, None, fold_number)

    train_dur = perf_counter() - start_train
    return (train_idx, val_idx, proba, clf.classes_, model_size, train_dur, model_id, fold_number)


class LokyStackerV10Base(BaseClassifier):
    _tags = {"capability:multivariate": True}

    DEFAULT_MODEL_NAMES = [
        "multirockethydra-bestk-p-ridgecv",
        "quant-etc",
        "rdst-p-ridgecv",
    ]
    STACKING_MODEL = "probability-ridgecv"
    # Class-level default so every estimator has the attribute; ``TSCGlueEnhancedV4``
    # promotes it to a constructor parameter. Read via the instance so a subclass that
    # exposes it as a real parameter shadows this.
    prune_constant: bool = False

    def _get_feature_names(self, model_name: str) -> tuple[str, ...]:
        """Return required feature type names for a model."""
        if model_name in (
            "multirockethydra-bestk-p-ridgecv",
            "multirockethydra-p-ridgecv",
            "multirockethydra-ridgecv",
            "multirockethydra-etc",
        ):
            return ("multirocket", "hydra")
        elif model_name in ("quant-etc", "quant-p-ridgecv"):
            return ("quant",)
        elif model_name in ("rdst-p-ridgecv", "rdst-ridgecv", "rdst-etc"):
            return ("rdst",)
        elif model_name in ("rstsf-random-etc", "rstsf-random-p-ridgecv"):
            return ("rstsf-random",)
        elif model_name in ("fm-p-ridgecv", "fm-etc"):
            return ("mantis", "chronos2")
        elif model_name in ("weasel-bestk-p-ridgecv", "weasel-etc"):
            return ("weasel",)
        else:
            raise ValueError(f"Unknown model {model_name}")

    def _make_feature_spec(self, feature_name: str, group_rng: np.random.Generator) -> FeatureSpec:
        """Create a single FeatureSpec. Seedless for deterministic transforms like quant."""
        use_subprocess = feature_name not in ("multirocket", "rdst", "rstsf-random")
        support_gpu = feature_name in ("hydra", "mantis", "chronos2")
        if feature_name in ("quant", "mantis", "chronos2"):
            return FeatureSpec(
                feature_name=feature_name,
                use_subprocess=use_subprocess,
                support_gpu=support_gpu,
            )
        return FeatureSpec(
            feature_name=feature_name,
            feature_seed=int(group_rng.integers(0, 2**31 - 1)),
            use_subprocess=use_subprocess,
            support_gpu=support_gpu,
        )

    def build_model_specs(self, model_names: list[str]) -> list[ModelSpec]:
        """Build ModelSpec list from a flat list of model names.

        Models are accumulated into groups that share feature seeds.
        A duplicate model name starts a new group.
        """
        # Split flat list into groups: a new group starts when a name is repeated
        groups: list[list[str]] = []
        seen: set[str] = set()
        for name in model_names:
            if name in seen:
                groups.append([])
                seen = set()
            if not groups:
                groups.append([])
            groups[-1].append(name)
            seen.add(name)

        all_models: list[ModelSpec] = []
        for group in groups:
            group_rng = np.random.default_rng(self._get_feature_seed())

            # Build FeatureSpecs per group, deduped by feature name within group
            group_features: dict[str, FeatureSpec] = {}
            for model_name in group:
                for ft_name in self._get_feature_names(model_name):
                    if ft_name not in group_features:
                        group_features[ft_name] = self._make_feature_spec(ft_name, group_rng)

            for model_name in group:
                features = tuple(
                    group_features[ft_name] for ft_name in self._get_feature_names(model_name)
                )
                model_seed = self._get_feature_seed()
                fold_seed_rng = np.random.default_rng(model_seed)
                fold_seeds = tuple(
                    int(fold_seed_rng.integers(0, 2**31 - 1)) for _ in range(self.n_repetitions)
                )
                spec = ModelSpec(
                    model_name=model_name,
                    model_seed=model_seed,
                    level=0,
                    features=features,
                    fold_seeds=fold_seeds,
                )
                all_models.append(spec)

        return all_models

    def __init__(
        self,
        random_state=None,
        k_folds=10,
        n_jobs=1,
        keep_features=False,
        verbose=0,
        model_names=None,
        n_repetitions=1,
        compute_dtype=None,
        stacking_models=None,
        selection=None,
        n_gpus=0,
        runs_dir=None,
        eval_metric="accuracy",
    ):
        super().__init__()
        self.k_folds = int(k_folds)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.n_gpus = int(n_gpus)
        self.keep_features = bool(keep_features)
        self.verbose = int(verbose)
        self.n_repetitions = int(n_repetitions)
        self.compute_dtype = np.dtype(compute_dtype) if compute_dtype is not None else None
        self.stacking_models = (
            stacking_models if stacking_models is not None else [self.STACKING_MODEL]
        )
        self.selection = selection
        self.runs_dir = runs_dir
        self.eval_metric = eval_metric

        self.cv_splits = None
        self.feature_seed = np.random.default_rng(random_state)

        self._run_id = uuid.uuid4().hex[:16]
        self._base_dir = Path(
            ".", runs_dir if runs_dir is not None else "tscglue_runs", self._run_id
        )
        self._model_dir = self._base_dir / "models"
        self._tmpdir: Path | None = self._base_dir / "features_training"

        self.model_names = model_names

        # Build model specs from flat list; derive unique features
        self.model_specs = self.build_model_specs(
            self.model_names if self.model_names is not None else self.DEFAULT_MODEL_NAMES
        )
        self.best_model = (
            self.stacking_models[0] if self.stacking_models else self.model_specs[0].get_model_id()
        )
        all_features: dict[str, FeatureSpec] = {}
        for spec in self.model_specs:
            for ft in spec.features:
                fid = ft.get_feature_id()
                if fid not in all_features:
                    all_features[fid] = ft
        self.features_list = list(all_features.values())

        self._oof_scores: list[dict] = []
        self._transform_times: list[dict] = []
        self._probability_columns: list[tuple[int, str, Any]] | None = None

        self._fallback_path: Path = self._model_dir / "fallback.pkl"

    # ----------------- utils -----------------

    @property
    def _device(self) -> str:
        return "cuda" if self.n_gpus != 0 else "cpu"

    def _feature_device(self, ft: FeatureSpec) -> str:
        """Device a feature transformer is built for."""
        return self._device if ft.support_gpu else "cpu"

    def _split_feature_lanes(self) -> tuple[list[FeatureSpec], list[FeatureSpec]]:
        """``features_list`` split into (device lane, cpu lane) on ``support_gpu``.

        The device lane runs sequentially in one background thread while the cpu lane
        runs on the main thread, so both processors are busy at once. Lane order is
        ``features_list`` order, which puts the cheap hydra transform ahead of the
        foundation models on every preset -- it releases its device memory before they
        allocate. Without a GPU the device lane is empty and everything runs on the
        main thread.
        """
        use_gpu = self._device != "cpu"

        def on_device(ft) -> bool:
            return use_gpu and ft.support_gpu

        return [ft for ft in self.features_list if on_device(ft)], [
            ft for ft in self.features_list if not on_device(ft)
        ]

    def _get_feature_seed(self) -> int:
        return int(self.feature_seed.integers(0, 2**31 - 1, dtype=np.int32))

    def _require_tmpdir(self) -> Path:
        if self._tmpdir is None:
            raise RuntimeError("Temporary directory not available.")
        return self._tmpdir

    def _label_to_python(self, value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    def _missing_classes(self, classes) -> list:
        """Labels seen in ``y`` that a fitted model cannot predict.

        Every fold model has to cover all of ``self.classes_``: the OOF matrix and
        the per-model prediction blocks are assembled with one column per class, so
        a model with a narrower ``classes_`` leaves NaN columns behind and cannot be
        stacked or served. When any model comes back short the stack is abandoned in
        favour of the fallback.
        """
        seen = {str(c) for c in np.asarray(classes)}
        return [self._label_to_python(c) for c in self.classes_ if str(c) not in seen]

    def _probability_key(self, level: int, model_name: str, cls: Any) -> tuple[int, str, Any]:
        return int(level), model_name, self._label_to_python(cls)

    def _probability_sort_key(self, key: tuple[int, str, Any]) -> tuple[int, str, Any]:
        level, model_name, cls = key
        return level, model_name, self._label_to_python(cls)

    def _aggregate_prediction_matrix(
        self,
        predictions: list[dict],
        n_samples: int,
        probability_columns: Iterable[tuple[int, str, Any]],
    ) -> np.ndarray:
        columns = list(probability_columns)
        if not columns:
            return np.empty((n_samples, 0), dtype=np.float64)

        col_to_idx = {col: i for i, col in enumerate(columns)}
        prob_sum = np.zeros((n_samples, len(columns)), dtype=np.float64)
        prob_count = np.zeros((n_samples, len(columns)), dtype=np.int32)

        for pred in predictions:
            key = self._probability_key(pred["level"], pred["model"], pred["class"])
            col_idx = col_to_idx.get(key)
            if col_idx is None:
                continue
            row_idx = int(pred["index"])
            prob_sum[row_idx, col_idx] += float(pred["probability"])
            prob_count[row_idx, col_idx] += 1

        prob_array = np.full((n_samples, len(columns)), np.nan, dtype=np.float64)
        np.divide(prob_sum, prob_count, out=prob_array, where=prob_count > 0)
        return prob_array

    def cleanup(self):
        if self._base_dir.exists():
            shutil.rmtree(self._base_dir)

    def _feature_input(self, ft: str, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float64) if ft == "rdst" else X

    # ----------------- aeon API -----------------

    def _predict_proba(self, X):
        if self._fallback_path.exists():
            fallback = read_model("fallback", str(self._model_dir))
            return fallback.predict_proba(X)
        return self.predict_proba_per_model(X)[self.best_model]

    def _predict(self, X):
        if self._fallback_path.exists():
            fallback = read_model("fallback", str(self._model_dir))
            return fallback.predict(X)
        probas = self._predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    # ----------------- prediction row helpers -----------------

    def add_probabilities(self, probas, classes, model_name, level, indices=None):
        preds = []
        row_indices = np.arange(len(probas)) if indices is None else np.asarray(indices)
        for idx, row in zip(row_indices, probas):
            for scls, prob in zip(classes, row):
                preds.append(
                    {
                        "index": int(idx),
                        "model": model_name,
                        "level": level,
                        "class": self._label_to_python(scls),
                        "probability": float(self._label_to_python(prob)),
                    }
                )
        return preds

    # ----------------- OOF persistence -----------------

    def _save_model_predictions(self, predictions, model_name, n_samples, level):
        model_preds = [p for p in predictions if p["model"] == model_name]
        if not model_preds:
            return predictions
        classes = sorted({self._label_to_python(p["class"]) for p in model_preds})
        class_to_idx = {c: i for i, c in enumerate(classes)}
        prob_sum = np.zeros((n_samples, len(classes)), dtype=np.float64)
        prob_count = np.zeros((n_samples, len(classes)), dtype=np.int32)
        for p in model_preds:
            idx = p["index"]
            cidx = class_to_idx[p["class"]]
            prob_sum[idx, cidx] += p["probability"]
            prob_count[idx, cidx] += 1
        prob_array = np.where(prob_count > 0, prob_sum / prob_count, np.nan)
        d = str(self._require_tmpdir())
        save_array(prob_array, f"pred_{model_name}", d)
        save_array(np.array([level] + classes, dtype=object), f"pred_{model_name}_meta", d)
        return [p for p in predictions if p["model"] != model_name]

    def _load_model_predictions(self, model_name):
        d = str(self._require_tmpdir())
        prob_array = read_array(f"pred_{model_name}", d)
        meta = read_array(f"pred_{model_name}_meta", d, allow_pickle=True, mmap_mode=None)
        level = int(meta[0])
        classes = list(meta[1:])
        return prob_array, level, classes

    def _compute_oof_score(self, y, model_name) -> float:
        prob_array, _level, classes = self._load_model_predictions(model_name)
        valid = ~np.isnan(prob_array).any(axis=1)
        if not np.any(valid):
            return 0.0
        y_true = y[np.where(valid)[0]]
        proba = prob_array[valid]

        if self.eval_metric == "accuracy":
            pred_idx = np.argmax(proba, axis=1)
            preds = np.asarray(classes)[pred_idx]
            return float(accuracy_score(np.asarray(y_true, dtype=str), np.asarray(preds, dtype=str)))
        elif self.eval_metric == "f1":
            from sklearn.metrics import f1_score
            pred_idx = np.argmax(proba, axis=1)
            preds = np.asarray(classes)[pred_idx]
            # Macro so every class weighs equally, matching the ovr/macro
            # convention already used for roc_auc and average_precision here.
            return float(
                f1_score(
                    np.asarray(y_true, dtype=str),
                    np.asarray(preds, dtype=str),
                    average="macro",
                )
            )
        elif self.eval_metric == "log_loss":
            from sklearn.metrics import log_loss
            return float(log_loss(y_true, proba, labels=classes))
        elif self.eval_metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            if len(classes) == 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes))
        elif self.eval_metric == "average_precision":
            from sklearn.metrics import average_precision_score
            from sklearn.preprocessing import label_binarize
            if len(classes) == 2:
                return float(average_precision_score(y_true, proba[:, 1]))
            y_bin = label_binarize(y_true, classes=classes)
            return float(average_precision_score(y_bin, proba, average="macro"))
        else:
            raise ValueError(f"Unknown eval_metric: {self.eval_metric!r}")

    def _build_probability_array(self, n_samples: int):
        d = self._require_tmpdir()
        prob_files = sorted(p for p in d.glob("pred_*.npy") if not p.name.endswith("_meta.npy"))
        cols, names = [], []
        for path in prob_files:
            model_name = path.stem[5:]  # strip pred_
            prob_array, level, classes = self._load_model_predictions(model_name)
            if level != 0:
                continue
            for i, cls in enumerate(classes):
                names.append(self._probability_key(level, model_name, cls))
                cols.append(prob_array[:, i])
        if not cols:
            return None
        order = sorted(range(len(names)), key=lambda i: self._probability_sort_key(names[i]))
        self._probability_columns = [names[i] for i in order]
        return np.column_stack([cols[i] for i in order])

    # ----------------- features: train transformers + compute arrays -----------------

    def fit_transform_features(self, X: np.ndarray, fit_start_time=None) -> None:
        """Fit transformers and compute features.

        When a GPU is available, ``support_gpu`` features run in a background thread
        while the rest run on the main thread, so both processors are used
        simultaneously. When no GPU is available all features run sequentially on the
        main thread.
        """
        os.makedirs(self._model_dir, exist_ok=True)
        directory = str(self._tmpdir)
        X_path = str(self._tmpdir / "X.npy")

        gpu_features, cpu_features = self._split_feature_lanes()
        gpu_error: list[BaseException] = []

        def _log_feature(ft, t0):
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            size_mb = Xt.nbytes / (1024 * 1024)
            elapsed = perf_counter() - t0
            device = self._feature_device(ft)
            log(
                f"Fit+transformed {ft.get_feature_id()} [{device}] features {Xt.shape} ({size_mb:.2f} MB) dtype={Xt.dtype} in {elapsed:.4f}s",
                level=1,
                start_time=fit_start_time,
                verbose=self.verbose,
            )
            self._transform_times.append(
                {
                    "model": ft.get_feature_id(),
                    "level": None,
                    "oof_accuracy": None,
                    "train_time": [elapsed],
                }
            )

        def _fit_one(ft):
            t0 = perf_counter()
            device = self._feature_device(ft)
            if ft.use_subprocess:
                _run_in_subprocess(
                    _fit_transform_in_subprocess,
                    (
                        ft.feature_name,
                        ft.feature_seed,
                        self.n_jobs,
                        X_path,
                        str(self._model_dir),
                        directory,
                        ft.get_feature_id(),
                        self.compute_dtype,
                        self.verbose,
                        device,
                        self.prune_constant,
                    ),
                )
            else:
                _fit_transform_inline(
                    ft.feature_name,
                    ft.feature_seed,
                    self.n_jobs,
                    X_path,
                    str(self._model_dir),
                    directory,
                    ft.get_feature_id(),
                    self.compute_dtype,
                    device,
                    self.prune_constant,
                )
            _log_feature(ft, t0)

        def _run_gpu_queue():
            try:
                for ft in gpu_features:
                    _fit_one(ft)
            except Exception as e:
                gpu_error.append(e)

        gpu_thread = threading.Thread(target=_run_gpu_queue, daemon=True)
        gpu_thread.start()

        for ft in cpu_features:
            _fit_one(ft)

        gpu_thread.join()
        if gpu_error:
            raise RuntimeError("GPU feature extraction failed") from gpu_error[0]

    def compute_features(self, X: np.ndarray, directory: str, start_time=None) -> None:
        compute_start = perf_counter()
        X_path = f"{directory}/X.npy"

        gpu_features, cpu_features = self._split_feature_lanes()
        gpu_error: list[BaseException] = []

        def _log_feature(ft, t0):
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            size_mb = Xt.nbytes / (1024 * 1024)
            device = self._feature_device(ft)
            log(
                f"Computed {ft.get_feature_id()} [{device}] features {Xt.shape} ({size_mb:.2f} MB) dtype={Xt.dtype} in {perf_counter() - t0:.4f}s",
                level=1,
                start_time=compute_start if start_time is None else start_time,
                verbose=self.verbose,
            )

        def _transform_one(ft):
            t0 = perf_counter()
            if ft.use_subprocess:
                _run_in_subprocess(
                    _transform_in_subprocess,
                    (
                        ft.get_feature_id(),
                        X_path,
                        str(self._model_dir),
                        directory,
                        self.compute_dtype,
                        self.verbose,
                    ),
                )
            else:
                _transform_inline(
                    ft.get_feature_id(),
                    X_path,
                    str(self._model_dir),
                    directory,
                    self.compute_dtype,
                )
            _log_feature(ft, t0)

        def _run_gpu_queue():
            try:
                for ft in gpu_features:
                    _transform_one(ft)
            except Exception as e:
                gpu_error.append(e)

        gpu_thread = threading.Thread(target=_run_gpu_queue, daemon=True)
        gpu_thread.start()

        for ft in cpu_features:
            _transform_one(ft)

        gpu_thread.join()
        if gpu_error:
            raise RuntimeError("GPU feature extraction failed") from gpu_error[0]

    # ----------------- fallback -----------------

    def _fit_fallback(self, X, y, fit_start_time):
        log(
            "Falling back to MultiRocketHydraClassifier", level=1, start_time=fit_start_time,
            verbose=self.verbose,
        )
        fallback = MultiRocketHydraClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        fallback.fit(X, y)
        save_model(fallback, "fallback", str(self._model_dir))
        log(
            "Fallback model trained successfully", level=1, start_time=fit_start_time,
            verbose=self.verbose,
        )

    # ----------------- training -----------------

    def _fit(self, X, y):
        fit_start = perf_counter()
        if self.compute_dtype is None:
            self.compute_dtype = np.asarray(X).dtype
        log(
            f"Starting fit, run_dir={self._base_dir}, n_jobs={self.n_jobs}",
            level=1,
            start_time=fit_start,
            verbose=self.verbose,
        )
        _cpu_max = os.cpu_count() or 1
        _cpu_used = _cpu_max if self.n_jobs == -1 else self.n_jobs
        log(
            f"CPUs set/available/used/ {_cpu_used}/{_cpu_max}/{_cpu_used}",
            level=1,
            start_time=fit_start,
            verbose=self.verbose,
        )
        try:
            import torch

            _gpu_torch = torch.cuda.device_count()
        except Exception:
            _gpu_torch = 0
        try:
            import subprocess

            _gpu_smi = len(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
                .splitlines()
            )
        except Exception:
            _gpu_smi = 0
        _gpu_used = 1 if self.n_gpus != 0 else 0
        log(
            f"GPUs set/available[torch]/available[smi]/used/ {_gpu_used}/{_gpu_torch}/{_gpu_smi}/{_gpu_used}",
            level=1,
            start_time=fit_start,
            verbose=self.verbose,
        )
        _direction = "minimize" if self.eval_metric == "log_loss" else "maximize"
        log(
            f"Eval metric: {self.eval_metric} ({_direction})",
            level=1,
            start_time=fit_start,
            verbose=self.verbose,
        )

        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._tmpdir, exist_ok=True)

        t0 = perf_counter()
        save_array(X, "X", str(self._tmpdir), dtype=self.compute_dtype)
        save_array(y, "y", str(self._tmpdir))
        log(
            f"Saved X and y to disk in {perf_counter() - t0:.2f}s (dtype={self.compute_dtype})",
            level=2,
            start_time=fit_start,
            verbose=self.verbose,
        )

        _, counts = np.unique(y, return_counts=True)
        if np.any(counts < 2):
            log(
                "Some classes have fewer than 2 instances, fold training not possible",
                level=1,
                start_time=fit_start,
                verbose=self.verbose,
            )
            self._fit_fallback(X, y, fit_start)
            return

        if self.cv_splits is None:
            self.cv_splits = []

        mp_ctx = multiprocessing.get_context("forkserver")

        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=mp_ctx) as executor:
                warm = [executor.submit(_noop) for _ in range(self.n_jobs)]
                predictions = []

                self.fit_transform_features(X, fit_start_time=fit_start)
                stacker_fold_seed = self._get_feature_seed()

                # -------- level 0 --------
                expected_folds = {
                    spec.get_model_id(): self.k_folds * spec.n_repetitions
                    for spec in self.model_specs
                }
                tasks = []
                for spec in self.model_specs:
                    fold_rng = np.random.default_rng(spec.model_seed)
                    fold_counter = 0
                    for fold_seed in spec.fold_seeds:
                        rep_splits = generate_folds(
                            X, y, n_splits=self.k_folds, n_repetitions=1, random_state=fold_seed
                        )
                        for _, (train_idx, val_idx) in enumerate(rep_splits):
                            fold_model_seed = int(fold_rng.integers(0, 2**31 - 1))
                            tasks.append(
                                (
                                    fold_counter,
                                    spec.get_model_id(),
                                    spec.model_name,
                                    train_idx,
                                    val_idx,
                                    fold_model_seed,
                                    str(self._tmpdir),
                                    list(spec.features),
                                    str(self._model_dir),
                                )
                            )
                            fold_counter += 1

                n_workers = min(self.n_jobs, len(tasks))
                log(
                    f"Starting training with {n_workers} workers for {len(tasks)} models",
                    level=2,
                    start_time=fit_start,
                    verbose=self.verbose,
                )

                base_kwargs = {"prepruned_features": self.prune_constant}
                futures = {
                    executor.submit(_train_one_model_v10, *t, model_kwargs=base_kwargs): t
                    for t in tasks
                }
                model_groups = defaultdict(list)
                model_train_times: dict[str, list[float]] = defaultdict(list)

                for future in as_completed(futures):
                    task = futures[future]
                    fold_number = task[0]
                    model_id_task = task[1]
                    try:
                        (
                            train_idx,
                            val_idx,
                            proba,
                            classes_,
                            model_size,
                            train_dur,
                            model_id_result,
                            fold_number,
                        ) = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed during training {model_id_task} fold {fold_number}: {e}"
                        ) from e

                    log(
                        f"Trained {model_id_result} in {train_dur:.4f}s for f-{fold_number} "
                        f"({model_size / (1024 * 1024):.2f} MB)",
                        level=2,
                        start_time=fit_start,
                        verbose=self.verbose,
                    )

                    missing = self._missing_classes(classes_)
                    if missing:
                        log(
                            f"{model_id_result} f-{fold_number} predicts "
                            f"{len(classes_)}/{self.n_classes_} classes (missing {missing}), "
                            "stacking not possible",
                            level=1,
                            start_time=fit_start,
                            verbose=self.verbose,
                        )
                        for pending in futures:
                            pending.cancel()
                        self._fit_fallback(X, y, fit_start)
                        return

                    new_preds = self.add_probabilities(
                        probas=proba,
                        classes=classes_,
                        model_name=model_id_result,
                        level=0,
                        indices=val_idx,
                    )
                    predictions.extend(new_preds)

                    model_groups[model_id_result].append(fold_number)
                    model_train_times[model_id_result].append(train_dur)
                    if len(model_groups[model_id_result]) == expected_folds[model_id_result]:
                        log(
                            f"Completed training for model {model_id_result}",
                            level=2,
                            start_time=fit_start,
                            verbose=self.verbose,
                        )
                        del model_groups[model_id_result]

                        predictions = self._save_model_predictions(
                            predictions, model_id_result, n_samples=X.shape[0], level=0
                        )
                        oof_score = self._compute_oof_score(y, model_id_result)
                        self._oof_scores.append(
                            {
                                "model": model_id_result,
                                "level": 0,
                                "eval_metric": self.eval_metric,
                                "oof_score": oof_score,
                                "train_time": model_train_times.pop(model_id_result),
                            }
                        )
                        log(
                            f"OOF {self.eval_metric} (base) {model_id_result}: {oof_score}",
                            level=1,
                            start_time=fit_start,
                            verbose=self.verbose,
                        )

                # -------- stacking --------
                prob_array = self._build_probability_array(n_samples=X.shape[0])
                if not self.stacking_models:
                    return
                log(
                    "Starting stacking model training", level=2, start_time=fit_start,
                    verbose=self.verbose,
                )
                if prob_array is None or np.isnan(prob_array).any():
                    log(
                        "NaN values detected in probability array, skipping stacking",
                        level=2,
                        start_time=fit_start,
                        verbose=self.verbose,
                    )
                    self._fit_fallback(X, y, fit_start)
                    return

                save_array(prob_array, "Xt_probabilities", str(self._tmpdir))

                stacker_splits = generate_folds(
                    X, y, n_splits=self.k_folds, n_repetitions=1, random_state=stacker_fold_seed
                )
                stack_tasks = []
                for model_name in self.stacking_models:
                    stack_fold_rng = np.random.default_rng(self._get_feature_seed())
                    for fold_no, (train_idx, val_idx) in enumerate(stacker_splits):
                        stack_fold_seed = int(stack_fold_rng.integers(0, 2**31 - 1))
                        stack_tasks.append(
                            (
                                fold_no,
                                model_name,  # model_id = model_name for stacking
                                model_name,
                                train_idx,
                                val_idx,
                                stack_fold_seed,
                                str(self._tmpdir),
                                [FeatureSpec(feature_name="probabilities")],
                                str(self._model_dir),
                            )
                        )

                n_workers = min(self.n_jobs, len(stack_tasks))
                log(
                    f"Starting stacking training with {n_workers} workers for {len(stack_tasks)} models",
                    level=2,
                    start_time=fit_start,
                    verbose=self.verbose,
                )

                futures = {executor.submit(_train_one_model_v10, *t): t for t in stack_tasks}
                model_groups = defaultdict(list)
                model_train_times: dict[str, list[float]] = defaultdict(list)

                for future in as_completed(futures):
                    task = futures[future]
                    fold_number = task[0]
                    model_id_task = task[1]
                    try:
                        (
                            train_idx,
                            val_idx,
                            proba,
                            classes_,
                            model_size,
                            train_dur,
                            model_id_result,
                            fold_number,
                        ) = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed during stacking training {model_id_task} fold {fold_number}: {e}"
                        ) from e

                    log(
                        f"Trained {model_id_result} in {train_dur:.4f}s for f-{fold_number} "
                        f"({model_size / (1024 * 1024):.2f} MB)",
                        level=2,
                        start_time=fit_start,
                        verbose=self.verbose,
                    )

                    missing = self._missing_classes(classes_)
                    if missing:
                        log(
                            f"Stacker {model_id_result} f-{fold_number} predicts "
                            f"{len(classes_)}/{self.n_classes_} classes (missing {missing}), "
                            "stacking not possible",
                            level=1,
                            start_time=fit_start,
                            verbose=self.verbose,
                        )
                        for pending in futures:
                            pending.cancel()
                        self._fit_fallback(X, y, fit_start)
                        return

                    new_preds = self.add_probabilities(
                        probas=proba,
                        classes=classes_,
                        model_name=model_id_result,
                        level=1,
                        indices=val_idx,
                    )
                    predictions.extend(new_preds)

                    model_groups[model_id_result].append(fold_number)
                    model_train_times[model_id_result].append(train_dur)
                    if len(model_groups[model_id_result]) == self.k_folds:
                        log(
                            f"Completed training for model {model_id_result}",
                            level=2,
                            start_time=fit_start,
                            verbose=self.verbose,
                        )
                        del model_groups[model_id_result]

                        predictions = self._save_model_predictions(
                            predictions, model_id_result, n_samples=X.shape[0], level=1
                        )
                        oof_score = self._compute_oof_score(y, model_id_result)
                        self._oof_scores.append(
                            {
                                "model": model_id_result,
                                "level": 1,
                                "eval_metric": self.eval_metric,
                                "oof_score": oof_score,
                                "train_time": model_train_times.pop(model_id_result),
                            }
                        )
                        log(
                            f"OOF {self.eval_metric} (stack) {model_id_result}: {oof_score}",
                            level=1,
                            start_time=fit_start,
                            verbose=self.verbose,
                        )

                log("Fit complete", level=1, start_time=fit_start, verbose=self.verbose)
                self._select_best_model()

        finally:
            if not self.keep_features and self._tmpdir and self._tmpdir.exists():
                cleanup_start = perf_counter()
                shutil.rmtree(self._tmpdir)
                log(
                    f"Cleaned up tmpdir in {perf_counter() - cleanup_start:.2f}s",
                    level=2,
                    start_time=fit_start,
                    verbose=self.verbose,
                )
                self._tmpdir = None
            if self.keep_features and self._tmpdir:
                self.features_training_dir_ = str(self._tmpdir)
            log("Executor shutdown complete", level=2, start_time=fit_start, verbose=self.verbose)

    def _select_best_model(self):
        if self.selection is None:
            return
        if self.selection == "best":
            candidates = self._oof_scores
        elif self.selection == "best-stacking":
            candidates = [s for s in self._oof_scores if s["level"] == 1]
        elif self.selection == "best-base":
            candidates = [s for s in self._oof_scores if s["level"] == 0]
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection!r}")
        if not candidates:
            return
        higher_is_better = self.eval_metric != "log_loss"
        self.best_model = (max if higher_is_better else min)(
            candidates, key=lambda s: s["oof_score"]
        )["model"]
        log(
            f"Selected best model ({self.selection}): {self.best_model}", level=1,
            verbose=self.verbose,
        )

    # ----------------- inspection helpers -----------------

    def _get_training_dir(self) -> str:
        d = getattr(self, "features_training_dir_", None) or (
            str(self._tmpdir) if self._tmpdir else None
        )
        if not self.keep_features or not d or not os.path.exists(d):
            raise RuntimeError(
                f"Not available. Set keep_features=True before fitting. keep_features={self.keep_features}, dir={d}"
            )
        return d

    def get_oof_predictions(self) -> pl.DataFrame:
        d = self._get_training_dir()
        frames = []
        for f in sorted(os.listdir(d)):
            if f.startswith("pred_") and f.endswith(".npy") and not f.endswith("_meta.npy"):
                model_name = f[5:-4]
                prob_array = read_array(f"pred_{model_name}", d)
                meta = read_array(f"pred_{model_name}_meta", d, allow_pickle=True, mmap_mode=None)
                _, classes = int(meta[0]), list(meta[1:])
                schema = [f"{model_name}|{cls}" for cls in classes]
                frames.append(pl.DataFrame(prob_array, schema=schema))
        return pl.DataFrame() if not frames else pl.concat(frames, how="horizontal")

    def get_features(self) -> pl.DataFrame:
        d = self._get_training_dir()
        frames = []
        for f in sorted(os.listdir(d)):
            if f.startswith("Xt_") and f.endswith(".npy") and f != "Xt_probabilities.npy":
                key = f[3:-4]
                arr = read_array(f[:-4], d)
                schema = [f"{key}|{i}" for i in range(arr.shape[1])]
                frames.append(pl.DataFrame(arr, schema=schema))
        return pl.DataFrame() if not frames else pl.concat(frames, how="horizontal")

    def summary(self, return_transforms: bool = False) -> list[dict]:
        if return_transforms:
            return self._transform_times + self._oof_scores
        return self._oof_scores

    # ----------------- inference -----------------

    def predict_proba_per_model(self, X: np.ndarray) -> dict[str, np.ndarray]:
        predict_start = perf_counter()
        log("Starting prediction", level=1, start_time=predict_start, verbose=self.verbose)

        mp_ctx = multiprocessing.get_context("forkserver")
        features_infer = self._base_dir / "features_inference"
        features_stack = self._base_dir / "features"

        os.makedirs(features_infer, exist_ok=True)
        self._tmpdir = features_infer

        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=mp_ctx) as executor:
                warm = [executor.submit(_noop) for _ in range(self.n_jobs)]

                # compute features (transform-only; transformers already trained)
                save_array(X, "X", str(features_infer), dtype=self.compute_dtype)
                self.compute_features(X, str(features_infer), start_time=predict_start)
                log(
                    "Computed and saved features for prediction", level=1, start_time=predict_start,
                    verbose=self.verbose,
                )

                predictions = []
                # ---- level 0 predictions ----
                tasks = []
                for spec in self.model_specs:
                    for fold in range(self.k_folds * spec.n_repetitions):
                        tasks.append(
                            (
                                spec.get_model_id(),
                                spec.model_name,
                                str(features_infer),
                                list(spec.features),
                                str(self._model_dir),
                                fold,
                            )
                        )

                log(
                    f"Starting prediction with {self.n_jobs} workers for {len(tasks)} first-level models",
                    level=1,
                    start_time=predict_start,
                    verbose=self.verbose,
                )

                futures = {executor.submit(_predict_one_model_v10, *t): t for t in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    model_id_task = task[0]
                    try:
                        proba, classes_, predict_dur, model_id_res = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed during prediction {model_id_task}: {e}"
                        ) from e

                    log(
                        f"Predicted {model_id_res} in {predict_dur:.4f}s",
                        level=2,
                        start_time=predict_start,
                        verbose=self.verbose,
                    )
                    predictions.extend(
                        self.add_probabilities(proba, classes_, model_id_res, level=0)
                    )

                log(
                    "Completed all first-level model predictions", level=1, start_time=predict_start,
                    verbose=self.verbose,
                )

                # ---- build stacking matrix ----
                if features_infer.exists():
                    shutil.rmtree(features_infer)
                os.makedirs(features_stack, exist_ok=True)
                self._tmpdir = features_stack

                if self._probability_columns is None:
                    raise RuntimeError(
                        "Probability column metadata missing. Fit the model before predicting."
                    )
                prob_array = self._aggregate_prediction_matrix(
                    predictions=predictions,
                    n_samples=X.shape[0],
                    probability_columns=self._probability_columns,
                )

                save_array(X, "X", str(features_stack), dtype=self.compute_dtype)
                save_array(prob_array, "Xt_probabilities", str(features_stack))

                # ---- stacking predictions ----
                stack_tasks = []
                for model_name in self.stacking_models:
                    for fold in range(self.k_folds):
                        stack_tasks.append(
                            (
                                model_name,  # model_id = model_name for stacking
                                model_name,
                                str(features_stack),
                                [FeatureSpec(feature_name="probabilities")],
                                str(self._model_dir),
                                fold,
                            )
                        )

                log(
                    f"Starting prediction with {self.n_jobs} workers for {len(stack_tasks)} stacking models",
                    level=1,
                    start_time=predict_start,
                    verbose=self.verbose,
                )

                futures = {executor.submit(_predict_one_model_v10, *t): t for t in stack_tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    model_id_task = task[0]
                    try:
                        proba, classes_, predict_dur, model_id_res = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed during stacking prediction {model_id_task}: {e}"
                        ) from e

                    log(
                        f"Predicted {model_id_res} in {predict_dur:.4f}s",
                        level=2,
                        start_time=predict_start,
                        verbose=self.verbose,
                    )
                    predictions.extend(
                        self.add_probabilities(proba, classes_, model_id_res, level=1)
                    )

            log(
                "Completed all stacking model predictions", level=1, start_time=predict_start,
                verbose=self.verbose,
            )

            model_ids = [spec.get_model_id() for spec in self.model_specs] + self.stacking_models
            out = {}
            for model_id in model_ids:
                level = 1 if model_id in self.stacking_models else 0
                cols = [self._probability_key(level, model_id, cls) for cls in self.classes_]
                out[model_id] = self._aggregate_prediction_matrix(
                    predictions=predictions,
                    n_samples=X.shape[0],
                    probability_columns=cols,
                )
            return out

        finally:
            for d in (features_infer, features_stack):
                if d.exists():
                    shutil.rmtree(d)
            self._tmpdir = None
            log(
                "Executor shutdown complete", level=1, start_time=predict_start,
                verbose=self.verbose,
            )

    def predict_per_model(self, X: np.ndarray) -> dict[str, np.ndarray]:
        proba_per_model = self.predict_proba_per_model(X)
        return {
            name: self.classes_[np.argmax(proba, axis=1)] for name, proba in proba_per_model.items()
        }


# ---------------------------------------------------------------------------
# Level-0 (base) pools, one per preset. Single source of truth: the classes
# below select among these rather than each redeclaring a list.
# ---------------------------------------------------------------------------

# The five representations every preset builds on.
_ENHANCED_BASE_MODELS = [
    "multirockethydra-bestk-p-ridgecv",
    "quant-etc",
    "rdst-p-ridgecv",
    "rstsf-random-etc",
    "fm-p-ridgecv",
]

# low = medium minus the ``weasel`` and ``fm`` representations (README preset table).
_ENHANCED_LOW_MODELS = [name for name in _ENHANCED_BASE_MODELS if name != "fm-p-ridgecv"]

# medium = the base five plus weasel.
_ENHANCED_MEDIUM_MODELS = _ENHANCED_BASE_MODELS + ["weasel-bestk-p-ridgecv"]

# high = every representation twice, once behind a RidgeCV head and once behind an
# ExtraTrees head. All names are distinct, so build_model_specs puts them in one
# group with a single FeatureSpec/seed per feature name: each representation is
# computed exactly once and feeds both of its heads.
_ENHANCED_HIGH_MODELS = [
    "multirockethydra-bestk-p-ridgecv",
    "multirockethydra-etc",
    "quant-etc",
    "quant-p-ridgecv",
    "rdst-p-ridgecv",
    "rdst-etc",
    "rstsf-random-etc",
    "rstsf-random-p-ridgecv",
    "fm-p-ridgecv",
    "fm-etc",
    "weasel-bestk-p-ridgecv",
    "weasel-etc",
]


def _robust_r2(y, pred):
    """Outlier-robust R²: ordinary R² on predictions clipped to the target range.

    Standard R² is dominated by its single largest squared residual, so one
    off-scale prediction (a high-leverage ridge extrapolation, e.g. -4 on a
    target in [0, 0.18]) can drive it to large negative values even when the
    model is good on the other 99% of samples. Clipping predictions to
    [min(y), max(y)] before scoring neutralises such nonsensical values — and is
    a no-op for well-behaved models (ETR, clipped variants), so it only changes
    the pathological cases. It also matches how ``ClippedRegressor`` actually
    serves predictions, and stays on the same scale as ``r2_score``.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(r2_score(y, np.clip(pred, np.nanmin(y), np.nanmax(y))))


def generate_folds(X, y, n_splits=5, n_repetitions=5, random_state=0, stratify=True):
    all_folds = []
    for i in range(n_repetitions):
        folds = utils.get_folds(
            X, y, n_splits=n_splits, random_state=random_state + i, stratify=stratify
        )
        all_folds.extend(folds)
    return all_folds


_VALID_EVAL_METRICS = {"accuracy", "f1", "log_loss", "roc_auc"}


_ENHANCED_PRESETS = ("low", "medium", "high")

# Level-1 pools. ``high`` is the union of the plain and
# balanced pools; ``probability-nn`` is a member of both (MLPClassifier has no
# ``class_weight``, so there is no balanced counterpart to add), which is why the
# union has 9 members and not 10.
_ENHANCED_PLAIN_STACKERS = [
    "probability-ridgecv",
    "probability-logisticcv",
    "probability-et",
    "probability-nn",
    "probability-rf",
]
_ENHANCED_BALANCED_STACKERS = [
    "probability-ridgecv-balanced",
    "probability-logisticcv-balanced",
    "probability-et-balanced",
    "probability-nn",
    "probability-rf-balanced",
]
_ENHANCED_HIGH_STACKERS = _ENHANCED_PLAIN_STACKERS + [
    name for name in _ENHANCED_BALANCED_STACKERS if name not in _ENHANCED_PLAIN_STACKERS
]

# The served head for every (preset, eval_metric) pair. At ``medium``/``high``
# every head is trained regardless, so eval_metric only picks which already
# fitted head is served — no extra compute. At ``low`` a single stacker is
# trained, so this table is also read in __init__ to decide *which* one.
_ENHANCED_SERVED_HEAD = {
    ("low", "accuracy"): "probability-ridgecv",
    ("low", "f1"): "probability-et",
    ("low", "roc_auc"): "probability-et",
    ("low", "log_loss"): "probability-et",
    ("medium", "accuracy"): "probability-stack-mean",
    ("medium", "f1"): "probability-stack-mean",
    ("medium", "roc_auc"): "probability-stack-mean",
    ("medium", "log_loss"): "probability-et",
    ("high", "accuracy"): "probability-stack-mean-balanced",
    ("high", "f1"): "probability-stack-mean-balanced",
    ("high", "roc_auc"): "probability-stack-mean-balanced",
    ("high", "log_loss"): "probability-et",
}


class TSCGlueEnhancedV3(LokyStackerV10Base):
    DEFAULT_MODEL_NAMES = list(_ENHANCED_MEDIUM_MODELS)
    MEAN_STACKER_NAME = "probability-stack-mean"
    BALANCED_MEAN_STACKER_NAME = "probability-stack-mean-balanced"
    MEAN_STACKER_EXCLUDE = ("probability-ridgecv", "probability-ridgecv-balanced")

    def __init__(
        self,
        random_state=None,
        k_folds=10,
        n_jobs=1,
        verbose=0,
        n_repetitions=1,
        n_gpus=0,
        runs_dir=None,
        eval_metric="accuracy",
        preset="medium",
        compute_dtype=None,
    ):
        assert n_gpus in (0, 1, -1), f"n_gpus must be 0, 1, or -1; got {n_gpus}"
        assert eval_metric in _VALID_EVAL_METRICS, (
            f"eval_metric must be one of {_VALID_EVAL_METRICS}; got {eval_metric!r}"
        )
        assert preset in _ENHANCED_PRESETS, (
            f"preset must be one of {_ENHANCED_PRESETS}; got {preset!r}"
        )
        self.preset = preset

        if preset == "low":
            model_names = list(_ENHANCED_LOW_MODELS)
            # Only one stacker is trained here, so the served head has to be
            # known pre-fit rather than picked from fitted candidates.
            stacking_models = [_ENHANCED_SERVED_HEAD[(preset, eval_metric)]]
        elif preset == "medium":
            model_names = list(_ENHANCED_MEDIUM_MODELS)
            stacking_models = list(_ENHANCED_PLAIN_STACKERS)
        else:  # high
            model_names = list(_ENHANCED_HIGH_MODELS)
            stacking_models = list(_ENHANCED_HIGH_STACKERS)

        super().__init__(
            random_state=random_state,
            n_repetitions=n_repetitions,
            k_folds=k_folds,
            n_jobs=n_jobs,
            keep_features=False,
            verbose=verbose,
            n_gpus=n_gpus,
            runs_dir=runs_dir,
            model_names=model_names,
            stacking_models=stacking_models,
            eval_metric=eval_metric,
            compute_dtype=compute_dtype,
        )

    def _mean_members(self, pool=None) -> list[str]:
        """Mean members of ``pool`` (default: every stacker), minus its ridge.

        The pool is an argument because ``high`` builds two means from two
        disjoint-except-``probability-nn`` halves of ``stacking_models``.
        """
        pool = self.stacking_models if pool is None else pool
        return [m for m in pool if m not in self.MEAN_STACKER_EXCLUDE]

    def _mean_pools(self) -> dict[str, list[str]]:
        """Mean candidate name -> the stackers it averages, for this preset."""
        if self.preset == "low":
            return {}
        pools = {self.MEAN_STACKER_NAME: _ENHANCED_PLAIN_STACKERS}
        if self.preset == "high":
            pools[self.BALANCED_MEAN_STACKER_NAME] = _ENHANCED_BALANCED_STACKERS
        return {name: self._mean_members(pool) for name, pool in pools.items()}

    def _select_best_model(self):
        if self.preset == "low":
            # One stacker, so there is nothing to select and no mean to build.
            # selection is None, so this keeps best_model = stacking_models[0]
            # from __init__ — the same head the table below resolves to.
            super()._select_best_model()
        self.best_model = _ENHANCED_SERVED_HEAD[(self.preset, self.eval_metric)]
        log(
            f"Serving {self.best_model} (preset={self.preset}, eval_metric={self.eval_metric})",
            level=1,
            verbose=self.verbose,
        )

    def _predict_proba(self, X):
        mean_pools = self._mean_pools()
        if self._fallback_path.exists() or self.best_model not in mean_pools:
            return super()._predict_proba(X)
        probas = self.predict_proba_per_model(X)
        return np.mean([probas[m] for m in mean_pools[self.best_model]], axis=0)

    def _fit_fallback(self, X, y, fit_start_time):
        # Local import: tscglue.fallback imports from this module.
        from tscglue.fallback import MRHydraET

        log("Falling back to MRHydraET", level=1, start_time=fit_start_time, verbose=self.verbose)
        fallback = MRHydraET(random_state=self.random_state, n_jobs=self.n_jobs)
        fallback.fit(X, y)
        save_model(fallback, "fallback", str(self._model_dir))
        log(
            "Fallback model trained successfully", level=1, start_time=fit_start_time,
            verbose=self.verbose,
        )


class TSCGlueEnhancedV4(TSCGlueEnhancedV3):
    def __init__(
        self,
        random_state=None,
        k_folds=10,
        n_jobs=1,
        verbose=0,
        n_repetitions=1,
        n_gpus=0,
        runs_dir=None,
        eval_metric="accuracy",
        preset="medium",
        prune_constant=True,
        compute_dtype=None,
    ):
        super().__init__(
            random_state=random_state,
            k_folds=k_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            n_repetitions=n_repetitions,
            n_gpus=n_gpus,
            runs_dir=runs_dir,
            eval_metric=eval_metric,
            preset=preset,
            compute_dtype=compute_dtype,
        )
        self.prune_constant = prune_constant


class TSCGlueClassifier(TSCGlueEnhancedV4):
    """The recommended TSCGlue stack: currently :class:`TSCGlueEnhancedV4`.

    A stable name for whichever version is current, so callers do not have to
    track the V-number. Everything -- presets, stacker pools, served-head table,
    GPU hydra, constant-column pruning -- is inherited from V4 untouched.
    """


def select_rows(arr, idx):
    """Rows of `arr` picked out by `idx`, or `arr` itself when `idx` is None."""
    if idx is None:
        return arr
    return arr[idx]


class DictMultiScaler(BaseEstimator, TransformerMixin):
    """
    Like MultiScaler but receives a dict of numpy arrays keyed by feature group name.

    Parameters
    ----------
    scalers : dict
        Maps feature group name to scaler instance.
        Example: {'hydra': SparseScaler(), 'multirocket': StandardScaler()}
    """

    def __init__(self, scalers):
        self.scalers = scalers

    def fit(self, X: dict[str, np.ndarray], y=None, idx=None):
        self.scalers_ = {}
        for key, scaler in self.scalers.items():
            if key in X:
                self.scalers_[key] = scaler
                if idx is not None:
                    scaler.fit(X[key][idx])
                else:
                    scaler.fit(X[key])
        return self

    def transform(self, X: dict[str, np.ndarray], idx=None):
        keys = [key for key in self.scalers_ if key in X]
        if not keys:
            return np.empty((next(iter(X.values())).shape[0], 0))

        widths = [X[key].shape[1] for key in keys]
        n_samples = X[keys[0]].shape[0] if idx is None else len(idx)
        dtype = np.result_type(*(X[key].dtype for key in keys))

        # Pre-allocate the full output once; fill it column-by-column so at most
        # one scaled feature-group chunk exists alongside it at any given time,
        # instead of holding every chunk plus a freshly hstack'd copy at once.
        out = np.empty((n_samples, sum(widths)), dtype=dtype)
        col = 0
        for key, width in zip(keys, widths):
            scaled = self.scalers_[key].transform(select_rows(X[key], idx))
            out[:, col:col + width] = scaled
            del scaled
            col += width

        return out

    def fit_transform(self, X: dict[str, np.ndarray], y=None, idx=None):
        return self.fit(X, y, idx=idx).transform(X, idx=idx)
