"""Tests for TSCGlueClassifier, TSCGlueRegressor, and LokyStackerV10Base."""

import tempfile

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from tscglue import utils
from tscglue.models import (
    LokyStackerV10Base,
    TSCAGGlueClassifier,
    TSCGlueClassifier,
    TSCGlueDual,
    TSCGlueEnhancedV2,
    TSCGlueEnhancedV3,
    TSCGlueRegressor,
    get_feature_transformer,
)


def test_model_accuracy_on_coffee():
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


def test_ag_stacking_on_coffee():
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCAGGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


def _make_classification_data(n_per_class=10, n_classes=15, n_timesteps=16, seed=0):
    """Synthetic multiclass series with INTEGER labels 0..n_classes-1.

    15 integer-labeled classes is what exercises the multiclass roc_auc
    label-ordering path (labels 10..14 sort differently by repr vs numerically).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        center = rng.standard_normal(n_timesteps) + c
        for _ in range(n_per_class):
            X.append((center + 0.3 * rng.standard_normal(n_timesteps))[None, :])
            y.append(c)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int)


@pytest.mark.parametrize("eval_metric", ["accuracy", "log_loss", "roc_auc"])
def test_classifier_eval_metrics_multiclass(eval_metric):
    # 15 integer-labeled classes -> roc_auc would raise "labels must be ordered"
    # before the label-sorting fix.
    X_train, y_train = _make_classification_data(seed=0)
    X_test, y_test = _make_classification_data(seed=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0,
            n_repetitions=1,
            k_folds=3,
            n_jobs=2,
            eval_metric=eval_metric,
            runs_dir=tmp_dir,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

    assert y_pred.shape == (len(X_test),)
    assert set(np.unique(y_pred)).issubset(set(np.unique(y_train)))
    assert proba.shape == (len(X_test), len(np.unique(y_train)))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("compute_dtype", [None, "float32", "float64"])
def test_v10base_compute_dtype(compute_dtype):
    """Test that LokyStackerV10Base fit+predict works with different compute_dtype values."""
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = LokyStackerV10Base(
            random_state=270,
            n_repetitions=1,
            k_folds=10,
            compute_dtype=compute_dtype,
            runs_dir=tmp_dir,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    expected_dtype = np.dtype(compute_dtype) if compute_dtype else X_train.dtype
    assert model.compute_dtype == expected_dtype, (
        f"Expected compute_dtype={expected_dtype}, got {model.compute_dtype}"
    )


def test_model_on_multivariate():
    X_train, y_train, X_test, y_test = utils.load_dataset("BasicMotions")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


@pytest.mark.parametrize("encode_labels", [False, True], ids=["string_labels", "int_labels"])
def test_label_dtype(encode_labels):
    """Test that inference works with both string and integer labels."""
    X_train, y_train, X_test, y_test = utils.load_dataset("BasicMotions")

    if encode_labels:
        labels, y_train_fit = np.unique(y_train, return_inverse=True)
        y_test_expected = np.array([np.where(labels == x)[0][0] for x in y_test])
    else:
        y_train_fit = y_train
        y_test_expected = y_test

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=270, n_repetitions=1, k_folds=10, n_jobs=1, runs_dir=tmp_dir
        )
        model.fit(X_train, y_train_fit)
        y_pred = model.predict(X_test)
        proba_per_model = model.predict_proba_per_model(X_test)
        best_proba = proba_per_model[model.best_model]

    assert y_pred.shape == y_test_expected.shape
    assert best_proba.shape == (X_test.shape[0], len(model.classes_))
    assert np.isfinite(best_proba).all()

    accuracy = accuracy_score(y_test_expected, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


def test_dual_shares_feature_specs():
    """Both heads of each representation must reuse the identical cached feature (same seed)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueDual(random_state=0, runs_dir=tmp_dir)

    assert len(model.model_specs) == 12
    # 12 models but only 8 transforms: multirocket, hydra, quant, rdst,
    # rstsf-random, mantis, chronos2, weasel
    assert len(model.features_list) == 8

    specs = {spec.model_name: spec for spec in model.model_specs}
    for ridge_name, etc_name in [
        ("multirockethydra-bestk-p-ridgecv", "multirockethydra-etc"),
        ("quant-p-ridgecv", "quant-etc"),
        ("rdst-p-ridgecv", "rdst-etc"),
        ("rstsf-random-p-ridgecv", "rstsf-random-etc"),
        ("fm-p-ridgecv", "fm-etc"),
        ("weasel-bestk-p-ridgecv", "weasel-etc"),
    ]:
        ridge_ids = [ft.get_feature_id() for ft in specs[ridge_name].features]
        etc_ids = [ft.get_feature_id() for ft in specs[etc_name].features]
        assert ridge_ids == etc_ids, f"{ridge_name} and {etc_name} do not share features"


def test_dual_fit_predict():
    X_train, y_train = _make_classification_data(seed=0)
    X_test, _ = _make_classification_data(seed=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueDual(
            random_state=0, n_repetitions=1, k_folds=3, n_jobs=2, runs_dir=tmp_dir
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

    assert y_pred.shape == (len(X_test),)
    assert set(np.unique(y_pred)).issubset(set(np.unique(y_train)))
    assert proba.shape == (len(X_test), len(np.unique(y_train)))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_missing_classes_helper():
    """_missing_classes reports labels of y that a fitted model cannot predict."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = LokyStackerV10Base(random_state=0, runs_dir=tmp_dir)

    model.classes_ = np.array(["a", "b", "c"])
    assert model._missing_classes(np.array(["a", "b", "c"])) == []
    assert model._missing_classes(np.array(["a", "c"])) == ["b"]
    assert model._missing_classes(np.array(["c"])) == ["a", "b"]

    # Integer labels: comparison is on str, so dtype differences must not matter.
    model.classes_ = np.array([0, 1, 2])
    assert model._missing_classes(np.array([0, 1, 2], dtype=np.int32)) == []
    assert model._missing_classes(np.array([0, 2])) == [1]


class _PhantomClassStacker(LokyStackerV10Base):
    """Stack whose ``classes_`` holds a label no fold model can ever be fitted on.

    Stands in for a base model that comes back with a narrower ``classes_`` than
    ``y`` — the condition the fallback guard exists for, which is otherwise hard
    to provoke through stratified folds.
    """

    def _fit(self, X, y):
        self.classes_ = np.append(self.classes_, "__phantom__")
        self.n_classes_ = len(self.classes_)
        return super()._fit(X, y)


def test_fallback_when_model_misses_class():
    X_train, y_train = _make_classification_data(n_per_class=6, n_classes=3, seed=0)
    X_test, _ = _make_classification_data(n_per_class=2, n_classes=3, seed=1)
    y_train = y_train.astype(str)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = _PhantomClassStacker(
            random_state=0,
            k_folds=2,
            n_jobs=2,
            model_names=["quant-etc"],
            runs_dir=tmp_dir,
        )
        model.fit(X_train, y_train)

        assert model._fallback_path.exists(), "Incomplete classes_ did not trigger the fallback"
        # No stack was built, so nothing was scored or selected.
        assert model.summary() == []

        y_pred = model.predict(X_test)

    assert y_pred.shape == (len(X_test),)
    assert set(np.unique(y_pred)).issubset(set(np.unique(y_train)))


def test_hydra_transformer_is_device_routed():
    """A non-cpu device swaps aeon's hydra for the torch one; cpu keeps aeon's."""
    from aeon.transformations.collection.convolution_based._hydra import HydraTransformer

    from tscglue.features_gpu import HydraTransformerDevice

    assert isinstance(get_feature_transformer("hydra", seed=0), HydraTransformer)
    assert isinstance(get_feature_transformer("hydra", seed=0, device="cpu"), HydraTransformer)

    gpu = get_feature_transformer("hydra", seed=0, device="cuda")
    assert isinstance(gpu, HydraTransformerDevice)
    assert gpu.device == "cuda"
    assert gpu.random_state == 0


def _lane_names(model):
    gpu_features, cpu_features = model._split_feature_lanes()
    return [ft.feature_name for ft in gpu_features], [ft.feature_name for ft in cpu_features]


def test_enhanced_v3_puts_hydra_on_the_gpu_first():
    """V3 moves hydra into the device lane, ahead of the foundation models."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        v3 = TSCGlueEnhancedV3(random_state=0, n_gpus=1, preset="high", runs_dir=tmp_dir)
        v2 = TSCGlueEnhancedV2(random_state=0, n_gpus=1, preset="high", runs_dir=tmp_dir)

        v3_gpu, v3_cpu = _lane_names(v3)
        v2_gpu, v2_cpu = _lane_names(v2)

        assert v3_gpu == ["hydra", "mantis", "chronos2"]
        assert "hydra" not in v3_cpu
        assert v3._feature_device("hydra") == "cuda"

        # V2 is untouched: hydra stays on the cpu lane and is built for the cpu.
        assert v2_gpu == ["mantis", "chronos2"]
        assert "hydra" in v2_cpu
        assert v2._feature_device("hydra") == "cpu"

        # Only hydra moves -- every other feature is still built for the cpu.
        assert v3._feature_device("quant") == "cpu"
        assert v3._feature_device("multirocket") == "cpu"
        assert v3._feature_device("mantis") == "cuda"


def test_enhanced_v3_without_gpu_matches_v2():
    """n_gpus=0 makes V3 exactly V2: no device lane, hydra built for the cpu."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        v3 = TSCGlueEnhancedV3(random_state=0, n_gpus=0, preset="high", runs_dir=tmp_dir)
        v2 = TSCGlueEnhancedV2(random_state=0, n_gpus=0, preset="high", runs_dir=tmp_dir)

        assert _lane_names(v3) == _lane_names(v2)
        assert _lane_names(v3)[0] == []
        assert v3._feature_device("hydra") == "cpu"
        assert v3._feature_device("mantis") == "cpu"


def test_hydra_device_falls_back_to_cpu_without_cuda():
    """A cuda-fitted transformer stays usable where there is no cuda to unpickle onto."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("guard only fires when CUDA is absent")

    from tscglue.features_gpu import HydraTransformerDevice

    X, _ = _make_classification_data(n_per_class=3, n_classes=2, n_timesteps=64)

    transformer = HydraTransformerDevice(random_state=0, device="cuda").fit(X)
    with pytest.warns(RuntimeWarning, match="no CUDA device"):
        Xt = transformer.transform(X)

    # Kernels are drawn on the cpu, so falling back must reproduce the cpu run exactly.
    expected = HydraTransformerDevice(random_state=0, device="cpu").fit_transform(X)
    np.testing.assert_array_equal(Xt, expected)
    assert transformer.device == "cuda", "device is a constructor param and must not be mutated"


def _make_regression_data(n_train=40, n_test=15, n_channels=1, n_timesteps=30, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_channels, n_timesteps)).astype(np.float32)
    X_test = rng.standard_normal((n_test, n_channels, n_timesteps)).astype(np.float32)
    y_train = X_train[:, 0, :].mean(axis=1) + 0.1 * rng.standard_normal(n_train)
    y_test = X_test[:, 0, :].mean(axis=1) + 0.1 * rng.standard_normal(n_test)
    return X_train, y_train, X_test, y_test


def test_regressor_fit_predict_basic():
    X_train, y_train, X_test, y_test = _make_regression_data()

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueRegressor(random_state=0, k_folds=3, n_jobs=1, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    assert y_pred.shape == (len(X_test),), f"Expected shape ({len(X_test)},), got {y_pred.shape}"
    assert np.isfinite(y_pred).all(), "Predictions contain NaN or Inf"
    assert y_pred.dtype in (np.float32, np.float64), f"Unexpected dtype {y_pred.dtype}"


def test_regressor_summary():
    X_train, y_train, X_test, _ = _make_regression_data()

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueRegressor(random_state=0, k_folds=3, n_jobs=1, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        scores = model.summary()
        scores_with_transforms = model.summary(return_transforms=True)

    assert len(scores) > 0
    for entry in scores:
        assert "model" in entry
        assert "level" in entry
        assert "oof_rmse" in entry
        assert "oof_r2" in entry
        assert "train_time" in entry
        assert np.isfinite(entry["oof_rmse"]), f"oof_rmse is not finite for {entry['model']}"
        assert np.isfinite(entry["oof_r2"]), f"oof_r2 is not finite for {entry['model']}"

    assert len(scores_with_transforms) >= len(scores)
