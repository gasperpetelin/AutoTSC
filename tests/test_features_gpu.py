"""Tests for tscglue.features_gpu."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from tscglue import utils
from tscglue.features_gpu import HydraTransformerDevice
from tscglue.models import SparseScaler
from tscglue.tabular import RidgeClassifierCVDecisionProba

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device available"
)

# Small UCR datasets, spanning short and long series while staying quick.
DATASETS = ["Coffee", "GunPoint", "ECG200", "ArrowHead", "SyntheticControl"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_gpu_features_match_cpu(dataset):
    """Same kernels on both devices, so only float32 rounding should separate them."""
    X = utils.load_dataset(dataset)[0]

    cpu = np.asarray(HydraTransformerDevice(random_state=0, device="cpu").fit_transform(X))
    gpu = np.asarray(HydraTransformerDevice(random_state=0, device="cuda").fit_transform(X))

    assert gpu.shape == cpu.shape
    rel_err = np.abs(cpu - gpu).max() / max(np.abs(cpu).max(), 1e-12)
    assert rel_err < 1e-4, f"{dataset}: max relative deviation {rel_err:.2e}"


def _accuracy(X_train, y_train, X_test, y_test, device):
    """Hydra features + the pipeline's ridge head (SparseScaler, then RidgeClassifierCV)."""
    transformer = HydraTransformerDevice(random_state=0, device=device)
    Xt_train = np.asarray(transformer.fit_transform(X_train), dtype=np.float64)
    Xt_test = np.asarray(transformer.transform(X_test), dtype=np.float64)

    scaler = SparseScaler()
    clf = RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10))
    clf.fit(scaler.fit_transform(Xt_train), y_train)
    return accuracy_score(y_test, clf.predict(scaler.transform(Xt_test)))


def test_gpu_accuracy_matches_cpu_across_datasets():
    """Training on gpu features must not cost accuracy relative to cpu features.

    Test sets this small make per-dataset accuracy jumpy -- one flipped prediction on
    Coffee is 3.6% -- so the per-dataset bound tolerates a single flip while the mean
    across datasets would expose a consistent bias in either direction.
    """
    deltas = {}
    for dataset in DATASETS:
        X_train, y_train, X_test, y_test = utils.load_dataset(dataset)
        deltas[dataset] = _accuracy(X_train, y_train, X_test, y_test, "cuda") - _accuracy(
            X_train, y_train, X_test, y_test, "cpu"
        )

    values = np.array(list(deltas.values()))
    detail = ", ".join(f"{k}={v:+.4f}" for k, v in deltas.items())

    assert np.abs(values).max() <= 0.05, f"per-dataset accuracy moved too far: {detail}"
    assert abs(values.mean()) <= 0.01, f"systematic accuracy bias on gpu: {detail}"
