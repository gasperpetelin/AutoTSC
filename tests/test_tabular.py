import numpy as np
import pytest
from sklearn.base import clone

from tscglue.tabular import (
    AutoSelectKBestClassifier,
    RidgeClassifierCVDecisionProba,
    RidgeClassifierCVIndicator,
)

HEADS = [RidgeClassifierCVDecisionProba, RidgeClassifierCVIndicator, AutoSelectKBestClassifier]


def _make_data(n_classes=2, n_per_class=25, n_features=12, seed=0, str_labels=False):
    """Separable-but-noisy tabular blobs, one centroid per class."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, n_features)) * 2.0
    X = np.repeat(centers, n_per_class, axis=0) + rng.standard_normal(
        (n_classes * n_per_class, n_features)
    )
    y = np.repeat(np.arange(n_classes), n_per_class)
    if str_labels:
        y = np.array([str(v) for v in y])
    return X, y


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("str_labels", [False, True])
def test_predict_proba_rows_are_bounded_and_sum_to_one(head, n_classes, str_labels):
    X, y = _make_data(n_classes=n_classes, str_labels=str_labels)
    clf = head().fit(X, y)

    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], len(clf.classes_))
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("str_labels", [False, True])
def test_predict_proba_columns_follow_classes_order(head, n_classes, str_labels):
    X, y = _make_data(n_classes=n_classes, str_labels=str_labels)
    clf = head().fit(X, y)

    proba = clf.predict_proba(X)

    np.testing.assert_array_equal(clf.classes_[proba.argmax(axis=1)], clf.predict(X))


@pytest.mark.parametrize("head", HEADS)
def test_clone_preserves_constructor_params(head):
    clf = head(n_jobs=4)

    cloned = clone(clf)

    assert cloned.n_jobs == 4
    assert cloned.get_params() == clf.get_params()
    assert head().n_jobs == 1


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_n_jobs_does_not_change_the_fitted_model(head, n_jobs):
    X, y = _make_data(n_classes=3)
    baseline = head().fit(X, y)
    other = head(n_jobs=n_jobs).fit(X, y)

    # allclose, not exact: a different BLAS thread count reorders the reductions.
    np.testing.assert_allclose(
        other.predict_proba(X), baseline.predict_proba(X), rtol=1e-8, atol=1e-10
    )

