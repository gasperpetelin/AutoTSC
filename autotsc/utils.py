"""Utility functions for AutoTSC."""

from aeon.datasets import load_classification


def load_dataset(dataset_name):
    """Load and normalize a dataset."""
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    return X_train, y_train, X_test, y_test
