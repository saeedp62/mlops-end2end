"""
class_weights.py - Utilities for handling imbalanced datasets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlops_utils.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


def compute_class_weights(y: pd.Series, verbose: bool = True) -> dict[Any, float]:
    """Compute balanced class weights for a Pandas Series of labels.
    
    Weights are calculated inversely proportional to class frequencies,
    similar to scikit-learn's `compute_class_weight("balanced", ...)`.
    
    Parameters
    ----------
    y:
        A Pandas Series containing class labels.
    verbose:
        If True, logs the computed weights.
        
    Returns
    -------
    A dictionary mapping each class label to its computed weight.
    """
    import numpy as np

    classes = np.unique(y)
    class_counts = y.value_counts().to_dict()
    n_samples = len(y)
    n_classes = len(classes)

    weights = {}
    for cls in classes:
        # formula: n_samples / (n_classes * count)
        count = class_counts.get(cls, 0)
        if count == 0:
            weights[cls] = 1.0
        else:
            weights[cls] = n_samples / (n_classes * count)

    if verbose:
        logger.info("Computed class weights: %s", weights)

    return weights
