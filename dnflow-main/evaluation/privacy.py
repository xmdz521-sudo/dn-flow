from typing import Dict

import numpy as np


def _minimum_l1(query: np.ndarray, reference: np.ndarray, chunk: int) -> np.ndarray:
    result = np.full(len(query), np.inf, dtype=np.float64)
    for query_start in range(0, len(query), chunk):
        query_block = query[query_start : query_start + chunk]
        minimum = np.full(len(query_block), np.inf, dtype=np.float64)
        for reference_start in range(0, len(reference), chunk):
            reference_block = reference[reference_start : reference_start + chunk]
            distances = np.abs(
                query_block[:, None, :] - reference_block[None, :, :]
            ).sum(axis=2)
            minimum = np.minimum(minimum, distances.min(axis=1))
        result[query_start : query_start + len(query_block)] = minimum
    return result


def dcr(
    train: np.ndarray, test: np.ndarray, synthetic: np.ndarray, chunk: int = 512
) -> Dict[str, float]:
    """Percentage of synthetic rows closer to train than held-out test.

    This comparison direction matches the paper's Table 11 and the uploaded
    original metric implementation: ``distance_to_train < distance_to_test``.
    A score close to 50% is preferred.
    """
    count = min(len(test), len(synthetic))
    syn = synthetic[:count]
    distance_to_train = _minimum_l1(syn, train, chunk)
    distance_to_test = _minimum_l1(syn, test, chunk)
    closer_to_train = distance_to_train < distance_to_test
    score = 100.0 * float(np.mean(closer_to_train))
    return {
        "dcr_percent": score,
        "dcr_absolute_deviation_from_50": abs(score - 50.0),
    }
