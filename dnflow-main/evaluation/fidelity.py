from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _quality_encoder() -> OneHotEncoder:
    # Matches the original alpha-Precision / beta-Recall evaluation script:
    # fit categorical levels on real data and use the resulting dense one-hot block.
    try:
        return OneHotEncoder(sparse_output=False)
    except TypeError:
        return OneHotEncoder(sparse=False)


def encoded(
    fit_frame: pd.DataFrame,
    frames: List[pd.DataFrame],
    numeric: List[str],
    categorical: List[str],
) -> List[np.ndarray]:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", _encoder(), categorical),
        ],
        sparse_threshold=0,
    )
    transformer.fit(fit_frame)
    return [
        np.asarray(transformer.transform(frame), dtype=np.float32)
        for frame in frames
    ]


def shape_trend(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
) -> Tuple[float, float]:
    try:
        from sdmetrics.reports.single_table import QualityReport
    except ImportError as error:
        raise RuntimeError("Shape and Trend require sdmetrics") from error
    metadata = {
        "columns": {
            column: {
                "sdtype": "numerical" if column in numeric else "categorical"
            }
            for column in real.columns
        }
    }
    report = QualityReport()
    report.generate(real, synthetic, metadata, verbose=False)
    properties = report.get_properties().set_index("Property")["Score"]
    return float(properties["Column Shapes"]), float(
        properties["Column Pair Trends"]
    )


def c2st(real: np.ndarray, synthetic: np.ndarray, seed: int = 42) -> float:
    count = min(len(real), len(synthetic))
    rng = np.random.RandomState(seed)
    x = np.concatenate((real[:count], synthetic[:count]))
    y = np.concatenate((np.zeros(count), np.ones(count)))
    order = rng.permutation(len(x))
    split = int(0.8 * len(x))
    train, test = order[:split], order[split:]
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(x[train], y[train])
    return float(
        roc_auc_score(y[test], classifier.predict_proba(x[test])[:, 1])
    )


def _quality_feature_frames(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the exact feature representation used by the original metric code."""
    real_parts = []
    synthetic_parts = []

    if numeric:
        real_parts.append(
            real[numeric].apply(pd.to_numeric, errors="coerce").to_numpy()
        )
        synthetic_parts.append(
            synthetic[numeric]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
        )

    if categorical:
        real_categorical = real[categorical].to_numpy().astype(str)
        synthetic_categorical = synthetic[categorical].to_numpy().astype(str)
        encoder = _quality_encoder()
        encoder.fit(real_categorical)
        real_parts.append(np.asarray(encoder.transform(real_categorical)))
        synthetic_parts.append(
            np.asarray(encoder.transform(synthetic_categorical))
        )

    if not real_parts:
        raise ValueError("alpha-Precision/beta-Recall require at least one feature")

    real_array = np.concatenate(real_parts, axis=1).astype(float)
    synthetic_array = np.concatenate(synthetic_parts, axis=1).astype(float)
    if not np.isfinite(real_array).all():
        raise ValueError("real data contain non-finite values after metric encoding")
    if not np.isfinite(synthetic_array).all():
        raise ValueError(
            "synthetic data contain non-finite values after metric encoding"
        )
    return pd.DataFrame(real_array), pd.DataFrame(synthetic_array)


def alpha_precision_beta_recall(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numeric: List[str],
    categorical: List[str],
) -> Tuple[float, float]:
    """Compute the paper's Synthcity naive alpha-Precision and beta-Recall.

    This follows the uploaded original evaluation script: numerical values are
    kept on their original scale, categorical values are one-hot encoded using
    levels fitted on real data, and Synthcity's ``AlphaPrecision`` evaluator is
    queried for ``delta_precision_alpha_naive`` and
    ``delta_coverage_beta_naive``. Returned values are fractions in [0, 1].
    """
    try:
        from synthcity.metrics import eval_statistical
        from synthcity.plugins.core.dataloader import GenericDataLoader
    except ImportError as error:
        raise RuntimeError(
            "alpha-Precision and beta-Recall require synthcity==0.2.12"
        ) from error

    real_encoded, synthetic_encoded = _quality_feature_frames(
        real, synthetic, numeric, categorical
    )
    evaluator = eval_statistical.AlphaPrecision()
    values = evaluator.evaluate(
        GenericDataLoader(real_encoded), GenericDataLoader(synthetic_encoded)
    )
    naive = {key: value for key, value in values.items() if "naive" in key}
    alpha_key = "delta_precision_alpha_naive"
    beta_key = "delta_coverage_beta_naive"
    missing = [key for key in (alpha_key, beta_key) if key not in naive]
    if missing:
        raise KeyError(
            "Synthcity AlphaPrecision result is missing: {}".format(
                ", ".join(missing)
            )
        )
    return float(naive[alpha_key]), float(naive[beta_key])
