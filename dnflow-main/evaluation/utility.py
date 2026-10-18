from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def _encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def mle(
    synthetic: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task_type: str,
    numeric: List[str],
    categorical: List[str],
    seed: int = 42,
) -> Dict[str, float]:
    features = [column for column in synthetic.columns if column != target]
    num_features = [column for column in numeric if column != target]
    cat_features = [column for column in categorical if column != target]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), num_features),
            ("categorical", _encoder(), cat_features),
        ]
    )
    if task_type == "classification":
        if synthetic[target].isna().any() or test[target].isna().any():
            raise ValueError("classification targets must not contain missing values")
        target_encoder = LabelEncoder()
        target_encoder.fit(synthetic[target])
        if len(target_encoder.classes_) != 2:
            raise ValueError(
                "classification MLE requires exactly two target classes"
            )
        synthetic_target = target_encoder.transform(synthetic[target])
        try:
            test_target = target_encoder.transform(test[target])
        except ValueError as error:
            raise ValueError(
                "test target contains a class absent from the synthetic target"
            ) from error
        if len(np.unique(synthetic_target)) != 2:
            raise ValueError(
                "synthetic target must contain both classes for classification MLE"
            )
        if len(np.unique(test_target)) != 2:
            raise ValueError(
                "test target must contain both classes for classification MLE"
            )
        estimator = XGBClassifier(
            random_state=seed, eval_metric="logloss", n_jobs=1
        )
        pipeline = make_pipeline(preprocessor, estimator)
        pipeline.fit(synthetic[features], synthetic_target)
        probability = pipeline.predict_proba(test[features])[:, 1]
        return {"mle_auc": float(roc_auc_score(test_target, probability))}
    estimator = XGBRegressor(random_state=seed, n_jobs=1)
    pipeline = make_pipeline(preprocessor, estimator)
    pipeline.fit(synthetic[features], synthetic[target])
    prediction = pipeline.predict(test[features])
    return {
        "mle_rmse": float(mean_squared_error(test[target], prediction) ** 0.5)
    }


def average_gap(
    synthetic_scores: List[float],
    real_scores: List[float],
    higher_is_better: Optional[bool] = None,
) -> float:
    """Return the paper's mean absolute task-relative MLE gap in percent.

    ``higher_is_better`` is retained for backward API compatibility. The paper
    aggregates the absolute relative difference, so metric direction does not
    change the calculation.
    """
    synthetic = np.asarray(synthetic_scores, dtype=float)
    real = np.asarray(real_scores, dtype=float)
    if synthetic.shape != real.shape:
        raise ValueError("synthetic and real score arrays must have equal shape")
    if np.any(real == 0):
        raise ValueError("real-data reference scores must be non-zero")
    gap = np.abs(synthetic - real) / np.abs(real)
    return 100.0 * float(np.mean(gap))
