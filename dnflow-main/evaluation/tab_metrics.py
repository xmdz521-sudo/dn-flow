import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .fidelity import alpha_precision_beta_recall, c2st, encoded, shape_trend
from .privacy import dcr
from .utility import average_gap, mle


def evaluate(
    real: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: Dict[str, Any],
) -> Dict[str, float]:
    numeric = schema["numeric"]
    categorical = schema["categorical"]
    train_array, test_array, synthetic_array = encoded(
        real, [real, test, synthetic], numeric, categorical
    )
    shape, trend = shape_trend(real, synthetic, numeric, categorical)
    alpha, beta = alpha_precision_beta_recall(
        real, synthetic, numeric, categorical
    )
    result = {
        "shape": shape,
        "trend": trend,
        # The paper reports these two Synthcity metrics as percentages.
        "alpha_precision": 100.0 * alpha,
        "beta_recall": 100.0 * beta,
        "c2st_auc": c2st(train_array, synthetic_array),
    }
    result.update(dcr(train_array, test_array, synthetic_array))

    synthetic_mle = mle(
        synthetic,
        test,
        schema["target"],
        schema["task_type"],
        numeric,
        categorical,
    )
    real_mle = mle(
        real,
        test,
        schema["target"],
        schema["task_type"],
        numeric,
        categorical,
    )
    score_key = "mle_auc" if schema["task_type"] == "classification" else "mle_rmse"
    synthetic_score = synthetic_mle[score_key]
    real_score = real_mle[score_key]
    result.update(synthetic_mle)
    result["mle_real_{}".format(score_key.split("_", 1)[1])] = real_score
    result["mle_gap_percent"] = average_gap(
        [synthetic_score], [real_score]
    )
    return result


def save_metrics(result: Dict[str, float], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in result.items()]
    ).to_csv(output / "metrics.csv", index=False)
