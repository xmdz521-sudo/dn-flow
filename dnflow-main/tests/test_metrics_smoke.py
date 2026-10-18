import sys
import types

import numpy as np
import pandas as pd

from evaluation.fidelity import alpha_precision_beta_recall, c2st
from evaluation.privacy import dcr
from evaluation.utility import average_gap, mle


def _install_fake_synthcity(monkeypatch):
    class FakeLoader:
        def __init__(self, frame):
            self.frame = frame

    class FakeAlphaPrecision:
        def evaluate(self, real, synthetic):
            assert real.frame.shape[1] == synthetic.frame.shape[1]
            return {
                "delta_precision_alpha_naive": 0.9888,
                "delta_coverage_beta_naive": 0.5279,
                "delta_precision_alpha_OC": 0.0,
            }

    synthcity = types.ModuleType("synthcity")
    metrics = types.ModuleType("synthcity.metrics")
    metrics.eval_statistical = types.SimpleNamespace(
        AlphaPrecision=FakeAlphaPrecision
    )
    plugins = types.ModuleType("synthcity.plugins")
    core = types.ModuleType("synthcity.plugins.core")
    dataloader = types.ModuleType("synthcity.plugins.core.dataloader")
    dataloader.GenericDataLoader = FakeLoader

    monkeypatch.setitem(sys.modules, "synthcity", synthcity)
    monkeypatch.setitem(sys.modules, "synthcity.metrics", metrics)
    monkeypatch.setitem(sys.modules, "synthcity.plugins", plugins)
    monkeypatch.setitem(sys.modules, "synthcity.plugins.core", core)
    monkeypatch.setitem(
        sys.modules, "synthcity.plugins.core.dataloader", dataloader
    )


def test_lightweight_metrics(monkeypatch):
    rng = np.random.RandomState(42)
    real_array = rng.normal(size=(20, 4)).astype("float32")
    synthetic_array = rng.normal(size=(10, 4)).astype("float32")
    assert 0 <= c2st(real_array[:10], synthetic_array) <= 1

    _install_fake_synthcity(monkeypatch)
    real = pd.DataFrame(
        {"number": [1.0, 2.0, 3.0], "category": ["a", "b", "a"]}
    )
    synthetic = pd.DataFrame(
        {"number": [1.1, 2.1, 2.9], "category": ["a", "b", "a"]}
    )
    alpha, beta = alpha_precision_beta_recall(
        real, synthetic, ["number"], ["category"]
    )
    assert alpha == 0.9888
    assert beta == 0.5279
    assert np.isclose(average_gap([0.8], [0.9]), 100.0 / 9.0)
    # The paper uses an absolute relative gap, including when synthetic > real.
    assert np.isclose(average_gap([1.0], [0.9]), 100.0 / 9.0)


def test_dcr_comparison_direction_matches_paper():
    train = np.array([[0.0]], dtype=np.float32)
    test = np.array([[10.0]], dtype=np.float32)
    closer_to_train = np.array([[1.0]], dtype=np.float32)
    closer_to_test = np.array([[9.0]], dtype=np.float32)
    assert dcr(train, test, closer_to_train, chunk=1)["dcr_percent"] == 100.0
    assert dcr(train, test, closer_to_test, chunk=1)["dcr_percent"] == 0.0


def test_classification_mle_encodes_binary_targets():
    label_pairs = [
        (" <=50K", " >50K"),
        ("no", "yes"),
        ("h", "g"),
        (0, 1),
        (False, True),
    ]
    for negative, positive in label_pairs:
        synthetic = pd.DataFrame(
            {
                "value": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0] * 2,
                "group": ["a", "a", "b", "a", "b", "b"] * 2,
                "target": [negative] * 3
                + [positive] * 3
                + [negative] * 3
                + [positive] * 3,
            }
        )
        test = synthetic.sample(frac=1.0, random_state=7).reset_index(drop=True)
        result = mle(
            synthetic,
            test,
            target="target",
            task_type="classification",
            numeric=["value"],
            categorical=["group", "target"],
        )
        assert 0.0 <= result["mle_auc"] <= 1.0


def test_regression_mle_returns_finite_rmse():
    synthetic = pd.DataFrame(
        {
            "value": np.linspace(0.0, 1.0, 20),
            "group": ["a", "b"] * 10,
            "target": np.linspace(1.0, 3.0, 20),
        }
    )
    test = synthetic.sample(frac=1.0, random_state=11).reset_index(drop=True)
    result = mle(
        synthetic,
        test,
        target="target",
        task_type="regression",
        numeric=["value", "target"],
        categorical=["group"],
    )
    assert np.isfinite(result["mle_rmse"])
    assert result["mle_rmse"] >= 0.0
