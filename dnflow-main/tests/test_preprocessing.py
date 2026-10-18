import numpy as np
import pandas as pd

from dnflow.preprocessing import TabularPreprocessor


def test_preprocessing_round_trip():
    frame = pd.DataFrame(
        {
            "number": pd.Series([1, 2, 3, 4], dtype="int64"),
            "category": pd.Series(["a", "b", "a", "c"], dtype="object"),
        }
    )
    processor = TabularPreprocessor(
        ["number"], ["category"], n_quantiles=4, integer_columns=["number"]
    )
    values = processor.fit_transform(frame)
    restored = processor.inverse_transform(values, frame)
    assert list(restored.columns) == list(frame.columns)
    assert restored.dtypes.equals(frame.dtypes)
    assert np.array_equal(restored["category"], frame["category"])


def test_metadata_declared_integer_is_rounded_and_clipped():
    # CSV ingestion may represent an integer-semantic column as float.
    frame = pd.DataFrame(
        {
            "count": pd.Series([1.0, 2.0, 5.0, 9.0], dtype="float64"),
            "category": ["a", "b", "a", "b"],
        }
    )
    processor = TabularPreprocessor(
        ["count"],
        ["category"],
        n_quantiles=4,
        integer_columns=["count"],
    )
    values = processor.fit_transform(frame)
    # Force a continuous latent value; inverse output must still be integer-semantic.
    values[:, 0] += 0.17
    restored = processor.inverse_transform(values, frame)
    assert pd.api.types.is_integer_dtype(restored["count"].dtype)
    assert restored["count"].between(1, 9).all()
