from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class TabularPreprocessor:
    def __init__(
        self,
        numeric_columns: List[str],
        categorical_columns: List[str],
        n_quantiles: int = 1000,
        output_distribution: str = "normal",
        random_state: int = 42,
        integer_columns: Optional[List[str]] = None,
    ):
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.integer_columns = list(integer_columns or [])
        invalid_integer_columns = sorted(
            set(self.integer_columns) - set(self.numeric_columns)
        )
        if invalid_integer_columns:
            raise ValueError(
                "integer columns must also be numerical columns: {}".format(
                    ", ".join(invalid_integer_columns)
                )
            )
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.random_state = random_state
        self.transformer = None
        self._numeric_bounds: Dict[str, Tuple[float, float]] = {}
        self._reference_dtypes: Dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        missing = [
            column
            for column in self.numeric_columns + self.categorical_columns
            if column not in frame.columns
        ]
        if missing:
            raise ValueError("missing preprocessing columns: {}".format(missing))

        self._reference_dtypes = {
            column: frame[column].dtype for column in frame.columns
        }
        self._numeric_bounds = {}
        for column in self.numeric_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                self._numeric_bounds[column] = (
                    float(numeric.min()),
                    float(numeric.max()),
                )

        transformers = []
        if self.numeric_columns:
            transformers.append(
                (
                    "numeric",
                    QuantileTransformer(
                        n_quantiles=min(self.n_quantiles, len(frame)),
                        output_distribution=self.output_distribution,
                        random_state=self.random_state,
                    ),
                    self.numeric_columns,
                )
            )
        if self.categorical_columns:
            transformers.append(
                ("categorical", _one_hot_encoder(), self.categorical_columns)
            )
        self.transformer = ColumnTransformer(
            transformers, remainder="drop", sparse_threshold=0
        )
        self.transformer.fit(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("preprocessor is not fitted")
        return np.asarray(self.transformer.transform(frame), dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)

    @property
    def num_dim(self) -> int:
        return len(self.numeric_columns)

    def _restore_integer(self, values: pd.Series, column: str) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        bounds = self._numeric_bounds.get(column)
        if bounds is not None:
            numeric = numeric.clip(lower=bounds[0], upper=bounds[1])
        fill_value = bounds[0] if bounds is not None else 0
        numeric = numeric.fillna(fill_value).round()
        original_dtype = self._reference_dtypes.get(column)
        if original_dtype is not None and pd.api.types.is_integer_dtype(original_dtype):
            return numeric.astype(original_dtype)
        # Metadata-declared integer columns may have been read as float from CSV.
        return numeric.astype(np.int64)

    def inverse_transform(
        self, values: np.ndarray, reference: pd.DataFrame
    ) -> pd.DataFrame:
        if self.transformer is None:
            raise RuntimeError("preprocessor is not fitted")
        output: Dict[str, Any] = {}
        offset = 0
        if self.numeric_columns:
            scaler = self.transformer.named_transformers_["numeric"]
            width = len(self.numeric_columns)
            numeric = scaler.inverse_transform(values[:, :width])
            for index, column in enumerate(self.numeric_columns):
                output[column] = numeric[:, index]
            offset = width
        if self.categorical_columns:
            encoder = self.transformer.named_transformers_["categorical"]
            group_sizes = [len(categories) for categories in encoder.categories_]
            crisp = np.zeros((len(values), sum(group_sizes)), dtype=np.float32)
            start = offset
            crisp_offset = 0
            for size in group_sizes:
                block = values[:, start : start + size]
                crisp[
                    np.arange(len(values)),
                    crisp_offset + np.argmax(block, axis=1),
                ] = 1
                start += size
                crisp_offset += size
            categorical = encoder.inverse_transform(crisp)
            for index, column in enumerate(self.categorical_columns):
                output[column] = categorical[:, index]

        result = pd.DataFrame(output)[list(reference.columns)]
        for column in reference.columns:
            reference_dtype = reference[column].dtype
            if column in self.integer_columns or pd.api.types.is_integer_dtype(
                reference_dtype
            ):
                result[column] = self._restore_integer(result[column], column)
            elif pd.api.types.is_float_dtype(reference_dtype):
                result[column] = pd.to_numeric(
                    result[column], errors="coerce"
                ).astype(reference_dtype)
            else:
                result[column] = result[column].astype(reference_dtype)
        return result

    def dimensions(self) -> Tuple[int, int]:
        if self.transformer is None:
            raise RuntimeError("preprocessor is not fitted")
        cat_dim = 0
        if self.categorical_columns:
            encoder = self.transformer.named_transformers_["categorical"]
            cat_dim = sum(len(categories) for categories in encoder.categories_)
        return self.num_dim, self.num_dim + cat_dim
