import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_metadata(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_schema(metadata: Dict[str, Any]) -> Dict[str, Any]:
    columns = metadata["column_names"]
    num_idx = list(metadata.get("num_col_idx", []))
    cat_idx = list(metadata.get("cat_col_idx", []))
    target_value = metadata["target_col_idx"]
    target_idx = int(target_value[0] if isinstance(target_value, list) else target_value)
    feature_idx = [index for index in range(len(columns)) if index != target_idx]
    # The target participates in generation. Its declared type is appended to the
    # corresponding group, preserving the formal data representation.
    num_all: List[int] = [index for index in num_idx if index in feature_idx]
    cat_all: List[int] = [index for index in cat_idx if index in feature_idx]
    if target_idx not in num_all and target_idx not in cat_all:
        if metadata["task_type"] == "regression":
            num_all.append(target_idx)
        else:
            cat_all.append(target_idx)

    integer_columns = list(metadata.get("int_columns", []))
    if not integer_columns:
        integer_columns = [
            columns[index]
            for index in metadata.get("int_col_idx", [])
            if 0 <= int(index) < len(columns)
        ]
    numeric_columns = [columns[index] for index in num_all]
    integer_columns = [
        column for column in integer_columns if column in numeric_columns
    ]

    return {
        "columns": columns,
        "numeric": numeric_columns,
        "categorical": [columns[index] for index in cat_all],
        "integer": integer_columns,
        "target": columns[target_idx],
    }


def read_table(path: Path, metadata: Dict[str, Any]) -> pd.DataFrame:
    header = metadata.get("header", "infer")
    frame = pd.read_csv(path, header=None if header is None else "infer")
    if header is None:
        frame.columns = metadata["column_names"]
        if len(frame) and [str(value) for value in frame.iloc[0].tolist()] == [
            str(value) for value in metadata["column_names"]
        ]:
            frame = frame.iloc[1:].reset_index(drop=True)
    expected = metadata["column_names"]
    if list(frame.columns) != expected:
        if len(frame.columns) != len(expected):
            raise ValueError("table column count does not match metadata")
        frame.columns = expected
    return frame
