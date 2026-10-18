import argparse
import json
import sys
from pathlib import Path

import joblib
import torch
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dnflow.config import load_config
from dnflow.datasets import dataset_schema, load_metadata, read_table
from dnflow.model import DNFlow
from dnflow.preprocessing import TabularPreprocessor
from dnflow.trainer import train_model
from dnflow.utils import output_directories, set_seed, write_json


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Train the formal DN-Flow model.")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--run-name", default="paper")
    result.add_argument("--seed", type=int)
    result.add_argument("--epochs", type=int, help="Override for smoke tests only.")
    result.add_argument("--batch-size", type=int)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config, ROOT)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    print(json.dumps(config, indent=2, sort_keys=True))
    set_seed(config["training"]["seed"])
    metadata = load_metadata(Path(config["dataset"]["metadata"]))
    schema = dataset_schema(metadata)
    frame = read_table(Path(config["dataset"]["real"]), metadata)
    development_size = (
        config["dataset"]["training_size"] + config["dataset"]["validation_size"]
    )
    if len(frame) < development_size:
        raise ValueError(
            "{} requires {} development rows by the paper protocol, but {} "
            "were found".format(
                config["dataset"]["name"], development_size, len(frame)
            )
        )
    if len(frame) > development_size:
        frame = frame.sample(
            n=development_size, random_state=config["dataset"]["split_seed"]
        ).reset_index(drop=True)
    validation_size = config["dataset"]["validation_size"]
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=validation_size,
        random_state=config["dataset"]["split_seed"],
        shuffle=True,
    )
    preprocessing = config["preprocessing"]
    processor = TabularPreprocessor(
        schema["numeric"],
        schema["categorical"],
        preprocessing["n_quantiles"],
        preprocessing["output_distribution"],
        preprocessing["random_state"],
        integer_columns=schema["integer"],
    )
    train_array = processor.fit_transform(train_frame)
    validation_array = processor.transform(validation_frame)
    num_dim, in_dim = processor.dimensions()
    model = DNFlow(in_dim=in_dim, num_dim=num_dim, **config["model"])
    paths = output_directories(ROOT, config["dataset"]["name"], args.run_name)
    checkpoint = paths["checkpoints"] / "best.pt"
    result = train_model(
        model, train_array, validation_array, config, checkpoint, torch.device(args.device)
    )
    joblib.dump(processor, paths["root"] / "preprocessor.joblib")
    write_json(paths["root"] / "resolved_config.json", config)
    write_json(paths["root"] / "training_metrics.json", result)


if __name__ == "__main__":
    main()
