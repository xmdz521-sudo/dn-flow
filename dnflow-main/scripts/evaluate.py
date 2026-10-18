import argparse
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dnflow.config import load_config
from dnflow.datasets import dataset_schema, load_metadata, read_table
from dnflow.utils import output_directories
from evaluation.tab_metrics import evaluate, save_metrics


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate DN-Flow synthetic data.")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--run-name", default="paper")
    result.add_argument("--samples", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config, ROOT)
    print(json.dumps(config, indent=2, sort_keys=True))
    paths = output_directories(ROOT, config["dataset"]["name"], args.run_name)
    sample_path = args.samples or paths["samples"] / "seed_0.csv"
    metadata = load_metadata(Path(config["dataset"]["metadata"]))
    schema = dataset_schema(metadata)
    schema["task_type"] = config["dataset"]["task_type"]
    real = read_table(Path(config["dataset"]["real"]), metadata)
    test = read_table(Path(config["dataset"]["test"]), metadata)
    synthetic = read_table(sample_path, metadata)
    development_size = (
        config["dataset"]["training_size"] + config["dataset"]["validation_size"]
    )
    if len(real) < development_size:
        raise ValueError("prepared development data is smaller than the paper split")
    if len(real) > development_size:
        real = real.sample(
            n=development_size, random_state=config["dataset"]["split_seed"]
        ).reset_index(drop=True)
    train, _ = train_test_split(
        real,
        test_size=config["dataset"]["validation_size"],
        random_state=config["dataset"]["split_seed"],
        shuffle=True,
    )
    result = evaluate(train, test, synthetic, schema)
    save_metrics(result, paths["metrics"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
