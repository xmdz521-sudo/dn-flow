import argparse
import json
import sys
from pathlib import Path

import joblib
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dnflow.config import load_config
from dnflow.datasets import load_metadata, read_table
from dnflow.model import DNFlow
from dnflow.sampler import sample_heun
from dnflow.utils import output_directories, set_seed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Sample from a trained DN-Flow model.")
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--run-name", default="paper")
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--num-samples", type=int)
    result.add_argument("--steps", type=int)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config, ROOT)
    if args.steps is not None:
        config["sampling"]["steps"] = args.steps
    print(json.dumps(config, indent=2, sort_keys=True))
    set_seed(args.seed)
    paths = output_directories(ROOT, config["dataset"]["name"], args.run_name)
    checkpoint = args.checkpoint or paths["checkpoints"] / "best.pt"
    processor = joblib.load(paths["root"] / "preprocessor.joblib")
    state = torch.load(checkpoint, map_location=args.device)
    num_dim, in_dim = processor.dimensions()
    model = DNFlow(in_dim=in_dim, num_dim=num_dim, **config["model"]).to(args.device)
    model.load_state_dict(state["model"])
    metadata = load_metadata(Path(config["dataset"]["metadata"]))
    reference = read_table(Path(config["dataset"]["real"]), metadata)
    count = args.num_samples or config["dataset"]["training_size"]
    values = sample_heun(
        model, count, in_dim, config["sampling"]["steps"], torch.device(args.device),
        self_conditioning=config["training"]["self_conditioning_probability"] > 0,
    )
    synthetic = processor.inverse_transform(values, reference)
    synthetic.to_csv(paths["samples"] / "seed_{}.csv".format(args.seed), index=False)


if __name__ == "__main__":
    main()
