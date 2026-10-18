from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED = {
    "dataset": ("name", "task_type", "target", "metadata", "real", "test", "training_size", "validation_size", "split_seed"),
    "model": ("hidden_dim", "layers", "heads", "feedforward_dim", "dropout", "alpha", "selector_init_bias"),
    "training": ("epochs", "batch_size", "optimizer", "learning_rate", "weight_decay", "seed", "training_seeds"),
    "sampling": ("steps", "solver", "sampling_seeds"),
    "loss": ("flow", "driver", "reconstruction", "correlation", "sparsity", "binary"),
    "preprocessing": ("n_quantiles", "output_distribution", "random_state"),
}


def _validate(config: Dict[str, Any]) -> None:
    for section, keys in REQUIRED.items():
        if section not in config or not isinstance(config[section], dict):
            raise ValueError("missing configuration section: {}".format(section))
        missing = [key for key in keys if key not in config[section]]
        if missing:
            raise ValueError("{} missing fields: {}".format(section, ", ".join(missing)))
    if config["dataset"]["task_type"] not in ("classification", "regression"):
        raise ValueError("task_type must be classification or regression")
    if config["sampling"]["solver"].lower() != "heun":
        raise ValueError("the formal DN-Flow configuration requires the Heun solver")
    if config["sampling"]["steps"] <= 0 or config["training"]["epochs"] <= 0:
        raise ValueError("epochs and sampling steps must be positive")


def load_config(path: Path, project_root: Path = None) -> Dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _validate(config)
    root = Path(project_root).resolve() if project_root else path.parent.parent.resolve()
    config["project_root"] = str(root)
    for key in ("metadata", "real", "test"):
        value = Path(config["dataset"][key])
        config["dataset"][key] = str(value if value.is_absolute() else root / value)
    return config
