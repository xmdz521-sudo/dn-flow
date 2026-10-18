from pathlib import Path

from dnflow.config import load_config


def test_all_dataset_configs_parse():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "configs").glob("*.yaml"))
    assert len(files) == 8
    for path in files:
        config = load_config(path, root)
        assert config["dataset"]["name"]
        assert config["dataset"]["split_seed"] == 42
