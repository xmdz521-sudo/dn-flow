import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "adult",
    "default",
    "shoppers",
    "magic",
    "beijing",
    "news",
    "bank_marketing",
    "bike_sharing",
)


def _write_mle_average_gap(run_name: str) -> None:
    rows = []
    for dataset in DATASETS:
        metrics_path = (
            ROOT
            / "outputs"
            / dataset
            / run_name
            / "metrics"
            / "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "mle_gap_percent" not in metrics:
            raise KeyError("{} does not contain mle_gap_percent".format(metrics_path))
        rows.append(
            {
                "dataset": dataset,
                "mle_gap_percent": float(metrics["mle_gap_percent"]),
            }
        )

    average = sum(row["mle_gap_percent"] for row in rows) / len(rows)
    summary_dir = ROOT / "outputs" / "summary" / run_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_dataset": {
            row["dataset"]: row["mle_gap_percent"] for row in rows
        },
        "mle_average_gap_percent": average,
    }
    (summary_dir / "mle_average_gap.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    table = pd.DataFrame(rows + [{
        "dataset": "Average",
        "mle_gap_percent": average,
    }])
    table.to_csv(summary_dir / "mle_average_gap.csv", index=False)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all eight DN-Flow datasets.")
    parser.add_argument("stage", choices=("train", "sample", "evaluate"))
    parser.add_argument("--run-name", default="paper")
    args, extra = parser.parse_known_args()
    for dataset in DATASETS:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "{}.py".format(args.stage)),
            "--config",
            str(ROOT / "configs" / "{}.yaml".format(dataset)),
            "--run-name",
            args.run_name,
        ] + extra
        subprocess.run(command, cwd=str(ROOT), check=True)
    if args.stage == "evaluate":
        _write_mle_average_gap(args.run_name)


if __name__ == "__main__":
    main()
