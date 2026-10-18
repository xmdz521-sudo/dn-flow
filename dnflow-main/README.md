# DN-Flow

DN-Flow is a Driver–Navigator structured flow-matching model for mixed-type
tabular data. A shared time-conditioned Transformer predicts the full transport
velocity (Driver). A learned column-wise selector permits the Navigator to refine
only numerical endpoints, while a dynamic gate controls the correction for each
sample, time and dimension. Categorical one-hot dimensions are generated only by
the Driver path.

No license is included. Add one only after the repository owner has selected it.

## Structure

- `src/dnflow/`: shared model, losses, preprocessing, training and sampling.
- `configs/`: paper configurations for eight datasets.
- `scripts/`: training, sampling, evaluation and batch entry points.
- `evaluation/`: Shape, Trend, MLE, alpha-Precision, beta-Recall, C2ST and DCR.
- `data/metadata/`: uploadable dataset schemas.
- `tests/`: lightweight tests that do not reproduce full training.


## Installation

Python 3.8 is supported.

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data

The supported datasets are Adult, Default, Shoppers, Magic, Beijing, News,
Bank Marketing and Bike Sharing. Raw data is not tracked. Follow
`data/README.md` and place `real.csv` and `test.csv` under
`data/<dataset>/`.

## Commands

Train one dataset:

```bash
python scripts/train.py --config configs/adult.yaml --run-name paper
```

Sample from its best checkpoint:

```bash
python scripts/sample.py --config configs/adult.yaml --run-name paper --seed 0
```

Evaluate:

```bash
python scripts/evaluate.py --config configs/adult.yaml --run-name paper
```

Run a stage over all eight datasets:

```bash
python scripts/run_all.py train --run-name paper
python scripts/run_all.py sample --run-name paper
python scripts/run_all.py evaluate --run-name paper
```

## Outputs

Each run writes to `outputs/<dataset>/<run_name>/`:

- `checkpoints/best.pt`
- `samples/seed_<seed>.csv`
- `metrics/metrics.json` and `metrics/metrics.csv`
- `outputs/summary/<run_name>/mle_average_gap.json` and `.csv` after batch evaluation
- `resolved_config.json`
- `preprocessor.joblib`
- `training_metrics.json`



## Metrics

Shape and Trend use the SDMetrics aggregate properties. MLE trains one predictor
on synthetic training data and a reference predictor on real training data, then
evaluates both on held-out real data. AUC is reported for classification and RMSE
for regression. The per-dataset MLE gap is `|synthetic - real| / |real| * 100`;
`run_all.py evaluate` averages the eight dataset gaps to reproduce Average Gap.
Alpha-Precision and beta-Recall use Synthcity 0.2.12 `AlphaPrecision` with the
`delta_precision_alpha_naive` and `delta_coverage_beta_naive` outputs from the
original evaluation code, and are reported as percentages. C2ST reports
logistic-regression AUC; 0.5 is ideal. DCR reports the percentage of synthetic
samples whose nearest training record is closer than their nearest held-out
record; 50% is ideal, and its absolute deviation from 50 is also reported.

Metric directions:

| Metric | Preferred direction |
|---|---|
| Shape, Trend, MLE AUC, alpha-Precision, beta-Recall | Higher |
| MLE RMSE, Average Gap | Lower |
| C2ST AUC | Closer to 0.5 |
| DCR | Closer to 50% |

## Common errors

- Missing CSV: verify the paths described in `data/README.md`.
- CUDA out of memory: reduce CLI batch size for diagnostics; do not report that
  run as the paper configuration.
- Missing SDMetrics: install the pinned requirements.
- Incompatible checkpoint: use a checkpoint and resolved configuration from the
  same run.
- Column mismatch: ensure CSV columns exactly match the corresponding metadata.
