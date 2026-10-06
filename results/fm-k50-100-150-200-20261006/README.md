# FM: k = 50, 100, 150, 200

New pilot requested by the experiment author to use these four k values.
This is a MovieLens adaptation with user/item features, not a numerical
reproduction of the Netflix paper. First sweep remains separately available
in `results/fm-k-sweep-20261006/`; use this folder for the requested four-point curve.

12 measured runs, seeds 42/43/44. Same Team 1 data and training settings as
the first sweep: Adam lr=0.01, batch=1024, reg_bias=0.01, reg_factors=0.05,
max epochs=200, patience=20, init_std=0.01; deterministic CPU, one thread.
Test not evaluated. Epoch selected only by validation, restored before scoring.

| k | Mean validation RMSE | Sample SD | Mean train RMSE | Parameters |
|---:|---:|---:|---:|---:|
| 50 | 1.018902 | 0.000433 | 0.584098 | 74920 |
| 100 | 1.023652 | 0.002024 | 0.598042 | 148370 |
| 150 | 1.022458 | 0.004194 | 0.498498 | 221820 |
| 200 | 1.025652 | 0.003221 | 0.433949 | 295270 |

## Handoff

`summary.csv` contains one row per k/seed with train and validation RMSE at
the selected checkpoint, settings, best epoch, parameter count, time and code
commit. `aggregate.csv` holds means/sample SD; `*_history.csv` records every
completed epoch. `manifest.json` contains exact environment, config/source
hashes, coverage, split ID and canonical row-order hashes. `selection.json`
records the minimum mean validation RMSE within this sweep at fixed regularization.
The plots read these CSVs without retraining.

Data: 140 train users, 1329 items; train=10572, validation=2209, test=2220.
Use the same data and ordered evaluation row IDs for cross-model comparisons.
CSV line endings can change split_id between Windows/Linux; also compare the
canonical validation_row_ids_sha256, source checksum and subset/split rules.
Test RMSE is intentionally blank. No bias model was run in this series.

All 12 checkpoints were reloaded and verified to reproduce the same validation
predictions and best epoch. Each run checked finite history, strict row
alignment, and restoration of the minimum validation score. Local weights and
predictions: `data/fm-k50-100-150-200-20261006/` (ignored by Git).

## Observations and limits

k=50 has the lowest mean validation RMSE in this four-point grid. Increasing
k does not improve validation consistently; train error is substantially
lower than validation error. After the selected early epochs, continuing
training lowers train loss while validation generally worsens.
Do not claim significance or universal superiority: only three seeds, one
split and fixed regularization are studied. SD is sample variability, not a
confidence interval. This sweep does not establish a win over bias.
Final cross-model protocol/settings should be reviewed by the team before test.

## Reproduce and plot

Executable commit: `4ee07212010fe77c4fd6903338667743e50c9fa1` (clean working tree at launch).

```powershell
python -m src.run_fm_sweep --config configs/fm-k50-100-150-200.json
python -m src.plot_fm_sweep --results results/fm-k50-100-150-200-20261006
```

The trainer refuses to overwrite existing results. For reproduction use a
new checkout or a copied config with new experiment_id/output_dir/local_dir.
Save configuration in a commit before measuring. Plotting alone can use the
committed results directly.
