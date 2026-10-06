# FM on full MovieLens 100K: k=50,100,150

The author requested retraining on the entire source dataset, replacing the
15,103-rating subset for this separate series. All 100,000 ratings and all
943 source users are included BEFORE splitting; the model is trained on
70,000 ratings, not on held-out validation/test ratings.

Random split seed 42, 70/15/15. Train-only ID mappings; unknown items are
excluded from the primary evaluation as in the original protocol.
Train vocabulary: 943 users, 1634 items.
Validation: 14971 / 15000 ratings
(99.8067%);
test: 14977 / 15000 (99.8467%).
Test ratings were not evaluated or used for training or model selection.

This series contains 9 measured runs (3 k values x 3 seeds).
Seeds=42/43/44. Same fixed training settings as previous FM runs:
Adam lr=0.01, batch=1024, reg_bias=0.01, reg_factors=0.05, init_std=0.01,
max epochs=200, patience=20; deterministic CPU, one thread, no clipping.

| k | Mean validation RMSE | Sample SD (3 seeds) | Mean train RMSE | Parameters |
|---:|---:|---:|---:|---:|
| 50 | 0.943125 | 0.002980 | 0.644036 | 131428 |
| 100 | 0.956798 | 0.000059 | 0.811092 | 260278 |
| 150 | 0.954139 | 0.001007 | 0.759224 | 389128 |

## Interpretation and handoff

The lowest mean validation RMSE within this fixed-regularization grid is
at k=50. This is a validation pilot, not a final test comparison.
Sample SD is seed variation, not a confidence interval. No bias run was added.
This full-data series must NOT be plotted against other models trained on
the earlier 15k subset as if they used the same split. Colleagues must use
`configs/full100k-data.json` and match the canonical evaluation row hash.
Earlier FM series and their data remain untouched.

- `summary.csv`: 9 per-run rows, scores at restored best validation checkpoint,
  settings, commit, timing, count, history/checkpoint paths; test_rmse blank.
- `aggregate.csv`: mean/sample SD per k.
- `*_history.csv`: train loss (MSE+L2), train RMSE, validation RMSE each epoch.
- `manifest.json`: source/config hashes, clean executable commit, environment,
  hardware, coverage, split ID and canonical little-endian int64 row-ID hashes.
- `data_checks.json`: full-dataset preparation checks (K5/K6 passed).
  Its legacy run_id is hardcoded by the data checker; config/command identify
  this actual full100k run.
- `selection.json`: provisional minimum mean validation score.
- PNGs: k curve and seed42 learning curves. Redrawing requires only CSVs.

Local weights and predictions are in `data/fm-full100k-k50-100-150-20261006/`
and ignored by Git. Every checkpoint was reloaded and reproduced identical
validation predictions. Every run checked finite histories, strict row IDs,
and restored minimum validation RMSE. No changes to model math or trainer.
Fit timing includes training and epoch evaluation, excludes setup/IO/plots.
CSV line endings may change split_id across OS; compare source/subset/split
rules and canonical ordered row hash as well.

## Reproduction

Executable configuration commit: `16595d729e5797803cc0bf0ec45f11e5393e7c65`.

```powershell
python -m src.prepare_data --config configs/full100k-data.json --report data/full100k-checks.json
python -m src.run_fm_sweep --config configs/fm-full100k-k50-100-150.json
python -m src.plot_fm_sweep --results results/fm-full100k-k50-100-150-20261006 --title "MovieLens 100K: FM at fixed regularization"
```

Trainer refuses to overwrite an existing nonempty results folder. Use a
fresh checkout or a copied config with new experiment/output/local dirs.
Next: team confirms the full100k comparison protocol; other model authors
produce scores on the same rows; freeze settings before final test evaluation.
