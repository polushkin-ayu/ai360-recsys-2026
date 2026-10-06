# FM(user,item): pilot sweep, 6 October 2026

12 measured CPU runs: k = 8, 16, 32, 64; seeds = 42, 43, 44.
Model is the existing user/item FM, equivalent to biased matrix factorization.
No test scores were calculated. No bias comparison was run in this task.
The proposed k grid and seeds still need team confirmation.

## Observations

| k | Validation RMSE mean | Sample SD (3 seeds) | Mean train RMSE at selected epoch | Parameters |
|---:|---:|---:|---:|---:|
| 8 | 1.031714 | 0.001476 | 0.835168 | 13,222 |
| 16 | 1.021908 | 0.002468 | 0.716727 | 24,974 |
| 32 | 1.021676 | 0.007762 | 0.692222 | 48,478 |
| 64 | 1.025772 | 0.007627 | 0.586021 | 95,486 |

The minimum mean validation RMSE is at k=32, but its advantage over k=16 is
only about 0.000232 and is small relative to the observed seed variability.
There is no evidence here of a reliable advantage for k=32. Larger k lowers
train RMSE without consistent validation improvement. Best epochs are 3-5;
the retained history includes subsequent epochs up to early stopping.
Regularization is fixed at the repository defaults, not tuned per k.
These are pilot observations on one subset/split, not final test results or
proof that FM beats bias. Sample SD is not a confidence interval.

## Files for combining results

- `summary.csv`: one row per model/k/seed, scores at the restored best-validation
  checkpoint, timing, parameter count, settings and code commit. `test_rmse` is blank.
- `aggregate.csv`: means and sample SD across three seeds for each k.
- `*_history.csv`: all completed epochs, train loss (MSE + L2), train RMSE and val RMSE.
- `manifest.json`: executable source/config hashes, clean code commit, environment,
  subset/split, coverage and canonical observation-order hashes.
- `selection.json`: provisional k selected by lowest mean validation RMSE.
- PNGs: validation RMSE against k and seed=42 learning curves.

Use the same source/subset/split rules, evaluation rows, metric and no clipping
for comparisons. Main evaluation covers known IDs only: train 10,572;
validation 2,209/2,265 (97.5276%); test 2,220/2,266 (97.9700%).
Train has 140 users and 1,329 items. Source and split seeds are both 42.

CSV artifact hashes and `split_id` can differ between Windows and Linux because
of line endings. Compare `validation_row_ids_sha256` (SHA-256 of ordered int64
little-endian row IDs), source checksum, subset/split config and counts as well.
Different code/settings/device/library versions must remain explicit.

To combine another model, produce an analogous per-run table with at least
`model,k,seed,val_rmse,test_rmse,split_id,validation_row_ids_sha256,code_commit`.
For a model without k (bias), leave k blank and plot its mean as a horizontal
baseline. Confirm evaluation alignment before combining; do not average
validation and test scores or different settings under one label.

Training weights and per-row predictions are local in
`data/fm-k-sweep-20261006/`; they are ignored by Git. All 12 checkpoints were
reloaded and reproduced the same validation predictions and best epoch.
Models are saved locally; collaborators can plot the committed CSVs without
weights, data download or retraining.

## Reproduce

Measured executable commit: `735f750f3c4eba07da8ee76a7df6730369d5e7d9`.
Environment and exact settings are in `manifest.json`; configuration is
`configs/fm-k-sweep.json`. Data preparation is documented in `data/README.md`.

```powershell
python -m src.run_fm_sweep --config configs/fm-k-sweep.json
python -m src.plot_fm_sweep --results results/fm-k-sweep-20261006
```

The trainer refuses to overwrite an existing nonempty result directory.
For another run, copy the config and change experiment_id/output_dir/local_dir;
keep original results. Commit the new config before measuring.
Plotting alone reads the committed CSV files and does not train a model.

Validation: repository suite ran 48 tests successfully, with 1 skip and 2
pre-existing expected failures. Each measured run additionally checked finite
histories, strict row alignment, best-validation restoration and checkpoint
prediction equality. A single-thread deterministic CPU loop was used.
Time covers the internal fit loop including epoch evaluation, excluding IO,
initialization, checkpoint checks and plotting; cross-person timing requires
recorded hardware and matching measurement boundaries.

Next: obtain team review, compare with bias on these rows, agree a small
regularization budget, and only then evaluate frozen choices on test.
