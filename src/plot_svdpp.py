"""Rebuild SVD++ figures from saved CSV files, without training."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

COLOR = '#156baf'
LABEL = 'Canonical SVD++'


def save(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def validate(table):
    if set(table.model) != {'svdpp'}:
        raise ValueError('only SVD++ measurements belong in this study')
    if len(table) != 15 or table.run_id.duplicated().any():
        raise ValueError('the complete study requires 15 unique SVD++ runs')
    if table.groupby('k').size().ne(3).any():
        raise ValueError('each k requires three completed seeds')


def rmse_curve(table, output, split):
    metric = f'{split}_rmse'
    if table[metric].isna().any():
        raise ValueError(f'{split} has incomplete measurements')
    group = table.groupby('k')[metric]
    means, std = group.mean(), group.std(ddof=1)
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    ax.errorbar(means.index, means, yerr=std, marker='o', color=COLOR,
                label=LABEL, linewidth=2, capsize=4)
    ax.set(xlabel='k (latent factors)', ylabel=f'{split.capitalize()} RMSE',
           xticks=means.index, title='SVD++ — MovieLens 100K subset, fixed random split')
    ax.grid(alpha=.25)
    ax.legend(loc='best')
    fig.text(.5, .005, 'Mean ± sample standard deviation over seeds 42, 43, 44; no clipping', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, 1))
    save(fig, output, f'rmse_vs_k_{split}')


def training_curves(table, metrics_path, output):
    best_k = table.groupby('k').validation_rmse.mean().sort_values(kind='stable').index[0]
    group = table[table.k == best_k]
    median = group.validation_rmse.median()
    run = group.assign(distance=(group.validation_rmse - median).abs()).sort_values(['distance', 'initialization_seed']).iloc[0]
    history = pd.read_csv(metrics_path.parent / 'histories' / f'{run.run_id}.csv')
    fig, axes = plt.subplots(3, 1, figsize=(8.7, 9.5), sharex=True)
    label = f'{LABEL}, k={int(run.k)}, seed={int(run.initialization_seed)}'
    for ax, metric, ylabel in zip(axes, ('train_mse', 'train_loss_with_regularization', 'val_rmse'),
                                 ('Train MSE', 'Train MSE + L2', 'Validation RMSE'), strict=True):
        ax.plot(history.epoch, history[metric], label=label, color=COLOR, linewidth=1.8)
        ax.axvline(run.best_epoch, color=COLOR, alpha=.55, linestyle=':')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel('Epoch (dotted line: selected checkpoint)')
    fig.suptitle('SVD++ training dynamics — run chosen using validation')
    fig.tight_layout(rect=(0, 0, 1, .98))
    save(fig, output, 'training_curves')
    pd.DataFrame([dict(model='svdpp', run_id=run.run_id, best_epoch=int(run.best_epoch))]).to_csv(output / 'training_curve_selection.csv', index=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--metrics', default='results/svdpp/metrics_by_run.csv')
    parser.add_argument('--output-dir', default='results/svdpp')
    args = parser.parse_args()
    path, output = Path(args.metrics), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(path)
    validate(table)
    for split in ('validation', 'test'):
        rmse_curve(table, output, split)
    training_curves(table, path, output)
    print(f'Created six SVD++ PNG/PDF figures from {path}')


if __name__ == '__main__':
    main()
