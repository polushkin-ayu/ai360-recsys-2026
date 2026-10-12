"""Обучить одну SVD++ на полном train MovieLens 1M.

python train.py --k 20 --lr 0.003 --reg 1 --epochs 100 --device cuda
Сами градиентные обновления находятся в src/training.py:fit_one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.data import ROOT, sha256
from src.experiment import context, digest, environment, validate_search_values, write_json
from src.training import cpu_state, fit_one, save_torch
from src.plotting import plot_history, plot_loss_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=20, help="размер эмбеддингов p, q, y")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate Adam")
    parser.add_argument("--reg", type=float, default=1.0, help="lambda в MSE + lambda/N * L2")
    parser.add_argument("--epochs", type=int, default=100, help="общее число эпох, включая уже сохранённые")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-users", type=int, default=128)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", help="папка результатов; по умолчанию results/single/k<K>")
    args = parser.parse_args()
    for field in ("k", "epochs", "patience", "batch_users", "checkpoint_every"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} должен быть положительным")
    if args.seed < 0:
        parser.error("--seed должен быть неотрицательным")
    try:
        validate_search_values([args.lr], [args.reg])
    except ValueError as error:
        parser.error(str(error))

    # context() проверяет split и загружает только train и validation.
    # Оценки test не используются для выбора лучшей эпохи.
    ctx = context("configs/svdpp-ml1m.json", args.device)
    cfg = dict(ctx["cfg"], max_epochs=args.epochs, patience=args.patience,
               batch_users=args.batch_users, checkpoint_every=args.checkpoint_every)
    output = ROOT / (args.output or f"results/single/k{args.k}")
    output.mkdir(parents=True, exist_ok=True)

    # Среда может смениться после переподключения Colab. Идентичность опыта
    # определяется данными, кодом и настройками, а сведения о среде сохраняются
    # отдельно. CPU/CUDA дают одинаковую формулу, но не битовые копии результатов.
    run_config = dict(k=args.k, lr=args.lr, regularization=args.reg, seed=args.seed,
                      patience=args.patience, batch_users=args.batch_users,
                      init_base_std=cfg["init_base_std"], dtype="float32",
                      split_id=ctx["meta"]["split"]["id"],
                      data_sha256=ctx["meta"]["artifact_sha256"],
                      source_sha256={name: sha256(ROOT / name) for name in
                                     ("src/model.py", "src/training.py", "src/data.py", "src/metrics.py", "src/plotting.py", "train.py")})
    signature = digest(run_config)
    run_file = output / "run_config.json"
    if run_file.exists():
        existing = json.loads(run_file.read_text(encoding="utf-8"))
        if existing["signature"] != signature:
            raise ValueError("В этой папке другой опыт. Укажите новую папку через --output.")
    write_json(run_file, dict(signature=signature, settings=run_config))
    print(f"device={ctx['device']}; train={len(ctx['train'])}; validation={len(ctx['validation'])}", flush=True)
    print(f"Результаты: {output}", flush=True)

    model, metrics, history = fit_one(
        ctx["train"], ctx["validation"], ctx["histories"],
        ctx["meta"]["n_users"], ctx["meta"]["n_items"], cfg, ctx["device"],
        args.k, args.lr, args.reg, args.seed,
        resume_path=output / "last.pt", signature=signature,
    )

    # model уже восстановлена на лучшую validation-эпоху.
    save_torch(output / "model.pt", dict(
        state=cpu_state(model), k=args.k, n_users=ctx["meta"]["n_users"],
        n_items=ctx["meta"]["n_items"], init_base_std=cfg["init_base_std"],
        split_id=ctx["meta"]["split"]["id"], histories_checksum=ctx["histories"].checksum,
        metrics=metrics,
    ))
    pd.DataFrame(history).to_csv(output / "history.csv", index=False, lineterminator="\n")
    write_json(output / "metrics.json", dict(
        **metrics, test_used=False, environment=environment(ctx["device"]),
        objective=ctx["protocol"]["objective"],
    ))
    plot_history(output / "history.csv", output / "learning_curve.png", metrics["best_epoch"])
    plot_loss_history(output / "history.csv", output / "train_loss.png", metrics["best_epoch"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
