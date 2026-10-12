"""Перерисовать кривые одной модели после обучения."""
import argparse
from pathlib import Path
from src.plotting import plot_history, plot_loss_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="results/single/k20/history.csv")
    parser.add_argument("--output", default="results/single/k20/learning_curve.png")
    parser.add_argument("--loss-output", help="путь графика loss; по умолчанию train_loss.png рядом с --output")
    args = parser.parse_args()
    plot_history(args.history, args.output)
    plot_loss_history(args.history, args.loss_output or Path(args.output).with_name("train_loss.png"))
