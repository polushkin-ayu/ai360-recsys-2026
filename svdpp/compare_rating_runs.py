"""Compare the finished rating-batch study, old fast study and recorded FM."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import ROOT

def main():
    output=ROOT/"results/svdpp-ml1m-ratings-active/seed11"
    if not (output/"status.json").exists():
        output=ROOT/"results/published-ratings/seed11"
    state=json.loads((output/"status.json").read_text(encoding="utf-8"))
    if not state["completed"] or not state["test_evaluated"]:
        raise ValueError("Wait for the entire frozen study, including test evaluation")
    current=pd.read_csv(output/"metrics_by_run.csv").sort_values("k")
    old_path=ROOT/"results/svdpp-ml1m-fast-completed/seed11/metrics_summary.csv"
    if not old_path.exists():old_path=ROOT/"results/published/seed11/metrics_summary.csv"
    old=pd.read_csv(old_path)
    fm=pd.read_csv(ROOT/"results/fm-reference/aggregate.csv")
    table=current.merge(old[["k","validation_rmse_mean","test_rmse_mean"]].rename(
          columns={"validation_rmse_mean":"previous_fast_validation","test_rmse_mean":"previous_fast_test"}),on="k")
    table=table.merge(fm[["k","val_rmse_mean","val_rmse_std"]].rename(
          columns={"val_rmse_mean":"fm_validation_mean","val_rmse_std":"fm_validation_sd"}),on="k",how="left")
    table.to_csv(output/"comparison_with_fm_and_fast.csv",index=False,lineterminator="\n")
    fig,ax=plt.subplots(figsize=(10,5))
    ax.errorbar(fm.k,fm.val_rmse_mean,yerr=fm.val_rmse_std,marker="o",capsize=3,label="FM: fixed settings, mean +/- SD of 3 seeds")
    ax.plot(current.k,current.validation_rmse,"s-",label="SVD++: 1024-rating batches, seed 11")
    ax.plot(old.k,old.validation_rmse_mean,":",alpha=.8,label="Previous fast SVD++: 512-user batches, seed 11")
    ax.set(xscale="log",xlabel="k (latent dimensions)",ylabel="Validation RMSE",title="MovieLens 1M: observed results under different training protocols")
    ax.set_xticks(current.k,labels=[str(k) for k in current.k],rotation=45)
    ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
    for extension in ["png","pdf"]:fig.savefig(output/f"validation_comparison.{extension}",dpi=180)
    plt.close(fig)
    best=current.loc[current.validation_rmse.idxmin()];previous=old.loc[old.validation_rmse_mean.idxmin()]
    caps=current.loc[current.hit_epoch_cap,"k"].tolist()
    report=f"""# SVD++ с 684 обновлениями за эпоху: результаты

Обучены все {len(current)} значений активной сетки k на полном MovieLens 1M, seed 11.
Активные k: {current.k.tolist()}.
k=6,7,8,9,15 отложены по просьбе пользователя и в эту серию не входят.
Батч — 1024 оценки, 684 отдельных обновления Adam за эпоху.
Максимум 25 эпох, patience=10, validation проверяется после каждой эпохи.
Каждый k обучается с нуля на train; после выбора k дополнительного
переобучения нет. Итоговая модель — лучшие validation-веса этого запуска.
Настройки фиксированы для всех k: lr=0.001, reg_bias=0.01,
reg_factors=10, init_std=0.01. Mean фиксирована по train.

Лучшая по validation модель: k={int(best.k)}.
Validation RMSE: {best.validation_rmse:.8f}.
Test RMSE этой модели: {best.test_rmse:.8f}.
Train RMSE: {best.train_rmse:.8f}.
Лучшая эпоха: {int(best.best_epoch)}; выполнено эпох: {int(best.epochs_run)}.
Всего обновлений Adam для этой модели: {int(best.optimizer_updates)}.
Сумма времени завершённых эпох активной сетки: {state['total_fit_seconds']/60:.2f} мин.
Включает обновления, train/validation scoring; исключает подготовку,
построение графиков, запись checkpoint, простой между продолжениями
и отброшенные ранние запуски с другим лимитом эпох.

Предыдущий быстрый протокол выбирал k={int(previous.k)} из другой, более широкой сетки,
validation RMSE={previous.validation_rmse_mean:.8f}.
В предоставленном эксперименте FM минимум среднего validation RMSE
при k=100: {fm.loc[fm.k==100,'val_rmse_mean'].iloc[0]:.8f}.

В новом опыте одновременно согласованы число обновлений, батч по оценкам,
learning_rate, начальная дисперсия факторов и отдельные коэффициенты
регуляризации. Поэтому изменение кривой нельзя приписать исключительно
числу обновлений; это сравнение двух полных протоколов обучения.
Один seed SVD++ сравнивается с тремя seeds FM, поэтому разброс SVD++ неизвестен.
Прежний лучший k=6 сейчас отложен; новый минимум относится только к 13
проверенным k и не устанавливает оптимум по исходной сетке из 18 значений.

До лимита 25 эпох дошли k={caps}. Достижение лимита не доказывает сходимость;
оптимум k относится к указанным настройкам, split, seed и бюджету.
Test вычислен после frozen.json и не использовался для выбора эпохи или k.
Поскольку результаты прошлых опытов на этом test уже известны,
это повторное сравнение на существующем test, а не новый независимый holdout.

Сводная таблица: comparison_with_fm_and_fast.csv.
Сравнение validation-кривых: validation_comparison.png/pdf.
Validation/test RMSE по k: rmse_vs_k.png/pdf, rmse_vs_k_log.png.
Train objective по эпохам: final_loss_curves.png и runs/k<K>/learning_and_loss.png.
Веса и состояние Adam: runs/k<K>/model.pt и last.pt (хранятся локально).
"""
    (output/"REPORT_RU.md").write_text(report,encoding="utf-8",newline="\n")
    print(report)

if __name__=="__main__":main()
