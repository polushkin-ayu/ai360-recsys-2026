# FM на общей выборке MovieLens 1M

9 запусков: k=50,100,150; seed=42,43,44. Все 1 000 209 исходных оценок
включены до общего разбиения 70/15/15, seed разбиения 42.
Обучение: 700 146 строк; validation известных ID: 150 002 строки.
Test не оценивался. Эти числа нельзя сравнивать с предыдущими сериями на 100K или малом срезе.

| k | Средний validation RMSE | Sample SD по трём seed |
|---|---|---|
| 50 | 0.913085 | 0.002013 |
| 100 | 0.956042 | 0.000745 |
| 150 | 1.005999 | 0.000344 |

Минимальное среднее в этой серии: k=50. Это сравнение при фиксированной
регуляризации, без подбора оптимальных настроек каждого k. SD описывает разброс
по инициализации, а не доверительный интервал по разбиениям.
Лучшая эпоха по validation: 1 во всех девяти запусках.
Ранняя лучшая эпоха указывает, что текущие настройки стоит обсудить с командой
до финальной оценки; увеличивать k само по себе недостаточно.

## Что передать для общего графика

- `summary.csv`: строка на каждый запуск, model/k/seed/val_rmse, split_id и hash строк validation.
- `aggregate.csv`: средние и SD для нашей линии FM.
- `manifest.json`: данные, настройки, версии библиотек, hash исходников и commit измеряемого кода.
- `*_history.csv`: истории эпох; PNG строятся из сохранённых таблиц без обучения.
- `selection.json`: выбор k только по validation.

Исполняемый commit: `b8c93b1ed025a5d7a4f7fd730c888402ec0948c8`.
Split ID: `44980861026c554d3b7d26ebd5bbf763ff0104d5629f40cc200ba2db38887663`.
Validation row hash: `21c29da556328c953b76824b5f8493116fe8b359ac3240add2221193cd0993dd`.

Веса и прогнозы по строкам находятся локально в
`data/fm-movielens1m-k50-100-150-20261006/`, Git их игнорирует.
Каждый checkpoint после записи загружен повторно: прогнозы validation совпали точно.
Истории конечны, восстановлена лучшая эпоха, RMSE проверен с выравниванием row_id.

## Воспроизведение

```bash
python -m src.prepare_movielens1m
python -m src.run_fm_sweep --config configs/fm-movielens1m-k50-100-150.json
python -m src.plot_fm_sweep --results results/fm-movielens1m-k50-100-150-20261006 --title "MovieLens 1M: FM at fixed regularization"
```

Для нового обучения укажите новые experiment_id/output_dir/local_dir в конфиге:
существующие результаты защищены от перезаписи. Последняя команда только рисует графики.

## Как объединить с другими моделями

Автор каждой модели сохраняет совместимый `summary.csv`, независимо от .py/.ipynb.
Перед объединением сравните split_id, validation_row_ids_sha256, clipping и определение
метрики. У всех должен быть один набор validation; test и validation не смешивают.
Результаты на старом датасете не входят в этот график. Seed обучения можно различать,
но единый список позволяет получить одинаковое число повторов.

```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Добавьте пути к таблицам коллег на той же выборке 1M.
paths = [Path('results/fm-movielens1m-k50-100-150-20261006/summary.csv')]
runs = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
assert runs['split_id'].notna().all() and runs['split_id'].nunique() == 1
assert runs['validation_row_ids_sha256'].notna().all()
assert runs['validation_row_ids_sha256'].nunique() == 1
assert runs['clipping'].eq(False).all()
assert runs['val_rmse'].notna().all()
assert not runs.duplicated(['model', 'k', 'seed']).any()
stats = runs.groupby(['model', 'k'])['val_rmse'].agg(['mean', 'std', 'count']).reset_index()
fig, ax = plt.subplots()
for name, rows in stats.groupby('model'):
    rows = rows.sort_values('k')
    ax.errorbar(rows['k'], rows['mean'], yerr=rows['std'].fillna(0),
                marker='o', capsize=3, label=name)
ax.set(xlabel='k', ylabel='Validation RMSE', title='MovieLens 1M')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('comparison_1m.png', dpi=180)
```

Объединение таблиц и рисунок не запускают модели. Если команда решит строить финальный
график по test, сначала фиксирует настройки; сохранённые веса позволяют получить
test-прогнозы без нового обучения. Bias без k изображают отдельной горизонтальной линией.
