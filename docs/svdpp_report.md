# Каноническая SVD++ на PyTorch: отдельный эксперимент

Репозиторий: https://github.com/polushkin-ayu/ai360-recsys-2026 .
Исходный commit: `37a9e2aa1d9f8792da3f3654275deae3ccc7de91`.
Ветка: `feat/svdpp-only-study`. Commit измеряемого кода: `2f880f746f514f33ddcdbeb91064c5c24af736b7`.

Материалы содержат только SVD++: 3 pilot-попытки и 15 основных запусков. Обучением других моделей занимаются другие участники проекта; комплексное сравнение будет выполнено отдельно. Результаты, таблицы, графики и CLI этого эксперимента ограничены SVD++.

## Формула и реализация

$$\widehat r_{ui}=\mu+b_u+b_i+q_i^\top\left(p_u+\frac{1}{\sqrt{|N_u|}}\sum_{j\in N_u}y_j\right).$$

Для пустого $N_u$ неявный вклад равен нулю. $P,Q,Y$ — независимые обучаемые таблицы; $\mu$ — фиксированный buffer со средним train, вычисленным в float32. Использована каноническая нормировка истории из [Surprise.SVDpp](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp) и формулы (15) [Корена, 2008](https://www.cs.cornell.edu/courses/cs6241/2020sp/readings/Koren-2008-Factorization.pdf). В формуле (17) приложенного перевода статьи нормировка отсутствует; этот вариант не обучался. Рисунок 6 служит образцом зависимости RMSE от k; значения из статьи не используются.

`src/svdpp.py` реализует forward и построение истории самостоятельно на PyTorch. Готовая SVD++ библиотеки и матричное SVD не используются. $N_u$ — отсортированное множество уникальных train-фильмов; рейтинг не является весом. Train target входит в историю; held-out события туда не добавляются. История фиксирована при обучении и оценке. Item index 0 — настоящий фильм, без padding.

Полные суммы Y вычисляются через `embedding_bag` с сохранением autograd; старого кэша после optimizer step нет. Повторяющиеся пользователи корректно накапливают градиенты. Общие training, prediction, loss и checkpoint helpers переиспользованы; в общий цикл добавлены явные train MSE, regularized loss, контроль конечности и синхронизация CUDA для тайминга. Старые интерфейсы проекта сохранены.

Checkpoint содержит параметры, фиксированное среднее, историю, нормировку, размеры mappings, split ID и checksums. Загрузчик проверяет идентичность данных.

$$L(B)=\operatorname{MSE}(B)+\frac{\lambda_b}{n_{train}}(\|b_U\|^2+\|b_I\|^2)+\frac{\lambda_f}{n_{train}}(\|P\|_F^2+\|Q\|_F^2+\|Y\|_F^2).$$

Нормы берутся по полным таблицам, $n_{train}$ — число всех train-строк. $\mu$ не регуляризуется; дополнительного weight decay нет. Используется minibatch Adam: его траектория и конвенция регуляризации не обязаны повторять SGD из Surprise. `train_rmse=sqrt(train_mse)`; regularized loss сохраняется отдельно. Clipping выключен, прогнозы вне [1,5] допустимы.

## Данные и протокол

Использован существующий MovieLens 100K pipeline и `load_prepared(..., verify=True)`, без новых split/mappings. `subset_seed=42`, `split_seed=42`. Срез: 140 пользователей, 15 103 рейтинга, 1 414 фильмов. Train: 10 572 строки; train mappings: 140 пользователей и 1 329 фильмов. Validation/test до фильтра: 2 265/2 266, после фильтра известных ID: 2 209/2 220. Coverage: 97.5276%/97.9700%. Это оценка известных ID на случайном split.

140 историй содержат 10 572 уникальных membership; длины min/median/mean/max: 10/43.5/75.5143/350, пустых нет. SHA-256 истории и split сохранены в `history_stats.json` и fingerprint. Все 15 запусков используют одинаковые row_id и порядок; RMSE рассчитана `rmse_by_row_id`.

Pilot SVD++: k=20, seed=0. Три кандидата `(lr, lambda_f)`: `(0.003,0.05)`, `(0.01,0.05)`, `(0.003,0.5)`, при `lambda_b=0.01`. Validation RMSE: 1.043844, 1.038647, 1.040110. Выбраны `lr=0.01`, `lambda_f=0.05`; все попытки сохранены отдельно.

Основная сетка: k=[10,20,50,100,200], seeds=[42,43,44]. Adam, batch=1024, максимум 200 эпох, patience=20, min_delta=0, lambda_b=0.01, init_std=0.01, sqrt normalization, fixed train mean. Каждый из 15 запусков начинается с нуля. После pilot меняются только k и seed. Checkpoint выбирается по минимальной validation RMSE; k — по среднему validation RMSE трёх seed.

До отдельного `final-evaluate` сохранены frozen protocol, manifest и `validation_manifest.json`, где все test RMSE пусты. Test затем оценён для всей заранее заданной сетки, без подбора по test и без refit на train+validation. Загрузчик проверяет все таблицы, но обучение и выбор не используют test targets.

**История доступа к test:** test уже просматривался в предыдущем запуске SVD++. Эта редакция меняет состав публикуемых материалов, а не настройки обучения. Те же 15 запусков повторены с неизменным численным протоколом; их метрики совпали точно. Повтор не считается новым независимым экспериментом на ранее не просмотренном test. Нового подбора после просмотра test не было; это явно отражено в конфигурации и `scope_audit.json`.

## Результаты SVD++

Значения — mean ± sample std (ddof=1) по трём seed, один split.

| k | Train RMSE | Validation RMSE | Test RMSE | Fit, с |
|---:|---:|---:|---:|---:|
| 10 | 0.854199 ± 0.003993 | 1.031823 ± 0.006952 | 1.051429 ± 0.009956 | 0.574 ± 0.034 |
| 20 | 0.843126 ± 0.100953 | 1.034524 ± 0.001749 | 1.055731 ± 0.004446 | 0.539 ± 0.041 |
| 50 | 0.818500 ± 0.006934 | 1.032918 ± 0.002054 | 1.059234 ± 0.006620 | 0.686 ± 0.019 |
| 100 | 0.680293 ± 0.108292 | 1.036788 ± 0.002943 | 1.058417 ± 0.006827 | 0.896 ± 0.044 |
| 200 | 0.405397 ± 0.223957 | 1.045085 ± 0.002358 | 1.072692 ± 0.008434 | 1.826 ± 0.635 |

Лучший k по validation — **10**. Его validation RMSE: **1.031823 ± 0.006952**; test RMSE: **1.051429 ± 0.009956**. При увеличении k train RMSE снижается, а held-out ошибка в этой сетке не улучшается; k=200 даёт наибольшую среднюю validation ошибку. Это согласуется с переобучением на малом срезе, но один split не устанавливает причину и не позволяет обобщить вывод на весь MovieLens.

Лучшие эпохи у 14 запусков находятся между 2 и 4; один запуск выбирает эпоху 23. Для training curves выбран k=10 по validation и seed=44 с медианным validation RMSE среди трёх seed этого k; best epoch=3. Панели отдельно показывают train MSE, train MSE+L2 и validation RMSE.

Графики `rmse_vs_k_test`, `rmse_vs_k_validation`, `training_curves` сохранены в PNG и PDF; каждая панель содержит только результаты SVD++. Mean/std описывают разброс инициализации, не доверительный интервал и не устойчивость к выбору split. Немонотонность сохранена, сглаживания нет. На всех test-прогнозах 15 запусков суммарно 138 значений выходят за [1,5]; они не обрезались.

## Проверки и воспроизводимость

44 теста прошли, 0 ошибок/падений/пропусков; 14 тестов относятся к новой SVD++. Независимый прямой forward: max error 0. Конечные разности в float64 с epsilon=1e-6: max abs error 1.5711e-10 при допуске 2e-8. Проверены полная история, отсутствие held-out утечки, пустая история, повторные пользователи, SGD step, фиксированное среднее, seed/reset, checkpoint и восстановление лучшей эпохи. При нулевых факторах используются независимые математические выражения.

Toy learning: MSE 0.79137349 → 9.71075e-14 за 600 эпох, порог 1e-4, seed=17. Проверки данных К5–К6 прошли. CSV-прогнозы, прочитанные обратно как float32, точно воспроизводят все 30 validation/test RMSE. До/после test неизменны настройки, выбор k, checkpoint и validation checksums. Повтор pilot/sweep пропустил завершённые запуски. PNG, перестроенные из CSV в отдельный каталог, побайтово совпали.

Python 3.12.14, torch 2.14.1+cpu, NumPy 2.3.5, pandas 2.2.3, matplotlib 3.10.8; CPU, float32, один поток, deterministic algorithms. Все зависимости закреплены в `requirements-svdpp.txt`; подробное окружение — в fingerprint. CUDA-проверка не выполнялась: CUDA недоступна.

`fit_seconds` включает обучение, train/validation evaluation, early stopping и восстановление лучшего состояния; исключает загрузку данных и запись файлов. Сумма fit для основных запусков: 13.5593 секунды. Это измерение на текущем CPU, не универсальная оценка производительности.

Код закоммичен до pilot; dirty_worktree отражает создание файлов результатов. Source hashes и `code_snapshot.zip` фиксируют исполняемые исходники и точные байты конфигурации. `scope_audit.json` содержит результаты сверки.

## Команды

Создание окружения macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Фактически выполненные команды из корня репозитория:

```bash
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-svdpp.txt
python -m src.prepare_data --config configs/team1-data.json --report results/team1-data-checks.json
python -m src.check_data --config configs/team1-data.json --report results/svdpp/data_checks.json
python -m src.check_svdpp
python -m src.run_svdpp --stage pilot --config configs/svdpp-study.json
python -m src.run_svdpp --stage sweep --config configs/svdpp-study.json
python -m src.run_svdpp --stage final-evaluate --manifest results/svdpp/manifest.json
python -m src.plot_svdpp --metrics results/svdpp/metrics_by_run.csv --output-dir results/svdpp
```

Подготовленные данные уже существовали и проверены повторно; скачивать их заново не потребовалось. Plot CLI читает CSV, не обучает модель и не требует весов.

Для нового обучения из переданного архива сохраните опубликованные результаты. Создайте копию JSON с `output_dir=results/svdpp-local` и `checkpoint_dir=artifacts/svdpp-local`, затем передавайте её в pilot/sweep и `results/svdpp-local/manifest.json` в final-evaluate. Для plot используйте CSV и каталог новой серии. Перенос завершённых измерений в продолжение допускается только при совпадении fingerprint и наличия всех checkpoint/CSV; локальные веса в архив и Git не включены.

После начала test новые pilot и sweep для того же manifest закрыты; final-evaluate поддерживает продолжение. Незавершённый run повторяется с нуля; завершённый переиспользуется только при совпадающих hashes. Ошибки протокола вызывают явный отказ.

## Файлы и границы выводов

Реализация/CLI/проверки: `src/svdpp.py`, `src/run_svdpp.py`, `src/plot_svdpp.py`, `src/check_svdpp.py`, `tests/test_svdpp.py`; интеграция: `src/models.py`. Протокол: `configs/svdpp-study.json`, `requirements-svdpp.txt`, `docs/svdpp_task.md`. Результаты: `results/svdpp/` — 15 run JSON, 3 pilot JSON, CSV, истории, прогнозы, manifest, audits и шесть графиков. Документация: текущий отчёт, README и журнал экспериментов.

Все итоговые показатели принадлежат одному восстановленному лучшему checkpoint каждого run. Dataset, venv и веса не публикуются. Результаты ограничены малым срезом, одним случайным split, известными ID и тремя инициализациями; исследование холодного старта и временного прогноза не проводилось. Удалённый push не выполнялся.
