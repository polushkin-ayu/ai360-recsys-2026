# Журнал экспериментов

Ниже записаны подготовка данных и отдельные эксперименты участников проекта. Не заполняйте метрики ожидаемыми значениями.

Для каждого запуска скопируйте блок:

## Run ID — дата

- Автор:
- Гипотеза:
- Commit и наличие незакоммиченных изменений:
- Конфигурация и команда запуска:
- Версия данных / checksum / способ подвыборки:
- Split ID и seed разбиения:
- Seed обучения:
- Python, версии библиотек, CPU/GPU:
- Метрики train/validation (test — после выбора модели):
- Число параметров с embeddings / без embeddings:
- Время обучения и что включено в измерение:
- Пути к таблице метрик и графикам:
- Наблюдение:
- Интерпретация и ограничения:
- Следующий шаг:

## team1-data-2026-10-04 — подготовка данных, 4 октября 2026

- Автор: техническая подготовка Team 1; ответственного и проверяющего в Issue назначает команда.
- Гипотеза: не проверялась; это получение данных и проверки протокола К5–К6, без обучения.
- Commit: базовый `ec255d2e9d7f825eeaa35582d99c06d3b5f875b0`, изменения незакоммичены.
  Базовый hash сам по себе этот запуск не воспроизводит: нужен patch и конфигурация.
  SHA-256 исполняемых исходников сохранены в `results/team1-data-checks.json`.
- Команда: `python -m src.prepare_data --config configs/team1-data.json --report results/team1-data-checks.json`.
- Источник: официальный MovieLens 100K; checksum ZIP и u.data — в конфигурации и отчёте.
- Срез: PCG64 seed 42, первый префикс случайно перемешанных users с не менее 15 000 ratings;
  140 пользователей, 15 103 оценки, все оценки выбранных пользователей.
- Split: seed 42, случайные строки, 70/15/15 с округлением вниз train/validation;
  10 572 / 2 265 / 2 266. Повторных пар в источнике нет.
- Известные ID: validation 2 209/2 265, test 2 220/2 266; unknown item rows 56/46,
  unknown user rows 0/0. Coverage 97.5276% / 97.9700%.
- Python 3.12.14, NumPy 2.3.5, Pandas 2.2.3, Linux, CPU;
  установленное окружение — `results/team1-environment.txt`.
- Результат: `results/team1-data-checks.json`; значения К5–К6 получены исполняемым кодом.
- Метрики моделей, параметры и время обучения: неприменимы; модели не запускались.
- Ограничения: сверка фактических bias/FM prediction IDs и проверка коллегой ещё впереди;
  интерфейс описан в `data/README.md`, назначение исполнителей/согласование не имитировались.
- Следующий шаг: передать интерфейс командам 2/3; после их predict вызвать `require_same_row_ids`
  и `rmse_by_row_id`, попросить коллегу воспроизвести инструкцию в новой копии.

## fm-k-sweep-20261006 — FM pilot, 6 October 2026

- Author: uliana roshchina; branch `experiment/fm-k-sweep-20261006`.
- Question: how user/item FM validation RMSE changes with k at fixed regularization.
- Executable commit: `735f750f3c4eba07da8ee76a7df6730369d5e7d9`; clean working tree at run start.
- Config: `configs/fm-k-sweep.json`; k=8/16/32/64, seeds=42/43/44, 12 runs.
- Command: `python -m src.run_fm_sweep --config configs/fm-k-sweep.json`.
- Same Team 1 MovieLens subset, train-only mappings and split seed 42; 10,572 train,
  2,209 known-ID validation, 2,220 known-ID test. Test not evaluated.
- Source/data/config/code checksums, split ID, canonical row hashes, CPU and library
  versions: `results/fm-k-sweep-20261006/manifest.json`.
- Settings: Adam lr=0.01, batch=1024, max epochs=200, patience=20;
  reg_bias=0.01, reg_factors=0.05, init_std=0.01, one deterministic CPU thread.
- Mean validation RMSE: k8=1.031714, k16=1.021908, k32=1.021676, k64=1.025772.
- Observation: k32 has the smallest mean, but improvement over k16 is much smaller
  than seed variability. Increasing k does not consistently improve validation.
- Results: `results/fm-k-sweep-20261006/summary.csv`, `aggregate.csv`, histories,
  plots, selection and README. Weights/predictions stay local under `data/`.
- Checks: 48 repository tests, 1 skip and 2 expected failures; all checkpoints
  reloaded with identical validation predictions; finite histories and aligned IDs.
- Limitations: no test evaluation or bias comparison, one split, default fixed
  regularization; proposed grid still requires team confirmation and peer review.
- Next: combine with colleagues' comparable tables, review protocol/settings,
  tune regularization within an agreed budget, then freeze choices for test.

## fm-k50-100-150-200-20261006 — requested FM grid, 6 October 2026

- Author: uliana roshchina; k=50/100/150/200, seeds=42/43/44, 12 runs.
- Executable commit: `4ee07212010fe77c4fd6903338667743e50c9fa1`; clean working tree before training.
- Config: `configs/fm-k50-100-150-200.json`.
- Command: `python -m src.run_fm_sweep --config configs/fm-k50-100-150-200.json`.
- Shared Team 1 MovieLens data/split and same hyperparameters as the first sweep.
- Test not evaluated; no bias comparison performed.
- Results and observation details: `results/fm-k50-100-150-200-20261006/README.md`.
- Exact environment, provenance, split/row hashes: manifest.json in that folder.
- All checkpoint reloads matched predictions; histories finite and rows aligned.
- Observation: k50 has the lowest mean within this grid; larger k does not
  consistently improve validation. Three seeds at fixed regularization.
- Next: team review, combine comparable model tables and freeze settings before test.

## svdpp-only-2026-10-06 — отдельный эксперимент SVD++

- Ветка: `feat/svdpp-only-study`; исходный commit `37a9e2aa1d9f8792da3f3654275deae3ccc7de91`; commit измеряемого кода `2f880f746f514f33ddcdbeb91064c5c24af736b7`.
- Объём: только canonical SVD++, sqrt normalization, fixed train mean; комплексное сравнение моделей проводится отдельно участниками проекта.
- Данные: существующий MovieLens pipeline, 140 users, 1 329 train items; train/validation/test известных ID 10 572/2 209/2 220.
- Pilot: k=20, seed=0, три попытки; выбран Adam lr=0.01, lambda_f=0.05 при lambda_b=0.01 по validation.
- Sweep: 15 запусков SVD++, k=[10,20,50,100,200], seeds=[42,43,44], batch=1024, максимум 200 эпох, patience=20, min_delta=0, init_std=0.01, clipping=False.
- Лучший k по среднему validation RMSE: 10; validation 1.031823 ± 0.006952, test 1.051429 ± 0.009956 (sample std по трём seed).
- Test оценён отдельной командой после frozen manifest. Test уже был просмотрен в предыдущем запуске SVD++; повтор выполнен без изменения численного протокола и нового подбора. Все метрики SVD++ совпали точно.
- Проверки: 44 теста, 0 failures/errors/skipped; finite differences max error 1.5711e-10; независимый forward max error 0; К5–К6 прошли; CSV восстанавливают RMSE точно в float32.
- Источники, конфигурация, checkpoint и CSV защищены SHA-256; до/после test protocol/checkpoint/validation hashes неизменны. Повтор plot из CSV даёт идентичные PNG.
- Окружение: Python 3.12.14, torch 2.14.1+cpu, NumPy 2.3.5, pandas 2.2.3, CPU/float32/один поток/deterministic. Сумма fit основных запусков 13.5593 с, включая epoch evaluation/early stopping/best restore, без I/O.
- Файлы: `results/svdpp/`, `docs/svdpp_report.md`; команды — в отчёте. Ограничения: малый срез, один split, известные ID, разброс только по инициализации.
- Удалённый push не выполнен; подготовлены отдельная ветка, patch, архив и bundle.

## fm-full100k-k50-100-150-20261006 — full source retraining

- Author: uliana roshchina; requested full MovieLens 100K, k=50/100/150, seeds=42/43/44.
- Executable config/code commit: `16595d729e5797803cc0bf0ec45f11e5393e7c65`; clean tree at launch.
- Data: all 100,000 source ratings before split; 70,000 train, 14971
  known-ID validation and 14977 known-ID test after filtering; split seed 42.
- Configs: `configs/full100k-data.json`, `configs/fm-full100k-k50-100-150.json`.
- Command: `python -m src.run_fm_sweep --config configs/fm-full100k-k50-100-150.json`.
- 9 measured runs; test not evaluated. Fixed original FM training settings.
- Results: `results/fm-full100k-k50-100-150-20261006/` (summary, aggregate,
  histories, plots, manifest, data checks, selection and README).
- Data K5/K6 passed; checkpoint reloads, row alignment, finite histories and
  best-validation restoration checked in all runs.
- Lowest mean validation score in this grid at k=50; no final test or bias claim.
- Previous subset series unchanged. Full100k results require matching full-data
  runs from other models; do not directly combine with their 15k-subset scores.
- Next: team review, comparable baseline runs, frozen settings, final test.
