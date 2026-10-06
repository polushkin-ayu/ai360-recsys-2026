# AI360 2026 — взаимодействия признаков в рекомендациях

Распределение ответственности и проверяемые утверждения: [координация работы](docs/coordination.md).

FM — основной проект группы: изучаем факторизованную полиномиальную регрессию и проверяем FM второго порядка. Общую справку по DL проходят все. DCN — необязательное параллельное направление на 1–2 человека; от него можно отказаться без ущерба основному проекту.

**Вопрос:** улучшают ли факторизованные взаимодействия user/item предсказание рейтинга относительно bias-модели на небольшом срезе MovieLens?

Статус: подготовлены данные и проверки К5–К6 Team 1. Дополнительно реализована и обучена каноническая SVD++: отдельный эксперимент RMSE(k), 15 основных запусков. [Отчёт и команды](docs/svdpp_report.md).

## Начать работу

1. Прочитать [общий план, цели и требуемые результаты](docs/work_plan.md), затем [протокол исследования](docs/research_plan.md).
2. Выбрать задачу в Issues и согласовать интерфейсы с другими парами.
3. Следовать [правилам совместной работы](CONTRIBUTING.md).
4. Записывать запуски в [журнал экспериментов](docs/experiment_log.md).

## Материалы

- **Основная статья:** Freudenthaler, Schmidt-Thieme, Rendle. Factorization Machines — Factorized Polynomial Regression Models. [PDF авторов](https://www.ismll.uni-hildesheim.de/pub/pdfs/FreudenthalerRendle_FactorizedPolynomialRegression.pdf).
- **Введение:** Steffen Rendle. Factorization Machines. ICDM 2010. [DOI](https://doi.org/10.1109/ICDM.2010.127), [PDF](https://jame-zhang.github.io/assets/algo/Factorization-Machines-Rendle2010.pdf).
- **Дополнительная статья, DCN:** Deep & Cross Network for Ad Click Predictions (2017), [v1](https://arxiv.org/abs/1708.05123v1). Изучаем исходную архитектуру; на MovieLens адаптируем её к регрессии.

## Основная работа и расписание

FM: проверка реализации → небольшой срез MovieLens 100K → сравнение bias/FM → анализ и постер. Обязательны FM второго порядка и baselines; высшие порядки и дополнительные признаки — расширения. FM по user/item ID эквивалентна MF с bias.

Распределяем задачи реализации, подготовки данных и экспериментов внутри одной команды. Если DCN продолжается, её участники используют общий протокол и сравнивают DNN/DCN; для рейтингов это адаптация к регрессии, а не воспроизведение CTR-эксперимента.

- [Весь проект: этапы, роли, цели и результаты](docs/work_plan.md).
- [Сегодня, 4 октября: 11–18, обед 14–15](docs/day2.md).
- 5 октября: самостоятельная работа по задачам из плана сегодняшнего дня.
- [Вторник, 6 октября: проверки, концепт и две репетиции](docs/tuesday.md).
- [7 октября: mid-review](docs/midreview/README.md), завершённый код не обязателен.
- 8–13 октября: обратная связь, полные эксперименты и постер; 14 октября — финальная защита.

## Структура

- `src/` — код подготовки данных, моделей, обучения и оценки.
- `notebooks/` — исследование данных и учебные демонстрации.
- `configs/` — воспроизводимые конфигурации запусков.
- `results/` — небольшие таблицы метрик и графики.
- `docs/` — план, журнал и материалы защиты.
- `data/` — локальные данные; в Git хранится только инструкция.

## Окружение и запуск

Окружение подготовки данных проверено и зафиксировано в `requirements-data.txt` и отчёте Team 1. Стартовый список остальных библиотек находится в `requirements.txt`; окружение обучения ещё предстоит проверить. Обучающего скрипта пока нет.

Создание отдельного окружения (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Для Windows PowerShell активация: `.venv\Scripts\Activate.ps1`.
Для FM можно использовать NumPy с явными градиентами или PyTorch/autograd. Для DNN/DCN планируем PyTorch; сборку устанавливаем согласно [официальной инструкции](https://pytorch.org/get-started/locally/).
Проверенные команды подготовки данных — ниже; команды обучения и оценки добавьте после реализации моделей.

### Данные и общая RMSE — Team 1

Подготовка и проверки К5–К6 реализованы. Из корня репозитория в Python 3.12:

```bash
python -m pip install -r requirements-data.txt
python -m src.prepare_data --config configs/team1-data.json --report results/team1-data-checks.json
python -m unittest discover -s tests -v
```

Точная процедура, форматы таблиц и интерфейс для моделей — в [data/README.md](data/README.md).
Фактическая проверка данных — [results/team1-data-checks.json](results/team1-data-checks.json).
Этот запуск готовит данные; результатов обучения bias/FM ещё нет.

## Отдельный эксперимент SVD++

Самостоятельная PyTorch-реализация: `src/svdpp.py`. CLI pilot/sweep/final-evaluate: `python -m src.run_svdpp`. Этот CLI обучает только SVD++; графики содержат только её результаты.

Конфигурация: `configs/svdpp-study.json`; окружение CPU: `requirements-svdpp.txt`; проверки: `python -m src.check_svdpp`; описание результатов и ограничения: [docs/svdpp_report.md](docs/svdpp_report.md).
