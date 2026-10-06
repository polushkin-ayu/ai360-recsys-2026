# MovieLens 100K — общий интерфейс Team 1

**Для новых запусков используем MovieLens 1M:** [единая инструкция](../docs/movielens1m-data.md), `python -m src.prepare_movielens1m`. Описание MovieLens 100K ниже сохранено для исторических экспериментов.


Предсказываем рейтинг 1–5 по user/item ID. Протокол: [research_plan.md](../docs/research_plan.md),
ТЗ: [team1-data.md](../docs/teams/team1-data.md). Жанры и обучение моделей не входят в подготовку.

## Проверенная процедура

Выполнять из корня репозитория. Проверено в отдельном окружении Python 3.12.14,
NumPy 2.3.5, Pandas 2.2.3, Linux, CPU. Требуется доступ к официальному серверу GroupLens.
`requirements-data.txt` фиксирует только две зависимости подготовки; общий список моделей не меняется.
Фактически установленное окружение: `results/team1-environment.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
python -m src.prepare_data --config configs/team1-data.json --report results/team1-data-checks.json
python -m src.check_data --config configs/team1-data.json --report data/team1-recheck.json
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m pip check
```

Первая команда pipeline скачивает источник при отсутствии локального ZIP, проверяет его,
сохраняет подготовленные таблицы, **заново читает их** и выполняет К5–К6.
`check_data` проверяет уже сохранённые данные без скачивания. При ошибке команда завершается
с ненулевым кодом; успешный отчёт появляется только после всех проверок.
Все пути из конфигурации разрешаются относительно корня репозитория.

Для Windows используйте `python -m venv .venv`, затем в PowerShell
`.venv\Scripts\Activate.ps1`; команды `python -m ...` те же. Это инструкция для Windows,
сам запуск на Windows здесь не проверялся.

Повторный запуск pipeline проверяет checksum сохранённого ZIP и пересоздаёт таблицы.
Для полностью чистого воспроизведения безопаснее использовать новую копию репозитория:
данные в неё не копируются. Повреждённый ZIP не принимается; удалите именно этот локальный
файл и повторите скачивание. Выходные файлы не предназначены для ручного редактирования:
`load_prepared` проверяет их SHA-256 по metadata.

## Источник и устойчивые ID

Официальная страница: https://grouplens.org/datasets/movielens/100k/ .
URL архива: https://files.grouplens.org/datasets/movielens/ml-100k.zip .

- Размер ZIP: 4 924 029 байт.
- SHA-256 ZIP: `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`.
- SHA-256 `u.data`: `06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490`.

Эти checksum вычислены при получении официального архива 4 октября 2026 года и закреплены
в конфигурации как проверка неизменности байтов; это не заявленные издателем цифровые подписи.
ZIP проверяется по размеру, SHA-256 и CRC. Извлекаются только `u.data`, `u.info` и `README`.
Checksum сохраняется в `data/raw/CHECKSUMS.json` и metadata.

`row_id` — номер исходной строки `u.data`, начиная с **0**, до любых выборок и разбиений.
ID сохраняется во всех таблицах и не перенумеровывается. Исходный порядок зафиксирован checksum.

Полный источник: 100 000 оценок, 943 пользователя, 1 682 фильма; пропусков и повторных
пар user/item нет, диапазон рейтингов 1–5. Частоты рейтингов 1/2/3/4/5:
6 110 / 11 370 / 27 145 / 34 174 / 21 201.
Оценок на пользователя: min 20, median 65, mean 106.0445, max 737;
на фильм: min 1, median 27, mean 59.4530, max 583.
Полные краткие статистики источника и среза сохранены в отчёте; большого EDA нет.

## Правила среза и split

Все параметры: `configs/team1-data.json`.
В документах и Issues #1–#3 на момент работы нет согласованных конкретных seed/размеров;
использованы стартовые значения seed 42 и target 15 000. Размер остаётся стартовым до пилота,
скорость обучения здесь не измерялась.

1. Отсортировать уникальные user ID; перемешать `numpy.random.Generator(PCG64(42))`.
2. Взять первый префикс пользователей, суммарное число оценок которого достигло 15 000.
3. Включить **все** их оценки. Используются только ID и числа наблюдений; рейтинги и модели
   не участвуют в выборе. Получено 140 пользователей, 15 103 оценки, 1 414 фильмов.
4. Отдельным генератором `PCG64(42)` случайно перемешать строки среза, изначально
   отсортированные по `row_id`. Train: `floor(N*0.70)`, validation: `floor(N*0.15)`,
   test: остаток. Получено **10 572 / 2 265 / 2 266** наблюдений.
5. Если пары повторяются, единица перемешивания — целая пара user/item, в порядке её
   первого появления. Доли применяются к числу групп; доли строк тогда приблизительные.
   Данные не удаляются. Эта ветвь проверена искусственным техническим тестом.
6. Выходные таблицы сортируются по исходному `row_id`. Mappings — возрастающие уникальные
   исходные ID **только train** → непрерывные индексы с 0. Пространства user и item отдельные.

Train содержит `n_users=140`, `n_items=1329`. Для общего FM можно использовать признаки
`user_idx` и `n_users + item_idx`, всего `n_users+n_items`. Плотная one-hot матрица не создаётся.
Идентификатор split — SHA-256 отсортированного JSON со checksum всех подготовленных артефактов;
он хранится в `metadata["split"]["id"]`.

## Неизвестные ID и основная оценка

Validation/test кодируются относительно train. Неизвестный индекс — **-1**,
`known_user`/`known_item` — false. Строка исключается из основной оценки, если неизвестен
хотя бы один ID. `*_all.csv` сохраняет все строки, флаги и исходные ID; по ним легко получить
исключённые `row_id`. В `validation.csv`/`test.csv` оставлены только известные ID.
Нельзя передавать индекс -1 как реальный последний элемент массива параметров модели.

| Часть | До фильтра | Строки с unknown user | Строки с unknown item | Исключено | После | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| validation | 2 265 | 0 | 56 | 56 | 2 209 | 97.5276% |
| test | 2 266 | 0 | 46 | 46 | 2 220 | 97.9700% |

Неизвестных уникальных фильмов: 51 в validation, 45 в test; неизвестных пользователей 0.
Coverage = число оставшихся строк / исходное число строк этой части. Неизвестные user/item
считаются и по строкам, и как уникальные ID; сумма строк двух типов может включать пересечение.
Это случайное разбиение для оценки на известных ID; холодный старт и временной прогноз
этот эксперимент не оценивает.

## Локальные файлы

CSV: UTF-8, запятая, заголовок, без индекса Pandas. Исходные и кодированные ID, рейтинг,
timestamp — целые; known-флаги — CSV `True`/`False`, coverage — float в JSON.

| Путь | Содержимое |
|---|---|
| `data/raw/ml-100k.zip` | Проверенный официальный архив |
| `data/raw/ml-100k/u.data`, `README`, `u.info` | Ratings, исходная инструкция/условия и размеры |
| `data/raw/CHECKSUMS.json` | URL и SHA-256 |
| `data/processed/team1/subset.csv` | `row_id,user_id,item_id,rating,timestamp` |
| `data/processed/team1/selected_users.json` | Выбранные исходные users в порядке seeded-перестановки |
| `data/processed/team1/train.csv` | Train, исходные колонки + `user_idx,item_idx,known_user,known_item` |
| `data/processed/team1/validation_all.csv`, `test_all.csv` | Все строки частей с теми же колонками, включая неизвестные ID |
| `data/processed/team1/validation.csv`, `test.csv` | Только известные ID, те же колонки и порядок |
| `data/processed/team1/validation_row_ids.csv`, `test_row_ids.csv` | Единственная колонка `row_id`; точный порядок основной оценки |
| `data/processed/team1/user_mapping.csv` | `user_id,user_idx` |
| `data/processed/team1/item_mapping.csv` | `item_id,item_idx` |
| `data/processed/team1/metadata.json` | Параметры, размеры, split ID, coverage, checksum, статистики, окружение |
| `results/team1-data-checks.json` | Маленький отчёт К5–К6, без таблиц данных |

`data/*` уже исключено существующим `.gitignore`, кроме этого README.
Ни исходный датасет, ни подготовленные CSV/JSON из `data/` не добавлять в Git.
Официальные условия и рекомендуемая ссылка на датасет доступны в локальном README источника.

## Интерфейс для команд 2 и 3

```python
from src.data import MODEL_COLUMNS, load_prepared
from src.metrics import require_same_row_ids, rmse_by_row_id

data = load_prepared("data/processed/team1")
train = data.train[MODEL_COLUMNS]  # row_id, user_idx, item_idx, rating
validation = data.validation[MODEL_COLUMNS]
test = data.test[MODEL_COLUMNS]
print(data.n_users, data.n_items, len(train), len(validation), len(test))

# predict должен возвращать один рейтинг и исходный row_id на каждую строку.
def evaluate_validation(prediction, prediction_row_ids):
    return rmse_by_row_id(
        validation.rating.to_numpy(), prediction,
        validation.row_id.to_numpy(), prediction_row_ids,
    )

# После получения результатов двух моделей:
# require_same_row_ids(bias_prediction_row_ids, fm_prediction_row_ids)
```

Вывод загрузки: `140 1329 10572 2209 2220`.
Mappings доступны как `data.user_mapping`/`data.item_mapping` (dict исходный int ID → int index),
описание и параметры — `data.metadata`. `load_prepared` по умолчанию проверяет checksum
**всех** сохранённых артефактов. Изменённая таблица вызывает ошибку.
Предсказания возвращать в порядке входного `row_id`; не делать свои split/mappings/фильтры.
Основную оценку можно однозначно воспроизвести по `*_row_ids.csv` и их checksum в отчёте.
Проверка фактических прогнозов bias/FM и согласование с живыми командами остаются на этапе
интеграции: моделей в исходном репозитории нет, организационное согласование не выдаётся за выполненное.

## К5–К6

К5 проверяет SHA-256, связь каждой строки с источником, все оценки выбранных users,
отсутствие пересечений `row_id` и user/item пар, точное объединение split, mappings по train,
unknown-флаги/индексы, размеры/coverage и порядок сохранённых оценочных ID.
Проверка не использует `assert`, который Python может отключить через `-O`.

Общая RMSE находится в `src.metrics.rmse`; для моделей используйте `rmse_by_row_id`.
Входы — непустые одномерные вещественные числовые векторы одинаковой формы, конечные.
Нельзя передавать столбец `(N,1)` или надеяться на broadcasting. Результат — Python float.
Метрика не выполняет clipping и не переупорядочивает наблюдения.

К6 фактически получает `sqrt(2.5) = 1.5811388300841898` для `[1,3]` и `[2,1]`.
Отклоняются разные длины, 2D/scalar/empty, NaN/Inf, complex/nonnumeric,
переставленные/повторные/пропущенные/нецелые ID. `require_same_row_ids` проверяет
одинаковый полный порядок ID двух моделей. Отдельные технические тесты проверяют дубликаты
пар, неизвестные ID всех типов и повреждения сохранённых данных/metadata.

После подготовки `unittest` выполняет 20 тестов. До скачивания 5 тестов сохранённых
артефактов явно пропускаются: их запуск после pipeline обязателен для полной проверки.
