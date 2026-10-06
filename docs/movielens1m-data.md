# Общие данные MovieLens 1M — протокол команды

Актуальный датасет с 6 октября 2026: полный MovieLens 1M вместо прежнего MovieLens 100K.
Источник пользователя: https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset .
Официальный источник: https://grouplens.org/datasets/movielens/1m/ .
Файл ratings.dat в обоих скачанных архивах совпал побайтно; его SHA-256:
`506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f`.

## Единый split

В источнике 1 000 209 оценок, 6 040 пользователей, 3 706 фильмов с оценками.
Полный каталог movies.dat может содержать фильмы без оценок; это не размер train-словаря.
Сохраняем все оценки. Для user/item-моделей нужны ratings.dat и общие train-only mappings.
Случайное разбиение 70/15/15 перенесено из research_plan.md, seed=42.
Это не хронологическое разбиение и не эксперимент холодного старта.

1. row_id = номер строки исходного ratings.dat с 0, до перемешивания.
2. Перемешивание numpy Generator(PCG64(42)) над строками в исходном порядке.
3. floor(N*0.70) строк идут в train, floor(N*0.15) в validation, остаток в test.
4. Выходные строки сортируются по исходному row_id.
5. Словари ID: возрастающие уникальные исходные ID только train -> индексы с 0.
6. Неизвестные ID кодируются как -1. Они остаются в *_all.csv, но исключаются
   из validation.csv/test.csv для основной RMSE. Использовать -1 как индекс запрещено.
7. Все CSV пишутся в UTF-8 с LF, чтобы hashes и split ID не зависели от Windows/Linux.

| Часть | До фильтра | Исключено unknown item | Для основной метрики |
|---|---:|---:|---:|
| train | 700146 | 0 | 700146 |
| validation | 150031 | 29 | 150002 |
| test | 150032 | 42 | 149990 |

Неизвестных пользователей в validation/test нет. Coverage: 99.9807% / 99.9720%.
Нет пересечений row_id или пар user/item; все исходные строки покрыты.

Ожидаемый split ID:
`44980861026c554d3b7d26ebd5bbf763ff0104d5629f40cc200ba2db38887663`.

## Получение данных каждым участником

Обновить main и выполнить из корня репозитория в Python-окружении:

```powershell
python -m pip install -r requirements-data.txt
python -m src.prepare_movielens1m --config configs/movielens1m-data.json
python -m src.prepare_movielens1m --check-only
```

Первая команда скачивает pinned Kaggle ZIP; при недоступности пытается скачать
официальный архив с идентичными ratings. Контрольные суммы обоих ZIP зафиксированы.
Повторный запуск переиспользует проверенный локальный архив; выходные таблицы
детерминированно пересоздаются. Отчёт по умолчанию: data/movielens1m-checks.json.
--check-only заново читает исходник и проверяет сохранённые таблицы/seed membership,
train-only mappings, coverage, пересечения, ID и checksums без изменения CSV.

Если Python сообщает об ошибке SSL, скачать официальный ZIP через PowerShell:

```powershell
New-Item -ItemType Directory -Path data/raw/movielens1m -Force
Invoke-WebRequest -Uri 'https://files.grouplens.org/datasets/movielens/ml-1m.zip' -OutFile 'data/raw/movielens1m/movielens1m.zip'
python -m src.prepare_movielens1m
```

Проверка сертификата не отключается. Можно также положить ZIP с Kaggle по тому же
пути, не распаковывая его вручную. Изменившийся архив/checksum не принимается.

## Подключение к моделям

```python
from src.data import MODEL_COLUMNS, load_prepared

data = load_prepared("data/processed/movielens1m")
train = data.train[MODEL_COLUMNS]
validation = data.validation[MODEL_COLUMNS]
test = data.test[MODEL_COLUMNS]
print(data.n_users, data.n_items, data.metadata["split"]["id"])
```

В конфигурации любой модели задать `data_dir: "data/processed/movielens1m"`.
Не делать свой split и не создавать новые mappings внутри обучения.
Истории SVD++ и любые агрегаты строить только по train. Настройки и лучшую эпоху
выбирать по validation; test не использовать для подбора. Для FM сохраняется
совместимость с fit_model/predict_frame и общей rmse_by_row_id.

Формат сохранённых файлов совпадает с Team 1: train.csv, validation/test.csv,
validation/test_all.csv, *_row_ids.csv, mappings, subset.csv, metadata.json.
Названия полей: row_id,user_idx,item_idx,rating для моделей; дополнительные
исходные ID/timestamp/known flags остаются в полных таблицах.

## Что в Git и что локально

В main находятся код, конфигурация, инструкция, тесты и небольшие отчёты
results/movielens1m-data/. Сам ZIP и таблицы остаются в data/, исключённом из Git.
Все участники получают одинаковые CSV из одной команды, не пересылая миллион строк.
Повторная подготовка из официального ZIP дала те же 11 артефактов и split ID,
что и подготовка из Kaggle ZIP. Тесты: 4 новых + 20 прежних Team 1.
Фактическое окружение и доказательства: checks.json и reproducibility.json.

Старые результаты на 15k/100k остаются историческими. Их нельзя объединять с
результатами 1M в одно сравнение качества. Новые запуски моделей должны записывать
этот split ID и те же оценочные строки. Этот этап только готовит данные, не обучает модели.
