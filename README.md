# AI360 2026 — Factorization Machines для рекомендаций

Проект исследует, улучшают ли факторизованные взаимодействия признаков предсказание
рейтинга относительно bias-модели. Основной эксперимент проводится на
воспроизводимом срезе MovieLens 100K с общей подготовкой данных, split и метрикой
RMSE.

Основная статья: Christoph Freudenthaler, Lars Schmidt-Thieme, Steffen Rendle,
[*Factorization Machines — Factorized Polynomial Regression Models*](https://www.ismll.uni-hildesheim.de/pub/pdfs/FreudenthalerRendle_FactorizedPolynomialRegression.pdf).

## Что реализовано

В [`src/models.py`](src/models.py) находятся три PyTorch-модели:

| Класс | Вход | Формула и назначение |
|---|---|---|
| `BiasModel` | `user_idx`, `item_idx` | `mu + b_user + b_item` — основной baseline |
| `FactorizationMachine` | `user_idx`, `item_idx` | Bias плюс `dot(p_user, q_item)` — FM второго порядка для двух one-hot полей, эквивалентная biased MF |
| `SparseFactorizationMachine` | произвольный `x` | Общая FM второго порядка для dense или sparse вещественных признаков |

Общая FM вычисляет

```text
y(x) = w0 + sum_i(w_i * x_i)
            + sum_{i<j}(dot(v_i, v_j) * x_i * x_j).
```

Дополнительно реализованы:

- детерминированная подготовка данных и train-only mappings;
- обучение `BiasModel` и user/item `FactorizationMachine`;
- выбор лучшей эпохи по validation RMSE и early stopping;
- предсказание с сохранением порядка `row_id`;
- общая RMSE с проверкой соответствия строк;
- сохранение и загрузка checkpoint;
- dense- и CSR-forward общей FM;
- тесты формул, градиентов, данных и checkpoint.

## Структура проекта

```text
configs/        воспроизводимые конфигурации
data/           инструкция и локальные raw/processed данные
docs/           план, аудит, журнал и материалы исследования
notebooks/      исследовательские ноутбуки
results/        небольшие отчёты, метрики и графики
src/data.py     загрузка и подготовка данных
src/metrics.py  RMSE и проверка row_id
src/models.py   модели, обучение, inference и checkpoint
tests/          автоматические проверки
```

Подробный формат подготовленных файлов описан в [`data/README.md`](data/README.md),
а математический аудит user/item FM — в [`docs/fm_audit.md`](docs/fm_audit.md).

## Установка и проверка

Для macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Для Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Подготовка MovieLens и запуск тестов выполняются из корня репозитория:

```bash
python -m src.prepare_data \
  --config configs/team1-data.json \
  --report results/team1-data-checks.json

python -m unittest discover -s tests -v
```

## Отдельный эксперимент SVD++

Самостоятельная PyTorch-реализация канонической SVD++ находится в `src/svdpp.py`. Истории строятся только по train, нормировка sqrt, среднее train фиксировано. Выполнены 3 pilot-попытки и 15 основных запусков; лучший k по validation — 10.

CLI `python -m src.run_svdpp` выполняет этапы pilot/sweep/final-evaluate только для SVD++; `python -m src.plot_svdpp` строит её графики из сохранённых CSV. Комплексное сравнение моделей проводится отдельно участниками проекта.

Конфигурация: `configs/svdpp-study.json`; окружение CPU: `requirements-svdpp.txt`; проверки: `python -m src.check_svdpp`; результаты и ограничения: [docs/svdpp_report.md](docs/svdpp_report.md).

## API подготовленных данных

```python
from src.data import MODEL_COLUMNS, load_prepared

data = load_prepared("data/processed/team1")

train = data.train[MODEL_COLUMNS]
validation = data.validation[MODEL_COLUMNS]
test = data.test[MODEL_COLUMNS]

print(data.n_users, data.n_items)
print(len(train), len(validation), len(test))
```

`PreparedData` содержит:

| Поле | Содержимое |
|---|---|
| `train` | обучающие строки с известными индексами |
| `validation` | validation-строки, неизвестные ID уже исключены |
| `test` | test-строки, неизвестные ID уже исключены |
| `user_mapping` | исходный `user_id -> user_idx` |
| `item_mapping` | исходный `item_id -> item_idx` |
| `metadata` | параметры split, размеры, coverage и checksums |
| `n_users`, `n_items` | размеры train-словарей |

`MODEL_COLUMNS` содержит:

```text
row_id, user_idx, item_idx, rating
```

- `row_id` идентифицирует исходное наблюдение и нужен для проверки порядка;
- `user_idx` и `item_idx` — непрерывные индексы, построенные только по train;
- `rating` — целевая переменная;
- значение `-1` означает неизвестный ID и не должно попадать в embedding.

По умолчанию `load_prepared` проверяет checksums всех артефактов. Не следует
создавать собственные mappings или заново разбивать данные внутри модели.

## Полный пример обучения user/item модели

```python
from pathlib import Path

import torch

from src.data import MODEL_COLUMNS, load_prepared
from src.metrics import rmse_by_row_id
from src.models import (
    FactorizationMachine,
    TrainingConfig,
    count_parameters,
    fit_model,
    predict_frame,
    save_checkpoint,
)


def index_tensor(frame, column):
    return torch.tensor(frame[column].to_numpy(copy=True), dtype=torch.long)


def rating_tensor(frame):
    return torch.tensor(frame.rating.to_numpy(copy=True), dtype=torch.float32)


data = load_prepared("data/processed/team1")
train = data.train[MODEL_COLUMNS]
validation = data.validation[MODEL_COLUMNS]
test = data.test[MODEL_COLUMNS]

model = FactorizationMachine(
    data.n_users,
    data.n_items,
    n_factors=16,
    init_std=0.01,
)

config = TrainingConfig(
    epochs=200,
    batch_size=1024,
    learning_rate=0.01,
    optimizer="adam",
    reg_bias=0.01,
    reg_factors=0.05,
    patience=20,
    min_delta=0.0,
    seed=42,
)

result = fit_model(
    model,
    index_tensor(train, "user_idx"),
    index_tensor(train, "item_idx"),
    rating_tensor(train),
    val_user_idx=index_tensor(validation, "user_idx"),
    val_item_idx=index_tensor(validation, "item_idx"),
    val_rating=rating_tensor(validation),
    config=config,
)

test_prediction, test_row_ids = predict_frame(model, test)
test_rmse = rmse_by_row_id(
    test.rating.to_numpy(),
    test_prediction,
    test.row_id.to_numpy(),
    test_row_ids,
)

print("best epoch:", result.best_epoch)
print("fit seconds:", result.fit_time_seconds)
print("parameters:", count_parameters(model))
print("test RMSE:", test_rmse)

save_checkpoint(Path("results/checkpoints/fm.pt"), model, result)
```

Для bias-baseline достаточно заменить создание модели:

```python
from src.models import BiasModel

model = BiasModel(data.n_users, data.n_items)
```

Остальной pipeline остаётся тем же.

### CPU и GPU

Модель обучается на том устройстве, на котором находятся её параметры:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FactorizationMachine(data.n_users, data.n_items).to(device)
```

`fit_model` и `predict_frame` сами переносят очередной batch на устройство модели.
Исходные полные тензоры разрешено оставить на CPU.

## Конфигурация обучения

`TrainingConfig` содержит:

| Поле | По умолчанию | Смысл |
|---|---:|---|
| `epochs` | `200` | максимальное число эпох |
| `batch_size` | `1024` | размер mini-batch |
| `learning_rate` | `0.01` | learning rate оптимизатора |
| `optimizer` | `"adam"` | `"adam"` или `"sgd"` |
| `reg_bias` | `0.01` | L2 для user/item или линейных bias |
| `reg_factors` | `0.05` | L2 для факторных параметров |
| `patience` | `20` | эпох без улучшения до остановки; `None` отключает early stopping |
| `min_delta` | `0.0` | минимальное улучшение validation RMSE |
| `seed` | `42` | инициализация и перестановка train |

Оптимизируется функция

```text
mean squared error
+ reg_bias / n_train * squared_norm(bias parameters)
+ reg_factors / n_train * squared_norm(factor parameters).
```

Глобальный intercept не регуляризуется. Test не передаётся в `fit_model` и не
участвует ни в выборе эпохи, ни в подборе гиперпараметров.

## Что возвращает обучение

`fit_model` возвращает объект `TrainingResult`:

```python
result.history
result.best_epoch
result.fit_time_seconds
```

`history` — список словарей, по одному на выполненную эпоху:

```python
[
    {
        "epoch": 1.0,
        "train_loss": 1.284,
        "train_rmse": 1.106,
        "val_rmse": 1.091,
    },
    {
        "epoch": 2.0,
        "train_loss": 1.103,
        "train_rmse": 1.021,
        "val_rmse": 1.034,
    },
]
```

Числа выше показывают только формат и не являются результатами эксперимента.

- `train_loss` включает MSE и регуляризацию;
- `train_rmse` не включает регуляризацию;
- `val_rmse` не включает регуляризацию и равен `None`, если validation не передан;
- `best_epoch` — эпоха с минимальным validation RMSE;
- `fit_time_seconds` — wall-clock время всего вызова `fit_model`.

Если передан validation, после обучения в модель восстанавливается `state_dict`
лучшей эпохи. Поэтому `predict_frame(model, test)` использует лучшую, а не последнюю
версию параметров. Если validation отсутствует, `best_epoch` равен последней эпохе.

## Предсказание и оценка

```python
prediction, prediction_row_ids = predict_frame(
    model,
    data.validation,
    batch_size=4096,
)
```

Функция возвращает два одномерных NumPy-массива:

- `prediction` — по одному вещественному прогнозу на строку;
- `prediction_row_ids` — `row_id` в исходном порядке таблицы.

Предсказания не ограничиваются диапазоном `[1, 5]`. Если clipping будет добавлен
в эксперимент, он должен быть одинаковым для всех моделей и зафиксирован в
конфигурации.

Оценивать модель следует через строгую проверку порядка:

```python
from src.metrics import rmse_by_row_id

score = rmse_by_row_id(
    data.validation.rating.to_numpy(),
    prediction,
    data.validation.row_id.to_numpy(),
    prediction_row_ids,
)
```

`rmse_by_row_id` отклоняет пропущенные, повторные, лишние или переставленные строки.

## Checkpoint API

```python
from src.models import load_checkpoint, save_checkpoint

save_checkpoint("results/checkpoints/model.pt", model, result)

restored_model, restored_result = load_checkpoint(
    "results/checkpoints/model.pt",
    map_location="cpu",
)
```

Checkpoint содержит:

- имя класса модели;
- результат `get_config()`;
- стандартный PyTorch `state_dict`;
- `TrainingResult`, если он был передан при сохранении.

`load_checkpoint` возвращает модель в режиме `eval()`.

## Общая FM для произвольного вектора x

### Dense API

```python
import torch

from src.models import SparseFactorizationMachine

model = SparseFactorizationMachine(
    n_features=5,
    n_factors=16,
    global_mean=3.5,
)

x = torch.tensor([
    [1.0, 0.0, 0.5, 0.0, 2.0],
    [0.0, 1.0, 0.0, 0.3, 0.0],
])

prediction = model(x)  # shape [2]
```

`x` обязан иметь форму `[batch, n_features]`, floating dtype и только конечные
значения. Этот путь удобен для тестов и небольшого числа признаков.

### Sparse CSR API

Для широких one-hot/multi-hot данных следует передавать только ненулевые элементы:

```python
# Строка 0: x[0]=1.0, x[2]=0.5, x[4]=2.0
# Строка 1: x[1]=1.0, x[3]=0.3
feature_indices = torch.tensor([0, 2, 4, 1, 3], dtype=torch.long)
feature_values = torch.tensor([1.0, 0.5, 2.0, 1.0, 0.3])
row_offsets = torch.tensor([0, 3, 5], dtype=torch.long)

prediction = model.forward_sparse(
    feature_indices,
    feature_values,
    row_offsets,
)
```

Контракт CSR:

- `feature_indices` — плоский список feature ID;
- `feature_values` — соответствующие значения `x_i`;
- `row_offsets` — границы строк, включая финальный `nnz`;
- индексы внутри строки должны быть уникальными и возрастающими;
- пустая строка разрешена повторяющимися offsets и даёт только `global_bias`.

Dense и sparse пути используют одни и те же параметры и дают одинаковые прогнозы
и градиенты. Sparse-сложность равна `O(nnz * n_factors)` без создания dense one-hot.

`fit_model` предназначен для моделей с сигнатурой `(user_idx, item_idx)`. Для общей
FM сейчас используется обычный PyTorch loop:

```python
from src.models import regularized_mse_loss

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

prediction = model.forward_sparse(
    feature_indices,
    feature_values,
    row_offsets,
)
loss = regularized_mse_loss(
    model,
    prediction,
    target,
    n_train=len(target),
    reg_bias=0.01,
    reg_factors=0.05,
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

Построитель design matrix и высокоуровневый trainer для CSR пока не входят в API.
Для воспроизводимого эксперимента их следует добавить до запуска расширенных моделей.

## Как кодировать новые группы признаков

Каждой категории или числовому признаку выделяется уникальный глобальный feature ID.
Диапазоны разных смысловых групп не должны пересекаться.

Например:

```text
[0, U)                    user ID
[U, U + I)                target item ID
[U + I, U + 2I)           item ID в истории пользователя
[U + 2I, U + 2I + T)      day/time category
[U + 2I + T, ...)         frequency, genre и другие признаки
```

Разделение `target item` и `history item` обязательно, если нужны независимые
таблицы факторов, как в представлении SVD++ из статьи.

Все преобразования должны обучаться только по train:

- vocabulary и category mappings;
- нормализация числовых признаков;
- пользовательские истории;
- частоты и агрегаты;
- target statistics.

Validation/test разрешено только преобразовывать уже обученными mappings. Нельзя
строить историю пользователя из validation/test, если она используется для их
предсказания.

## Как добавить новую user/item модель

Модель, совместимая с существующими `fit_model` и `predict_frame`, должна соблюдать
следующий контракт:

```python
from typing import Any

from torch import Tensor, nn


class NewRatingModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, *, hidden_size: int = 32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_size = hidden_size
        # Объявить все обучаемые nn.Parameter/nn.Module.

    def forward(self, user_idx: Tensor, item_idx: Tensor) -> Tensor:
        # Входы: torch.long [batch]. Выход: floating Tensor [batch].
        ...

    def reset_parameters(self, global_mean: float = 0.0) -> None:
        # fit_model вызывает метод перед обучением.
        ...

    def bias_l2(self) -> Tensor:
        # Сумма квадратов параметров линейной/bias части.
        ...

    def factor_l2(self) -> Tensor:
        # Сумма квадратов остальных регуляризуемых параметров.
        ...

    def get_config(self) -> dict[str, Any]:
        # Аргументы, достаточные для повторного __init__.
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "hidden_size": self.hidden_size,
        }
```

Обязательные свойства:

1. `forward` возвращает ровно один прогноз на входную строку.
2. Все параметры зарегистрированы как `nn.Parameter`, `nn.Embedding` или другой
   дочерний `nn.Module`.
3. В `forward` нет `detach()`, NumPy-конверсий или `torch.no_grad()`.
4. `reset_parameters` возвращает модель к воспроизводимой инициализации;
   глобальное среднее берётся только из train.
5. `bias_l2` и `factor_l2` возвращают scalar Tensor на устройстве модели.
6. `get_config()` плюс `state_dict()` полностью восстанавливают модель.
7. Модель корректно работает для batch размера 1 и переменного размера.

Для загрузки checkpoint новый класс нужно зарегистрировать в `model_classes`
внутри `load_checkpoint`:

```python
model_classes = {
    "BiasModel": BiasModel,
    "FactorizationMachine": FactorizationMachine,
    "SparseFactorizationMachine": SparseFactorizationMachine,
    "NewRatingModel": NewRatingModel,
}
```

Минимальные тесты новой модели:

- ручной forward с заранее заданными параметрами;
- shape и dtype выхода;
- конечность прогнозов и градиентов;
- gradient check или независимый oracle;
- воспроизводимость seed;
- уменьшение train loss на маленьком примере;
- checkpoint round trip;
- сохранение порядка `row_id` через `predict_frame`.

## Как добавить модель с другим входным API

SVD++, DCN или общая sparse FM могут требовать историю, контекст или целый feature
batch. Не следует скрыто читать эти данные из глобальных переменных.

Есть два варианта:

1. Подготовить неизменяемый контекст внутри модели до обучения, сохранив внешний
   `forward(user_idx, item_idx)`. Например, SVD++ может хранить train-only историю
   пользователей в зарегистрированных buffers.
2. Создать отдельные `fit_*` и `predict_*`, но сохранить общий результат:
   `TrainingResult` и пару `(prediction, row_ids)`.

Второй вариант необходим, если batch принципиально не сводится к паре user/item.
Экспериментальный код при этом должен использовать общие split, `row_id` и RMSE.

## Воспроизводимый эксперимент

Для каждого запуска необходимо сохранить:

- commit и наличие локальных изменений;
- конфигурацию модели и обучения;
- checksum/версию данных и split ID;
- seed обучения;
- версии Python/PyTorch/NumPy и CPU/GPU;
- историю train loss и validation RMSE;
- лучшую эпоху;
- число параметров;
- время обучения и границы измерения;
- validation/test RMSE и coverage;
- checkpoint выбранной версии модели.

Шаблон находится в [`docs/experiment_log.md`](docs/experiment_log.md). Test следует
оценивать только после выбора модели и гиперпараметров по validation.

## Дополнительная документация

- [`docs/research_plan.md`](docs/research_plan.md) — исследовательский протокол;
- [`docs/work_plan.md`](docs/work_plan.md) — этапы и ожидаемые результаты;
- [`docs/coordination.md`](docs/coordination.md) — ответственность и проверки;
- [`data/README.md`](data/README.md) — источник, split и формат данных;
- [`src/README.md`](src/README.md) — краткая справка по исходному API;
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — правила совместной работы.
