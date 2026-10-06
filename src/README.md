# Модели

Каноническая SVD++ добавлена в `svdpp.py`: `SVDPlusPlus` и `build_train_histories(train, n_users, n_items)`. Истории — unique train items, нормировка sqrt; mu фиксировано, P/Q/Y независимы.

`run_svdpp.py` выполняет только SVD++: 3 pilot attempts и 15 основных запусков. `plot_svdpp.py` строит её графики из CSV. Содержательные проверки модели — `tests/test_svdpp.py`; полная инструкция — [отчёт](../docs/svdpp_report.md).

В `models.py` находятся две PyTorch-модели для уже подготовленных целочисленных
`user_idx` и `item_idx`, а также общая FM:

- `BiasModel`: `mu + b_user + b_item`;
- `FactorizationMachine`: та же линейная часть плюс
  `dot(user_factors, item_factors)` — FM второго порядка для двух one-hot полей.
- `SparseFactorizationMachine`: FM второго порядка для произвольного dense-вектора
  `[batch, n_features]` или разреженного CSR batch через `forward_sparse`.

Подготовка данных, словари категорий, split и обработка неизвестных ID намеренно
не входят в модуль моделей.

Обе модели наследуют `torch.nn.Module`: вызываются через `model(user_idx,
item_idx)`, поддерживают `.to(device)`, `.train()`, `.eval()`, `state_dict()` и
обычные оптимизаторы `torch.optim`.

```python
import torch
from src.data import MODEL_COLUMNS, load_prepared
from src.models import FactorizationMachine, TrainingConfig, fit_model, predict_frame

data = load_prepared("data/processed/team1")
train = data.train[MODEL_COLUMNS]
validation = data.validation[MODEL_COLUMNS]

model = FactorizationMachine(data.n_users, data.n_items, n_factors=16)
result = fit_model(
    model,
    torch.as_tensor(train.user_idx, dtype=torch.long),
    torch.as_tensor(train.item_idx, dtype=torch.long),
    torch.as_tensor(train.rating, dtype=torch.float32),
    val_user_idx=torch.as_tensor(validation.user_idx, dtype=torch.long),
    val_item_idx=torch.as_tensor(validation.item_idx, dtype=torch.long),
    val_rating=torch.as_tensor(validation.rating, dtype=torch.float32),
    config=TrainingConfig(seed=42),
)

prediction, prediction_row_ids = predict_frame(model, data.test)
```

`predict_frame` реализует контракт команды данных: принимает таблицу из
`load_prepared`, не меняет порядок и возвращает прогноз вместе с исходным
`row_id`. В него следует передавать `validation`/`test`, а не варианты
`*_all.csv`: sentinel `-1` для неизвестного ID отклоняется явно.

При наличии validation после early stopping восстанавливается версия с
минимальным validation RMSE. `result.history` содержит train loss, train RMSE и
validation RMSE по эпохам. Для checkpoint используются `save_checkpoint` и
`load_checkpoint`; внутри сохраняется стандартный `state_dict`.

Проверки моделей:

```bash
python -m unittest discover -s tests -v
```

Полное описание данных, обучения, результатов, общей sparse FM и контракта для
добавления моделей находится в корневом [`README.md`](../README.md).
