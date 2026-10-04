# Модели

В `models.py` находятся две PyTorch-модели для уже подготовленных целочисленных
`user_idx` и `item_idx`:

- `BiasModel`: `mu + b_user + b_item`;
- `FactorizationMachine`: та же линейная часть плюс
  `dot(user_factors, item_factors)` — FM второго порядка для двух one-hot полей.

Подготовка данных, словари категорий, split и обработка неизвестных ID намеренно
не входят в модуль моделей.

Обе модели наследуют `torch.nn.Module`: вызываются через `model(user_idx,
item_idx)`, поддерживают `.to(device)`, `.train()`, `.eval()`, `state_dict()` и
обычные оптимизаторы `torch.optim`.

```python
import torch
from src.models import FactorizationMachine, TrainingConfig, fit_model

model = FactorizationMachine(n_users, n_items, n_factors=16)
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

model.eval()
with torch.no_grad():
    prediction = model(test_user_idx, test_item_idx)
```

При наличии validation после early stopping восстанавливается версия с
минимальным validation RMSE. `result.history` содержит train loss, train RMSE и
validation RMSE по эпохам. Для checkpoint используются `save_checkpoint` и
`load_checkpoint`; внутри сохраняется стандартный `state_dict`.

Проверки моделей:

```bash
python -m unittest discover -s tests -v
```
