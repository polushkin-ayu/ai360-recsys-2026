from pathlib import Path

import pandas as pd

from src.data import load_prepared


# бюджеты по числу обучаемых параметров
BUDGETS = [
    100_000,
    200_000,
    500_000,
    1_000_000,
]


def main():
    # Загружаем именно общий MovieLens 1M
    data = load_prepared("data/processed/movielens1m")

    n_users = data.n_users
    n_items = data.n_items

    train = data.train.copy()

    # Для contextual FM нужен t = day
    if "timestamp" not in train.columns:
        raise ValueError(
            "timestamp not found in MovieLens 1M train"
        )

    train["day"] = (
        train["timestamp"].astype("int64") // 86400
    )

    n_days = train["day"].nunique()

    print("Dataset:")
    print("users:", n_users)
    print("items:", n_items)
    print("train days:", n_days)
    print()

    # ============================
    # Количество параметров
    # ============================

    # Pure MF:
    #
    # user factors: n_users * k
    # item factors: n_items * k
    def params_mf(k):
        return (
            (n_users + n_items) * k
        )

    # FM(u,i):
    #
    # global bias
    # user bias
    # item bias
    # user factors
    # item factors
    def params_fm_ui(k):
        return (
            1
            + n_users
            + n_items
            + (n_users + n_items) * k
        )

    # SVD++:
    #
    # user bias
    # item bias
    # p_u: users * k
    # q_i: items * k
    # y_j: items * k
    def params_svdpp(k):
        return (
            n_users
            + n_items
            + (
                n_users
                + 2 * n_items
            ) * k
        )

    # FM(u,i,t)
    def params_fm_uit(k):
        return (
            1
            + n_users
            + n_items
            + n_days
            + (
                n_users
                + n_items
                + n_days
            ) * k
        )

    # FM(u,i,t,f)
    #
    # + frequency linear weight
    # + frequency latent vector
    def params_fm_uitf(k):
        return (
            2
            + n_users
            + n_items
            + n_days
            + (
                n_users
                + n_items
                + n_days
                + 1
            ) * k
        )

    models = {
        "MF": params_mf,
        "SVD++": params_svdpp,
        "FM(u,i)": params_fm_ui,
        "FM(u,i,t)": params_fm_uit,
        "FM(u,i,t,f)": params_fm_uitf,
    }

    # ============================
    # Ищем k, который ближе всего
    # к каждому parameter budget
    # ============================

    def closest_k(param_fn, budget):
        best_k = None
        best_params = None
        best_diff = None

        # 1..500 более чем достаточно
        for k in range(1, 501):
            n_params = param_fn(k)

            diff = abs(
                n_params - budget
            )

            if (
                best_diff is None
                or diff < best_diff
            ):
                best_diff = diff
                best_k = k
                best_params = n_params

        return best_k, best_params

    rows = []

    for model, param_fn in models.items():
        for budget in BUDGETS:
            k, actual_params = closest_k(
                param_fn,
                budget,
            )

            rows.append({
                "model": model,
                "budget": budget,
                "k": k,
                "actual_params": actual_params,
                "difference": (
                    actual_params - budget
                ),
            })

    df = pd.DataFrame(rows)

    # Для красивого вывода
    df = df.sort_values(
        ["model", "budget"]
    ).reset_index(drop=True)

    print(df.to_string(index=False))

    # ============================
    # Сохраняем
    # ============================

    output_dir = Path(
        "results/parameter-budget-study"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "budget_plan.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Saved:")
    print(output_path.resolve())


if __name__ == "__main__":
    main()