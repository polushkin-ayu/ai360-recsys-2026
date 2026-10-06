import json
from pathlib import Path
import plotly.graph_objects as go

# Загружаем результаты sweep'а
sweep_path = Path("results/pure_mf_sweep.json")
with open(sweep_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sweep_results = data["sweep"]
k_values = [int(k) for k in sweep_results.keys()]
rmse_values = [res["val_rmse"] for res in sweep_results.values()]

# Строим график
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=k_values,
    y=rmse_values,
    mode="lines+markers",
    name="PureMF Validation RMSE",
    line=dict(color="royalblue", width=3),
    marker=dict(size=8)
))

fig.update_layout(
    title="PureMatrixFactorization: Hyperparameter Sweep (k vs Validation RMSE)",
    xaxis_title="Number of Factors (k)",
    yaxis_title="Validation RMSE",
    xaxis=dict(type="log", tickvals=k_values),
    template="plotly_white"
)

output_path = Path("results/pure_mf_sweep_plot.html")
fig.write_html(output_path)
print(f"График успешно сохранен в {output_path}!")