#!/usr/bin/env python3
"""SHAP analysis for DS2: t = log(tau0), using seven geometric descriptors."""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "DS2.csv")
df.columns = df.columns.astype(str).str.strip()

features = ["BA", "A1", "A2", "E1", "E2", "E3", "E4"]
target = "t"

X = df[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[target], errors="coerce")
valid = X.notna().all(axis=1) & y.notna()
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# Match the supplied SHAP workflow: default GBR hyperparameters, random_state=42,
# fitted to all valid DS2 records for global feature interpretation.
model = GradientBoostingRegressor(random_state=42)
model.fit(X, y)

# TreeExplainer computes tree SHAP values directly for this estimator.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance = pd.DataFrame({
    "Feature": features,
    "Mean_absolute_SHAP": mean_abs_shap
}).sort_values("Mean_absolute_SHAP", ascending=False)
out = ROOT / "figures"
out.mkdir(parents=True, exist_ok=True)
importance.to_csv(out / "DS2_log_tau0_SHAP_importance.csv", index=False)

top5_idx = np.argsort(mean_abs_shap)[-5:]
top5_idx = top5_idx[np.argsort(mean_abs_shap[top5_idx])[::-1]]
X_top5 = X.iloc[:, top5_idx].copy()
shap_top5 = shap_values[:, top5_idx]

plt.rcParams.update({
    "font.size": 18, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.labelsize": 18,
    "xtick.labelsize": 18, "ytick.labelsize": 18
})
plt.figure(figsize=(5.5, 4.5))
shap.summary_plot(
    shap_top5, X_top5, plot_type="violin",
    show=False, plot_size=(5.5, 4.5)
)
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(4)
ax.tick_params(width=4.5, length=6, axis="both", labelsize=18)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")
plt.xlabel("SHAP value (impact on log τ₀)", fontsize=18, fontweight="bold")
plt.grid(False)
plt.subplots_adjust(left=0.35)
if len(plt.gcf().axes) > 1:
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("Feature value", fontsize=16, fontweight="bold")
    cbar.tick_params(labelsize=16)
    for label in cbar.get_yticklabels():
        label.set_fontweight("bold")
plt.tight_layout()
plt.savefig(out / "DS2_log_tau0_SHAP_top5.png", dpi=600, bbox_inches="tight")
plt.savefig(out / "DS2_log_tau0_SHAP_top5.pdf", bbox_inches="tight")
print(importance.to_string(index=False))
print("Saved SHAP PNG/PDF and importance CSV in", out)
