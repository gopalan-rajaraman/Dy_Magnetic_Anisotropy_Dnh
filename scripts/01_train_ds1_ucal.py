#!/usr/bin/env python3
"""Compare GBR, RFR, DTR and KRR for Ucal prediction using DS1.csv."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "DS1.csv"
OUT = ROOT / "results" / "DS1"
PRED = ROOT / "results" / "predictions" / "DS1"
OUT.mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df.columns = df.columns.astype(str).str.strip()

features = ["CN", "A1", "A2", "BA", "E1", "E2", "E3", "E4"]
target = "Ucal"  # The uploaded CSV column is 'Ucal ' with trailing whitespace.

X = df[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[target], errors="coerce")
valid = X.notna().all(axis=1) & y.notna()
X, y = X.loc[valid], y.loc[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "GBR": GradientBoostingRegressor(
        learning_rate=0.1, max_depth=3, n_estimators=300, random_state=42
    ),
    "RFR": RandomForestRegressor(
        n_estimators=400, max_depth=3, random_state=42, n_jobs=-1
    ),
    "DTR": DecisionTreeRegressor(max_depth=7, random_state=42),
    # Kept unscaled here to match the DS1 code supplied by the user.
    "KRR": KernelRidge(alpha=0.1, kernel="poly")
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Target": target,
        "N_total": len(X),
        "N_train": len(X_train),
        "N_test": len(X_test),
        "Training_R2": r2_score(y_train, y_train_pred),
        "Test_R2": r2_score(y_test, y_test_pred),
        "Training_MAE_K": mean_absolute_error(y_train, y_train_pred),
        "Test_MAE_K": mean_absolute_error(y_test, y_test_pred),
        "Test_RMSE_K": mean_squared_error(y_test, y_test_pred) ** 0.5
    })

    pd.DataFrame({
        "row_index_original_csv_0based": X_test.index,
        "actual_Ucal_K": y_test.to_numpy(),
        "predicted_Ucal_K": y_test_pred,
        "absolute_error_K": abs(y_test.to_numpy() - y_test_pred)
    }).to_csv(PRED / f"DS1_Ucal_{name}_test_predictions.csv", index=False)

results_df = pd.DataFrame(results)
results_df.to_csv(OUT / "DS1_Ucal_model_results.csv", index=False)
print("\\nDS1 Ucal model performance:\\n")
print(results_df.to_string(index=False))
