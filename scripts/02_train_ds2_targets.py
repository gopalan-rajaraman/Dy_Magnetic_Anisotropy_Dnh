#!/usr/bin/env python3
"""Compare GBR, RFR, DTR and KRR for Ueff and log(tau0) prediction using DS2.csv.

The feature list is taken from the supplied DS2 code. It includes Ucal as an input
descriptor for predicting Ueff and t (= log(tau0)).
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "DS2.csv"
OUT = ROOT / "results" / "DS2"
PRED = ROOT / "results" / "predictions" / "DS2"
OUT.mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df.columns = df.columns.astype(str).str.strip()

features = [
    "CN", "A1", "A2", "BA", "E1", "E2", "E3", "E4",
    "Dy", "LA1", "LA2", "LE1", "LE2", "LE3", "LE4",
    "gz", "B20", "Ucal"
]
targets = {
    "Ueff": ("Ueff", "K"),
    "log_tau0": ("t", "log units")
}

models = {
    "GBR": GradientBoostingRegressor(
        learning_rate=0.1, max_depth=3, n_estimators=300, random_state=42
    ),
    "RFR": RandomForestRegressor(
        n_estimators=400, max_depth=3, random_state=42, n_jobs=-1
    ),
    "DTR": DecisionTreeRegressor(max_depth=7, random_state=42),
    "KRR": Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(alpha=0.1, kernel="poly"))
    ])
}

all_results = []
for target_name, (target_column, unit) in targets.items():
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_column], errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    X_target, y_target = X.loc[valid], y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X_target, y_target, test_size=0.2, random_state=42
    )

    rows = []
    for name, estimator in models.items():
        # Recreate each estimator for an independent fit.
        from sklearn.base import clone
        model = clone(estimator)
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        row = {
            "Model": name,
            "Target": target_name,
            "Target_column": target_column,
            "N_total": len(X_target),
            "N_train": len(X_train),
            "N_test": len(X_test),
            "Features": ";".join(features),
            "Split": "80:20 random",
            "Random_state": 42,
            "Training_R2": r2_score(y_train, y_train_pred),
            "Test_R2": r2_score(y_test, y_test_pred),
            f"Training_MAE ({unit})": mean_absolute_error(y_train, y_train_pred),
            f"Test_MAE ({unit})": mean_absolute_error(y_test, y_test_pred),
            f"Test_RMSE ({unit})": mean_squared_error(y_test, y_test_pred) ** 0.5
        }
        rows.append(row)

        pd.DataFrame({
            "row_index_original_csv_0based": X_test.index,
            f"actual_{target_name}": y_test.to_numpy(),
            f"predicted_{target_name}": y_test_pred,
            "absolute_error": abs(y_test.to_numpy() - y_test_pred)
        }).to_csv(PRED / f"DS2_{target_name}_{name}_test_predictions.csv", index=False)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT / f"DS2_{target_name}_model_results.csv", index=False)
    all_results.extend(rows)
    print(f"\\nDS2 {target_name} model performance:\\n")
    print(result_df.to_string(index=False))

pd.DataFrame(all_results).to_csv(OUT / "DS2_all_targets_model_results.csv", index=False)
