#!/usr/bin/env python3
"""Check the two repository datasets for row counts, missing values and target ranges."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

report = {}
for filename, name, targets in [
    ("DS1.csv", "DS1", ["Ucal"]),
    ("DS2.csv", "DS2", ["Ucal", "Ueff", "t"])
]:
    df = pd.read_csv(DATA / filename)
    df.columns = df.columns.astype(str).str.strip()
    report[name] = {
        "file": filename,
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "CN_counts": {str(k): int(v) for k, v in df["CN"].value_counts().items()},
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "target_ranges": {
            target: [float(pd.to_numeric(df[target], errors="coerce").min()),
                     float(pd.to_numeric(df[target], errors="coerce").max())]
            for target in targets
        }
    }
    if "CCDC" in df.columns:
        report[name]["duplicate_CCDC_rows"] = int(df["CCDC"].duplicated(keep=False).sum())

(OUT / "dataset_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
