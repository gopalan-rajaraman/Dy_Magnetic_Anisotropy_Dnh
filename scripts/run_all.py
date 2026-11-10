#!/usr/bin/env python3
"""Run dataset audit, DS1/DS2 model comparisons, and DS2 SHAP analysis."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "00_audit_datasets.py",
    "01_train_ds1_ucal.py",
    "02_train_ds2_targets.py",
    "03_shap_ds2_logtau.py"
]
for script in SCRIPTS:
    print(f"\\n===== Running {script} =====", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
print("\\nAll analyses completed.")
