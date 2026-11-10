# Dy(III) SMM Machine-Learning Repository

This repository contains only the two datasets, model-training scripts, SHAP analysis,
and generated results/figures required to reproduce the analyses described in the scripts.

## Dataset naming

The uploaded datasets have been renamed as follows:

| Repository file | Original uploaded dataset | Purpose |
|---|---|---|
| `data/DS1.csv` | `DS1_finalK` | Main Ucal model |
| `data/DS2.csv` | `TB_Dnh_final - Copy.csv` | Experimental Ueff and log(tau0) models |

I interpreted the second occurrence of “DS2” in the request as **DS1**, because the
repository needs two distinct dataset names.

The CSV contents are preserved. The scripts strip whitespace from column headers in
memory. This is needed because the DS1 file contains the header `Ucal ` with a trailing
space; the scripts therefore refer to it as `Ucal`.

## Models and features

### DS1: Ucal prediction
- Target: `Ucal` (K)
- Features: `CN, A1, A2, BA, E1, E2, E3, E4`
- Models: GBR, RFR, DTR, KRR
- Split: 80:20 random train/test, `random_state=42`

### DS2: Ueff and log(tau0) prediction
- `Ueff` target: `Ueff` (K)
- `log(tau0)` target: `t` (the log relaxation-time value in the uploaded table)
- Features for both targets: `CN, A1, A2, BA, E1, E2, E3, E4, Dy, LA1, LA2, LE1, LE2, LE3, LE4, gz, B20, Ucal`
- Models: GBR, RFR, DTR, KRR
- Split: 80:20 random train/test, `random_state=42`

**Important:** Ucal is included as an input feature for DS2 Ueff and `t` models because
that is what the supplied code specifies. These models therefore require a Ucal value
to be available when making a prediction.

### SHAP analysis
- Dataset: DS2
- Target: `t` = log(tau0)
- Features: `BA, A1, A2, E1, E2, E3, E4`
- Model: default `GradientBoostingRegressor(random_state=42)` fitted to all valid DS2 rows
- Output: top-five-feature violin summary plot at 600 dpi (PNG and PDF), plus mean
  absolute SHAP feature importance table.

The SHAP model is a full-dataset explanatory model, not the held-out test model from
the model-comparison script.

## Reproduce the analysis

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
```

Or run each step separately:

```bash
python scripts/00_audit_datasets.py
python scripts/01_train_ds1_ucal.py
python scripts/02_train_ds2_targets.py
python scripts/03_shap_ds2_logtau.py
```

## Outputs

- `results/DS1/DS1_Ucal_model_results.csv`
- `results/DS2/DS2_Ueff_model_results.csv`
- `results/DS2/DS2_log_tau0_model_results.csv`
- `results/DS2/DS2_all_targets_model_results.csv`
- `results/predictions/` — held-out test predictions and absolute errors
- `results/dataset_audit.json` — row counts, missing values, and target ranges
- `figures/DS2_log_tau0_SHAP_top5.png`
- `figures/DS2_log_tau0_SHAP_top5.pdf`
- `figures/DS2_log_tau0_SHAP_importance.csv`

## Reproducibility and interpretation notes

1. DS2 has 130 rows with CN counts D4h=23, D5h=66, D6h=41. It contains repeated CCDC
   identifiers. The requested random split is retained to reproduce the supplied code;
   because of repeated CCDC values, a stricter grouped validation should also be
   considered before making claims about generalization to unseen compounds.
2. The DS1 KRR implementation is left unscaled to match the DS1 code supplied. The
   DS2 KRR implementation uses `StandardScaler` in a pipeline, matching the DS2 code.
3. The result tables report R² and MAE for both training and test sets, and additionally
   report test RMSE. Ucal and Ueff errors are in K; `t` errors are in log units.
4. No target values, descriptor values, or rows were modified in the packaged datasets.
