# Dataset notes

## DS1.csv
- Source: uploaded `DS1_finalK` file.
- 1,429 rows; coordination-number counts: CN=4: 257, CN=5: 872, CN=6: 300.
- Target: `Ucal` in K.
- The original CSV header contains `Ucal ` (trailing whitespace); code strips header whitespace.
- One `SN` entry is missing, but `SN` is not used as a model feature or target.

## DS2.csv
- Source: uploaded `TB_Dnh_final - Copy.csv` file.
- 130 rows; coordination-number counts: CN=4: 23, CN=5: 66, CN=6: 41.
- Targets used: `Ueff` in K and `t` as log(tau0).
- All features and targets required by the supplied model scripts are present.
- 24 rows belong to repeated CCDC identifiers; random train/test splitting may place
  records sharing a CCDC identifier in both sets.
