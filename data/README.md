# Data card

All files contain a timestamp column named `5 Minutes` and a single target named `Lane 1 Flow (Veh/5 Minutes)`.

| File | Rows | First timestamp | Last timestamp | Intended role |
|---|---:|---|---|---|
| `train.csv` | 7,776 | 2016-01-04 00:00 | 2016-02-29 23:55 | Paper 2016 train/validation source |
| `test.csv` | 4,320 | 2016-03-04 00:00 | 2016-03-31 23:55 | Paper 2016 held-out test |
| `pems_2024/workdays/train.csv` | 36,187 | 2024-02-01 00:00 | 2024-07-31 23:55 | Paper 2024 training source |
| `pems_2024/workdays/test.csv` | 6,618 | 2024-01-01 00:00 | 2024-01-31 23:55 | Paper 2024 retrospective test |
| `pems_2024/all_days_train.csv` | 51,045 | 2024-02-01 00:00 | 2024-07-31 23:55 | Additional version including non-workdays |
| `pems_2024/all_days_test.csv` | 8,913 | 2024-01-01 00:00 | 2024-01-31 23:55 | Additional version including non-workdays |

## Provenance

The paper identifies the source as the California Department of Transportation Performance Measurement System (Caltrans PeMS), sampled at five-minute resolution for one lane/sensor. The repository does not contain the original download query, station identifier, raw untransformed export, or original min/max values. Those missing provenance details limit exact independent reconstruction.

## Scale caveat

The target values in the supplied CSV files lie between 0 and 100. The paper describes normalizing the original series, multiplying it by 100, and rounding to integers before a final model-input scaling step. The stored column name still states `Veh/5 Minutes`, but the available values may therefore represent a quantized normalized scale rather than raw vehicle counts. Do not claim raw physical units unless the original transformation parameters are recovered.

## Temporal caveat for 2024

The 2024 test month (January) precedes the training period (February-July). This is not a valid forward-chaining forecast split. The corrected data pipeline raises an error when asked to use this ordering. To create a defensible 2024 experiment, obtain a later held-out period or chronologically repartition the source data, then retrain all compared models under the same protocol.

## Missing intervals

The retained periods omit some dates (for example weekends). Ordinary LSTM windows that cross a timestamp gap are excluded by `traffic_flow/data.py`. The DeltaRelax-LSTM exploratory model receives the actual normalized time gaps.

