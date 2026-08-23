# Dataset

The repository includes the two CSV files used by the runnable 2016 pipeline.

| File | Rows | First timestamp | Last timestamp | Role |
|---|---:|---|---|---|
| `train.csv` | 7,776 | 2016-01-04 00:00 | 2016-02-29 23:55 | Training and chronological validation source |
| `test.csv` | 4,320 | 2016-03-04 00:00 | 2016-03-31 23:55 | Held-out evaluation source |

Both files contain:

- `5 Minutes`: timestamp;
- `Lane 1 Flow (Veh/5 Minutes)`: prediction target;
- `# Lane Points` and `% Observed`: source metadata.

The data originate from the California Department of Transportation Performance Measurement System (Caltrans PeMS) and are sampled at five-minute resolution. The stored target values use the processed 0-100 representation supplied with the project.

Some dates are not present in the retained periods. `traffic_flow/data.py` checks timestamps and excludes ordinary LSTM windows that cross a non-five-minute interval.

