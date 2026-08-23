# Methodology audit: paper versus corrected code

This document is intentionally conservative. It records what can be supported by the paper, the supplied code, and the supplied CSV files. It does not claim that the cleaned code reproduces the published tables.

## Executive conclusion

The paper's central qualitative claim is narrow: on the reported single-sensor experiments, LSTM with temporal attention slightly outperformed a plain LSTM. The current code expresses that architecture clearly and fixes several leakage and maintenance problems. However, some paper claims and comparisons require qualification:

- the 2024 train/test direction is reversed;
- the original internal validation procedure mixed heavily overlapping windows after shuffling;
- TWDGCN has no meaningful graph in a one-node setting;
- the data appear to be quantized to a 0-100 scale while retaining a physical-unit column name;
- advanced-model code changed after publication, so its current implementation cannot be used as evidence that the paper's old numbers are reproduced.

## Findings and disposition

| Topic | Paper / earlier implementation | Corrected repository | Interpretation |
|---|---|---|---|
| Internal validation | Earlier code shuffled overlapping 12-step windows and used a random validation split. Neighboring train/validation samples could share 11 inputs, and a validation target could appear inside a later training input. | Raw rows are split chronologically first; windows are then built independently. `validation_data` is explicit and `shuffle=False`. | The current validation is leakage-safe, but new validation losses are not directly comparable to old ones. |
| Train/test scaling | The paper says preprocessing and normalization were performed independently on each split (p. 4). Independent fitting on test uses test-wide min/max and is not appropriate for prospective evaluation. | `MinMaxScaler` is fitted on the training portion only, then applied to validation and test. | Current behavior is defensible; paper wording should not be repeated as a best practice. |
| Quantization | The paper describes normalization to `[0,1]`, multiplication by 100, rounding to integers, and final scaling back to `[0,1]` (p. 5). | The supplied CSV target is already in `[0,100]`; the code does not quantize again. | Quantization loses precision. If these are normalized percentages, MAE is in quantized scale units, not recoverable raw vehicle counts. |
| Timestamp gaps | Workdays/weekends are removed, so some adjacent rows are separated by more than five minutes. Earlier ordinary LSTM windows could cross those gaps. | Ordinary LSTM windows crossing a non-five-minute interval are excluded. DeltaRelax-LSTM receives normalized real time gaps. | This avoids treating Friday-to-Monday or missing-day boundaries as ordinary five-minute transitions. |
| 2024 chronology | Paper: train = 2024-02-01 to 2024-07-31; test = 2024-01-01 to 2024-01-31 (pp. 4 and 10). | Timestamp checks reject a test period that precedes training. Files are preserved for provenance but are not a valid default forward forecast. | The reported 2024 experiment is a retrospective/backward holdout, not evidence of forecasting a later unseen period. |
| TWDGCN | Reported as a graph-based baseline in Tables 1-3; the discussion itself acknowledges an artificial graph and a single-sensor setting (p. 11). Earlier code had one node and no usable adjacency/message passing. | Removed. | With one node, graph convolution cannot learn spatial relations. Calling the earlier implementation a meaningful GCN is not defensible. |
| GSA-KAN | Earlier code did not contain a genuine spline-based KAN. | Current exploratory model includes learnable cubic B-spline edge functions plus gated self-attention. | Current code is more faithful to the KAN name, but it is post-publication and does not validate the paper's old GSA-KAN row. |
| DeltaRelax-LSTM | Earlier code supplied constant delta values and did not constrain decay reliably. | Current code reads actual timestamp gaps and uses `softplus(alpha)` to ensure non-negative decay. | Current code is exploratory and post-publication; old and new results must not be conflated. |
| Error plots | Earlier plots labeled one error value at each sample as "MAE" or "MSE." | Plots are labeled pointwise absolute error and squared error; aggregate metrics remain MAE/MSE. | The numeric curves are meaningful, but the earlier statistical labels were imprecise. |
| Percentage error | Paper reports MAPE. MAPE is unstable when true traffic is zero or near zero, and earlier code silently excluded zero targets. | Main evaluation reports SMAPE and WAPE in addition to MAE/MSE/RMSE/R2. | Use MAE/RMSE as primary measures and explain any zero-handling when discussing MAPE. |
| Computational cost | Paper reports millisecond CPU latency, but hardware and timing protocol are not fully specified. | Training metadata records device, parameter count, warm-up, sample count, total time, latency, and RSS delta. | Latency comparisons remain hardware- and software-specific; they are not universal model properties. |

## Architecture actually implemented

### LSTM with temporal attention

1. Input shape: `(batch, 12, 1)`.
2. First LSTM: 64 units, returns a sequence.
3. Batch normalization.
4. Second LSTM: 64 units, returns a sequence.
5. Additive-style scalar score per time step: `tanh(HW + b)`.
6. Softmax across the 12 time steps.
7. Weighted sum of hidden states to create the context vector.
8. Dropout and a one-unit sigmoid output on the scaled target.

The attention weights sum to one for each sample. They describe how the model distributes weight over hidden states; they do **not** prove causality or automatically provide a physical explanation.

### Plain LSTM baseline

The baseline uses the same two 64-unit LSTM layers, batch normalization, dropout, and output constraint, but it takes the final hidden state instead of an attention-weighted context. This makes the core comparison reasonably controlled.

## Published result tables

### 2016 (paper Table 1)

| Model | MAE | MSE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|---:|
| LSTM + Attention | 3.8967 | 28.6424 | 5.3519 | 18.61 | 0.9416 |
| LSTM | 3.9239 | 29.2472 | 5.4081 | 18.47 | 0.9403 |
| GSA-KAN | 5.7131 | 64.6572 | 8.0410 | 24.74 | 0.8681 |
| TWDGCN | 4.0517 | 30.5908 | 5.5309 | 19.60 | 0.9376 |
| DeltaRelax-LSTM | 4.0578 | 30.5902 | 5.5308 | 19.82 | 0.9376 |

### 2024 (paper Table 2)

| Model | MAE | MSE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|---:|
| LSTM + Attention | 5.1170 | 55.0752 | 7.4213 | 17.44 | 0.9286 |
| LSTM | 5.1183 | 55.1830 | 7.4285 | 17.01 | 0.9285 |
| GSA-KAN | 5.8502 | 68.7041 | 8.2888 | 20.35 | 0.9109 |
| TWDGCN | 5.2418 | 58.1848 | 7.6279 | 17.79 | 0.9246 |
| DeltaRelax-LSTM | 5.2923 | 59.1844 | 7.6931 | 18.06 | 0.9233 |

## Claims that are safe to make

- The paper studied single-sensor, univariate, five-minute traffic-flow prediction.
- The central model uses LSTM hidden states and a learned temporal softmax weighting over a 12-step history.
- In the paper's reported tables, attention produced a small improvement over the matched LSTM baseline.
- Attention analysis showed non-uniform temporal patterns, but attention is an interpretability aid rather than causal evidence.
- The corrected code prevents the known internal validation leakage and excludes non-contiguous ordinary LSTM windows.

## Claims to avoid

- "The 2024 experiment proves forward generalization to future data." It does not; the test month is earlier than the training months.
- "TWDGCN learned spatial traffic relationships." The supplied setting has one sensor/node.
- "The current GSA-KAN and DeltaRelax-LSTM code reproduces the paper rows." Those implementations were changed after publication.
- "Every reported error is measured in raw vehicles per five minutes." The supplied 0-100 data and paper's quantization step make that uncertain.
- "Attention explains why traffic changed." It only reports learned weighting inside the model.

