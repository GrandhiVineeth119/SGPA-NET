# SGPA-Net: Architecture

SGPA-Net is a seizure-prediction model for scalp EEG. It classifies 5-second windows as **preictal** (within 30 minutes before a seizure) or **interictal** (more than 60 minutes from any seizure). It is evaluated with leave-one-subject-out (LOSO) on the CHB-MIT dataset, so each test patient is unseen during training.

## Pipeline overview

```
EEG window (18 channels × 5 s, 256 Hz)
        │
        ├── Wavelet scattering branch
        │     Scattering1D (J=6, Q=8), time-averaged
        │     → 126 features per channel
        │
        └── PLV functional graph
              Hilbert phase locking value per band:
                θ 4–8 Hz, α 8–13 Hz, β 13–30 Hz, γ 30–45 Hz
              Edge kept if PLV ≥ 0.3
              edge_attr = the 4 band PLVs
                    │
                    ▼
Spatial encoder (applied to each window)
    GATv2Conv with 4-dim edge attributes
    (GCNConv is the ablation baseline)
    → global mean pooling → one vector per window
                    │
                    ▼
Sequence of 8 consecutive windows (about 40 s of context)
    Temporal encoder: Mamba-style selective SSM
    (d_state 16, d_conv 4, expand 2)
    (BiLSTM, BiGRU and Transformer are ablation baselines)
                    │
                    ▼
Classifier head → preictal vs interictal logit
```

## Components

### 1. Input and preprocessing
- **Channels:** 18 common bipolar channels from CHB-MIT, sampled at 256 Hz.
- **Windows:** 5 s, 50% overlap.
- **Labels:** preictal = 30 min before onset; interictal = more than 60 min from any seizure.
- **Filtering:** 60 Hz notch, then 0.5–45 Hz bandpass.
- **Artifact removal:** ICA with 15 components; at most 2 EOG-like components removed.
- **Normalization:** per-window, per-channel z-score.

### 2. Wavelet scattering branch
- `kymatio` `Scattering1D` with J = 6 and Q = 8.
- Coefficients are time-averaged, giving 126 features per channel.
- Captures the spectral shape of each channel.

### 3. PLV functional graph
- Nodes are the 18 channels.
- Edges connect channel pairs whose phase locking value is at least 0.3.
- Each edge has 4 attributes: the PLV in each of the θ, α, β and γ bands.
- Captures synchrony between channels, which changes before seizures.

### 4. Spatial encoder
- **Proposed:** GATv2Conv that uses the 4-dimensional edge attributes.
- **Baseline:** GCNConv, which ignores edge attributes. Used for the ablation.
- Global mean pooling turns the node embeddings of one window into a single vector.

### 5. Temporal encoder
- Takes a sequence of 8 window vectors.
- **Proposed:** a Mamba-style selective state-space model. Implemented in pure PyTorch with d_state 16, d_conv 4 and expand 2.
- **Baselines:** BiLSTM, BiGRU and a Transformer encoder.
- Stability: LayerNorm, a clamp on the Δt term at 10, a clamp on the state at ±100, and gradient clipping at 0.5. These were needed to stop NaN losses during training.

### 6. Classifier head
- Maps the temporal output to one logit for preictal vs interictal.

## Training

| Setting | Value |
|---|---|
| Loss | Class-weighted cross-entropy |
| Training balance | 3 interictal windows to 1 preictal window |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-5 |
| Scheduler | CosineAnnealingWarmRestarts, T_0 = 10, T_mult = 2 |
| Gradient clipping | 0.5 |
| Batch size | 64 |
| Epochs | 50 |
| Sequence length | 8 windows |

## Evaluation

- **Protocol:** leave-one-subject-out. The held-out patient is never seen in training.
- **Metrics:** sensitivity, specificity, F1, AUC.
- **Clinical metric:** false alarms per hour, computed as false positives divided by the hour's 720 windows.
- **Threshold search:** `find_optimal_threshold` looks for a threshold that meets a target FPR per hour.

## Ablation design

| Experiment | Spatial | Temporal | PLV edges |
|---|---|---|---|
| 1 | GCN | LSTM | No |
| 2 | GAT | LSTM | Yes |
| 3 | GAT | Transformer | Yes |
| 4 | GAT | Mamba | Yes |

Each row changes one component against the previous row, so the effect of each choice can be read from the table.

## Design rationale

- **Graph over flat features:** seizures change how channels synchronize, and a graph represents synchrony directly.
- **Edge attributes:** the PLV value per band tells the attention layer how strong each connection is, which GCN cannot use.
- **Sequence model over single windows:** a single 5 s window is often ambiguous. Eight windows give the model the trend leading up to a seizure.
- **Mamba as the temporal model:** a selective state-space model handles long sequences with linear cost, which suits continuous monitoring.

## Current status

- The 6-patient GAT + Mamba run has a mean default-threshold F1 of 0.548 and an AUC of 0.694.
- False alarms are about 209 per hour at the default threshold, far above the clinical target. The threshold search did not reach the target on several folds.
- The proposed architecture has not yet been shown to beat the baselines on the 10-patient set.
