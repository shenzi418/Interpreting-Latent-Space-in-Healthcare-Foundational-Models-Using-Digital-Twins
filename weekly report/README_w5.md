# Week 5 - Physics Head for MedalCare Digital-Twin Parameters

## 1. Overview
- Added a physics head to probe whether the shared encoder captures mechanistic signal by predicting the MedalCare-XL digital-twin parameter vector (θ).
- Formalized a fixed θ contract (ordered list + scaling policy) sourced from MedalCare-XL parameter files.
- Extended MedalCare dataset ingestion to attach θ + availability mask and normalization metadata.
- Integrated physics regression loss into the multi-head training loop for MedalCare batches only.
- Evaluated on a synthetic held-out test split and produced physics metrics + sanity plots.

## 2. Physics Head Design (Key Decisions)
- **θ contract:** fixed, ordered 51-dim vector derived from atrial + ventricular parameter files; ordering frozen in config (`config/theta.json` and `config/theta_core.json`).
- **Input representation:** physics head uses the shared encoder pooled feature `z` (same representation as classification heads).
- **Architecture:** MLP head (2–3 FC layers + ReLU, optional dropout) mapping `z → θ̂`, configurable hidden size and dropout.
- **Loss + routing:** masked regression loss (MSE/Huber) in normalized θ space; computed only for MedalCare batches. PTB-XL batches skip physics loss.
- **Normalization:** per-component transforms (e.g., log/logit when needed) followed by z-score. Stats computed on MedalCare train split only and stored in `outputs/theta_stats.json`.
- **Training schedule:** Stage A baseline or joint MMD training; Stage B freeze encoder/heads and optimize physics loss only.

## 3. Issues Encountered and Fixes
- **Parameter file mapping ambiguity:** ECG CSV paths pointed to noisy waveform directories, not parameter files. Fixed by deterministic path swap (`WP2_largeDataset_Noise → WP2_largeDataset_ParameterFiles`) and filename rewrite (`run_XXXXXX_filtered.csv → run_XXXXXX_{Atrial,Ventricular}Parameters.txt`).
- **Parameter parsing inconsistencies:** values arrived with suffixes/units and occasional malformed entries. Implemented a robust key=value parser with integrity checks and a per-component availability mask `m` to ignore missing/NaN components in loss.
- **θ scaling stability:** raw parameter scales vary widely and would dominate optimization. Chosen per-component transforms and z-score normalization with train-only stats; stored metadata for consistent inference and de-normalization.
- **Loss contamination across datasets:** physics loss must not update on PTB-XL batches. Loss routing explicitly gated by dataset source to keep gradients correct.
- **Leakage risk in evaluation:** split by `run_id` on synthetic MedalCare to keep parameter families disjoint between train/test.

## 4. Results (Physics Metrics)
- Evaluated on the MedalCare synthetic test split (physics-only core run).
- Summary from `outputs/physics_only_core_v1/physics_metrics.json`:
  - Normalized MAE mean: **0.509** (median 0.452)
  - Raw MAE mean: **7853.85** (median 5785.82)
  - R² mean: **0.064** (median 0.328)
  - R² tiers: **21 strong**, **5 moderate**, **25 weak**
- Interpretation: several θ components are well predicted (strong R²), but many remain weak; the head demonstrates partial mechanistic signal capture and highlights which parameters need targeted modeling improvements.

## 5. Artifacts Produced
- θ contracts: `config/theta.json`, `config/theta_core.json`
- θ statistics: `outputs/theta_stats.json`, `outputs/theta_core_stats.json`
- Physics evaluation: `outputs/physics_only_core_v1/physics_metrics.json`
- Sanity plots: `outputs/physics_only_core_v1/physics_plots/` (θ̂ vs θ and distribution overlays)


