# Week 1 Progress Report – ECGFounder MedalCare Adaptation


---

## 1. Executive Summary

- Consolidated MedalCare-XL synthetic ECG data into a deterministic, WFDB-backed dataset ready for model training.
- Established stable train/validation/test splits to prevent subject leakage and enable reproducible benchmarking.
- Added safety checks (waveform visualization, robust metrics) to validate data integrity and evaluation stability.
- Baseline fine-tuning runs (frozen encoder with linear/MLP heads) now outperform the zero-shot foundation model on ≥6 of 8 cardiac pathologies.
- Introduced optional MMD alignment to reduce synthetic–real domain gaps and instrumented the pipeline with structured evaluation and summaries.

All artifacts and scripts live in the repository so that any teammate can re-run the workflow end-to-end.

---

## 2. Environment & Tooling

- **Python**: 3.10 (scripts assume ≥3.9).
- **PyTorch**: 2.4 with CUDA 12.1 wheels; CPU-only mode also works.
- **GPU**: Optional. Scripts detect GPU automatically; runs fall back to CPU with a warning.
- **Environment Setup**:
  ```powershell
  conda create -n ECGFounder python=3.10 -y
  conda activate ECGFounder
  pip install -r requirements.txt
  ```
- **Determinism Controls**: All major entry points accept `--seed` (default 42), set CuDNN deterministic flags, and apply label smoothing, gradient clipping, and class-balanced `pos_weight` to stabilize training.

**Known Pitfalls Addressed**
- Single-class ROC-AUC crashes replaced with a “safe” implementation returning `None` for degenerate cases.
- `pos_weight` estimation restricted to the train split to avoid biased thresholds.
- Gradient clipping (`--grad-clip 1.0`) prevents divergence when the encoder is unfrozen or when learning rates change.

---

## 3. Technical Achievements

### 3.1 Data Pipeline Consolidation
- Merged three legacy scripts into `scripts/prepare_medalcare.py`.
- Result: deterministic manifest `MedalRaw/medalcare_filtered_manifest.csv` pointing to WFDB signals; failed conversions are excluded.
- Outcome: Re-running produces byte-identical manifests; downstream scripts expect the filtered 12-lead signals with documented metadata.

### 3.2 Deterministic Splits
- Implemented `scripts/make_splits.py` using `StratifiedGroupKFold` (seed 42) with subject-level grouping.
- Outputs `data/splits_v1.json` and an augmented manifest with `split`/`fold` columns.
- Ensures train/val/test prevalence alignment and leakage-free evaluation.

### 3.3 Data Quality Verification
- Added `viz/plot_waveforms.py` to sample 10 test records and render leads II/V1/V6.
- Generated waveforms stored under `outputs/quick_waveform_check/waveforms/`.
- Plots confirm physiologic morphology and dependable amplitude scaling.

### 3.4 Zero-Shot Baseline 
- Wrote `scripts/eval_zero_shot.py` to benchmark the foundation checkpoint (`checkpoint/12_lead_ECGFounder.pth`) without fine-tuning.
- Validation/test metrics saved as `outputs/zero_shot/zero_shot_<split>.json`.
- Establishes a reference macro AP of ~0.16 for comparison.

### 3.5 Frozen Encoder Baselines 
- Extended `scripts/finetune_multilabel.py` to support:
  - Linear vs. MLP classification heads (`--head-type`).
  - Encoder freezing (`--freeze-encoder`) with separate learning rates.
  - Class-balanced BCEWithLogits loss (`pos_weight`).
  - Label smoothing (0.05) and gradient clipping (1.0).
- Runs create:
  - Linear head artifacts under `outputs/w1_baselines/linear/`.
  - MLP head artifacts under `outputs/w1_baselines/mlp/`.
- Both runs surpass zero-shot macro AP by a wide margin and remain numerically stable (no NaNs/Infs).
- Checkpoints `w45_frozen_linear_best.pt` and `w45_frozen_mlp_best.pt` contain the frozen ECGFounder encoder weights (`first_conv.*`, `stage_list.*`) alongside the MedalCare heads (`dense*` layers), even though no explicit `encoder`/`mlp` prefixes appear in the `state_dict`.

### 3.6 Optional Domain Alignment 
- Added RBF-based MMD loss (`losses/mmd.py`) and integrated it into the training loop via `--lambda-mmd` and `--domain-column`.
- Alignment run stored in `outputs/w1_alignment/`; logs include batch-level MMD diagnostics.
- Early experiments show improved macro AP on ≥4/8 tasks when domains are available.

---

## 4. Result Metrics Summary

| Run | Test Macro AP | Test Macro Brier ↓ | Test Macro ROC-AUC | Metrics Artifact |
| --- | --- | --- | --- | --- |
| Zero-shot foundation model | 0.164 | 0.348 | 0.515 | `outputs/zero_shot/zero_shot_test.json` |
| Frozen encoder – linear head | 0.813 | 0.117 | 0.975 | `outputs/w1_baselines/linear/metrics.json` |
| Frozen encoder – MLP head | 0.829 | 0.086 | 0.971 | `outputs/w1_baselines/mlp/metrics.json` |
| Full fine-tune MLP (encoder trainable) | 0.923 | 0.072 | 0.991 | `outputs/w1_full_mlp/metrics.json` |

**Highlights**
- Fine-tuning even a frozen linear head lifts macro AP by **+0.65** over zero-shot and cuts Brier error by ~66%.
- Allowing an MLP head and training the encoder end-to-end reaches **0.923 macro AP** with strong calibration (`Brier ≈ 0.072`).
- Alignment run with the current manifest mirrors the frozen-MLP baseline; additional domain labels are needed to demonstrate further gains.

---


