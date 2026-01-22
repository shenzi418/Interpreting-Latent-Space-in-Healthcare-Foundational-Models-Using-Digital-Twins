# Interpreting-Latent-Space-in-Healthcare-Foundational-Models-Using-Digital-Twins

This project explores biomedical digital twins as a tool to interpret latent space representations and assesses their potential to improve trust in healthcare foundational models (FMs).

## Project Structure

The repository currently contains scripts for preparing MedalCare-XL ECG data, creating deterministic splits, fine-tuning the ECGFounder backbone, and evaluating zero-shot or baseline performance.

- `scripts/` – CLI utilities for data preparation (`prepare_medalcare.py`), deterministic splits (`make_splits.py`), training (`finetune_multilabel.py`), zero-shot evaluation (`eval_zero_shot.py`), and dataset definitions.
- `metrics/`, `losses/` – Robust multilabel metrics (AP, Brier, safe ROC-AUC) and an optional MMD alignment loss.
- `viz/plot_waveforms.py` – Visual sanity checks for converted ECG waveforms.
- `data/` – Deterministic MedalCare manifest and split summary artifacts.
- `README_w1.md` – Week-one progress report with step-by-step reproduction instructions and result summaries.

Refer to `README_w1.md` to reproduce the current experiments end-to-end.

## MedalCare splits
- `data/medalcare_filtered_manifest_dataset_split.csv` stores the original
  dataset split (`train`/`val`/`test`) and a `run_id` derived from
  `original_csv_path`.
- Generate it with: `python scripts/add_medalcare_splits.py`