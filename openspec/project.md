# Project Context

## Purpose
Investigate how biomedical digital twins can interpret latent space
representations in ECG foundation models, with two thrusts: (a) aligning
synthetic and real ECG distributions, and (b) decoding mechanistic simulation
parameters (θ) from the latent space to provide physics-grounded
interpretability.

## Tech Stack
- Python 3.10 (Conda env: `env-ECGFounder.yml`)
- PyTorch (GPU-optional; CUDA wheels for NVIDIA GPUs)
- NumPy, pandas, SciPy, scikit-learn
- WFDB for ECG signal I/O
- Matplotlib for waveform/embedding visualization

## Project Conventions

### Code Style
- Python scripts and utilities under `scripts/`, with argparse CLI flags.
- Snake_case for functions/variables; class names in CamelCase.
- Prefer explicit file paths and deterministic outputs (seeded splits).

### Architecture Patterns
- Data preparation and splits are separated (`scripts/prepare_medalcare.py`,
  `scripts/make_splits.py`, `scripts/add_medalcare_splits.py`,
  `scripts/datasets.py`) from training (`scripts/finetune_multilabel.py`)
  and post-hoc analysis (`analysis/`, `scripts/export_latents.py`).
- Shared Net1D encoder (ECGFounder backbone) with three execution modes:
  single-dataset, joint dual-head (per-domain heads), and joint shared-head
  (single 3-class head over the {NORM, MI, CD} ontology).
- Stage-level residual adapters (`ConvAdapter1D`) provide lightweight
  domain adaptation; the backbone is otherwise frozen in linear-probe mode.
- Optional MMD alignment loss (class-agnostic or class-conditional)
  between MedalCare and PTB-XL features when `--lambda-mmd > 0`.
- Latent features and predictions are exported to
  `outputs/latents/<exp_prefix>_<domain>/latents.npz` for downstream
  evaluation; analysis scripts consume these via the
  `latent-evaluation` capability spec.

### Testing Strategy
- No formal unit test suite; validation is via deterministic scripts,
  saved metrics JSON, per-class CSVs, and visual waveform / latent-space
  checks in `outputs/`.
- Each fine-tuning run produces a versioned `outputs/<run_id>/` directory
  with checkpoints, `metrics.json`, and per-class CSVs.

### Git Workflow
- Default workflow: work on feature branches, keep dataset artifacts out
  of version control where possible.

### Spec-driven Development
- Use OpenSpec for new capabilities, breaking changes, or architecture
  shifts: scaffold a proposal under `openspec/changes/<id>/`, get approval,
  implement against `tasks.md`, then archive into
  `openspec/changes/archive/YYYY-MM-DD-<id>/` and merge spec deltas into
  `openspec/specs/`. See `openspec/AGENTS.md` for full details.

## Domain Context
- ECG founder model fine-tuning using synthetic MedalCare-XL ECGs and
  real PTB-XL ECGs.
- Two label spaces are in use: the native per-dataset spaces (8 MedalCare
  labels, 5 PTB-XL superclasses) and the shared 3-class space
  ({NORM, MI, CD}) introduced in Exp 7. Reported metrics are
  accuracy, F1, recall, specificity, precision, Brier, and ROC-AUC.
- Focus on latent-space alignment and mechanistic interpretability via
  MedalCare-XL ground-truth simulation parameters (θ).

## Important Constraints
- Data comes from PhysioNet (PTB-XL) and MedalCare-XL; follow dataset
  licensing and attribution.
- Deterministic splits and seeds are required for reproducibility.
- GPU is optional, but training is heavy without CUDA.

## External Dependencies
- PhysioNet PTB-XL dataset (`ptb-xl-a-large-publicly-available-...`)
- MedalCare-XL synthetic ECG dataset (raw simulation parameter files in
  `MedalCare-XL/WP2_largeDataset_ParameterFiles/`)
- ECGFounder checkpoints in `checkpoint/`
