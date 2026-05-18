*Apply when about to run a CLI command in this repo (install, data prep, train, latent export, analysis).*

## Install
- `conda env create -f env-ECGFounder.yml && conda activate ECGFounder` — canonical; includes `torch==2.9.1+cu128`.
- `pip install -r requirements.txt` — partial install: **no torch** (install separately), and `neurokit2==0.2.10` is here but NOT in the conda env.

## One-time data prep (run in order)
- `python scripts/prepare_medalcare.py` — MedalCare CSV → WFDB.
- `python scripts/make_splits.py` — deterministic train/val/test (seeded from `original_csv_path` hash).
- `python scripts/add_medalcare_splits.py` — attach split column to manifest.

## Training
- `python scripts/finetune_multilabel.py --run-id <id> --mode {single|joint_dual|shared_head} [--lambda-mmd 0.1 --mmd-class-conditional]`.
- `python scripts/finetune_bottleneck.py --run-id <id> --bottleneck-dim K` — K ∈ {16, 64, 256, 1024}.

## Latent export
- `python scripts/export_latents.py --run-id <id>` (1024-d) or `python scripts/export_bottleneck_latents.py --run-id <id>` (K-d).
- Output path: `outputs/latents/<prefix>_<domain>/latents.npz`.

## Analysis
- `python analysis/tier1_evaluation.py --run-id <id>` — alignment + class-structure + mechanism evaluation suite.
- `python analysis/dim_scan.py` — PCA K-dimension sweep (post-hoc, no retrain).
- `python analysis/phase_b2_infarct_decoding.py` — cross-domain territory decoding (Track 3 active diagnostic).

## Tests / lint / format / typecheck
- **None configured.** No `pytest`, `ruff`, `black`, or `mypy` in this repo. Validate work via `metrics.json` and visual checks under `outputs/`.