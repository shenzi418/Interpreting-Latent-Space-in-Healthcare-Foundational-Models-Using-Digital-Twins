# Reproducing the thesis results

There are three levels of reproduction, cheapest first. Table and figure numbers refer to the thesis.

## Level 1 — Verify the reported numbers (no computation)

Every number in the thesis traces to a frozen file under `results/`; the README maps each thesis table and figure to its file. The files under `results/verify/` are the outputs of independent re-implementations (`analysis/*_verify.py`) of the three central analyses, run against the same inputs.

## Level 2 — Re-run the central analyses (CPU only, fresh clone, no downloads)

The extracted feature tables, the θ target files and the label manifests are committed under `data/`, so the two analyses that need no encoder run directly from a clone:

```
python analysis/fidelity_audit.py        # audit: thesis Tables 4.9-4.10, Figure 4.2  -> f1_fidelity.json
python analysis/block_transfer.py        # block transfer: Table 4.11, Figure 4.4    -> f2_blocks.json
python analysis/fidelity_audit_verify.py # independent re-implementation of the audit
python analysis/block_transfer_verify.py # independent re-implementation of block transfer
```

Outputs are written under `outputs/analysis/fidelity_audit/` and should match the frozen copies in `results/` (seeds are fixed in the script headers; the analyses are deterministic). All remaining analyses read exported latent representations, which are large and not committed — complete Level 3 first.

## Level 3 — The full pipeline (external data + one GPU)

### 3.1 Environment

Python 3.10. Install PyTorch 2.9.1 with CUDA 12.8 first (see pytorch.org for the platform command), then `pip install -r requirements.txt`. Fine-tuning fits on a single 16 GB consumer GPU (roughly 10 GPU-hours for all six runs); every analysis step is CPU.

### 3.2 External inputs

All three are public and are expected at these repo-relative paths (every script also takes explicit path flags — see `--help`):

| Input | Source | Expected path |
|---|---|---|
| PTB-XL v1.0.3 | PhysioNet (CC BY 4.0) | `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/` |
| MedalCare-XL | Zenodo (CC BY 4.0) | `MedalCare-XL/` |
| ECGFounder 12-lead weights | the authors' release (MIT) | `checkpoint/12_lead_ECGFounder.pth` |

### 3.3 Data preparation (optional — all outputs already committed under `data/`)

The committed manifests, θ targets and feature tables were produced by, in order: `scripts/prepare_medalcare.py`, `scripts/make_splits.py`, `scripts/add_medalcare_splits.py`, `scripts/build_medalcare_isch_targets.py`, `scripts/build_ptbxl_mi_subclass.py`, and the two feature extractors `scripts/extract_ecg_features_neurokit2.py` and `scripts/extract_ecg_features_spatial.py`. Re-running them should reproduce the committed files. **Do not regenerate `data/medalcare_filtered_manifest_dataset_split.csv` ad hoc**: every committed NPZ assumes its row order.

### 3.4 Fine-tuning (the six encoders of thesis Table 3.3)

These are the exact recorded invocations (seed 42 throughout):

```
python scripts/finetune_multilabel.py --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --shared-head --run-id exp8_leadfix_baseline
python scripts/finetune_multilabel.py --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --shared-head --run-id exp8_leadfix_ccmmd --lambda-mmd 0.1 --class-cond-mmd
python scripts/finetune_multilabel.py --epochs 20 --batch-size 128 --num-workers 0 --seed 42 --dual-head-shared-labels --lambda-mmd 0 --run-id exp8_leadfix_dual
python scripts/finetune_multilabel.py --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --shared-head --run-id exp8_leadfix_globalz --global-z
python scripts/finetune_multilabel.py --epochs 30 --batch-size 128 --num-workers 0 --seed 42 --shared-head --medalcare-only --run-id exp8_leadfix_medalonly
python scripts/finetune_bottleneck.py  --checkpoint outputs/exp8_leadfix_baseline/checkpoints/linear_best.pt --bottleneck-dim 64 --epochs 20 --seed 42 --run-id exp8_leadfix_K64
```

Each run writes `outputs/<run-id>/{args.json, metrics.json, per_class_metrics.csv, checkpoints/linear_best.pt}`. The `metrics.json` files are thesis Table 4.1; the frozen copies are `results/encoder_metrics/*.json`. Note that retraining on different hardware or driver versions may not be bit-identical (GPU nondeterminism); the frozen results are the reference.

### 3.5 Latent export

One directory per run × domain × split, named `outputs/latents/<run-id>_<domain>_<split>/`. Template (repeat for each domain and for splits train/val/test):

```
python scripts/export_latents.py --checkpoint outputs/exp8_leadfix_medalonly/checkpoints/linear_best.pt \
  --model-type auto --use-adapter --dataset medalcare --split test \
  --outdir outputs/latents/exp8_leadfix_medalonly_medalcare_test
```

Two pairings matter: pass `--global-z` on the MedalCare exports of the `exp8_leadfix_globalz` run (it must match how the checkpoint was trained), and export the K64 run with `scripts/export_bottleneck_latents.py` (same `--checkpoint/--dataset/--split/--outdir` interface).

### 3.6 Analyses

With the latents in place, the remaining thesis results regenerate as follows:

| Script | Thesis location | Frozen result file |
|---|---|---|
| `scripts/_diag_c2st_leadfix.py` | Table 4.2 (loading conditions) | `results/c2st_loader_conditions.json` |
| `analysis/domain_mechanism_replication.py` | Table 4.3 (over-determination) | — |
| `scripts/_t42_leadperm_sweep.py` | Table 4.4 (77-cell sweep) | `results/lead_permutation_sweep.json` |
| `analysis/phase_b2_infarct_decoding.py` + `scripts/_audit_paired_grid.py` | Tables 4.6–4.7 (decision rule; twelve-cell grid) | — |
| `analysis/circular_geometry.py`, `analysis/floor_audit.py`, `analysis/cyclic_order_test.py` | §4.3 floors, anchors, angular structure | `results/floor_audit.json`, `results/floor_free_scores.txt` |
| `analysis/acuity_stratified_transport.py` | §4.4 acuity test | (feeds `floor_audit.json`) |
| `analysis/channel_repair.py` (+ `_verify`) | Table 4.12, Figure 4.5 (repair) | `results/f3_repair.json` |
| `analysis/inlp_alignment.py`, `analysis/nonlinear_c2st.py`, `analysis/label_shift_bound.py` | §4.2 and Appendix A.4 | — |

The supplementary grids of Table 4.8 and Appendix A.3 are frozen at `results/alpha_sweep_grid.txt` and `results/alpha_sweep_supplement.txt`.

## Determinism and provenance

Fine-tuning uses seed 42; the audit family uses seed 20260813 with the bootstrap draw counts stated in each script header; the P1 pipeline derives every random stream from a hash of the cell identity, so any cell's intervals are independent of which other cells are computed. The tagged commit `freeze-2026-08-13` is the analysis state that produced the frozen results; the pre-correction pipeline state is preserved at the tag `pre-leadfix`.
