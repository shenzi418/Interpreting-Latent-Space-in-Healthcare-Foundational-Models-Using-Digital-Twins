*Apply when interpreting `outputs/<run_id>/` artifacts, comparing Exp 5/6/7 runs, or designing a new run.*

## Run-ID glossary
- `exp5_3class` — dual-head, no alignment, 3-class shared label space.
- `exp6_3class` — dual-head + class-conditional MMD (λ=0.1).
- `exp7_baseline` — shared-head, 1024-d linear head, no alignment.
- `exp7_ccmmd` — shared-head + class-conditional MMD (λ=0.1).
- `exp7_bottleneck_K{16,64,256,1024}` — shared-head with `Linear(1024,K) → GELU → Linear(K, n_classes)` head.
- `dim_scan` — PCA K-sweep on existing latents (post-hoc, no retrain).
- `*_inlp` suffix — same config, INLP-aligned latents under `outputs/latents/<id>_<domain>_inlp/`.

## The 2×2 ablation (Exp 7 era)

|                   | dual-head       | shared-head      |
|-------------------|-----------------|------------------|
| **no alignment**  | `exp5_3class`   | `exp7_baseline`  |
| **ccMMD (λ=0.1)** | `exp6_3class`   | `exp7_ccmmd`     |

Headline: shared-head architecture (NOT the 3-class relabeling) drives cross-domain transfer. LR M→P AUC ≈ 0.59 dual-head vs ≈ 0.76 shared-head. See `reports/exp7_progress_report.md` §8.

## Output artifact contract per `<run_id>`
- `outputs/<run_id>/checkpoints/linear_best.pt` — best-val-score checkpoint. This
  is the file every `--checkpoint` flag wants. The name is the default of
  `--best-checkpoint-name` (`scripts/finetune_multilabel.py:1583`), so a run that
  overrode that flag has a different filename — check `args.json`.
  ⚠ There is **no `best_model.pt`**; that name appeared in these rules until
  2026-08-12 and never matched anything on disk.
- `outputs/<run_id>/checkpoints/checkpoint_{epoch}_{score}.pth` — best-so-far
  snapshots, one per improving epoch. Useful for reading off the selected epoch
  and its val score without opening `metrics.json`.
- `outputs/<run_id>/args.json` — full argparse Namespace as JSON.
- `outputs/<run_id>/metrics.json` — final test metrics (per-class + macro).
- `outputs/<run_id>/per_class_metrics.csv` — same data, tabular.
- Latent export step writes separately to `outputs/latents/<run_id>_<domain>/latents.npz`.

## Shared label space (Exp 7)
- 3 classes: NORM, MI, CD.
- `MEDALCARE_REMAP`, `PTBXL_REMAP`, `remap_labels()` live at `scripts/finetune_multilabel.py:52-83`.
- Excluded: MedalCare `lae`/`fam`, PTB-XL `STTC`/`HYP` (no clean clinical correspondence).