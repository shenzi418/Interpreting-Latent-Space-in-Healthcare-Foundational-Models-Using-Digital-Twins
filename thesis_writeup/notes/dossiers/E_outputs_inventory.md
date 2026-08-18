# Dossier E — `outputs/` Inventory (Reproducibility Appendix Source)

Read-only inventory of everything under `outputs/` (75 top-level entries), built from filesystem mtimes/sizes and `json.load`/`np.load` introspection only — no scripts were run and no repo files besides this one were modified. Intended to let the thesis trace any reported number back to the artifact file that produced it.

## Summary

- **75 top-level entries**: 20 loose files (logs + one JSON) and 55 directories.
- **19 training-run directories** (`args.json`/no-`args.json` + `metrics.json` + `checkpoints/`), plus one more nested one level down (`ptbxl_baselines/linear/ptbxl_baseline/`) — 20 trained models total, spanning three architectural eras: `joint_*` (Mar 2026, dual-head), `exp5/6_3class`+`exp7_*` (Apr–May 2026, shared-head + bottleneck/tier2 variants), `exp8_leadfix_*` (Aug 2026, post lead-order/z-score bugfix).
- **127 latent-export subdirectories** under `outputs/latents/`, all holding a single `latents.npz` with `Z` (features), `P` (probabilities), `Y` (labels) — bottleneck runs add `Z_post_gelu`; the two `exp8_leadfix_medalonly_ptbxl*` exports add `ecg_id`.
- **34 additional analysis-run top-level directories** (`dim_scan*`, `inlp*`, `phase_b2*` ×17, `tier1_eval*`, `latent_analysis*` ×4, `ptbxl_baselines`, `quick_waveform_check`) plus `outputs/analysis/` itself, which holds the **fidelity-audit centrepiece** (`fidelity_audit/`, `circular_geometry/`) and the domain/probe/lead-swap diagnostics (`domain_signal/`, `probe_map/`, `leadperm_sweep/`, `leadswap_diag/`, `label_shift/`).
- **19 log-like files** at the top level (`_log_*.txt`, `_t*.log`, `stage3_retry.log`) plus `outputs/logs/` (2 files) and a handful of per-analysis run logs (`inlp_lowK/run_log_*.txt`, `phase_b2/_smoke_log.txt`).
- **Total size on disk: 34.76 GB.** Three `exp7`/`exp8` full-backbone runs (`exp7_baseline`, `exp7_ccmmd`, `exp8_leadfix_baseline`) each exceed 2 GB of checkpoints; `ptbxl_baselines/linear/ptbxl_baseline/checkpoints/` alone is ~3 GB (24 full-backbone `.pth` snapshots).
- **Date range: 2026-01-05 (`quick_waveform_check`) to 2026-08-17.** The bulk of activity clusters in three bursts: **Mar 12–13** (`joint_*`), **Apr 16 – May 26** (`exp5`–`exp7` family, dim-scan, INLP, Tier-1/Tier-2), and **Aug 10–13** (`exp8_leadfix_*` retrain after the lead-order/z-score bugfix, fidelity audit, floor audit). A handful of `outputs/analysis/fidelity_audit/` and `circular_geometry/floor_audit.json` files carry an **mtime of 2026-08-17**, four days after the declared 2026-08-13 experimental freeze — flagged below, cause not determined (see Caveats).
- All 19 top-level training runs and all 127 latent exports are cross-domain (MedalCare↔PTB-XL); every training run's `metrics.json` reports `test.medalcare.*` and `test.ptbxl.*` blocks in parallel, confirming the shared-head cross-domain evaluation convention described in `.claude/rules/experiments.md`.

---

## 1. Training runs (`args.json`/`metrics.json`/`checkpoints/` present)

All 19 are shared- or dual-head classifiers trained on MedalCare+PTB-XL jointly (except `exp8_leadfix_medalonly`, MedalCare-only). `exp5_3class`, `exp6_3class`, `exp7_baseline`, `exp7_baseline_norm`, `exp7_ccmmd`, `joint_adapter_cls`, `joint_adapter_mmd`, `joint_baseline` **predate the `args.json` convention** — their config is inferable only from `run_id`/checkpoint naming, not stored.

### 1a. Config (from `args.json` where present)

| run_id | created | modified | size on disk | best epoch / max ckpt epoch (of configured `epochs`) | batch | lr | λ_mmd | bottleneck K | medalcare_only | seed | base checkpoint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `exp5_3class` | 2026-05-03 | 2026-05-03 | 1.3GB | 13/13 (of —, no args.json) | — | — | — | — | — | — | — |
| `exp6_3class` | 2026-05-03 | 2026-05-03 | 1.3GB | 13/13 (of —) | — | — | — | — | — | — | — |
| `exp7_baseline` | 2026-04-16 | 2026-04-16 | 2.5GB | 29/29 (of —) | — | — | — | — | — | — | — |
| `exp7_baseline_norm` | 2026-04-17 | 2026-04-17 | 2.1GB | 30/30 (of —) | — | — | — | — | — | — | — |
| `exp7_bottleneck_K16` | 2026-05-17 | 2026-05-17 | 1.2GB | 17/17 (of 20) | 128 | 0.001 | — | 16 | — | 42 | `outputs\exp7_baseline\checkpoints\linear_best.pt` |
| `exp7_bottleneck_K64` | 2026-05-17 | 2026-05-17 | 713.2MB | 10/10 (of 20) | 128 | 0.001 | — | 64 | — | 42 | `outputs\exp7_baseline\checkpoints\linear_best.pt` |
| `exp7_bottleneck_K256` | 2026-05-17 | 2026-05-17 | 717.7MB | 9/9 (of 20) | 128 | 0.001 | — | 256 | — | 42 | `outputs\exp7_baseline\checkpoints\linear_best.pt` |
| `exp7_ccmmd` | 2026-04-16 | 2026-04-16 | 2.5GB | 29/29 (of —) | — | — | — | — | — | — | — |
| `exp7_tier2_K64_A_5050` | 2026-05-26 | 2026-05-26 | 119.1MB | 11 (of 15) | 128 | 0.001 | — | 64 | — | 42 | `outputs\exp7_baseline\checkpoints\linear_best.pt` (λ_cls=0.5, λ_bio=0.5, adapter_trainable=true) |
| `exp7_tier2_K64_B_bioonly` | 2026-05-26 | 2026-05-26 | 119.0MB | 6 (of 15) | 128 | 0.001 | — | 64 | — | 42 | same base (λ_cls=0.0, λ_bio=1.0) |
| `exp8_leadfix_K64` | 2026-08-11 | 2026-08-11 | 1.3GB | 18/18 (of 20) | 128 | 0.001 | — | 64 | — | 42 | `outputs\exp8_leadfix_baseline\checkpoints\linear_best.pt` |
| `exp8_leadfix_baseline` | 2026-08-11 | 2026-08-11 | 2.1GB | 30/30 (of 30) | 128 | 0.0001 | 0.0 | — | — | 42 | `checkpoint\12_lead_ECGFounder.pth` |
| `exp8_leadfix_ccmmd` | 2026-08-11 | 2026-08-11 | 1.8GB | 15/15 (of 30) | 128 | 0.0001 | 0.1 | — | — | 42 | `checkpoint\12_lead_ECGFounder.pth` |
| `exp8_leadfix_dual` | 2026-08-11 | 2026-08-11 | 1.5GB | 15/15 (of 20) | 128 | 0.0001 | 0.0 | — | — | 42 | `checkpoint\12_lead_ECGFounder.pth` |
| `exp8_leadfix_globalz` | 2026-08-11 | 2026-08-11 | 1.7GB | 14/14 (of 30) | 128 | 0.0001 | 0.0 | — | — | 42 | `checkpoint\12_lead_ECGFounder.pth` |
| `exp8_leadfix_medalonly` | 2026-08-11 | 2026-08-11 | 1.2GB | 11/11 (of 30) | 128 | 0.0001 | 0.0 | — | **True** | 42 | `checkpoint\12_lead_ECGFounder.pth` |
| `joint_adapter_cls` | 2026-03-12 | 2026-03-13 | 2.2GB | 27/27 (of —) | — | — | — | — | — | — | — |
| `joint_adapter_mmd` | 2026-03-13 | 2026-03-13 | 1.6GB | 14/14 (of —) | — | — | — | — | — | — | — |
| `joint_baseline` | 2026-03-12 | 2026-03-12 | 1.7GB | 18/18 (of —) | — | — | — | — | — | — | — |

Notes: `exp8_leadfix_globalz` is the global-z-score ablation named in `data-pipeline.md` (`per_lead_norm=False`). `exp8_leadfix_dual` reintroduces the dual-head architecture at Exp-8 lead-fix state (per `stage3_retry.log`'s header: "dual-head at the same 3-class label space — restores the 2×2"). All `exp8_leadfix_*` runs carry `_argv`/`_written_at` provenance fields in `args.json` (not present pre-Aug); `_written_at` matches the checkpoint-mtime span closely (see §5 Compute evidence).

### 1b. Checkpoints and headline test metrics (from `metrics.json.best`)

`checkpoints/` filenames encode `checkpoint_{epoch}_{val_score}.pth`; every run also has a `linear_best.pt` (the file every `--checkpoint` flag consumes, per `experiments.md`). Metric columns are `test.medalcare.{f1,roc_auc}` / `test.ptbxl.{f1,roc_auc}` read verbatim off `best.test.medalcare` / `best.test.ptbxl` — **printed as found, not interpreted**.

| run_id | n checkpoints (incl. `linear_best.pt`) | best-val ckpt file | test medalcare F1 | test medalcare AUC | test ptbxl F1 | test ptbxl AUC | `metrics.json` top-level keys |
|---|---|---|---|---|---|---|---|
| `exp5_3class` | 11 | `checkpoint_13_0.8423.pth` | 0.8667 | 0.9950 | 0.7860 | 0.9438 | run_id, metrics_requested, evaluations, best |
| `exp6_3class` | 11 | `checkpoint_13_0.8433.pth` | 0.8659 | 0.9948 | 0.7870 | 0.9438 | run_id, metrics_requested, evaluations, best |
| `exp7_baseline` | 21 | `checkpoint_29_0.8587.pth` | 0.9166 | 0.9956 | 0.7876 | 0.9351 | run_id, metrics_requested, evaluations, best |
| `exp7_baseline_norm` | 18 | `checkpoint_30_0.8582.pth` | 0.9381 | 0.9980 | 0.7931 | 0.9340 | run_id, metrics_requested, evaluations, best |
| `exp7_bottleneck_K16` | 10 | `checkpoint_17_0.8663.pth` | 0.9440 | 0.9978 | 0.7904 | 0.9384 | run_id, config, metrics_requested, evaluations, best, completed_at |
| `exp7_bottleneck_K64` | 6 | `checkpoint_10_0.8736.pth` | 0.9371 | 0.9971 | 0.7950 | 0.9405 | run_id, config, metrics_requested, evaluations, best, completed_at |
| `exp7_bottleneck_K256` | 6 | `checkpoint_9_0.8783.pth` | 0.9429 | 0.9981 | 0.7941 | 0.9393 | run_id, config, metrics_requested, evaluations, best, completed_at |
| `exp7_ccmmd` | 21 | `checkpoint_29_0.8589.pth` | 0.9161 | 0.9954 | 0.7869 | 0.9346 | run_id, metrics_requested, evaluations, best |
| `exp7_tier2_K64_A_5050` | 1 (only `linear_best.pt`) | — | 0.9887 | 0.9999 | 0.7893 | 0.9303 | run_id, config, bio_channels, z_score_stats, metrics_requested, evaluations, best, completed_at |
| `exp7_tier2_K64_B_bioonly` | 1 (only `linear_best.pt`) | — | 0.1875 | 0.3019 | 0.2774 | 0.3721 | same keys — this arm (λ_cls=0, bio-loss-only) collapses classification, as expected |
| `exp8_leadfix_K64` | 11 | `checkpoint_18_0.9023.pth` | 0.9521 | 0.9967 | 0.8319 | 0.9435 | run_id, config, metrics_requested, evaluations, best, completed_at |
| `exp8_leadfix_baseline` | 18 | `checkpoint_30_0.8881.pth` | 0.9275 | 0.9975 | 0.8280 | 0.9404 | run_id, metrics_requested, evaluations, best |
| `exp8_leadfix_ccmmd` | 15 | `checkpoint_15_0.8797.pth` | 0.9289 | 0.9972 | 0.8234 | 0.9380 | run_id, metrics_requested, evaluations, best |
| `exp8_leadfix_dual` | 13 | `checkpoint_15_0.8804.pth` | 0.9269 | 0.9979 | 0.8345 | 0.9451 | run_id, metrics_requested, evaluations, best |
| `exp8_leadfix_globalz` | 14 | `checkpoint_14_0.8791.pth` | 0.9104 | 0.9950 | 0.8234 | 0.9392 | run_id, metrics_requested, evaluations, best |
| `exp8_leadfix_medalonly` | 10 | `checkpoint_11_0.9230.pth` | 0.9268 | 0.9980 | **0.4567** | **0.6267** | run_id, metrics_requested, evaluations, best — PTB-XL numbers collapse because this encoder never saw PTB-XL gradients (MedalCare-only training; matches the CLAUDE.md medalonly-encoder discussion) |
| `joint_adapter_cls` | 18 | `checkpoint_27_0.6622.pth` | 0.6793 | 0.9814 | 0.7065 | 0.9159 | run_id, metrics_requested, evaluations, best |
| `joint_adapter_mmd` | 13 | `checkpoint_14_0.6540.pth` | 0.6691 | 0.9816 | 0.7037 | 0.9172 | run_id, metrics_requested, evaluations, best |
| `joint_baseline` | 14 | `checkpoint_18_0.6474.pth` | 0.6624 | 0.9801 | 0.7081 | 0.9167 | run_id, metrics_requested, evaluations, best, physics — this is the only run with a `physics_metrics.json` sidecar (θ-decoding heads; 50 θ-names, `mae_norm`/`mae_raw`/`r2`/`pearson`/`spearman`/`effective_n` per-name arrays, `summary.r2_mean=0.311`, `r2_tiers={strong:26, moderate:4, weak:21}`) |

### 1c. Nested 20th run: `ptbxl_baselines/linear/ptbxl_baseline/`

Single-domain PTB-XL-only baseline, one directory below `outputs/ptbxl_baselines/linear/`. `metrics.json` top keys: `run_id, metrics_requested, evaluations, best`; `best.epoch=29`, `best.primary_metric={name:"f1", value:0.6998}`, `best.test` keys `[accuracy, f1, recall, specificity, precision, brier, roc_auc]` (single-domain — no `medalcare`/`ptbxl` split). 24 `checkpoint_*.pth` files + `linear_best.pt`, each ~126.9MB (full backbone, not linear-probe-only weights) — this single subdirectory is **~3.0GB**, the single largest checkpoint stash in `outputs/`.

---

## 2. Latent exports (`outputs/latents/`, 127 subdirectories)

Every subdirectory holds exactly one `latents.npz`. Universal schema: `Z` = encoder features (N×D), `P` = model output probabilities (N×n_classes), `Y` = ground-truth multi-hot labels (N×n_classes: 3 for the shared 3-class label space post-Exp7, 5 for PTB-XL's native label count, 8 for MedalCare's native 8-class count). Bottleneck/tier2 exports (`K16`/`K64`/`K256`) additionally store `Z_post_gelu` (post-activation bottleneck features, same shape as `Z`). The two `exp8_leadfix_medalonly_ptbxl*` exports (the "all-folds" evaluation set used for the n=4324 Track-3 scoring) additionally store `ecg_id` (int64) for row-level traceability.

| subdir | mtime | array shapes (`latents.npz`) |
|---|---|---|
| `exp1_ptbxl` | 2026-03-13 | Z(2198,1024):f32, P(2198,5):f32, Y(2198,5):f32 |
| `exp4_ptbxl` | 2026-03-13 | Z(2198,1024):f32, P(2198,5):f32, Y(2198,5):f32 |
| `exp5_3class_medalcare` | 2026-05-04 | Z(2386,1024):f32, P(2386,3):f32, Y(2386,8):f32 |
| `exp5_3class_medalcare_train` | 2026-05-05 | Z(12019,1024):f32, P(12019,3):f32, Y(12019,8):f32 |
| `exp5_3class_ptbxl` | 2026-05-04 | Z(2198,1024):f32, P(2198,3):f32, Y(2198,5):f32 |
| `exp5_medalcare` | 2026-03-13 | Z(2386,1024):f32, P(2386,8):f32, Y(2386,8):f32 |
| `exp5_ptbxl` | 2026-03-13 | Z(2198,1024):f32, P(2198,5):f32, Y(2198,5):f32 |
| `exp6_3class_medalcare` | 2026-05-04 | Z(2386,1024):f32, P(2386,3):f32, Y(2386,8):f32 |
| `exp6_3class_medalcare_train` | 2026-05-05 | Z(12019,1024):f32, P(12019,3):f32, Y(12019,8):f32 |
| `exp6_3class_ptbxl` | 2026-05-04 | Z(2198,1024):f32, P(2198,3):f32, Y(2198,5):f32 |
| `exp6_medalcare` | 2026-03-13 | Z(2386,1024):f32, P(2386,8):f32, Y(2386,8):f32 |
| `exp6_ptbxl` | 2026-03-13 | Z(2198,1024):f32, P(2198,5):f32, Y(2198,5):f32 |
| `exp7_bottleneck_K16_medalcare_test` | 2026-05-17 | Z(2386,16), Z_post_gelu(2386,16), P(2386,3), Y(2386,8) — all f32 |
| `exp7_bottleneck_K16_medalcare_test_inlp` | 2026-08-11 | same shapes (INLP-projected) |
| `exp7_bottleneck_K16_medalcare_train` | 2026-05-17 | Z/Z_post_gelu(12019,16), P(12019,3), Y(12019,8) |
| `exp7_bottleneck_K16_medalcare_train_inlp` | 2026-08-11 | same shapes (INLP-projected) |
| `exp7_bottleneck_K16_medalcare_val` | 2026-05-17 | Z/Z_post_gelu(2434,16), P(2434,3), Y(2434,8) |
| `exp7_bottleneck_K16_medalcare_val_inlp` | 2026-08-11 | same shapes (INLP-projected) |
| `exp7_bottleneck_K16_ptbxl_test` | 2026-05-17 | Z/Z_post_gelu(2198,16), P(2198,3), Y(2198,5) |
| `exp7_bottleneck_K16_ptbxl_test_inlp` | 2026-08-11 | same shapes (INLP-projected) |
| `exp7_bottleneck_K16_ptbxl_train` | 2026-05-17 | Z/Z_post_gelu(17418,16), P(17418,3), Y(17418,5) |
| `exp7_bottleneck_K16_ptbxl_train_inlp` | 2026-08-11 | same shapes (INLP-projected) |
| `exp7_bottleneck_K16_ptbxl_val` | 2026-05-17 | Z/Z_post_gelu(2183,16), P(2183,3), Y(2183,5) |
| `exp7_bottleneck_K16_ptbxl_val_inlp` | 2026-08-11 | same shapes (INLP-projected) |
| `exp7_bottleneck_K256_medalcare_test` | 2026-05-17 | Z/Z_post_gelu(2386,256), P(2386,3), Y(2386,8) |
| `exp7_bottleneck_K256_medalcare_test_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K256_medalcare_train` | 2026-05-17 | Z/Z_post_gelu(12019,256), P(12019,3), Y(12019,8) |
| `exp7_bottleneck_K256_medalcare_train_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K256_medalcare_val` | 2026-05-17 | Z/Z_post_gelu(2434,256), P(2434,3), Y(2434,8) |
| `exp7_bottleneck_K256_medalcare_val_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K256_ptbxl_test` | 2026-05-17 | Z/Z_post_gelu(2198,256), P(2198,3), Y(2198,5) |
| `exp7_bottleneck_K256_ptbxl_test_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K256_ptbxl_train` | 2026-05-17 | Z/Z_post_gelu(17418,256), P(17418,3), Y(17418,5) |
| `exp7_bottleneck_K256_ptbxl_train_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K256_ptbxl_val` | 2026-05-17 | Z/Z_post_gelu(2183,256), P(2183,3), Y(2183,5) |
| `exp7_bottleneck_K256_ptbxl_val_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K64_medalcare_test` | 2026-05-17 | Z/Z_post_gelu(2386,64), P(2386,3), Y(2386,8) |
| `exp7_bottleneck_K64_medalcare_test_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K64_medalcare_test_inlpv2` | 2026-05-24 | same shapes (INLP v2 re-projection) |
| `exp7_bottleneck_K64_medalcare_train` | 2026-05-17 | Z/Z_post_gelu(12019,64), P(12019,3), Y(12019,8) |
| `exp7_bottleneck_K64_medalcare_train_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K64_medalcare_train_inlpv2` | 2026-05-24 | same shapes |
| `exp7_bottleneck_K64_medalcare_val` | 2026-05-17 | Z/Z_post_gelu(2434,64), P(2434,3), Y(2434,8) |
| `exp7_bottleneck_K64_medalcare_val_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K64_medalcare_val_inlpv2` | 2026-05-24 | same shapes |
| `exp7_bottleneck_K64_ptbxl_test` | 2026-05-17 | Z/Z_post_gelu(2198,64), P(2198,3), Y(2198,5) |
| `exp7_bottleneck_K64_ptbxl_test_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K64_ptbxl_test_inlpv2` | 2026-05-24 | same shapes |
| `exp7_bottleneck_K64_ptbxl_train` | 2026-05-17 | Z/Z_post_gelu(17418,64), P(17418,3), Y(17418,5) |
| `exp7_bottleneck_K64_ptbxl_train_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K64_ptbxl_train_inlpv2` | 2026-05-24 | same shapes |
| `exp7_bottleneck_K64_ptbxl_val` | 2026-05-17 | Z/Z_post_gelu(2183,64), P(2183,3), Y(2183,5) |
| `exp7_bottleneck_K64_ptbxl_val_inlp` | 2026-08-11 | same shapes |
| `exp7_bottleneck_K64_ptbxl_val_inlpv2` | 2026-05-24 | same shapes |
| `exp7_ccmmd_medalcare` | 2026-04-16 | Z(2386,1024), P(2386,3), Y(2386,8) — f32 |
| `exp7_ccmmd_medalcare_inlp` | 2026-08-11 | same shapes |
| `exp7_ccmmd_medalcare_train` | 2026-05-05 | Z(12019,1024), P(12019,3), Y(12019,8) |
| `exp7_ccmmd_medalcare_train_inlp` | 2026-08-11 | same shapes |
| `exp7_ccmmd_ptbxl` | 2026-04-16 | Z(2198,1024), P(2198,3), Y(2198,5) |
| `exp7_ccmmd_ptbxl_inlp` | 2026-08-11 | same shapes |
| `exp7_ccmmd_ptbxl_train` | 2026-05-08 | Z(17418,1024), P(17418,3), Y(17418,5) |
| `exp7_ccmmd_ptbxl_train_inlp` | 2026-08-11 | same shapes |
| `exp7_medalcare` | 2026-04-16 | Z(2386,1024), P(2386,3), Y(2386,8) |
| `exp7_medalcare_inlp` | 2026-08-11 | same shapes |
| `exp7_medalcare_inlpv2` | 2026-05-08 | same shapes |
| `exp7_medalcare_train` | 2026-05-05 | Z(12019,1024), P(12019,3), Y(12019,8) |
| `exp7_medalcare_train_inlp` | 2026-08-11 | same shapes |
| `exp7_medalcare_train_inlpv2` | 2026-05-08 | same shapes |
| `exp7_medalcare_train_unswapped` | 2026-08-10 | same shapes — pre-leadfix diagnostic re-export (aVL/aVF unswapped) |
| `exp7_medalcare_train_unswapped_perlead` | 2026-08-10 | same shapes — unswapped + per-lead z-score diagnostic |
| `exp7_medalcare_unswapped` | 2026-08-10 | Z(2386,1024), P(2386,3), Y(2386,8) — unswapped diagnostic |
| `exp7_medalcare_unswapped_perlead` | 2026-08-10 | same shapes — unswapped + per-lead diagnostic |
| `exp7_norm_medalcare` | 2026-04-17 | Z(2386,1024), P(2386,3), Y(2386,8) |
| `exp7_norm_ptbxl` | 2026-04-17 | Z(2198,1024), P(2198,3), Y(2198,5) |
| `exp7_ptbxl` | 2026-04-16 | Z(2198,1024), P(2198,3), Y(2198,5) |
| `exp7_ptbxl_inlp` | 2026-08-11 | same shapes |
| `exp7_ptbxl_inlpv2` | 2026-05-08 | same shapes |
| `exp7_ptbxl_leadswap` | 2026-08-10 | same shapes — deliberate lead-swap diagnostic export (leadswap_diag) |
| `exp7_ptbxl_train` | 2026-05-08 | Z(17418,1024), P(17418,3), Y(17418,5) |
| `exp7_ptbxl_train_inlp` | 2026-08-11 | same shapes |
| `exp7_ptbxl_train_inlpv2` | 2026-05-08 | same shapes |
| `exp7_tier2_K64_A_5050_medalcare_test` | 2026-05-26 | Z/Z_post_gelu(2386,64), P(2386,3), Y(2386,8) |
| `exp7_tier2_K64_A_5050_medalcare_train` | 2026-05-26 | Z/Z_post_gelu(12019,64), P(12019,3), Y(12019,8) |
| `exp7_tier2_K64_A_5050_medalcare_val` | 2026-05-26 | Z/Z_post_gelu(2434,64), P(2434,3), Y(2434,8) |
| `exp7_tier2_K64_A_5050_ptbxl_test` | 2026-05-26 | Z/Z_post_gelu(2198,64), P(2198,3), Y(2198,5) |
| `exp7_tier2_K64_A_5050_ptbxl_train` | 2026-05-26 | Z/Z_post_gelu(17418,64), P(17418,3), Y(17418,5) |
| `exp7_tier2_K64_A_5050_ptbxl_val` | 2026-05-26 | Z/Z_post_gelu(2183,64), P(2183,3), Y(2183,5) |
| `exp7_tier2_K64_B_bioonly_medalcare_test` | 2026-05-26 | Z/Z_post_gelu(2386,64), P(2386,3), Y(2386,8) |
| `exp7_tier2_K64_B_bioonly_medalcare_train` | 2026-05-26 | Z/Z_post_gelu(12019,64), P(12019,3), Y(12019,8) |
| `exp7_tier2_K64_B_bioonly_medalcare_val` | 2026-05-26 | Z/Z_post_gelu(2434,64), P(2434,3), Y(2434,8) |
| `exp7_tier2_K64_B_bioonly_ptbxl_test` | 2026-05-26 | Z/Z_post_gelu(2198,64), P(2198,3), Y(2198,5) |
| `exp7_tier2_K64_B_bioonly_ptbxl_train` | 2026-05-26 | Z/Z_post_gelu(17418,64), P(17418,3), Y(17418,5) |
| `exp7_tier2_K64_B_bioonly_ptbxl_val` | 2026-05-26 | Z/Z_post_gelu(2183,64), P(2183,3), Y(2183,5) |
| `exp8_leadfix_baseline_medalcare_test` | 2026-08-11 | Z(2386,1024), P(2386,3), Y(2386,8) |
| `exp8_leadfix_baseline_medalcare_train` | 2026-08-11 | Z(12019,1024), P(12019,3), Y(12019,8) |
| `exp8_leadfix_baseline_medalcare_val` | 2026-08-11 | Z(2434,1024), P(2434,3), Y(2434,8) |
| `exp8_leadfix_baseline_ptbxl_test` | 2026-08-11 | Z(2198,1024), P(2198,3), Y(2198,5) |
| `exp8_leadfix_baseline_ptbxl_train` | 2026-08-11 | Z(17418,1024), P(17418,3), Y(17418,5) |
| `exp8_leadfix_baseline_ptbxl_val` | 2026-08-11 | Z(2183,1024), P(2183,3), Y(2183,5) |
| `exp8_leadfix_ccmmd_medalcare_test` | 2026-08-11 | Z(2386,1024), P(2386,3), Y(2386,8) |
| `exp8_leadfix_ccmmd_medalcare_train` | 2026-08-11 | Z(12019,1024), P(12019,3), Y(12019,8) |
| `exp8_leadfix_ccmmd_medalcare_val` | 2026-08-11 | Z(2434,1024), P(2434,3), Y(2434,8) |
| `exp8_leadfix_ccmmd_ptbxl_test` | 2026-08-11 | Z(2198,1024), P(2198,3), Y(2198,5) |
| `exp8_leadfix_ccmmd_ptbxl_train` | 2026-08-11 | Z(17418,1024), P(17418,3), Y(17418,5) |
| `exp8_leadfix_ccmmd_ptbxl_val` | 2026-08-11 | Z(2183,1024), P(2183,3), Y(2183,5) |
| `exp8_leadfix_dual_medalcare_test` | 2026-08-11 | Z(2386,1024), P(2386,3), Y(2386,8) |
| `exp8_leadfix_dual_medalcare_train` | 2026-08-11 | Z(12019,1024), P(12019,3), Y(12019,8) |
| `exp8_leadfix_dual_medalcare_val` | 2026-08-11 | Z(2434,1024), P(2434,3), Y(2434,8) |
| `exp8_leadfix_dual_ptbxl_test` | 2026-08-11 | Z(2198,1024), P(2198,3), Y(2198,5) |
| `exp8_leadfix_dual_ptbxl_train` | 2026-08-11 | Z(17418,1024), P(17418,3), Y(17418,5) |
| `exp8_leadfix_dual_ptbxl_val` | 2026-08-11 | Z(2183,1024), P(2183,3), Y(2183,5) |
| `exp8_leadfix_globalz_medalcare_test` | 2026-08-11 | Z(2386,1024), P(2386,3), Y(2386,8) |
| `exp8_leadfix_globalz_medalcare_train` | 2026-08-11 | Z(12019,1024), P(12019,3), Y(12019,8) |
| `exp8_leadfix_globalz_medalcare_val` | 2026-08-11 | Z(2434,1024), P(2434,3), Y(2434,8) |
| `exp8_leadfix_globalz_ptbxl_test` | 2026-08-11 | Z(2198,1024), P(2198,3), Y(2198,5) |
| `exp8_leadfix_globalz_ptbxl_train` | 2026-08-11 | Z(17418,1024), P(17418,3), Y(17418,5) |
| `exp8_leadfix_globalz_ptbxl_val` | 2026-08-11 | Z(2183,1024), P(2183,3), Y(2183,5) |
| `exp8_leadfix_K64_medalcare_test` | 2026-08-11 | Z/Z_post_gelu(2386,64), P(2386,3), Y(2386,8) |
| `exp8_leadfix_K64_medalcare_train` | 2026-08-11 | Z/Z_post_gelu(12019,64), P(12019,3), Y(12019,8) |
| `exp8_leadfix_K64_medalcare_val` | 2026-08-11 | Z/Z_post_gelu(2434,64), P(2434,3), Y(2434,8) |
| `exp8_leadfix_K64_ptbxl_test` | 2026-08-11 | Z/Z_post_gelu(2198,64), P(2198,3), Y(2198,5) |
| `exp8_leadfix_K64_ptbxl_train` | 2026-08-11 | Z/Z_post_gelu(17418,64), P(17418,3), Y(17418,5) |
| `exp8_leadfix_K64_ptbxl_val` | 2026-08-11 | Z/Z_post_gelu(2183,64), P(2183,3), Y(2183,5) |
| `exp8_leadfix_medalonly_medalcare_test` | 2026-08-11 | Z(2386,1024), P(2386,3), Y(2386,8) |
| `exp8_leadfix_medalonly_medalcare_train` | 2026-08-11 | Z(12019,1024), P(12019,3), Y(12019,8) |
| `exp8_leadfix_medalonly_ptbxl` | 2026-08-11 | Z(21799,1024), P(21799,3), Y(21799,5), **ecg_id(21799,):i64** — full PTB-XL "all folds" export (n=21799) used for the 9.9× power scoring |
| `exp8_leadfix_medalonly_ptbxl_test` | 2026-08-12 | Z(2198,1024), P(2198,3), Y(2198,5), **ecg_id(2198,):i64** — fold-10-only subset of the above |

Note the `*_inlp`/`*_inlpv2` naming: array shapes are identical to their non-INLP counterpart in every case (INLP is a post-hoc orthogonal projection, dimension-preserving) — the projection matrices themselves live separately in `outputs/inlp*/**/projection.npz`, not in the latent exports.

---

## 3. Analysis outputs

### 3.1 `outputs/analysis/` — the fidelity-audit / domain-signal / lead-swap centrepiece

Two loose top-level JSON files plus 7 subdirectories, 72 files total, modified 2026-08-10 to **2026-08-17**.

**Top-level:**
| file | keys | note |
|---|---|---|
| `scaler_domain_ab.json` | arm_a, arm_b, in_domain_identical, rows | the scaler-dichotomy A/B test referenced repeatedly in CLAUDE.md (target vs strict scaler) |
| `scaler_domain_ab_target_vs_target_pool.json` | arm_a, arm_b, in_domain_identical, rows | companion run: target scaler vs target-pool scaler |

**`circular_geometry/`** (8 files) — the circular/cyclic-order geometry line of analysis (S(b), floor audit, acuity):
| file | keys / note |
|---|---|
| `acuity_stratified_transport.json` | encoder, n_medalcare_fit_rows, anchor_deg, strata, power_matched_chronic_vs_acute |
| `circular_geometry.json` | anchors_deg, n_folds, n_perm, encoders (91KB — main geometry sweep) |
| `cyclic_order_test.json` | one key per exp8_leadfix_* encoder (6 encoders) — the S(b) statistic now flagged **RETRACTED at non-integer b** per the banner in `analysis/cyclic_order_test.py` |
| `floor_audit.json` | anchors_deg, floors, rescored, cross_domain_below_floor, cyclic_order_M_to_P, acuity_strata, granularity_renormalised, synthetic_prior_vs_floor, null_identity — **source of the 0.29216 PTB-XL / 0.09319 MedalCare constant-predictor floor** cited throughout CLAUDE.md |
| `floor_audit_report.txt` | plain-text rendering of the above (34KB→13KB after stripping) |
| `granularity_control.json` | one key per exp8_leadfix_* encoder |
| `latent_geometry_correspondence.json` | n_boot, n_perm, encoders |
| `synthetic_prior_value.json` | n_grid, n_repeat, encoders (51KB) |

**`domain_signal/`** (34 files) — one-off domain-alignment / MMD / C2ST / subspace mechanism probes, almost all keyed to `exp8_leadfix_baseline` (a few to `exp7_baseline`/`exp7_ccmmd`/`exp5_3class` for the pre-leadfix comparison):
| file | keys / note |
|---|---|
| `b2_coral_exp5_3class.json`, `b2_coral_exp7_baseline.json`, `b2_coral_exp7_ccmmd.json` | config, n_ptbxl, ci_clears_null, treatments — CORAL alignment probe, one per pre-leadfix encoder |
| `coral_controls.json` | one key per {exp8_leadfix_baseline, exp8_leadfix_ccmmd, exp7_ccmmd} |
| `coral_exp8_leadfix_baseline.json` | run, coral_rank, k_inlp, conditions |
| `direction_agreement_exp8_leadfix_baseline.json` | run, n_dims, participation_ratio, random_reference, classes |
| `domain_mechanism_exp8_leadfix_baseline.json` | run, k_removed, representations |
| `domain_mechanism_replication.json` | k, seed, subsample, runs |
| `domain_rank_exp8_leadfix_baseline.json` | run, whiten_rank, euclidean, whitened |
| `domain_signal_structure_exp7.json` | run, per_dim_auroc, inlp_curve, support_overlap (13.9KB) |
| `inlp_controls_exp7.json` | run, k, ambient_dim, baseline, inlp, random_control, verdict_alignment, verdict_transfer |
| `nonlinear_c2st_exp8_leadfix_baseline.json` / `..._euclidean.json` | run, (geometry), whiten_rank, n_inlp, curve — GBDT C2ST curve, matches the "GBDT C2ST 9e-5 spread" discovery |
| `nonlinear_structure_exp8_leadfix_baseline.json` | run, k_removed, gbdt_auc, n_positive_importance, top_share, top_coords |
| `quantile_controls.json`, `quantile_exp8_leadfix_baseline.json` | quantile-based domain probes |
| `rank_reconciliation_exp8_leadfix_baseline.json` | run, k, implementations |
| `subspace_identity_exp7.json`, `subspace_mechanism_exp8_leadfix_baseline*.json` (×3: base/whitened/whitened128), `subspace_theta_exp7.json` | subspace/whitened-subspace domain-mechanism probes |
| `tradeoff_frontier_exp7*.json/.png`, `tradeoff_frontier_exp8_leadfix_baseline*.json/.png` (×4 variants: base, gbdt, knee) | run, ks, rows / c2st_detector — the K-sweep tradeoff-frontier plots (6 PNGs total) |
| `transfer_control_bootstrap_exp8_leadfix_baseline.json`, `transfer_control_exp8_leadfix_baseline*.json` (×2), `transfer_reality_exp8_leadfix_baseline.json` | transfer-control bootstrap CIs and the "transfer reality" macro-F1 check |
| `whitened_frontier_exp8_leadfix_baseline.json` | run, whiten_rank, n_dims, n_inlp, ks, curve, k_at_chance |

**`fidelity_audit/`** (6 files) — **the thesis centrepiece** (F1/F2/F3 = the three audit stages):
| file | keys / note |
|---|---|
| `f1_fidelity.json` (73.7KB) | meta, rank_agreement, marginal_vs_informativeness, missingness, missingness_holm_sig_features, axis, features, lists, prediction_check, addendum |
| `f1_fidelity_out.txt` | plain-text log, first line: "F1 per-feature informativeness-fidelity audit (seed=20260813, n_boot=500)" |
| `f2_blocks.json` (19.8KB) | block_structure, pre_stated_prediction, floors, per_block, efficiency, paired_cross, paired_in, axis2_floor_check, block_eta2_features, spearman |
| `f2_blocks_out.txt` | first line: "F2: per-block fidelity-audit prediction test" |
| `f3_repair.json` (25KB) | exp8_leadfix_medalonly, exp8_leadfix_dual, axis_reconciliation, verdict — the channel-repair attempt |
| `f3_repair_out.txt` | plain-text log |

**`label_shift/`** (1 file): `label_shift.json` — keys `shared_classes, medalcare_test, ptbxl_test_fold10, js_divergence_bits, d_js, joint_error_floor_perfect_alignment, observed, caveat, reference`.

**`leadperm_sweep/`** (1 file): `leadperm_sweep.json` (39.9KB) — keys `metadata, rows`; the 77-cell lead-permutation sweep (identity + 66 transpositions + 10 random perms) discussed at length in CLAUDE.md; produced by `_t42_sweep.log` (see §5).

**`leadswap_diag/`** (5 files): `c2st_leadfix.json` (as_shipped, leadfix), `c2st_leadfix_trained.json` (one key per 5 exp8_leadfix_* encoders), `pipeline_a_leadswap.json`, `pipeline_a_medalcare_unswapped.json`, `pipeline_a_medalcare_unswapped_perlead.json` (all: best_C, cv_scores, in_domain, ptbxl_orig, ptbxl_leadswap [+ med_train/med_test for the unswapped variants]) — the diagnostic set behind the 2026-08-10 lead-swap bugfix.

**`probe_map/`** (13 files): `probe_control.json`/`.csv` and `probe_map.json` (metadata/results; `probe_map.json` is 313KB — the largest single JSON in the whole tree) plus per-encoder `probe_map_exp8_leadfix_{K64,baseline,ccmmd,dual,globalz}.csv` (17-column: `kind,lead,feature,status,n_train,n_in,n_cross,n_cross_primary4,alpha,rho_in,rho_cross_source,rho_cross_target_pool,rho_cross_source_primary4,d_rho,perm_p_cross_source,r2_in,scaler_sign_disagreement`) and 5 `grid_exp8_leadfix_*.png` heatmaps.

### 3.2 Other top-level analysis directories

| dir | mtime range | size | contents |
|---|---|---|---|
| `dim_scan` | 2026-05-11 | 702KB | PCA K-sweep on pre-leadfix latents (`exp7_baseline`/`exp7_ccmmd` × combined/medalcare/ptbxl, 6 `*_summary_*.json` each with `config, pca_mode, n_train_pool, n_test_pool_valid, n_mi_train, n_mi_test, explained_variance_ratio_cumsum_at_K, ks, per_K, k_star`), `kstar_table.json` (list of 6 `{config, pca_mode, k_star}`), 3 `frontier_*.png` |
| `dim_scan_exp8` | 2026-08-11 | 700KB | same sweep re-run on `exp8_leadfix_{baseline,ccmmd}` post-leadfix, plus a `nonlinear_c2st.json` (seed, subsample, note, rows) not present in the pre-leadfix `dim_scan/` |
| `exp7_latent_analysis` | 2026-04-16 | 1.7MB | `exp7_analysis.json` (experiment, samples, domain_alignment, class_separability, fisher_lda) + 3 PCA PNGs (class / class×domain overlay / domain) |
| `exp7_ccmmd_latent_analysis` | 2026-04-16 | 1.7MB | same structure, ccMMD arm |
| `exp7_norm_latent_analysis` | 2026-04-17 | 1.7MB | same structure, `baseline_norm` arm |
| `exp7_full_evaluation` | 2026-04-16→**2026-05-04** | 17.7KB | `full_evaluation.json` (configs, kmeans, cross_domain_transfer, cosine_similarity) + 2 `dtw_validation_exp7*.json` (added ~3 weeks later than the main file) |
| `exp7_full_evaluation_norm` | 2026-04-17 | 11.4KB | `full_evaluation.json` only, `baseline_norm` arm |
| `exp7_latent_analysis_inlp` | 2026-05-08 | 9.2KB | `full_evaluation.json`, post-INLP-projection arm |
| `latent_analysis` | 2026-03-13→2026-05-18 | 6.9MB | earliest/broadest latent-analysis dir: `LATENT_SPACE_ANALYSIS_REPORT.{md,html}` (10KB md / 4.1MB html), `DISCUSSION_NOTES.md` (24KB), `class_separability_metrics.json` (linear_probes, fisher_lda, silhouette), `domain_alignment_metrics.json` (keyed `"Exp 5"`/`"Exp 6"`), 9 PNGs (PCA by domain/class, scree plots, silhouette scores, Fisher-LDA heatmap) |
| `inlp` | 2026-05-08→2026-08-11 | 26.0MB | `inlp_summary.json` (exp7_baseline, exp7_ccmmd), `inlp_summary_inlpv2.json` (exp7_baseline only), `preflight.json` (primary, conditional, b2_inputs); per-encoder subdirs `exp7_baseline/`, `exp7_baseline_inlpv2/`, `exp7_ccmmd/`, each with `iteration_log.json` (14 keys incl. `metrics_orig`, `metrics_inlp`, `iterations`, `ptp_residual`), `pca_before_after.png`, `projection.npz` (8.4MB each — the actual INLP projection matrices) |
| `inlp_lowK` | 2026-05-24→2026-08-11 | 2.8MB | low-K (bottleneck) INLP: `concept5_classifier.json` (probe_in_domain_metrics, main, ablations, gt_upper_bound), `concept5_*_per_territory.csv` ×2, `concept5_predictions.npz`, `eval_decoding_lowK.json` (K16/K64/K256/K1024), `inlp_lowK_summary.json` (asym_K16/asym_K64/asym_K256/sym_K64), `tier2_A_5050.json`/`tier2_B_bioonly.json` (run_prefix, label, shapes, alignment, probes, pipeline_a), 3 `run_log_*.txt`, per-K subdirs (`exp7_bottleneck_K{16,64,256}`, `exp7_bottleneck_K64_inlpv2`) each with `iteration_log.json` (22 keys, richer than the plain `inlp/` version — adds `delta_c2st_inlp`, `delta_c2st_rand_control`, `delta_c2st_inlp_specific`), `pca_before_after.png`, `projection.npz` |
| `tier1_eval` | 2026-05-17 | 542KB | `cross_config_table.json`/`.md`, 4 `*_summary.json` (one per {exp7_baseline_ref, exp7_bottleneck_K16, K64, K256}, each 7 keys: config, K, shapes, alignment, class_structure, mechanism, anatomy_pipeline_a — ~105KB each), `frontier_tier1.png` |
| `tier1_eval_exp8` | 2026-08-11 | 526KB | same structure, re-run on the 5 `exp8_leadfix_*` encoders (K64, baseline, ccmmd, dual, globalz) |
| `ptbxl_baselines` | 2026-03-12 | 3.0GB | nested training run, see §1c |
| `quick_waveform_check` | 2026-01-05 | 2.5MB | earliest artifact in the whole tree — 10 `waveforms/medalcare_*.png` sanity-check plots, pre-dates all experiment numbering |

**`phase_b2*` family (17 directories)** — the Track-3 cross-domain θ-territory decoding diagnostic (Phase B2). All share a common file pattern: `cm_*.png` (confusion matrices: overall, `_8c`, `_A_2c`, `_A_4c`, `_B_cal_4c`, `_B_hard_4c`, one set per encoder evaluated in that run), `hist_predphi_by_territory_*.png`, `polar_*.png` (where present), and JSON result files `cross_domain.json` / `cross_domain_4c_pipelineA.json` / `cross_domain_4c_pipelineB.json` / `in_domain.json` / `in_domain_8c.json` (all: `metadata, results[, primary_endpoints_not_computed, primary_endpoints_family_size, primary_endpoints_holm]`). Per-directory summary:

| dir | mtime range | size | encoders evaluated | JSON files present | note |
|---|---|---|---|---|---|
| `phase_b2` | 2026-05-07→2026-08-11 | 2.2MB | exp5_3class, exp6_3class, exp7_baseline, exp7_ccmmd | cross_domain, cross_domain_4c_pipelineA/B, in_domain, in_domain_8c | pre-leadfix baseline; `_smoke_in.json`/`_smoke_log.txt` also present (earliest smoke test, 2026-05-07) |
| `phase_b2_smoke` | 2026-08-11 | 418KB | exp7_baseline only | cross_domain, cross_domain_4c_pipelineA/B, in_domain, in_domain_8c | smoke test ahead of the exp8 rerun |
| `phase_b2_smoke_paired` | 2026-08-11 | 114KB | exp8_leadfix_baseline | cross_domain_4c_pipelineA, in_domain | paired-test smoke |
| `phase_b2_exp8` | 2026-08-11 | 3.0MB | K64, baseline, ccmmd, dual, globalz (5 encoders) | cross_domain, cross_domain_4c_pipelineA/B, in_domain, in_domain_8c, in_domain_K64 | main post-leadfix run; has `_snapshot_4cfg/` and `_snapshot_K64/` sub-snapshots preserving earlier cuts |
| `phase_b2_exp8_poolscaler` | 2026-08-11 | 1.7MB | same 5 encoders | cross_domain, cross_domain_4c_pipelineA/B, in_domain | pool-scaler ablation; has `_snapshot_2cfg/` |
| `phase_b2_exp8_srcscaler` | 2026-08-11 | 1.1MB | baseline, ccmmd | cross_domain, cross_domain_4c_pipelineA/B, in_domain, in_domain_8c | source-scaler ablation |
| `phase_b2_exp8_tgtscaler` | 2026-08-11 | 653KB | baseline, ccmmd | cross_domain, cross_domain_4c_pipelineA/B, in_domain | target-scaler ablation (the "strict scaler" arm) |
| `phase_b2_exp8_spatial54` | 2026-08-11 | 1.6MB | K64, baseline, ccmmd, dual, globalz | cross_domain, cross_domain_4c_pipelineA/B, in_domain | 54-feature spatial control, 5 encoders |
| `phase_b2_exp8_spatial54_measscaler` | 2026-08-11→2026-08-12 | 1.6MB | same 5 | same 4 | measured-scaler variant |
| `phase_b2_exp8_spatial54_srcscaler` | 2026-08-11→2026-08-12 | 1.6MB | same 5 | same 4 | source-scaler variant |
| `phase_b2_exp8_spatial54_poolscaler` | 2026-08-11 | 274KB | baseline only | none (PNGs only) | pool-scaler variant, baseline-only |
| `phase_b2_inlp` | 2026-05-14 | 820KB | exp7_baseline, exp7_ccmmd | cross_domain, cross_domain_4c_pipelineA/B, in_domain, in_domain_8c | post-INLP-projection arm |
| `phase_b2_mi_stage` | 2026-08-11 | 11.3KB | — | `mi_stage_control.json` (metadata, results) only | the MI-stage/acuity control referenced in the 2026-08-13 "consistent with" retraction |
| `phase_b2_baseline_fold10_measscaler_paired` | 2026-08-12 | 161.5KB | exp8_leadfix_baseline, fold-10 only | cross_domain, cross_domain_4c_pipelineA, in_domain | matches `_log_baseline_fold10_meas_paired.txt`; last line reports `in_domain_4c +0.1367 [+0.1020,+0.1731] p=0.0001 n=1200` |
| `phase_b2_medalonly_fold10_target` | 2026-08-12 | 163.4KB | exp8_leadfix_medalonly, fold-10 | cross_domain, cross_domain_4c_pipelineA, in_domain | target scaler, matches `_log_medalonly_fold10.txt` (`in_domain_4c +0.1523 [+0.1156,+0.1862] p=0.0001 n=1200`) |
| `phase_b2_medalonly_fold10_measscaler` | 2026-08-12 | 159.2KB | exp8_leadfix_medalonly, fold-10 | cross_domain, cross_domain_4c_pipelineA, in_domain | measured-scaler variant, identical headline number to the target-scaler run above |
| `phase_b2_medalonly_allfolds_target` | 2026-08-11→2026-08-12 | 362.2KB | exp8_leadfix_medalonly, all 10 folds | cross_domain, cross_domain_4c_pipelineA/B, in_domain | the n=4324 all-folds Track-3 scoring run |
| `phase_b2_medalonly_allfolds_target_pool_measured` | 2026-08-12 | 361.7KB | exp8_leadfix_medalonly, all folds | cross_domain, cross_domain_4c_pipelineA/B, in_domain | pool-measured-scaler variant of the above |

---

## 4. Log files

First/last non-empty line and mtime for every top-level log-like file plus the nested run logs. All files use `Windows-1252`/UTF-8-with-BOM-ish encodings inconsistently (three files — `_t1b_sweep_log.txt`, `phase_b2/_smoke_log.txt`, the three `inlp_lowK/run_log_*.txt` — carry stray BOM/UTF-16 artifacts in their first bytes; content is otherwise readable ASCII).

| file | mtime | size | first line | last line |
|---|---|---|---|---|
| `_log_baseline_fold10_meas_paired.txt` | 2026-08-12 02:24 | 7.2KB | `====...====` | `exp8_leadfix_baseline  in_domain_4c  +0.1367  [+0.1020,+0.1731]  0.0001  1200  577` |
| `_log_exp8_medalonly.txt` | 2026-08-11 22:14 | 165.5KB | `Using device: cuda:0` | `[Exp 7] Shared-head training complete.` |
| `_log_feat_allfolds.txt` | 2026-08-11 22:04 | 223.8KB | `[PTB-XL folds [1..10]] total=21799, MI=5469, processing=5469` | `Wrote summary -> data\ecg_features_spatial_ptbxl_allfolds_summary.json` |
| `_log_medalonly_fold10.txt` | 2026-08-12 02:00 | 7.1KB | `====...====` | `exp8_leadfix_medalonly  in_domain_4c  +0.1523  [+0.1156,+0.1862]  0.0001  1200  595` |
| `_log_medalonly_fold10_meas.txt` | 2026-08-12 02:06 | 7.2KB | `====...====` | same as above, measured-scaler variant, identical headline value |
| `_log_medalonly_target.txt` | 2026-08-12 00:10 | 8.8KB | `====...====` | `exp8_leadfix_medalonly  logreg_l2  0.582  0.330 [0.314,0.344]  0.0001  0.338 [0.322,0.352]  0.0001  -0.008  0.602  0.603` |
| `_log_medalonly_target_pool_measured.txt` | 2026-08-12 01:53 | 8.9KB | `====...====` | `exp8_leadfix_medalonly  logreg_l2  0.582  0.326 [0.312,0.340]  0.0001  0.342 [0.326,0.355]  0.0001  -0.016  0.614  0.613` |
| `_log_probe_control.txt` | 2026-08-11 21:22 | 26.3KB | `=== control: exp8_leadfix_baseline ===` | `[done] wrote outputs\analysis\probe_map\probe_control.json` |
| `_log_probe_map.txt` | 2026-08-11 21:16 | 22.1KB | `=== exp8_leadfix_baseline ===` | `[done] wrote outputs\analysis\probe_map\probe_map.json` |
| `_log_smoke_medalonly.txt` | 2026-08-11 21:54 | 12.4KB | `Using device: cuda:0` | `[Exp 7] Shared-head training complete.` |
| `_log_spatial54_meas.err.txt` | 2026-08-11 21:44 | 0B | *(empty)* | *(empty)* |
| `_log_spatial54_meas.txt` | 2026-08-12 01:29 | 19.2KB | `====...====` | `exp8_leadfix_K64  logreg_l2  0.482  0.199 [0.163,0.234]  0.7694  ...` |
| `_log_spatial54_pool.txt` | 2026-08-11 21:34 | 6.2KB | `====...====` | `-> rho_eps_max (Logistic)` |
| `_log_spatial54_src.txt` | 2026-08-12 00:08 | 19.0KB | `====...====` | `exp8_leadfix_K64  logreg_l2  0.482  0.210 [0.174,0.250]  0.4835  ...` |
| `_t1b_sweep_log.txt` | 2026-05-17 21:10 | 431.0KB | `=== SKIP TRAIN K=256 (already done: ...)` | `=== 2026-05-17 21:10:43 === SWEEP COMPLETE` |
| `_t3_pipeline_a_dump.txt` | 2026-05-18 05:17 | 18.1KB | `############## outputs\phase_b2\cross_domain_4c_pipelineA.json ##############` | `Inferior [5, 223]` |
| `_t42_smoke.log` | 2026-08-11 12:07 | 905B | `device=cuda  encoder=exp8_leadfix_baseline` | `wrote ...leadperm_sweep.json (3 cells, 0.7 min)` |
| `_t42_sweep.log` | 2026-08-11 12:26 | 7.8KB | `device=cuda  encoder=exp8_leadfix_baseline` | `wrote ...leadperm_sweep.json (77 cells, 16.2 min)` |
| `stage3_retry.log` | 2026-08-11 06:12 | 15.7KB | `### exp8_leadfix_dual -- dual-head at the same 3-class label space -- restores the 2x2` | `-> reports\stage3_logs\stage3_status.json` |
| `logs/domain_mechanism_replication.log` | 2026-08-11 05:36 | 1.8KB | `====...====` | `wrote outputs\analysis\domain_signal\domain_mechanism_replication.json` |
| `logs/post_phase_b2_tgtscaler.log` | 2026-08-11 06:18 | 8.9KB | `====...====` | `exp8_leadfix_ccmmd  logreg_l2  0.579  0.308 [0.264,0.356]  0.0002  ...` |
| `phase_b2/_smoke_log.txt` | 2026-05-07 02:37 | 4.3KB | `[targets] train MI=5347, test MI=1200` | `exp7_baseline  0.223 [0.193,0.248]  1.0000  0.265 [0.237,0.292]  0.312  0.404` |
| `inlp_lowK/run_log_asym.txt` | 2026-05-24 22:50 | 25.0KB | `====...====` | `Wrote outputs\inlp_lowK\inlp_lowK_summary.json` |
| `inlp_lowK/run_log_eval.txt` | 2026-05-24 23:27 | 10.0KB | `[load] PTB-XL primary 4c subset: n=438; counts={'Inferior':196,'Anteroseptal':168,'Anterolateral':42,'Inferolateral':32}` | `Wrote outputs\inlp_lowK\eval_decoding_lowK.json` |
| `inlp_lowK/run_log_sym.txt` | 2026-05-24 22:54 | 21.0KB | `[merge] loaded existing summary with keys=['asym_K16','asym_K64','asym_K256']` | `Wrote outputs\inlp_lowK\inlp_lowK_summary.json` |

None of the log files contain embedded `YYYY-MM-DD HH:MM:SS` timestamps except `_t1b_sweep_log.txt` (21 timestamps, `2026-05-17 20:20:57` → `2026-05-17 21:10:43`, 50 min span). Wall-clock duration for the other runs is only recoverable from file mtimes or (for training runs) checkpoint-file mtime spans — see §5.

---

## 5. Compute evidence

No log anywhere in `outputs/` records a GPU model string (no `GeForce`/`RTX`/`NVIDIA-SMI` hits in any text log; the only binary-file hits are torch pickle metadata inside `.pth` checkpoints and PNG bytes, not readable device names). Every device string found is the generic PyTorch string `cuda` / `cuda:0` — a specific GPU model cannot be confirmed from these artifacts alone.

**Training-run duration** (most reliable source: min/max mtime across `checkpoints/*.pth`, since one file is written per improving epoch):

| run | duration (first→last checkpoint mtime) | epochs (ckpt count) | device string | source |
|---|---|---|---|---|
| `joint_baseline` | 29.9 min (2026-03-12 22:17→22:47) | 13 | — | checkpoint mtimes |
| `joint_adapter_cls` | 50.5 min (2026-03-12 23:11→2026-03-13 00:01) | 17 | — | checkpoint mtimes |
| `joint_adapter_mmd` | 25.5 min (2026-03-13 00:11→00:37) | 12 | — | checkpoint mtimes |
| `exp7_baseline` | 41.8 min (2026-04-16 00:50→01:32) | 20 | — | checkpoint mtimes |
| `exp7_ccmmd` | 42.1 min (2026-04-16 01:53→02:35) | 20 | — | checkpoint mtimes |
| `exp7_baseline_norm` | 43.0 min (2026-04-17 00:03→00:46) | 17 | — | checkpoint mtimes |
| `exp5_3class` | 18.0 min (2026-05-03 05:58→06:16) | 10 | — | checkpoint mtimes |
| `exp6_3class` | 17.9 min (2026-05-03 08:44→09:02) | 10 | — | checkpoint mtimes |
| `exp7_bottleneck_K256` | 9.4 min (2026-05-17 20:04→20:13) | 5 | — | checkpoint mtimes |
| `exp7_bottleneck_K64` | 10.8 min (2026-05-17 20:24→20:35) | 5 | — | checkpoint mtimes |
| `exp7_bottleneck_K16` | 19.0 min (2026-05-17 20:45→21:04) | 9 | — | checkpoint mtimes |
| `_t1b_sweep_log.txt` (K256+K64+K16 train+export sweep, wraps the 3 rows above) | **50 min total** (2026-05-17 20:20:57→21:10:43, 21 embedded timestamps) | max epoch mention = 20 | `Device: cuda` | log body |
| `exp7_tier2_K64_A_5050` | ~0 min (single checkpoint write, `linear_best.pt` only) | 0 numbered ckpts / best=11 of 15 | — | checkpoint mtimes |
| `exp7_tier2_K64_B_bioonly` | ~0 min (single checkpoint write) | 0 numbered ckpts / best=6 of 15 | — | checkpoint mtimes |
| `exp8_leadfix_baseline` | 45.8 min (2026-08-11 02:52→03:37; `_written_at`=02:50:14) | 17 | — (see `_log_exp8_medalonly.txt` for a same-day sibling run's device string) | checkpoint mtimes + args.json `_written_at` |
| `exp8_leadfix_ccmmd` | 22.6 min (2026-08-11 03:43→04:05; `_written_at`=03:41:32) | 14 | — | checkpoint mtimes + `_written_at` |
| `exp8_leadfix_dual` | 27.9 min (2026-08-11 04:28→04:56; `_written_at`=04:25:53) | 12 | — | checkpoint mtimes + `_written_at` |
| `exp8_leadfix_globalz` | 22.5 min (2026-08-11 05:06→05:28; `_written_at`=05:04:34) | 13 | — | checkpoint mtimes + `_written_at` |
| `exp8_leadfix_K64` | 24.3 min (2026-08-11 05:42→06:07) | 10 | — | checkpoint mtimes |
| `exp8_leadfix_medalonly` | 11.3 min (2026-08-11 21:57→22:08; `_written_at`=21:56:32) | 9 | `Using device: cuda:0` (`_log_exp8_medalonly.txt`, epoch mentions max=15) / `_log_smoke_medalonly.txt` shows a 1-epoch smoke run immediately prior | checkpoint mtimes + `_written_at` + sibling log |
| `ptbxl_baselines/linear/ptbxl_baseline` | not separately logged; 24 checkpoints, mtime range within 2026-03-12 | 24 | — | checkpoint mtimes |

**Analysis-run duration** (from embedded duration strings, the only ones found anywhere in the tree):

| run | duration hint | cells / scope | device string | source file |
|---|---|---|---|---|
| leadperm sweep (smoke) | 0.7 min | 3 cells | `device=cuda  encoder=exp8_leadfix_baseline` | `_t42_smoke.log` |
| leadperm sweep (full) | 16.2 min | 77 cells (identity + 66 transpositions + 10 random) | `device=cuda  encoder=exp8_leadfix_baseline` | `_t42_sweep.log`, output → `outputs/analysis/leadperm_sweep/leadperm_sweep.json` |

`stage3_retry.log` mentions "epoch" 57 times (max value 20) across its body but embeds no timestamps of its own — it is an orchestration/dispatch log (header: "exp8_leadfix_dual — dual-head at the same 3-class label space — restores the 2×2") that re-summarizes child-run epoch counts rather than logging a single run's device/duration; cross-reference against the `exp8_leadfix_dual` checkpoint-mtime row above for the actual wall-clock figure.

---

## Caveats

- **`args.json` provenance is incomplete for 8 of 19 training runs** (`exp5_3class`, `exp6_3class`, `exp7_baseline`, `exp7_baseline_norm`, `exp7_ccmmd`, `joint_adapter_cls`, `joint_adapter_mmd`, `joint_baseline`) — these predate the convention documented in `experiments.md`; their hyperparameters are only inferable from `run_id` naming and `checkpoints/` epoch/val-score filenames, not from a stored config.
- **`_written_at` appears only in the six `exp8_leadfix_*` runs' `args.json`** — no earlier run stores a run-start timestamp, so checkpoint-file mtimes are the only duration proxy for anything before Aug 2026.
- **No GPU model string exists anywhere in `outputs/`** — every device reference found is the generic `cuda`/`cuda:0`; if the thesis needs a specific GPU model for the reproducibility appendix, it must come from outside this directory (e.g. `env-ECGFounder.yml`, a shell history, or direct recall).
- **Seven files under `outputs/analysis/fidelity_audit/` and one under `outputs/analysis/circular_geometry/` (`floor_audit.json`) carry mtimes of 2026-08-17**, four days after the declared 2026-08-13 experimental freeze in CLAUDE.md. `outputs/*` is fully gitignored (except `quick_waveform_check/`), so this is not a `git checkout` mtime artifact; cause not established by this read-only inventory (could be a verification re-run, a filesystem re-touch by another process, or an editor/antivirus scan) — flagged for the user to confirm before citing these files' *content* dates in the thesis. Their sizes match the description already logged in CLAUDE.md for the 2026-08-13 fidelity audit, so content is presumptively the same; this was not diffed byte-for-byte.
- **Table 1a/1b hyperparameter and metric values are transcribed verbatim from the JSON, not re-derived or sanity-checked** against the claims elsewhere in CLAUDE.md — per the task brief, this dossier is a file inventory, not a re-analysis. Where a printed number is surprising (e.g. `exp8_leadfix_medalonly` test-PTB-XL F1=0.4567/AUC=0.6267, or `exp7_tier2_K64_B_bioonly`'s classification collapse to F1=0.1875), that is flagged inline but the underlying JSON was not re-opened for a second pass.
