# Interpreting the Latent Space in ECG Digital-Twin Foundation Models

Code and frozen results for an MRes thesis (Artificial Intelligence and Machine Learning, Imperial College London; supervisor: Dr Marta Varela). The project adapts a pre-trained ECG foundation model (ECGFounder) to a cardiac digital twin (MedalCare-XL) and to a real clinical cohort (PTB-XL), asks whether infarct territory can be read from the latent space within and across the synthetic-to-real gap, and introduces an informativeness-fidelity audit that measures, per ECG feature, how much territory information each domain carries.

## Repository contents

| Path | Contents |
|---|---|
| `scripts/` | Data preparation, split generation, fine-tuning, latent export, feature extraction |
| `analysis/` | The analysis pipelines: territory decoding, circular geometry and floors, the fidelity audit, block transfer, channel repair, alignment diagnostics |
| `losses/`, `metrics/`, `net1d.py`, `finetune_model.py`, `util.py` | Model code: backbone, adapters, MMD losses, evaluation metrics |
| `data/` | Seeded split manifests and simulation-parameter (θ) target files |
| `results/` | Frozen result files behind the thesis numbers (see tracing table below) |

The two datasets are public and are not redistributed here: PTB-XL v1.0.3 (PhysioNet, CC BY 4.0) and MedalCare-XL (Zenodo, CC BY 4.0). ECGFounder weights are released by their authors under the MIT licence.

## Tracing thesis numbers to files

Every number in the thesis traces to a result file. The frozen copies live in `results/`:

| Thesis location | File |
|---|---|
| Table 4.1 (encoder classification) | `results/encoder_metrics/*.json` |
| Table 4.2 (loading configurations, C2ST/MMD/kNN) | `results/c2st_loader_conditions.json` |
| Table 4.4 (77-cell lead-permutation sweep) | `results/lead_permutation_sweep.json` |
| Constant-predictor floors and anchor angles (§3.6, §4.3) | `results/floor_audit.json` |
| Table 4.8 and Appendix A.3 (α sweep) | `results/alpha_sweep_grid.txt`, `results/alpha_sweep_supplement.txt` |
| Floor-free scores and nulls (§4.3) | `results/floor_free_scores.txt` |
| Figure 4.1 (cross-domain confusion matrices) | `results/confusion_medalcare_only.json` |
| Tables 4.9–4.10, Figure 4.2 (fidelity audit) | `results/f1_fidelity.json` |
| Table 4.11, Figure 4.3 (block transfer and prediction) | `results/f2_blocks.json` |
| Table 4.12, Figure 4.4 (repair experiments) | `results/f3_repair.json` |
| Independent re-implementation checks | `results/verify/` |

Remaining values (for example the paired latent-versus-control grid of Table 4.7) regenerate from the scripts in `analysis/` and `scripts/`, whose seeds are fixed in their headers.

## Reproducibility

Step-by-step instructions, including the exact fine-tuning commands, are in [REPRODUCING.md](REPRODUCING.md). In short: the tagged commit `freeze-2026-08-13` is the frozen state of the analysis code at the end of the experimental phase; the files in `results/` were produced under it. Fine-tuning uses seed 42 throughout; the audit and repair scripts carry their own seeds and bootstrap draw counts in their headers. Trained checkpoints and exported latent arrays are large and regenerable and are not committed. The two central analyses (the fidelity audit and the block transfer) run CPU-only from the committed files in `data/`, with no dataset download and no training.

## Environment

Python 3.10, PyTorch 2.9 (CUDA 12.8), wfdb, neurokit2, scikit-learn. See `requirements.txt`.
