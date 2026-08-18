# GPU-hours estimate for the Sustainability declaration (computed 2026-08-18 00:xx)

**Hardware** (nvidia-smi on the workstation that ran everything): NVIDIA GeForce RTX 5080, 16 303 MiB.
Board power limit ≈ 360 W (vendor spec); fine-tuning a frozen backbone + adapters typically draws well below that — 250 W is used as a working average.

**Method**: no run logged wall-clock time, so each run's duration is bounded by the file-modification span of its
`outputs/<run_id>/` directory (args.json / first checkpoint → metrics.json / last checkpoint). Spans are upper bounds
for the run itself (they include evaluation and export) but exclude runs whose folders were later deleted or overwritten.

| Group | Runs on disk | Dates | Sum of spans |
|---|---|---|---|
| ptbxl_baselines/linear, joint_baseline, joint_adapter_cls, joint_adapter_mmd | 4 | 2026-03-12/13 | ≈ 2.9 h |
| exp7_baseline, exp7_ccmmd, exp7_baseline_norm | 3 | 2026-04-16/17 | ≈ 2.2 h |
| exp5_3class, exp6_3class | 2 | 2026-05-03 | ≈ 0.8 h |
| exp7_bottleneck_K16/K64/K256, exp7_tier2_K64_A/B | 5 | 2026-05-17, 05-26 | ≈ 1.5 h |
| exp8_leadfix_{baseline,ccmmd,dual,globalz,K64,medalonly} | 6 | 2026-08-11 | ≈ 3.3 h |
| **fine-tuning total on disk** | **20** | | **≈ 9.3 h** (upper-bound spans; the model-only time is lower) |

**Inference**: 127 latent-export directories under `outputs/latents/` (≈ 2.0 GB); export bursts last a few minutes each
(e.g. seven exports 08-11 05:08–05:11) → ≈ 2–3 GPU-h in total. The 77-cell lead-permutation sweep (PTB-XL test forward
passes) and the C2ST/GBDT diagnostics add ≈ 1 GPU-h (GBDT is CPU). All audit / transfer / repair / circular-statistics
analyses are CPU-only from cached latents.

**Not on disk**: the first months' exploratory runs (zero-shot baseline, frozen-encoder linear/MLP heads, early MMD runs,
Oct 2025 – Feb 2026; the earliest surviving file in outputs/ is dated 2026-01-05). Allowing the same order of magnitude
again as the surviving fine-tuning runs is a conservative allowance.

**Estimate to quote**: fine-tuning ≈ 10 GPU-h (surviving) + inference ≈ 3–4 GPU-h + deleted early runs ≈ 10 GPU-h
→ **≈ 25 GPU-hours, upper estimate ≈ 40 GPU-hours** on one RTX 5080; at ≈ 250 W average that is **≈ 6–10 kWh**
(≈ 1–2 kg CO2e at ≈ 0.2 kg CO2e/kWh, roughly the recent UK grid average — quote as an order of magnitude only).
