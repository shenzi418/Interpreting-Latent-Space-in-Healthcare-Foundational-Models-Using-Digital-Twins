# Project history dossier, part 1 — start of project → July 2026

*Compiled 2026-08-18 from `reports/` (the written record), `openspec/` (the dated capability
list) and `git log`. Every number below carries a pointer of the form `path:line` or `path→§`.
Paths are relative to the repo root. **`reports/` is gitignored** — these are owner-local
documents. Part 2 (2026-08-10 audit → experimental freeze) is a separate dossier.*

---

## Executive summary (10 lines)

1. The project ran three eras: **infrastructure + baselines** (Nov 2025 – Jan 2026), **controlled adapter experiments Exp 1–7** (Mar – early May 2026), and **the interpretability pivot: Phase B/B2, INLP, Track 1/2/3, Tier 1/2** (May 2026), then a **June–July gap** with zero repo activity.
2. The load-bearing negative was established early and never moved: a domain classifier separates synthetic from real latents perfectly (**C2ST AUROC = 1.000**) under every training objective, every post-hoc projection and every bottleneck dimension tested.
3. The load-bearing positive was that a frozen ECGFounder latent linearly encodes MedalCare's simulator parameters in-domain (φ, z, size, transmurality), while the same read-out fails on PTB-XL.
4. Four independent read-outs (raw Z, INLP-aligned Z, a 5-concept vector, a multi-task K=64 bottleneck) hit the *same* cross-domain wall — that repetition is what eventually motivated the fidelity audit.
5. The 2026-04-29 supervisor meeting converted the project from "domain adaptation" to "digital twins as an interpretability oracle"; the 2026-05-13 brief added the pre-registered Track 3 decision rule.
6. **The 2026-08-10 repo audit made every pre-`exp8` number provisional**: MedalCare aVL/aVF transposed, global vs per-lead z-score, the PTB-XL 3-class filter keeping STTC instead of CD, and an AUC column transposition (`2026-08-10_repo_audit_and_rerun_plan.md:18-23`).
7. Four *conclusions*, not just magnitudes, were retracted outright — the φ sign-flip story, the Inferior→Anteroseptal "domain-gap" collapse, "per-lead normalisation hurts", and the Pipeline-B in-domain calibrator (`…rerun_plan.md:118-127`).
8. The C2ST=1.0 wall **survives** the corrections (re-measured on existing latents, no retraining: `…rerun_plan.md:28-32`), so the alignment dead-end is the safest thing to carry into the thesis.
9. Compute was modest throughout: one RTX 5080, ~19 trained runs, longest single run ≈1 h, whole Tier-1 sweep ≈50 min, Tier-2 pair ≈40 min including exports and evaluation.
10. The May code that produced the May headline numbers **was never committed until 2026-08-11** (`…rerun_plan.md:168-179`; tag `pre-leadfix`) — a reproducibility fact the Declarations chapter should state plainly.

---

## 1. Dated timeline

Status key: **KEPT** = still quotable as written · **SUPERSEDED-by-leadfix** = produced on pre-`exp8`
data/code, direction may survive but the magnitude must be requoted from an `exp8_*` run ·
**RETRACTED** = the conclusion itself is wrong · **abandoned** = discontinued, reason given.

| When | What was built / run | Headline number(s) | Source | Status today |
|---|---|---|---|---|
| 2025-03 → 2025-06 | Upstream ECGFounder code inherited (commits by NickLJLee, Shenda Hong) | — | `git log --date=short` (2025-03-26, 2025-03-27, 2025-06-04, 2025-06-09) | KEPT — third-party provenance; matters for the Declarations "code reuse" line |
| 2025-11-09 → 11-14 (W1) | Project repo initialised; MedalCare-XL CSV→WFDB consolidation (`scripts/prepare_medalcare.py`), deterministic manifest, `StratifiedGroupKFold` subject-level splits (seed 42), waveform QC plots | byte-identical manifests on re-run | `reports/README_w1.md:41-53`; commits 2025-11-09/12/14 | KEPT — the split artefact (`data/medalcare_filtered_manifest_dataset_split.csv`) is still the source of truth; split integrity re-verified in 2026 (`…rerun_plan.md:104-105`) |
| W1 | **Zero-shot baseline** of the frozen ECGFounder checkpoint on MedalCare 8-class | macro AP **0.164**, Brier 0.348, ROC-AUC 0.515 | `reports/README_w1.md:83` | SUPERSEDED-by-leadfix (MedalCare batches carried the aVL/aVF transposition and the global-scalar z-score) |
| W1 | **Frozen-encoder baselines**: linear head, MLP head, plus one full fine-tune | macro AP **0.813** (linear) / **0.829** (MLP) / **0.923** (full FT) | `reports/README_w1.md:84-86` | SUPERSEDED-by-leadfix. Full fine-tune **abandoned** — every subsequent experiment freezes the encoder (`reports/experiment_report.md:5`) |
| W1 | RBF-MMD loss added (`losses/mmd.py`), `--lambda-mmd`, `--domain-column` | "improved macro AP on ≥4/8 tasks" — no table produced | `reports/README_w1.md:73-75` | **abandoned** — never quantified; subsumed by Exp 6 / ccMMD |
| ~W4 (commit 2025-12-05) | PTB-XL ingestion; shared Net1D encoder with two heads (MedalCare 8-class + PTB-XL 5-superclass); `--joint-datasets medalcare+ptbxl`; frozen-encoder joint training | PTB-XL linear AP **0.782** / MLP **0.791**; joint val AP 0.797 (Medal) / 0.777 (PTB-XL); PTB-XL zero-shot AP **0.184**, ROC-AUC 0.318 | `reports/README_w4.md:14-19` | PTB-XL-only baseline **KEPT** (never touches MedalCare — `…rerun_plan.md:103`); joint arms SUPERSEDED-by-leadfix |
| W4 | **t-SNE of joint latents** — the first look at the latent space | "two largely disjoint clouds… a domain classifier on z would likely have very high AUROC" | `reports/README_w4.md:25` | KEPT as the *origin* of the domain-gap question. Visualisation later switched t-SNE→PCA (`openspec/changes/archive/2026-01-21-update-latent-viz-pca/proposal.md`) |
| 2026-01-21 (OpenSpec batch) | Five capabilities landed: domain-separability probe (logistic domain classifier + AUROC), MMD wired into joint mode, MedalCare split manifest, physics-pipeline bug fixes, PCA visualisation | domain-classifier AUROC becomes a standing metric | `openspec/changes/archive/2026-01-21-*/proposal.md` | KEPT — this probe is the direct ancestor of every later C2ST number |
| W5 (commit 2026-01-22) | **Physics head** — MLP `z → θ̂` over a frozen, ordered **51-dim** θ contract from MedalCare parameter files; masked regression loss, MedalCare batches only; θ z-scored with train-only stats | normalized MAE **0.509** (median 0.452); raw MAE 7853.85; R² mean **0.064** (median 0.328); **21 strong / 5 moderate / 25 weak** of 51 dims | `git show 30057ce:"weekly report/README_w5.md":28-31` (file deleted in the 2026-05-10 restructure) | **abandoned** — the 50-dim `theta_core` was killed by the April parameter audit (only 9 of 50 dims usable) and replaced by the 4-parameter `isch[0]` target set |
| 2026-05-03 (OpenSpec, work earlier) | `ConvAdapter1D` stage-level residual adapters; θ_core refinement; **AP replaced by threshold metrics** (accuracy/F1/recall/precision/specificity + ROC-AUC/Brier) on supervisor + literature grounds | metric suite changes here — this is why W1/W4 report AP and Exp 1–7 report F1 | `openspec/changes/archive/2026-05-03-{add-stage-residual-adapters,refine-physics-eval,update-classification-metrics}/proposal.md` | KEPT — explains the metric discontinuity across the record |
| ~2026-03 → 04 (`experiment_report.html` 2026-03-13; `.md` 2026-04-16) | **Exp 1 / 4 / 5 / 6** — frozen encoder, 30 epochs, best-by-val-F1 | PTB-XL macro-F1 **0.6982 / 0.7081 / 0.7065 / 0.7037**; MedalCare macro-F1 0.6624 / 0.6793 / 0.6691; MMD driven 0.078 → 0.009 during training | `reports/experiment_report.md:31-34,40-42,79` | SUPERSEDED-by-leadfix **and** by filter defect **F** (all PTB-XL metrics were computed on NORM/MI/**STTC**, not CD — `…rerun_plan.md:123`). The *conclusion* — all four sit inside a 1-point F1 band — is KEPT directionally |
| — | Exp 2 (MedalCare adapter pretrain) and Exp 3 (sequential transfer) | never run | `reports/experiment_report.md:99` | **abandoned** — "skipped for now; can be revisited later" |
| 2026-04 | **Exp 7 shared head** (single `Linear(1024,3)` over {NORM, MI, CD}), alternating MedalCare/PTB-XL batches, 312,643 trainable params | MedalCare test F1 **0.9166**, PTB-XL **0.7876**, best epoch 29; LR M→P accuracy **0.373 → 0.655**, AUC 0.555 → **0.759** | `reports/experiment_report.md:50-51`; `reports/exp7_progress_report.md:58,151-156` | SUPERSEDED-by-leadfix + F. The audit explicitly lists "does shared-head > dual-head survive?" as **UNKNOWN** pending the `exp8` rerun (`…rerun_plan.md:134`) |
| 2026-04 | **Phase-A latent battery**: MMD, kNN mixing, C2ST, k-means(3), linear probes, cosine intra/inter, DTW | **C2ST AUROC 1.000 in every config**; ccMMD −13% MMD with zero downstream change; cross-domain same-class cosine **−0.054**; DTW ratio 0.549 (Medal) / 0.881 (PTB-XL) | `reports/exp7_progress_report.md:107-112,144-147,164,173-176` | **C2ST wall KEPT** — re-measured post-fix at 1.0000 (`…rerun_plan.md:28-32`). Everything else SUPERSEDED |
| 2026-04 | Normalisation ablation `exp7_baseline_norm` (per-lead z-score retrain) | "kept C2ST at 1.000 and *worsened* cross-domain transfer 65.5% → 55.7%" | `reports/exp7_progress_report.md:202` | **RETRACTED** — that run still had swapped leads, so the comparison is untested (`…rerun_plan.md:122`) |
| 2026-04-28 | **MedalCare parameter audit** — programmatic parse of ~7,500 `*Parameters.txt` | only **9 of 50** θ dims usable: 5 APD (`max,min,rho_d,v_d,z_d`) + 4 MI-only `isch[0]` (`phi, z, size, rho_eps_max`); ~30 dims zero-variance, 4 class-conditionally missing, 12 unit-inconsistent | `reports/supervisor_meeting_2026_04_29_supplement.md:15-22,28-34,42-47` | KEPT — still the justification for "θ is 4-dimensional". One later correction: `transmural` is a *duplicate* of `rho_eps_max`, so θ has 4 members not 5 (`…rerun_plan.md:191`, m2) |
| **2026-04-29** | **Supervisor meeting — the pivot.** "Domain adaptation" → "digital twins as an interpretability oracle"; 5-claim / 5-figure paper structure proposed; manifold mixup, SSL/SupCon, TCAV, Z→ECG decoder counterfactuals and 50-dim θ CCA all dropped | timeline stated as ~17 weeks technical + 4 weeks writing → mid-September | `reports/supervisor_meeting_2026_04_29.md:17-21,39-49,51,53-61` | KEPT — this is the hinge of the whole narrative |
| 2026-05-04 | **2×2 ablation re-run** requested by supervisor: `exp5_3class`, `exp6_3class` on the *same* 3-class space | dual-head KMeans 0.654 / LR M→P AUC 0.589 vs shared-head 0.733 / **0.759**; label-space coarsening alone changes nothing (0.654 → 0.654) | `reports/exp7_progress_report.md:242-245,257-260,264` | The 2×2 **design** is KEPT; the numbers are SUPERSEDED-by-leadfix + F |
| 2026-05-06 | **Phase B2 in-domain** — Ridge/Logistic probes MedalCare-train MI (n=5,347) → MedalCare-test MI (n=1,200) | φ circular R² **0.47** [0.42, 0.51]; z R² 0.39; size R² 0.24; `rho_eps_max` AUC **0.92**; NK2 baseline 0.07 / 0.00 / 0.02 / 0.66 | `reports/b2_infarct_localization_log.md:45-48` | **Direction KEPT** ("θ is linearly decodable from frozen latents" survives every defect — `…rerun_plan.md:112`); magnitudes SUPERSEDED; every `permutation_p_r2` **RETRACTED** (vacuous null, defect A3 — `…rerun_plan.md:125`) |
| 2026-05-07 | **B2-CD smoke test** — MedalCare φ probe → PTB-XL 3-class territory (n=444; Lateral n=12) | macro-F1 **0.22–0.27**, permutation p = **1.0**; five standardisation variants all in [0.22, 0.26]; reversed φ bins score best (0.34–0.38); cosine LAD↔{Ant, Inf, Lat} = 0.5694 / 0.5719 / 0.5791 | `reports/b2_infarct_localization_log.md:65-91` | **RETRACTED as a domain-gap finding** — the "φ axis is sign-flipped between synth and real" reading is an artifact of the frontal-plane reflection caused by the aVF/aVL swap (`…rerun_plan.md:119`) |
| 2026-05-08 | **INLP v1** (Ravfogel et al. ACL 2020) on `exp7_baseline` / `exp7_ccmmd`; then **v2** symmetric-pool sensitivity | 46 / 40 iterations (literature default 20 insufficient); fit-pool C2ST 1.000 → **0.609 / 0.605**, MMD ~7–8× ↓; **held-out C2ST 0.963** (v1) / **0.969** (v2, fit pool doubled to 29,437); mechanism byte-identical (φ 0.465, z 0.397, size 0.240, rho AUC 0.920); LR M→P **−7–8 AUC points**; B2-CD 0.195–0.225, p=1.0 | `reports/inlp_alignment_summary.md:26-30,41-42,54-55,65-66,101-108,128-131,209-213` | "Linear post-hoc alignment cannot close the gap" **KEPT** (survives the leadfix per `CLAUDE.md` active-state). "INLP converged / domains became indistinguishable" **RETRACTED** (defect A4 — `…rerun_plan.md:127`). B2-CD magnitudes SUPERSEDED |
| 2026-05-08 | Email drafted to supervisor reporting the PARTIAL INLP outcome and asking A/B/C direction | — | `reports/email_drafts/2026-05-08_inlp_outcome_to_marta.md:19-51` | KEPT as record of the decision point |
| 2026-05-12 | **Consolidated supervisor summary** — glossary + every number to date in one document | consolidation only | `reports/supervisor_summary_2026_05_12.md` (§0 glossary, §2–§7) | KEPT as a narrative source; its numbers inherit the statuses above |
| **2026-05-13** | **Supervisor brief** — three asks, incl. "use a classifier trained on MedalCare MI folder labels to define φ thresholds, then apply the φ regressor cross-domain with corrected bins"; also "does reducing latent dimension help alignment without dropping performance?" | pre-registration of the Track 3 rule | `reports/b2cd_redux_log.md:5-8`; `reports/track1_latent_dim_log.md:9-13` | KEPT — pre-registration is a methodological asset; the rule was scored on 2026-08-12 |
| 2026-05-15 | **Track 3 redux** — new 4-class coronary territory design; Pipeline A (direct classifier on Z) and Pipeline B (φ regressor + wedges/calibrator); in-domain 8-class audit | in-domain 4c **0.506** [0.477, 0.535] (NK2 0.301); CD 4c **0.213** [0.172, 0.255] p=0.746; CD 2c 0.399 p=0.995; Pipeline B hardcoded 0.211 / learned calibrator 0.194; in-domain 8c **0.488**, 4c collapse 0.513, **2c transmurality 0.850** (NK2 0.627) | `reports/b2cd_redux_log.md:193-194,220-221,262-263,322-325,410-414` | 4-class **label design KEPT** (smallest class n=12→32, `:113-119`) but carries defect **D1**: `LCX_0.3_ant` and `LCX_0.3_post` are the same θ distribution under two labels → oracle ceiling macro-F1 **0.867**, not 1.0 (`…rerun_plan.md:140-153`). Cross-domain numbers RETRACTED (lead artifact); in-domain numbers SUPERSEDED-by-leadfix |
| 2026-05-17 | **Track 1a — PCA dim scan** (`analysis/dim_scan.py`), K ∈ {1024…8}, 3 pooling modes, pre-registered K* rule (C2ST ≤ 0.85 **and** LR M→P ≥ 0.65 **and** φ R² ≥ 0.35) | **no K satisfies the rule**; C2ST 1.000 at every K; MMD *rises* as K falls (0.126 → 0.342 median; 0.263 → 0.807 multi-bw); φ R² 0.474 → 0.125 | `reports/track1_latent_dim_log.md:34-37,43-46,54-59` | "Dimension is not the alignment bottleneck" **KEPT**; per-K magnitudes SUPERSEDED; the fallback-K* table is **RETRACTED as a selection rule** (K* chosen on the same test data it reports — `…rerun_plan.md:136`) |
| 2026-05-17 | **Track 1b′ / Tier 1 — trained bottleneck heads** `Linear(1024,K)+GELU+Linear(K,3)`, K ∈ {256, 64, 16}, head-only, frozen backbone | test avg-F1 0.852 → **0.867** (K=1024→16, i.e. classification is free); LR P→M AUC 0.640 → **0.988**; φ R² 0.494 → **0.002**; MMD-multibw 0.532 → 1.743; C2ST 1.000 at every K; in-domain 4c anatomy 0.506 → 0.346 | `reports/tier1_bottleneck_log.md:29-34,70-92` | The **Pareto structure** (classification free, transfer up, mechanism destroyed, alignment unmoved) is KEPT directionally; magnitudes SUPERSEDED. Caveat on record: K=1024 reference uses a **1-layer** head vs 2-layer for K∈{256,64,16}, so the +0.015 conflates depth with bottleneck (`tier1_bottleneck_log.md:43-46`) |
| 2026-05-18 | **Track 2 explainer** (no new compute) — why MMD fell 6× while C2ST stayed at 0.96 | MMD 0.12 → 0.02; C2ST 1.00 → 0.96; kNN mixing 0.005 → 0.013 | `reports/track2_inlp_asymmetry_explainer.md:15-18,45-47` | KEPT as *framing* ("we changed the instrument, not the gap"); its component numbers inherit the INLP statuses |
| 2026-05-24 | **Task 1 — INLP at low K** (K ∈ {16, 64, 256, 1024}, asymmetric + symmetric K=64) | held-out C2ST 0.761 (K=16) / 0.814 (K=64) / 0.958 (K=256) / ~1.0 (K=1024); best cross-domain F1 **0.272** at K=256+INLP (p=0.012); **K=1024 INLP is a numerical no-op** | `reports/2026-05-24_task1_task2a_results.md:129-134,191-198` | The K-ordering claim is **RETRACTED** — `rank(P_total)` is 4 at K=16, 16 at K=64, 208 at K=256, so "K=16 + INLP" is a 4-dimensional representation and the pattern is pure capacity loss (`…rerun_plan.md:126`) |
| 2026-05-24 | **Task 2a — 5-concept classifier** (`phi_sin, phi_cos, ẑ, size, trans_logit` → territory) | in-domain F1 0.482; cross-domain F1 **0.204** (p=0.84), macro-AUC **0.433** (p=1.0, worse than chance); 132/196 PTB-XL Inferior → Anteroseptal; **GT-θ upper bound 0.869**; dropping `phi_sin` *improves* CD F1 to 0.221 | `reports/2026-05-24_task1_task2a_results.md:240-245,263-268,301-308` | The failure *mechanism* (φ_sin sign flip) is **RETRACTED** as a domain phenomenon (lead artifact). The **GT-θ upper bound 0.869 is KEPT** as evidence the concept ontology is sufficient — it uses only MedalCare ground truth, no ECG lead ordering |
| 2026-05-26 | **Tier 2 — multi-task bottleneck K=64**, exactly as specified: Config A `0.5·cls + 0.5·physics`, Config B `0.0/1.0` | A: own-head φ R² **0.663** vs 0.467 at K=1024, disease F1 0.989 (Medal) / 0.789 (PTB-XL); refit-probe φ R² 0.587, MMD 0.451 → **0.081**, C2ST still **1.000**; cross-domain F1 **0.268**, p=0.098, AUC 0.510. B: best physics (φ 0.686) but worst transfer (F1 0.197, p=0.97) | `reports/2026-05-24_task1_task2a_results.md:589-594,631-646` | "What the bottleneck keeps is decided by the loss, not by K" is KEPT directionally; all magnitudes SUPERSEDED-by-leadfix. The CD improvement is **not** evidence (p=0.098, AUC≈0.5) and its confusion pattern is the retracted lead artifact (`:681-690`) |
| 2026-05-27 | **Poster planning** for the MRes poster (A1 portrait, due 2026-06-10 16:00, 2-min pitch + Q&A, mixed audience) — three candidate narratives, (A) recommended | (A) "what transfers, what doesn't": pairs the in-domain decoding positive with the cross-domain diagnostic figure | `reports/poster_planning_discussion.md:5,17-22,59-70` | Planning KEPT; **the poster PDF is not in the repo** (`find -iname "*poster*"` returns only the planning note). The second marker for the viva is the same person who marked this poster (`CLAUDE.md` header) |
| **2026-06 → 2026-07** | **No repo activity**: 0 commits, 0 files with a June or July 2026 mtime | — | `git log --date=short`; `find . -printf "%TY-%Tm"` month histogram (2026-05: 236 files; 2026-08: 960; June/July: absent) | Gap in the record. Only dated trace: "the *train an SAE on an ECG FM* lane closed on 28–29 July 2026" (`reports/2026-08-12_pivot_representational_geometry.md:148`). Poster delivery and literature reading are the presumed activity — **do not assert this in the thesis without a non-repo source** |
| 2026-08-10 | **Repo audit** — four confirmed defects (L lead order, F PTB-XL filter, A1 AUC columns, D1 LCX label contradiction) + ~5 substantive and ~15 minor statistical issues; staged `exp8_*` rerun plan | see §5 below | `reports/2026-08-10_repo_audit_and_rerun_plan.md:18-23,116-127` | **Boundary of this dossier.** Everything above is provisional relative to it |

---

## 2. What each named experiment / phase actually was

### 2.1 Weekly-baseline era (no "Exp" numbers yet)

| Name | Purpose | Setup (one line) | Headline result | How it ended |
|---|---|---|---|---|
| **Zero-shot baseline** | Reference point for "what does the pretrained checkpoint already do?" | `scripts/eval_zero_shot.py` on `checkpoint/12_lead_ECGFounder.pth`, no fine-tuning | MedalCare macro AP 0.164 (`README_w1.md:83`); PTB-XL macro AP 0.184, ROC-AUC 0.318 (`README_w4.md:19`) | Served its purpose; superseded by the leadfix era. Retained only as "a tiny head beats zero-shot by a mile" |
| **Frozen-encoder baselines** | Does a linear head suffice on a frozen backbone? | `finetune_multilabel.py --freeze-encoder --head-type {linear,mlp}`, class-balanced BCE, label smoothing 0.05, grad clip 1.0 | AP 0.813 / 0.829; full fine-tune 0.923 (`README_w1.md:84-86`) | Set the "frozen encoder + small head" convention for the entire project |
| **Joint multi-head (W4)** | First synthetic+real co-training | shared Net1D, two heads (MedalCare 8-class, PTB-XL 5-superclass), frozen encoder | val AP 0.797 / 0.777 (`README_w4.md:16-18`) | Became Exp 4/5 in the numbered series |
| **t-SNE latent visualisation** | Look at the shared latent space | joint checkpoint → z for both test sets → t-SNE | two largely disjoint clouds (`README_w4.md:25`) | Replaced by PCA (OpenSpec `2026-01-21-update-latent-viz-pca`), then by quantitative C2ST/MMD/kNN |
| **Physics head (W5)** | First direct test that the encoder carries mechanistic signal | MLP `z → θ̂` over a frozen ordered **51-dim** θ contract; masked MSE/Huber in normalised θ space; MedalCare batches only; train-only normalisation stats | normalized MAE 0.509; R² mean 0.064 (median 0.328); 21 strong / 5 moderate / 25 weak (`git show 30057ce:"weekly report/README_w5.md":28-31`) | **Abandoned.** The April audit showed ~30 of those 50 dims are dataset-wide constants and several are class tokens in disguise, so most of the "strong R²" dims were unusable. Replaced by 4-parameter `isch[0]` decoding (Phase B2). The spec survives as `openspec/specs/physics-head/spec.md` and `openspec/specs/physics-eval/spec.md` |

### 2.2 The numbered experiments (Exp 1–7)

All freeze the ECGFounder encoder; only `ConvAdapter1D` residual adapters (reduction 16, zero-init) and heads train; 30 epochs; best checkpoint by validation F1 (`reports/experiment_report.md:5,94-98`).

| Exp | Definition | Setup | Headline | How it ended |
|---|---|---|---|---|
| **Exp 1** | PTB-XL baseline | frozen encoder + fresh adapters, PTB-XL only, 5 superclasses | PTB-XL macro-F1 0.6982, best epoch 29 (`experiment_report.md:31`) | Kept as the "real data alone" reference |
| **Exp 2** | MedalCare adapter pretrain (feeder for Exp 3) | — | **never run** (`experiment_report.md:99`) | Abandoned — sequential-transfer line dropped |
| **Exp 3** | Sequential transfer (Exp 2 adapters → PTB-XL) | — | **never run** | Abandoned with Exp 2 |
| **Exp 4** | Joint baseline, **no adapters** | MedalCare + PTB-XL, dual-head 8+5, heads only | PTB-XL F1 0.7081 (best in the group), MedalCare 0.6624 (`experiment_report.md:32,40`) | Showed adapters are ~neutral on PTB-XL, +0.017 F1 on MedalCare |
| **Exp 5** | Joint + adapters (dual-head) | as Exp 4 with fresh zero-init adapters | PTB-XL 0.7065, MedalCare 0.6793 (`experiment_report.md:33,41`) | Became the "dual-head, no alignment" cell of the 2×2 |
| **Exp 6** | Joint + adapters + **MMD** | class-agnostic MMD, λ=1.0 in the original run | PTB-XL 0.7037; MMD driven 0.078 → 0.009; early-stopped epoch 18 (`experiment_report.md:34,79-80`) | The λ=1.0 class-agnostic variant was **retired** from the headline comparison: it inflates KMeans NMI/ARI (0.427/0.386) while giving *worse* cross-domain transfer than ccMMD λ=0.1 (`exp7_progress_report.md:268`) |
| **Exp 7** | **Shared head** — one `Linear(1024, 3)` over {NORM, MI, CD} for both domains | alternating MedalCare/PTB-XL batches, labels remapped on the fly, combined `pos_weight` (NORM 1.945, MI 1.572, CD 3.162), Adam with differential LRs (head 1e-3, adapters 1e-5), ReduceLROnPlateau, checkpoint on mean per-domain macro-F1; 312,643 trainable params | MedalCare F1 0.917 / PTB-XL 0.788 at epoch 29; LR M→P accuracy 0.373 → 0.655 (`exp7_progress_report.md:77-80,151-156`) | Became **the** encoder for everything downstream (Phase B, INLP, Track 1/2/3, Tier 1/2). Its PTB-XL metrics are contaminated by filter defect F |
| **`exp5_3class` / `exp6_3class`** | The 2026-05-04 re-run of Exp 5/6 on the **same** 3-class space, `exp6_3class` using ccMMD λ=0.1 to match Exp 7 | identical filtering, remapping, hyperparameters; both early-stopped at epoch 17, best epoch 13 | dual-head LR M→P AUC 0.589 / 0.609 vs shared-head 0.759 / 0.766 (`exp7_progress_report.md:238-245,257-260`) | Closed the supervisor's confound: label-space coarsening alone changes nothing (KMeans 0.654 → 0.654) |
| **The 2×2 ablation** | {dual, shared} × {no alignment, ccMMD λ=0.1} | cells: `exp5_3class`, `exp6_3class`, `exp7_baseline`, `exp7_ccmmd` | headline: architecture, not relabeling, drives transfer; ccMMD adds +0.02 (dual) / +0.007 (shared) | KEPT as the design; numbers SUPERSEDED. Audit lists the claim as UNKNOWN pending `exp8` (`…rerun_plan.md:134`) |
| **`exp7_baseline_norm`** | Normalisation ablation (per-lead z-score on MedalCare) | retrain of `exp7_baseline` | "C2ST still 1.000, transfer worsened 65.5% → 55.7%" (`exp7_progress_report.md:202`) | **RETRACTED** — the run still had swapped leads, so it tested a confounded pair |

### 2.3 Latent-space evaluation machinery

| Name | Purpose | Setup | Headline | How it ended |
|---|---|---|---|---|
| **C2ST / domain classifier** | Can a classifier tell synthetic from real using only the latents? | 5-fold logistic regression on pooled latents, report AUROC (`analysis/exp7_analysis.py`; ancestor spec `openspec/specs/analysis/spec.md`) | **1.000 in every configuration ever tested** (`exp7_progress_report.md:109,194`) | The project's central negative. Survives the leadfix at 1.0000 (`…rerun_plan.md:28-32`). Later shown to be *constitutionally blind* to lead permutations (`CLAUDE.md` active-state, 2026-08-11) |
| **MMD** | Distributional distance | RBF, median heuristic and 5-bandwidth variants; also used as a training loss (class-agnostic and class-conditional) | −13% under ccMMD (`exp7_progress_report.md:107`); *rises* as K falls, both PCA and bottleneck arms (`track1_latent_dim_log.md:43-46`) | Reinterpreted, not trusted: at K=1024 single-bandwidth MMD is underpowered (the AAAI 2020 effect, empirically confirmed) — `tier1_bottleneck_log.md:139-141` |
| **k-means / cosine / DTW battery** | Is the latent space organised by disease, and does latent proximity mean waveform proximity? | k-means(3) with Hungarian matching + NMI/ARI; intra/inter/cross-domain cosine; DTW ratio vs random pairs | KMeans combined 0.734; cosine gap 0.140 with cross-domain same-class **−0.054**; DTW ratio 0.549 / 0.881 (`exp7_progress_report.md:144-147,164,173-176`) | Delivered the sharpest one-liner of the era — *the shared head aligns decision boundaries, not feature geometry* (`exp7_progress_report.md:198`) |
| **`dim_scan`** (Track 1a) | Is the gap a high-dimensionality artifact? | post-hoc PCA of existing latents to K ∈ {1024…8}, three pooling modes, re-run the whole battery; pre-registered K* rule | no K passes; C2ST 1.000 throughout (`track1_latent_dim_log.md:34-37,54-63`) | Answered "no" without spending GPU time; the K* fallback selection is retracted as a rule |
| **Tier 1 bottleneck evaluation** | Same question with a *learned* projection | `analysis/tier1_evaluation.py`, 4 blocks (alignment / class structure / mechanism / 4c anatomy), N_BOOT 200 + N_PERM 50 for the first three, 1000/200 for anatomy; 49 integrity checks | see timeline; the Pareto table at `tier1_bottleneck_log.md:70-92` | Both arms agree ⇒ the "dimension is the bottleneck" hypothesis is dead from two directions (`track1_latent_dim_log.md:141`) |
| **Tier 2 evaluation** | Fair comparison of the multi-task bottleneck | `analysis/eval_tier2.py`: L1 alignment, L2 refit probes, L3 Pipeline-A transfer | `2026-05-24_task1_task2a_results.md:631-646` | Known defect: `eval_tier2.py:158` passes the same array as both PTB-XL "pool" and "test" legs (`…rerun_plan.md:194`, m5) |

### 2.4 Alignment attempts

| Name | Purpose | Setup | Headline | How it ended |
|---|---|---|---|---|
| **MMD / ccMMD (training-time)** | Force distributions together during training | class-agnostic λ=1.0 (Exp 6) and class-conditional λ=0.1 (`exp6_3class`, `exp7_ccmmd`) | statistic moves, nothing downstream does (`exp7_progress_report.md:196`) | Retired as a headline; ccMMD λ=0.1 kept as the 2×2's alignment axis |
| **INLP (Track 2)** | Post-hoc, retraining-free removal of the linear domain direction — **the supervisor's homework from her 2026-04-29 email** | iterative rank-1 nullspace projection, `max_iter=50` (20 was insufficient), `stop_acc=0.55`, balanced logistic domain head, fit pool MedalCare-train + PTB-XL-test (n=14,217), seed 42 | 46/40 iterations; fit-pool C2ST → 0.61; **held-out C2ST 0.96**; mechanism byte-identical; class transfer −7–8 AUC (`inlp_alignment_summary.md:26-30,41-42,54-55,65-66,101-108`) | **PARTIAL** against its own pre-registered gates (in-domain R² > 0.37 ✅, cross-domain F1 > 0.45 ❌ — `:163-165`). Aligned latents kept as a **separate evaluation arm**, never promoted to canonical (`:253-256`) |
| **INLP v2 (symmetric pool)** | Pre-empt "you fit and tested on the same PTB-XL split" | fit pool MedalCare-train + PTB-XL-**train** (n=29,437); both test splits truly unseen | held-out C2ST 0.969 vs v1's 0.963 (`inlp_alignment_summary.md:209-213`) | The methodological-rigor exhibit of the era: doubling the pool changed nothing |
| **INLP at low K (Task 1)** | Does INLP work once the domain subspace is small? | `analysis/inlp_lowK.py` on the bottleneck latents, K ∈ {16, 64, 256}, plus a symmetric K=64 arm | held-out C2ST 0.76–0.96; best CD F1 0.272 at K=256 (`2026-05-24…:129-134,191-198`) | Conclusion **RETRACTED** — rank collapse confound (defect A4) |
| **Ruled-out family** | manifold mixup, SSL/contrastive/SupCon, further alignment-via-training | never implemented | — | Dropped at the 04-29 meeting on the strength of C2ST=1.0 across four methods (`supervisor_meeting_2026_04_29.md:53-61`; talking point at `…supplement.md:101-103`). Manifold mixup was a standing supervisor request (`SUPERVISOR_TODO.md:43-50`) that was deliberately deprioritised |

### 2.5 Mechanistic decoding (Phase B / B2 / Track 3)

| Name | Purpose | Setup | Headline | How it ended |
|---|---|---|---|---|
| **Phase B (planned)** | Ridge/CCA between latents and MedalCare θ; APD (B1), infarct geometry (B2), clinical-feature transfer (B3) | `reports/exp7_progress_report.md:286` | — | B1 (APD → QTc on PTB-XL) **deferred and never run** (`supervisor_summary_2026_05_12.md`→§7 open follow-ups; `SUPERVISOR_TODO.md:109`). B2 became the whole of Phase B |
| **Phase B2 in-domain** | Can a linear probe read `isch[0].{phi, z, size, rho_eps_max}` off the frozen latent? | Ridge (φ as sin/cos, circular R²) + logistic (transmurality) fit on MedalCare-train MI, tested on MedalCare-test MI, with NK2 6-feature paired baseline, bootstrap CIs, permutation nulls | φ 0.47, z 0.39, size 0.24, rho AUC 0.92 — identical within CIs across all four trained configs, i.e. **the signal is in the frozen backbone, not in our adapters** (`b2_infarct_localization_log.md:45-50`) | The positive that carried the project; direction survives the audit, magnitudes and p-values do not |
| **Phase B2-CD (cross-domain)** | Does the same probe localise MI on real ECGs? | apply the MedalCare probe to PTB-XL MI latents; bin predicted φ into territories; compare to SCP-derived labels | chance, p=1.0 (`b2_infarct_localization_log.md:64-67`) | Failure mechanism as diagnosed in May (φ sign flip / Inferior→Anterior collapse) is **retracted**; the *fact* of a cross-domain deficit persists into the August work |
| **Track 3 (B2-CD redux)** | The supervisor's 2026-05-13 ask, done properly | new 4-class territory (`territory_4c`) expressible in both datasets; **Pipeline A** = multinomial logistic on Z; **Pipeline B** = φ regressor + {hardcoded wedges, learned calibrator}; run over 6 configs | `b2cd_redux_log.md:20-29` | Numbers retracted/superseded; the **label design and its three constraints** (`:36-106`) are reusable Methods material |
| **Track 3 decision rule (pre-registered)** | Prevent post-hoc goalpost moving | POSITIVE = 4-class macro-F1 ≥ 0.45 with p < 0.01; PARTIAL = 2-class ≥ 0.55 (later ≥ 0.65) with p < 0.01 | May scoring: NEGATIVE on both (`b2cd_redux_log.md:20-29,515-522`) | Re-scored on the clean `exp8_leadfix_medalonly` encoder on 2026-08-12 — 4-class not met; 2-class PARTIAL met under the strict scaler only (`CLAUDE.md` active-state). Note the PARTIAL bar is quoted as **0.55** in May and **0.65** in the August record — reconcile before citing |
| **In-domain 8-class audit** | Does Z encode anatomy × transmurality jointly? | `territory_8c` = 4 territories × {subendocardial, transmural}, logistic on Z, three confusion-matrix collapses | 8c 0.488, 4c collapse 0.513 (matches Pipeline A's 0.506 — a genuine consistency check), **2c transmurality 0.850** vs NK2 0.627 (`b2cd_redux_log.md:410-414,421-426`) | Was the strongest in-domain claim of the May era. SUPERSEDED-by-leadfix but the *shape* of the argument (in-domain works ⇒ the labels are not the problem) is reusable |
| **"concept5 classifier"** | An interpretable synth→real bridge: instead of transferring 1024-d Z, transfer the 5 numbers the probes predict | `analysis/concept5_classifier.py`: fit φ/z/size/transmurality probes on MedalCare-train MI, assemble `[phi_sin, phi_cos, ẑ, size, trans_logit]`, train a multinomial LR on that 5-vector, apply to PTB-XL; sensitivity = drop-one ablations + MLP head + ground-truth-θ upper bound | CD F1 0.204 (p=0.84), CD AUC 0.433 (p=1.0); **GT-θ upper bound 0.869**; dropping `phi_sin` *helps* (`2026-05-24…:240-245,301-308`) | Ended as the third read-out to hit the same wall. The GT-θ upper bound is its durable contribution: the concept ontology is sufficient, so the failure is in the synth→real probe transfer |
| **Tier 2 multi-task bottleneck** | Marta's constructive sequel to Tier 1: make the K=64 bottleneck keep the physics | `scripts/finetune_bottleneck_multitask.py`: shared `z_k = Linear(1024,64)`, two heads (3-class + 5-channel physics), physics loss masked to MedalCare infarct rows, adapters trainable, seed 42; Config A `0.5/0.5`, Config B `0.0/1.0` | A doubles every Tier-1 physics score and beats the full 1024-d latent (φ 0.663 vs 0.467) at no classification cost; B is best on physics and worst on transfer (`2026-05-24…:589-594`) | Answered Marta's question decisively **in-domain**; cross-domain remained borderline (F1 0.268, p=0.098). Loss-mix sweep, K=16 version and the two-stage curriculum (`:775-785`) were proposed and **never run** |

### 2.6 The poster

MRes AI&ML poster, A1 portrait ~300 dpi PDF, **due 2026-06-10 16:00**, conference sections, 30-second readability, 2-minute pitch + Q&A, mixed audience (`reports/poster_planning_discussion.md:5,17-22`). Three narratives were drafted; **(A) "what transfers, what doesn't"** was recommended over (B) "the gap is intrinsic" (all-negative, "risks reading as nothing worked") and (C) "cross-domain classification via shared-head fine-tuning" ("honest but boring… loses the digital-twin angle that justifies the thesis title") — `:59-96`. Four open questions were logged and never resolved in the repo (`:99-112`). **No poster file exists in the repo**; the second marker of that poster is the second marker for the viva (`CLAUDE.md` header), so its framing has downstream consequences.

---

## 3. What survives into the thesis

### 3.1 Main text — "The road to the audit" (compressed history)

| Item | Why it earns main-text space | Pointer |
|---|---|---|
| **The alignment dead-end, stated once with all four attacks** — training-time MMD (class-agnostic and class-conditional), post-hoc INLP (at K=1024 and at K ∈ {16,64,256}), post-hoc PCA to K=8, and a trained bottleneck to K=16 — all leaving C2ST at 1.0 | It is the only early result that **survived the audit intact** and it is what licenses the whole pivot. Also the cleanest "we did the obvious thing and it failed" paragraph in the project | `exp7_progress_report.md:194,206`; `track1_latent_dim_log.md:141`; `inlp_alignment_summary.md:65-66`; survival: `…rerun_plan.md:28-32` |
| **The pivot itself** (domain adaptation → interpretability oracle), dated and attributed to the 04-29 meeting | The thesis needs one sentence explaining why the title is about *interpreting* and not about *aligning*; the meeting record makes it a documented decision rather than a retrofit | `supervisor_meeting_2026_04_29.md:17-21` |
| **In-domain θ decodability as the enabling positive** — but requoted from `exp8`, not from May | The audit certifies the *direction* survives every defect (`…rerun_plan.md:112`); the magnitudes must come from the corrected runs | `b2_infarct_localization_log.md:45-48` (historical); current numbers from the August record |
| **The four-read-outs-one-wall observation** (raw Z → INLP Z → concept-5 → Tier-2 multitask) | This is the logical bridge to the fidelity audit: after the fourth read-out fails identically, the question stops being "which read-out?" and becomes "what information is missing?" | `2026-05-24_task1_task2a_results.md:681-690` |
| **Pre-registration as a practice**, with Track 3's rule as the worked example | Examiner-facing credibility; the rule was written on 2026-05-13, long before the numbers existed, and was honoured when it scored NEGATIVE | `b2cd_redux_log.md:5-8,515-522` |
| **One paragraph of self-correction**: the August audit, what it invalidated, and that the pre-fix state is tagged `pre-leadfix` | Declarations and integrity both need it, and volunteering it is far stronger than being asked about it in the viva | `…rerun_plan.md:18-23,116-127,214-217` |

### 3.2 Appendix

| Item | Why appendix rather than main text | Pointer |
|---|---|---|
| Weeks 1/4/5 baselines (zero-shot, frozen linear/MLP, PTB-XL baselines, physics head) | Provenance and honest scaffolding, but every number is pre-leadfix and none is load-bearing | `README_w1.md:83-86`; `README_w4.md:14-19`; `README_w5` via `git show 30057ce` |
| Exp 1/4/5/6 table | The conclusion is "no configuration mattered by more than 1 F1 point", which is one sentence; the table itself belongs in an appendix, flagged with defect F | `experiment_report.md:31-34,40-42` |
| The full 2×2 ablation table | Design is main-text-worthy; the numbers are contaminated and being re-run, so the table sits in the appendix with an `exp8` cross-reference | `exp7_progress_report.md:242-245,257-260` |
| Track 1 (PCA dim scan + Tier 1 Pareto) | A genuinely interesting four-axis trade-off — classification free at K=16, transfer up, mechanism to zero, alignment unmoved — but it answers a question (dimensionality) that the thesis no longer asks | `track1_latent_dim_log.md:43-46`; `tier1_bottleneck_log.md:70-92` |
| Tier 2 multi-task bottleneck | The most constructive result of the May era ("supervision decides what the bottleneck keeps") and directly supervisor-commissioned; pre-leadfix, so appendix with the caveat | `2026-05-24_task1_task2a_results.md:589-594,631-646` |
| INLP protocol + v1/v2 symmetric sensitivity | The methodological-rigor exhibit; a reviewer's first objection was pre-empted and answered | `inlp_alignment_summary.md:194-227` |
| MedalCare parameter audit (9 of 50) and the θ-4 contract | Methods-critical: it explains why θ is four numbers. Still valid | `supervisor_meeting_2026_04_29_supplement.md:15-22` |
| Track 3 4-class label design (three constraints, exclusions, n=32) + the D1 oracle ceiling | Methods-critical and reusable; must be presented **with** the 0.867 ceiling | `b2cd_redux_log.md:36-119`; `…rerun_plan.md:140-153` |
| The GT-θ upper bound (0.869) | A clean "the ontology is sufficient" control that does not depend on lead ordering | `2026-05-24…:301-308` |

### 3.3 One sentence only

- **ccMMD**: reduces the MMD statistic by 8–13% and changes nothing downstream (`exp7_progress_report.md:196`).
- **Week-1 MMD "improves ≥4/8 tasks"**: never tabulated (`README_w1.md:75`) — mention only if the narrative needs to date the first alignment attempt.
- **k-means / cosine / DTW battery**: one sentence for the durable line — the shared head aligns decision boundaries, not feature geometry (`exp7_progress_report.md:198`).
- **concept-5 classifier**: one sentence as the third failed read-out, plus its GT-θ upper bound in the appendix.
- **Exp 2/3 never run** (`experiment_report.md:99`) — one clause, so the numbering gap is not mysterious.
- **The poster** — one sentence in Declarations/timeline; there is no artefact in the repo to cite.
- **Physics head (51-dim θ)** — one sentence explaining why it was replaced.

### 3.4 Best omitted

| Omit | Reason |
|---|---|
| The **φ sign-flip** narrative and the **Inferior→Anteroseptal collapse** as a domain-gap phenomenon | Retracted: both are artifacts of the aVF/aVL transposition (151/196 → 44/196 after the fix) — `…rerun_plan.md:119-120`. They were the most vivid figures of the May era, which is exactly why they must not survive |
| **`exp7_baseline_norm`** "per-lead normalisation makes things worse" | Retracted as untested (`…rerun_plan.md:122`) |
| **Pipeline B in-domain learned calibrator (macro-F1 0.998)** | Pure resubstitution with best-of-3 model selection on the scored rows; the audit says "indefensible; delete" (`…rerun_plan.md:124`) |
| **The `dim_scan` fallback K\* table** (K*=8/16) | Selected on the same test data it reports (`…rerun_plan.md:136`) |
| **The INLP K-ordering claim** ("low K is more alignable") | Confounded by rank collapse: rank(P_total) = 4 / 16 / 208 (`…rerun_plan.md:126`) |
| **Every `permutation_p_r2` from `phase_b2/in_domain.json` and `dim_scan/*_summary.json`** | The null omits the ridge intercept, so a constant-zero predictor sits at the p-floor (`…rerun_plan.md:125`) |
| **Week-1 full fine-tune (AP 0.923)** | Off the project's own methodological line — every later result freezes the encoder; quoting it invites "why didn't you just fine-tune?" without a defensible answer |
| **t-SNE→PCA visualisation change** | Housekeeping; zero information |
| **Idea A / Idea B fall-back list** as *plans* | Idea A was executed as the concept-5 classifier; Idea B (transmurality vs PTB-XL `INJ*` codes) was never run (`b2_infarct_localization_log.md:110-118`). Describing unexecuted plans as if they were findings is the failure mode to avoid |

---

## 4. Numbers to be careful with

**Rule of thumb: anything produced before the `exp8_*` runs sits downstream of at least one of the four confirmed defects. Do not quote a pre-`exp8` number as current without either (a) a corrected re-measurement or (b) an explicit "as reported in May, since superseded" framing.**

### 4.1 Conclusions the record itself retracted

| Retracted claim | Where it was asserted | Correction |
|---|---|---|
| "Cross-domain territory transfer is at chance (macro-F1 0.213, p=0.76)" | `b2cd_redux_log.md:220-221` | 0.328, p=0.002 with correct leads, no retraining (`…rerun_plan.md:118`). ⚠ `CLAUDE.md` warns: cite the lead-identity check `aVL = (I − III)/2`, **not** this p-value — 39/66 arbitrary lead transpositions also improve transfer |
| "The φ axis is sign-flipped between synthetic and real" | `b2_infarct_localization_log.md:83`; `2026-05-24…:278-283` | Frontal-plane reflection artifact (`…rerun_plan.md:119`) |
| "Inferior→Anteroseptal collapse is a domain-gap phenomenon" (151/196; repeated for concept-5 at 132/196 and Tier-2 at 111/196) | `b2cd_redux_log.md:241-245`; `2026-05-24…:263-268,681-690` | Artifact; 151/196 → 44/196 on the fix (`…rerun_plan.md:120`) |
| "In-domain territory decoding is weak (macro-F1 0.51, AUC 0.58)" | `b2cd_redux_log.md:193-194`; `2026-05-24…:191-198` | macro-F1 0.59, AUC 0.78 against an oracle ceiling of 0.867 (`…rerun_plan.md:121`) |
| "Per-lead normalisation makes things worse (65.5% → 55.7%)" | `exp7_progress_report.md:202` | Untested — that run had swapped leads (`…rerun_plan.md:122`) |
| "The learned φ→territory calibrator beats fixed wedges in-domain (0.998)" | Pipeline B in-domain outputs | Resubstitution; delete (`…rerun_plan.md:124`) |
| "INLP converged / the domains became indistinguishable"; "K=16 is more alignable than K=256" | `inlp_alignment_summary.md:41-42,54-55`; `2026-05-24…:129-134` | Not supported at any K; K=256 hit `max_iter` with domain accuracy 0.742 and C2ST 0.958; rank collapse confounds the K comparison (`…rerun_plan.md:126-127`) |
| Every `permutation_p_r2` in `phase_b2/in_domain.json` and `dim_scan/*_summary.json` | e.g. `b2_infarct_localization_log.md:45-48` ("p=0.001") | Vacuous null (`…rerun_plan.md:125`). Conclusions survive a corrected null for z and size; the test as written rejects nothing |

### 4.2 Numbers that are merely superseded (direction probably fine, magnitude not)

- All Week-1 / Week-4 AP numbers (`README_w1.md:83-86`, `README_w4.md:14-19`) — pre-leadfix MedalCare, and pre-metric-suite change (AP was later replaced by threshold metrics).
- **All PTB-XL classification metrics from every shared-head run** — trained and evaluated on NORM/MI/**STTC**, ~42% of CD dropped and ~315 all-zero rows retained (`…rerun_plan.md:123`). This includes the headline "PTB-XL F1 = 0.788".
- All Exp 1/4/5/6 and 2×2 numbers (`experiment_report.md:31-42`; `exp7_progress_report.md:242-260`).
- All Phase-B2 in-domain magnitudes (φ 0.47 / z 0.39 / size 0.24 / rho 0.92) and the 8-class audit (8c 0.488, 2c transmurality 0.850).
- All Track 1 per-K numbers and the whole Tier 1 Pareto table.
- All Tier 2 numbers, including the "beats the full 1024-d latent" comparison.
- The INLP alignment numbers as *magnitudes* (the qualitative conclusion — held-out C2ST stays ~0.96–0.97 — is the part that survives).

### 4.3 Internal inconsistencies to reconcile before quoting

- **PTB-XL 3-class test-set size**: **1,787** at `exp7_progress_report.md:36` vs **1,891** at `experiment_report.md:101`, `exp7_progress_report.md:251` and `supervisor_summary_2026_05_12.md`→§4.2. PTB-XL CD test positives likewise **274** (`exp7_progress_report.md:44`) vs **285** (`experiment_report.md:62`). Pick one, from the manifest, and state it once.
- **MedalCare test-set size**: 2,386 total (`b2_infarct_localization_log.md:15`) vs 2,126 after 3-class filtering (`exp7_progress_report.md:36`) vs 1,200 MI rows for B2 — three different denominators that are easy to conflate.
- **PTB-XL MI evaluation subset**: n=**444** under the old 3-class design (`b2_infarct_localization_log.md:16`) vs n=**438** under the 4-class design (`b2cd_redux_log.md:119`).
- **Track 3 PARTIAL threshold**: 2-class ≥ **0.55** in the May record (`b2cd_redux_log.md:28`) vs ≥ **0.65** in the August record (`CLAUDE.md` active-state). The pre-registration text must be quoted verbatim from whichever document is primary.
- **Li et al. venue**: cited as *MICCAI 2024* throughout the April/May documents (`supervisor_meeting_2026_04_29.md:44`; `…supplement.md:80`); the verified venue is **IEEE TMI 2024** (memory `reference_prior_work_infarct_inverse_inference`; `poster_planning_discussion.md:110`). Every re-use must be re-verified via `papersflow.verify_citation`.
- **θ arity**: the April audit says four MI-only parameters "phi, z, size, transmurality" (`…supplement.md:42-47`); `transmural` is a duplicate array of `rho_eps_max`, so θ has **4** members and not 5 (`…rerun_plan.md:191`).
- **"Six experiments"** in the overview vs a seven-row table (`experiment_report.md:5,11-19`) — trivially confusing if quoted.

### 4.4 Structural caveats that must travel with any 4-class number

- **D1 oracle ceiling 0.867**: `LCX_0.3_ant` and `LCX_0.3_post` are the same θ distribution under two labels; Inferolateral recall is pinned at exactly 0.500 by construction. Every 4-class number in the project was implicitly compared against a ceiling of 1.0 (`…rerun_plan.md:140-153`).
- **Tier 1's 1-layer vs 2-layer head confound**: the K=1024 reference is a 1-layer head, so the "+0.015 F1 from the bottleneck" conflates depth with compression; the clean control was deliberately not run (`tier1_bottleneck_log.md:43-46`).
- **INLP v1 pool contamination**: PTB-XL-**test** is inside the v1 fit pool, so the v1 "test-pool" metric mixes held-out synthetic with in-pool real (`inlp_alignment_summary.md:194-201`; also `…rerun_plan.md:189`, M4).
- **Permutation p-value floors**: `N_PERM_BINARY=200` ⇒ minimum p = 1/201 ≈ 0.005; 105 of 605 reported p-values sit at a floor and none is multiplicity-corrected (`…rerun_plan.md:187`). A floor p is not evidence of effect size.
- **Reproducibility**: the code that produced the May results was uncommitted until 2026-08-11 (tag `pre-leadfix`) — `…rerun_plan.md:168-179,214-217`.

---

## 5. Supervisor guidance recorded

*Marta's inputs reach the repo second-hand — as the student's meeting agendas, action-item lists and draft replies. Where the record is the student's own framing rather than her words, that is flagged.*

### 5.1 The standing action list (`reports/SUPERVISOR_TODO.md`)

Six items, "meeting notes — interpreted and formalized" (`:3`), i.e. the student's rendering:

| # | Ask | Status in the record |
|---|---|---|
| 1 | K-means clustering on 1024-d Z (k=5 PTB-XL, k=8 MedalCare) with Hungarian accuracy, NMI, ARI, compared across Exp 1/4/5/6 | **Done** (`:14-19,92`) — though what was actually reported is k=3 on the shared label space |
| 2 | **Multiple random seeds** {42, 123, 456, 789, 1024}, mean ± std, paired tests | **Never started** (`:23-30`). Re-listed as an open follow-up on 2026-05-12 and again as Phase D in `exp7_progress_report.md:288`. **This is the single most-repeated unmet supervisor request in the project** |
| 3 | **Prototype learning** (arXiv:2508.01521) — per-class prototypes, link back to nearest ECGs and to θ | **Never started** (`:32-39`); Phase C in every plan; dropped at the freeze |
| 4 | **Manifold mixup** (Verma et al. ICML 2019) as a better alignment approach | **Never started, deliberately** (`:43-50,96`) — deprioritised at the 04-29 meeting because C2ST=1.0 across four methods made a fifth alignment method unattractive (`supervisor_meeting_2026_04_29_supplement.md:101-103`) |
| 5 | **Better latent metrics** than PCA pictures — k-means accuracy, cosine intra/inter, DTW | **Done** (`:54-63`) — this ask is the direct cause of the whole Phase-A battery |
| 6 | **Exp 7 shared head** for overlapping classes, with a proposed mapping incl. a tentative `lae ↔ HYP` | **Done** (`:67-86`); the `lae ↔ HYP` pairing was dropped, `lae`/`fam` excluded (`exp7_progress_report.md:27`) |

### 5.2 The 2026-04-29 meeting

The meeting document is the **student's proposal and question list**, not a minute of Marta's answers (`reports/supervisor_meeting_2026_04_29.md`). What it records:

- The proposed **pivot** from domain adaptation to "digital twins as an interpretability oracle" (`:17-21`) and the reframing of digital twins from an alignment objective to an interpretability oracle.
- A **parameter-audit-driven simplification** of θ from 50 dims to 5 APD + 4 isch (`:23-36`).
- A proposed 5-claim paper with the working title *Cardiac Digital Twins as One-Pass Mechanistic Interpretability for ECG Foundation Models*, with **B2 (infarct localisation) nominated as "the killer figure"** and D2 (closed-loop retrieval) as the headline (`:39-49`).
- **Explicit drops**: manifold mixup, SSL/contrastive/SupCon, TCAV, Z→ECG decoder counterfactuals, 50-dim θ_core CCA (`:53-61`).
- **Ten questions put to the supervisor** (`:65-85`), the load-bearing ones being: Q1 direction shift; Q2 θ scope reduction; Q4 whether B2 or B1 is the central figure; Q5 thesis-only vs concurrent paper and venue; Q6 whether AMI≈anterior / IMI≈inferior mapping is clinically safe; Q7 n=3 vs n=5 seeds; Q8 whether the optional multi-task Exp 8 is worth committing to.
- The **timeline** as of that meeting: ~17 weeks technical + 4 weeks writing → mid-September, with mandatory pilot gates at W1/W4/W8 (`:51`).

**The one directly attributed supervisor instruction from this window** is INLP, from her 2026-04-29 email, together with her caveat, quoted twice in the record:

> *"It may not work well if the distribution of data across latent space for both datasets has different shapes."* — `inlp_alignment_summary.md:175-177`; `exp7_progress_report.md:308`

That caveat turned out to be exactly right, and the project treats it as such.

### 5.3 The re-run she asked for (concern → experiment)

Her concern that Exp 7's advantage might be the smaller 3-class label space rather than the shared head produced the `exp5_3class` / `exp6_3class` re-run and the clean 2×2 (`exp7_progress_report.md:218-234`; OpenSpec `2026-05-04-redo-exp5-exp6-with-shared-labels/proposal.md`). This is the clearest example in the record of a supervisor question converting directly into a controlled experiment, and it is worth saying so in the thesis.

### 5.4 The 2026-05-13 brief

- **Track 3 ask (point 3)**: "use a classifier trained on MedalCare MI labels (the folder names encoding coronary territory, transmurality, and lateral sub-location) to define phi thresholds, then apply the phi regressor cross-domain on PTB-XL with the corrected bins" (`b2cd_redux_log.md:5-8`). This is Pipeline B, and it is the origin of the pre-registered decision rule.
- **Track 1 ask**: "Does reducing latent dimension help domain alignment (MMD/C2ST)… without dropping too much performance?" and "Was 1024-d necessary?" (`track1_latent_dim_log.md:9-13`). Answered with a deliberately two-armed design (post-hoc PCA **and** trained bottleneck) so the answer could not be blamed on either method (`:15-22`).
- The answer was pre-drafted in her language for the end-of-week email (`track1_latent_dim_log.md:145-149`).

### 5.5 The 2026-05-18 meeting

`reports/track2_inlp_asymmetry_explainer.md` was written **for** that meeting (`:3`), in response to her question: *"MMD says the gap is almost gone, but C2ST says it is still there, and kNN says local geometry didn't move. Which one is telling the truth?"* (`:20`). It ran **no new experiments** — the answer came from data already on disk (`:4`) — and ends with a two-option menu (Tier 2 vs the digital-twin axis-naming chapter, `:89-94`).

### 5.6 The 2026-05-24 and Tier-2 asks

- **Macro-OvR AUC alongside macro-F1** was an explicit supervisor requirement; a dedicated helper was written for it (`2026-05-24_task1_task2a_results.md:111-118`), with 1000-bootstrap CIs and 1000-permutation p-values (`:449-450`).
- **Tier 2's two configurations were specified by her**: "We run two versions, exactly as Marta specified" — Config A 50/50 and Config B physics-only (`2026-05-24…:493-501`). Her framing question is quoted: *"what if we also tell the bottleneck to keep the physics?"* (`:485`).

### 5.7 Poster steer

The brief (A1 portrait, 30-second readability, 1–2 key messages, 2-minute pitch, mixed audience) is recorded at `poster_planning_discussion.md:17-22`; the four unresolved questions — narrative direction, AI/ML vs cardiology emphasis, which future claim to show as the research plan, and citation scope — are at `:99-112`. The second marker who assessed it is the viva's second marker.

### 5.8 Outside the part-1 window, but it shapes the framing

The 2026-08-14 draft email puts **five decisions** to her, the first gating: **(1) scaler primacy** (strict/source vs recalibrated/target — it decides three results in opposite directions), (2) audit resolution wording, (3) how much of the repair negative goes in the main text, (4) framing and chapter titles, (5) publication (ML4H 2026 Findings, Sep 10) — `reports/email_drafts/2026-08-14_scaler_and_framing_to_marta.md:28-62`. Same email carries the logistics: thesis **Aug 28**, viva Sept 7–18, slot to be agreed with both markers early (`:64-73`).

---

## 6. Compute usage (for the Declarations "compute" statement)

- **Hardware**: a single workstation GPU — **NVIDIA GeForce RTX 5080, 16,303 MiB** (`reports/EXECUTION_LOG_2026-08-10.md:19`). Earlier logs describe it only as "a single RTX-class GPU" (`tier1_bottleneck_log.md:41`). Training logs confirm `Device: cuda` (`outputs/_t1b_sweep_log.txt`). No cluster, no multi-GPU, no cloud.
- **Software**: Python 3.10; PyTorch 2.4 + CUDA 12.1 at project start (`README_w1.md:20-22`), PyTorch 2.9.1+cu128 at the end (`CLAUDE.md`→Tech Stack); scikit-learn 1.6.1; seed **42** everywhere (`2026-05-24…:449-450`). Scripts fall back to CPU with a warning; GPU is nominally optional (`README_w1.md:22`).
- **Trained runs on disk**: **19** run directories containing checkpoints (`ls -d outputs/*/checkpoints`) — `joint_{baseline,adapter_cls,adapter_mmd}`, `exp5_3class`, `exp6_3class`, `exp7_{baseline,ccmmd,baseline_norm}`, `exp7_bottleneck_K{16,64,256}`, `exp7_tier2_K64_{A_5050,B_bioonly}`, and six `exp8_leadfix_*` runs (the last group is August, i.e. part 2).
- **Latent exports**: **127** directories under `outputs/latents/` (originals, `_inlp`, `_inlpv2`, bottleneck-K, and diagnostic exports).
- **Epoch budgets**: Exp 1/4/5/6 and Exp 7 ran 30 epochs, best epochs 29/18/27/14 and 29 respectively (`experiment_report.md:31-34,50-51`; `metrics.json` confirms 30 evaluation entries for `exp7_baseline`/`exp7_ccmmd`). `exp5_3class`/`exp6_3class` early-stopped at epoch 17, best 13 (`exp7_progress_report.md:238`; `metrics.json`).
- **Tier 1 bottleneck sweep**: max 20 epochs, patience 5, batch 128, Adam lr 1e-3; actual epochs run 14 (K=256) / 15 (K=64) / 20 (K=16) per `outputs/exp7_bottleneck_K*/metrics.json`; **whole three-K sweep ≈ 50 minutes**, and the sweep log timestamps run 19:20 → 21:10 on 2026-05-17 including latent exports (`tier1_bottleneck_log.md:41`; `outputs/_t1b_sweep_log.txt`).
- **Tier 2**: Config A ≈ **21 min** (15 epochs × 85 s), Config B ≈ **9 min** (10 epochs × 53 s), latent export ≈ 3 min per run, evaluation ≈ 1 min per run — **≈ 40 min total** (`2026-05-24…:787-789`).
- **Evaluation cost is negligible next to training**: the full 4-block Tier-1 evaluation over four configs took **110 seconds** (`tier1_bottleneck_log.md:61`).
- **Trainable-parameter footprint**: Exp 7 trains **312,643** parameters (head 3,075 + adapters 309,568) on a frozen backbone (`exp7_progress_report.md:58`); Tier 2 trains ~**1.2%** of all weights (`2026-05-24…:551`).
- **Storage**: each bottleneck checkpoint ≈ 119 MB, the 18 K-d latent NPZs ≈ 95 MB total (`tier1_bottleneck_log.md:220-228`); per-config Tier-1 evaluation summaries ≈ 103 KB each (`:233-241`).
- **Analysis budgets** (CPU, scikit-learn): dim scan and Tier-1 blocks used N_BOOT 200 / N_PERM 50 for tractability; Phase B2 and anatomy blocks used N_BOOT 1000 / N_PERM 200; the 2026-05-24 runs used 1000/1000 (`track1_latent_dim_log.md:33`; `tier1_bottleneck_log.md:59-61`; `2026-05-24…:449-450`).
- **Order-of-magnitude estimate for one seed of the headline pipeline**: the student's own quote to the supervisor was "2 extra ~1 hr GPU runs per seed. Total 4 extra hours" for two extra seeds (`supervisor_meeting_2026_04_29_supplement.md:119-120`), i.e. **≈1 GPU-hour per Exp-7-class run**. Summing the documented runs, the entire pre-August experimental programme is on the order of **tens of GPU-hours on one consumer GPU** — worth stating plainly, because it is a genuine sustainability point for the Declarations chapter.
- **What was never spent**: the multi-seed replication (Phase D, 3–5 seeds) was never run, so no error bars exist on any pre-`exp8` headline number (`SUPERVISOR_TODO.md:23-30`).

---

## 7. Gaps in the record (state these rather than guessing)

1. **June–July 2026 is empty**: no commits and no files with a June or July 2026 mtime. The only dated marker inside that window is a literature note ("the SAE lane closed on 28–29 July 2026", `reports/2026-08-12_pivot_representational_geometry.md:148`). The poster was due 2026-06-10 but its PDF is not in the repo.
2. **Weeks 2, 3 and 6+ have no weekly report.** Only `README_w1`, `README_w4` and `README_w5` exist, and `README_w5` was deleted from the tree in the 2026-05-10 restructure (recoverable at `git show 30057ce:"weekly report/README_w5.md"`).
3. **No minutes exist for any meeting.** Every "supervisor said" in the record is the student's rendering in an agenda, an action list or a draft email. Attribute accordingly in the thesis.
4. **`reports/` is gitignored**, so the written record has no commit history of its own; file mtimes are the only dating evidence for several documents (e.g. `experiment_report.html` 2026-03-13 vs `experiment_report.md` 2026-04-16, which is the basis for dating Exp 1–6 to ~March–April).
5. **B1 (APD → QTc on PTB-XL) was never run**, and neither were Phase C prototypes, Phase D multi-seed, D2 closed-loop retrieval or D3 counterfactuals — all appear in plans and none in results. The August memory confirms D2/D3 are out of scope (memory `project_research_state`).
