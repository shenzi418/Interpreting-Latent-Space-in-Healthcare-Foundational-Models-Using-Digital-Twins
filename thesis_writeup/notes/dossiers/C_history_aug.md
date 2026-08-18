# Project history dossier, part 2 — 10–13 August 2026

*Audit → leadfix → breakthrough → geometry → fidelity audit.*
Compiled 2026-08-17 from `reports/` (read-only). Feeds Results §4.4, §4.5 and Methods.
Every number below carries its pipeline, encoder, scaler, n and source file. Where the
2026-08-13 adversarial audit corrected an earlier number, the corrected value is given
and the correction is flagged **[corrected 08-13]**.

---

## 0. Executive summary (10 lines)

1. **10 Aug**: a silent aVL/aVF transposition in every MedalCare batch was found by reading `scripts/datasets.py`, confirmed by limb-lead physics identities, and shown to have masked cross-domain territory transfer (p 0.756 → 0.002, no retraining).
2. The same day's repo audit found three further defects (PTB-XL filter kept STTC not CD; `predict_proba` columns transposed into `roc_auc_score`; MedalCare global-scalar vs per-lead z-score) plus a structural label contradiction (`LCX_0.3_post`) that capped the 4-class oracle at macro-F1 0.8643.
3. The decision point was measured before any retraining: **C2ST stays 1.0000** on lead-corrected latents (MMD −18%, kNN mixing 4×) — the alignment dead-end survives the fix.
4. **11 Aug (overnight)**: five `exp8_leadfix_*` encoders retrained on corrected data; an INLP `max_iter=20` artifact briefly resurrected "alignment is achievable" and was then **withdrawn** by a nonlinear C2ST (GBDT 0.9999 at the same k).
5. The replacement mechanism — the gap is **over-determined** (marginals alone → 1.0000; dependence alone → 0.9993 mean) — replicated on 6/6 checkpoints down to a 16-d bottleneck.
6. A pre-registered decisive run (Part 16, written 09:15 before the file existed) returned the **spatial54 control beating the latent cross-domain 5/5** — and a 77-cell lead-permutation sweep showed C2ST is constitutionally blind (spread 1e-5) while transfer is only a *gross*-corruption detector (39/66 transpositions improve it).
7. **11–12 Aug**: an integrity audit plus a per-lead probe map located the mechanism — MedalCare writes territory into **ST**, PTB-XL into **Q/R** — and a MedalCare-only encoder (9.9× power, n=4324) **retracted the control's win**: the primary endpoint is a null (p=0.162 / 0.695).
8. The pre-registered Track 3 rule was scored: **4-class POSITIVE not met** (0.3440 / 0.3357 vs ≥0.45); **2-class PARTIAL met under the strict scaler, missed under the legacy one** (0.6521 / 0.6299 vs ≥0.65).
9. **12 Aug**: pivot from distribution alignment to representational geometry; Tier-1 delivered six encoders × n=4324 group-disjoint — in-domain decoding strong in both domains, readout planes at the **random-plane floor**, and the synthetic prior **worse than scratch at every label budget**.
10. **13 Aug** (already digested elsewhere): the adversarial audit retracted CLAIM 2 / S(b) / b̂, imposed the constant-predictor floor (0.29216 PTB-XL), renormalised the anchor ranks, and reopened the scaler question as supervisor Q1.

---

## 1. Timeline, 10 → 13 August 2026

| date/time | what happened | source (file § / part) |
|---|---|---|
| **08-10** | `scripts/datasets.py:93-95` read: MedalCare `input_leads` declared `…aVR, aVF, aVL…` while `prepare_medalcare.py:53` writes standard order. Bug confirmed by limb-lead identities on 6 records. | `2026-08-10_lead_order_bug_diagnostic.md` §1 |
| 08-10 | Involution 2×2 diagnostic on frozen `exp7_baseline` (no retraining): correct/correct cross-domain macro-F1 **0.3278, p=0.0020** vs as-shipped 0.2132, p=0.756; sign-reversal interaction is the load-bearing evidence. | lead-order diag §2–§4 |
| 08-10 | Second mismatch identified: MedalCare global-scalar z-score vs PTB-XL per-lead. Inference-time test gives best cross-domain AUC in project history (0.587) but worse macro-F1 (0.242) — prior/calibration shift, must be retested after retraining. | lead-order diag §3 |
| 08-10 | Full repo audit: 4 confirmed defects (**L**, **F**, **A1**, **D1**) + 5 substantive statistical issues + ~15 minor; verdict table on every existing result (SURVIVES / INVALIDATED / UNKNOWN / STRUCTURAL). | `2026-08-10_repo_audit_and_rerun_plan.md` §0, §2 |
| 08-10 | **Decision point measured**: C2ST = 1.0000 (cv and held-out) under swapped, leadfix, and leadfix+per-lead-z; MMD 0.17568 → 0.14471; kNN mixing 0.0026 → 0.0109. Alignment wall is real. | repo audit §5; `outputs/analysis/leadswap_diag/c2st_leadfix.json` |
| 08-10 | Reproducibility failure recorded: the May headline results were produced by uncommitted code (10 modified + 29 untracked files). | repo audit §2.5 |
| **08-10 → 11 (overnight)** | Stage 0 freeze: commit `7839113`, annotated tag **`pre-leadfix`**, 40 files. Not pushed. | `EXECUTION_LOG_2026-08-10.md` Stage 0 |
| 08-11 | Stage 1 code fixes 1a–1j (L, N, F, A1, A2, A3, D1, M6, m1–m11) + doc inversion of the lead rule in `CLAUDE.md` / `data-pipeline.md`. | EXEC LOG Stage 1 |
| 08-11 | Stage 1b: `data/theta_mi_*.npz` regenerated under the D1 fix — 450/100/100 rows relabelled `LCX_0.3_post` Inferolateral → Anterolateral; all other fields bit-identical to backups. | EXEC LOG Stage 1b |
| 08-11 | Stage 2 free re-measurements: phase_b2 ×4 configs, eval_decoding_lowK, concept5, dim_scan, inlp_alignment (**not converged**, 19 directions, bal-acc 0.7603), inlp_lowK (crashed cosmetically after all science completed). | EXEC LOG Stage 2.1–2.5, Part 4.5 |
| 08-11 02:50 | Stage 3 training begins: `exp8_leadfix_baseline` (2864 s, best epoch 30). Driver later dies on a `UnicodeEncodeError` from an em-dash print; 3 of 5 runs then fail with `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED` (self-inflicted resource contention). | EXEC LOG Stage 3 |
| 08-11 04:25–06:14 | Retry driver recovers `dual`, `globalz`, `K64`; a second export defect (`--model-type single` hard-coded) is found and fixed with `--model-type auto`; all six dual exports rc=0 at 06:14:51. **Stage 3 complete, 5/5.** | EXEC LOG Part 6.2, 6.3 |
| 08-11 (overnight) | **Breakthrough Part 1**: `inlp_alignment.py DEFAULT_MAX_ITER = 20` identified; frontier to k=90 drives linear C2ST 1.0000 → 0.5001 at a cost of 21 AUC points of transfer. Claim: "alignment is achievable and costly". | EXEC LOG Part 1; `2026-08-11_breakthrough_analysis.md` §1 |
| 08-11 | **Part 2 — three withdrawals.** Euclidean-cosine orthogonality was a metric artifact (corr under S_ptbxl 0.4306, 4.7× random); linear domain identity is **2-dimensional**, not 90; and the alignment claim itself fails — at k=90 GBDT C2ST 0.9999, MMD² p=0.005. **"There is no alignment/transfer tradeoff."** | EXEC LOG Part 2; breakthrough §11 |
| 08-11 | **Part 3 — first positive**: residual is a per-coordinate **dispersion** mismatch (mean std ratio 1.95); diagonal CORAL recovers **+0.0645 ± 0.020** macro-AUC (3 checkpoints × 5 seeds), split-half control flat at −0.001, saturates at 50 unlabelled target samples. | EXEC LOG Part 3; breakthrough §12 |
| 08-11 | **Part 4** — a C2ST of exactly 0.5000 withdrawn (greedy-tree induction never started on an exact-ECDF rank map); corrected verdict flips MARGINAL → **OVER-DETERMINED**; quantile matching gives +0.1254 net transfer. | EXEC LOG Part 4; breakthrough §13 |
| 08-11 | **Part 5** — the **target-scaler defect** fixed (`--scaler-domain {target,source}`, default `target`); over-determination replicates **6/6** at k=0. | EXEC LOG Part 5; breakthrough §13.6 |
| 08-11 | **Parts 6–7** — scaler A/B lands: source→target improves all 32 cross-domain blocks (mean +0.0775) but flatters the control; the strict `target_pool` arm then takes the latent to **16/16 blocks clearing their own null (max p 0.0217)** while the control's best block collapses 0.554 (p<0.001) → 0.381 (p=0.82). | EXEC LOG Parts 6, 7; breakthrough §14, §14.6 |
| 08-11 | **Part 8** — the 6-feature NeuroKit2 control is diagnosed as **structurally incapable** (one ST feature averaged over V2–V6, one T amplitude on lead II); `extract_ecg_features_spatial.py` builds the 54-feature replacement; drop-in verified bit-identical on the 6 shared columns. | EXEC LOG Part 8, 8.3b; breakthrough §15 |
| 08-11 07:46–08:38 | **Parts 9–13** — replication over 4 configs (5/8 blocks significant in every one); C2ST **1.0000 on all five corrected encoders**; K64 dissociation (in-domain survives, 0/8 cross-domain blocks significant); dim_scan linear C2ST first sub-1.0 values then **withdrawn** by a GBDT re-score (min 0.9948); an imputation leak in `transfer_control.py` caught by a too-good 0.9918; tier1 pass 2 over five encoders. | EXEC LOG Parts 9–13 |
| 08-11 | **Part 14** — Stage 4.1 (MI-stage control) declared **not runnable** on the exported split: fold 10 holds 21 acute rows (14 with territory) vs 201 chronic. | EXEC LOG Part 14 |
| 08-11 09:15 | **Part 16 — pre-registration** written before the output file existed: primary endpoint `pipelineA / exp8_leadfix_baseline / cross_domain_4c`, ±0.03 band, six decision rules, four quoting constraints, and a recorded expectation. | EXEC LOG Part 16 |
| 08-11 09:55–11:35:59 | **Parts 17–18** — strict-scaler all-5 (4/5 transfer, K64 fails); then the decisive spatial54 arm: **control 0.3442 (p=0.0001) vs latent 0.2786 (p=0.0435)**, control wins 5/5; the finding is the **sign reversal** (in-domain Δ +0.1367…+0.1493, cross-domain Δ −0.0396…−0.0878). | EXEC LOG Parts 17, 18; breakthrough §16 Rank 1 |
| 08-11 | **Part 19 — Stage 4.2** 77-cell lead-permutation sweep (16.2 min): linear C2ST spread **1e-5**, GBDT **9e-5**, neither correlating with damage; transfer detects random permutations but **0/66 transpositions**; the historical aVL↔aVF cell ranks **70/77** in damage. | EXEC LOG Part 19; breakthrough §17 |
| 08-11 ~21:00 | Integrity audit run on request; spatial54 scaler ablation launched (`source` and `target_pool_measured`, 5 configs each). | `2026-08-11_integrity_audit_and_probe_map.md` §1, §4 |
| 08-11 | **Probe map** (`analysis/probe_feature_map.py`): ST is the outlier — ρ_in 0.799 → ρ_cross 0.150; 20 of 24 non-significant cells are ST_J60. Control excludes both boring explanations (within-real CV ρ=0.502). η²: MedalCare ST 0.056 / Q 0.007 / R 0.010; PTB-XL ST 0.016 / Q 0.077 / R 0.065. | integrity audit §2.1–§2.3 |
| 08-11 22:14 | **`exp8_leadfix_medalonly` trained** (MedalCare-only, no PTB-XL gradient step) — frees all ten PTB-XL folds, n=438 → **4324** (9.9×). | integrity audit §6.1–§6.4 |
| **08-12** | **§6.4b — the clean-arm verdict**: the control's cross-domain win **does not survive**. Primary endpoint at n=4324: Δ=+0.0145 (p=0.162, `target`) and +0.0041 (p=0.695, strict). Control reproduces to 4 dp in every fold-10 cell, so only the latent moved. | integrity audit §6.4b |
| 08-12 | **§6.4c — Track 3 scored**: 4-class 0.3440 / 0.3357 (POSITIVE not met); 2-class 0.6299 / **0.6521** (PARTIAL met under strict only). Twelve-row paired grid extracted; 2 of 12 nominal hits, opposite directions, neither surviving Holm. | integrity audit §6.4c |
| 08-12 | Missing paired cell rerun (`outputs/phase_b2_baseline_fold10_measscaler_paired/`): control 0.3304, latent 0.2797, Δ=−0.0507, p=0.098 — grid complete. | integrity audit §6.4b |
| 08-12 | **Pivot proposal**: C2ST is the wrong instrument (formal literature reason); territory is a **circular manifold** in both domains; the two circular codes sit in near-orthogonal subspaces; SOTA/novelty audit (SAE lane closed 28–29 Jul 2026; sim→real cell empty). | `2026-08-12_pivot_representational_geometry.md` §0–§4 |
| 08-12 | **Acuity test** (`analysis/acuity_stratified_transport.py`): the ST-vs-Q/R *explanation* fails — acute stratum (n=97) at chance (R 0.137, p=0.53), power-matched chronic scores higher in 88% of draws. | pivot §3.5 |
| 08-12 | **Tier-1 geometry results**: 6 encoders, n=4324 PTB-XL (3794 patients) / n=6547 MedalCare (1100 runs), group-disjoint CV, permutation-calibrated. Six findings incl. subspace overlap at the random floor and negative prior value. | `2026-08-12_tier1_geometry_results.md` §0–§4 |
| **08-13** | Adversarial audit: **CLAIM 2 / S(b) / b̂ RETRACTED** (Kempter 2012 prior art); constant-predictor floor **0.29216 / 0.09319** confirmed as a supremum; anchor ranks renormalised to 9–11/24 source, 7/24 target; scaler dichotomy reopened as supervisor **Q1**; fidelity audit F1/F2/F3 established as the centrepiece. | `2026-08-13_fidelity_audit_and_final_verification.md` A.1–A.7, C.1–C.4 |

---

## 2. Fact cards

### Card A — the four repo-audit defects, the physics-identity check, and the 39/66 caveat

**A1. What "the four defects" means — two different framings exist; do not conflate them.**

`reports/2026-08-10_repo_audit_and_rerun_plan.md` §0 tables four defects as **L, F, A1, D1**.
`CLAUDE.md` lists four as **lead order, normalisation, PTB-XL filter, AUC columns** (i.e. L, N, F, A1).
The union is five: L, N, F, A1, D1. The audit treats **N** as part of the lead-order writeup and
**D1** as "STRUCTURAL — needs a decision, not just a rerun" (§2.4). Safest thesis wording:
*"four coding defects (lead order, normalisation convention, class filter, probability-column order)
plus one structural labelling contradiction in the simulator's own metadata."*

| code | defect | location (pre-fix) | how it was fixed | file that fixed it |
|---|---|---|---|---|
| **L** | MedalCare aVL/aVF transposed in **every batch ever trained on**; `wfdb.rdsamp` discarded `sig_name` so nothing could disagree | `scripts/datasets.py:93-95`, applied at `:173`, loaded at `:159` | new `_reorder_leads()` reindexes by WFDB `sig_name`, mirroring `PTBXLDataset._reorder_leads` (`:661-674`), and **raises** if `sig_name` is missing or a target lead absent | `scripts/datasets.py` (EXEC LOG 1a) |
| **N** | MedalCare `z_score_normalization` used a single **global scalar** mean/std over the whole (12, T) array; PTB-XL normalised **per lead** | `scripts/datasets.py:126-127` | `per_lead_norm: bool = True` added, defaulting to the PTB-XL-matching convention; `False` reproduces the old behaviour exactly for ablation (`exp8_leadfix_globalz`) | `scripts/datasets.py` (EXEC LOG 1b) |
| **F** | PTB-XL 3-class filter kept columns `[0,1,2]` = (NORM, MI, **STTC**) instead of `[0,1,4]` = (NORM, MI, **CD**); came from `list(PTBXL_REMAP.values())` where `.keys()` was meant | `scripts/finetune_multilabel.py:664` | `keep_indices = list(PTBXL_REMAP.keys())` | `scripts/finetune_multilabel.py` (EXEC LOG 1c) |
| **A1** | `macro_ovr_auc` scored `y_true` in `TERRITORIES_4C` (anatomical) order against `proba` in `model.classes_` (alphabetical) order — first two columns swapped on **every** call; a bare `except` turned ~32 AUCs into `null` | `analysis/eval_decoding_lowK.py:261-295` | `macro_ovr_auc` now takes `proba_labels` and reorders columns, raising on a missing label; `score_block` gained passthrough; all call sites pass `proba_labels=list(model.classes_)`; subset branch renormalises rows; bare `except` now reports type/labels/shape | `analysis/eval_decoding_lowK.py` (+ `analysis/concept5_classifier.py`, `scripts/_diag_leadswap_ptbxl.py`) (EXEC LOG 1d) |
| **D1** | `LCX_0.3_ant` and `LCX_0.3_post` occupy the **identical φ wedge** ([+2.003, +3.139] / [+2.003, +3.140], circ-mean +2.57 both) but carry different territory labels → oracle ceiling acc 0.9158 / macro-F1 **0.8643** with Inferolateral recall pinned at exactly 0.500 | `scripts/build_medalcare_isch_targets.py:86-100` | `derive_territory_4c_from_phi` (same wedges as `hardcoded_phi_to_4c`); folder labels preserved as `territory_4c_folder`; build **raises** on any disagreement other than the documented D1 case | `scripts/build_medalcare_isch_targets.py` (EXEC LOG 1g) |

Also fixed in the same pass, and thesis-relevant: **A2** (Pipeline-B calibrator resubstitution —
the reported in-domain 0.998 was fit and scored on the same 1,200 rows; now fit on MedalCare
train, `calibrator_fit_pool: "medalcare_train"` recorded in JSON); **A3** (ridge permutation null
had no intercept — old null R² mean −77.730 vs new −0.156, so *any* non-catastrophic model pinned
p at the floor; verified against sklearn to 1.4e-14); **M6** (two incompatible `circular_r2`
definitions, one with a docstring falsely asserting they matched; the local one now raises
`NotImplementedError`); **A5** (`N_PERM_BINARY` 200 → 10000); **m1–m11**.

**Downstream data consequence of D1 (carry into every 4c table):** class balance shifted —
train Anterolateral 850 → **1300**, Inferolateral 900 → **450** (now the minority class at 8.4%).
Cross-domain comparisons against PTB-XL (Inferolateral n=32) are minority-vs-minority, and any
macro-F1 comparison against pre-fix 4c numbers is **not like-for-like**.

**A2. The physics-identity lead check — values and tolerances.**

Identities: `aVR = −(I+II)/2`, `aVL = (I−III)/2`, `aVF = (II+III)/2`. Relative RMS error of each
stored channel against each identity, on **raw** (un-normalised) data:

| | ch4 vs aVL | ch4 vs aVF | ch5 vs aVL | ch5 vs aVF |
|---|---|---|---|---|
| MedalCare (6 records) | **0.0092–0.2611** | 1.3843–1.7622 | 2.0652–2.9683 | **0.0080–0.1163** |
| PTB-XL (`00001_hr`) | **0.0057** | 1.8434 | 1.1444 | **0.0093** |

Re-verified in Stage 1a on raw WFDB: ch4-vs-aVL RMS **0.034**, ch5-vs-aVF **0.019**; manifest
`lead_order` column agrees with `sig_name`.

Loader-level verification, 4 records × 12 leads, correlating loader output channel *j* against
`raw[sig_name.index(TARGET[j])]`: **worst corr = 1.000000 (PASS)**. Against the *old* index list
`[0,1,2,3,5,4,6,7,8,9,10,11]`: ch4 corr **−0.9001**, ch5 corr **−0.9001**, all ten other channels
+1.0000. aVL and aVF are near-antiphase, which is why swapping them reflects the frontal-plane axis.

> **Methodological note, from EXEC LOG 1a, worth reproducing in Methods.** The identity check is
> valid **only on raw data**. Checking `aVL=(I−III)/2` on *loader output* gave 0.62–0.70 and looked
> like a failure: neither normalisation convention preserves linear relations among leads (global
> z adds a per-record offset; per-lead z rescales each lead independently). On normalised data the
> correct test is the name-indexed correlation above.

**A3. The 39/66 caveat — what the lead fix may and may not be argued from.**

From the 77-cell sweep (Part 19 §19.4): introducing aVL↔aVF on the **target** side at inference
gives macro-F1 **0.2817 vs identity 0.2599 (+0.0218)**, i.e. the historical bug cell ranks
**70/77 in damage** — it scores *above* correct order. **39 of 66** arbitrary transpositions also
improve transfer.

- **The fix is not in doubt**: it rests on the physics identity verified on the signals themselves,
  independent of any transfer number. The two interventions also differ — the audit retrained the
  encoder with corrected **source** leads; the sweep permutes the **target** at inference with the
  probe never refit.
- **What is overturned is the evidentiary weight of the corroborating statistic.** Anywhere the
  writeup leans on p 0.756 → 0.002 as *independent confirmation*, it must instead lean on the
  lead-identity check and report the transfer movement as **"consistent with"** the fix.
- `CLAUDE.md` already carries this wording. It is on the 26-item forbidden-phrasings list by
  implication; treat "transfer confirmed the lead fix" as forbidden.

---

### Card B — the `exp8_leadfix` run family

All five trained under Stage 3 with defects L, N (as configured), F fixed, on regenerated
(D1-corrected) θ targets. `exp8_leadfix_medalonly` was added 08-11 22:14 as option (c) of the
power fix. Artifact contract per run: `outputs/<run_id>/{args.json, metrics.json,
per_class_metrics.csv, checkpoints/linear_best.pt}` + six latent cells
(2 domains × 3 splits) under `outputs/latents/<run_id>_<domain>_<split>/latents.npz`.

| run_id | what it is | best epoch | val MedalCare acc / F1 / AUC | val PTB-XL acc / F1 / AUC | train wall-clock |
|---|---|---|---|---|---|
| `exp8_leadfix_baseline` | shared-head, correct leads + filter — **the new reference** | 30/30 | 0.9799 / 0.9438 / 0.9974 | 0.8839 / 0.8325 / 0.9449 | 2864 s |
| `exp8_leadfix_ccmmd` | shared-head + class-conditional MMD (λ=0.1) — "does alignment still fail once the basis is right?" | 15/19 | 0.9739 / 0.9323 / 0.9964 | 0.8788 / 0.8270 / 0.9431 | 1830 s |
| `exp8_leadfix_dual` | dual-head (exp5-equivalent) — restores the 2×2 architecture ablation | 15/19 | 0.9750 / 0.9217 / 0.9983 | 0.8847 / 0.8392 / 0.9496 | 2301 s |
| `exp8_leadfix_globalz` | normalisation ablation: legacy **global-scalar** z-score, now interpretable | 14/18 | 0.9727 / 0.9268 / 0.9954 | 0.8805 / 0.8314 / 0.9441 | 1965 s |
| `exp8_leadfix_K64` | bottleneck `Linear(1024,64)→GELU→Linear(64,n)` on the baseline checkpoint, 20 epochs | 18/20 | 0.9876 / 0.9644 / 0.9986 | 0.8879 / 0.8402 / 0.9471 | 1735 s |
| `exp8_leadfix_medalonly` | **`--shared-head --medalcare-only`**: no PTB-XL gradient step, `pos_weight` from MedalCare only, checkpoint selected on MedalCare val F1 alone | 11/15 | 0.9730 / 0.9230 / 0.9980 | **0.6557 / 0.4569 / 0.6406** (passive readout) | — |

Test-split figures (best epoch), for the four with test rows logged: baseline MedalCare
0.9737/0.9275/0.9975, PTB-XL 0.8801/0.8280/0.9404; ccmmd 0.9726/0.9289/0.9972 and
0.8743/0.8234/0.9380; dual 0.9738/0.9269/0.9979 and 0.8829/0.8345/0.9451; globalz
0.9655/0.9104/0.9950 and 0.8752/0.8234/0.9392; medalonly 0.9724/0.9268/0.9980 and
0.6542/0.4567/0.6267. Source: `outputs/exp8_leadfix_*/metrics.json`. Integrity audit §1
states the group ranges as val MedalCare F1 0.9217–0.9644 / AUC 0.9954–0.9986 and val PTB-XL
F1 0.8270–0.8402 / AUC 0.9431–0.9496 — **no degenerate encoders**.

**In-domain θ-territory decoding by encoder** (Phase-B2 pipeline, MedalCare test n=1200,
Pipeline A 4-class macro-F1): baseline 0.6404, ccmmd 0.6520, dual 0.6529, globalz 0.6432,
K64 0.5234, **medalonly 0.6560**. (EXEC LOG Part 18 §18.4; integrity audit §6.4b.)

**Why `medalonly` is the headline encoder** (integrity audit §3, §6.1–§6.4b):

1. PTB-XL folds 1–8 were the other encoders' training set and fold 9 their validation set
   (`scripts/datasets.py:569-575`), so only fold 10 (n=438 endpoint rows) was ever clean.
   The 54-feature control never trains at all — so using folds 1–9 would bias **exactly** the
   comparison that produced the sign reversal.
2. `--medalcare-only` is otherwise **byte-identical** to `exp8_leadfix_baseline`: architecture,
   shared 3-class label space, adapters, LRs, schedule, seed. `args.json` differs in 5 of 50 keys
   (`medalcare_only`, `run_id`, `overwrite`, two timestamps); checkpoint key layout identical
   (523 keys, 14 adapter keys). So the pair isolates **PTB-XL exposure** as the single variable.
3. Power: PTB-XL rows 2198 → 21799; MI-present 550 → 5469; **primary 4-class endpoint 438 → 4324
   (9.9×)**. Per-class support {Inf 196, AS 168, AL 42, IL 32} → {1914, 1624, 455, 331}. The clean
   9.9× ratio shows fold 10 was representative — a power gain, not a population change.
4. Row-alignment hardened at the same time: `export_latents.py` now writes `ecg_id` for PTB-XL;
   `phase_b2` asserts feature-rows == CSV-rows == latent-rows and compares `ecg_id` element-wise;
   the all-folds export matches `data/ptbxl_mi_subclass_allfolds.csv` row-for-row, and the fold-10
   CSV is an exact in-order subset.
5. Fold-10 vs all-folds exports of the same encoder agree at per-row cosine ≥ 0.9999999 and
   argmax agreement 1.000; the 1.3e-3 max elementwise difference is TF32 convolution
   nondeterminism from batch grouping (mean relative 2.5e-6).

⚠ **Open observation, not a claim** (integrity audit §5 item 4): medalonly is better than the
PTB-XL-supervised encoders at *both* legs — in-domain 0.6560 vs 0.6404 and cross-domain 0.3111 vs
0.2786 at matched rows. That is backwards from intuition, rests on **n=1 training run, one seed**,
and needs a seed replicate before being called an effect.

---

### Card C — the pre-registered Track 3 decision rule and its verdict

**The rule, verbatim** (fixed long before the numbers existed; memory `project_research_state`,
"Key outcome to watch"; quoted in integrity audit §6.4c):

> **POSITIVE**: 4-class macro-F1 ≥ 0.45 with permutation p < 0.01
> **PARTIAL**: 2-class Anterior-vs-Inferior macro-F1 ≥ 0.65 with p < 0.01
> **NEGATIVE**: confirms the synth-real representational gap is fundamental

**Scored on** `exp8_leadfix_medalonly`, all folds, **n=4324**, latent arm, Pipeline A cross-domain,
`analysis/phase_b2_infarct_decoding.py`; artifacts `outputs/phase_b2_medalonly_allfolds_target/`
and `outputs/phase_b2_medalonly_allfolds_target_pool_measured/`:

| endpoint | `target` (legacy) | strict (`target_pool_measured`) | threshold | verdict |
|---|---|---|---|---|
| 4-class (**primary**) | 0.3440, p=1e-4 | 0.3357, p=1e-4 | ≥0.45, p<0.01 | **not met** — p passes, F1 short by ~0.11 |
| 2-class (secondary) | 0.6299, p=1e-4 | **0.6521**, p=1e-4 | ≥0.65, p<0.01 | **met under strict, missed under `target`** (misses by 0.0201) |

**Frozen wording — use exactly this** (integrity audit §6.4c and `CLAUDE.md`):

> *"PARTIAL met under the pre-specified strict scaler; not met under the legacy one"* — with the
> 4-class **primary** endpoint leading, and **do NOT** flatten this to either a clean NEGATIVE or
> a clean PARTIAL.

**Both sides of the scaler argument, as recorded:**

- *For strict being primary*: §1.5 ranked the three scaler modes **before** any of these numbers
  existed and while the live concern ran the *other* way (that the control was flattered). It called
  `source` "a defect, not a baseline", `target` transductive **and** label-selected, `target_pool`
  corrupted for the feature arm. On this exact cell the latent also **beats the control**
  (Δ=+0.0195, paired swap p=0.0439, bootstrap p=0.0270, CI95 [−0.0000, +0.0390]).
- *Against*: margin is **0.0021** macro-F1; the arm-vs-arm sign flips under `target` (Δ=−0.0050,
  p=0.62); the Δ CI touches zero exactly; `target_pool_measured` **did not exist at
  pre-registration time** (garden-of-forking-paths exposure); and the rule's `p<0.01` is an
  against-chance p which at n=4324 is at the 1e-4 floor for *everything*, control included.

⚠ **The permutation-floor caveat is load-bearing.** At n=4324 `permutation_p_macro_f1` = 9.999e-05
(= 1/(10000+1)) for **both arms on both endpoints**. "Beats chance" is uninformative at this power.
**Never quote a floor p-value as evidence the latent works** — only the paired arm-vs-arm test
discriminates. Contrast the fold-10 baseline runs where the latent's own 4-class p was 0.043
(baseline), 0.074 (ccmmd), 0.0013 (dual), 0.0016 (globalz), 0.081 (K64) — 2 of 5 encoders failed
to beat chance at all.

**Status**: which scaler is primary is *"the single most consequential open decision in the
project"* and is a **supervisor call, not a run** (integrity audit §5 item 5 = supervisor Q1 on the
08-13 list). The pivot notes (§6 item 3) that under a representational-geometry framing the
scaler-primacy question becomes much less load-bearing, since the headline no longer rests on a
macro-F1 threshold.

---

### Card D — the twelve-cell paired latent-vs-control54 grid

**Statistic.** `paired_macro_f1()` in `analysis/phase_b2_infarct_decoding.py`, reported under
`paired_Z_vs_features` in `cross_domain_4c_pipelineA.json`. Both arms are fit on identical
MedalCare rows and score identical PTB-XL rows; the run refuses to report a "paired" number if the
two arms' truth vectors ever differ. Two nulls: a **bootstrap on Δ** over shared resample indices,
and a **paired swap permutation** (exchange the two arms' predictions per row, conditioning on the
predictions rather than the labels). It draws its own `derive_rng(cfg, "paired_Z_vs_features",
endpoint)` stream, so it shifts no downstream p-value.

Δ = latent − control54. Verified against the JSONs by `scripts/_audit_paired_grid.py`
(integrity audit §6.4c):

| # | scaler | encoder | rows | endpoint | Z | ctrl | Δ | CI95 | p_swap |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `target` | baseline | 438 | 4-class | 0.2786 | 0.3442 | −0.0656 | [−0.1230, +0.0005] | **0.0400** |
| 2 | `target` | baseline | 438 | 2-class | 0.5919 | 0.6268 | −0.0349 | [−0.1091, +0.0370] | 0.2544 |
| 3 | `target` | medalonly | 438 | 4-class | 0.3111 | 0.3442 | −0.0331 | [−0.0915, +0.0234] | 0.2857 |
| 4 | `target` | medalonly | 438 | 2-class | 0.6197 | 0.6268 | −0.0072 | [−0.0649, +0.0531] | 0.7715 |
| 5 | `target` | medalonly | **4324** | 4-class | 0.3440 | 0.3295 | +0.0145 | [−0.0061, +0.0353] | 0.1617 |
| 6 | `target` | medalonly | **4324** | 2-class | 0.6299 | 0.6349 | −0.0050 | [−0.0246, +0.0146] | 0.6207 |
| 7 | strict | baseline | 438 | 4-class | 0.2797 | 0.3304 | −0.0507 | [−0.1083, +0.0039] | 0.0978 |
| 8 | strict | baseline | 438 | 2-class | 0.6162 | 0.6209 | −0.0047 | [−0.0660, +0.0508] | 0.8364 |
| 9 | strict | medalonly | 438 | 4-class | 0.3193 | 0.3304 | −0.0111 | [−0.0744, +0.0525] | 0.7194 |
| 10 | strict | medalonly | 438 | 2-class | 0.6270 | 0.6209 | +0.0061 | [−0.0563, +0.0687] | 0.8893 |
| 11 | strict | medalonly | **4324** | 4-class | 0.3357 | 0.3316 | +0.0041 | [−0.0152, +0.0238] | 0.6949 |
| 12 | strict | medalonly | **4324** | 2-class | **0.6521** | 0.6326 | **+0.0195** | [−0.0000, +0.0390] | **0.0439** |

Row provenance: rows 1–2 from `outputs/phase_b2_smoke_paired/` (the run that first produced the
paired statistic, despite the directory name); rows 7–8 from
`outputs/phase_b2_baseline_fold10_measscaler_paired/` (rerun 2026-08-12 to complete the grid —
control returned 0.3304 and latent 0.2797, reproducing the stored point estimates exactly, so the
rerun only *added* the paired p).

**Verdict wording (frozen).** *"On cross-domain MI-territory decoding the 1024-d latent and the
54-feature hand-crafted control are **statistically indistinguishable**."* The sign flips but
nothing is detectable at any power. Exactly **two of twelve** tests reach nominal p<0.05, and they
point in **opposite directions** (row 1 favours the control, row 12 the latent); **neither survives
Holm** across the twelve.

**Encoder-independence is proven, not assumed.** The control never touches the latent and its `C`
is tuned on MedalCare features, so it must be encoder-invariant — and it reproduces to four
decimals in every fold-10 cell (**0.3442** in both `target` rows, **0.3304** in both strict rows).
Only the latent moved.

**The two changes disentangled** (integrity audit §6.4b):

| contribution | `target` | strict |
|---|---|---|
| **encoder** (fold 10, matched rows: 0.2786→0.3111 / 0.2797→0.3193) | **+0.0325** | **+0.0396** |
| evaluation set (medalonly, fold 10 → all folds) | +0.0476 | +0.0152 |
| total gap change | +0.0801 | +0.0548 |

The encoder effect is consistent in sign and magnitude across both scalers; the eval-set
contribution is not, and lies inside the fold-10 CI [−0.0915, +0.0234] — sampling noise, not a
population shift.

**Imputation confound excluded in the control's favour**: spatial54 all-54-finite coverage on the
endpoint rows is **75.3% all-folds vs 72.4% fold-10** (per-kind ≥99% both). The clean arm gives
the control marginally *better* inputs.

**The in-domain leg — the robust result.** It scores identical MedalCare rows (n=1200) in every
run, so it was always a controlled encoder comparison:

| encoder | latent | control54 | Δ | p |
|---|---|---|---|---|
| baseline (saw PTB-XL) | 0.6404 | 0.5037 | **+0.1367** | 0.0005 / 0.0001 (two paired draws; point estimate identical) |
| **medalonly** | **0.6560** | 0.5037 | **+0.1523** | **0.0001** |

`in_domain_4c` is **byte-identical** across the two clean-arm scaler runs, as it must be since
`--scaler-domain` applies to the PTB-XL leg only. Full in-domain sign-reversal table (EXEC LOG
§18.4, `target` scaler, fold 10): baseline +0.1367 / −0.0656; ccmmd +0.1483 / −0.0709; dual
+0.1493 / −0.0404; globalz +0.1395 / −0.0396; K64 +0.0198 / −0.0878. Pipeline B's in-domain blocks
say the same more loudly (Z +0.148 to +0.227).

**Corrected shape of the finding** (integrity audit §6.4b point 4): *the latent carries a large,
real, in-domain θ-territory advantage that does not transfer; the part that does transfer is no
better and no worse than 54 hand-crafted per-lead measurements.*

⚠ **Retention percentages** ("latent retains 42–49%, control 68%") are **illustration, not
evidence** — a retention ratio is normalised by each arm's own in-domain score, so it flatters
whichever arm starts lower, and the control does (0.5037 vs 0.6404). Quote the absolute
cross-domain numbers against their own nulls, and the sign flip.

⚠ **Not a dimensionality artifact**: `C` is tuned per arm on source CV and the latent selects the
*heaviest* regularisation available (C=0.01 vs control C=0.1, K64 C=1); and K64 — 64-d, narrower
than the 54-d control — transfers *worse*, not better.

---

### Card E — the spatial54 control: construction, claim, retraction

**Why it exists.** The predecessor control was the 6-feature NeuroKit2 vector
(`scripts/extract_ecg_features_neurokit2.py:72-84, :177-196`): `QRS_duration_ms`, `QT_interval_ms`,
`P_duration_ms`, `heart_rate_bpm` (no spatial content), `ST_J60_avg_mV` (**averaged over V2–V6**,
anterior only) and `T_amplitude_mV` (**lead II only**). Infarct territory is defined by *which*
leads deviate, so the control could not represent the anterior-vs-inferior contrast it was scored
on — **structurally incapable, not merely weak** (EXEC LOG Part 8.1; breakthrough §15).

**Construction** (`scripts/extract_ecg_features_spatial.py`), 54 columns:

```
ST_J60_<lead>   J+60 ms voltage            ST elevation / depression      12
Q_amp_<lead>    min in [R_onset, R]        pathological Q waves           12
R_amp_<lead>    voltage at R peak          R-wave progression loss        12
T_amp_<lead>    voltage at T peak          T-wave inversion territory     12
+ the original 6 global features                                           6
```

Two design decisions on record: **all 12 leads are read at the same sample index**, at fiducials
delineated once on lead II, so the measurement is the instantaneous QRS/T vector across the frontal
and precordial planes (per-lead re-delineation would misalign leads and destroy the spatial
quantity); and **`Q_amp` is included deliberately** because PTB-XL's MI population is predominantly
chronic, where ST elevation has resolved and pathological Q waves are the localisation marker.

**Strict-superset verification** (`scripts/_preflight_spatial54.py`, EXEC LOG Part 15.1;
column-name matched, `atol=0, rtol=0`): the 6 shared columns are **bit-identical** on all three
splits (12019 / 2386 / 2198 rows); fully-finite row fractions are **unchanged to four decimals**
(0.3088 / 0.2938 / 0.1815), i.e. the 48 new columns add **zero** missingness — every incomplete
row is incomplete because of the original 6. `nk2_ok` is elementwise equal to the 6-feature mask.

**Instrument check** (`scripts/_audit_spatial_features.py`, breakthrough §15.1; PTB-XL fold 10,
Anterior n=210 vs Inferior n=228, predictions fixed before looking). Cohen's d, anterior minus
inferior: P1 mean `Q_amp` d over V1–V4 = **−0.37**; P2 mean `Q_amp` d over II/III/aVF = **+0.97**
(region separation **1.35 d**); P3 lead specificity mean |d| 0.59 on territory leads vs 0.29 on
I/aVR/aVL/V5/V6 = **2.06×**. R-wave progression correct (V2/V3 R_amp d −0.62/−0.69); residual ST
elevation survives in V1–V3 (+0.36/+0.45/+0.52). **A 5-fold in-domain probe on just the 24 Q/R
columns reaches AUROC 0.909** (PTB-XL-only protocol, *not* comparable to any transfer number).

**The "control beats latent" claim, as published on 08-11** (EXEC LOG Part 18, pre-registered at
Part 16): primary endpoint `pipelineA / exp8_leadfix_baseline / cross_domain_4c`, `target` scaler,
n=438 — control **0.3442 (p=0.0001)** vs latent **0.2786 (p=0.0435)**, dC−dZ = **+0.0656**, more
than twice the ±0.03 pre-declared band, replicating **5/5** across configs
(latent 0.2564–0.3046, control 0.3442 in all five). Preconditions all checked first: determinism
guard passed (55 latent blocks, macro-F1 identical, 0.00 MC σ), control encoder-invariant to six
decimals, identical rows and class counts on both arms (438 cross: 168/42/196/32; 1200 in-domain:
400/300/400/100), `C` tuned per arm on source rows only.

**Scaler ablation on the same (fixed) encoder** (integrity audit §4): control wins 5/5 under
`target_pool_measured` too, mean gap **+0.0537** (baseline +0.051, ccmmd +0.045, dual +0.026,
globalz +0.049, K64 +0.098) — i.e. 76% of the published gap. Under the `source` scaler the control
wins only 2/5, mean gap +0.009 — but `source` is *"a defect, not a baseline"* and closes the gap by
**damaging the control** (control 0.3442 → 0.2657, −23%, identical in all five configs, while the
latent moves −7% to +6%).

**THE RETRACTION** (integrity audit §6.4b, 2026-08-12). The claim was **conditional on the encoder
having itself been supervised on PTB-XL**. On `exp8_leadfix_medalonly` at the very same 438 rows
the effect drops from p=0.040 to p=0.286 (`target`) and p=0.719 (strict); at n=4324 the primary
endpoint is a null in both (p=0.162, p=0.695). It was **never robust even with that supervision**:
the same baseline encoder under the strict scaler gives p=0.098 — *"the published claim rests on
exactly one significant cell out of the four (encoder × scaler) eventually run at n=438."*

**Imputation-artifact checks on the control (all passed).** (i) PTB-XL fold 10 is ~75% all-NaN
overall, but the **n=438 evaluation subset is 0% all-NaN in every class**; imputation is
source-median only and no rows are dropped, so both arms score identical rows (integrity audit
§1.4). (ii) The Part-12 missingness gate — score the missingness indicator alone — reads worst
**0.5364** against a 0.55 abort threshold, versus 0.847/0.863 for the leaking version. (iii) Three
of the 54 features carry substantial imputation (`QRS_duration_ms` 22.0% train / 33.6% test;
`QT_interval_ms` 20.4 / 30.7; `P_duration_ms` 19.9 / 26.1), the other 51 at ≤1.8% — and the
direction is conservative, since it handicaps the arm that won (EXEC LOG §18.7).

⚠ **`target_pool` is corrupted for the feature arm** (integrity audit §1.5; EXEC LOG Part 12.4):
it pools all 2198 PTB-XL rows, ~75% entirely imputed to MedalCare-train medians, deflating the
feature arm's per-column std by **1.78× (median)** and shifting its mean by **0.456 measured-std
units (median, max 2.93)**. The latent arm is unaffected (all 2198 latents real). This is why
`target_pool_measured` was added and why the `target_pool` ablation was killed mid-run.
**Every `ecg_features` column in `outputs/phase_b2_exp8_poolscaler/` is VOID** (control p-values
pinned at 0.9996/0.9994/0.9991/0.9993 across four configs).

**Earlier, superseded comparison** (EXEC LOG Part 13, breakthrough §16 Rank 5): one-vs-rest macro
AUROC on 438 rows gave Z 0.5674, global6 0.5465, spatial54 0.6123 (tuned C, leak-free); paired
bootstrap Z − spatial54 = −0.0451, 95% CI [−0.1003, +0.0064], P(control wins) = **0.952**. Reported
as *"the latent does not beat the control, and the direction is not resolved at n=438"* — this arm
is superseded by the macro-F1 permutation grid and by §6.4b.

---

### Card F — the 77-cell lead-permutation sweep (Stage 4.2)

**Design** (EXEC LOG Part 19; breakthrough §17). Script `scripts/_t42_leadperm_sweep.py`,
summariser `scripts/_t42_summarise.py`, artifact
`outputs/analysis/leadperm_sweep/leadperm_sweep.json`, 77 cells, 16.2 min.

- **Intervention**: PTB-XL test (the *target*) permuted at inference. Probe fit once on
  correctly-ordered MedalCare-train latents and **never refit** (`metadata.probe.refit = false`).
- **Encoder**: `exp8_leadfix_baseline`. **Cells**: identity + all C(12,2)=66 transpositions +
  10 seeded random full permutations.
- **Metrics**: territory macro-F1 (n=438, 2000-draw permutation null, 400-draw bootstrap CI),
  linear C2ST, GBDT C2ST on 19 pre-declared cells, multi-bandwidth MMD².
- **Precondition, passed exactly**: identity reproduced the stored
  `exp8_leadfix_baseline_ptbxl_test/latents.npz` to `max|d| = 0.000e+00`.
- Thresholds were **fixed in the script docstring before the table existed**.

**Pre-declared reading fired SUPPORTED:**

| criterion | threshold | measured | verdict |
|---|---|---|---|
| transfer macro-F1 spread, 66 transpositions | ≥ 0.05 | **0.0929** | YES |
| linear C2ST pinned | min > 0.99, spread < 0.01 | min 1.0000, spread **1e-5** | YES |
| GBDT C2ST pinned | min > 0.99, spread < 0.01 | min 0.9999, spread **9e-5** | YES |

Across cells spanning macro-F1 **0.1618 → 0.3201** (a factor of two), Spearman ρ vs macro-F1 =
**−0.040 (p=0.73)** linear and **−0.031 (p=0.90)** GBDT.

**Class breakdown:**

| class | n | mean F1 | min | max | mean MMD² |
|---|---|---|---|---|---|
| identity | 1 | 0.2599 | — | — | 0.1778 |
| limb–limb | 15 | 0.2802 | 0.2586 | 0.3201 | 0.2014 |
| limb–precordial | 36 | 0.2584 | 0.2272 | 0.2849 | 0.1940 |
| precordial–precordial | 15 | 0.2561 | 0.2312 | 0.2867 | 0.1945 |
| **random (full)** | 10 | **0.2012** | 0.1618 | 0.2245 | 0.2673 |

- **Gross corruption is detected**: randoms mean 0.2012 vs identity 0.2599; **7/10 below identity's
  95% CI [0.2162, 0.3011]**; **0/10 reach p<0.05**; random-vs-transposition Mann–Whitney
  **p = 2.1e-07**.
- **Single transpositions are not**: **0 of 66** fall below identity's CI; the two outside it are
  *above* (II↔aVR 0.3191, aVR↔aVF 0.3201); **39 of 66 score higher than correct order**.
- **MMD² reacts but not usefully**: range 0.1639–0.3123 (spread 72% of mean), separates randoms
  from transpositions (0.2673 vs 0.1958, Mann–Whitney **p = 3.9e-07**), identity has the **8th
  smallest MMD² of 77**; but within transpositions ρ = −0.146 (**p = 0.20**). It flags *change*,
  not *damage*.

⚠ **Governing caveat**: **identity itself is not significant — macro-F1 0.2599, p = 0.1139**; mean
bootstrap CI width 0.0793 (~30% of the point estimate). So the "transpositions undetected" result
is partly a power statement at n=438. The 2 cells beating identity's CI are 2 of 66 uncorrected
tests where ~3 are expected by chance — **not real improvements**; the 20/66 with p<0.05 are
likewise uncorrected.

⚠ **Protocol note**: 0.2599 here is a *different protocol* from Part 18's 0.2786 (probe fit on
source with source-fit scaler, no calibration) and **must not be quoted as a restatement of it**.

**Defensible claim (narrowed)**: *C2ST saturates at AUROC ≈ 1.0 in sim-to-real ECG and is therefore
constitutionally unable to detect a lead-order corruption; transfer detects gross corruption but
neither statistic resolves single-channel transpositions at n=438.* **Do not claim transfer is a
lead-order detector — it is a corruption detector.** Recommend MMD² over C2ST for gap monitoring,
with the caveat that it flags change without grading harm.

---

### Card G — the alignment dead-end, as restated in the August documents

**The three-condition table that settled it before any retraining** (`scripts/_diag_c2st_leadfix.py`
→ `outputs/analysis/leadswap_diag/c2st_leadfix.json`; frozen `exp7_baseline` checkpoint, MedalCare
test 2386 vs PTB-XL test 2198, domain-balanced by seeded subsampling, C2ST scaler fit on fold-train
rows only; "held" = fit on train pools, scored on untouched test pools):

| MedalCare condition | C2ST cv | C2ST held-out | MMD² (unbiased, multi-bw) | kNN mixing (k=10) |
|---|---|---|---|---|
| as shipped — aVL/aVF swapped, global z | 1.0000 | 1.0000 | 0.17568 | 0.0026 |
| leadfix — correct order, global z | **1.0000** | **1.0000** | 0.16604 | 0.0049 |
| leadfix + per-lead z | **1.0000** | **1.0000** | 0.14471 | 0.0109 |

Both fixes reduce first-order distances (MMD −18%, kNN mixing 4×) while C2ST does not move at all —
the *same* dissociation already recorded for single-bandwidth MMD, multi-bandwidth MMD,
class-conditional MMD and INLP in `reports/inlp_alignment_summary.md`. Three mutually independent
interventions produce it, which upgrades it from "our methods failed" to a property of the
representation (repo audit §5).

**On the retrained encoders** (EXEC LOG §9.3b, `1_c2st`, 11 s):

| config | C2ST cv | C2ST held | MMD | kNN mix |
|---|---|---|---|---|
| exp8_leadfix_baseline | 1.0000 | 1.0000 | 0.14447 | 0.0105 |
| exp8_leadfix_ccmmd | 1.0000 | 1.0000 | 0.13378 | 0.0117 |
| exp8_leadfix_dual | 1.0000 | 1.0000 | 0.13986 | 0.0121 |
| exp8_leadfix_globalz | 1.0000 | 1.0000 | 0.16296 | 0.0051 |
| exp8_leadfix_K64 | 1.0000 | 1.0000 | **0.32089** | 0.0020 |

Confirmed again by tier1 pass 2 over the same five encoders (EXEC LOG §13.1:
`outputs/tier1_eval_exp8/cross_config_table.md`, C2ST 1.0000 on all five; MMD_med 0.2002 for ccmmd,
the lowest of the five, with C2ST unmoved).

**The INLP story, in order** (this is the part most at risk of being told wrong):

1. `analysis/inlp_alignment.py` carries `DEFAULT_MAX_ITER = 20`. At k=20 linear C2ST reads 0.847
   and is still falling — the historical "wall" was partly a **search-budget artifact**
   (breakthrough §1).
2. Removing 90 linear directions drives held-out **linear** C2ST 1.0000 → 0.5001 while M→P transfer
   falls 0.7678 → 0.5606 (exp7_baseline). Rank-matched random projection control: C2ST 1.0000 ±
   0.0000, transfer 0.7598 — so not capacity loss (specificity gap +0.4999). In-domain AUC moves
   only −0.004.
3. **§11 withdrawal.** Every C2ST in the project's history is a **logistic regression**, and INLP
   removes linear directions — instrument and procedure share a blind spot. Re-measured held-out at
   the same k: whitened-128/k=2 → linear 0.5000, **GBDT 0.9987, MLP 0.9997, kNN 0.9961**, MMD²
   p=0.005; euclidean/k=90 → linear 0.5132, **GBDT 0.9999**. **The domains were never aligned.**
4. **There is no alignment/transfer tradeoff.** With the GBDT axis, the whole frontier is flat
   (1.0000 → 0.9999 from k=0 to k=160) while transfer falls to chance; the random control holds
   transfer at ~0.70 throughout. "A tradeoff requires that something was bought."
5. **Metric correction**: Euclidean cosine assumed isotropy (participation ratio **71.2 of 1024**);
   under the data covariance the class directions agree at 0.4306 mean (4.7× random). Linear domain
   identity is **2-dimensional**, not 90; removing those 2 costs nothing (transfer 0.6120 → 0.6163).
6. **Over-determination is the replacement mechanism** (§13.6, `domain_mechanism_replication.py`,
   6 checkpoints, k=0 raw latents): as-is ≥0.9998; **dependence-only mean 0.9993 (min 0.9968)**;
   **marginals-only mean 0.9999 (min 0.9997)**; both-destroyed floor mean 0.5032 (max 0.5197).
   The sharpest single pair in the project: on `exp7_bottleneck_K16`, INLP drives held-out **linear**
   C2ST 1.0000 → **0.5808** (beating a rank-matched random projection by 0.38), while a GBDT reads
   **0.9968 off the dependence structure alone with every marginal destroyed**.
7. Stage 2.5a re-ran `inlp_alignment.py` *as configured* (the historical measurement): **NOT
   CONVERGED**, 19 directions, domain balanced accuracy 0.7603 > stop_acc 0.55.
   Stage 2.5b `inlp_lowK` (linear C2ST): K=16 → 0.5278 pool / 0.5808 held-out, rank 7/16, converged;
   K=64 → 0.5832 / 0.8144, rank 37/64, not converged; K=256 → 0.7876 / 0.9582, rank 214/256, not
   converged.
8. Ruled out along the way: **label shift** (Zhao et al. ICML 2019 Thm 4.3) — JS divergence 0.145
   bits → joint-error floor 0.072 vs observed joint error 0.145, **bound slack by +0.073**; and the
   removed subspace is **neither clinical nor biophysical** (every excess R² negative: 6 NeuroKit2
   features best −0.069; θ: phi −0.076, z −0.120, size −0.057, rho_eps_max −0.137, territory_4c
   −0.024). The tempting headline "alignment destroys infarct-geometry directions" is **FALSE**.

**Standing rule established**: *a linear probe can only license a claim about linear structure.*
Every "aligned / not aligned" statement in `inlp_alignment_summary.md` and `exp7_progress_report.md`
is a statement about **linear decodability only**.

**Also withdrawn**: dim_scan's sub-1.0 linear C2ST readings (0.947 / 0.902 at K=8 under
single-domain PCA) — GBDT re-score returns 0.9948–0.9968, lowest GBDT anywhere in the 30-cell grid
= **0.9948** vs linear low 0.9038 (`analysis/dim_scan_nonlinear_c2st.py`,
`outputs/dim_scan_exp8/nonlinear_c2st.json`). *"Nothing about 'the first crack in C2ST' goes in a
chapter."*

---

### Card H — the probe map (ST vs Q/R) and the acuity test

**The instrument** (`analysis/probe_feature_map.py`; integrity audit §2). For each of the 48
per-lead spatial features (4 physiology kinds × 12 leads), a ridge probe is fit on MedalCare-train
latents and read out in both domains. Method choices that matter: the primary metric is **Spearman
ρ, not R²** (the domains disagree on per-coordinate scale by ~3×, so cross-domain R² would mostly
measure amplitude calibration — cross-domain R² is stored under a `_DO_NOT_QUOTE` suffix);
**targets are never imputed** (rows with NaN target dropped per feature, n recorded per cell);
1000 bootstrap, 10000 permutations, RNG keyed per (config, feature).

**Result — ST is the outlier** (median over 12 leads × 5 `exp8_leadfix_*` encoders):

| physiology | ρ in-domain | ρ cross-domain | Δρ | cells sig. |
|---|---|---|---|---|
| **ST at J+60 ms** | **0.799** | **0.150** | **0.637** | 40/60 |
| Q amplitude | 0.555 | 0.333 | 0.246 | 58/60 |
| R amplitude | 0.677 | 0.496 | 0.192 | 60/60 |
| T amplitude | 0.643 | 0.368 | 0.267 | 57/60 |

**20 of the 24 non-significant cells across all five encoders are ST_J60.** Worst individual cells
(`exp8_leadfix_baseline`): `ST_J60_III` 0.843 → −0.028 (p=0.510); `ST_J60_aVL` 0.840 → −0.008
(p=0.840); `ST_J60_V3` 0.765 → 0.022 (p=0.612). Insensitive to the evaluation pool (median ρ_cross
0.344 on all 549 MI rows vs 0.338 on the 438-row subset) and to the probe-side X-scaler
(source vs target_pool: median |Δ| 0.046, corr 0.877).

**Control — both boring explanations excluded** (`analysis/probe_feature_control.py`). A probe fit
*inside* the real domain (5-fold CV on PTB-XL latents, out-of-fold), against an in-domain probe
subsampled to the same ~439 training rows:

| | ρ in-domain @ matched n | ρ **within-real** (CV) | ρ cross-domain | transfer efficiency |
|---|---|---|---|---|
| **ST_J60** | 0.782 | **0.502** | 0.088 | **18%** |
| Q_amp | 0.481 | 0.613 | 0.350 | 57% |
| R_amp | 0.588 | 0.751 | 0.503 | 67% |
| T_amp | 0.519 | 0.662 | 0.383 | 58% |

ST **is** measurable on PTB-XL and the latent **does** encode it (ρ=0.50 within-real). What fails
is specifically the **MedalCare-fit readout direction** — a subspace mismatch, not an absence.
Robust across all five encoders (ST lowest efficiency in every one, 0.21–0.43 vs 0.45–0.75).

**The mechanism — η² of the 4-class territory label on each measurement, per domain**
(encoder-independent; a property of the *data*):

| | η² in MedalCare | η² in PTB-XL |
|---|---|---|
| ST_J60 | **0.056** | 0.016 |
| Q_amp | 0.007 | **0.077** |
| R_amp | 0.010 | **0.065** |
| T_amp | 0.034 | 0.031 |

Corroborated by scale: MedalCare ST deviations are 3–4× larger (std 0.20–0.58 vs 0.05–0.16).

**Clinical grounding** (integrity audit §2.6; both DOIs `papersflow.verify_citation`-resolved,
**do not re-key from memory**): acute ST criteria — Thygesen et al., *Eur Heart J*
2018;40(3):237–269, DOI 10.1093/eurheartj/ehy462; Q-wave regression over time — Das et al.,
*Circulation* 2006;113(21):2495–2501, DOI 10.1161/circulationaha.105.595892.

**Caveats attached in the source document** (§2.4): the η² values are modest in absolute terms —
*"the ratio between domains is the finding, not the magnitudes"*; multivariate structure could
differ from these univariate reads; ST transfer efficiency is unexplainedly higher for `globalz`
(0.425) and `K64` (0.408) than for baseline/ccmmd/dual (0.21–0.23), so the robust claim is the
**within-encoder ordering**.

**THE ACUITY TEST — the explanation fails.** Two independent measurements, both post-dating the
mechanism claim:

1. `analysis/acuity_stratified_transport.py` (pivot §3.5), readout fit on n=6547 MedalCare θ rows,
   never refit. Correction recorded there: PTB-XL acuity **is** on disk after all —
   `ptbxl_database.csv` carries `infarction_stadium1` (whole DB: Stadium I 166, I-II 5, II 88,
   II-III 943, III 980; inside the 4324-row all-folds cohort: acute 97, Stadium II 61, II-III 703,
   III 756).

   | stratum | n | R | null mean | null p95 | perm p | median err |
   |---|---|---|---|---|---|---|
   | acute (I, I–II) | 97 | 0.137 | 0.143 | 0.238 | 0.53 | 93.7° |
   | Stadium II | 61 | **0.482** | 0.165 | 0.281 | **0.0002** | 48.4° |
   | Stadium II–III | 703 | 0.295 | 0.272 | 0.304 | 0.12 | 54.5° |
   | Stadium III (old) | 756 | 0.196 | 0.192 | 0.225 | 0.43 | 79.5° |

   **The prediction fails**: acute sits at chance, and power-matched chronic (subsampled to n=97,
   400 draws) scores *higher* in **88%** of draws (R 0.207 vs 0.137). Per-territory means show all
   four territories mapping to essentially one angle in every stratum (acute: −66°, −59°, −72°,
   −78° against truths +57°, +147°, −147°, −57°). **The collapse is universal and
   acuity-independent.** The Stadium II cell (n=61) is flagged as *"a lead to replicate, not a
   result"* — its correspondence is scrambled, with Anteroseptal landing antipodal (−169°).

2. The `acuity_trend()` rank test (`CLAUDE.md`, 08-13): territory-centred alignment vs acuity rank
   gives ρ = **−0.034 (p=0.097)** under the strict scaler and ρ = **+0.017 (p=0.743, wrong sign)**
   under `target`, **non-monotone in both**; the apparent effect is entirely the n=61 Stadium-II
   stratum, at the wrong end of the axis. PTB-XL chronic dominance is now *measured*:
   1617/4324 graded, **90.2% Stadium II-III/III**.

⚠ **[corrected 08-13] The per-stratum b̂ analysis is RETRACTED.** It used the S(b) statistic, which
is well-defined only at integer b (see Card I). Every non-integer b̂ is an artifact of the
anchor-unwrap origin. The surviving statistic is `acuity_trend()` above, which does not use S(b).

⚠ **Required wording**: call the ST-vs-Q/R divergence *"consistent with"* the acute/chronic
account, **never "the mechanism for"** — that phrasing is on the forbidden list.

**Why the divergence still predicts the control's behaviour** (integrity audit §2.4, updated): the
54-feature control is fit only on MedalCare too, so it learns the same wrong channel and is harmed
the same way — which is exactly what the mechanism predicts, and is why the previously-unexplained
question "why does the control survive anyway" had no answer (it had a false premise).

**Stage 4.1 as originally specified is not runnable** (EXEC LOG Part 14): the rerun plan's
"Stadium III n=980, II-III n=943, II n=88 vs n=166 acute" are **whole-database** counts; the
exported latents were fold 10 only, holding **21 acute rows, 14 with a territory label**, vs 201
chronic. Measured anyway for the record (`exp7_baseline`): acute n=9 macro-F1 0.125 p=0.859;
transitional n=5 0.000 p=1.000; chronic n=165 0.203 p=0.599; unknown n=259 0.187 p=0.991; ALL n=438
0.189 p=0.994; Δ(acute − chronic) = −0.078 [−0.223, +0.031], p(acute>chronic)=0.907. **No stratum,
including the pooled n=438, beats its own within-stratum null.** The honest sentence is *"we could
not test it"*, with the count table as evidence — not *"stage is not the cause"*. (This constraint
was later relieved by the medalonly all-folds export, which is how §3.5 above became possible.)

**Method note for the thesis** (EXEC LOG §14.3): *the generalisable error is quoting a population
count to justify an experiment that will run on a split.* Standing check: compute stratum counts on
the rows the analysis will actually see.

---

### Card I — the 08-12 pivot and the Tier-1 geometry results

**The pivot** (`2026-08-12_pivot_representational_geometry.md`). Trigger: instruction to treat the
pipeline as possibly off-track, discard C2ST as the governing metric, research SOTA independently.

- **Why C2ST was the wrong instrument, stated precisely** (§2): *"C2ST estimates whether P ≠ Q. For
  interpretability we care whether there exists a structure-preserving correspondence… Translate one
  cloud by a large constant vector. C2ST → 1.0. Every internal relation is untouched."* The 77-cell
  sweep is the empirical half; the theoretical half is Gröger et al. (arXiv 2602.14486), which shows
  that after permutation calibration global spectral metrics (CKA/CCA/RV) disappear while local
  neighbourhood metrics survive, and that CKA's null scales as **O(d/n)** — our regime is
  d/n = 1024/2386 ≈ 0.43.
- **Novelty audit** (§1.5, re-confirmed 08-13): the ECG-SAE lane closed 28–29 Jul 2026 (CADENCE
  arXiv 2607.25244; ECG-InterpBench arXiv 2607.27404); the difference-of-means concept-direction +
  causal-injection method is scooped by Physics Steering (arXiv 2511.20798). **Genuinely
  unoccupied**: the sim→real cell; a mechanistic simulator as an external audit substrate;
  continuous biophysical parameters as probe targets; MedalCare-XL + any FM (zero papers); a
  characterised sim→real transfer *failure* with a measured mechanism. **Never claim "first sim2real
  ECG"** (Doste et al. *Front Physiol* 2022; Luongo et al. 2021 precede us).
- **Field context** (§1.6): our cross-domain null is level with the field's frontier — PAA-Net
  (arXiv 2605.22044, MICCAI 2026) reports in-silico Dice 0.7391 collapsing to **0.2284 zero-shot on
  17 real cases**; Li et al. *IEEE TMI* 2024;43(7):2466–2478 is **in-silico only** (Dice 0.457 ±
  0.317). Non-identifiability is an established field result and a **citation asset** (Grandits
  et al. arXiv 2411.00165; Álvarez-Barrientos et al. *Med Image Anal* 2025;101:103460). **Nobody in
  this field uses C2ST or MMD** — its use here was an import from domain adaptation.
  ⚠ Unverified and to be re-checked before the thesis: PAA-Net's 0.2284 figure, the Grandits
  identifiability cohort numbers, the Du et al. 2026 angiography licence.

**Tier-1 geometry results** (`2026-08-12_tier1_geometry_results.md`). Six encoders; **n=4324**
PTB-XL MI rows (3794 patients) and **n=6547** MedalCare; **group-disjoint CV** (`patient_id` on
PTB-XL; `f"{split}:{run_id}"` on MedalCare — run_id is unique only within a split);
permutation-calibrated; ridge α GCV-selected once per domain. Code `analysis/geom_common.py`,
`circular_geometry.py`, `latent_geometry_correspondence.py`, `granularity_control.py`,
`synthetic_prior_value.py`; artifacts `outputs/analysis/circular_geometry/*.json`.

**§2.1 in-domain decoding — SURVIVES** (the doc's own header: "§2.4 (subspace), §2.5 (kNN) and
§2.1's in-domain numbers survive unchanged"):

| encoder | d | MedalCare R | median err | PTB-XL R | median err |
|---|---|---|---|---|---|
| medalonly | 1024 | 0.657 | 29.1° | 0.798 | 16.0° |
| baseline | 1024 | 0.648 | 29.5° | 0.801 | 15.5° |
| ccmmd | 1024 | 0.653 | 29.3° | 0.801 | 15.7° |
| dual | 1024 | 0.654 | 29.2° | 0.800 | 15.7° |
| globalz | 1024 | 0.663 | 28.5° | 0.801 | 15.5° |
| K64 | 64 | 0.466 | 42.1° | 0.777 | 16.6° |

All p at the permutation floor (0.0020) in both domains, 6/6.

**Already retracted inside the 08-12 doc**: the pivot's *"territory is recovered **better** in real
ECG than synthetic"*. The two R values are not comparable — label-shuffle nulls are 0.043
(MedalCare continuous φ) vs 0.245 (PTB-XL 4-level). The `granularity_control.py` matched-target run
shows the confound is **class imbalance**, not granularity (PTB-XL puts 81.9% of its MI cohort in
two territories whose anchors both point rightward; MedalCare is 33.6/33.6/24.4/8.4). The two
principled normalisations **disagree in sign**. **Write: "territory is decodable to a comparable
degree in both domains, with real ECG at least as good."**
⚠ **[corrected 08-13]** the normalisation used the label-shuffle null (0.245) as chance; the correct
bar is the constant floor (**0.29216**) — the comparability conclusion survives, the normalised
numbers do not.

**§2.2 transport — every cell is below the constant floor [corrected 08-13]:**

| encoder | M→P source | M→P target | P→M source | P→M target | arc M→P src | arc M→P tgt |
|---|---|---|---|---|---|---|
| medalonly | 0.175 | 0.259 | 0.027 | 0.084 | 36.4° | 200.1° |
| baseline | 0.195 | 0.239 | 0.033 | 0.098 | 28.4° | 191.9° |
| ccmmd | 0.192 | 0.241 | 0.033 | 0.087 | 35.6° | 189.2° |
| dual | 0.187 | 0.261 | 0.030 | 0.087 | 38.5° | 208.7° |
| globalz | 0.211 | 0.206 | 0.046 | 0.035 | 70.0° | 243.5° |
| K64 | 0.248 | 0.244 | 0.110 | 0.086 | 10.8° | 120.1° |

> **[corrected 08-13] — the single most important edit.** The constant-predictor floor on the M→P
> target marginal is **R = 0.29216** (proven the supremum over all label-independent predictors);
> on P→M it is **0.09319**. Every M→P cell above (0.175–0.261) is therefore **below
> chance-as-a-constant**, not "modest transfer". The permutation nulls in this table sit *below* the
> floor, so **passing them does not mean beating a constant**. Rescored by `analysis/floor_audit.py`
> → `outputs/analysis/circular_geometry/floor_audit.json` (+ `floor_audit_report.txt`).
> Under floor-free metrics a real zero-shot signal survives **under the target scaler only**.

Note in the same section: `ccmmd` is indistinguishable from `baseline` on every cell — consistent
with the alignment dead-end.

**§2.3 anchor sensitivity — [corrected 08-13], quote the renormalised ranks only.** Ranking the 24
territory→anchor assignments by raw R compares different chance levels: each assignment defines its
own truth marginal and hence its own floor, and the floors span **0.148–0.593 (4.0×)**. Renormalised
by (R−floor)/(1−floor):

| encoder | raw rank [source] | **renorm [source]** | raw [target] | **renorm [target]** |
|---|---|---|---|---|
| medalonly | 19 | **11** | 3 | **7** |
| baseline | 19 | **10** | 3 | **7** |
| ccmmd | 18 | **11** | 3 | **7** |
| dual | 19 | **11** | 3 | **7** |
| globalz | 12 | **9** | 4 | **7** |
| K64 | 16 | **11** | 4 | **7** |

Identity headroom is **negative in every cell (−0.17 to −0.04)** and identity is never best, so the
qualitative conclusion — *the residual transport signal is largely not territory-specific* —
**stands**; but both dramatic versions die: identity is neither bottom-quartile (source) nor top-3
(target), it is **middling under both scalers**. Rerun script:
`reports/2026-08-13_audit_artifacts/scripts/anchor_renorm.py`; output `anchor_renorm_out.txt`.
The anchors themselves are MedalCare's own circular mean of φ per `territory_4c` bucket
(AS +57.27°, AL +147.25°, IL −147.26°, Inf −57.30°) — but **[corrected 08-13]** they are the
midpoints of the φ wedges that *define* `territory_4c` (to 0.043°), i.e. a construction of the
simulator's labelling, not an independent measurement; only the *semantic* correspondence is
assumed, and that is what the 24-assignment test probes.

**§2.4 subspace overlap — SURVIVES THE 08-13 AUDIT UNTOUCHED.** Statistic: mean cos² of principal
angles between the two 2-D readout planes; reported as cross-domain (200 group-bootstrap draws),
within-domain split-half (the estimator ceiling), and the random 2-D baseline (analytically 2/d);
headline = **normalised overlap = cross / sqrt(within_M × within_P)**.

| encoder | cross | random floor | ceiling M | ceiling P | **normalised** | 95% CI |
|---|---|---|---|---|---|---|
| medalonly | 0.0026 | 0.0019 | 0.360 | 0.311 | **0.0076** | [0.0016, 0.0168] |
| baseline | 0.0019 | 0.0019 | 0.346 | 0.293 | **0.0059** | [0.0007, 0.0144] |
| ccmmd | 0.0024 | 0.0019 | 0.350 | 0.302 | **0.0075** | [0.0018, 0.0159] |
| dual | 0.0026 | 0.0019 | 0.357 | 0.306 | **0.0080** | [0.0016, 0.0166] |
| globalz | 0.0032 | 0.0019 | 0.387 | 0.299 | **0.0093** | [0.0020, 0.0197] |
| K64 | 0.0170 | 0.0309 | 0.798 | 0.713 | **0.0225** | [0.0053, 0.0512] |

The estimator recovers 0.29–0.40 (1024-d) and 0.71–0.80 (64-d) of a domain's plane from an
independent half of that *same* domain; across the domain gap it recovers **0.6–2.3% of that**, and
the random-plane floor sits inside or above the cross-domain CI in every encoder — for K64 the
cross-domain overlap (0.0170) is **below** the random floor (0.0309). The within-domain split-half
doubles as a **real→real control**: PTB-XL patients transfer to *other* PTB-XL patients at
0.29–0.31 while MedalCare→PTB-XL sits at 0.002.
**The 08-13 report cites this explicitly** (C.4): the repair failure is *"evidence the residual gap
lives at representation level, consistent with the subspace-orthogonality result (2026-08-12 §2.4,
which survives this audit untouched)."*

**§2.5 k-NN — survives; not a linearity artifact.** Nonparametric transfer (k=25 nearest MedalCare
rows by cosine, predict circular mean of their φ; 500-draw permutation null; ceiling = same
procedure with neighbours drawn from PTB-XL itself, excluding self and same-patient rows):
R cross 0.220–0.294 vs null 0.132–0.160, **32–42% of ceiling**, significant in **5/6** (dual
p=0.054 misses). **Correct statement: the linear readout directions do not correspond at all; local
neighbourhood structure partially does** — ~40× more than the ~1% the linear subspace probe finds.

**§2.6 RSA — corroborative only, do not oversell.** ρ = +0.143 (four encoders), +0.029 (globalz),
+0.257 (K64); identity rank 7/24, 13/24, 10/24; each domain's own RDM highly reliable (+0.89 to
+0.99). But 4 territories give only 6 RDM pairs, so Spearman is quantised to 36 attainable values
and identical values across encoders are **not four confirmations**; rank 7 corresponds to p ≈ 0.29.

**§2.7 label efficiency — the synthetic prior has NEGATIVE value** (`exp8_leadfix_medalonly`,
PTB-XL, patient-disjoint held-out, 25 repeats; **[corrected 08-13]: this is a source-scaler result**
— `_pooled` shared basis + MedalCare moments, deliberate and verified, not a bug — state the scaler
when quoting):

| n labels | scratch | + synthetic prior | constrained to synthetic plane |
|---|---|---|---|
| 10 | 0.372 | 0.299 | 0.297 |
| 20 | 0.452 | 0.324 | 0.292 |
| 50 | 0.583 | 0.390 | 0.307 |
| 100 | 0.651 | 0.460 | 0.322 |
| 200 | 0.703 | 0.552 | 0.339 |
| 500 | 0.747 | 0.630 | 0.340 |
| 1000 | 0.765 | 0.695 | 0.339 |
| 2000 | 0.787 | 0.748 | 0.347 |
| zero-shot (frozen) | — | 0.176 | — |

Both findings replicate in 6/6 encoders: the prior is **worse than scratch at every budget**
including 10 labels (actively harmful, not neutral), and **constraining the readout to the synthetic
plane caps performance at ~0.29–0.36 regardless of n**, against 0.79 achievable free. K64 is the
informative exception in degree (its prior catches up by n≈500: 0.759 scratch vs 0.751 prior),
because a 2-D plane inside 64 dimensions is far less of a constraint than inside 1024. Zero-shot
frozen transport across encoders: 0.176–0.253.

**What the 08-13 audit retracted from this cluster:**

- **CLAIM 2 / S(b) / b̂ — RETRACTED.** `S(b) = |mean exp(i(pred − b·t))|` is invariant to which
  territory the anchors are unwrapped from **only at integer b**: spread across the four origins is
  ~1e-17 at b ∈ {−1,0,1,2} but **1.70e-01 at b=0.5** and **1.81e-02 at b=1.09**. b̂ across origins
  (medalonly) 0.718 → 1.116; K64 −0.390 / +1.172 / +0.558 / +0.546. The b̂ bootstrap is a two-point
  alias mass (1916 draws at +1.09, 84 at −0.42, **zero between**), so the reported CI spanned a
  region of zero bootstrap mass. The S(b) permutation null pins b̂ at 0 under any random labelling
  and fires at the p-floor in **all twelve cells including the "no correspondence" cell.**
  **Prior art**: Kempter et al. 2012, *J Neurosci Methods* 207:113–124, DOI
  10.1016/j.jneumeth.2012.03.007 — arg-max-R slope is "never unique" on a cylinder ("barber's pole"
  solutions). A banner is in `analysis/cyclic_order_test.py`.
  **What survives**: the branch-invariant **integer** points — **S(1) = 0.24–0.26 > S(0) = 0.14–0.21
  in all six encoders**, far outside the label-permutation null, i.e. territory-dependent angular
  structure exists. **Nothing about its gain or orientation survives.**
- **"Five of six encoders" is a forbidden phrasing** — the encoders are pseudo-replicates, ~2
  effective observations. Likewise: no b̂ numeric; no "unit gain"; no "correctly oriented";
  "22/24 cells at/below floor" → **18/24** under group bootstrap, and the tally is α-dependent.
- **§4 threat table row "scaler choice — resolved, conclusions identical under both" is FALSE.**
  Under floor-free metrics the two scalers give **opposite verdicts**: target survives the
  norm-matched random-projection and shuffled-source-refit nulls (**p=0.0033** each); source is
  indistinguishable from an arbitrary projection (η² p=0.34). Separately, F2's
  fidelity-predicts-transfer correlation exists under **source only** (ρ=−0.90 vs +0.10). This is
  supervisor **Q1**.
- **The supervised increment fails its null**: axis+latent[target] Δ=+0.0165 (paired-swap p=0.011)
  but the arbitrary norm-matched 1024→2 projection null gives **p=0.093** (63% of random projections
  add something). **Do not claim a supervised increment.**

**Still open in §4** (state as a limitation): the only real→real control is a **split-half of PTB-XL
itself** — same acquisition protocol, same hospital, same era. A second real cohort is the strongest
addition a reviewer will ask for, and was out of scope in the remaining time.

**Tier-2 recommendation, on record** (§6): **do not** run the SAE arm. Recommendation was
*"Nothing. Freeze the claim set and spend the remaining time writing."*

---

### Card J — scaler definitions (quote these verbatim)

**Phase-B2 / Track-3 pipeline** — `standardise_target(X_target, source_scaler, mode, pool,
pool_raw)` in `analysis/phase_b2_infarct_decoding.py:451-508`. CLI flag `--scaler-domain`;
**CLI default is `target`** (so a bare invocation reproduces the `outputs/phase_b2_exp8_tgtscaler/`
snapshot), while the **function-level default is `target_pool`** so new callers get the safe
convention. Docstring, quoted:

- **`source`** — *"reproduces the historical path: reuse the scaler fitted on MedalCare train.
  **That is a defect, not a baseline.** The two domains disagree about per-coordinate spread by up
  to ~3x, so a Ridge or LogReg fitted on MedalCare-standardised inputs is then handed PTB-XL
  features at the wrong scale and its decision boundaries sit at the wrong distances."*
  (Called "legacy/strict-transport" in the geometry docs; measured to cost the feature arm far more
  than the latent — global6 −34% vs latent −7%.)
- **`target`** — *"fits a scaler on the PTB-XL matrix itself. This is per-domain diagonal
  standardisation — transductive unsupervised domain adaptation in the AdaBN/CORAL lineage. It reads
  unlabelled target features only; no target label is touched anywhere in this function."*
  ⚠ It fits on the ~438-row **primary subset**, *"and that subset is chosen BY LABEL. Fitting scaler
  statistics on it therefore conditions the standardisation on the very labels being predicted,
  which is a subtle leak even though no label array is read."* This is the **legacy** mode in
  Track-3 scoring.
- **`target_pool`** — fits on the *"full PTB-XL matrix for the same split, all rows, no label
  selection: same transduction, but nothing label-dependent enters the scaler."* Raises if
  `pool is None` rather than falling back. A fully **disjoint** pool exists for the latent (PTB-XL
  train, ~17k rows) but not for the NeuroKit2 features, so both arms use the unselected same-split
  pool — *"This removes label-selection dependence; it does not remove row overlap."*
- **`target_pool_measured`** (= **"strict"** throughout the Track-3 grid) — *"`target_pool` with the
  imputed rows excluded from the **statistics** (not from the evaluation). It exists because
  `target_pool` is corrupted for the feature arm: the pool is every row of the split, but only the
  MI-subclass rows were ever run through the feature extractor, so ~75% of the pooled rows are
  entirely MedalCare-train medians… i.e. `target_pool` partially reconstructs the `source` defect it
  was introduced to avoid — and it does so for the feature arm only, since every latent row is real.
  That asymmetry biases the very Z-vs-features comparison the strict mode is meant to adjudicate.
  This mode fits column-wise nanmean/nanstd on the **un-imputed** pool instead, so it is exactly
  equivalent to `target_pool` for the latent arm and changes only the feature arm."*

**Tier-1 geometry pipeline** — `analysis/circular_geometry.py:134-139`, two transport scalers, both
reported everywhere: *"`source` scaler is the strict transport: no target-domain information at all.
`target` scaler re-centres on the target's own unlabelled statistics"* — a legitimate
deployment-time adaptation but **not information-free**. `analysis/synthetic_prior_value.py` uses a
`_pooled` shared basis + MedalCare moments (a source-scaler result; **not** a bug — verified 08-13).

**CORAL / diagonal standardisation** (breakthrough §12, EXEC LOG Part 3) — the alignment-side
sibling of `target`: rescale each coordinate per domain rather than deleting directions.
Diagonal CORAL gave M→P 0.6714 → **0.7604** (+0.089 single seed; **+0.0645 ± 0.020** over
3 checkpoints × 5 seeds), split-half control −0.0008, saturating at **50 unlabelled target samples**;
full CORAL at r=128 gave **−0.27** (the same failure as full-rank whitening — off-diagonal
covariance of a 1024-d space from a few thousand samples is mostly noise). Linear C2ST drops to
0.5000 while GBDT stays 1.0000, so *"the honest word is calibration"*, not alignment.
`quantile` (full marginal matching, `QuantileTransformer`) roughly doubles CORAL's gain: net
+0.1254 / +0.1671 / +0.0449 across three checkpoints.

---

## 3. What survives into the thesis — recommendation for §4.5 (~4 pages main text)

§4.5 is "the road to the audit": it must explain **why** the fidelity audit (§4.1–§4.3) and the
floor-aware circular evaluation (§4.4) are the right instruments, by showing what the previous
instruments did and did not establish. It is a *narrative of instrument failure*, not a results
dump. Suggested budget:

**Page 1 — the alignment dead-end, stated once and properly (≈1 page, MAIN TEXT).**
- The three-condition C2ST table (Card G, first table): 1.0000 in every condition while MMD falls
  18% and kNN mixing rises 4×. This is the cleanest single exhibit and it costs one small table.
- The 5-encoder confirmation row (C2ST 1.0000 on all five `exp8_leadfix_*`), one sentence.
- **Over-determination as the mechanism**: marginals alone 0.9999 mean, dependence alone 0.9993
  mean, floor 0.5032, **6/6 checkpoints at k=0**, with the K16 pair (linear 0.5808 vs GBDT 0.9968 on
  the dependence structure alone) as the closing sentence. One table, six rows.
- One paragraph on **the withdrawn frontier**: the `DEFAULT_MAX_ITER = 20` artifact, the linear-only
  measurement, the GBDT re-measurement, and the standing rule *"a linear probe can only license a
  claim about linear structure."* This paragraph is the methodological hinge of the whole chapter —
  it is what licenses §4.4's floor-aware framework and §4.1's audit-instead-of-alignment turn.
- **Omit from main text**: the label-shift bound, the subspace-identity/subspace-theta negatives,
  CORAL/quantile transfer gains, the whitening failures, `direction_agreement`, `domain_rank`.
  These are Appendix material (they are *ruled-out branches*, and the chapter cannot afford to
  narrate every one).

**Page 2 — the lead-order bug and the corruption-detector sweep (≈1 page, MAIN TEXT).**
- The bug in two sentences + **the physics identity table** (Card A2, raw-data RMS). This is the
  evidence that carries the fix; the 2×2 transfer table is illustration.
- The 2×2 involution diagnostic as a *method* (one small table or four numbers inline): 0.2132 →
  0.3278, and the **interaction/sign-reversal** argument, which is what distinguishes it from "the
  classifier got better".
- The 77-cell sweep as the honest sequel: linear C2ST spread **1e-5**, GBDT **9e-5**, ρ ≈ −0.04 /
  −0.03 across a 2× range in transfer; randoms 7/10 below identity's CI and 0/10 significant;
  **0/66 transpositions detected, 39/66 improve transfer, the historical bug cell ranks 70/77**;
  identity itself p=0.1139. End with the narrowed claim and the explicit self-correction: the
  transfer movement is *consistent with* the fix, not confirmation of it.
- This page is also where the **methods contribution** lands, and where the "we attacked our own
  favourable result" through-line is established — which buys credibility for §4.1–§4.3.
- **Appendix**: the full 77-row cell table, the class-breakdown table, MMD² details.

**Page 3 — Track 3: the pre-registered rule and its verdict (≈1 page, MAIN TEXT).**
- The rule quoted verbatim, with the date-ordering made explicit (fixed before the numbers existed).
- The four-cell verdict table (Card C) and the **frozen wording**, leading with the 4-class primary.
- The permutation-floor caveat (all four cells at 9.999e-05; "beats chance" uninformative at n=4324;
  only the paired test discriminates). This paragraph is mandatory — it is what stops an examiner
  reading p=1e-4 as strong evidence.
- One paragraph on the scaler dichotomy as an **open, declared** decision (supervisor Q1), with both
  cases stated in one sentence each. Do not adjudicate it in the text.
- **Appendix**: the scaler-mode docstring quotations (Card J), the `target_pool` corruption numbers.

**Page 4 — the latent-vs-control comparison, retraction included (≈1 page, MAIN TEXT).**
- The 12-cell grid: **put all twelve rows in the main text** (it is one compact table and its whole
  point is that selective quotation is what went wrong the first time). Follow with: two nominal
  hits, opposite directions, neither surviving Holm → **"statistically indistinguishable"**.
- The **in-domain** advantage as the surviving positive: **+0.1523 (p=0.0001, n=1200, medalonly)**,
  with **+0.1367** as its PTB-XL-supervised predecessor, and the independent replication at
  **+0.1660** on the circular-geometry readout (n=6513) — noting explicitly that these are
  *different pipelines*.
- The retraction paragraph: what was claimed on 08-11 (control +0.0656, 5/5, pre-registered), why it
  failed (encoder had been supervised on PTB-XL; one significant cell of four at n=438), and the
  encoder-vs-eval-set decomposition (+0.033/+0.040 encoder, inconsistent eval-set contribution).
  Include the imputation-coverage check (75.3% vs 72.4%) in a footnote.
- A short "what the control is" box: the 54 columns, lead-II fiducials, strict-superset bit-identity,
  and the instrument check (region separation 1.35 d, lead specificity 2.06×, in-domain Q/R AUROC
  0.909). This is needed or the comparison is not interpretable.
- **Appendix**: the 5/5 spatial54 tables under three scalers, the AUROC arm and its bootstrap
  (Z − spatial54 = −0.045, CI [−0.100, +0.006], P=0.952), the Part-12 imputation leak.

**Cross-cutting placement decisions.**

| item | verdict | why |
|---|---|---|
| ST-vs-Q/R probe map (η² table + transfer-efficiency table) | **Main text — but in §4.1/§4.2**, not §4.5 | It is the seed of the fidelity audit, not part of the road to it. §4.5 should reference it forward in one sentence. |
| Acuity test (ρ=−0.034 / +0.017; §3.5 stratum table) | **Main text, short** — wherever the mechanism is stated | It is the falsification of our own explanation and must travel with the claim. One paragraph + the four-row table (appendix). |
| K64 capacity dissociation (in-domain 0.523 survives, 0/8 cross-domain blocks) | **Main text, 3–4 sentences** in §4.5 | It kills "just use a narrower bottleneck" and pre-empts the capacity objection to the control comparison. |
| Circular-manifold in-domain result (16.0° / R 0.798 PTB-XL) | **§4.4**, not §4.5 | It is Part A of the 08-13 report. §4.5 only needs the sentence "measuring macro-F1 on a circular variable was itself an instrument error". |
| Tier-1 subspace/kNN/prior-value results | **§4.4** | Same. |
| MI-stage control (Stage 4.1) count table | **Appendix + one limitations sentence** | "We could not test it" is the honest claim; the count table is the evidence. |
| Stage-5 repo hygiene, run-id collisions, `args.json`, split systems | **Appendix / Methods footnote** | Real work, zero narrative value in Results. |
| CORAL / quantile transfer gains (+0.065 / +0.13) | **Appendix** | They are alignment-adjacent positives that complicate the negative story and are not needed for any claim in the ladder. |
| The three "process" self-corrections (rank-transform 0.5000, imputation leak, capacity confound) | **Methods chapter**, 1 paragraph each, or a single boxed "how errors were caught" | These are the integrity evidence the Declarations chapter benefits from; they do not belong in Results. |

**Sequencing note.** Write §4.5 *last*, after §4.1–§4.4 exist, and cut it to fit. Its only job is to
leave the reader believing three things: (1) alignment was properly, repeatedly, and finally ruled
out — with the mechanism measured, not assumed; (2) the standard instruments (C2ST, MMD, macro-F1
on a circular target, "beats its own shuffle null") were each shown to be inadequate *by measurement*;
(3) the surviving positive claim is in-domain-only, and the cross-domain endpoint is a declared,
pre-registered, honestly-scored null.

---

## 4. Numbers to be careful with

**A. Retracted or withdrawn — must never appear as live results.**

| number | status | replacement |
|---|---|---|
| Any **b̂** numeric; "unit gain"; "correctly oriented" | **RETRACTED 08-13** (S(b) branch artifact; Kempter 2012) | `S(1)=0.24–0.26 > S(0)=0.14–0.21`, 6/6, integer points only |
| Per-stratum acuity b̂ (+0.89/+0.88/−0.15/−0.51) | **VOID** | `acuity_trend()`: ρ=−0.034 (p=0.097) / +0.017 (p=0.743) |
| "Territory recovered **better** in real than synthetic" | **RETRACTED 08-12** (class-imbalance confound) | "comparable, real at least as good" |
| "Alignment is achievable and costs 21 AUC points" (frontier headline) | **WITHDRAWN 08-11 §11** | linear-only removal; GBDT 0.9999 at the same k; "there is no tradeoff" |
| "Per-coordinate quantile normalisation moved nonlinear C2ST off 1.0" (C2ST = 0.5000) | **WITHDRAWN 08-11 §13.2** (greedy induction never started; train C2ST also 0.5000; prediction range 0.00e+00) | jittered rank transform → 1.0000; floor 0.4980 |
| "The first crack in C2ST = 1.0" (dim_scan 0.947 / 0.902 at K=8) | **WITHDRAWN 08-11 §15.2b** | GBDT re-score: min 0.9948 over 30 cells |
| "A 54-feature control **beats** the latent cross-domain" (+0.0656, 5/5) | **RETRACTED 08-12 §6.4b** (conditional on PTB-XL-supervised encoder) | "statistically indistinguishable"; 12-cell grid |
| "The latent carries territory information hand-crafted features do not" (vs global6) | **WITHDRAWN 08-11 §15** (control structurally blind) | comparison re-instrumented with spatial54 |
| In-domain Pipeline-B calibrator macro-F1 **0.998** | **INVALID** (resubstitution, defect A2) | out-of-sample calibrator fit on MedalCare train |
| Every pre-fix `permutation_p_r2` (`phase_b2/in_domain.json`, `dim_scan/*_summary.json`) | **VACUOUS** (defect A3, null had no intercept) | recomputed with centred hat matrix |
| Every pre-fix 4c AUC from `eval_decoding_lowK` / `concept5` / tier2 | **CORRUPTED** (defect A1) | recomputed with `proba_labels` reordering |
| All `ecg_features` columns in `outputs/phase_b2_exp8_poolscaler/` | **VOID** (75% imputed pool) | `target_pool_measured` arm |
| `transfer_control.py` first two runs (spatial54 M→P 0.716, MI 0.9918) | **VOID** (missingness leak; indicator alone AUROC 0.847/0.863) | leak-free tuned run: 0.6123 |
| "Five of six encoders" as a replication claim | **FORBIDDEN 08-13** (pseudo-replicates, ~2 effective observations) | state the encoder set, not a count |
| "22/24 cells at/below floor" | **CORRECTED** → 18/24 under group bootstrap, α-dependent | — |
| `outputs/phase_b2_inlp/` | **LEGACY — do not cite** (folder-derived labels, 200 permutations, pre-leadfix encoders) | — |

**B. Pipeline-specific — never mix.** Four distinct scoring pipelines produce numbers that look
comparable and are not:

1. **Phase-B2 territory macro-F1** (`analysis/phase_b2_infarct_decoding.py`): in-domain n=1200
   MedalCare test, cross-domain n=438 (fold 10) or n=4324 (all folds). Latent 0.6560 / control
   0.5037 in-domain; 0.3440 / 0.3295 cross-domain. Permutation nulls at 10 000 draws.
2. **Circular-geometry macro-F1 / R / η²** (`analysis/circular_geometry.py` + 08-13 audit):
   group-disjoint CV, n=6513 MedalCare / n=4315–4324 PTB-XL. Latent 0.6195, control54 0.4535, axis
   0.2050 in-domain; 0.3402 / 0.3189 / 0.3043 cross-domain. **These are a different task and a
   different readout** — the +0.1660 here is an *independent replication* of the +0.1523 in (1),
   not the same number.
3. **Transfer AUROC** (`analysis/transfer_reality.py`, `transfer_control.py`, tier1's LR M→P):
   3-class shared label space, macro one-vs-rest AUROC, n=438 or the shared-class test splits.
   0.5674 / 0.6123 / 0.8270 etc. **Never compare an AUROC to a macro-F1.**
4. **Lead-permutation sweep macro-F1**: probe fit on source with a **source-fit scaler, no
   calibration**, n=438. Identity = 0.2599, p=0.1139. **Not** a restatement of Part 18's 0.2786.

**C. Scaler-specific.** Any cross-domain number is meaningless without its scaler. `target` and
strict (`target_pool_measured`) disagree on the Track-3 verdict (0.6299 vs 0.6521), on the sign of
the arm-vs-arm comparison, on whether the floor-free zero-shot positive exists at all (p=0.0033 vs
p=0.34), and on the direction of F2's fidelity-transfer correlation (ρ=−0.90 source vs +0.10
target). `source` is a *defect*, not a conservative baseline, and it damages the feature arm ~3×
more than the latent.

**D. Encoder-specific.** `exp8_leadfix_baseline`/`ccmmd`/`dual`/`globalz`/`K64` **all trained on
PTB-XL folds 1–8 with fold 9 as validation** — any cross-domain comparison against a control that
never trains is biased on those encoders, and only fold 10 (n=438) is clean. `exp8_leadfix_medalonly`
is the only encoder for which all ten folds are evaluable. `dual` is the PTB-XL-contaminated encoder
that carries the one Holm-surviving floor-free cell in the 08-13 audit — flag it when quoting.

**E. n-specific and power-specific.**
- At **n=4324** every `permutation_p_macro_f1` sits at the 1e-4 floor for both arms on both
  endpoints. Floor p-values are not evidence.
- At **n=438** the 95% CI on a macro-AUROC difference is ~0.11 wide; the lead-permutation identity
  cell is itself p=0.1139; "not detected" is partly "not powered".
- Permutation nulls in the circular pipeline sit **below** the constant floor
  (E|V|² = R_p²R_t² + (1/n)(1−R_p²)(1−R_t²) + O(n⁻²); Jensen term +3.9% for diffuse predictors), so
  passing a permutation test does **not** mean beating a constant predictor.
- Multiplicity: 605 permutation p-values existed pre-fix with no correction; 20/66 and 2/66 in the
  sweep are uncorrected; 2/12 nominal hits in the paired grid do not survive Holm.

**F. Ceiling-specific.** The 4-class task carries a **built-in oracle ceiling of macro-F1 0.8643
(acc 0.9158)** under the *folder* labelling, with Inferolateral recall pinned at 0.500 by
construction. Post-D1-fix (φ-derived) the oracle is 1.0000, but **all pre-fix 4c numbers were
implicitly compared against 1.0**. Post-fix class balance also changed (Anterolateral 850→1300,
Inferolateral 900→450), so pre/post 4c comparisons are not like-for-like.

**G. Wording traps carried from `CLAUDE.md`'s forbidden list.** "the mechanism for" the ST/Q-R
divergence (→ "consistent with"); "the latent beats the axis" unqualified; "MedalCare only checked
marginals"; "first sim2real ECG"; any permutation-floor p as evidence; "transfer confirmed the lead
fix".

---

## 5. Compute

| item | value | source |
|---|---|---|
| GPU | **NVIDIA GeForce RTX 5080, 16,303 MiB** | EXEC LOG "Environment" |
| Interpreter | `F:\anaconda3\envs\ECGFounder\python.exe` — **bare `python` on PATH resolves to base anaconda with torch 2.10.0+cpu (no CUDA)**; every campaign command used the absolute path | EXEC LOG "Environment" |
| torch / numpy / sklearn / scipy | 2.9.1+cu128 / 2.2.4 / 1.6.1 / 1.15.2 | EXEC LOG |
| Required env var | `KMP_DUPLICATE_LIB_OK=TRUE` (duplicate `libiomp5md.dll` otherwise aborts); `PYTHONIOENCODING=utf-8` added after a cp1252 em-dash crash | EXEC LOG Stage 3 |
| **Training (Stage 3, GPU)** | baseline 2864 s (30 epochs, best 30) · ccmmd 1830 s (19 logged, best 15) · dual 2301 s (19, best 15) · globalz 1965 s (18, best 14) · K64 1735 s (20, best 18) | `reports/stage3_logs/train_exp8_leadfix_*.log` |
| Stage 3 driver 1 | finished 04:21:39 after **2516 s**, 2/5 complete (3 cuDNN host-allocation failures from concurrent CPU analysis load — *scheduling contention, not a training defect*) | EXEC LOG Stage 3 |
| Stage 3 retry driver | launched 04:25:47; `stage3_status.json` finished 2026-08-11T06:12:21, elapsed **6394 s** | `reports/stage3_logs/stage3_status.json` |
| Latent export | 10–12 s per test/val cell, 41–85 s per train cell; 6 cells per run | `reports/stage3_logs/export_*.log` |
| **Analysis (CPU)** | phase_b2 4-config pass **91.2 min**; phase_b2 K64 **15.4 min**; poolscaler all-5 **76.2 min**; spatial54 decisive arm **100.8 min**; dim_scan **35.4 min**; tier1 pass 2 **13.8 min**; inlp_alignment **25.5 min**; inlp_lowK 3.3 min; leadperm sweep **16.2 min**; dim_scan_nonlinear_c2st ~4 min; c2st on 5 encoders **11 s** | EXEC LOG Parts 9–19 |
| Tier-1 geometry (08-12) | four scripts, read-only, **≈50 min total on CPU**, no GPU | tier1 geometry §7 |
| 08-13 audit scripts | CPU only, run from repo root, seeds inside each script | 08-13 report "Reproduction" |
| Ablation cost note | the spatial54/scaler reruns spend **~4 of every 5 minutes** recomputing the in-domain θ leg, which `--scaler-domain` cannot affect; a working `--n-perm-in-domain 0` would cut the ablation from ~6 h to <1 h. `--cross-domain-only` is **a dead flag** (declared in argparse, referenced nowhere) | integrity audit §4 "Note for future reruns"; EXEC LOG Part 5.1 |

---

## 6. Artifact inventory (as of 2026-08-17)

**`outputs/analysis/`** — directories: `circular_geometry`, `domain_signal`, `fidelity_audit`,
`label_shift`, `leadperm_sweep`, `leadswap_diag`, `probe_map`; loose files
`acuity_stratified_transport.json`, `scaler_domain_ab.json`,
`scaler_domain_ab_target_vs_target_pool.json`.

- `leadperm_sweep/` → **`leadperm_sweep.json`** (77 cells).
- `circular_geometry/` → `acuity_stratified_transport.json`, `circular_geometry.json`,
  `cyclic_order_test.json`, **`floor_audit.json`**, **`floor_audit_report.txt`**,
  `granularity_control.json`, `latent_geometry_correspondence.json`,
  `synthetic_prior_value.json`.
- `fidelity_audit/` → `f1_fidelity.json` + `f1_fidelity_out.txt`, `f2_blocks.json` +
  `f2_blocks_out.txt`, `f3_repair.json` + `f3_repair_out.txt`.
- `leadswap_diag/` → `c2st_leadfix.json`, `c2st_leadfix_trained.json`,
  `pipeline_a_leadswap.json`, `pipeline_a_medalcare_unswapped.json`,
  `pipeline_a_medalcare_unswapped_perlead.json`.
- `probe_map/` → `probe_map.json`, `probe_map_exp8_leadfix_{baseline,ccmmd,dual,globalz,K64}.csv`,
  `grid_<config>.png` (5), `probe_control.{csv,json}`.
- `domain_signal/` (36 files) — key ones: `coral_exp8_leadfix_baseline.json`,
  `coral_controls.json`, `quantile_controls.json`, `nonlinear_c2st_exp8_leadfix_baseline.json`,
  `nonlinear_structure_exp8_leadfix_baseline.json`, `domain_mechanism_replication.json`,
  `subspace_{identity,theta}_exp7.json`, `tradeoff_frontier_exp8_leadfix_baseline*.json/png`,
  `transfer_control_exp8_leadfix_baseline{,_tunedC}.json`,
  `transfer_control_bootstrap_exp8_leadfix_baseline.json`,
  `transfer_reality_exp8_leadfix_baseline.json`, `rank_reconciliation_*.json`,
  `b2_coral_{exp5_3class,exp7_baseline,exp7_ccmmd}.json`, `inlp_controls_exp7.json`.
- `label_shift/` → `label_shift.json`.

**`reports/2026-08-13_audit_artifacts/`** — `anchor_renorm_out.txt`, `tmp_f1_fidelity.json`,
`tmp_f1_fidelity_out.txt`, `tmp_f2_blocks.json`, `tmp_f2_blocks_out.txt`, `tmp_f3_repair.json`,
`tmp_f3_repair_out.txt`, `tmp_floorfree_condmi_out.txt`, `tmp_floorfree_out.txt`,
`tmp_s2_sota_out.txt`, `tmp_s3_validation_lit_out.txt`, `tmp_s4_venues_out.txt`, `tmp_t1_full.log`,
`tmp_t4_alpha_grid.txt`, `tmp_t4_alpha_results.json`, `tmp_t4_supplement.txt`,
`tmp_v_f1-fidelity-audit_out.txt`, `tmp_v_f2-block-transfer_out.txt`, `tmp_v_f2_verify.json`,
`tmp_v_f3-channel-repair_out.txt`; plus `scripts/` containing `anchor_renorm.py`,
`tmp_f1_fidelity.py`, `tmp_f1_fidelity_addendum.py`, `tmp_f2_blocks.py`, `tmp_f3_repair.py`,
`tmp_floorfree_condmi.py`, `tmp_floorfree_metrics.py`, `tmp_t1_combination.py`,
`tmp_t1_randproj_null.py`, `tmp_t35_control_mirror.py`, `tmp_t4_alpha_protocol.py`,
`tmp_t4_supplement.py`, `tmp_v_f1_fidelity_verify.py`, `tmp_v_f2_verify.py`, `tmp_v_f3_verify.py`,
`tmp_verify_audit.py`, `tmp_verify_floorfree.py`, `tmp_verify_window.py`.

**Phase-B2 result directories under `outputs/`** (which number lives where):
`phase_b2` (pre-exp8, D1-fixed re-run) · `phase_b2_inlp` (**LEGACY, do not cite**) ·
`phase_b2_exp8` (pass-2, `target`) · `phase_b2_exp8_srcscaler` · `phase_b2_exp8_tgtscaler` ·
`phase_b2_exp8_poolscaler` (**ecg_features columns VOID**) · `phase_b2_exp8_spatial54` (the
decisive 08-11 run) · `_spatial54_srcscaler` · `_spatial54_poolscaler` · `_spatial54_measscaler` ·
`phase_b2_smoke_paired` (rows 1–2 of the paired grid) ·
`phase_b2_baseline_fold10_measscaler_paired` (rows 7–8) · `phase_b2_medalonly_fold10_target` ·
`phase_b2_medalonly_fold10_measscaler` · `phase_b2_medalonly_allfolds_target` ·
`phase_b2_medalonly_allfolds_target_pool_measured` (rows 5–6, 11–12 — the Track-3 verdict) ·
`phase_b2_mi_stage` (Stage 4.1) · `dim_scan_exp8` · `tier1_eval_exp8`
(`cross_config_table.md`).
