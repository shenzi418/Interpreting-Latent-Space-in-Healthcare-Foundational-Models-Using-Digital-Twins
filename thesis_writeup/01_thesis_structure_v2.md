# Thesis structure v2 — content plan for every section

**Written overnight 17→18 Aug 2026 (supersedes `00_thesis_structure.md` §2–3; §0 template constraints, §4 Declarations plan, §5 Q1–Q5 and §7 discipline in v1 still apply).**
**Status**: DRAFT for morning sign-off. Chapter prose written so far: `06_declarations.tex` (facts you supplied) and a **v0 of Ch3 §3.2 Data** (`03_contribution.tex`, clearly marked; disposable if you change the plan) — everything else below is a content brief.

Sources for every claim in this plan: the four dossiers in `notes/dossiers/` (A data & setup, B history Nov→Jul, C history 10–13 Aug, E outputs inventory), the two 2026-08-13 reports, and the frozen artifacts. Each brief lists its **materials** (tables/figures with the artifact they come from) and its **cautions** (retracted / superseded / pipeline-specific numbers).

---

## 0. What changed since v1, and why

Your critique: v1 had no experimental setup, no data definitions, no exp1–8 history, and risked being an over-long "AI-shaped" document. v2 fixes this by:

1. **Ch3 "Contribution" is now a full Data & Methods chapter** — datasets with counts, label spaces, θ, splits, lead order; ECGFounder and every fine-tuning mode; the run families exp5–8 as a glossary table; the two feature sets; the three analysis pipelines; the audit/transport/repair protocols. (§3 below.)
2. **Ch4 "Experimental Results" is re-ordered as the year's question-logic** (recommended; see the decision in §6): adapt the FM → try to align → try to decode → *why does it fail?* → the audit → the audit predicts → repair fails. The audit remains the centrepiece by page share and by what the Introduction/Conclusion lead with. Option A (audit-first, as in the decision doc) is kept as an alternative — the section briefs are identical either way, only the order changes.
3. **Realistic sizing**: ~55 pages of main text (Ch1–5), tables and three figures carrying the numbers, prose kept to what a one-year project honestly supports. Everything pre-`exp8` goes to an appendix table with its "superseded" flag rather than into main-text prose.
4. **A dated experiment glossary** (Table 3.4) so an examiner can see what was done when, including what was *not* done (Exp 2/3 never run; multi-seed never run; June–July = poster period with no repo activity).
5. **New integrity items surfaced by the dossiers** are listed in §6 for your decision (most important: the Track 3 PARTIAL bar is 0.55 in the dated May pre-registration, 0.65 only in undated notes).

---

## 1. Headline, register, page budget

- Headline claim, register (negative-as-measurement; instrument + mechanism), and claim ladder C1–C6: unchanged from the decision doc / v1 §1.
- **Page budget (main text ≈ 55 pp)**: Ch1 4 · Ch2 8 · Ch3 14 · Ch4 24 · Ch5 5 · (Ch6 Declarations 3) · Appendices ≈ 10 · References.
- **Figures**: Fig 1 audit map (built), Fig 2 block transfer (built), Fig 3 repair forest (built), + Fig 3.1 pipeline overview schematic (to draw), + Fig 4.1 C2ST-vs-transfer from the 77-cell sweep (from `outputs/analysis/leadperm_sweep/`; optional). Nothing else.
- **Tables** (main text): ~14, listed per section below. Big grids → appendix.

---

## 2. Chapters 1, 2, 5 (unchanged in shape; briefs sharpened)

### Ch 1 Introduction — 4 pp (write last)
1.1 Cardiac digital twins as a source of labelled ECGs; ECG foundation models; the natural hope: fit a mechanistic readout on the twin, read it out on patients. State the standard belief fairly ("a synthetic cohort validated for distributional realism is usable").
1.2 The project's question in its two forms: (i) *as registered* — can the latent space of an ECG FM adapted to a digital twin be interpreted in terms of the twin's parameters θ, and does that interpretation carry to real ECG? (ii) *as answered* — the transfer fails, and the failure is explained and predicted by a per-feature informativeness audit of the twin against the real cohort. Hook (Q4c): MedalCare-XL's own 39 % vs 62 % Turing result, quoted fairly.
1.3 Contributions = C1–C6 + the methodology bundle, each with its section number.
1.4 Outline.

### Ch 2 Background — 8 pp (write at D7; citations from the verified list, papersflow for every new one)
2.1 The 12-lead ECG and infarct localisation (acute ST vs chronic Q; frontal QRS axis) — Thygesen 2018, Das 2006. 1 p.
2.2 Cardiac digital twins and MedalCare-XL: what is simulated (θ = φ, z, size, ρ; 8 pathology classes; 16,839 records), how it was validated (marginal overlays vs PTB-XL+, Fig 6 hand-picked class-level features, Turing test 77.33 / 83 / 62-vs-39) — Gillette 2023 (verified). 1.5 p. Forbidden #25 governs wording.
2.3 PTB-XL: 21,799 records / 18,869 patients / 10 folds / SCP codes / MI sub-labels / `infarction_stadium` — Wagner 2020, Strodthoff 2021 (verified). 0.5 p.
2.4 ECG foundation models; ECGFounder (Net1D, 10 M recordings) — Li 2024 arXiv (verified; NEJM AI 2025 version noted); the FM-interpretability lane is real-data only (CADENCE, ECG-InterpBench — verify). 1 p.
2.5 Inverse inference of infarct parameters from ECG (in-silico state of the art: Li TMI 2024, Li RBME 2025); sim-trained/real-tested exists for other localisation tasks (Doste 2022, Luongo 2021) → never "first sim2real ECG". 1 p.
2.6 Domain gap and alignment: MMD (Gretton — verify), class-conditional MMD, INLP (Ravfogel — verify), C2ST (Lopez-Paz & Oquab — verify), CORAL (Sun — verify). 1 p. *New vs v1: needed because §4.2 uses these.*
2.7 Evaluating synthetic data beyond marginals: TSTR → utility frameworks → feature-importance agreement → DT credibility (V&V-40, Pathmanathan, Viceconti) → the gap the audit fills. 1 p.
2.8 Circular statistics for angular readouts (Fisher & Lee 1983; Jammalamadaka & Sarma **1988** — resolved; Kempter 2012; van Driel 2015; Kovach 2017; Combrisson & Jerbi 2015). 0.5 p.
2.9 Negative results as measurements: the exemplar skeleton (Zech, DeGrave, Oakden-Rayner, Christodoulou, Ghassemi). 0.5 p.

### Ch 5 Conclusion — 5 pp
5.1 Findings against the ladder (one paragraph per C1–C6, status words exactly as the ladder). 5.2 Marginal realism vs informativeness fidelity: what it implies for DT validation practice; the audit as a cheap pre-training gate. 5.3 Limitations (one simulator / one cohort / one backbone family with ~2 effective encoder observations / one delineator; η² linearity; label-granularity mismatch and its out-of-sample answer; scaler dependence; n=5 blocks; best-of-8 parity selection; MNAR globals; PTB-XL angles are *assigned* anchors, not measured; anatomy models `run_S64`/`run_S67` straddle splits — see §6; no multi-seed replication before `exp8`). 5.4 Future work framed as future (second simulator, second backbone, rank-based informativeness, delineator sensitivity, acute cohort). 5.5 Closing statement.

---

## 3. Ch 3 "Contribution" = Data, models, methods and the instrument — 14 pp

**Overall stance**: written as a methods chapter with the *contribution* being (a) the experimental programme and (b) the audit instrument + circular-evaluation methodology. Every number here is a *setup* number (counts, dimensions, hyper-parameters), traced to `dossiers/A_data_and_setup.md` (pointer given per item as A§x).

### 3.1 Overview of the study (1 p)
- One paragraph on the design: two domains → one frozen FM with adapters → latents → three questions (align? decode? why not?) → the audit.
- **Fig 3.1** pipeline schematic (draw: data → encoder → latents/features → four analysis blocks).
- Novelty fences A–F in one paragraph with the "to our knowledge + search scope" caveat.

### 3.2 Data (3 pp)
**3.2.1 MedalCare-XL** (A§1.1): 16,839 WFDB records, 12 leads, 500 Hz, 10 s; 8 mutually exclusive pathology labels in fixed column order `(sinus, mi, rbbb, lbbb, lae, iab, fam, avblock)` (glossary follows the dataset README, not the mislabelled inline comments — A§7.8); splits train 12,019 / val 2,434 / test 2,386, **taken from the dataset's own split directories** (`add_medalcare_splits.py`; agreement 100 %) — *not* the SHA-256/StratifiedGroupKFold description in the rules files, which never ran (A§7.1); MI subset 7,797 rows (5,347 / 1,250 / 1,200) across 8 territory folders; θ = `isch[0].{φ, z, size, ρ_eps_max}` (4-dim; `transmural` is a byte-identical duplicate); `territory_4c` from φ wedges (±2.0 rad, 0.0 rad); anchor angles AS +57.27°, AL +147.25°, IL −147.26°, INF −57.30°; the `LCX_0.3_post` folder/φ disagreement and the resulting oracle ceiling (0.867 macro-F1 for any 4-class readout — Card A / dossier B §4.4). **Table 3.1** MedalCare cohort table.
**3.2.2 PTB-XL** (A§1.2): v1.0.3, 21,799 records / 18,869 patients, 500 Hz, 10 s, official 10-fold stratification; the 5 superclasses; the shared 3-class space {NORM, MI, CD} and how it is built (`MEDALCARE_REMAP` / `PTBXL_REMAP`; the pre-fix filter kept STTC instead of CD — defect F); the MI-territory 4-class label from SCP sub-codes (rules + exclusions; MI-present 5,469 all-folds → **4,324** clean 4-class rows; fold 10 alone 438; the older 3-class single-territory rule gives 444 — name the rule whenever a count is quoted); acuity `infarction_stadium` (1,617/4,324 graded; 90.2 % Stadium II–III/III). **Table 3.2** PTB-XL cohort table.
**3.2.3 Preprocessing** (A§1.3, §1.1.9, §1.2.2): reindex by WFDB `sig_name` on both domains; the historical positional aVL↔aVF transposition in the MedalCare loader (found 10 Aug), verified by the limb-lead identities aVL = (I−III)/2, aVF = (II+III)/2 on the raw signals (Card A2 table); per-lead z-score on both domains (MedalCare previously global scalar); tag `pre-leadfix` preserves the pre-fix state. Two short paragraphs; the *consequences* go to §4.2.
**3.2.4 Territory as an angle**: PTB-XL rows carry no φ — their target angle is the MedalCare anchor of their territory (A§7.11); this is why circular metrics need a constant floor (§3.6).

### 3.3 The foundation model and its adaptation (2.5 pp) (A§2)
- ECGFounder Net1D: `filter_list=[64,160,160,400,400,1024,1024]`, 1024-d pooled features, pretrained `12_lead_ECGFounder.pth`; `ConvAdapter1D` residual adapters (309,568 trainable params) + linear head (3,075 params for 3 classes); backbone frozen; the `linear_prob+use_adapter` freezing bug and the manual-freeze workaround (one sentence, footnote).
- Fine-tuning modes: single-domain; joint dual-head (per-domain heads, native 8/5 labels; optional 51-name physics head — a *different* θ object from the 4-member `isch[0]` θ: use θ_phys vs θ); joint shared-head (one 3-class head, alternating batches, `pos_weight`, Adam head 1e-3 / adapters 1e-5, ReduceLROnPlateau, checkpoint on mean per-domain macro-F1); MMD / class-conditional MMD loss (λ=0.1); bottleneck heads `Linear(1024,K)→GELU→Linear(K,3)`, K∈{16,64,256}; Tier-2 multitask bottleneck (cls + bio targets).
- **Table 3.3 run families** (from A§2.3.1): rows = `joint_*` (Mar), `exp5/6_3class` (May, dual-head ± ccMMD), `exp7_baseline/ccmmd` (Apr, shared-head), `exp7_bottleneck_K16/64/256`, `exp7_tier2_*`, `exp8_leadfix_{baseline, ccmmd, dual, globalz, K64, medalonly}` (Aug, post-fix); columns = mode, label space, epochs, K, λ, trained on, lead-fix status, headline in-domain macro-F1 (MedalCare / PTB-XL). Mark pre-`exp8` rows "provisional (pre-fix)"; mark `args.json` missing where so (A§7.3). Note that PTB-XL columns pre/post fix are **not comparable** (different filtered subset) — A§7.7.
- Which encoder is the headline and why: `exp8_leadfix_medalonly` (never saw a PTB-XL gradient; checkpoint chosen on MedalCare val alone) → all 10 PTB-XL folds usable (n=4324); `exp8_leadfix_dual` as sensitivity; **~2 effective encoder observations** caveat.
- Latent export: `outputs/latents/<run>_<domain>/latents.npz` (`Z` 1024-d, `P`, `Y`; unfiltered split row order; 127 exports).

### 3.4 The experimental programme, dated (1 p) — **Table 3.4 experiment glossary**
| ID / name | when | what | status in thesis |
|---|---|---|---|
| Weekly baselines (zero-shot, frozen linear/MLP, PTB-XL baseline, physics head) | Nov 25–Feb 26 | establish the frozen-encoder regime | one sentence + Appendix |
| Exp 1 / 4 (Exp 2, 3 never run) | Feb–Mar | joint dual-head ± adapters ± MMD | Appendix table (pre-fix) |
| Exp 5 / 6 | May 3 | dual-head on the shared 3-class space, ± ccMMD | 2×2 design in main text; numbers Appendix |
| Exp 7 | Apr 16 | shared head; +ccMMD; the 2×2 conclusion | design main; numbers Appendix (pre-fix) |
| Track 1 (PCA dim scan, bottleneck K, Tier 1/2) | May 17–26 | dimensionality vs alignment/transfer/mechanism | one paragraph + Appendix |
| Track 2 (INLP v1/v2, low-K) | May 8–24 | post-hoc linear alignment | §4.2 main (qualitative), Appendix |
| Phase B / B2 (θ decoding, 8-class audit) | May | in-domain θ decodability; cross-domain territory | §4.3 (in-domain re-quoted from exp8), Appendix |
| Track 3 (pre-registered territory decoding) | May 13–18; re-scored Aug 12 | the pre-registered rule and its verdict | §4.3 main |
| Poster | Jun–Jul | (no repo artefact) | one sentence in timeline |
| Repo audit + lead-fix + `exp8` reruns | Aug 10–11 | four/five defects; retrained encoders | §3.2.3, §4.2 |
| Fair control (spatial54), 12-cell grid, 77-cell sweep, probe map | Aug 11–12 | control comparison, C2ST blindness, ST vs Q/R | §4.2, §4.3 |
| Circular geometry Tier 1 + 13 Aug audit | Aug 12–13 | floors, S(b) retraction, floor-free metrics | §4.3 |
| Fidelity audit F1/F2/F3 | Aug 13 | the centrepiece | §4.4–4.6 |
Also state: multi-seed replication was requested and never run (no pre-`exp8` error bars); the June–July gap.

### 3.5 Hand-crafted ECG features (1.5 pp) (A§4)
- Two sets, both delineated once on lead II with NeuroKit2 `dwt` on raw voltages: `global6` (QRS_dur, QT, P_dur, ST_J60_avg, T_amp, HR-type globals — list from A§4.1) and `spatial54` = 48 per-lead (Q_amp, R_amp, ST_J60, T_amp × 12 leads) + the 6 globals; `T_amplitude_mV` ≡ `T_amp_II` (53 distinct); `ST_J60_avg_mV` is not the mean of the 12 leads; coverage/missingness per block and domain, incl. the MNAR flag on sim interval globals; frontal QRS axis = atan2(R_aVF, R_I). **Table 3.5** blocks × coverage.
- What the 54-feature *control arm* is for (the fair competitor to the 1024-d latent) and its instrument checks (Card E: region separation, lead specificity, in-domain Q/R AUROC 0.909).

### 3.6 Analysis pipelines (2.5 pp) (A§5) — **Table 3.6 pipeline glossary** (forbidden #23: never mix)
- **P1 Phase-B2 classifier pipeline** (`analysis/phase_b2_infarct_decoding.py`): multinomial logistic regression on Z or on features, C by 5-fold CV on the source; the four scaler modes quoted verbatim (`target`, strict/`source`, `target_pool`, `measured` — Card J / A§5.1); label-permutation p (floor 1e-4 at n=4324 — "certifies label use only"); `paired_macro_f1` (same resampled rows, per-row prediction swap) as the *only* arm-vs-arm test.
- **P2 circular-geometry pipeline** (`geom_common.py`, `circular_geometry.py`, `floor_audit.py`): ridge readout Z→(cos φ, sin φ), GCV α; nearest-anchor macro-F1; circular R̄ and its **constant-predictor floor** (0.29216 PTB-XL 4-anchor / 0.09319 MedalCare continuous φ / 0.12270 quantised) as the supremum over label-independent predictors; permutation nulls sit *below* the floor (Jensen identity); floor-free metrics (nearest-anchor macro-F1 with floor 0.1534/0.1257, circular η², Miller–Madow MI); patient/run-block bootstrap; the two mandatory nulls (norm-matched random projection; shuffled-source-label refit); the zero-parameter axis baseline. Box: floor derivation (half a page).
- **P3 fidelity-audit trio** (`fidelity_audit.py`, `block_transfer.py`, `channel_repair.py`) — definitions in §3.7–3.8.
- Alignment diagnostics: linear and GBDT C2ST (held-out), MMD (single/multi-bandwidth), kNN mixing, INLP protocol (rank-1 nullspace, `max_iter`, `stop_acc`; v1 vs v2 pools), the lead-permutation sweep protocol (77 cells, probe fit once on MedalCare-train, never refit).
- The self-corrections that shaped the pipelines, one sentence each (rank-transform, imputation leak, capacity confound, `max_iter=20`, linear-vs-GBDT) — the integrity evidence the Declarations refer to.

### 3.7 The informativeness-fidelity audit (1 p)
Per-feature ANOVA η² vs 4-class territory in each domain; 500-draw block bootstrap (runs / patients); blind-spot / spurious criteria (diff CI excludes 0); block-aware permutation for rank agreement (why per-feature ρ is not the licensed statistic); |SMD| for marginal realism; the 2-class collapse as granularity check.

### 3.8 Transport, prediction and repair protocols (1.5 pp)
Readouts fit on MedalCare only; the scalers defined precisely **per pipeline, because the word "strict" means different things in P1 and P2** (dossier A §5.1, dossier C Card J): in P1 (Track 3 / 12-cell grid) `target` = "legacy" (fresh scaler on the label-selected PTB-XL subset — a subtle leak) and `target_pool_measured` = "strict" (scaler on the *un-imputed* full PTB-XL split — still target-side, not information-free; `source` is a documented defect and is not reported); in P2 (circular geometry, F2/F3) `source` = strict transport (no target information) and `target` = re-centring on the PTB-XL MI cohort (diagonal-CORAL lineage, mildly transductive). **Table 3.6 carries this mapping; every results table names the mode by its code name.** — **Q1 declaration slot** (`\qone{}`); transfer efficiency = (F1_cross − floor_cross)/(F1_in − floor_in); block readouts (F2); channel restriction via a latent→feature map fit on MedalCare only (F3-A); importance reweighting with ESS-matched control (F3-B); diagonal CORAL as the recalibration boundary; paired tests on identical resampled rows for every arm-vs-arm claim; pre-statement practice (F2's prediction written to disk before computing).

---

## 4. Ch 4 "Experimental Results" — 24 pp, ordered as the question-logic (Option B)

Each brief: purpose → content → materials → cautions. Numbers below are the ones the section will carry (all traced in the dossiers / 08-13 report; the draft will attach `\tracenote{}` per number).

### 4.1 Adapting ECGFounder to the digital twin (2 pp)
*Purpose*: establish that the encoders work in-domain and which one is the headline.
- In-domain fine-tuning: `exp8` family test macro-F1 — MedalCare 0.91–0.95 across runs; PTB-XL 0.82–0.83 for the jointly trained runs; `medalonly` 0.9268 MedalCare / **0.4567 PTB-XL as a passive readout** (never trained on it) — A§2.3.2. The pre-fix `exp5/6/7` numbers exist only in Appendix Table A.2 with the "different PTB-XL subset" caveat.
- The 2×2 design (dual vs shared head × ± ccMMD) as the May finding that shared-head drives cross-domain class transfer (LR M→P AUC ~0.59 → ~0.76 pre-fix); the audit lists whether that survives the fix as **UNKNOWN**; state so. Bottleneck K: classification free at K=16 (pre-fix; appendix).
- **Materials**: Table 4.1 (exp8 family metrics; from `outputs/*/metrics.json`); Appendix A.2 (pre-fix table).
- **Cautions**: no pre/post before-after; ccMMD "changes nothing downstream" one sentence.

### 4.2 The two latent distributions cannot be aligned — and the standard detector is blind (3.5 pp)
*Purpose*: the first negative, stated once with the mechanism measured; and the instrument critique that licenses everything after.
- The four attacks (training-time MMD/ccMMD; post-hoc INLP at K=1024 and low K; PCA to K=8; trained bottleneck K=16) all leave held-out C2ST ≈ 1.0 (Card G table: 1.0000 under swapped / leadfix / leadfix+per-lead-z while MMD −18 % and kNN mixing 4×; all five `exp8` encoders 1.0000). Over-determination: marginals alone 0.9999, dependence alone 0.9993 (6/6 checkpoints down to K=16); the withdrawn INLP `max_iter=20` frontier and the linear-vs-GBDT lesson ("a linear probe licenses only linear claims").
- The lead-order bug as an *instrument* episode: physics-identity table (Card A2), 2×2 involution diagnostic (0.2132 → 0.3278, sign-reversal interaction), then the 77-cell sweep: linear C2ST spread 1e-5, GBDT 9e-5, ρ ≈ −0.04 with damage across a 2× range in transfer; randoms 7/10 below identity's CI; 0/66 transpositions detected; 39/66 *improve* transfer; the historical bug cell ranks 70/77; identity p = 0.1139 → transfer is a **corruption detector, not a lead-order detector**; the fix is carried by physics, not by p (forbidden #9).
- **Materials**: Table 4.2 (three-condition C2ST/MMD/kNN); Table 4.3 (over-determination, 6 rows); Table 4.4 (sweep summary); optional Fig 4.1 (transfer vs C2ST scatter, 77 cells). Appendix: full 77-cell table; INLP protocol v1/v2; label-shift bound; subspace/CORAL/quantile branches.
- **Cautions**: never "INLP converged"; never the K-ordering claim; every C2ST is held-out.

### 4.3 Decoding simulator territory from the latent: in-domain success, cross-domain null (6 pp)
*Purpose*: the second negative — pre-registered, honestly scored, with the retracted control-win included and the floor-aware re-reading of the circular readout.
- 4.3.1 In-domain: the latent decodes θ-territory far beyond the 54-feature control (**+0.1523 macro-F1, p=0.0001, n=1200, P1 pipeline**; predecessor +0.1367; independent replication +0.1660 on P2, n=6513); the axis is *below the constant floor in-domain* (η² 0.0020) — the seed of the audit. K64 capacity dissociation (in-domain survives, cross-domain 0/8 blocks). One paragraph on the 8-class anatomy×transmurality audit *as superseded May evidence* or omit (dossier B: appendix).
- 4.3.2 Track 3 [**ruled 2026-08-18, N1 accepted**]: the pre-registered rule quoted verbatim from the dated May log with its provenance (**POSITIVE 4-class ≥ 0.45, p<0.01; PARTIAL 2-class ≥ 0.55, p<0.01**; May scoring NEGATIVE at 0.235 / 0.461 on pre-fix encoders); re-scored on `exp8_leadfix_medalonly`, n=4324: 4-class **0.3440 (`target`) / 0.3357 (`target_pool_measured`) — POSITIVE not met under either mode**; 2-class **0.6299 / 0.6521 — PARTIAL met under BOTH modes at the dated 0.55 bar** (footnote: the 0.65 in later working notes would be met under `target_pool_measured` only). The verdict no longer hinges on the scaler; the mode question survives only for the paired latent-vs-control comparison (4.3.3) and the circular pipeline (4.3.4, Q1b). Permutation p at the 1e-4 floor for both arms on both endpoints → uninformative; only paired tests order arms. **Table 4.5** verdict table (both bars shown, both scalers).
- 4.3.3 Latent vs 54-feature control across the gap: **all twelve paired cells** (encoder × eval-set × scaler × endpoint; Card D) → "statistically indistinguishable" on the 4-class primary (+0.0145 p=0.162 / +0.0041 p=0.695 at n=4324); one 2-class strict hit (+0.0195, p=0.044) matched by one opposite hit; neither survives Holm. The retraction paragraph (control "won" 5/5 only while the encoder was PTB-XL-supervised; encoder vs eval-set decomposition; imputation coverage check). **Table 4.6** the 12-cell grid (main text — Q3c; dossier C recommends main).
- 4.3.4 The circular readout under floor-aware evaluation (Part A): every transported R̄ below the 0.29216 floor at label-free α (Table 4.7 α sweep, compact); the reductio (constant predictor "beats" the empirical floor at α=1e7); the axis wins R̄ by being a well-placed constant (arc 41.9°, wrong cyclic order); floor-free metrics × scaler (**zero-shot frame positive under target only**: macro-F1 0.3402 vs random-projection and shuffled-refit nulls p=0.0033; strict: η² p=0.34; the supervised increment over the axis fails its random-projection null p=0.093 — forbidden #4); conditional MI non-redundancy; the in-domain mirror (latent 0.6195 ≫ control54 0.4535 ≫ axis 0.2050); S(b): only the integer points S(1)>S(0) survive; no b̂, no "unit gain"; Kempter 2012. **Table 4.8** three-arm mirror; **Table 4.9** floor-free × scaler. Acuity test (ρ = −0.034 strict / +0.017 target, non-monotone) travels with the ST-vs-Q/R mechanism statement in 4.4.
- **Cautions**: label the pipeline on every macro-F1 (P1 vs P2 vs F3 ridge+anchor); Track 3 threshold footnote; "statistically indistinguishable"; anchors are constructions of the simulator; oracle ceiling 0.867 for any 4-class number.

### 4.4 The simulator relocates pathology information: the fidelity audit (6 pp) — **Fig 1**
Content exactly as v1 §2 (Ch4 §4.1 brief) — C1, C2, C6: the map; blind spots (28 + axis; Q_amp_III 0.006→0.324; axis 0.0020 vs 0.1297 at matched dispersion, real-side direction physiological); spurious channels (14 columns / 13 distinct; 9/12 ST_J60 leads; ST not uniformly spurious — V4/avg real>sim); block-level inversion (block-mean ρ −0.587; per-feature −0.354 NOT significant, block-aware p 0.10/0.14; T within-block +0.706 POSITIVE); marginal-realism *dissociation* (never anti-correlated); MNAR globals; 4c/2c robustness with the R_amp_V6 exception; MedalCare's own 39 % vs 62 % as clinician-level corroboration; the strongest objection (label-space mismatch) and its three-part answer. **Tables 4.10** block-mean η² + within-block ρ; **4.11** top blind spots / spurious with CIs; full 54-row table → Appendix A.1 (auto-generated from `f1_fidelity.json`).

### 4.5 Fidelity predicts transfer (3.5 pp) — **Fig 2**
As v1 brief (C3 + axis frame-vs-fit + T-block): per-block in→cross under both scalers; ST at the constant floor under source (0.1551 vs 0.1534, CI contains floor; eff 0.006) vs Q best (0.612); paired block orderings (ST−Q −0.083, p=0.001) as the robust core; ρ(η²_sim, eff_src) = −0.900, exact one-sided p=0.0417 (two-sided 0.0833 misses — say so), fold-seed sensitivity (seed 7: −0.700); η²_real ↔ eff ρ=+1.00; CORAL boundary (target ρ≈+0.10; ST 0.006→0.296, still second-worst); axis fit-free 0.3043 vs sim-fitted 0.178; globals eff 0.936 not headlined. **Table 4.12** F2 block table.

### 4.6 Restriction and reweighting cannot repair it (3 pp) — **Fig 3**
As v1 brief (C4, C5): Q+R restriction loses (−0.0552, p<0.001); ST12 at/below the random-projection bulk; Q+R vs ST ordering holds (+0.0855); inferior-lead 12-d **parity** (+0.0025, p=0.79; both nulls; beats measured inferior features +0.0441) — best-of-8 caveat, main text (Q3a); reweighting harms (axis-pair −0.0732, survives ESS control; six-feature only with the ESS caveat); dual-encoder mirror; the paired p attaches to F1 levels not ratio ordering. **Table 4.13** F3 arm grid + nulls. Closing paragraph: what the three results compose to (C.4 of the report) and the subspace-orthogonality result as the representation-level residual.

*(Option A ordering, if preferred: 4.4→4.5→4.6 first, then 4.1→4.3 as "the road to the audit" — same briefs, ~1 page less because the history compresses when it comes last.)*

---

## 5. Appendices (≈10 pp)
A.1 full 54-feature audit table · A.2 pre-fix run table (exp1–7, Track 1/Tier 1–2 headline numbers, all flagged) · A.3 α-grid / floor tables + renormalised anchor ranks · A.4 77-cell sweep table + INLP protocol details · A.5 Track 3 label design (SCP rules, exclusions, oracle ceiling) + scaler-mode docstrings · A.6 reproducibility (seeds, commands, artifact map, script provenance).

---

## 6. Decisions for the morning (new items first)

| # | Decision | My recommendation | Why it matters |
|---|---|---|---|
| N1 | Track 3 PARTIAL bar 0.55 vs 0.65 | **RULED 18 Aug: 0.55 (dated) — accepted; CLAUDE.md, decision doc, email, memory updated by owner** | — |
| N2 | Ch4 ordering: Option B (question-logic, recommended) vs Option A (audit-first) | B | shows the year's arc; audit still gets the most pages and leads Ch1/abstract |
| N3 | Split provenance wording | describe what actually ran (dataset split directories); drop the SHA-256/StratifiedGroupKFold text from Methods and fix `.claude/rules/data-pipeline.md` later | A§7.1 |
| N4 | `run_S64`/`run_S67` straddle splits | report as a limitation (in-domain CV group key is finer than the anatomy model) | A§7.2; no re-run under the freeze |
| N5 | 12-cell grid main text vs appendix (Q3c) | main text (dossier C: selective quotation was the failure mode) | |
| N6 | PTB-XL count conventions | one sentence naming the rule for 444 vs 438, 1,787 vs 1,891 | dossier B §4.3 |
| N7 | Commit `reports/2026-08-13_audit_artifacts/` (un-ignore) for examiners | yes | Declarations data statement |
| N8 | **RULED 18 Aug — accepted; Q1 split into Q1a (Track-3 pipeline: `target` vs `target_pool_measured`) / Q1b (circular: `source` vs `target`); never write "strict" unqualified (now grepped).** "strict" is two different scalers: P1's strict = `target_pool_measured` (target-side, un-imputed pool) vs P2's strict = `source` (no target info). Q1 in the Marta email lists (a) Track 3 PARTIAL "under strict" and (c) F2 "under strict/source" as if they were the same axis | name modes by code name everywhere; re-phrase Q1 for Marta as *two* choices: P1 (legacy `target` vs `target_pool_measured`) and P2 (`source` vs `target`) | otherwise the thesis (and Marta) will read Track 3's "strict" as "no target information", which is false |
| Q1–Q5 | as in v1 §5 | unchanged except Q1 wording (N8) | |

---

## 7. Updated schedule (from D1 = Tue 18 Aug)

Overnight already done: Declarations draft; Figs 1–3 (v1); GPU estimate; 6 verified references; the dossiers; this plan. That frees ~1 day.

| Day | Date | Work |
|---|---|---|
| D1 | Tue 18 | sign-off on v2 (+ N1–N7); **Ch3 §3.2–3.4** (data, model, glossary) drafted; Fig 3.1 schematic |
| D2 | Wed 19 | **Ch3 §3.5–3.8** + Tables 3.5/3.6; start §4.4 (audit) prose |
| D3 | Thu 20 | **§4.4** complete (Fig 1, Tables 4.10–4.11, A.1); §4.5 |
| D4 | Fri 21 | **§4.5, §4.6** complete (Figs 2–3, Tables 4.12–4.13) |
| D5 | Sat 22 | **§4.3** (Track 3, 12-cell grid, floor-aware Part A) |
| D6 | Sun 23 | **§4.1, §4.2**; Ch4 assembled → **send Ch3+Ch4 to Marta** |
| D7 | Mon 24 | Ch1, Ch2 (verify citations as they enter) |
| D8 | Tue 25 | Ch5, abstract, Declarations final, appendices → **full draft to Marta** |
| D9 | Wed 26 | **M8 integrity pass** (check_draft.py hard failures = 0; number trace; both scalers; floors; DOIs) |
| D10 | Thu 27 | M8 finish AM; feedback in PM |
| D11 | Fri 28 | final build, submit |

Sacrifice order unchanged (Fig 4.1 → §4.3 compresses → §4.2 compresses → Ch2 shrinks). M8 protected.

---

## 8. Overnight artifacts (all committed under `thesis_writeup/`)
`chapters/06_declarations.tex` (draft) · `figures/fig1_eta2_scatter.*`, `fig2_block_transfer.*`, `fig3_repair_forest.*` + `figures/src/*.py` + `figures/fig*_points.csv` (trace tables) · `references.bib` (6 verified entries) · `notes/dossiers/{A,B,C,E}_*.md` · `notes/gpu_hours_estimate.md` · `notes/check_draft.py` · `notes/forbidden_phrases.txt` · `notes/overnight_log_2026-08-17.md`.
