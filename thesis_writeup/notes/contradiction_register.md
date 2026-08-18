# Contradiction / open-issue register (from dossiers A, B, C, E) — evaluated 2026-08-18

Legend: **W** = affects thesis wording (must be applied); **A** = appendix/limitation only; **N** = no thesis impact (repo hygiene). Resolution = what the thesis text will say.

| # | Item (source) | Class | Resolution |
|---|---|---|---|
| 1 | Track 3 PARTIAL bar 0.55 (dated May) vs 0.65 (undated) — B §4.3, mine | **W** (ruled) | Quote the May rule; "met under both modes at the dated 0.55 bar"; footnote 0.65. Done in plan/CLAUDE.md. |
| 2 | "strict" = `target_pool_measured` (P1) vs `source` (P2) — A §5.1, C Card J | **W** (ruled) | Name every mode by code name; grep enforces it. Q1 split into Q1a/Q1b. |
| 3 | Split provenance: SHA-256/StratifiedGroupKFold text vs `add_medalcare_splits.py` reading directory names — A §7.1 | **W** | §3.2.1 already states the true mechanism (dataset's own partition, 100 % agreement). Rules file to be corrected after submission. |
| 4 | `run_S64`/`run_S67` straddle splits — A §7.2 | **A** | Limitation sentence in §3.2.3 (written) and §5.3. No re-run. |
| 5 | `args.json` missing for 8/19 runs — A §7.3 | **W** | Table 3.3 caption: pre-Aug hyper-parameters "from run conventions, not recorded". Done. |
| 6 | Pre/post-fix PTB-XL F1 not comparable (STTC vs CD subset) — A §7.7 | **W** | † marks in Table 3.3; never a before/after sentence. Done. |
| 7 | `exp5/6_3class` dual-head status unverifiable from artifacts — A §7.5 | **A** | Appendix A.2 row note: "dual-head by run convention; no args.json". Main text uses "dual-head" as the design name only. |
| 8 | `exp4_ptbxl`, `exp6_*` latent dirs of unknown source encoder — A §7.4 | **N** | No figure/table uses them; excluded from the artifact map. |
| 9 | `config/theta.json` referenced but absent — A §7.6 | **A** | Reproducibility appendix: physics-head θ list recoverable only from `joint_baseline/physics_metrics.json`; head unused in results. |
| 10 | 51-name θ_phys vs 4-member θ share a symbol — A §7.6 | **W** | θ_phys subscript introduced in §3.3 (written). |
| 11 | `prepare_medalcare.py` comments mislabel iab/fam — A §7.8 | **N** | Thesis glossary follows the dataset README (Table 3.1 does). |
| 12 | Scaler CLI default `target` vs function default `target_pool` — A §7.9 | **W** | §3.6 states which mode produced each reported cell (by code name); results tables name the mode. |
| 13 | 444 (legacy 3-class single-territory) vs 438 (4-class) on fold 10 — A §7.10, B §4.3 | **W** | §3.2.2 names the rule beside each count (written). |
| 14 | PTB-XL "angles" are assigned anchors — A §7.11 | **W** | §3.2.4 states it (written); the constant floor follows from it. |
| 15 | 24.7 % of the n=4324 rows carry ≥1 imputed feature column (control arm) — A summary 13 | **W** | §3.5 states coverage; §4.3 restates beside the 12-cell grid with the "imputation confound excluded in the control's favour" check. |
| 16 | PTB-XL 3-class test n 1,787 vs 1,891; CD positives 274 vs 285 — B §4.3 | **A** | Both are pre-fix numbers (STTC-not-CD subset); appear only in Appendix A.2 with the manifest count quoted once. Main text uses post-fix counts from Table 3.2 only. |
| 17 | MedalCare denominators 2,386 / 2,126 / 1,200 — B §4.3 | **W** | Table 3.1 fixes them: 2,386 = test split; 1,200 = MI test rows; 2,126 (3-class-filtered test) appears only where the shared-label task is scored, and is named. |
| 18 | θ arity 4 vs 5 (transmural duplicate) — B §4.3 | **W** | §3.2 θ paragraph (written). |
| 19 | Li et al. venue MICCAI vs IEEE TMI 2024 — B §4.3 | **W** | Cite the papersflow-verified TMI record only. |
| 20 | "Six experiments" vs seven-row table — B §4.3 | **N** | Not quoted. |
| 21 | "Four defects" vs five — C Card A | **W** | Wording: "four confirmed defects in the code path (lead order, normalisation, PTB-XL filter, AUC column order) plus one structural label contradiction (LCX_0.3_post)". §3.2.3 to be aligned. |
| 22 | Retracted May narratives (φ sign flip; Inferior→Anteroseptal collapse; per-lead-norm worsens; 0.998 calibrator; INLP converged / K-ordering; vacuous permutation_p_r2) — B §4.1 | **W** | Never appear as findings. The lesson (instrument errors caught by controls) appears in the Methods "how errors were caught" box without the retracted numbers. |
| 23 | Superseded magnitudes (AP era, Exp 1–7, 2×2, Phase-B2 in-domain, Track 1/Tier 1–2, INLP magnitudes) — B §4.2 | **A** | Appendix A.2 only, flagged "pre-fix; superseded". Main text: designs + qualitative conclusions that survived (C2ST≈1 under every attempt; dimensionality not the bottleneck; in-domain θ decodable). |
| 24 | Oracle ceiling 0.867 for folder-labelled 4-class — B §4.4 | **W** | §3.2 θ paragraph: applies to folder labels; φ-derived labels have no ceiling (written, corrected). |
| 25 | INLP v1 pool contamination; permutation p floors (N_PERM=200) — B §4.4 | **A** | Appendix INLP protocol note; the p-floor caveat is already a standing rule (never quote a floor p as evidence). |
| 26 | June–July 2026 no repo activity; no poster artefact — B §7 | see timeline decision | The thesis carries no month-by-month timeline (see §2 of this note); the poster is mentioned in the acknowledgements/declarations only if the owner wants. |
| 27 | Multi-seed never run — B §5.1 | **W** | Stated in §3.4 (written) and §5.3; uncertainty from block bootstrap instead. |
| 28 | No meeting minutes; supervisor guidance is the student's rendering — B §7 | **W** | Attribute as "following the supervisor's suggestion" without quoting her, except the one emailed caveat (INLP) which is quoted as an email. |
| 29 | Fidelity-audit JSON mtimes 08-17 — E | **N** | Byte-identical copies of the 08-13 archive; resolved. |
| 30 | `exp8_leadfix_medalonly` PTB-XL F1 0.4567 — E | **W** | Marked ‡ passive readout (done). |

## Timeline decision (owner question, 2026-08-18)
The thesis will not carry a dated month-by-month timeline. Table 3.4 becomes a **phase table without dates** (question → setup → where reported), and every result is presented as the final, corrected pipeline's result. Dates stay in the repository and the reports (which remain the truthful record); the thesis is organised by logic, not chronology. Nothing in the thesis will assert a date that is not true.
