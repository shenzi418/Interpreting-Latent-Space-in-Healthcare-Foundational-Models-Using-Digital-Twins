# Thesis structure — *Interpreting the Latent Space in ECG Digital-Twin Foundation Models*

**Status**: DRAFT for sign-off, 2026-08-17 (day 0 of writing). Nothing below is chapter prose.
**Governing documents**: `reports/2026-08-13_thesis_endgame_decision.txt` (claim ladder C1–C6, Section-4 forbidden list, MUST/SHOULD/OPTIONAL) and `reports/2026-08-13_fidelity_audit_and_final_verification.md` (the corrected record, Parts A–E). Where this file and those disagree, those win.
**Deadline**: thesis due **Fri 28 Aug 2026** (programme-director email 2026-08-17). Viva Sept 7–18.

---

## 0. Binding constraints from the course template (read 2026-08-17)

Source: `C:\Users\haowe\Downloads\MResAIML_thesis\MResAIML_thesis\{main,includes,titlepage}.tex` (this revision already contains the Sustainability section from the programme director's second email). Copied verbatim into `thesis_writeup/`; the original `main.tex` is preserved at `thesis_writeup/notes/template_main_ORIGINAL.tex`.

| # | Constraint | Where it bites |
|---|---|---|
| T1 | `\documentclass[12pt,twoside]{report}`, A4, hmargin 2.8 cm, vmargin 2.0 cm, Bitstream Charter body font (`bch`), fancyhdr headers/footers, `\parindent 0` | page-count arithmetic below assumes this geometry (~350–400 words/page of prose) |
| T2 | **APA bibliography**: `\bibliographystyle{apa}` + natbib (`\citep`, `\citet`) | every entry in `references.bib` needs author/year/title/journal/vol/pages/DOI; author-year in text |
| T3 | **Mandated chapter names, in order**: Introduction · Background · Contribution · Experimental Results · Conclusion · Declarations | the decision doc's Ch1–Ch8 skeleton must map INTO these six (§2 below). Declarative titles (Q4b) can only live at *section* level unless Marta okays renaming chapters — I recommend **keeping the six template names untouched** and putting the declarative phrasing in section titles |
| T4 | **Declarations chapter has four mandatory sections**, names verbatim: *Use of Generative AI* · *Ethical Considerations* · *Sustainability* · *Availability of Data and Materials* | §4 below |
| T5 | AI disclosure must (a) name tool + version, publisher, URL; (b) describe use against the ICLR-2027 **required** list (synthetic data; theoretical models/conceptual frameworks; mathematical claims; proof ingredients; proof writing; hypotheses; methodology/experiment design or feedback; implementing methods; translation; data cleaning/reformatting; qualitative/thematic analysis; interpreting results) and the **recommended** list; (c) describe the review process; (d) confirm the work is the student's own; (e) delete the template's instructional paragraphs | "a substantial falsehood … produced by an LLM will be regarded as academic misconduct" — the number-tracing discipline in §6 is therefore an integrity requirement, not just hygiene |
| T6 | Ethical Considerations must state whether ethics approval was needed, risks considered, and (encouraged) broader impact / misuse | public de-identified + synthetic data → "no approval required", stated explicitly |
| T7 | Sustainability: how the research was done in an energy-efficient way | needs GPU model + approx GPU-hours from the owner (not derivable from the repo) |
| T8 | Availability: repo link; state whether personal / proprietary data were used (with the ICO personal-data link) | PTB-XL is de-identified public data; MedalCare-XL is synthetic; repo is public on GitHub |
| T9 | Front matter: abstract, acknowledgments, ToC (roman numerals) then arabic from Ch1 | already wired in `main.tex` |
| T10 | `\date{September 2026}` on the title page; degree line reads "**MSc** degree in Artificial Intelligence and Machine Learning" | ⚠ the template says *MSc*; the programme is an *MRes*. **Left as-is pending your call** — it is the course leader's own text, but it looks like a template bug. One-word change in `titlepage.tex`. |

**Other things found in the template (flagged, not fixed unless noted):**

- `main.tex` does `\input{notation}` but `notation.tex` is not shipped → **created** (`thesis_writeup/notation.tex`; holds all project macros: `\medalcare`, `\ptbxl`, `\macroF`, `\etasq`, `\circR`, `\Rfloor`, `\Rfloorptb`=0.29216, `\Rfloormedal`=0.09319, `\Ffloorptb`=0.1534, `\todo{}`, `\tracenote{}`, `\qone{}`).
- No `\bibliography{…}` line in the template (only the style) → **added** `\bibliography{references}`.
- `includes.tex` redefines `\@makechapterhead` **without** `\makeatletter`, so the custom chapter heading silently never applies (it redefines `\@` instead). Harmless; the default report-class heading is what every other student's PDF will show. Left alone.
- `\usepackage[pdftex,…]{hyperref}` → must compile with **pdflatex** (not xelatex/lualatex). `latexmk -pdf` does that.
- `fncylab` emits "not required after 2019" — cosmetic.
- No page/word limit is stated in the template, and a web search found none published for the MRes AI&ML → **open logistics item L1**: check the programme handbook / the programme-director email. Working assumption below: **~60 pages of main text** (Ch1–Ch5), excluding front matter, Declarations, references, appendix.

**Toolchain**: MiKTeX is installed (pdflatex, latexmk, bibtex, `apa.bst`, Charter fonts). `thesis_writeup/` **compiles cleanly today** (`latexmk -pdf main.tex` → 15-page skeleton PDF, bibliography resolved, no errors). No Overleaf needed; if you want Overleaf for Marta's comments, the folder uploads as-is.

---

## 1. Headline and register (unchanged from the decision doc)

- **Headline claim** (Section 1 of the decision doc, verbatim in spirit): a per-feature *informativeness-fidelity audit* of MedalCare-XL against PTB-XL is a new, validated instrument that **explains** why simulator-fitted territory readouts fail on real ECG, **predicts** which channels transfer before transfer is attempted, and shows the failure is **unrepairable** downstream because the information is absent from the synthetic signal, not misaligned with it. Conceptual claim: *marginal realism is not informativeness fidelity*.
- **Register**: instrument + mechanism, negative-as-measurement (Zech / DeGrave / Oakden-Rayner skeleton: fair statement of the belief → the negative as a measurement with a named mechanism → reusable instrument → prescription). Never a failure log.
- **World**: audit-as-instrument (F1+F2). The repair negative (F3) is load-bearing but not the headline.
- The registered title stays. The latent remains central to §4.3–4.5 (Q4a decides how loud the audit is in the abstract).

---

## 2. Chapter map: decision-doc Ch1–Ch8 → template's six chapters

Page budgets sum to ~60 pp main text (+ ~3 pp Declarations, + appendix, + references). Adjust once L1 is answered.

### Ch 1 — Introduction (template name) ← decision-doc Ch1  — **5 pp**
| § | Content | Claims | Figures/tables |
|---|---|---|---|
| 1.1 | The promise of cardiac digital twins for pathology readouts, stated fairly (MedalCare-XL scale; FM latents; the sim→real hope) | — | — |
| 1.2 | The question, and the standard belief being tested ("a marginally realistic synthetic cohort is a usable training/validation cohort") | C6 as the frame | — |
| 1.3 | Contributions = the claim ladder C1–C6 (+ the methodology bundle), each with the section that carries it | C1–C6 | — |
| 1.4 | Thesis outline | — | — |
Written at M7 (last), after results are stable.

### Ch 2 — Background (template name) ← decision-doc Ch2 — **9 pp**
| § | Content | Cite (all via papersflow.verify_citation; Part D of the main report is the verified list) |
|---|---|---|
| 2.1 | The 12-lead ECG and infarct localisation: acute-ST vs chronic-Q criteria; frontal QRS axis as the oldest localisation channel | Thygesen EHJ 2018; Das Circulation 2006 (both verified) |
| 2.2 | Cardiac digital twins and MedalCare-XL: what it simulates (θ = φ, z, size, ρ_eps_max), how it was validated — **quoted fairly**: per-lead marginal overlays vs PTB-XL+ (Fig 6, hand-picked class-level features), clinician Turing test 77.33% / 83% / **62% vs 39%** diagnosability | Gillette Sci Data 2023 (verified) — forbidden #25 governs the wording |
| 2.3 | PTB-XL: cohort, MI sub-labels, `infarction_stadium` acuity, folds | Wagner Sci Data 2020 (**verify**); Strodthoff JBHI 2021 (verify) |
| 2.4 | ECG foundation models; ECGFounder; the FM-interpretability lane is real-data only | ECGFounder paper (**verify**); CADENCE arXiv 2607.25244, ECG-InterpBench arXiv 2607.27404 (verify) |
| 2.5 | Inverse inference of infarct parameters from ECG — in-silico state of the art; sim-trained/real-tested exists for *other* localisation tasks | Li TMI 2024; Li RBME 2025; Doste Front Physiol 2022; Luongo CVDHJ 2021 (all verified) — forbidden "first sim2real ECG" |
| 2.6 | Evaluating synthetic data: TSTR → utility frameworks → feature-importance agreement → DT credibility/V&V; the gap the audit fills (nearest neighbours credited) | Esteban 2017; Goncalves 2020; Alaa 2021; El Emam 2024; Hudovernik 2024; Ortuno 2026; Morrison 2019; Pathmanathan 2024; Viceconti 2021; Ledezma 2019 (verified list) |
| 2.7 | Circular statistics for angular readouts: mean resultant length, phase-clustering bias, non-uniqueness of arg-max-R slope, chance-level caution | van Driel 2015; Kovach 2017; Fisher & Lee 1983; Kempter 2012; Combrisson & Jerbi 2015 (verified); **Jammalamadaka–Sarma year (1988 vs 1993) and Mardia & Jupp (2000) / Fisher (1993) monograph check still PENDING** — flag in draft |
| 2.8 | Negative results as measurements: the exemplar skeleton this thesis follows | Zech 2018; DeGrave 2021; Oakden-Rayner 2020; Christodoulou 2019; Ghassemi 2021 (verified); Locatello / Adebayo venue attributions UNVERIFIED → cite the arXiv record |
Written at M7. Only gap-filling literature search (papersflow + brave in tandem); no broad new survey.

### Ch 3 — Contribution (template name) ← decision-doc Ch3 (Data & Methods) + the instrument definition — **14 pp**
The template's "Contribution" chapter is where the *methods and the instrument* live; the empirical validation goes to Ch4.
| § | Content | Claims | Figures/tables |
|---|---|---|---|
| 3.1 | The contribution in one page: audit → prediction → repair; what is new (novelty fences A–F, "to our knowledge" + search-scope caveat) | ladder | **Fig 0** overview diagram (optional, S-level) |
| 3.2 | Data: MedalCare-XL (θ 4-dim, `territory_4c` derived from φ, 8-folder structure, seeded splits), PTB-XL (MI subclass, 4-class territory, acuity stadium, folds), **lead order** (reindex by `sig_name` on both domains; the historical aVL/aVF transposition and its fix — cite the physics identity aVL=(I−III)/2, forbidden #9), per-lead z-score on both domains | — | Table 3.1 cohort table |
| 3.3 | Encoder and latents: ECGFounder backbone + adapters; `exp8_leadfix_medalonly` (headline, never saw PTB-XL) vs `exp8_leadfix_dual` (sensitivity only); 1024-d export; **~2 effective encoder observations** (forbidden #3) | — | — |
| 3.4 | The 54-feature extraction: neurokit2 delineation; blocks ST_J60×12, Q_amp×12, R_amp×12, T_amp×12, globals×6; `T_amplitude_mV` ≡ `T_amp_II` (53 distinct); `ST_J60_avg_mV` is not the lead mean; coverage/missingness incl. the **MNAR flag** on sim interval globals (forbidden #17); frontal QRS axis = atan2(R_aVF, R_I) | — | Table 3.2 feature blocks + coverage |
| 3.5 | Circular methodology (Part B): territory as an angle; circular R̄ and its **constant-predictor floor as the supremum over label-independent predictors** (0.29216 PTB-XL / 0.09319 MedalCare); **permutation nulls sit below the floor** (Jensen identity); floor-free metrics (nearest-anchor macro-F1 with its own constant floor 0.1534/0.1257, circular η², MI); paired patient/run-block bootstrap; the **two mandatory nulls** (norm-matched random projection; shuffled-source-label refit); the zero-parameter axis baseline | methodology (ladder item 7) | Box: floor derivation |
| 3.6 | The informativeness-fidelity audit: per-feature η² vs territory per domain, 500-draw block bootstrap, blind-spot / spurious criteria (diff CI excludes 0), block-aware permutation for rank agreement, |SMD| for marginal realism | C1 definition | — |
| 3.7 | Transport protocol: readout fit on MedalCare only; **both scalers defined precisely** (strict/source = zero target information; target = target-cohort re-centring, mild transductive); efficiency = (F1_cross − floor_cross)/(F1_in − floor_in); paired tests only for arm-vs-arm | C3 definition | — |
| 3.8 | Repair arms: channel-restricted readouts via latent→feature map fit on MedalCare only; importance reweighting (+ ESS-matched control); diagonal CORAL as the recalibration boundary | C4/C5 definition | — |
| 3.9 | Pre-registration practice and the **analysis-pipeline glossary** (Phase-B2 classifier pipeline vs this-week CV pipeline vs F3 ridge+anchor pipeline — forbidden #23; every macro-F1 in Ch4 carries its pipeline label) | — | Table 3.3 pipeline glossary |
**Q1 declaration slot** = §3.7: "strict primary for raw-transport claims, target primary for information-present-after-recalibration claims, both always reported" — written as `\qone{…}` so it can be swapped when Marta answers.
First chapter drafted (M3).

### Ch 4 — Experimental Results (template name) ← decision-doc Ch4 + Ch5 + Ch6 + Part A — **25 pp**
| § | Title (declarative, section-level) | Content | Claims | Figures/tables |
|---|---|---|---|---|
| 4.1 | *The simulator relocates pathology information: the fidelity audit* (F1) — **7 pp** | the map; blind spots (28 + axis; Q_amp_III 0.006→0.324; axis 0.0020 vs 0.1297 at matched dispersion) and spurious channels (14 cols / 13 distinct; 9/12 ST_J60 leads; ST not uniformly spurious — V4/avg real>sim); **block-level inversion** (block-mean ρ=−0.587; per-feature −0.354 NOT significant, block-aware p=0.10/0.14; T within-block +0.706 POSITIVE) — forbidden #11; marginal-realism **dissociation** (never "anti-correlated", forbidden #12); MNAR confound; 4c/2c robustness with the R_amp_V6 exception (#16); MedalCare's own 39%-vs-62% as the clinician-level corroboration | **C1, C2, C6** | **Fig 1** (THE figure: sim-vs-real η² scatter, 55 points, block-coloured, CI whiskers, axis starred) · Table 4.1 block-mean η² sim/real + within-block ρ · Table 4.2 top blind spots / spurious with CIs · full 54-row audit table → **Appendix A.1** |
| 4.2 | *Fidelity predicts transfer* (F2) — **5 pp** | per-block in-domain → cross F1 under **both scalers**; ST at the constant floor (0.1551 vs 0.1534, CI contains floor; eff 0.006) vs Q best (0.612); paired block orderings (ST−Q −0.083 p=0.001) as the robust core; ρ(η²_sim, eff_src)=−0.900 exact one-sided p=0.0417 (two-sided 0.0833 misses — say so) + fold-seed sensitivity (seed 7: −0.700) — forbidden #13; η²_real↔eff ρ=+1.00; CORAL boundary (target: ρ≈+0.10, repairs scale only); axis fit-free 0.3043 vs sim-fitted 0.178 (#24); T block transfers best after recalibration; globals eff_target 0.936 NOT headlined (#18) | **C3** (+C5 axis frame, +C6 T-block) | **Fig 2** (per-block in→cross slope chart, both scalers side by side) · Table 4.3 = F2 block table with efficiencies + exact p |
| 4.3 | *Restriction and reweighting cannot repair it* (F3) — **5 pp** | Q+R restriction loses (−0.0552 p<0.001); ST12 at/below the random-projection bulk (#20); Q+R vs ST ordering holds (+0.0855); **inferior-lead 12-d parity** (+0.0025 p=0.79; both nulls; beats measured inferior features +0.0441) — reported as PARITY, best-of-8 caveat (#19; placement Q3a); reweighting HARMS (axis-pair −0.0732, survives ESS control; six-feature only with ESS caveat, #21); dual-encoder sensitivity mirrors; the paired p<0.001 attaches to F1 levels not ratio ordering (#14) | **C4, C5** | **Fig 3** (repair forest plot vs unrestricted, with null bands) · Table 4.4 = F3 arm grid + nulls |
| 4.4 | *What the latent readout carries, under floor-aware evaluation* (Part A) — **4 pp** | the constant floor in action: every transported latent R̄ below 0.29216 at label-free α; the α-window reductio (constant predictor still "beats" the empirical floor at α=1e7); the axis "wins R̄ by being a well-placed constant" (arc 41.9°, wrong cyclic order) — both arms fail differently and R̄ hides both; floor-free metrics: **zero-shot frame positive under target scaler only** (macro-F1 0.3402 vs random-projection and shuffled-refit nulls p=0.0033; strict: nothing survives, η² p=0.34); **supervised increment over the axis fails its random-projection null** (p=0.093) — forbidden #4; conditional MI non-redundancy; **in-domain mirror**: latent 0.6195 ≫ control54 0.4535 ≫ axis 0.2050 (axis η²=0.0020, below floor) with paired +0.1660 replicating +0.1523; S(b): only the branch-invariant integer points survive (S(1)>S(0) in every encoder; **no b̂ numerics, no "unit gain"**, Kempter 2012 cited) | **C5, C6**, methodology | Table 4.5 α sweep (compact; full grid → Appendix A.2) · Table 4.6 in-domain vs cross-domain three-arm mirror · Table 4.7 floor-free metrics × scaler |
| 4.5 | *The road to the audit* (decision-doc Ch6, compressed) — **4 pp** | (a) alignment dead-end: MMD variants + INLP reduce first-order distances, C2ST stays ~1.0; (b) C2ST is constitutionally blind — 77-cell lead-permutation sweep, spread 1e-5; transfer is a **corruption detector, not a lead-order detector**; (c) the lead-order bug, its physics-identity fix (#9); (d) **Track 3 pre-registered verdict quoted verbatim** (#7): 4-class POSITIVE not met (0.3440/0.3357 vs ≥0.45); 2-class PARTIAL met under strict (0.6521), missed under legacy (0.6299); (e) latent vs control54 cross-domain **null** (#6) — the 12-cell paired grid (main text vs appendix = Q3c; my recommendation: summary row in main, full grid Appendix A.3); (f) the in-domain advantage +0.1523 (Phase-B2 pipeline label) | context for C2/C5; forbidden #6, #7, #8, #9, #10 | Table 4.8 Track 3 verdict · Table 4.9 12-cell grid summary (or A.3) |
Written M4 (4.1), M5 (4.2–4.4), M6 (4.5).

### Ch 5 — Conclusion (template name) ← decision-doc Ch7 (Discussion) + Ch8 (Conclusion) — **7 pp**
| § | Content |
|---|---|
| 5.1 | Findings against the claim ladder (one paragraph per C1–C6, status MEASURED / CONDITIONAL / CONSISTENT-WITH exactly as the ladder states) |
| 5.2 | Marginal realism vs informativeness fidelity: implications for digital-twin validation practice; the audit as a pre-training gate; what MedalCare-XL's Turing result was telling us |
| 5.3 | Limitations: one simulator, one cohort, one backbone family (~2 effective encoder observations), one delineator; η² linearity; label-granularity mismatch (with the out-of-sample answer); scaler dependence (Q1); n=5 blocks for the ρ; best-of-8 selection on the parity cell; MNAR globals; sim-vs-real label construction |
| 5.4 | Recommended validation practice + future work (second simulator; second backbone; rank-based informativeness; delineator sensitivity — all framed as future, none run) |
| 5.5 | Concluding statement |
Written M7.

### Ch 6 — Declarations (template name) — **~3 pp** — content plan in §4 below.

### Appendix (after Ch6, before references)
A.1 full 54-feature audit table (η² sim/real, CIs, |SMD|, missingness, verdict) · A.2 α-grid / floor tables + renormalised anchor table (Part A.2/A.7) · A.3 twelve-cell paired grid (if Q3c → appendix) · A.4 reproducibility: seeds, commands, artifact inventory, script-provenance table (S4, thinned to one page).

---

## 3. Figures and tables — inventory and artifact source

Every figure is generated by a script under `thesis_writeup/figures/src/` that reads ONLY the frozen artifacts (plotting existing JSON is not a new experiment).

| Item | Lands in | Source artifact | Status |
|---|---|---|---|
| **Fig 1** η² scatter (55 pts, block colours, CI whiskers, axis starred) | §4.1 | `outputs/analysis/fidelity_audit/f1_fidelity.json` → `features[*]`, `axis` (byte-identical to `reports/2026-08-13_audit_artifacts/tmp_f1_fidelity.json`) | to build (M4) |
| **Fig 2** per-block in→cross slope chart, both scalers | §4.2 | `f2_blocks.json` → `per_block`, `efficiency`, `floors` | to build (M5) |
| **Fig 3** repair forest plot with null bands | §4.3 | `f3_repair.json` → `exp8_leadfix_medalonly.{paired,nulls,reweight}` | to build (M5) |
| Fig 0 overview diagram | §3.1 | none (schematic) | SHOULD — only if slack |
| Table 3.1 cohorts | §3.2 | `data/theta_mi_build_summary.json`, `data/ptbxl_mi_subclass_summary.json`, manifest | M3 |
| Table 3.2 feature blocks + coverage | §3.4 | `f1_fidelity.json` → `missingness` | M3 |
| Table 3.3 pipeline glossary | §3.9 | decision doc #23 | M3 |
| Table 4.1 block-mean η² + within-block ρ | §4.1 | `f1_fidelity.json` → `rank_agreement`, `addendum` | M4 |
| Table 4.2 top blind spots / spurious | §4.1 | `f1_fidelity.json` → `lists` | M4 |
| Table 4.3 F2 block table | §4.2 | `f2_blocks.json` | M5 |
| Table 4.4 F3 arm grid + nulls | §4.3 | `f3_repair.json` | M5 |
| Table 4.5 α sweep (compact) | §4.4 | `tmp_t4_alpha_grid.txt`, `tmp_t4_supplement.txt` | M5 |
| Table 4.6 three-arm mirror (in-domain vs cross) | §4.4 | `tmp_t35_control_mirror.py` output (main report A.6) | M5 |
| Table 4.7 floor-free × scaler | §4.4 | `tmp_floorfree_out.txt`, `tmp_floorfree_condmi_out.txt` | M5 |
| Table 4.8 Track 3 verdict (verbatim) | §4.5 | `reports/2026-08-11_integrity_audit_and_probe_map.md` §6.4c; `scripts/_audit_paired_grid.py` output | M6 |
| Table 4.9 / A.3 twelve-cell grid | §4.5 or A.3 | same | M6 |
| A.1 54-row audit table | Appendix | `f1_fidelity.json` (auto-generated LaTeX via script) | M4 |
| A.2 α grid | Appendix | `tmp_t4_alpha_grid.txt` | M5/M8 |
| A.4 reproducibility table | Appendix | main report "Reproduction" section | M8 |

Every number in prose gets an invisible `\tracenote{<artifact>:<key>}` in the source so M8 can grep source→artifact.

---

## 4. Declarations chapter — content plan

### 4.1 Use of Generative AI (required; template list must be answered item by item)
Draft facts to state (owner to confirm tool versions and dates):
- **Tool**: Claude (Anthropic) — used through **Claude Code** (CLI/IDE agent) and claude.ai; model versions used across the project **[owner to list, e.g. Claude Opus 4.x / Sonnet 4.x during Oct 2025–Jul 2026, Claude Fable 5 from Aug 2026]**; publisher Anthropic; URLs https://claude.ai and https://claude.com/claude-code. Also state MCP tooling used for citation verification (PapersFlow) and search (Brave) if you consider them "AI tools" — recommend a one-line mention.
- **Required-disclosure items — honest answers**:
  - design or provide feedback on research methodology or experiments: **YES, extensively** (experiment design, pre-registration rules, null constructions, the audit design)
  - implement methods: **YES** (most analysis and training code was written with Claude, then run and inspected by the student)
  - clean and reformat datasets: **YES** (manifests, lead-order fix, feature extraction pipelines)
  - interpret results: **YES** (results were discussed and interpreted collaboratively; final interpretation the student's)
  - propose or refine hypotheses: **YES** (e.g. the ST-vs-Q/R channel divergence, the audit-predicts-transfer hypothesis)
  - help develop theoretical models or conceptual frameworks: **YES** (the informativeness-fidelity audit as a concept; the "marginal realism ≠ informativeness fidelity" frame)
  - formulate mathematical claims / provide ingredients for proofs / assist in writing proofs: **YES** for the constant-predictor floor (supremum statement) and the permutation-below-floor identity — say so plainly
  - generate synthetic data sets: **NO** (all synthetic ECGs come from MedalCare-XL, produced by its authors; no AI-generated data)
  - assist with translation; qualitative/thematic analysis: **not applicable**
- **Recommended-disclosure items** — YES to: creating/modifying figures; suggesting experimental parameters; creating/editing code; drafting parts of the thesis; summarising/analysing literature; identifying gaps and relevant literature; brainstorming; sourcing information; editing for readability; formatting references; suggesting structure; title/keywords (partially). NO/NA: surveys/interviews, transcription.
- **Review process** (this is what makes the disclosure credible): every quantitative result was (i) produced by scripts committed at tag `freeze-2026-08-13`, (ii) re-implemented from scratch by an independent adversarial-verifier pass whose corrections were adopted (list: 9 not 8 spurious ST leads; block-aware p; exact permutation p; ESS control), (iii) traced number-by-number from the thesis text to the on-disk artifact in an integrity pass; every citation was verified against a bibliographic database before entering the bibliography; a fixed list of over-claims was grepped out of every draft; all AI-drafted prose was read, edited and approved by the student. Confirmation of own work + responsibility statement (template's closing sentence, adapted).
- Wording register: complete sentences, no minimising. Given "AI-produced falsehood = misconduct", the disclosure should *reference the integrity pass* explicitly.

### 4.2 Ethical Considerations
- No ethics approval required: PTB-XL is a public, de-identified clinical dataset (PhysioNet); MedalCare-XL is fully synthetic; no new human data, no patient contact, no re-identification attempt. Ethics review with supervisor early in the project → state outcome. **[owner: confirm what the early ethics review recorded]**
- Risks considered: mis-use of the *result* (over-reading a negative sim→real result as "digital twins are useless" — the thesis says the opposite: validate informativeness before use); mis-use of the *audit* (cherry-picking faithful channels to over-sell a simulator). Broader impact: safer validation practice for synthetic-data pipelines in cardiology.

### 4.3 Sustainability
- Reused a pre-trained backbone (ECGFounder) rather than pre-training; fine-tunes were single-GPU, few epochs, adapters + linear head only; every audit/transfer/repair analysis is CPU-only; no large hyper-parameter sweeps; the experimental freeze prevented redundant re-runs. **[owner: GPU model, approximate total GPU-hours, location/energy mix if known]** — I can estimate GPU-hours from `outputs/*/metrics.json` timestamps if you want a defensible number.

### 4.4 Availability of Data and Materials
- PTB-XL v1.0.3 (PhysioNet; licence: **verify** CC-BY 4.0), MedalCare-XL (Zenodo/Sci Data; licence: **verify**), ECGFounder weights (public release; licence: **verify**), code: https://github.com/shenzi418/Interpreting-Latent-Space-in-Healthcare-Foundational-Models-Using-Digital-Twins (tag `freeze-2026-08-13`; `thesis_writeup/` in the same repo). Derived artifacts (latents, features, JSON results) — state which are in the repo (`data/*.npz`, `outputs/quick_waveform_check/`) and which are regenerable (`outputs/` is gitignored; regeneration commands in Appendix A.4). ⚠ `reports/` is gitignored — decide whether the audit-artifact folder (`reports/2026-08-13_audit_artifacts/`, ~small) should be un-ignored and committed for the examiners; **recommendation: yes, commit it** (it is the provenance for every thesis number).
- Personal data: none held or processed beyond the public de-identified PTB-XL release; no proprietary data. Include the ICO link as the template does.

---

## 5. Supervisor-dependent branch points (Q1–Q5) and the both-way-proof rule

Until Marta answers, all results prose is written **both-scaler-proof** (every transport number under both scalers; every "positive" with its scaler named). Text whose *wording* would change is wrapped in `\qone{}` so it is greppable.

| Q | Decision | Default I will draft to (swap on answer) | Sections touched |
|---|---|---|---|
| Q1 scaler primacy | strict primary for raw-transport claims + target primary for information-present-after-recalibration claims, both always reported (decision-doc proposal) | as proposed; declared in §3.7; §4.2 (F2 lives under strict), §4.4 (zero-shot positive lives under target), §4.5 (Track 3 secondary endpoint) each state both | 3.7, 4.2, 4.4, 4.5, 5.1, abstract |
| Q2 audit resolution | block-level inversion is the licensed claim; within-block "uncorrelated-to-positive" | as proposed | 4.1, 1.3, abstract |
| Q3a inferior-lead parity in main text? | main text as "measured boundary of repair", parity + best-of-8 caveat | main text | 4.3 |
| Q3b chapter/section title "restriction and reweighting cannot repair it" | keep at section level; reweighting harm quoted on the axis-pair summary with ESS caveat for six-feature | keep | 4.3 |
| Q3c 12-cell grid main vs appendix | summary row in §4.5, full grid Appendix A.3 | that | 4.5, A.3 |
| Q4a abstract leads with the audit? | yes, within the registered title | yes | abstract, 1.3 |
| Q4b declarative titles | section-level only (template chapter names fixed) | section-level | all |
| Q4c 39%-vs-62% as the hook | yes, quoted fairly with Fig 6 credited | yes | 1.1, 2.2, 4.1 |
| Q5 publication | ML4H Findings Sep 10 (post-thesis); skip NeurIPS workshops (~Aug 29) | skip workshops; ML4H after Aug 28 | none in-thesis |
| **L1** (new, logistics) page/word limit; submission time-of-day and portal on Aug 28 | assume ~60 pp main text; submit by midday | budgets in §2 |
| **L2** (new) supervisor's full name/title for the title page; "MSc" vs "MRes" wording | TODO macro on the title page | titlepage |

---

## 6. Recompressed schedule — M3–M9 mapped day-by-day to Fri 28 Aug

Original budget (decision doc §5): M3 1.5 d, M4 3 d, M5 3 d, M6 1 d, M7 2.5 d, M8 2 d, M9 1.5 d = 14.5 d over Aug 16–30. Available now: **Aug 17 (today, half day) → Aug 28 = 11.5 working days**, i.e. −3 d. Constraints honoured: **M8 ≥ 1.5 d protected**; a **supervisor-feedback window before submission**; the M-ordering, the MUST/SHOULD/OPTIONAL split, and the freeze stand.

| Day | Date | Milestone | Deliverable (each pauses at section boundaries for your review) |
|---|---|---|---|
| D0 | **Mon 17 Aug** | boot + structure (this doc) + freeze commit ✔ · **send Marta email + second-marker viva-slot email** · start M3: Ch3 outline agreed | structure sign-off; Ch3 outline |
| D1 | Tue 18 Aug | **M3** Ch3 Contribution (3.2–3.9), Tables 3.1–3.3 | Ch3 draft complete |
| D2 | Wed 19 Aug | **M4** §4.1 audit: Fig 1 script + figure, Tables 4.1–4.2, appendix A.1 auto-table; prose for the map + blind spots/spurious | §4.1 first half |
| D3 | Thu 20 Aug | **M4** §4.1 finish (block inversion, dissociation, MNAR, 4c/2c, Turing corroboration) · **M5** starts: Fig 2 + Table 4.3 | §4.1 complete; §4.2 tables |
| D4 | Fri 21 Aug | **M5** §4.2 prose (F2, both scalers, exact p, seed sensitivity, CORAL, axis frame-vs-fit) · Fig 3 + Table 4.4 | §4.2 complete |
| D5 | Sat 22 Aug | **M5** §4.3 (F3 repair, parity, reweighting) · §4.4 (Part A: floors, axis-as-constant, floor-free × scaler, in-domain mirror, S(b) integer points) | §4.3–4.4 complete |
| D6 | Sun 23 Aug | **M6** §4.5 road to the audit (Track 3 verbatim, 12-cell grid summary, C2ST sweep, lead-order fix) · **send Ch3+Ch4 draft to Marta** (feedback window opens) | Ch4 complete |
| D7 | Mon 24 Aug | **M7** Ch1 Introduction · Ch2 Background 2.1–2.5 (verify every citation as it enters) | Ch1, half Ch2 |
| D8 | Tue 25 Aug | **M7** Ch2 2.6–2.8 · Ch5 Conclusion/Discussion · abstract · Ch6 Declarations (owner facts inserted) · **send full draft to Marta** | full draft v1 |
| D9 | Wed 26 Aug | **M8** integrity pass day 1: number-by-number trace (every `\tracenote` resolves to an artifact value), Section-4 forbidden-phrase grep, both-scaler check, every R̄ with its floor, pipeline labels on every macro-F1, DOI check | integrity log |
| D10 | Thu 27 Aug | **M8** day 2 (AM: figures/tables/captions/front matter, `\todo{}` count → 0) · **M9** (PM): incorporate Marta's feedback | v2 |
| D11 | **Fri 28 Aug** | **M9**: final revision, PDF build, submission dry-run, **submit** (target: by midday, well before the deadline time — L1) | submitted PDF + tagged commit `thesis-submitted` |

M8 = D9 full + D10 AM = **1.5 d** ✔. Supervisor window: partial draft D6 → full draft D8 → feedback absorbed D10 PM–D11 AM.

### What is cut or thinned to fit (starting from the decision doc's SHOULD/OPTIONAL lists)
| Item | Original | Now | Rationale |
|---|---|---|---|
| M4 (audit) | 3 d | **1.5 d** | figure + tables are script-generated from JSON; the prose is well-specified by C1/C2/C6 |
| M5 (prediction+repair) | 3 d | **2 d** (incl. §4.4 Part A) | same |
| M6 (history) | 1 d | **0.75 d** | §4.5 is compressed to 4 pp; the 12-cell grid goes to the appendix (Q3c default) |
| M7 (intro/background/conclusion) | 2.5 d | **2 d** | Background limited to the already-verified Part D list + gap-filling only |
| M9 | 1.5 d | **1 d** (D10 PM + D11) | feedback window shifted earlier by sending Ch3+Ch4 on D6 |
| S1 extra figures | 0.5 d | **Fig 3 forest plot KEPT (it is a core figure); polar axis plot DROPPED** unless slack on D5 | |
| S2 ML4H 4-page skeleton | 0.5 d | **after Aug 28** (Sep 1–9) | zero days inside the window, as the decision doc already intended |
| S3 skip NeurIPS workshops | decision | **skip** (recommend to Marta, Q5) | collides with M8 |
| S4 reproducibility appendix | 0.5 d | **thinned to one page** inside M8 (A.4) | provenance is already in the archived reports |
| S5 viva crib | 0.5 d | **after Aug 28** (viva prep, doubles with the 4-pager) | |
| O1 η² linearity sensitivity | 1 d | **NO** | new numbers 3 days before deadline; becomes a limitation sentence (§5.3) |
| O2 53-distinct-feature rerun | 0.5 d | **NO** | cite the verifier's spot-check instead |
| Fig 0 overview diagram | — | only if D1 finishes early | schematic, no data |
| Ch2 §2.7 monograph check (Mardia & Jupp / Fisher) | pending | **do at D7 via papersflow/brave (≤30 min)**; if unresolvable, claim 1 of Part B is written as "we could not find … in standard references" | honesty over novelty |

Slack: none formal. If a day slips, the order of sacrifice is: Fig 0 → §4.4 shrinks to 3 pp (α table to appendix) → §4.5 shrinks to 3 pp → Ch2 shrinks to 7 pp. **M8 and the both-scaler/floor/trace discipline are never sacrificed.**

---

## 7. Writing discipline (applies to every draft; enforced by grep at M8 and at every section hand-off)

1. Section-4 forbidden phrasings (26 items) — grep list kept in `thesis_writeup/notes/forbidden_phrases.txt`; run before any section is shown to you.
2. Every circular R̄ appears with its constant floor (`\Rfloorptb`, `\Rfloormedal`) or not at all.
3. Every transport number appears under BOTH scalers or not at all (`\qone{}` marks Q1-dependent wording).
4. Arm-vs-arm claims cite paired tests only; every macro-F1 carries its pipeline label (Table 3.3 glossary).
5. Every number traces to `reports/2026-08-13_audit_artifacts/` or `outputs/analysis/fidelity_audit/` via `\tracenote{}`; untraceable → flag, do not write.
6. Every citation enters `references.bib` only after `papersflow.verify_citation`, with the verification date in a comment.
7. No new experiments. A gap → an honest limitation sentence.
8. `\todo{}` count must be 0 at submission (`grep -c "\\todo" chapters/*.tex`).

---

## 8. Day-1 reminders (today, Mon 17 Aug)

- [ ] **Send the supervisor email** — updated draft at `reports/email_drafts/2026-08-14_scaler_and_framing_to_marta.md` (already includes the Aug-28 date and the viva-slot request). Q1 gates the results wording.
- [ ] **Email the second marker** (same as poster second marker) about a viva slot in Sept 7–18 — programme director says book EARLY; ask for availability across the whole window and propose 2–3 concrete slots once Marta replies.
- [ ] Confirm L1 (page limit / submission time & portal) from the programme handbook or the programme-director email.
- [ ] Give me: supervisor's full name/title for the title page; GPU model + rough GPU-hours (Sustainability); which Claude versions to name (AI disclosure); what the early ethics review recorded.
