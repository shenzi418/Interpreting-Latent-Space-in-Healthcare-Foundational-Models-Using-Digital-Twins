# MRes Thesis — Proposed Structure for Review

**Interpreting the Latent Space in ECG Digital-Twin Foundation Models**

Author: Haowen (Owen) Shen · Supervisor: Marta · Date: 19 August 2026
Submission: Friday 28 August 2026

---

## Purpose of this document

This is the proposed chapter-and-section structure of the thesis, for your
review before I draft the remaining chapters. It follows the required template
(Introduction / Background / Contribution / Experimental Results / Conclusion /
Declarations). For each chapter it gives the role, the sections, the key result
each section carries, and the figures and tables it will hold, plus a provisional
page budget. The two places where I'd most value your steer are marked **[for your
steer]**, and there is a short list of them at the end.

The methods chapter (Contribution) and the Declarations chapter are already
drafted; the results and framing chapters are outlined below and will follow.

---

## 1. The thesis in one page

**The question (as registered).** Can the latent space of an ECG foundation model
(ECGFounder), adapted to a cardiac digital twin (MedalCare-XL), be interpreted in
the twin's underlying parameters — in particular infarct *territory* — and does
that interpretation carry across to real ECGs (PTB-XL)?

**The answer, and the turn it took.** In-domain (on the simulator) the latent
carries territory information strongly, well beyond hand-crafted features. Across
the domain gap this largely collapses: under a decision rule fixed in May, the
coarse anterior-vs-inferior distinction transfers but the finer four-way one does
not, and a 1024-dimensional foundation-model representation does no better than a
single number a cardiologist reads by hand (the heart's frontal electrical axis).
Rather than stop at that negative, the thesis asks *why*, and answers it with a new
instrument.

**The contribution.** A per-feature *informativeness-fidelity audit* of the digital
twin: for each standard ECG measurement, how much infarct-territory information it
carries in the simulator versus in real patients. The finding is that the simulator
does not merely lose information — it **relocates** it. It writes territory into the
ST segment (the signature of an *acute* infarct, which is what it simulates), while
real clinical infarcts are mostly *old* and carry territory in the Q/R waves and the
electrical axis. A readout trained on the simulator is therefore taught to rely on
exactly the channels that are uninformative in reality. The audit **predicts** which
channels transfer (before any transfer is attempted) and shows the failure is
**not repairable** by any reweighting or channel-selection trick — the information
is absent from the synthetic signal, not merely misaligned.

**The message.** A simulator can pass distributional realism checks — its feature
distributions match real data, and it was validated that way — and still be the
wrong thing to train on, because matching distributions is not the same as
preserving what is diagnostic. *Marginal realism is not informativeness fidelity.*
To our knowledge no prior work audits a mechanistic cardiac simulator this way; the
MedalCare-XL authors' own validation hinted at the gap (their simulated ECGs were
correctly diagnosed by clinicians 39% of the time versus 62% for real ones) without
localising it, which is exactly what this instrument does.

**A secondary methodological contribution.** The angular readouts are scored with a
circular metric (the mean resultant length) that has a large "chance" floor a
constant predictor attains for free. The thesis states this floor explicitly, shows
the usual permutation test does not substitute for it, and re-reads every result
against it. This corrects a pitfall that, uncaught, would have turned a below-chance
result into an apparent success.

**Contributions, itemised** (each maps to a results section):

- **C1 — Interpretable in-domain, partial transfer.** Territory is decodable from
  the latent in-domain (+0.15 macro-F1 over hand-crafted features); across the gap
  the pre-registered two-class endpoint is met, the four-class is not, and the
  latent is statistically indistinguishable from the hand-crafted control.
- **C2 — The fidelity audit and its map.** 28 features plus the electrical axis are
  informative in reality but flat in the simulator ("blind spots"); nine ST-segment
  leads are the reverse ("spurious"); the mismatch is a relocation across
  measure-blocks (ST/T carry the simulator's signal, Q/R and the axis carry
  reality's).
- **C3 — The audit predicts transfer.** Real-domain informativeness ranks which
  channels transport perfectly; simulator informativeness ranks it backwards — the
  simulator's strongest channel transports at the chance floor.
- **C4 — The failure is unrepairable downstream.** Restricting the readout to the
  faithful channels does not beat the full latent; reweighting the simulator toward
  real feature distributions makes transfer worse — the deficit is missing
  information, not covariate shift.
- **C5 — What does transfer is small and localised.** A 12-dimensional
  inferior-lead subspace matches the full latent; the axis transfers only as a
  coordinate frame, and fitting it on the simulator destroys it.
- **C6 — The conceptual claim.** Marginal realism and informativeness fidelity are
  dissociated properties of a synthetic cohort; validating the first does not
  validate the second.
- **M — Floor-aware circular evaluation** (the methodological contribution above).

---

## 2. Chapter-by-chapter structure

Page budgets are provisional and assume the ~350–400 words/page geometry of the
template. The total main-text budget is flagged as an open item — see §5.

### Chapter 1 — Introduction  (≈ 4 pp)

Role: state the promise, the question, and the contributions; no results.

| § | Content |
|---|---|
| 1.1 Motivation | Digital twins produce labelled ECGs at scale with the generating parameters known; foundation models produce reusable representations; combining them promises readouts of mechanistic quantities no clinical label set provides. States fairly the assumption this rests on — that a distributionally-validated synthetic cohort is a usable training cohort — and what is at stake if that assumption fails invisibly. |
| 1.2 The question | The registered question, then how it developed: the in-domain answer, the cross-domain shortfall under a fixed rule, and the "why" the shortfall forces. |
| 1.3 Contributions | The C1–C6 + M list above, each naming the section that delivers it. |
| 1.4 Thesis outline | One paragraph. |

### Chapter 2 — Background  (≈ 8 pp)

Role: the clinical, model, data, and methodological background; each subsection
ends with one sentence on what the thesis takes from it.

| § | Content |
|---|---|
| 2.1 ECG and infarct localisation | The 12-lead frontal-plane logic and the QRS axis; why *acute* injury shows in the ST segment and *established* infarct in Q waves and R-wave loss (Thygesen 2018 criteria; Das 2006 on Q-wave regression) — hence a chronic-dominated cohort writes territory into Q/R. |
| 2.2 Digital twins and MedalCare-XL | The simulation pipeline and the released dataset; the parameters θ; and its published validation quoted fairly (feature-marginal overlays; clinician Turing test 77%/83%/62-vs-39%) — distributional and clinician-level, never per-feature or transfer-linked. |
| 2.3 PTB-XL | The real cohort, its diagnostic statements, folds, MI sub-labels and acuity grading; why the real MI cohort is chronic-dominated. |
| 2.4 ECG foundation models | ECGFounder; adapters/linear probing on frozen backbones; the interpretability lane to date is real-data only. |
| 2.5 Inverse inference of infarct parameters | The in-silico state of the art (Li, IEEE TMI 2024) and sim-trained/real-tested localisation for *other* tasks — establishing that the sim→real step for infarct parameters from a foundation-model latent is unclosed; the thesis measures and explains the gap, it does not close it. |
| 2.6 Domain gap and alignment | MMD, class-conditional MMD, INLP, the classifier-two-sample-test detector, CORAL/AdaBN recalibration — the tools §4.2 uses, and their known blind spot. |
| 2.7 Evaluating synthetic data beyond marginals | Train-on-synthetic-test-on-real; utility and feature-importance-agreement frameworks; digital-twin credibility/V&V — none audits per-feature label informativeness of a mechanistic simulator against a real cohort, which is the slot the audit fills. |
| 2.8 Circular statistics for angular readouts | Mean resultant length and its biases; circular correlation coefficients; the arg-max non-uniqueness result; the chance-level caution — motivating the floor. |
| 2.9 Negative results as measurements | The register the thesis adopts (Zech, DeGrave, Oakden-Rayner, Christodoulou, Ghassemi): fair statement of the belief, negative as a measurement with a mechanism, a reusable instrument as the durable contribution. |

### Chapter 3 — Contribution (Methods)  (≈ 12 pp)  — *drafted*

Role: everything built and every protocol, so the results chapter is pure findings.

| § | Content | Objects |
|---|---|---|
| 3.1 Overview of the study | One frozen encoder, two cohorts, three questions, one instrument. | Fig 3.1 (pipeline) |
| 3.2 Data | MedalCare-XL and PTB-XL cohorts; the infarct-territory label design; the pre-processing and a corrected loading pipeline (a lead-order bug found and fixed). | Tables 3.1–3.3 |
| 3.3 The foundation model and its adaptation | ECGFounder; adapters and heads; the fine-tuning modes; **which single encoder carries the headline results** (the one never exposed to real labels) and why the six encoders are not independent replicates; latent export. | — |
| 3.4 The experimental programme | The sequence of experiments and how each fed the next. | — |
| 3.5 Hand-crafted ECG features and the control arm | The 54-feature spatial control the latent is measured against, and the frontal-axis baseline. | — |
| 3.6 Analysis pipelines | The classifier pipeline (P1) and the circular-geometry pipeline (P2), the two transport modes, and the alignment diagnostics. | Tables 3.4–3.6 |
| 3.7 The informativeness-fidelity audit | The instrument: per-feature information measure per domain, confidence intervals, and the separation of marginal realism from informativeness. | — |
| 3.8 Transport, prediction and repair protocols | How the audit is turned into transfer predictions, and the channel-restriction and reweighting interventions, with their statistical tests and null models. | — |

### Chapter 4 — Experimental Results  (≈ 24 pp)

Role: the findings, in the order that makes the argument. This is the longest
chapter; its six sections are the spine of the thesis.

| § | What it delivers | Objects |
|---|---|---|
| 4.1 Adapting ECGFounder to the twin | The encoders classify well in-domain and transfer at class level only weakly; the encoders differ little — what matters is what the latent *carries*, not the encoders as replicates. | Table 4.1 |
| 4.2 The domain gap, and why the standard detector misses it | The two latent distributions are perfectly separable however we align them; a 77-input-corruption sweep shows the detector is blind to gross corruption — so distribution-level alignment is not the route, and the question becomes *what information* the latent carries. | Tables 4.2–4.4, opt. Fig 4.4 |
| 4.3 Territory decodes in-domain, partially across the gap | In-domain the latent beats the control by +0.15; the pre-registered rule (two-class met under both data-normalisation modes at the dated 0.55 bar, four-class not met); latent vs control indistinguishable across the gap; and the floor-aware circular reading, including the zero-parameter axis baseline matching the latent. | Tables 4.5–4.9 |
| 4.4 The fidelity audit: where each domain writes territory | **The centrepiece.** For every ECG measurement, we compare how much infarct-location information it carries in the simulator versus in real patients. This produces a map of which measurements the simulator reproduces faithfully and which it does not; shows the mismatch is a systematic *relocation* (the simulator carries location in the ST segment, reality in the Q/R waves and the electrical axis); and demonstrates that how *realistic* a measurement looks does not predict whether it carries the right information. (Expanded in plain terms below the table.) | **Fig 1**, Tables 4.10–4.11 |
| 4.5 The audit predicts which channels transfer | Per-block transfer, both modes; the simulator's strongest channel transporting at the chance floor while real informativeness ranks transport perfectly; recalibration repairs scale but not information; fitting on the simulator destroys a channel that works fit-free. | **Fig 2**, Table 4.12 |
| 4.6 What transfers, what does not, and the limits of repair | Channel restriction does not beat the full latent; the one arm at parity (a 12-dimensional inferior-lead subspace, reported as parity, not a win); reweighting harms; the residual gap is at representation level; a closing statement that fixing it requires a better *simulator*, not a better encoder. | **Fig 3**, Table 4.13 |

**What Section 4.4 does, in plain terms** (the thesis's central section).
We take the ~50 standard ECG measurements and, for each one, ask the same simple
question in each domain separately: *how strongly does this measurement distinguish
the four infarct territories?* Computing that once on the simulator and once on real
patients gives two "informativeness scores" per measurement, and the section is
built around comparing them. Five things come out of that comparison:

1. **A map of faithful vs unfaithful measurements.** Some measurements carry real
   information about location in patients but are flat in the simulator (the
   simulator is effectively *blind* to them - mostly the Q/R waves and the electrical
   axis); others are the reverse, where the simulator relies on them but they carry
   little in reality (the ST-segment leads). We list both, with confidence intervals.

2. **The mismatch is systematic, not scattered.** It is not that individual
   measurements are randomly off - whole *families* of measurements are swapped. The
   simulator concentrates location information in the ST/T-wave family, while reality
   concentrates it in the Q/R-wave family and the axis.

3. **A clean worked example - the electrical axis.** It varies just as much in the
   simulator as in real data (so the simulator has not simply frozen it), yet it
   carries about 65 times less location information in the simulator than in reality,
   and in real patients it points in the physiologically correct direction. This is
   the single clearest number in the thesis.

4. **The key evidence for the thesis's message.** How *realistic* a measurement looks
   - whether its distribution matches real data - does not predict whether it carries
   the right information: some of the least realistic-looking measurements carry the
   right information, and some of the most realistic hide the biggest gaps. This is
   what "realism is not informativeness" means, shown directly.

5. **Why this happens, and an independent check.** The pattern fits a clinical
   explanation - the simulator models *acute* (fresh) infarcts, which show in the ST
   segment, whereas the real cohort is dominated by *old* infarcts, which show in the
   Q/R waves - which we present as *consistent with* the data, not proven. And it
   lines up with the dataset authors' own finding that clinicians correctly diagnosed
   the simulated ECGs only 39% of the time versus 62% for real ones: the same deficit
   they saw at the whole-ECG level, which our audit pins down measurement by
   measurement.

### Chapter 5 — Conclusion  (≈ 5 pp)

| § | Content |
|---|---|
| 5.1 Findings | One paragraph per contribution C1–C6 + M, each with its section and its honest strength. |
| 5.2 What it means | For validating synthetic cohorts (the audit as a cheap, encoder-free pre-training gate); for MedalCare-XL's users and authors (which channels to trust, the Turing result explained at feature level). |
| 5.3 Limitations | One simulator, one real cohort, one backbone family, one feature-extraction toolchain; the information measure is univariate; label-granularity differences between domains. |
| 5.4 Future work | Framed as future, not as gaps. |
| 5.5 Close | The registered question answered as far as one simulator, one cohort and one backbone allow. |

### Chapter 6 — Declarations  (≈ 3 pp)  — *drafted*

Required by the template: **Use of Generative AI** (a full, honest disclosure —
this project used an AI coding assistant extensively for implementation, analysis
scaffolding, and drafting, all verified by me; I take responsibility for every
number and claim), **Ethical Considerations** (public de-identified datasets, no
new data collection), **Sustainability** (compute footprint; all analysis is
CPU-only over pre-computed representations, no retraining in the write-up phase),
and **Availability of Data and Materials** (PTB-XL and MedalCare-XL are public; the
code repository link).

### Appendix

The full 54-feature audit table; the superseded pre-fix result tables (flagged as
such); the metric-floor and normalisation-sweep tables; the input-corruption sweep;
the decision-rule label design; and a reproducibility section (seeds, commands,
artifact map).

---

## 3. Figures and tables

**Figures.** Fig 3.1 pipeline schematic *(built)*; **Fig 1** the audit map — sim vs
real informativeness for all features, the electrical axis starred *(built; the
single most important figure)*; **Fig 2** per-block transfer, both modes *(built)*;
**Fig 3** the repair comparison with null bands *(built)*; optional Fig 4.4 the
corruption-sweep scatter.

**Tables.** Six methods tables (3.1–3.6, built); thirteen results tables (4.1–4.13)
covering the encoder metrics, the domain-gap conditions, the in-domain and
cross-domain decoding, the decision-rule verdict, the twelve paired latent-vs-control
cells, the floor-aware circular reading, the block informativeness map, the
blind-spot/spurious lists, the block-transfer predictions, and the repair arms.

---

## 4. Framing decisions I'd value your steer on  **[for your steer]**

These affect the shape of the thesis, so I'd rather agree them with you before
drafting the results chapters than present them as faits accomplis.

1. **Emphasis.** The plan leads on the audit instrument and the "marginal realism is
   not informativeness fidelity" message, with the original latent-interpretability
   question as the framing that motivates it. Is that the emphasis you want, or would
   you prefer the latent-interpretability angle kept more central throughout?

2. **Reporting the two normalisation modes.** A few cross-domain results depend on
   which of two reasonable data-normalisation choices is treated as primary. My plan
   is to report *both* everywhere and never pick the flattering one, stating plainly
   where they disagree. I believe this is the honest choice; I'd like your
   endorsement of it, since it costs some crispness.

3. **How prominently to present the "repair fails" result** (Ch 4.6) and the
   detailed null-model grid (Ch 4.2) — main text (my preference, as rigour) or
   appendix (as decluttering)?

---

## 5. Page budget and the one open logistics item

| Chapter | Provisional pages |
|---|---|
| 1 Introduction | 4 |
| 2 Background | 8 |
| 3 Contribution (Methods) | 12 |
| 4 Experimental Results | 24 |
| 5 Conclusion | 5 |
| **Main text total** | **≈ 53** |
| 6 Declarations | 3 |
| References + Appendix | ~10 |

**Open item:** I could not find a stated page or word limit for the MRes AI&ML
thesis in the template or online. Could you confirm the limit (and the submission
time-of-day / portal for the 28th)? If it is tighter than ~53 pages of main text,
the natural place to compress is Chapter 4 by moving more of the supporting grids to
the appendix.

---

## 6. Writing plan to 28 August

Results chapter first (it exists in full and verified form as analysis outputs), in
the order 4.4 → 4.5 → 4.6 → 4.3 → 4.1 → 4.2; then Chapters 1, 2, 5, the abstract and
appendix; then a dedicated integrity pass that checks every number in the draft
against its source artifact and every citation, before a buffer day for your
comments. I'll send you **Chapters 3 and 4 first** once 4 is assembled, then the
**full draft**, so you can steer early rather than all at once.
