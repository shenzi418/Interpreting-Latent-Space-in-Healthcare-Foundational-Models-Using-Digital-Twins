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

Page budgets are provisional and assume the template's page geometry; the total is
an open item (see section 5). Each row says, in plain terms, what that section will
be about.

### Chapter 1 - Introduction  (~4 pp)

*What this chapter does: sets up the promise, the question, and the contributions. No results yet.*

| Section | What it will say |
|---|---|
| 1.1 Motivation | A cardiac "digital twin" is a physics simulator that produces ECGs for which the exact underlying cause is known (which artery, how large the infarct, and so on) - labels that real hospital data never has. A "foundation model" is a large pre-trained network that turns an ECG into a compact numerical summary. Combining the two promises a way to read physical properties of a heart attack directly off a real ECG. This section states the assumption that idea rests on - that if a simulator's ECGs look statistically realistic, it is safe to train on them - and why it matters if that assumption quietly fails. |
| 1.2 The question | The question set at the start of the project: can the model's internal summary of an ECG be interpreted in terms of the simulator's physical settings, and does that interpretation carry over to real ECGs? Then how it actually turned out - it works on the simulator's own ECGs, only partly on real ones - and the "why" that gap forces us to answer. |
| 1.3 Contributions | The list of contributions (the six findings plus the methodological one), each pointing to the section that delivers it. |
| 1.4 Thesis outline | One short paragraph mapping out the chapters. |

### Chapter 2 - Background  (~8 pp)

*What this chapter does: gives the clinical, model, data and statistical background a reader needs. Each part ends with one sentence on what the thesis takes from it.*

| Section | What it will say |
|---|---|
| 2.1 The ECG and locating a heart attack | How a standard 12-lead ECG encodes the direction of the heart's electrical activity, and the key clinical fact the thesis relies on: a *fresh* heart attack and an *old* one leave their marks on different parts of the ECG (fresh ones in the "ST segment", old ones in the "Q and R waves"). Supported by standard cardiology references. |
| 2.2 Digital twins and the MedalCare-XL simulator | How the simulator generates ECGs, the dataset it produced, and - stated fairly - how its authors checked it was realistic (comparing the overall shape of the measurements to real data, and asking clinicians to spot the fakes). The point to carry forward: those checks were about overall realism; they never checked measurement by measurement, and never tested whether models trained on it transfer to real patients. |
| 2.3 The PTB-XL real-patient dataset | The real clinical dataset we test against: its diagnoses, its standard train/test splits, its heart-attack sub-types, and how it records the age of each infarct - which is how we can show its heart attacks are mostly old. |
| 2.4 ECG foundation models | The specific pre-trained model we use, and how we attach a small trainable layer to it without retraining the whole thing. The gap: prior work interpreting these models has only ever used real data, never a simulator. |
| 2.5 Reading heart-attack properties back off an ECG | The best prior work on inferring infarct properties from an ECG was done entirely in simulation; related tasks have been taken to real patients, but this particular step - inferring infarct properties from a foundation model's summary of a *real* ECG - has not been achieved. This thesis measures and explains the gap; it does not claim to close it. |
| 2.6 Making two datasets "look alike", and the standard test for it | The standard toolbox for forcing two datasets' numerical summaries to line up, and the standard test for whether they have - plus what that test is known to miss. These are the tools used in section 4.2. |
| 2.7 Judging synthetic data beyond "does it look realistic" | Existing ways people judge synthetic data (train a model on synthetic and test it on real; various utility scores). The gap this thesis fills: none of them measures, feature by feature, whether a physics simulator carries the same diagnostic information as real data. |
| 2.8 Statistics for angle-valued predictions | Some results are angles (a direction on the heart), which need special statistics. This covers the standard measure, its known pitfalls, and why it needs a proper "chance level" baseline - the small methodological point the thesis makes. |
| 2.9 Negative results done well | A short survey of respected papers whose main contribution is a rigorous negative result, to establish the style this thesis follows: state the belief fairly, turn the negative into a measurement with a mechanism, and leave behind a reusable tool. |

### Chapter 3 - Contribution (Methods)  (~12 pp)  - *already drafted*

*What this chapter does: describes everything we built and every procedure, so that Chapter 4 can be pure results.*

| Section | What it will say | Figures/Tables |
|---|---|---|
| 3.1 Overview | The whole study on one page: one frozen pre-trained model, two datasets (simulator and real), three questions, and one new audit tool. | Fig 3.1 (diagram) |
| 3.2 The data | The two datasets; how we defined which region of the heart each infarct sits in; how the ECGs are prepared; and a data-loading bug (two ECG leads swapped) that we found and fixed. | Tables 3.1-3.3 |
| 3.3 The model and how we adapt it | The pre-trained model, the small trainable pieces we add, and the ways we train them. Also: which single trained model our headline results use (the one never shown any real-patient labels), and why the six models we trained should not be counted as six independent experiments. | - |
| 3.4 The sequence of experiments | The order the experiments were run in, and how each one led to the next. | - |
| 3.5 The hand-crafted comparison | A set of 54 standard, hand-measured ECG features, plus a single clinical number (the heart's electrical axis), that we hold the AI model up against as fair baselines. | - |
| 3.6 The two analysis methods | The two ways we read infarct location out of the model, the two ways of moving a reader from the simulator to real data, and the checks on how far apart the two datasets are. | Tables 3.4-3.6 |
| 3.7 The fidelity-audit tool | The core instrument: how it measures, for each ECG feature, how much infarct-location information it carries in each dataset (with error bars), and how it separates "looks realistic" from "carries the right information". | - |
| 3.8 Prediction and repair procedures | How we turn the audit into predictions about what will transfer, and the two attempts to repair the transfer, each with its statistical tests. | - |

### Chapter 4 - Experimental Results  (~24 pp)

*What this chapter does: the findings, in the order that builds the argument. This is the longest chapter and the spine of the thesis.*

| Section | What it shows | Figures/Tables |
|---|---|---|
| 4.1 Adapting the model to the simulator | The trained models diagnose well on their own dataset but transfer only weakly to the other, and they are all very similar to each other - so the interesting question is not which model is best, but what information the model's summary contains. | Table 4.1 |
| 4.2 The gap between simulator and real, and why the usual test misses it | A simple classifier can always tell simulator ECGs from real ones, however hard we try to align them; and a stress-test shows the usual "are they aligned?" check fails to notice even severe damage to the input. Conclusion: trying to align the two datasets is a dead end, and the real question is what information the model carries. | Tables 4.2-4.4, opt. Fig 4.4 |
| 4.3 Reading infarct location: strong on the simulator, only partial on real data | On the simulator the model reads infarct location far better than the hand-crafted features. On real data, under a rule we fixed in advance, it gets the coarse region right (front vs bottom of the heart) but not the finer four-way split, and it does no better than the hand-crafted features - or even than a single clinical number. | Tables 4.5-4.9 |
| 4.4 The fidelity audit: where each dataset stores infarct-location information | **The centrepiece.** For every ECG measurement, we compare how much infarct-location information it carries in the simulator versus in real patients. This produces a map of which measurements the simulator gets right and which it does not; it shows the mismatch is systematic (the simulator stores location in one family of measurements, real patients in a different family); and it shows that how realistic a measurement looks does not predict whether it carries the right information. Explained in plain terms just below this table. | **Fig 1**, Tables 4.10-4.11 |
| 4.5 The audit predicts what transfers | We test the audit's predictions: the measurements it flags as unfaithful are exactly the ones that fail to carry over to real ECGs. The simulator's favourite measurement transfers no better than chance, while how useful a measurement is in real patients predicts its transfer almost perfectly. A simple recalibration helps a little but does not supply the missing information. | **Fig 2**, Table 4.12 |
| 4.6 What does transfer, and why the failure cannot be patched | We try to repair the transfer two ways - keeping only the faithful measurements, and reweighting the simulator to resemble real data. Neither beats the full model, and reweighting makes it worse; the one restriction that ties the full model is small and specific. The conclusion: the missing information cannot be recovered after the fact - it needs a better simulator, not a better AI model. | **Fig 3**, Table 4.13 |

**What Section 4.4 does, in plain terms** (the thesis's central section).
We take the ~50 standard ECG measurements and, for each one, ask the same simple
question in each dataset separately: *how strongly does this measurement distinguish
the four infarct regions?* Computing that once on the simulator and once on real
patients gives two "informativeness scores" per measurement, and the section is
built around comparing them. Five things come out of that comparison:

1. **A map of faithful vs unfaithful measurements.** Some measurements carry real
   information about location in patients but are flat in the simulator (the
   simulator is effectively *blind* to them - mostly the Q/R waves and the electrical
   axis); others are the reverse, where the simulator relies on them but they carry
   little in reality (the ST-segment measurements). We list both, with error bars.

2. **The mismatch is systematic, not scattered.** It is not that individual
   measurements are randomly off - whole *families* of measurements are swapped. The
   simulator concentrates location information in the ST/T-wave family, while real
   patients concentrate it in the Q/R-wave family and the electrical axis.

3. **A clean worked example - the electrical axis.** It varies just as much in the
   simulator as in real data (so the simulator has not simply frozen it), yet it
   carries about 65 times less location information in the simulator than in real
   patients, and in real patients it points in the physiologically correct direction.
   This is the single clearest number in the thesis.

4. **The key evidence for the thesis's message.** How *realistic* a measurement looks
   - whether its distribution matches real data - does not predict whether it carries
   the right information: some of the least realistic-looking measurements carry the
   right information, and some of the most realistic hide the biggest gaps. This is
   what "realism is not informativeness" means, shown directly.

5. **Why this happens, and an independent check.** The pattern fits a clinical
   explanation - the simulator models *acute* (fresh) infarcts, which show up in the
   ST segment, whereas the real dataset is dominated by *old* infarcts, which show up
   in the Q/R waves - which we present as the *likely* explanation, consistent with
   the data rather than separately proven. It also lines up with the dataset authors'
   own finding that clinicians correctly diagnosed the simulated ECGs only 39% of the
   time versus 62% for real ones: the same shortfall they saw for the whole ECG,
   which our audit pins down measurement by measurement.

### Chapter 5 - Conclusion  (~5 pp)

| Section | What it will say |
|---|---|
| 5.1 What we found | One short paragraph per contribution, each stated with its honest strength (proven, well-supported, or consistent-with). |
| 5.2 What it means | For anyone using a simulator to train a model: the audit is a cheap check to run first - it needs only the features and the labels, not the model. For the makers and users of this particular simulator: which measurements to trust, and their own clinician test explained measurement by measurement. |
| 5.3 Limitations | Honest scope: one simulator, one real dataset, one family of models, one measurement toolkit; the information measure looks at one feature at a time; and the two datasets label infarcts at slightly different levels of detail. |
| 5.4 Future work | Framed as opportunities, not gaps. |
| 5.5 Closing | The original question answered as far as one simulator, one dataset and one model allow. |

### Chapter 6 - Declarations  (~3 pp)  - *already drafted*

Required by the template: a full and honest **AI-use disclosure** (this project used
an AI coding assistant heavily for writing code, running analyses, and drafting text,
all checked by me - I take responsibility for every number and claim); **ethics**
(public, de-identified datasets, no new data collected); **sustainability** (the
compute used - all analysis runs on a CPU over already-computed results, with no
model retraining during the write-up); and **data availability** (both datasets are
public; a link to the code).

### Appendix

The full table of all 54 audited features; the older results from before the
data-loading fix (clearly flagged as superseded); the supporting statistics tables;
the input stress-test; how the infarct-region labels were defined; and a
reproducibility section (the exact settings and commands needed to regenerate every
number).

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
