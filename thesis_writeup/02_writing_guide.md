# Writing guide for this thesis

**Purpose.** Every chapter is drafted with AI assistance and that is disclosed. The goal of this guide is not concealment; it is that the thesis reads as what it is — one person's considered account of a year's work — and that every sentence earns its place. Machine-drafted prose has recognisable habits (documented below from public field guides¹) that also happen to be bad academic style: it over-signposts, over-hedges, over-decorates, and lets rhythm and vocabulary do the work that evidence should do. Removing those habits makes the text better, not just less recognisable. The mechanical parts of this guide are enforced by `notes/check_draft.py` (section 7 "STYLE"); the rest is applied by reading.

¹ Sources consulted 2026-08-18: Wikipedia, *Signs of AI writing* (vocabulary lists, negative parallelisms, rule of three, copula avoidance, structural tells); several detector write-ups (em-dash density, triplets, uniform paragraphs, hedge clusters, "delve/underscore/pivotal" lexicon); Imperial DoC *Guide to Individual Projects* (no page limit; "a 150-page dissertation is not twice as good as a 75-page one"; a project is not awarded a distinction if the write-up is not good enough; a critical appraisal is expected).

² Refresh 2026-08-24 (polish phase): (a) Turnitin ships a dedicated **AI-paraphrase / bypasser detector** (announced Aug 2025) aimed at synonym-swap passes over machine text — so polishing must change sentence *structure and rhythm*, never word-by-word substitution, which is exactly the texture that detector targets; (b) **burstiness** (variance in sentence length, and separately in paragraph length) is one of the strongest statistical signals GPTZero-class detectors use — a run of 25–30-word sentences reads as generated even when every word is clean, and one very long sentence followed by a short one reads as human; (c) **over-long sentences** (40+ words, stacked subordinate clauses) and **balanced constructions** (matched semicolon pairs, "X is A; and Y is B" on repeat) are per-sentence tells: split them, and let some sentences be short; (d) em-dash overuse is now the folk fingerprint — there is active discourse about em-dashes causing false accusations in peer review; (e) known false-positive risk factors: very formal register, short texts (<300 words), non-native-English phrasing that has been heavily edited — another reason the fix is structural variety, not more polish.

---

## 1. Voice and stance

- **First person singular for decisions, interpretations and admissions** ("I chose the MedalCare-only encoder because…", "I read this as…", "I did not run…"). Impersonal or passive for procedures ("Latents were exported once per run"). Never "we" for a single-author thesis except in the Declarations formula the template dictates. Never "the author".
- **Tense**: past for what was done ("the readout was fitted"), present for what is the case ("the floor is 0.292", "the simulator relocates…"), present perfect sparingly.
- **Confidence calibrated to evidence, once.** State the finding plainly, give the number and its interval, name the caveat once, move on. Do not stack hedges ("may potentially suggest"), and do not repeat a caveat in every paragraph that touches the result — put it where the number first appears and in the limitations.
- **No sales, no drama.** No "crucially", "strikingly", "remarkably", "the key insight". If something matters, the sentence structure and the number make it matter.
- **No meta-talk about honesty.** Do not write "honestly", "the honest answer is", "to be transparent". Honesty is shown by reporting both scaler modes and the failed arms, not asserted.

## 2. Paragraph architecture

- One idea per paragraph. First sentence states the point (usually with the number). Middle sentences give the evidence and the comparison. Last sentence, if any, gives the consequence for the argument — **not a punchline, not a rhetorical turn.** Many paragraphs should simply end when the evidence is done.
- **Vary length.** Two-sentence paragraphs and eight-sentence paragraphs both belong. A run of paragraphs of equal length reads as generated.
- Prose carries argument; **tables and figures carry lists.** Do not bullet-point inside results prose. If three things need enumerating, use a sentence with "first… second… third" only when the order matters; otherwise a table.
- Do not open paragraphs with "Notably,", "Importantly,", "Interestingly,", "It is worth noting that", "In this section, we". Start with the subject.

## 3. Sentence-level habits to remove

| Habit | What to do instead |
|---|---|
| **Negative parallelism**: "not X but Y", "not just X, it is Y", "X, not Y" as a rhetorical device | Say Y. Use "not X" only when the reader would otherwise assume X and it must be denied explicitly — and then at most once per section. |
| **Enumerative setup**: "The value of X is twofold." / "Three things follow." before a list | Give the items directly; the count announces itself. (Caught by the structural scan 2026-08-24; triads and balanced semicolon sequences are the cadence that survives vocabulary filters.) |
| **Rule of three**: three adjectives, three clauses, three examples by reflex | Use the number of items the content has. Two is fine. Four is fine. |
| **Over-long, over-balanced sentences**: 40+ words with stacked subordinate clauses; runs of sentences of near-identical length; matched pairs ("X is A; and Y is B") sentence after sentence | Split them. Vary length deliberately — some sentences should be short. Detectors measure sentence-length variance ("burstiness") directly, so uniform rhythm is a statistical tell even when every word is clean. The linter reports the count of >40-word sentences per chapter. |
| **Em dashes** as the default connective (my own habit) | Target ≤ 1 per 200 words. Prefer a full stop, a comma, a colon, or parentheses. Keep the dash for a genuine interruption. |
| **Copula avoidance**: "serves as", "stands as", "represents", "functions as", "constitutes" | "is". |
| **Present-participle tails**: "…, highlighting the…", "…, underscoring…", "…, reflecting…" | Full stop. New sentence with a subject and a verb. |
| **AI lexicon**: delve, underscore, highlight (verb), pivotal, crucial, robust (as praise), nuanced, landscape, tapestry, testament, leverage, harness, foster, bolster, showcase, meticulous, intricate, comprehensive, holistic, seamless, notably, additionally (sentence-initial), moreover, furthermore, "in the realm of", "it is important to note", "plays a role" | Plain verbs and nouns: shows, is, gives, uses, because, so. "Robust" only in its statistical sense with a test named. |
| **Elegant variation** (calling the same thing by five names) | Call things by one name. The latent is "the latent"; the 54-feature arm is "the control"; the frontal QRS axis is "the axis". Consistency beats variety in technical prose. |
| **Hedge stacks** ("could potentially", "may possibly", "seems to suggest") | One hedge, chosen for its exact strength: "is consistent with", "suggests", "shows". |
| **Over-signposting** ("As discussed above", "As will be shown in Section…", "This chapter has presented") | A cross-reference in parentheses when needed; no narration of the document. |
| **Rhetorical questions** | State the question once in the Introduction; elsewhere, statements. |
| **Sycophancy toward the reader or the field** ("the reader will appreciate", "this exciting area") | Delete. |
| **Bold in prose** for emphasis | None. Bold only in table headers and, sparingly, for defined terms at first use (italic preferred). |
| **Curly quotes / smart apostrophes** | LaTeX ``…'' and ' only. |
| **Every paragraph ending in a generalisation** ("…which is exactly the point.", "…and that is the finding.") | End on the fact. |

## 4. Numbers and evidence (nothing from nothing)

- Every quantitative statement names: the pipeline (P1/P2/P3), the encoder if it matters, the scaler mode by code name, the evaluation rows (n), and the uncertainty (CI or paired test). One `\tracenote{}` per number in the source.
- Every circular R̄ carries its floor. Every cross-domain number appears under both scaler modes of its pipeline or not at all. Arm-vs-arm claims use paired tests only. Permutation floors are never evidence of effect size.
- Every claim about prior work is a citation verified through papersflow, cited at the point of use, with the "to our knowledge, within the searches described" caveat on novelty statements. No citation from memory.
- Every interpretation is labelled with its strength: "shows" (measured, CI-backed), "predicts" (validated out of sample), "is consistent with" (mechanism supported but not established — the acuity test failed), "suggests" (one simulator, one cohort). The claim ladder in `01_thesis_structure_v3.md` fixes which word each finding gets.
- Retracted results are not mentioned as findings. Superseded numbers appear only in the appendix, flagged.

## 5. Proportion (what a year supports)

- ≈55 pages of main text; three data figures + one schematic; ~14 tables; ≈10 pages of appendix. The DoC guide's line applies: length is not quality.
- Background: only literature that a claim or a design choice rests on. No survey of the field for its own sake. Each background subsection ends by stating what the thesis takes from it.
- Novelty statements: fenced (Part D of the main report). Never "first sim2real ECG"; never "the mechanism for"; never "MedalCare only validated marginals".
- Limitations are a section with content, not a paragraph of ritual modesty: one simulator, one cohort, one backbone (≈2 effective encoder observations), one delineator, η² linearity, label granularity with the out-of-sample answer, scaler dependence (Q1b), n=5 blocks for the ρ, best-of-8 parity selection, MNAR globals, anchors assigned not measured, `run_S64/S67`, no multi-seed.

## 6. Human markers to keep

- **Specific, concrete detail** that only the person who did the work has: the exact check that caught the lead transposition; the fact that the T block was the only informativeness-matched block *and* the most marginally unrealistic; that 24.7 % of the real rows carry an imputed column and what was done about it; that a permutation p at 10⁻⁴ was reached by both arms.
- **Decisions with reasons**: why the MedalCare-only encoder is the headline; why the four-class label; why two scaler modes; why the axis baseline.
- **Admissions with causes**: what did not work and the measured reason; what was planned and not run (Exp 2/3, multi-seed) — stated once, plainly.
- **Judgement in the discussion**: "I read this as…", "the reading I find most defensible is…" — one voice, taking a position, with the evidence beside it.

## 7. Process per section

1. Outline (in v3) → draft → run `python notes/check_draft.py chapters/<file>` → fix hard failures and read the STYLE report → read the section aloud once → cut 10–15 % → check every number against its `\tracenote` → hand-off.
2. On hand-off, the owner reads for voice: does it sound like *you* explaining it to Marta? Rewrite any sentence you would not say.

## 8. Style linter (implemented in `notes/check_draft.py`, section 7 "STYLE")

Report-only (never a hard failure), per chapter file:
- em dashes per 1,000 words (target ≤ 5); "not X but Y" constructions per 1,000 words (target ≤ 1); AI-lexicon hits (list in `notes/style_lexicon.txt`, target 0 outside quotations); sentence-initial "Additionally/Moreover/Furthermore/Notably/Importantly/Interestingly" (target 0); "serves as/stands as/represents/functions as" (target 0); participle tails ", -ing …" at sentence end (report count); triplet heuristic "A, B, and C" adjective/noun runs (report count); paragraph-length coefficient of variation (flag < 0.35 as too uniform); mean sentence length and its SD (flag SD/mean < 0.35); sentences over 40 words (report count and maximum; flag if > 8 % of sentences); bold in prose outside tables (target 0); "we" outside Declarations (target 0).
