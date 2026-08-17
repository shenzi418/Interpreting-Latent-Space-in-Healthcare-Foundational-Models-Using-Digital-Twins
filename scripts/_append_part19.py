"""Append Part 19 (Stage 4.2 lead-permutation sweep) to the execution log."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FP = REPO / "reports" / "EXECUTION_LOG_2026-08-10.md"

PART = """

---

# PART 19 — STAGE 4.2: LEAD-PERMUTATION SENSITIVITY SWEEP

*Run 2026-08-11. Script `scripts/_t42_leadperm_sweep.py`, summariser
`scripts/_t42_summarise.py`, artifact
`outputs/analysis/leadperm_sweep/leadperm_sweep.json` (77 cells, 16.2 min).*

Plan item 4.2: *"sweep single channel permutations and measure C2ST/transfer
under each. Establishes the diagnostic as a general tool, not a one-off bug fix."*

## 19.1 What was actually done

The 2026-08-10 audit found **one** instance of a lead-order corruption that
transfer noticed and C2ST did not. One instance is an anecdote. This sweeps the
whole space:

* **Intervention.** PTB-XL test (the *target* domain) is permuted at inference.
  The territory probe is fit once on correctly-ordered MedalCare-train latents
  and **never refit** (`metadata.probe.refit = false`), so nothing downstream can
  absorb the permutation.
* **Encoder.** `exp8_leadfix_baseline`.
* **Cells.** identity + all C(12,2)=66 transpositions + 10 seeded random full
  permutations = 77.
* **Metrics.** territory macro-F1 (n=438, 2000-draw permutation null, 400-draw
  bootstrap CI); linear C2ST; GBDT C2ST on 19 pre-declared cells; multi-bandwidth
  MMD².

**Precondition, passed exactly.** The identity cell reproduced the stored
`exp8_leadfix_baseline_ptbxl_test/latents.npz` to `max|d| = 0.000e+00` — bit
identical. The sweep is measuring the permutation, not a re-implementation.
(`use_adapter` was derived from the checkpoint's own keys rather than from
`args.json`, which records `use_adapter: false`; the exact match confirms the
checkpoint's keys are authoritative.)

## 19.2 The pre-declared reading fired SUPPORTED

Thresholds were fixed in the script docstring before the table existed (and the
summariser's, after only the 3-cell smoke run — disclosed in its docstring).

| criterion | threshold | measured | verdict |
|---|---|---|---|
| transfer macro-F1 spread, 66 transpositions | ≥ 0.05 | **0.0929** | YES |
| linear C2ST pinned | min > 0.99, spread < 0.01 | min 1.0000, spread **1e-5** | YES |
| GBDT C2ST pinned | min > 0.99, spread < 0.01 | min 0.9999, spread **9e-5** | YES |

Across cells spanning macro-F1 **0.1618 → 0.3201** — a factor of two — the linear
C2ST moves by one part in 10⁵ and the GBDT C2ST by nine. Neither tracks the
damage: Spearman ρ vs macro-F1 = **−0.040** (p=0.73) and **−0.031** (p=0.90).

**This part is clean and is the deliverable.** C2ST at AUROC 1.0 is saturated,
and a saturated statistic cannot rank anything. Reporting "C2ST ≈ 1.0, therefore
the domains are far apart" says nothing about whether the gap is a physiological
difference or a channel transposition.

## 19.3 But it is supported for a weaker reason than the framing implied

The honest breakdown of where the 0.0929 spread comes from:

| class | n | mean F1 | min | max | mean MMD² |
|---|---|---|---|---|---|
| identity | 1 | 0.2599 | — | — | 0.1778 |
| limb–limb | 15 | 0.2802 | 0.2586 | 0.3201 | 0.2014 |
| limb–precordial | 36 | 0.2584 | 0.2272 | 0.2849 | 0.1940 |
| precordial–precordial | 15 | 0.2561 | 0.2312 | 0.2867 | 0.1945 |
| **random (full)** | 10 | **0.2012** | 0.1618 | 0.2245 | 0.2673 |

* **Gross corruption is detected.** Random permutations: mean 0.2012 vs identity
  0.2599, **7/10 below identity's 95% CI**, and **0/10 reach p<0.05** — transfer
  collapses to non-significance. Random vs transposition Mann–Whitney
  **p = 2.1e-07**.
* **Single transpositions are not detected.** **0 of 66** fall below identity's
  95% CI [0.2162, 0.3011]. The two that fall outside it are *above* it
  (II↔aVR 0.3191, aVR↔aVF 0.3201). **39 of 66 transpositions score higher than
  the correct lead order.**

So the instrument separates *scrambled* from *plausible*. It does not resolve
single-channel errors, which is the class of bug that actually occurs in practice
and the class the audit found.

## 19.4 The finding that cuts against our own audit

**The historical bug cell ranks 70/77 in damage — it scores *above* identity.**

| cell | macro-F1 | Δ vs identity | perm p | C2ST_lin | C2ST_gbdt |
|---|---|---|---|---|---|
| identity (correct order) | 0.2599 | — | 0.1139 | 1.0000 | 0.9999 |
| **aVL↔aVF** (the 2026-08-10 bug) | **0.2817** | **+0.0218** | 0.0385 | 1.0000 | 1.0000 |

The audit reported that *correcting* aVL/aVF lifted transfer from p≈0.76 to
p≈0.002 and treated that as corroboration. Here, *introducing* the same
transposition on the target side moves transfer the other way.

**This does not overturn the lead-order fix.** The fix rests on a physics
identity verified empirically on the signals themselves — `aVL = (I − III)/2`,
`aVF = (II + III)/2`, channel 4 *is* aVL (`data-pipeline.md`). That evidence is
independent of any transfer number. The two interventions also differ: the audit
retrained the encoder with corrected leads on the **source** domain, changing what
was learned; this sweep holds the encoder fixed and permutes the **target** at
inference.

**What it does overturn is the evidentiary weight of the corroborating
statistic.** "Transfer improved after the fix" is much weaker evidence than it
appeared, because in this sweep 39/66 arbitrary transpositions also improve
transfer. Any writeup that leans on the p 0.76 → 0.002 movement as *independent
confirmation* of the fix must be re-worded to lean on the lead-identity check
instead. See §19.7.

## 19.5 MMD² reacts, but not usefully

MMD² is not saturated — it spans 0.1639–0.3123, a spread of 72% of its mean — so
unlike C2ST it *can* rank. It gets the coarse question right: random permutations
sit well above transpositions (0.2673 vs 0.1958, Mann–Whitney **p = 3.9e-07**),
and identity has the **8th smallest MMD² of 77 cells**, so correct ordering is
near-minimal distance.

But it fails the question that matters: within the 66 transpositions it does not
track damage (ρ = −0.146, **p = 0.20**). It registers *that* the input changed,
not *whether the change hurt*. A practitioner watching MMD² would see it rise and
have no way to tell a harmful transposition from a harmless one.

## 19.6 The load-bearing caveat

**The identity cell is itself not significant: macro-F1 0.2599, p = 0.1139.**

The reference condition — correct lead order, corrected encoder — does not clear
p<0.05 in this protocol. Mean bootstrap 95% CI width per cell is 0.0793, roughly
30% of the point estimate. This is a coarse instrument being asked a fine
question, and it bounds every claim above:

* the "single transpositions are undetected" result is partly a statement about
  n=438 power, not only about the diagnostic;
* the two cells that beat identity's CI are 2 of 66 tests with no multiplicity
  correction — at α=0.05 one expects ~3 by chance, so **they should not be
  interpreted as real improvements**;
* the 20/66 transpositions with p<0.05 are likewise uncorrected.

Consistent with Part 18: cross-domain territory decoding from the latent is weak
in absolute terms (0.2786, p=0.0435 under pipeline A). The number here (0.2599,
p=0.1139) is a *different protocol* — probe fit on source with source-fit scaler,
no calibration — and must not be quoted as a restatement of the Part 18 figure.

## 19.7 What changes in the report

* **The methods contribution stands, narrowed.** Claim: *C2ST saturates at
  AUROC≈1.0 in sim-to-real ECG and is therefore constitutionally unable to detect
  a lead-order corruption, while transfer detects gross corruption. Neither
  resolves single-channel transpositions at n=438.* That is defensible on this
  table and is more useful than the version we set out to show.
* **Do not claim transfer is a lead-order detector.** It is a corruption detector.
  39/66 transpositions beat the correct order.
* **Re-word the audit's corroboration.** The lead-order fix is justified by the
  `aVL = (I − III)/2` identity check. The transfer improvement should be reported
  as *consistent with* the fix, not as independent confirmation of it.
* **Recommend MMD² over C2ST for gap monitoring, with the caveat** that it flags
  change without grading harm.
"""


def main() -> int:
    txt = FP.read_text(encoding="utf-8")
    if "# PART 19" in txt:
        print("Part 19 already present -- not appending")
        return 1
    n0 = len(txt.splitlines())
    FP.write_text(txt.rstrip("\n") + PART, encoding="utf-8")
    n1 = len(FP.read_text(encoding="utf-8").splitlines())
    print(f"{FP.name}: {n0} -> {n1} lines (+{n1-n0})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
