"""Append Part 18: the spatial54 verdict, read against the Part 16 pre-registration."""
from pathlib import Path

TEXT = '''
---

# PART 18 — The decisive run: the hand-crafted control beats the latent cross-domain

`run_spatial54_arm.py`, all five `exp8_leadfix_*` configs against a 54-column
hand-crafted spatial ECG control (48 per-lead + the original 6). rc=0, 100.8 min,
finished 11:35:59. This is the run Part 16 pre-registered at 09:15, before the
output file existed.

**The pre-registered reading is row 4 of the §16.2 table, and it fires cleanly.**

## 18.1 Preconditions — all four checked before any verdict was read

1. **Determinism guard PASSED** (§16.3 rule 3, the hard precondition):
   `55 latent blocks compared; macro_f1 identical, largest p deviation 0.00 MC sigma`.
   The Z arm is bit-identical to the reference; only the control changed. Without
   this the file would have been unquotable in full, not in part.
2. **The control is encoder-invariant.** Its macro-F1 is identical to 6 decimal
   places across all five configs (0.344187 on `cross_domain_4c`, 0.626825 on
   `cross_domain_2c`, 0.503683 in-domain). It must be — the control never sees the
   encoder — and a control that *varied* across configs would have indicated
   cross-contamination. It doesn't.
3. **Z and control are scored on the same rows.** n=438 cross-domain, n=1200
   in-domain, with identical `n_per_class_truth` on both arms
   (168/42/196/32 cross; 400/300/400/100 in-domain). No row-subset asymmetry of
   the kind Part 12 found.
4. **The capacity confound is controlled** (the Part 13 §13.3 lesson). `C` is
   tuned per arm by CV **on source rows only**: Z selects C=0.01 (source CV
   macro-F1 0.656–0.670), the control selects C=0.1 (0.518), K64 selects C=1
   (0.546). Neither arm is handicapped by a shared default.

## 18.2 The primary endpoint

`pipelineA / exp8_leadfix_baseline / cross_domain_4c` — the single block and
single config fixed in §16.1.

| arm | macro-F1 | permutation p |
|---|---|---|
| Z (1024-d latent) | 0.2786 | 0.0435 |
| control (54 spatial features) | **0.3442** | **0.0001** |

dC − dZ = **+0.0656**, more than double the pre-declared ±0.03 band, with both
arms significant. Pre-registered reading, quoted verbatim from §16.2:

> **§14.6 measured instrumentation; state it plainly and prominently.**

This is also the outcome §16.4 recorded in advance as the expected one. That
paragraph was written so it could be wrong; it wasn't.

## 18.3 It replicates 5/5 — the control wins every config

§16.3 rule 2 requires all five configs, not a favourable subset. All five deliver:

| config | Z f1 | Z p | control f1 | control p | winner |
|---|---|---|---|---|---|
| baseline | 0.2786 | 0.0435 | 0.3442 | 0.0001 | control |
| ccmmd | 0.2733 | 0.0740 | 0.3442 | 0.0001 | control (only control significant) |
| dual | 0.3038 | 0.0013 | 0.3442 | 0.0001 | control |
| globalz | 0.3046 | 0.0016 | 0.3442 | 0.0001 | control |
| K64 | 0.2564 | 0.0809 | 0.3442 | 0.0001 | control (only control significant) |

Note also that **baseline's own significance is marginal** (p=0.0435 under the
`target` scaler; the same cell was p=0.0098 under `target_pool` in Part 17 with
the point estimate essentially unmoved, 0.2786 → 0.2797). A point estimate that
stable with a p that mobile means the permutation null is wide and baseline sits
near its edge. `dual` (p=0.0013) and `globalz` (p=0.0016) are the robust members.

## 18.4 The finding this run actually produced: a sign reversal, not a ranking

The interesting result is not "the control wins". It is that **the two arms swap
places between in-domain and cross-domain, in all five configs**:

| config | in-domain Z | in-domain ctl | Δ | cross Z | cross ctl | Δ |
|---|---|---|---|---|---|---|
| baseline | 0.6404 | 0.5037 | **+0.1367** | 0.2786 | 0.3442 | **−0.0656** |
| ccmmd | 0.6520 | 0.5037 | **+0.1483** | 0.2733 | 0.3442 | **−0.0709** |
| dual | 0.6529 | 0.5037 | **+0.1493** | 0.3038 | 0.3442 | **−0.0404** |
| globalz | 0.6432 | 0.5037 | **+0.1395** | 0.3046 | 0.3442 | **−0.0396** |
| K64 | 0.5234 | 0.5037 | +0.0198 | 0.2564 | 0.3442 | **−0.0878** |

Pipeline B's in-domain blocks say the same thing more loudly (Z +0.148 to +0.227).

Stated as retention of each arm's own in-domain performance:

* the latent retains **42–49%** of its in-domain territory decoding under the
  sim-to-real shift;
* the control retains **68%** (0.5037 → 0.3442).

So the latent carries *more* θ-territory information than 54 hand-crafted spatial
features — by a wide margin, +0.14 macro-F1 in-domain and +0.14 on source CV — and
**that surplus is the part that does not survive the domain shift.** The
information the latent has in excess of the hand-crafted features is
synthetic-specific.

That is a sharper statement of the sim-to-real problem than anything else in this
project, and it is the honest headline: the foundation-model latent's advantage
over classical spatial ECG features is real, large, and **entirely in-domain**.

## 18.5 Why this is not just "1024 dimensions overfit"

The obvious deflation is that a 1024-d probe overfits the source domain and a
54-d probe cannot. Three things argue against that being the whole story:

* `C` was tuned per arm on source CV, and the latent arm selected the *heaviest*
  regularisation available to it (C=0.01 vs the control's C=0.1). The probe is
  already being pushed toward the low-capacity regime and still fails to transfer.
* **K64 is 64-dimensional — comparable to the control's 54 — and transfers
  *worse* than the control, not better** (0.2564, p=0.0809; and on the 3-class
  block p=0.9556). Cutting the latent to control-comparable width does not recover
  portability; it destroys it. Dimensionality alone therefore does not predict the
  ordering.
* The control's advantage is in *retention*, not raw score. It is the weaker
  representation in-domain and still wins cross-domain.

What remains genuinely open is whether a differently-shaped probe (not merely a
smaller one) could extract portable territory structure from the 1024-d latent.
§16 Rank 3's K=128/256 sweep is the direct test and is now the best-motivated
follow-up in the queue.

## 18.6 Pipeline B — the cells that favour Z, and why they do not rescue the claim

Pipeline B (φ → territory via a CV-selected calibrator) is exploratory by §16.3
rule 4. It shows Z winning `cross_calibrator_4c` on ccmmd (+0.0358), dual
(+0.0437) and globalz (+0.0440), with baseline indistinguishable (+0.0182).

Quoting those cells as a counterweight would be wrong, and the reason is visible
in the numbers: **the control drops from 0.3442 (pipeline A) to 0.2725 (pipeline
B calibrator), while Z rises from 0.2786 to 0.2907.** The Z-favourable margin is
~85% control-degradation, not Z improvement. Pipeline B's calibrator is a second
fitted object that costs the control far more than it costs the latent — exactly
the failure mode rule 4 was written to anticipate. The comparison degrades the
reference; it does not raise the latent.

The control also wins **every** 2-class block in both pipelines (e.g. pipeline A
baseline 0.6268 vs 0.5919), so there is no direction in which the primary reading
reverses.

## 18.7 The one caveat that goes with the control's number

Three of the 54 features are global intervals with substantial imputation:
`QRS_duration_ms` (22.0% train / 33.6% test), `QT_interval_ms` (20.4 / 30.7),
`P_duration_ms` (19.9 / 26.1). The remaining 51 sit at ≤1.8%, mean 2.1% / 2.6%.

This does **not** reopen the Part 12 defect — Part 15.3 scored the finiteness
indicator alone and got 0.5364, below the 0.55 gate, so missingness does not carry
the label. And the direction is conservative: the imputation handicaps the
*control*, which wins anyway. A cleaner interval extractor could only widen the
gap the control already holds.

## 18.8 What changes in the report

* **§16 Rank 1 is resolved.** The attribution claim it was testing does not
  survive: cross-domain territory transfer is real, and a 54-column hand-crafted
  spatial control does it *better* than the latent, 5/5.
* **§16 Rank 5 must be restated.** Its "Z 0.280, p=0.0098 vs NK2 0.207" comparison
  used the 6-feature global control. Against the 54-feature spatial control the
  sign flips. The earlier number was not wrong, it was under-powered as a control
  — which is precisely what §13.4's P(control wins)=0.952 predicted.
* **The over-determination result (§16 Rank 2) becomes the thesis headline**, as
  §16.4 anticipated, with §18.4's sign reversal as the second result.
'''


def main() -> int:
    p = Path("reports/EXECUTION_LOG_2026-08-10.md")
    t = p.read_text(encoding="utf-8")
    if "# PART 18 —" in t:
        print("Part 18 already present")
        return 0
    p.write_text(t.rstrip("\n") + "\n" + TEXT, encoding="utf-8")
    print(f"appended Part 18 ({len(t.splitlines())} -> "
          f"{len(p.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
