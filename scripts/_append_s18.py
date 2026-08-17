"""Append S18 (post-plan: where to break through, 20 days out) to the report."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FP = REPO / "reports" / "2026-08-11_breakthrough_analysis.md"

SEC = """

---

## 18. The plan is complete — where the breakthrough actually lands

*Written 2026-08-11, after Stage 4.2 closed the 2026-08-10 rerun plan. 20 days to
the 2026-08-31 thesis deadline.*

### 18.1 Status

| stage | outcome |
|---|---|
| 0–3 | done — code fixes, `exp8_*` retraining, free re-measurements |
| 4.1 MI-stage control | **not runnable** on the exported split (fold 10 holds 21 acute records, 14 territory-labelled, vs 201 chronic). Documented as a measured limitation, not skipped. Part 14. |
| 4.2 lead-permutation sweep | done. Part 19, §17. |
| 5 repo hygiene | done — 6 defects, 24 assertions in `scripts/_verify_stage5.py`. |

### 18.2 The three results that survived, in the order they should be written

1. **The gap is over-determined** (§13.6). Marginals alone and dependence alone
   *each* identify the domain at AUROC ≈ 1.0, across thirteen checkpoints. This is
   why alignment is unreachable — not a tuning failure, a structural one.
2. **The sign reversal** (Part 18). The latent beats a 54-feature hand-crafted
   spatial control in-domain by +0.14 macro-F1 in all five configs, and loses to it
   cross-domain in all five. The foundation model's advantage over classical ECG
   features is real, large, and **entirely in-domain**.
3. **The diagnostics are the wrong instruments** (Part 19). C2ST saturated at 1.0
   moves by 1e-5 across corruptions spanning 2× in transfer; MMD² flags change
   without grading harm.

Together these are a coherent negative-result thesis with a methods contribution
attached. That is a defensible submission. **The writing, not another experiment,
is now the critical path.**

### 18.3 The one experiment worth the remaining time

The sharpest reviewer objection to result 2 is a capacity objection: *"you compared
a 1024-d latent and a 64-d bottleneck against a 54-d control — maybe the latent
transfers fine at some intermediate capacity you never tried."* Part 18 §18.5
partially answers it (K64 transfers *worse*, and `C` was tuned per arm), but only
at two points on the curve.

**Recommended: a post-hoc PCA-K sweep of cross-domain territory decoding against
the fixed control baseline of 0.3442.** K ∈ {2, 4, 8, 16, 32, **54**, 64, 128, 256,
512, 1024}, on existing `exp8_leadfix_*` latents, no retraining — the machinery
already exists in `analysis/dim_scan.py`. K=54 is the point of the design: it
matches the control's dimensionality exactly, making the comparison a clean
capacity control rather than an argument about it.

Why this and not a trained bottleneck at K=128/256: it costs hours instead of days,
and it is the *screening* experiment. If some K beats 0.3442, that is a positive
result worth spending compute to confirm with a trained bottleneck. If no K beats
it — the outcome I expect, given both endpoints already lose — then the capacity
objection is closed on eleven points instead of two, and result 2 hardens
considerably for one afternoon of compute.

Pre-declare before running, per the Part 16 discipline: the endpoint is
cross-domain `territory_4c` macro-F1 against the control's 0.3442; the reading is
that the capacity objection survives only if some K beats it **and** clears its own
permutation null.

### 18.4 What not to do

* **No new alignment-via-training method.** Thirteen checkpoints, two independent
  sufficient causes, C2ST 1.0000 after every correction — and `ccmmd`, trained
  explicitly to close the gap, lowers MMD to the lowest of the five while moving
  C2ST not at all.
* **Do not open Stage 4.1** (nine more PTB-XL fold exports). It is the highest-value
  *unopened* question (§16 Rank 7), and that is exactly why it is wrong to start now
  — it would open a new axis 20 days out. The 12:1 chronic:acute ratio belongs in
  the limitations chapter as a measured fact.
* **Do not re-litigate the lead fix.** §17.4 narrowed what the transfer movement
  proves; the physics identity still settles it.
"""


def main() -> int:
    txt = FP.read_text(encoding="utf-8")
    if "## 18. The plan is complete" in txt:
        print("S18 already present -- not appending")
        return 1
    n0 = len(txt.splitlines())
    FP.write_text(txt.rstrip("\n") + SEC, encoding="utf-8")
    n1 = len(FP.read_text(encoding="utf-8").splitlines())
    print(f"{FP.name}: {n0} -> {n1} lines (+{n1-n0})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
