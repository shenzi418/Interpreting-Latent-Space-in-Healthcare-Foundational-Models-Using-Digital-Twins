"""Append Part 13 to EXECUTION_LOG: tier1 pass 2 + the Rank 5 verdict.

Kept as a file rather than a heredoc: the body contains both quote styles and
markdown pipes, which a shell heredoc mangles (Part 12 process note).
"""
from pathlib import Path

TEXT = '''
---

# PART 13 — tier1 pass 2, and §16 Rank 5 answered against a control that ties

Two things landed while the poolscaler driver ran: the tier1 replication over all
five corrected encoders (rc=0, 13.8 min, 08:37:58), and the leak-free rerun of
`analysis/transfer_control.py`. They belong together because the second one
changes how the first one should be read.

## 13.1 tier1 pass 2 — five encoders, C2ST 1.0000 on every one

`reports/stage3_logs/post_tier1.log`, full table in
`outputs/tier1_eval_exp8/cross_config_table.md`.

| config | K | MMD_med | C2ST | kNN-mix | LR M→P | LR P→M | phi_R2c | rho_AUC | anat in-dom 4c | anat cross 4c |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1024 | 0.2153 | **1.0000** | 0.0049 | 0.827 | 0.616 | 0.570 | 0.955 | 0.640 | 0.260 |
| ccmmd    | 1024 | 0.2002 | **1.0000** | 0.0051 | 0.806 | 0.599 | 0.575 | 0.955 | 0.652 | 0.266 |
| dual     | 1024 | 0.2092 | **1.0000** | 0.0050 | 0.601 | 0.462 | 0.581 | 0.956 | 0.653 | 0.270 |
| globalz  | 1024 | 0.2420 | **1.0000** | 0.0015 | 0.701 | 0.646 | 0.599 | 0.942 | 0.643 | 0.296 |
| K64      | 64   | 0.4484 | **1.0000** | 0.0018 | 0.893 | 0.974 | 0.385 | 0.871 | 0.523 | 0.193 |

Three readings worth keeping:

* **C2ST is 1.0000 to four decimals on all five**, including the two encoders
  built specifically to reduce the gap (ccmmd) and to remove the normalisation
  mismatch (globalz). That is checkpoint twelve and thirteen for §13.6
  over-determination. ccMMD does lower MMD (0.2002 vs 0.2153) and does not move
  C2ST at all — the same dissociation §13 documents, now on corrected data.
* **The shared-head architecture claim survives the audit.** LR M→P is 0.827
  (baseline, shared-head) vs 0.601 (dual-head) — the effect `experiments.md`
  records at ≈0.76 vs ≈0.59 pre-fix. Bigger after the lead fix, same direction.
* **K64 is the outlier in both directions.** Best class-structure numbers
  (LR M→P 0.893, P→M 0.974, KMeans 0.827) and worst mechanism numbers
  (phi_R2c 0.385 vs ≈0.58, rho_AUC 0.871 vs ≈0.955, anat cross-4c 0.193). A 64-d
  bottleneck keeps what separates the three shared classes and discards what
  encodes θ. That is §16 Rank 3's capacity threshold showing up in a second,
  independent measurement — and it is evidence the two are genuinely different
  quantities rather than one quantity measured twice.

`best_C=0.01` is selected by tier1's own CV for Z[1024d] on all four 1024-d
configs, and `best_C=1` for K64. Noted because it is what flagged the confound in
§13.2 below.

## 13.2 §16 Rank 5 — the fixed comparison, and a capacity confound caught first

The leak-free script (Part 12) ran clean: the missingness guard reports 0.5364 on
both feature arms, comfortably under the 0.55 abort threshold, versus 0.847/0.863
for the leaked version. Shuffle nulls now sit near 0.5 where before they sat
below it. The instrument is sound.

**But the first run of it was still wrong**, and in a way that would have produced
a headline. Every arm ran at scikit-learn's default `C=1.0`, while Z is 1024-d
over 5347 rows and the controls are 54-d and 6-d. §13.1's own CV selects
`best_C=0.01` for Z[1024d] on these exact rows for this exact task — two orders
of magnitude off the default. So the untuned run handicapped the latent arm
specifically, and "control wins" and "the latent was under-regularised" are
indistinguishable from that output.

`probe_auc` now takes a `--Cs` grid and tunes per arm by 5-fold CV on the
**MedalCare rows only** (target rows never enter selection, so no leakage is
added). Both numbers are kept:

| representation | dim | M→P untuned | M→P tuned | selected C (per class) | P→P ceiling | frac |
|---|---|---|---|---|---|---|
| Z         | 1024 | 0.5617 | **0.5674** | 0.01, 0.01, 0.01, 0.001 | 0.804 | 0.706 |
| global6   | 6    | 0.5459 | 0.5465 | 0.1, 10, 0.01, 0.01 | 0.559 | 0.978 |
| spatial54 | 54   | 0.6170 | **0.6123** | 1, 1, 0.1, 10 | 0.784 | 0.781 |

Tuning confirmed the diagnosis and did **not** rescue the latent: Z picks
C=0.01/0.001 on every class, exactly as predicted, and gains +0.0057. The delta
moves from −0.0553 to −0.0449. Capacity explains about a fifth of the gap; the
rest is real.

**Verdict: TIE, and it should be reported as a tie rather than as a reversal.**
−0.0449 lands inside the ±0.05 band declared in the script before it ran, but it
is 0.001 inside it — one class-level perturbation from flipping. The defensible
statement is the second of the three pre-declared outcomes:

> On cross-domain MI-territory transfer, a 54-column hand-crafted spatial feature
> set matches the 1024-d foundation-model latent (0.612 vs 0.567 macro AUROC,
> both far above their shuffle nulls of 0.485 / 0.525). The transfer number
> measures the task, not the representation.

Per-class, the control's advantage is concentrated where anatomy is most
lead-local — Inferior 0.688 vs 0.552 and Anteroseptal 0.645 vs 0.515 — while the
latent leads on Inferolateral (0.599 vs 0.582) and Anterolateral (0.604 vs 0.535),
the two rarest PTB-XL territories (32 and 42 rows). That pattern is what the
textbook predicts of per-lead Q/ST/T columns, and §15.1's instrument check
already established this control can represent territory.

**What this does to the report.** §6's Rank 5 framing — "transfer is real, lead
the thesis with it" — survives on the *reality* of transfer (both arms beat their
nulls decisively) but not on its attribution to the latent. Any sentence of the
form "the foundation model's representation carries transferable territory
information" now needs "…no better than 54 hand-crafted per-lead measurements"
attached, or it overclaims. §15 already made this correction once for §14.6; this
is the same correction arriving at the report's headline claim.

Note the two effects also compose in the *unfavourable* direction for the first
time today: §14 found the scaler flattering the control, §15 found the control's
definition flattering the latent, and here — with both fixed and capacity
controlled — the control still ties. That is the strongest form of the result,
because it is the one that survives corrections made in both directions.

## 13.3 What is still owed

The spatial54 Track-3 arm (§16 Rank 1) remains the decisive run and is unchanged
by this: it measures the same comparison on macro-F1 with permutation p-values
and 8 cross-domain blocks, rather than one-vs-rest AUROC on 438 PTB-XL rows. If
it also ties, the two agree by different statistics on different splits and the
claim is settled. If it splits from this result, the difference is informative on
its own — and Rank 5 collapsing into Rank 1 (Part 12 §12.5) is exactly why both
were kept.

Artifacts: `outputs/analysis/domain_signal/transfer_control_exp8_leadfix_baseline.json`
(untuned, superseded — kept for the comparison) and
`..._tunedC.json` (the reportable one; carries `selected_C` per class and the
`C_grid` used).
'''


def main() -> int:
    p = Path("reports/EXECUTION_LOG_2026-08-10.md")
    t = p.read_text(encoding="utf-8")
    if "# PART 13 —" in t:
        print("Part 13 already present; nothing appended")
        return 0
    p.write_text(t.rstrip("\n") + "\n" + TEXT, encoding="utf-8")
    print(f"appended Part 13 -> {p} ({len(t.splitlines())} -> "
          f"{len((t + TEXT).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
