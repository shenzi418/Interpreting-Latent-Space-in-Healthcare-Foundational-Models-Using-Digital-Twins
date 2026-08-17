"""Insert §14.9 (tier1 pass 2) into the breakthrough report, before §15.

Kept as a file rather than a heredoc: the body contains both quote styles and
markdown pipes, which a shell heredoc mangles (EXECUTION_LOG Part 12 process note).
"""
from pathlib import Path

ANCHOR = ("## 15. The hand-crafted control cannot localise an infarct "
          "— the comparison is instrumented unfairly")

SECTION = '''### 14.9 tier1 pass 2 — the full evaluation suite on five corrected encoders

`analysis/tier1_evaluation.py` re-run over all five `exp8_leadfix_*` configs
(rc=0, 13.8 min). This is the broadest single replication in the project: 21
metrics × 5 encoders, all post-audit. Full table in
`outputs/tier1_eval_exp8/cross_config_table.md`; the load-bearing columns:

| config | K | MMD_med | C2ST | kNN-mix | LR M→P | LR P→M | phi_R2c | rho_AUC | anat in-dom 4c | anat cross 4c |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1024 | 0.2153 | **1.0000** | 0.0049 | 0.827 | 0.616 | 0.570 | 0.955 | 0.640 | 0.260 |
| ccmmd    | 1024 | 0.2002 | **1.0000** | 0.0051 | 0.806 | 0.599 | 0.575 | 0.955 | 0.652 | 0.266 |
| dual     | 1024 | 0.2092 | **1.0000** | 0.0050 | 0.601 | 0.462 | 0.581 | 0.956 | 0.653 | 0.270 |
| globalz  | 1024 | 0.2420 | **1.0000** | 0.0015 | 0.701 | 0.646 | 0.599 | 0.942 | 0.643 | 0.296 |
| K64      | 64   | 0.4484 | **1.0000** | 0.0018 | 0.893 | 0.974 | 0.385 | 0.871 | 0.523 | 0.193 |

**C2ST = 1.0000 to four decimals on every one.** Checkpoints twelve and thirteen
for §13.6 over-determination, and the two most informative ones yet: `ccmmd` was
trained specifically to close the gap and `globalz` removes the normalisation
mismatch the audit found. ccMMD *does* lower MMD (0.2002 vs 0.2153, the lowest of
the five) and moves C2ST not at all — the MMD/C2ST dissociation §13 documents,
now reproduced on corrected data with a corrected label space.

**The architecture claim survives the audit.** LR M→P is 0.827 for the shared-head
baseline vs 0.601 dual-head. `experiments.md` records this pre-fix at ≈0.76 vs
≈0.59; the lead fix widens the gap without changing its direction. The Exp-7
headline — *shared-head architecture, not the 3-class relabeling, drives
cross-domain transfer* — is one of the few pre-`exp8` claims that comes through
the audit intact and larger.

**K64 splits the two capability axes cleanly.** It is simultaneously best on
class structure (LR M→P 0.893, P→M 0.974, KMeans 0.827 — all five-way maxima) and
worst on mechanism (phi_R2c 0.385 vs ≈0.58; rho_AUC 0.871 vs ≈0.955; cross-domain
4c 0.193 vs ≈0.27). A 64-d bottleneck keeps what separates NORM/MI/CD and discards
what encodes θ.

That dissociation matters beyond the capacity question. Throughout this report
"class structure" and "θ decodability" have been reported as separate numbers on
the assumption that they measure different things; K64 is the first encoder where
they move in **opposite** directions, which is the strongest available evidence
that the assumption holds rather than that one quantity is being measured twice.
It also sharpens §16 Rank 3: the transferability floor sits in (64, 1024], and
K=128/256 is the informative probe.

One methodological note carried forward: tier1's internal CV selects `best_C=0.01`
for Z[1024d] on these rows and `best_C=1` for K64. That is what exposed the
capacity confound in the Rank 5 control comparison (EXECUTION_LOG Part 13 §13.2) —
a 1024-d arm run at scikit-learn's default `C=1.0` is under-regularised by two
orders of magnitude, and any comparison against a low-dimensional control that
does not tune C is measuring capacity, not representation.

'''


def main() -> int:
    p = Path("reports/2026-08-11_breakthrough_analysis.md")
    t = p.read_text(encoding="utf-8")
    if "### 14.9 tier1 pass 2" in t:
        print("14.9 already present; nothing inserted")
        return 0
    if ANCHOR not in t:
        print("ANCHOR NOT FOUND -- nothing written")
        return 1
    p.write_text(t.replace(ANCHOR, SECTION + ANCHOR, 1), encoding="utf-8")
    print(f"inserted 14.9 before section 15 ({len(t.splitlines())} -> "
          f"{len(p.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
