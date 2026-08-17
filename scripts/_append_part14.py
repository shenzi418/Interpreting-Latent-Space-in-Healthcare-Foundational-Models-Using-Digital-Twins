"""Append Part 14 to EXECUTION_LOG: Stage 4.1 feasibility, measured not assumed."""
from pathlib import Path

TEXT = '''
---

# PART 14 — Stage 4.1 (MI-stage control) is underpowered on the exported split

Stage 4.1 of the rerun plan calls the acute-vs-chronic MI stratification "the
strongest remaining candidate for the residual gap and a genuine scientific result
either way". It is the last unrun free measurement, so I costed it before queueing
it. It does not survive the costing, and the reason is worth recording because the
plan's own numbers are what make it look feasible.

## 14.1 The counts the plan quotes are whole-database

`reports/2026-08-10_repo_audit_and_rerun_plan.md` §4 Stage 4.1 quotes
"Stadium III n=980, II-III n=943, II n=88 vs n=166 acute". Those reproduce exactly
against `ptbxl_database.csv` over **all 21,799 records**:

| `infarction_stadium1` | whole DB | exported test split | MI rows in split | primary 4c subset |
|---|---|---|---|---|
| Stadium I | 166 | 14 | — | 9 |
| Stadium II | 88 | 7 | — | 5 |
| **acute (I + II)** | **254** | **21** | **21** | **14** |
| Stadium II-III | 943 | 107 | — | 85 |
| Stadium III | 980 | 100 | — | 80 |
| **chronic (III + II-III)** | **1923** | **207** | **201** | **165** |
| unknown | 3430 | 332 | 328 | 259 |
| NaN (no MI annotation) | 16187 | 1638 | — | — |

The exported latents are the **test split only** (fold 10 of PTB-XL's 10-fold
stratification, n=2198). Acute records are spread evenly across folds — 19 to 35
per fold — so the split holds ~1/10th of them: **21 acute rows, of which 14 carry
a primary territory label.**

## 14.2 What that permits, and what it does not

A stratified transfer comparison needs an AUROC per stratum. §13.4 just measured
what the *chronic-dominated* 438-row evaluation supports: a 95% CI of width 0.11
on a macro-AUROC difference. The acute stratum is 14 rows against those 438 — an
interval on it would be wider than the entire range the metric can take, and a
per-territory breakdown (the form the question actually needs, since territory is
what the probe predicts) puts 3-5 rows in most cells.

So the honest statement is: **Stage 4.1 cannot be run as specified on the exported
artifacts.** Not "the result was null" — the measurement is not defined at this
sample size, which is a different and weaker thing to have to say.

Three ways it could become answerable, none free:

1. **Export latents for the remaining folds.** ~254 acute records total, of which
   ~170 would carry territory labels. That is a real experiment: 9 more export
   runs on an existing checkpoint, no retraining. It is the only route that makes
   the plan's own framing achievable.
2. **Drop territory, ask the coarser question.** Acute-vs-chronic as a *binary
   probe target* on 21 vs 201 rows is still thin but at least defined, and it
   tests something adjacent: whether the latent separates MI stage at all.
3. **Report the imbalance as the finding.** MedalCare simulates acute ischemia;
   the PTB-XL rows the thesis evaluates on are 90%+ chronic where annotated, and
   two-thirds carry no stage annotation at all. That mismatch is a documented
   property of the sim-to-real setup and belongs in the limitations section
   whether or not the probe is ever run.

## 14.3 Recommendation

**(3) now, (1) only if time permits after the decisive spatial54 run.** Option 3
costs nothing and strengthens the thesis's limitations chapter with a measured
domain-composition fact rather than a hedge. Option 1 is the scientifically
interesting one but it is a new export campaign on the critical path of a
deadline, and §16 Rank 1 has a stronger claim on that time — it settles a question
the report currently leads with, whereas Stage 4.1 opens a new one.

Recording this rather than quietly skipping the stage: "we did not run it" and
"it is not runnable on these artifacts" are different statements, and only the
second one is true here.

**Method note for the thesis.** The generalisable error is quoting a population
count to justify an experiment that will run on a *split*. Whole-DB n=254 reads as
adequate; split n=21 does not; nothing in the plan's wording flagged the
difference. Worth a standing check: before costing any stratified analysis,
compute the stratum count **on the rows the analysis will actually see**.
'''


def main() -> int:
    p = Path("reports/EXECUTION_LOG_2026-08-10.md")
    t = p.read_text(encoding="utf-8")
    if "# PART 14 —" in t:
        print("Part 14 already present")
        return 0
    p.write_text(t.rstrip("\n") + "\n" + TEXT, encoding="utf-8")
    print(f"appended Part 14 ({len(t.splitlines())} -> "
          f"{len(p.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
