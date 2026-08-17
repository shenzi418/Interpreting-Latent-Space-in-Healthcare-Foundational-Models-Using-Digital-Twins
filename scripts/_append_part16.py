"""Append Part 16 — pre-registration for the decisive run, written before the numbers."""
from pathlib import Path

TEXT = '''
---

# PART 16 — Pre-registration for the decisive run (written before it produced a number)

The spatial54 arm is chained and will run unattended. Its docstring already fixes
three outcomes in advance, which is the right instinct, but it fixes them at the
level of "latent wins / control wins / neither is significant" — and Part 13
demonstrated three times over that the interesting failure modes live *below* that
resolution. So the decision rules go on record now, at 09:15, while the output
file does not yet exist.

This is not ceremony. §13.2's "TIE" and §13.4's correction to it differ only in
what was considered a permissible reading of the same numbers, and the reason the
first version was wrong is that the reading rule was chosen *after* seeing the
point estimate. Writing the rule first is the only structural defence against
that, and it costs nothing while the GPU is busy.

## 16.1 What the run reports, and which number is the endpoint

`run_spatial54_arm.verdict()` prints a latent-vs-control row per cross-domain
block over three JSONs (`cross_domain`, `pipelineA`, `pipelineB`) × 5 configs.
That is ~55 comparisons. **Fifty-five comparisons is not fifty-five results**, and
the temptation to quote whichever rows favour a preferred story is exactly what a
pre-registration is for.

**The primary endpoint is `pipelineA / exp8_leadfix_baseline / cross_domain_4c`**
— one block, one config, chosen because it is the block §16 Rank 5 already
reports a number for (Z 0.280, p=0.0098 against NK2 0.207 under global6) and
because `baseline` is the encoder every other claim in the report is stated on.
Everything else is secondary and will be labelled as such.

## 16.2 Decision rules, fixed now

Let **dZ** = Z macro-F1 and **dC** = control macro-F1 on the primary endpoint,
with permutation p-values pZ and pC (n_perm=10000).

| condition | reading | what goes in the thesis |
|---|---|---|
| pZ < 0.05, pC >= 0.05 | latent transfers, control does not | §16 Rank 5's attribution claim is **restored**; the 438-row AUROC arm was underpowered, not wrong |
| pZ < 0.05, pC < 0.05, dZ − dC > +0.03 | both transfer, latent better | attribution claim restored **with the control named**; quote both |
| pZ < 0.05, pC < 0.05, \\|dZ − dC\\| <= 0.03 | both transfer, indistinguishable | §16 Rank 5 stands as written: transfer real, attribution not established |
| pZ < 0.05, pC < 0.05, dC − dZ > +0.03 | control better | §14.6 measured instrumentation; state it plainly and prominently |
| pZ >= 0.05, pC < 0.05 | control transfers, latent does not | the strongest negative available; leads the limitations chapter |
| both >= 0.05 | neither transfers | cross-domain territory decoding is below the noise floor for both representations — itself a result about the sim-to-real gap |

The ±0.03 band is narrower than §13.2's ±0.05 because macro-F1 over 8 permutation
blocks at n_perm=10000 is a tighter instrument than one-vs-rest AUROC on 438 rows.
**It is declared here, before any number exists, and will not be revised after.**

## 16.3 Constraints on how the result may be quoted

1. **No cell may be quoted without its permutation p.** A macro-F1 difference
   with both arms non-significant is not a comparison, it is two noise draws.
2. **"Replicates across configs" requires all five**, not the favourable subset.
   The report has used 4-of-4 and 5-of-5 replication as a load-bearing claim
   (§14.7, §14.9); a 3-of-5 result is a *split*, and must be called one.
3. **The determinism guard is a precondition, not a result.** If
   `check_latent_unchanged()` fails, no verdict may be quoted from the file at
   all — not even a partial one. Part 15.4 verified the guard's own baseline
   precisely so this cannot be argued around after the fact.
4. **Pipeline B is exploratory.** It selects a calibrator by CV, which is a second
   fitted object and a second opportunity for the comparison to reflect fitting
   rather than representation. It may support the primary endpoint; it may not
   substitute for it.
5. **The prior is against the latent.** §13.4 gives P(control wins) = 0.952 on the
   AUROC arm. A latent win on the primary endpoint is therefore a *reversal* of
   the currently best-supported reading, and reversals need their disagreement
   explained — not just reported. If Rank 1 favours the latent, the honest write-up
   must say why two instruments disagree, and §13.4's power argument is the first
   hypothesis to test, not a convenient explanation to reach for.

## 16.4 The outcome I expect, recorded so it can be wrong

Given §13.4 (P(control wins)=0.952) and §15.1 (the control reaches in-domain
AUROC 0.909 from Q/R columns alone), I expect the **control to match or beat the
latent** on the primary endpoint, with both significant. If that is what lands,
the thesis's headline becomes the over-determination result (§16 Rank 2), and the
territory-transfer result survives as "transfer is real under 6× prevalence shift,
and a 54-column hand-crafted control does it at least as well" — which is a
weaker claim about foundation models and a *stronger* claim about the sim-to-real
setup, because it says the transferable signal is spatially localised and
measurable rather than distributed and opaque.

Recording the expectation makes it falsifiable. If the latent wins cleanly, this
paragraph is the evidence that the result was not fitted to a preferred narrative.
'''


def main() -> int:
    p = Path("reports/EXECUTION_LOG_2026-08-10.md")
    t = p.read_text(encoding="utf-8")
    if "# PART 16 —" in t:
        print("Part 16 already present")
        return 0
    p.write_text(t.rstrip("\n") + "\n" + TEXT, encoding="utf-8")
    print(f"appended Part 16 ({len(t.splitlines())} -> "
          f"{len(p.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
