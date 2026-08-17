"""Append §13.4 to EXECUTION_LOG: the bootstrap around the latent-vs-control tie."""
from pathlib import Path

TEXT = '''
## 13.4 The tie, bootstrapped — "tie" was a statement about the point estimate

§13.2 reported a delta of −0.0449 against a ±0.05 band declared before the run,
and called it a TIE while flagging that it lands 0.001 inside the boundary. That
flag was right to be there, and it deserved a number rather than a caveat. 438
PTB-XL rows, two territories at n=32 and n=42 — an interval was owed.

`analysis/transfer_control_bootstrap.py` resamples the **evaluation rows only**,
1000 draws. Probes are fit once per arm per class on the MedalCare rows (C tuned
exactly as in §13.2, on source rows only), and each draw rescores those same
fitted objects on a resampled PTB-XL row set. All three arms see the **same**
resampled rows in each draw, so the delta is paired and the shared row-sampling
noise cancels rather than swamping the comparison.

The point estimates reproduce to four decimals — 0.5674 / 0.5465 / 0.6123 —
which is the check that the bootstrap is scoring the same objects §13.2 scored.

| representation | point | boot mean | 95% CI |
|---|---|---|---|
| Z         | 0.5674 | 0.5680 | [0.5273, 0.6077] |
| global6   | 0.5465 | 0.5470 | [0.5016, 0.5884] |
| spatial54 | 0.6123 | 0.6131 | [0.5738, 0.6540] |

| comparison | delta | 95% CI | P(control wins) |
|---|---|---|---|
| Z − global6   | +0.0210 | [−0.0338, +0.0798] | 0.251 |
| Z − spatial54 | −0.0451 | [−0.1003, +0.0064] | **0.952** |

**Both intervals include zero, and both are wider than the ±0.05 band the verdict
was declared against.** The Z − spatial54 interval runs to −0.10 on one side and
barely clears zero on the other. So:

* **"TIE" as written in §13.2 overstates what 438 rows can support.** A tie is a
  claim of *equivalence*, and equivalence requires an interval narrow enough to
  exclude a material difference. This one does not — it is consistent with the
  control beating the latent by a full 0.10 AUROC.
* What the data **does** support: the latent does not outperform the control
  (P(control wins) = 0.952 against spatial54), and it is not distinguishable from
  the 6-feature control at all (P = 0.251, interval straddling zero symmetrically).
* The correct verdict is therefore **"the latent does not beat the hand-crafted
  control, and the direction of the difference is not resolved at this sample
  size"** — weaker than "tie" in what it asserts, and stronger in what it rules
  out, because the one thing the interval *does* exclude is the latent winning by
  any margin worth quoting.

This does not rescue the attribution claim, and it is not meant to: every reading
above is bad for "the representation is what carries territory information", which
is the claim §16 Rank 5 retired. The change is to how the *comparison* is stated,
not to which side it favours.

**Why this matters beyond the wording.** The decisive run (spatial54 Track-3,
§16 Rank 1) evaluates the same comparison on 8 cross-domain blocks with
permutation p-values instead of one-vs-rest AUROC on 438 rows. §13.2 predicted
agreement would settle the question. This bootstrap shows the AUROC arm is too
underpowered to settle anything on its own — so Rank 1 is not a confirmation
step, it is the measurement, and its permutation p-values are the statistic that
should be quoted in the thesis.

Artifact:
`outputs/analysis/domain_signal/transfer_control_bootstrap_exp8_leadfix_baseline.json`.

**Process note.** This is the third correction to the same number in one sitting:
a missingness leak (Part 12), a capacity confound (§13.2), and now an
underpowered interval. None was caught by a review pass; each was caught by
asking what would have to be true for the previous version to be wrong. Worth
recording in the methods chapter — the failure mode is not "wrong arithmetic", it
is "a defensible number reported with more confidence than its sample supports".
'''


def main() -> int:
    p = Path("reports/EXECUTION_LOG_2026-08-10.md")
    t = p.read_text(encoding="utf-8")
    if "## 13.4 The tie, bootstrapped" in t:
        print("13.4 already present")
        return 0
    p.write_text(t.rstrip("\n") + "\n" + TEXT, encoding="utf-8")
    print(f"appended 13.4 ({len(t.splitlines())} -> "
          f"{len(p.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
