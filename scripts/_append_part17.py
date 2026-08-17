"""Append Part 17: the poolscaler all-5 result, including the K64/globalz split."""
from pathlib import Path

TEXT = '''
---

# PART 17 — Poolscaler all-5: the strict scaler holds, and K64 fails cross-domain in a new way

`analysis/phase_b2_infarct_decoding.py --scaler-domain target_pool` over all five
`exp8_leadfix_*` configs. rc=0, 76.2 min. This is the strict arm from report §14.6:
the scaler is fitted on the **full unselected same-split PTB-XL matrix**, so it
cannot see which rows the MI-subclass filter picked — the leakage channel §14
found and §14.6 closed.

**Determinism first.** The driver re-verified the two original configs against the
2-config snapshot taken before the run: **44 blocks compared, all reproduced.** The
5-config file is a strict superset of the 2-config result, so §14.6's published
numbers stand unchanged and the three new configs are additive rather than a
restatement.

## 17.1 In-domain (Z), target_pool scaler

| config | phi_R2c | phi_MAE° | z_R2 | size_R2 | rho_AUC |
|---|---|---|---|---|---|
| baseline | 0.5546 | 45.01 | 0.3220 | 0.2644 | 0.9509 |
| ccmmd | 0.5642 | 44.30 | 0.3234 | 0.2712 | 0.9491 |
| dual | 0.5736 | 43.74 | 0.3121 | 0.2647 | 0.9502 |
| **globalz** | **0.5912** | **42.43** | **0.3768** | **0.2759** | 0.9388 |
| K64 | 0.3839 | 57.22 | 0.1867 | 0.1899 | 0.8709 |

In-domain θ decoding is essentially scaler-invariant — these match the tier1 pass-2
figures (§13.1) to within rounding, as they must: the source rows fit the scaler in
every variant, so only cross-domain blocks can move. That is the built-in control,
and it passed.

## 17.2 Cross-domain, pipeline A 4c — the primary comparison

| config | Z f1 | Z p | verdict |
|---|---|---|---|
| baseline | 0.2797 | **0.0098** | transfers |
| ccmmd | 0.2854 | **0.0057** | transfers |
| **dual** | **0.3050** | **0.0003** | transfers, strongest |
| globalz | 0.2817 | **0.0091** | transfers |
| K64 | 0.2321 | 0.1896 | **fails** |

**Four of five transfer under the strict scaler.** The lead-fix result survives the
tightest leakage control the project has: closing the scaler channel does not
remove cross-domain territory transfer.

## 17.3 The two findings worth extracting

**(a) `dual` is the best cross-domain config, and that is awkward for §14.9.**
p=0.0003 and macro-F1 0.3050, the strongest of the five — yet `dual` is the *worst*
config on class-structure transfer (§14.9: LR M→P 0.601 vs 0.827 shared-head). So
the architecture that transfers **labels** worst transfers **θ-territory** best.
Two readings, and they are not equivalent:

* the shared head buys label transfer by discarding exactly the biophysical
  structure the territory probe needs — a genuine trade-off, and the more
  interesting claim;
* or the dual-head latent is simply higher-variance and this is one draw.

§14.9's K64 dissociation (best on class structure, worst on mechanism) points the
same direction from the opposite end, which makes the trade-off reading more
plausible than a variance artifact — two independent encoders now separate the two
axes in opposite directions. **This is not currently in the report and should be.**

**(b) K64 fails cross-domain here too (p=0.19), and now under the strict scaler.**
§14.8 recorded all 8 cross-domain blocks losing significance for K64; the strict
scaler reproduces it. Combined with K64's in-domain survival (0.5234, p=0.0001)
this is the capacity-floor result stated twice under different leakage controls.
§16 Rank 3's K=128/256 probe is the right follow-up and is now better motivated.

**(c) The 3-class `globalz` anomaly.** globalz is the *only* config that fails the
primary 3-class block (0.2754, p=0.1759) while passing pipeline-A 4c (p=0.0091).
Worth a note, not a claim: the 3-class block is the coarser target, so failing it
while passing the finer one is odd enough to flag but rests on one config and one
block.

## 17.4 What may NOT be quoted from this run

**Every `ecg_features` column in this directory is void** (Part 12): the feature
NPZs cover MI rows only, so non-MI rows are scored against a median-imputed
constant, and the resulting "control" numbers measure imputation. The pipeline-B
calibrator columns make this visible — control p-values of **0.9996, 0.9994,
0.9991, 0.9993** across four configs. A p-value pinned at ~1.0 in four independent
configs is not a weak control, it is a broken one, and it is the same defect from
the other side: the calibrator fits phi-bins on a constant.

The `_summarise_poolscaler_all5.py` helper prints those columns under a literal
`[VOID]` header for exactly this reason — a number that must not be quoted is
safer visible-and-labelled than omitted, because omission invites someone to
recompute it later without the context.

The fair control comparison is the spatial54 arm, running now.

## 17.5 Scaler sensitivity — the Z arm does move

target_pool minus target, cross-domain, the two configs present in both:

| block | target | target_pool | delta |
|---|---|---|---|
| baseline / cross_domain_4c | 0.2786 | 0.2797 | +0.0012 |
| baseline / cross_domain_2c | 0.5919 | 0.6162 | **+0.0242** |
| ccmmd / cross_domain_4c | 0.2733 | 0.2854 | +0.0121 |
| ccmmd / cross_domain_2c | 0.5782 | 0.6392 | **+0.0610** |
| both / in_domain_4c | — | — | +0.0000 (exact) |

In-domain is bit-identical, as predicted. Cross-domain moves, and **the strict
scaler moves it *up* in all four cells** — up to +0.061. That direction matters:
the concern motivating §14.6 was that a target-fitted scaler might *flatter* the
latent by peeking at the selected rows. It does the opposite. Removing the peek
improves cross-domain transfer, so the effect §14.6 was guarding against was not
inflating the result — if anything the guarded version was conservative.
'''


def main() -> int:
    p = Path("reports/EXECUTION_LOG_2026-08-10.md")
    t = p.read_text(encoding="utf-8")
    if "# PART 17 —" in t:
        print("Part 17 already present")
        return 0
    p.write_text(t.rstrip("\n") + "\n" + TEXT, encoding="utf-8")
    print(f"appended Part 17 ({len(t.splitlines())} -> "
          f"{len(p.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
