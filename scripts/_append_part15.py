"""Append Part 15 (pre-flight for the decisive run) to EXECUTION_LOG."""
from pathlib import Path

TEXT = '''
---

# PART 15 — Pre-flight for the decisive run: the superset claim, measured

§15 of the report justifies the spatial54 control with a strict-superset argument:
the 6 global columns are computed by calling the original implementation verbatim,
so any drop against `global6` is attributable to estimation cost on 54 columns and
never to lost information. That argument is about *how the extractor was written*.
It had not been checked against *the artifact the extractor produced*, and those
are different claims — Part 12's leak was exactly a case where the artifact did
not have the property the code implied.

`scripts/_preflight_spatial54.py` (read-only) checks it, plus the three other
invariants `run_spatial54_arm.py` would otherwise only discover after launch.

## 15.1 Results

| check | result |
|---|---|
| superset bit-identical, medalcare_train | **exact on all 6 shared cols** (12019 rows) |
| superset bit-identical, medalcare_test | **exact on all 6 shared cols** (2386 rows) |
| superset bit-identical, ptbxl_test | **exact on all 6 shared cols** (2198 rows) |
| fully-finite rows, med_train | 0.3088 -> 0.3088 (unchanged) |
| fully-finite rows, med_test | 0.2938 -> 0.2938 (unchanged) |
| fully-finite rows, ptb_test | 0.1815 -> 0.1815 (unchanged) |
| feature/latent row alignment | 12019 / 2198 both sides, scoring 5347 med / 438 ptb |
| missingness-AUROC (Part 12 gate, 0.55) | worst **0.5364** (Inferolateral, ptb) |

Column-name matched, compared with `atol=0, rtol=0` — bitwise, not approximate.
So the superset property holds as an *artifact* property, and the report's §15
reasoning can be quoted as measured.

## 15.2 The two results worth noting beyond a green light

**Adding 48 columns adds zero missingness.** The fully-finite fraction is identical
to four decimals on all three splits. That is not the expected outcome — 48 more
per-lead estimates are 48 more chances to fail — and it means the per-lead block
succeeds exactly whenever the global block does. §15's smoke test hinted at this
(the spatial block completed 12/12 while P-duration/QT failed on 2/12); at full
scale the implication is stronger: **every incomplete row is incomplete because of
the original 6 features, not the new 48.** If the control underperforms, the
6-column legacy block is where to look.

**The missingness gate passes with room, and its margin is informative.** Worst
cell 0.5364 against a 0.55 threshold, versus 0.847/0.863 for the leaking version
in Part 12. The residual above 0.5 is not noise — it is the same 4:1
missingness-by-territory imbalance in both domains, just far too weak to carry a
label. Worth keeping the gate in place rather than declaring the leak class closed.

## 15.3 Why this ran before the patch rather than after

`run_spatial54_arm.py` already refuses to start on a mismatched latent block, so
these invariants would have been caught eventually. The reason to check first is
sequencing: the arm can only run after `_apply_featureset_patch.py` mutates
`analysis/phase_b2_infarct_decoding.py`, and that patch is itself gated on the
poolscaler finishing. A failure discovered post-patch would have to be diagnosed
with a modified analysis script in the tree — the one configuration in which it is
hardest to tell an artifact problem from a patch problem. Ten seconds of read-only
checking removes that ambiguity from the critical path.

Status at time of writing: poolscaler all-5 still in flight (pid 43608, config 2
of 5 complete). Patch verified (`--check`: all anchors matched) and still unapplied,
per the §8.4 constraint. Pre-flight clean. The decisive run is one gate away.
'''


def main() -> int:
    p = Path("reports/EXECUTION_LOG_2026-08-10.md")
    t = p.read_text(encoding="utf-8")
    if "# PART 15 —" in t:
        print("Part 15 already present")
        return 0
    p.write_text(t.rstrip("\n") + "\n" + TEXT, encoding="utf-8")
    print(f"appended Part 15 ({len(t.splitlines())} -> "
          f"{len(p.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
