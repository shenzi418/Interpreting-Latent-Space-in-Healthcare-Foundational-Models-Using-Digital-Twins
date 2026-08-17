"""Append 15.4 (reference provenance) to EXECUTION_LOG Part 15."""
from pathlib import Path

TEXT = '''
## 15.4 The determinism guard's baseline, verified — the check that would have cost a run

`run_spatial54_arm.check_latent_unchanged()` compares the new run's `Z` blocks
against `outputs/phase_b2_exp8` and aborts on any drift beyond `TOL=1e-12`. That
is the right guard, and it has two silent failure modes that the guard itself
cannot report:

**(a) An incomplete reference passes for the wrong reason.** The comparison
intersects config keys, so a reference missing configs shrinks the comparison
rather than failing it — the guard would print a success line having checked a
subset. `scripts/_check_global6_reference.py`: **5/5 configs present in all three
JSONs, 55 Z blocks total** (10 + 15 + 30). The intersection is the full set.

**(b) The reference could be the wrong protocol.** The arm runs at the default
`--scaler-domain target`; the reference JSONs record configs, territories, class
counts and the `logreg_Cs` grid under a `metadata` key, but **no `scaler_domain`
field**. So "the reference is the target-scaler run" was an assumption inherited
from how the directory was produced, not a property readable from it. If it were
wrong, "latent identical to 1e-12" would be an impossible bar and the run would
abort at the guard *after* paying its full cost.

`scripts/_identify_reference_scaler.py` settles it by fingerprinting the Z blocks
against all three scaler variants on disk:

| candidate | Z blocks bit-identical to reference | max abs delta |
|---|---|---|
| `target_pool` | 2/6 | 0.0610 |
| **`target`** | **6/6** | **0.000000** |
| `source` | 2/6 | 0.0628 |

Unambiguous: the reference is the `target`-scaler run, which is what the arm
defaults to. The guard's baseline is correct and the bar is achievable.

The 2/6 partial matches on the other two are worth a note rather than alarm — the
in-domain MedalCare blocks are scaler-invariant by construction (source rows fit
the scaler in every variant), so exactly the cross-domain blocks differ. That the
partial matches land on precisely the blocks that *should* be invariant is a
positive control on the fingerprint: a spurious match would not respect that
structure.

**Why this is worth 30 seconds.** The guard protects the comparison's validity,
but nothing protects the guard's own premises, and both premises here were
unstated. This is the same shape as §13.2's capacity confound and Part 12's leak:
the instrument was fine, the thing it was silently assuming was not. Checking a
guard's baseline before trusting its verdict is cheap; discovering it at abort
time costs the run.

## 15.5 Pre-flight status — all gates green except the one that is not mine to open

| gate | status |
|---|---|
| spatial54 NPZs exist, aligned 12019/2198 | ok (Part 15.1) |
| strict-superset bit-identical | ok, all 6 cols x 3 splits |
| missingness added by 48 new cols | none (finite fractions unchanged) |
| Part 12 leak gate (0.55) | ok, worst 0.5364 |
| patch anchors (`--check`) | ok, all three matched |
| reference completeness | ok, 5/5 configs, 55 Z blocks |
| reference protocol == arm default | ok, `target`, 6/6 bit-identical |
| **poolscaler all-5 finished** | **IN FLIGHT — pid 43608, config 2/5 done** |

Everything the decisive run depends on has been verified read-only. The single
remaining blocker is the §8.4 constraint: the patch mutates
`analysis/phase_b2_infarct_decoding.py`, which the in-flight poolscaler is
executing, so it must not be applied until pid 43608 exits.
'''


def main() -> int:
    p = Path("reports/EXECUTION_LOG_2026-08-10.md")
    t = p.read_text(encoding="utf-8")
    if "## 15.4 The determinism guard" in t:
        print("15.4 already present")
        return 0
    p.write_text(t.rstrip("\n") + "\n" + TEXT, encoding="utf-8")
    print(f"appended 15.4/15.5 ({len(t.splitlines())} -> "
          f"{len(p.read_text(encoding='utf-8').splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
