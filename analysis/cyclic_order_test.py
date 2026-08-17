"""RETRACTED 2026-08-13 -- do not use b_hat or any non-integer S(b) value.

The adversarial audit (reports/2026-08-13_fidelity_audit_and_final_verification.md,
Part A.1) established that S(b) is invariant to which territory the anchors are
unwrapped from ONLY at integer b (spread ~1e-17 at b in {-1,0,1,2}; 1.7e-01 at
b=0.5). Every non-integer b_hat this script reports is therefore an artifact of the
arbitrary Anteroseptal unwrap origin: b_hat moves 0.718 -> 1.116 across the four
equally defensible origins, the bootstrap of b_hat is a two-point alias mass (not a
distribution -- the printed CI spans a region of zero bootstrap mass), and the
permutation null pins b_hat at ~0 under any random labelling, so the maximisation is
NOT charged against the null as claimed below. This non-uniqueness is documented
prior art: Kempter et al. 2012, J Neurosci Methods 207:113-124 ("barber's pole"
solutions of argmax-R circular-linear regression). What survives of this script's
output: only the branch-invariant integer points -- S(1) exceeds S(0) in all six
encoders, far outside the label-permutation null, i.e. territory-dependent angular
structure exists. Nothing about its gain, scale, or orientation survives.
The original docstring is retained below for the record.

--------------------------------------------------------------------------------

Is the transported territory map monotone-but-compressed, or is it noise?

`floor_audit.py` Part 4 shows that under the target scaler the four predicted
territory means traverse the true anatomical cycle AS -> AL -> IL -> Inf in 5 of 6
encoders. That descriptive fact cannot be tested as stated, for two reasons:

  * Four territories admit only 3! = 6 cyclic orders, so ANY order statistic built
    on the four group means has a floor of p = 1/6 = 0.167. It can never reach
    significance, however clean the picture looks.
  * The six `exp8_leadfix_*` encoders are six fine-tunes of one backbone on one
    dataset pair -- a robustness sweep over training configuration, not six
    independent replicates. No binomial may be taken across them.

So the test has to move from the four group means down to the 4324 rows.
[FALSE -- see banner: partials() reduces the 4324 rows to 4 complex numbers, and
78-83% of the statistic's mass sits on two of the four anchors.]

The curve
---------
Allow the transported readout one rotation and one gain on the anatomical angle,
and read off the resultant as a function of the gain:

    S(b) = | mean_i exp( i ( pred_i - b * t_i ) ) |

where t_i is the true anchor angle of row i's territory, unwrapped from
Anteroseptal into [0, 2pi) so that b is a gain on "degrees around the ventricle
starting at the septum". The rotation is free: the modulus is invariant to it.
Three points on this one curve carry the whole argument.

    S(0)     the predictor's own angular concentration, with the territory labels
             switched off entirely -- the no-correspondence baseline
    S(1)     exactly the circular resultant reported in Tier 1
    S(b_hat) the best order-preserving affine map

b_hat ~ 1 with S(b_hat) >> S(0) means the angular correspondence is there and
correctly scaled. 0 < b_hat < 1 means correct order, compressed range -- the
hypothesis. b_hat < 0 means reversed. b_hat ~ 0 means absent.
[FALSE -- see banner: every one of these readings assumes b_hat is well-defined
off the integers, and it is not; it is a function of the unwrap origin.]

Inference
---------
  null        permute territory labels between whole patients, recompute max_b S(b)
              with the same maximisation, so the two free parameters are charged
              against the null rather than assumed free
              [FALSE -- see banner: under random labels the four partials become
              near-parallel and the maximiser pins at b~0 (null b_hat never left
              [-0.16,+0.16] in 4000 draws); the null charges ~nothing, and the
              test fires at the p-floor in all twelve cells. Also: this
              patient-block shuffle preserves the patient-level label marginal,
              not the row-level one.]
  bootstrap   resample patients with replacement -> CI on b_hat and S(b_hat)

This is what the transport permutation null cannot do: there R_null = R_pred *
R_floor identically, so it measures predictor diffuseness rather than pairing.

Note this is NOT zero-shot transport -- b_hat and the rotation are two parameters
read off the target domain. The claim it supports is the weaker and more defensible
one: an order-preserving angular correspondence exists up to a 2-parameter
recalibration, which is far less than the 4 free offsets a saturated model needs.

Read-only over stored latents. Usage:
  python analysis/cyclic_order_test.py
  python analysis/cyclic_order_test.py --encoders exp8_leadfix_medalonly
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from geom_common import (  # pylint: disable=wrong-import-position
    ENCODERS,
    TERRITORIES,
    RidgeSVD,
    angles_from_cs,
    load_medalcare,
    load_ptbxl,
    medalcare_anchor_angles,
    resultant,
)

OUT_JSON = REPO_ROOT / "outputs/analysis/circular_geometry/cyclic_order_test.json"
B_GRID = np.arange(-2.0, 3.0 + 1e-9, 0.002)
N_BOOT = 2000
N_PERM = 2000


def _canonical_anchors(anchors: dict) -> np.ndarray:
    """The four anchor angles unwrapped from Anteroseptal into [0, 2pi).

    Fixing the branch is what makes the gain `b` interpretable: it is a gain on
    arc length travelled around the ventricle starting at the septum.
    """
    base = anchors[TERRITORIES[0]]
    return np.array([(anchors[t] - base) % (2 * np.pi) for t in TERRITORIES])


def s_curve(part: np.ndarray, t_anchor: np.ndarray) -> np.ndarray:
    """S(b) for the whole grid, from the 4 per-territory partial sums.

    `part[k]` = (1/n) * sum over rows of territory k of exp(i * pred), so
    S(b) = | sum_k part[k] * exp(-i b t_k) |. Four terms, whatever n is.
    """
    phase = np.exp(-1j * np.outer(B_GRID, t_anchor))      # (n_b, 4)
    return np.abs(phase @ part)


def partials(pred_c: np.ndarray, terr: np.ndarray, n: int) -> np.ndarray:
    return np.array([pred_c[terr == t].sum() / n for t in TERRITORIES])


def order_of_means(part: np.ndarray) -> bool:
    """Do the four predicted territory means traverse the true cycle?"""
    m = np.angle(part)
    u = (m - m[0]) % (2 * np.pi)
    return bool(np.all(np.diff(u) > 0))


def run_encoder(encoder: str, anchors: dict, n_boot: int, n_perm: int,
                seed: int) -> dict:
    medal = load_medalcare(encoder)
    ptb = load_ptbxl(encoder, anchors)
    t_anchor = _canonical_anchors(anchors)
    n = len(ptb)

    model = RidgeSVD().fit(medal.z, np.c_[np.cos(medal.angle), np.sin(medal.angle)])
    groups = np.asarray(ptb.group)
    uniq = np.unique(groups)
    idx_by_group = [np.where(groups == g)[0] for g in uniq]

    res: dict = {
        "encoder": encoder, "n": int(n), "n_groups": int(len(uniq)),
        "anchor_unwrapped_deg": [round(float(np.degrees(x)), 1) for x in t_anchor],
    }

    for mode in ("source", "target"):
        if mode == "source":
            pred = angles_from_cs(model.predict(ptb.z))
        else:
            zt = (ptb.z - ptb.z.mean(0)) / (ptb.z.std(0) + 1e-8)
            pred = angles_from_cs(model.predict(zt * model.sd_ + model.mu_))
        pred_c = np.exp(1j * pred)

        part = partials(pred_c, ptb.territory, n)
        curve = s_curve(part, t_anchor)
        k_hat = int(curve.argmax())
        b_hat, s_hat = float(B_GRID[k_hat]), float(curve[k_hat])
        s0 = float(np.abs(part.sum()))                    # b = 0, labels switched off
        s1 = float(np.abs(part @ np.exp(-1j * t_anchor)))  # b = 1, the Tier 1 R

        rng = np.random.default_rng(seed)
        b_boot, s_boot = np.empty(n_boot), np.empty(n_boot)
        for j in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            rows = np.concatenate([idx_by_group[i] for i in pick])
            c = s_curve(partials(pred_c[rows], ptb.territory[rows], len(rows)),
                        t_anchor)
            k = int(c.argmax())
            b_boot[j], s_boot[j] = B_GRID[k], c[k]

        # Null: whole patients keep their latents, but swap territory labels.
        rng = np.random.default_rng(seed + 1)
        terr = ptb.territory
        s_null = np.empty(n_perm)
        b_null = np.empty(n_perm)
        for j in range(n_perm):
            perm = rng.permutation(len(uniq))
            shuf = terr.copy()
            for dst, src in enumerate(perm):
                a, b = idx_by_group[dst], idx_by_group[src]
                shuf[a] = terr[b][rng.integers(0, len(b), len(a))]
            c = s_curve(partials(pred_c, shuf, n), t_anchor)
            k = int(c.argmax())
            b_null[j], s_null[j] = B_GRID[k], c[k]

        res[mode] = {
            "S_at_b0_no_correspondence": s0,
            "S_at_b1_equals_tier1_R": s1,
            "tier1_R_check": resultant(pred - ptb.angle),
            "b_hat": b_hat,
            "S_at_b_hat": s_hat,
            "gain_over_b0": s_hat - s0,
            "b_hat_boot_ci95": [float(np.percentile(b_boot, 2.5)),
                                float(np.percentile(b_boot, 97.5))],
            "S_hat_boot_ci95": [float(np.percentile(s_boot, 2.5)),
                                float(np.percentile(s_boot, 97.5))],
            "boot_frac_b_positive": float(np.mean(b_boot > 0)),
            "null_S_mean": float(s_null.mean()),
            "null_S_p95": float(np.percentile(s_null, 95)),
            "null_b_hat_mean": float(b_null.mean()),
            "perm_p": (1 + int((s_null >= s_hat).sum())) / (n_perm + 1),
            "four_means_order_preserved": order_of_means(part),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoders", nargs="*", default=ENCODERS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    anchors = medalcare_anchor_angles()
    out = {}
    for enc in args.encoders:
        try:
            out[enc] = run_encoder(enc, anchors, args.n_boot, args.n_perm, args.seed)
        except FileNotFoundError as exc:
            print(f"=== {enc}: SKIPPED ({exc}) ===")
            continue
        r = out[enc]
        print(f"\n=== {enc}  n={r['n']} ({r['n_groups']} patients) ===")
        for mode in ("source", "target"):
            m = r[mode]
            print(f"  [{mode}]  S(0)={m['S_at_b0_no_correspondence']:.3f}   "
                  f"S(1)={m['S_at_b1_equals_tier1_R']:.3f}   "
                  f"S(b_hat)={m['S_at_b_hat']:.3f}   "
                  f"b_hat={m['b_hat']:+.3f} "
                  f"CI[{m['b_hat_boot_ci95'][0]:+.2f},{m['b_hat_boot_ci95'][1]:+.2f}]")
            print(f"{'':11}null max_b S: mean {m['null_S_mean']:.3f}, "
                  f"p95 {m['null_S_p95']:.3f}  ->  perm_p={m['perm_p']:.4f}   "
                  f"4-mean order {'OK' if m['four_means_order_preserved'] else '--'}")

    print("\n" + "=" * 100)
    print(f"{'encoder':<24}{'scaler':<9}{'S(0)':>7}{'S(1)':>7}{'S(bhat)':>9}"
          f"{'b_hat':>8}{'b CI95':>16}{'nullS95':>9}{'perm p':>9}")
    print("-" * 100)
    for enc, r in out.items():
        for mode in ("source", "target"):
            m = r[mode]
            ci = f"[{m['b_hat_boot_ci95'][0]:+.2f},{m['b_hat_boot_ci95'][1]:+.2f}]"
            print(f"{enc:<24}{mode:<9}{m['S_at_b0_no_correspondence']:>7.3f}"
                  f"{m['S_at_b1_equals_tier1_R']:>7.3f}{m['S_at_b_hat']:>9.3f}"
                  f"{m['b_hat']:>+8.3f}{ci:>16}{m['null_S_p95']:>9.3f}"
                  f"{m['perm_p']:>9.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
