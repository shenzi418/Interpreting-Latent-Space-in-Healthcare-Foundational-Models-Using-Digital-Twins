"""Measure the Zhao et al. (ICML 2019) label-shift quantity for MedalCare vs PTB-XL.

Why this matters
----------------
The project's central negative result is "alignment fails: C2ST stays ~1.0 no
matter what we do". As written that reads as a report of five failed attempts --
a methods problem, and a reviewer's natural response is "you didn't try hard
enough".

Zhao, Tachet des Combes, Zhang & Gordon (ICML 2019, "On Learning Invariant
Representations for Domain Adaptation") prove an information-theoretic LOWER
BOUND (their Thm 4.3): if d_JS(D_Y^S, D_Y^T) >= d_JS(D_Z^S, D_Z^T), then for any
hypothesis h on top of representation g,

    err_S(h.g) + err_T(h.g)  >=  (1/2) * ( d_JS(D_Y^S, D_Y^T)
                                           - d_JS(D_Z^S, D_Z^T) )^2

where d_JS is the Jensen-Shannon *distance* (the square root of the JS
divergence). Driving the representation toward domain-invariance sends
d_JS(D_Z^S, D_Z^T) -> 0, which MAXIMISES the right-hand side: the floor becomes
(1/2) * JS(labels), set entirely by the label marginal gap that no amount of
feature alignment can touch.

So if the label marginals differ substantially, perfect alignment is not merely
hard here -- it is provably counterproductive, and the project's negative result
becomes a CONFIRMED PREDICTION rather than a failure to make something work.

IMPORTANT CAVEAT (do not drop this when citing the number)
----------------------------------------------------------
Thm 4.3 is stated for single-label classification, where D_Y is a genuine
distribution over classes. This project's task is MULTI-LABEL: a row can be both
MI and CD. The histogram computed below normalises per-class positive counts to
sum to 1, which is a PROXY for D_Y, not D_Y itself. The bound therefore
motivates and is consistent with the observed behaviour; it is not a literal
proof about this exact setup. The per-class positive rates are reported
alongside precisely so the reader can judge the gap without leaning on the
proxy. Report it as "the mechanism Zhao et al. identify is present here at
magnitude X", not "Zhao et al. prove our result".

Labels are built by the SAME code path the shared-head training used
(`_filter_medalcare_manifest`, `_filter_ptbxl_dataset`, `MEDALCARE_REMAP`,
`PTBXL_REMAP`), so this measures the marginal the model actually saw -- including
the Stage-1 defect-F fix that restored CD to the PTB-XL side. It reads CSVs only
(no signals, no latents, no model), so it is cheap and cannot be contaminated by
any defect the 2026-08-10 audit found in the latent pipeline.

Writes: outputs/analysis/label_shift/label_shift.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from scripts.datasets import PTBXLDataset  # noqa: E402
from scripts.finetune_multilabel import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_PTBXL_ROOT,
    MEDALCARE_REMAP,
    N_SHARED,
    PTBXL_REMAP,
    SHARED_LABELS,
    _filter_medalcare_manifest,
    _filter_ptbxl_dataset,
)

OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "label_shift"


def kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p||q) in bits; zero-probability terms contribute 0."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (in [0, 1] for two distributions)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_bits(p, m) + 0.5 * kl_bits(q, m)


def medalcare_shared_counts(split: str = "test") -> np.ndarray:
    """Per-class positive counts in the shared 3-class space, MedalCare side."""
    df = pd.read_csv(DEFAULT_MANIFEST)
    df = df[df["split"] == split].reset_index(drop=True)
    df = _filter_medalcare_manifest(df)
    counts = np.zeros(N_SHARED, dtype=np.float64)
    for src_idx, dst_idx in MEDALCARE_REMAP.items():
        counts[dst_idx] += float(df[f"label_{src_idx}"].sum())
    return counts, len(df)


def ptbxl_shared_counts(split: str = "test") -> np.ndarray:
    """Per-class positive counts in the shared 3-class space, PTB-XL side.

    Uses PTBXLDataset so the SCP->superclass mapping is the pipeline's own
    (SUPERCLASS_LABELS = NORM, MI, STTC, HYP, CD -- note CD is index 4, which is
    why PTBXL_REMAP maps 4 -> 2). Constructing the dataset reads CSVs only;
    signals are loaded lazily in __getitem__, which we never call.
    """
    ds = PTBXLDataset(root=DEFAULT_PTBXL_ROOT, split=split, sampling_rate=500,
                      signal_duration=10.0, use_high_res=True)
    ds = _filter_ptbxl_dataset(ds)
    counts = np.zeros(N_SHARED, dtype=np.float64)
    for src_idx, dst_idx in PTBXL_REMAP.items():
        counts[dst_idx] += float((ds.targets[:, src_idx] > 0).sum())
    return counts, len(ds.records)


def observed_joint_error(run_id: str = "exp7_baseline"):
    """(err_medalcare, err_ptbxl, joint) from a run's test metrics, or None.

    Uses 1 - accuracy on each domain's test split. Accuracy here is multi-label
    element-wise, so this is an indicative comparison against a bound stated for
    0/1 error -- but it is the right order of magnitude and it is the number the
    thesis already reports.
    """
    path = REPO_ROOT / "outputs" / run_id / "metrics.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    try:
        test = d["best"]["test"]
        err_med = 1.0 - float(test["medalcare"]["accuracy"])
        err_ptb = 1.0 - float(test["ptbxl"]["accuracy"])
    except (KeyError, TypeError, ValueError):
        return None
    return err_med, err_ptb, err_med + err_ptb


def main() -> int:
    print("=" * 78)
    print("Label-shift measurement: the Zhao et al. (ICML 2019) bound quantity")
    print("=" * 78)
    print(f"shared classes: {SHARED_LABELS}")

    c_med, n_med = medalcare_shared_counts("test")
    c_ptb, n_ptb = ptbxl_shared_counts("test")
    p_med = c_med / c_med.sum()
    p_ptb = c_ptb / c_ptb.sum()

    print(f"\nMedalCare test split  (n={n_med} rows after lae/fam filter)")
    print(f"  {'class':<6} {'pos_rate':>9} {'hist':>9} {'n_pos':>8}")
    for name, p, c in zip(SHARED_LABELS, p_med, c_med):
        print(f"  {name:<6} {c / n_med:>9.4f} {p:>9.4f} {int(c):>8}")

    print(f"\nPTB-XL test  (official fold 10, n={n_ptb} rows after STTC/HYP filter)")
    print(f"  {'class':<6} {'pos_rate':>9} {'hist':>9} {'n_pos':>8}")
    for name, p, c in zip(SHARED_LABELS, p_ptb, c_ptb):
        print(f"  {name:<6} {c / n_ptb:>9.4f} {p:>9.4f} {int(c):>8}")

    js = js_divergence_bits(p_med, p_ptb)
    d_js = float(np.sqrt(js))
    # Under perfect invariance d_JS(Z_S,Z_T) -> 0, so the Thm 4.3 floor is
    # (1/2)*(d_JS(Y) - 0)^2 = (1/2)*JS(Y).
    floor = 0.5 * js

    print(f"\n  JS divergence (bits) between label histograms : {js:.5f}")
    print(f"  d_JS = sqrt(JS)                               : {d_js:.5f}")
    print(f"  joint-error floor under PERFECT alignment     : {floor:.5f}")
    print("    = lower bound on (err_medalcare + err_ptbxl) as d_JS(Z) -> 0")

    # --- is the bound actually BINDING? --------------------------------------
    # This is the whole point. A floor only explains the project's failure if
    # the observed joint error is AT or BELOW it (i.e. the model is already
    # pressed against the wall). If observed error sits well ABOVE the floor,
    # label shift has slack and CANNOT be the reason alignment failed.
    observed = observed_joint_error()
    verdict = None
    if observed is not None:
        err_med, err_ptb, joint = observed
        slack = joint - floor
        print(f"\n  observed err_medalcare (1-acc, exp7_baseline test) : {err_med:.5f}")
        print(f"  observed err_ptbxl                                 : {err_ptb:.5f}")
        print(f"  observed JOINT error                               : {joint:.5f}")
        print(f"  slack above the floor                              : {slack:+.5f}")
        binding = slack <= 0.0
        verdict = "BINDING" if binding else "SLACK"
        if binding:
            print("\n  => BOUND IS BINDING: label shift alone can explain the failure.")
        else:
            print("\n  => BOUND IS SLACK. Observed joint error is above the floor, so")
            print("     label shift is NOT the binding constraint here. Perfect")
            print("     alignment would still permit a much better joint error than")
            print("     what we observe. The alignment failure needs another")
            print("     explanation -- it is NOT a corollary of Zhao et al.")

    print("\n  CAVEAT: multi-label proxy for D_Y, not D_Y itself -- see module docstring.")
    print("  CAVEAT: accuracy here is multi-label element-wise, so the comparison")
    print("          against a 0/1-error bound is indicative, not exact.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "shared_classes": list(SHARED_LABELS),
        "medalcare_test": {
            "n_rows": int(n_med),
            "counts": c_med.tolist(),
            "positive_rate": (c_med / n_med).tolist(),
            "histogram": p_med.tolist(),
        },
        "ptbxl_test_fold10": {
            "n_rows": int(n_ptb),
            "counts": c_ptb.tolist(),
            "positive_rate": (c_ptb / n_ptb).tolist(),
            "histogram": p_ptb.tolist(),
        },
        "js_divergence_bits": js,
        "d_js": d_js,
        "joint_error_floor_perfect_alignment": floor,
        "observed": (
            None if observed is None else {
                "run_id": "exp7_baseline",
                "err_medalcare_test": observed[0],
                "err_ptbxl_test": observed[1],
                "joint_error": observed[2],
                "slack_above_floor": observed[2] - floor,
                "verdict": verdict,
            }
        ),
        "caveat": (
            "Thm 4.3 is stated for single-label classification. This task is "
            "multi-label; the histogram is a normalised positive-count proxy for "
            "D_Y, not D_Y itself. Cite as 'the mechanism is present at this "
            "magnitude', not as a proof about this setup."
        ),
        "reference": (
            "Zhao, Tachet des Combes, Zhang & Gordon, 'On Learning Invariant "
            "Representations for Domain Adaptation', ICML 2019, pp. 7523-7532, "
            "Thm 4.3."
        ),
    }
    out_path = OUT_DIR / "label_shift.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
