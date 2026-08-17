"""Audit part 2 -- the artifacts that specifically carry the headline results.

Result 2 (the sign reversal: 54-d spatial control beats the 1024-d latent
cross-domain 5/5, loses in-domain 5/5) is the paper's centrepiece. Its validity
rests on things the generic verifiers never look at:

1. CONTROL INTEGRITY. If neurokit2 delineation fails often, the 54-d control is
   mostly imputed values. A control that is heavily imputed could be inflated
   (imputation leaks the target mean) or deflated (signal destroyed). Either way
   the comparison is not what it claims. Also: whatever imputation/scaling is
   used must be fit on SOURCE only -- fitting on the pooled or target set leaks.
2. CONVERGENCE. A degenerate encoder (collapsed head, no learning) would make
   "the latent loses cross-domain" trivially true for the wrong reason.
3. THE ACTUAL HEADLINE NUMBERS re-read from the JSON, not from the writeup.
4. phi for the LCX_*_post groups -- audit part 1 found LCX_0.3_post disagrees
   with its folder label while LCX_1.0_post does not. Transmurality should not
   move an infarct's angular position, so this asymmetry needs an explanation
   before it is trusted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

print("=" * 72)
print("1. SPATIAL-54 CONTROL INTEGRITY")
print("=" * 72)

FEATS = {
    "medalcare_train": "data/ecg_features_spatial_medalcare_train.npz",
    "medalcare_test": "data/ecg_features_spatial_medalcare_test.npz",
    "ptbxl_test": "data/ecg_features_spatial_ptbxl_test.npz",
}
for name, rel in FEATS.items():
    fp = REPO / rel
    if not fp.exists():
        cands = sorted(p.name for p in (REPO / "data").glob("ecg_features_spatial*"))
        print(f"  [miss] {rel}  (on disk: {cands})")
        continue
    z = np.load(fp, allow_pickle=True)
    key = "X" if "X" in z.files else ("features" if "features" in z.files else z.files[0])
    X = np.asarray(z[key], dtype=float)
    nan_frac = np.isnan(X).mean()
    per_col_nan = np.isnan(X).mean(axis=0)
    all_nan_rows = np.isnan(X).all(axis=1).mean()
    print(f"  {name}: shape={X.shape} keys={sorted(z.files)}")
    print(f"      overall NaN fraction : {nan_frac:.4f}")
    print(f"      rows entirely NaN    : {all_nan_rows:.4f}")
    print(f"      worst column NaN     : {per_col_nan.max():.4f}  "
          f"(cols >50% NaN: {(per_col_nan > 0.5).sum()}/{X.shape[1]})")
    if "ok" in z.files:
        print(f"      extraction ok flag   : {np.asarray(z['ok']).mean():.4f}")

print()
print("  --- where is the imputer/scaler fit? (must be SOURCE only) ---")
tc = (REPO / "analysis" / "transfer_control.py")
if tc.exists():
    txt = tc.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(txt, 1):
        s = line.strip()
        if any(t in s for t in ("fit(", "fit_transform(", "SimpleImputer",
                                "StandardScaler", "nan_to_num")):
            print(f"      L{i}: {s[:110]}")

print()
print("=" * 72)
print("2. exp8 TRAINING CONVERGENCE -- no degenerate encoders")
print("=" * 72)
for enc in ["exp8_leadfix_baseline", "exp8_leadfix_K64", "exp8_leadfix_ccmmd",
            "exp8_leadfix_dual", "exp8_leadfix_globalz"]:
    fp = REPO / "outputs" / enc / "metrics.json"
    if not fp.exists():
        print(f"  [miss] {enc}/metrics.json")
        continue
    j = json.loads(fp.read_text(encoding="utf-8"))
    hits = {}

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (int, float)) and any(
                    t in k.lower() for t in ("macro_f1", "macro_auc", "auc", "f1")
                ):
                    hits[f"{path}/{k}"] = v
                walk(v, f"{path}/{k}")

    walk(j)
    show = [(k, v) for k, v in hits.items() if "macro" in k.lower()][:4]
    if not show:
        show = list(hits.items())[:4]
    print(f"  {enc}:")
    for k, v in show:
        flag = "  <-- DEGENERATE?" if isinstance(v, float) and v < 0.2 else ""
        print(f"      {k} = {v}{flag}")

print()
print("=" * 72)
print("3. HEADLINE NUMBERS re-read from JSON (latent vs spatial54, cross-domain)")
print("=" * 72)


def dig(o, want, path="", out=None):
    if out is None:
        out = []
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}/{k}"
            if k == want and isinstance(v, (int, float)):
                out.append((p, v))
            dig(v, want, p, out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            dig(v, want, f"{path}[{i}]", out)
    return out


for label, rel in [
    ("LATENT   ", "outputs/phase_b2_exp8/cross_domain_4c_pipelineA.json"),
    ("CONTROL54", "outputs/phase_b2_exp8_spatial54/cross_domain_4c_pipelineA.json"),
]:
    fp = REPO / rel
    if not fp.exists():
        print(f"  [miss] {rel}")
        continue
    j = json.loads(fp.read_text(encoding="utf-8"))
    f1s = dig(j, "macro_f1")
    ps = dict(dig(j, "p_macro_f1"))
    nperm = dig(j, "n_permutation_macro_f1")
    print(f"  {label}  ({rel})")
    for p, v in f1s[:8]:
        pv = ps.get(p.rsplit("/", 1)[0] + "/p_macro_f1")
        print(f"      {p} = {v:.4f}" + (f"   p={pv}" if pv is not None else ""))
    if nperm:
        print(f"      n_permutation values seen: "
              f"{sorted({v for _, v in nperm})}")

print()
print("=" * 72)
print("4. phi FOR THE LCX_*_post GROUPS -- why does only 0.3 disagree?")
print("=" * 72)
for split in ["train", "val", "test"]:
    z = np.load(REPO / "data" / f"theta_mi_{split}.npz", allow_pickle=True)
    t8 = np.asarray(z["territory_8c"]).astype(str)
    phi = np.asarray(z["phi"], dtype=float)
    t4 = np.asarray(z["territory_4c"]).astype(str)
    t4f = np.asarray(z["territory_4c_folder"]).astype(str)
    print(f"  --- {split}")
    for grp in sorted(set(t8.tolist())):
        if "LCX" not in grp:
            continue
        m = t8 == grp
        disagree = (t4[m] != t4f[m]).mean()
        print(f"      {grp:16s} n={m.sum():5d}  "
              f"phi mean={np.rad2deg(phi[m]).mean():8.2f} deg  "
              f"[{np.rad2deg(phi[m]).min():7.2f}, {np.rad2deg(phi[m]).max():7.2f}]  "
              f"disagree={disagree:.2f}  -> 4c={sorted(set(t4[m].tolist()))}")
