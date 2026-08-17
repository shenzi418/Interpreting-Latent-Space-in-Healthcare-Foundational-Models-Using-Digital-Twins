"""Is the spatial-54 extractor broken, or is 75% all-NaN a real property?

neurokit2 failing to delineate three quarters of PTB-XL is not credible -- these
are clean 10 s clinical recordings. If the extractor is broken then result 2's
control arm is mostly a constant vector and the headline sign reversal is
measuring something other than what it claims.

Determines: (a) the all-NaN rate on the n=438 evaluation subset specifically,
(b) whether all-NaN lines up with the stored nk2_ok flag, (c) what happens when
the extractor is re-run live on a handful of the failing records.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SD = REPO / "scripts"
if str(SD) not in sys.path:
    sys.path.insert(0, str(SD))

TERR4 = ("Anteroseptal", "Anterolateral", "Inferior", "Inferolateral")

f = np.load(REPO / "data" / "ecg_features_spatial_ptbxl_test.npz", allow_pickle=True)
X = np.asarray(f["features"], dtype=float)
ok = np.asarray(f["nk2_ok"]).astype(bool)
names = [str(s) for s in np.asarray(f["feature_names"])]

all_nan = np.isnan(X).all(axis=1)
any_nan = np.isnan(X).any(axis=1)
print(f"PTB-XL fold-10 export: n={len(X)}")
print(f"  nk2_ok True      : {ok.sum()} ({ok.mean():.3f})")
print(f"  rows all-NaN     : {all_nan.sum()} ({all_nan.mean():.3f})")
print(f"  rows any-NaN     : {any_nan.sum()} ({any_nan.mean():.3f})")
print(f"  all_nan == ~ok   : {np.array_equal(all_nan, ~ok)}")
print(f"  ok & all_nan     : {(ok & all_nan).sum()}   (ok but no features)")
print(f"  ~ok & ~all_nan   : {(~ok & ~all_nan).sum()} (not-ok but has features)")

# the n=438 evaluation subset
sub = pd.read_csv(REPO / "data" / "ptbxl_mi_subclass.csv")
m4 = sub["territory_4c"].isin(TERR4)
idx = sub.loc[m4, "row_idx"].to_numpy()
print(f"\nn=438 evaluation subset (territory_4c in {TERR4}): n={idx.size}")
Xe = X[idx]
ae = np.isnan(Xe).all(axis=1)
print(f"  rows all-NaN     : {ae.sum()} ({ae.mean():.3f})")
print(f"  => the control sees {(~ae).sum()} rows with real measurements and "
      f"{ae.sum()} identical imputed rows")
lab = sub.loc[m4, "territory_4c"].to_numpy()
print("  all-NaN rate by class:")
for t in TERR4:
    mm = lab == t
    if mm.sum():
        print(f"      {t:15s} n={mm.sum():4d}  all-NaN={ae[mm].mean():.3f}")

# global6 columns are written unconditionally before delineation is attempted,
# so an all-NaN row means the *global* extractor failed too, not just delineation
g6 = [i for i, n in enumerate(names) if not any(
    n.startswith(k + "_") for k in ("ST_J60", "Q_amp", "R_amp", "T_amp"))]
print(f"\nglobal-6 columns at {g6} -> {[names[i] for i in g6]}")
print(f"  rows where global6 is entirely NaN: "
      f"{np.isnan(X[:, g6]).all(axis=1).mean():.3f}")

# --- live re-run on failing records
print("\n" + "=" * 68)
print("LIVE RE-RUN on 5 records the export marked as failed")
print("=" * 68)
try:
    from scripts.datasets import PTBXLDataset  # pylint: disable=wrong-import-position
    from scripts.extract_ecg_features_spatial import (  # pylint: disable=wrong-import-position
        extract_spatial_one_ecg,
    )

    ds = PTBXLDataset(
        root=REPO / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        sampling_rate=500, signal_duration=10.0, use_high_res=True,
        split="test", return_metadata=False,
    )
    bad = np.where(all_nan)[0][:5]
    for i in bad:
        sig = ds[int(i)][0]
        sig = sig.numpy() if hasattr(sig, "numpy") else np.asarray(sig)
        feats, good, reason = extract_spatial_one_ecg(sig, 500)
        nn = int(np.isnan(feats).sum())
        print(f"  row {i:5d}: shape={sig.shape} ok={good} reason={reason} "
              f"NaN={nn}/54")
except Exception as exc:  # noqa: BLE001
    print(f"  live re-run unavailable: {type(exc).__name__}: {exc}")
