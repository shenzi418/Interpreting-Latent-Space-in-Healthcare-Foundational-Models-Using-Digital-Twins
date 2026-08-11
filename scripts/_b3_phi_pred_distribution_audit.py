"""Audit predicted-phi distribution per truth_4c on MedalCare-test vs PTB-XL.

For exp7_baseline (the headline config), refit the phi-Ridge regressor on
MedalCare-train Z, then report median + interquartile range of the predicted
phi on (a) MedalCare-test in-domain and (b) PTB-XL primary 4c subset, broken
down by truth_4c territory. If the in-domain medians match the MedalCare phi
bin centers (~+1, +2.5, -1, -2.5 for the four territories) but the PTB-XL
medians collapse, that visualises the cross-domain shift that Pipeline B fails
to recover.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make analysis/ importable.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from sklearn.linear_model import RidgeCV

from analysis.phase_b2_infarct_decoding import (
    load_targets, load_features, load_config_latents, load_ptbxl_latents,
    load_ptbxl_subclass_csv,
    fit_scaler,
    TERRITORIES_4C,
)


def main() -> None:
    targets = load_targets()
    _ = load_features()
    Z_train, Z_test = load_config_latents("exp7_baseline")
    idx_train = targets["train"]["idx_in_split"]
    idx_test = targets["test"]["idx_in_split"]
    Z_train_mi = Z_train[idx_train].astype(np.float64)
    Z_test_mi = Z_test[idx_test].astype(np.float64)

    sc = fit_scaler(Z_train_mi)
    Z_train_std = sc.transform(Z_train_mi)
    Z_test_std = sc.transform(Z_test_mi)

    phi_train = targets["train"]["phi"]
    Y_train = np.stack([np.sin(phi_train), np.cos(phi_train)], axis=1)
    model = RidgeCV(alphas=np.logspace(-3, 3, 7)).fit(Z_train_std, Y_train)

    df = load_ptbxl_subclass_csv()
    mask = df["territory_4c"].isin(TERRITORIES_4C)
    primary_4c_idx = df[mask]["row_idx"].to_numpy()
    primary_4c_truth = df[mask]["territory_4c"].to_numpy()
    Z_ptbxl = load_ptbxl_latents("exp7_baseline")[primary_4c_idx].astype(np.float64)
    Y_ptbxl = model.predict(sc.transform(Z_ptbxl))
    phi_pred_ptbxl = np.arctan2(Y_ptbxl[:, 0], Y_ptbxl[:, 1])

    Y_test = model.predict(Z_test_std)
    phi_pred_test = np.arctan2(Y_test[:, 0], Y_test[:, 1])
    territory_4c_test = targets["test"]["territory_4c"]
    # convert to flat str array
    territory_4c_test = np.asarray(territory_4c_test.tolist(), dtype=object)

    print("=== Predicted phi (radians) distribution by truth_4c ===")
    print()
    print(f"{'territory':>14s} |"
          f"  MedalCare-test (in-domain)            |"
          f"  PTB-XL (cross-domain)")
    print(f"{'':>14s} |"
          f"  {'n':>4s}  {'median':>7s} {'(q25,':>7s} {'q75)':>7s}   |"
          f"  {'n':>4s}  {'median':>7s} {'(q25,':>7s} {'q75)':>7s}")
    print("-" * 100)
    for t in TERRITORIES_4C:
        m_mask = territory_4c_test == t
        p_mask = primary_4c_truth == t
        m_n = int(m_mask.sum())
        p_n = int(p_mask.sum())
        if m_n > 0:
            mq = np.percentile(phi_pred_test[m_mask], [25, 50, 75])
            m_str = f"{m_n:>4d}  {mq[1]:>+7.2f} ({mq[0]:>+5.2f}, {mq[2]:>+5.2f})  "
        else:
            m_str = " n=0                                "
        if p_n > 0:
            pq = np.percentile(phi_pred_ptbxl[p_mask], [25, 50, 75])
            p_str = f"{p_n:>4d}  {pq[1]:>+7.2f} ({pq[0]:>+5.2f}, {pq[2]:>+5.2f})"
        else:
            p_str = " n=0"
        print(f"{t:>14s} |  {m_str} |  {p_str}")

    print()
    print("Expected MedalCare ground-truth phi bin centers:")
    print("  Anteroseptal  ~ +1.0 rad  (LAD wedge midpoint)")
    print("  Anterolateral ~ +2.5 rad  (LCX_*_ant wedge midpoint)")
    print("  Inferior      ~ -1.0 rad  (RCA wedge midpoint)")
    print("  Inferolateral ~ -2.5 rad  (LCX_*_post wedge midpoint)")


if __name__ == "__main__":
    main()
