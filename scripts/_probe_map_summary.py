"""Aggregate the 48-cell probing map into the physiology x anatomy summary."""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "outputs" / "analysis" / "probe_map"
CFGS = ["exp8_leadfix_baseline", "exp8_leadfix_ccmmd", "exp8_leadfix_dual",
        "exp8_leadfix_globalz", "exp8_leadfix_K64"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
KINDS = ["ST_J60", "Q_amp", "R_amp", "T_amp"]

fr = pd.concat([pd.read_csv(D / f"probe_map_{c}.csv").assign(config=c) for c in CFGS])
ok = fr[fr.status == "ok"]

print("=== by physiology kind (median over 12 leads x 5 encoders) ===")
g = ok.groupby("kind")[["rho_in", "rho_cross_source", "d_rho"]].median().round(3)
g["n_sig"] = ok[ok.perm_p_cross_source < 0.05].groupby("kind").size()
g["n_cells"] = ok.groupby("kind").size()
print(g.loc[KINDS])

print("\n=== by lead (median over 4 kinds x 5 encoders) ===")
print(ok.groupby("lead")[["rho_in", "rho_cross_source", "d_rho"]].median().round(3).loc[LEADS])

b = ok[ok.config == "exp8_leadfix_baseline"]
cols = ["feature", "rho_in", "rho_cross_source", "d_rho", "perm_p_cross_source"]
print("\n=== baseline: 8 worst-transferring cells ===")
print(b.nlargest(8, "d_rho")[cols].round(3).to_string(index=False))
print("\n=== baseline: 8 best-transferring cells ===")
print(b.nsmallest(8, "d_rho")[cols].round(3).to_string(index=False))

print("\n=== cells NOT significant cross-domain (any encoder) ===")
ns = ok[ok.perm_p_cross_source >= 0.05]
print(ns[["config", "feature", "rho_in", "rho_cross_source",
          "perm_p_cross_source"]].round(3).to_string(index=False))

print("\n=== probe robustness: source vs target_pool X-scaler ===")
dd = (ok.rho_cross_source - ok.rho_cross_target_pool).abs()
print(f"  max |diff| = {dd.max():.4f}   median |diff| = {dd.median():.4f}")
print(f"  corr = {ok[['rho_cross_source', 'rho_cross_target_pool']].corr().iloc[0, 1]:.4f}")

print("\n=== evaluation pool: all 549 vs 438 primary-4c subset ===")
print(f"  median rho_cross full549   = {ok.rho_cross_source.median():.3f}")
print(f"  median rho_cross primary4  = {ok.rho_cross_source_primary4.median():.3f}")
