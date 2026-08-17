"""Verify every number quoted in EXECUTION_LOG Part 19 / report S17 against the
sweep artifact. Numbers in a writeup are claims; this re-derives each one.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parent.parent
J = json.loads((REPO / "outputs" / "analysis" / "leadperm_sweep"
                / "leadperm_sweep.json").read_text(encoding="utf-8"))
rows = J["rows"]
by = {r["name"]: r for r in rows}
I = by["identity"]
LIMB = {"I", "II", "III", "aVR", "aVL", "aVF"}
tr = [r for r in rows if r["name"] != "identity" and not r["name"].startswith("random")]
rd = [r for r in rows if r["name"].startswith("random")]
lo, hi = I["macro_f1_ci95"]
f1 = lambda g: [r["macro_f1"] for r in g]
gb = [r for r in rows if r.get("c2st_gbdt") is not None]

checks = [
    ("77 cells", len(rows), 77),
    ("66 transpositions", len(tr), 66),
    ("10 randoms", len(rd), 10),
    ("19 GBDT cells", len(gb), 19),
    ("identity max|d| == 0", J["metadata"]["identity_max_abs_dev_vs_stored"], 0.0),
    ("probe never refit", J["metadata"]["probe"]["refit"], False),
    ("n_eval 438", J["metadata"]["n_eval_rows"], 438),
    ("identity macro-F1 0.2599", round(I["macro_f1"], 4), 0.2599),
    ("identity p 0.1139", round(I["p_macro_f1"], 4), 0.1139),
    ("min macro-F1 0.1618", round(min(f1(rows)), 4), 0.1618),
    ("max macro-F1 0.3201", round(max(f1(rows)), 4), 0.3201),
    ("transposition spread 0.0929",
     round(max(f1(tr)) - min(f1(tr)), 4), 0.0929),
    ("C2ST lin spread 1e-5",
     round(max(r["c2st_linear"] for r in rows)
           - min(r["c2st_linear"] for r in rows), 5), 1e-5),
    ("C2ST gbdt spread 9e-5",
     round(max(r["c2st_gbdt"] for r in gb)
           - min(r["c2st_gbdt"] for r in gb), 5), 9e-5),
    ("randoms mean 0.2012", round(np.mean(f1(rd)), 4), 0.2012),
    ("transpositions above identity = 39",
     sum(r["macro_f1"] > I["macro_f1"] for r in tr), 39),
    ("transpositions below identity CI = 0",
     sum(r["macro_f1"] < lo for r in tr), 0),
    ("transpositions above identity CI = 2",
     sum(r["macro_f1"] > hi for r in tr), 2),
    ("randoms below identity CI = 7", sum(r["macro_f1"] < lo for r in rd), 7),
    ("randoms with p<0.05 = 0", sum(r["p_macro_f1"] < 0.05 for r in rd), 0),
    ("aVL<->aVF F1 0.2817", round(by["aVL<->aVF"]["macro_f1"], 4), 0.2817),
    ("aVL<->aVF delta +0.0218",
     round(by["aVL<->aVF"]["macro_f1"] - I["macro_f1"], 4), 0.0218),
    ("aVL<->aVF damage rank 70/77",
     sorted(rows, key=lambda x: x["macro_f1"]).index(by["aVL<->aVF"]) + 1, 70),
    ("mean CI width 0.0793",
     round(np.mean([b - a for a, b in (r["macro_f1_ci95"] for r in rows)]), 4),
     0.0793),
    ("identity MMD2 rank 8/77",
     sorted(r["mmd2"] for r in rows).index(I["mmd2"]) + 1, 8),
    ("MMD2 transposition mean 0.1958",
     round(np.mean([r["mmd2"] for r in tr]), 4), 0.1958),
    ("MMD2 random mean 0.2673",
     round(np.mean([r["mmd2"] for r in rd]), 4), 0.2673),
    ("rho C2ST-lin -0.040",
     round(spearmanr([r["c2st_linear"] for r in rows], f1(rows)).statistic, 3),
     -0.040),
    ("rho MMD2 -0.146",
     round(spearmanr([r["mmd2"] for r in rows], f1(rows)).statistic, 3), -0.146),
    ("rho C2ST-gbdt -0.031",
     round(spearmanr([r["c2st_gbdt"] for r in gb], f1(gb)).statistic, 3), -0.031),
]

bad = 0
for label, got, want in checks:
    ok = (got == want) if isinstance(want, (bool, int, str)) else \
         abs(got - want) < 1e-9
    print(f"  {'OK ' if ok else 'FAIL'}  {label:<38} got={got}")
    bad += not ok

# cross-check the two documents actually contain the headline figures
docs = {
    "EXECUTION_LOG Part 19": (REPO / "reports" / "EXECUTION_LOG_2026-08-10.md",
                              "# PART 19"),
    "report S17": (REPO / "reports" / "2026-08-11_breakthrough_analysis.md",
                   "## 17. Stage 4.2"),
}
QUOTED = ["0.2599", "0.3201", "0.1618", "0.0929", "0.2012", "39", "70/77",
          "0.2817", "0.1139"]
print()
for name, (fp, marker) in docs.items():
    txt = fp.read_text(encoding="utf-8")
    seg = txt[txt.index(marker):]
    seg = seg[:seg.index("\n# PART 20")] if "\n# PART 20" in seg else seg
    miss = [q for q in QUOTED if q not in seg]
    print(f"  {'OK ' if not miss else 'FAIL'}  {name}: "
          f"{len(QUOTED)-len(miss)}/{len(QUOTED)} headline figures present"
          + (f"  missing={miss}" if miss else ""))
    bad += bool(miss)

print(f"\n{'ALL CHECKS PASS' if not bad else f'{bad} FAILURES'}")
sys.exit(1 if bad else 0)
