"""Integrity checks on outputs/dim_scan/ deliverables.

Validates:
  * All 6 (config x pca_mode) summary JSONs parse and contain per_K for the 8
    expected Ks.
  * EVR(K=1024) == 1.0 (full-rank PCA).
  * EVR is monotonically increasing in K.
  * MMD goes UP as K decreases (AAAI 2020 prediction).
  * C2ST saturates near 1.0 at high K -- pre-registered evidence that the
    synth-real separator is dim-robust.
  * kstar_table.json values match a re-application of the pre-registered rule.

Exit code 0 only if all checks pass.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "outputs" / "dim_scan"

CONFIGS = ("exp7_baseline", "exp7_ccmmd")
PCA_MODES = ("combined", "medalcare", "ptbxl")
EXPECTED_KS = (1024, 512, 256, 128, 64, 32, 16, 8)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    msg = f"[{mark}] {label}"
    if detail:
        msg += "  -- " + detail
    print(msg)
    return ok


def reapply_kstar(per_k: Dict[str, Dict]) -> Dict[str, object]:
    """Re-apply the pre-registered K* rule from analysis/dim_scan.py."""
    eligible: List[int] = []
    backup: List[tuple] = []
    for ks_str in per_k.keys():
        k = int(ks_str)
        c2st = per_k[ks_str]["alignment"]["c2st_auc"]
        lr = per_k[ks_str]["class_structure"]["lr_m2p"]["macro_auc"]
        phi = per_k[ks_str]["mechanism"]["phi"]["circular_r2"]
        if c2st <= 0.85 and lr >= 0.65 and phi >= 0.35:
            eligible.append(k)
        if c2st <= 0.95:
            backup.append((lr, k))
    if eligible:
        return {"value": min(eligible), "rule": "primary", "rule_pass": True}
    if backup:
        backup.sort(key=lambda t: (-t[0], t[1]))
        return {"value": backup[0][1], "rule": "fallback", "rule_pass": False}
    return {"value": None, "rule": "none-satisfied", "rule_pass": False}


def main() -> int:
    all_ok = True
    summaries: Dict[tuple, Dict] = {}

    # 1) presence + parse
    for cfg in CONFIGS:
        for mode in PCA_MODES:
            p = DIR / f"{cfg}_summary_{mode}.json"
            ok = p.exists()
            all_ok &= _check(f"file {p.name} exists", ok)
            if not ok:
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                summaries[(cfg, mode)] = d
                all_ok &= _check(f"  {p.name} parses (n_keys={len(d)})", True)
            except Exception as e:
                all_ok &= _check(f"  {p.name} parses", False, str(e))

    for mode in PCA_MODES:
        p = DIR / f"frontier_{mode}.png"
        all_ok &= _check(f"frontier_{mode}.png exists", p.exists())

    p_kstar = DIR / "kstar_table.json"
    all_ok &= _check("kstar_table.json exists", p_kstar.exists())
    kstar_table = json.loads(p_kstar.read_text(encoding="utf-8")) if p_kstar.exists() else []

    # 2) per-summary structural checks
    for (cfg, mode), d in summaries.items():
        label = f"{cfg}/{mode}"
        per_k = d.get("per_K", {})
        ks_present = sorted([int(k) for k in per_k.keys()], reverse=True)
        all_ok &= _check(f"{label}: 8 Ks present", set(ks_present) == set(EXPECTED_KS),
                         f"got {ks_present}")
        # EVR=1.0 at K=1024
        evr = d.get("explained_variance_ratio_cumsum_at_K", {})
        evr_1024 = evr.get("1024", 0.0)
        all_ok &= _check(f"{label}: EVR(K=1024) approx 1.0", abs(evr_1024 - 1.0) < 1e-6,
                         f"got {evr_1024:.6f}")
        # EVR monotonic
        evr_vals = [evr[str(k)] for k in sorted(EXPECTED_KS)]
        mono = all(evr_vals[i] <= evr_vals[i + 1] for i in range(len(evr_vals) - 1))
        all_ok &= _check(f"{label}: EVR monotonic in K", mono)
        # MMD increases as K decreases
        mmd_seq = [per_k[str(k)]["alignment"]["mmd_median"] for k in EXPECTED_KS]
        # Expected: mmd_seq is ascending when listed [1024,512,...,8]
        ascending = all(mmd_seq[i] <= mmd_seq[i + 1] + 0.02 for i in range(len(mmd_seq) - 1))
        all_ok &= _check(f"{label}: MMD increases as K decreases (AAAI 2020 power)", ascending,
                         f"mmd_seq={[round(v,3) for v in mmd_seq]}")
        # C2ST saturated near 1.0 at K=1024
        c2st_1024 = per_k["1024"]["alignment"]["c2st_auc"]
        all_ok &= _check(f"{label}: C2ST(K=1024) saturated >= 0.99", c2st_1024 >= 0.99,
                         f"got {c2st_1024:.3f}")
        # Re-applied K*
        recomp = reapply_kstar(per_k)
        all_ok &= _check(f"{label}: K* matches re-applied rule",
                         recomp == d["k_star"],
                         f"saved={d['k_star']} recomputed={recomp}")

    # 3) Cross-check kstar_table.json entries align with per-summary k_star
    for row in kstar_table:
        key = (row["config"], row["pca_mode"])
        if key not in summaries:
            all_ok &= _check(f"kstar_table row {key} has matching summary", False)
            continue
        ok = row["k_star"] == summaries[key]["k_star"]
        all_ok &= _check(f"kstar_table {key} matches summary", ok,
                         f"row={row['k_star']} summary={summaries[key]['k_star']}")

    # 4) Headline summary
    print("\n--- Headline numbers (K=1024 baseline -> K=128 vs K=32) ---")
    for cfg in CONFIGS:
        for mode in PCA_MODES:
            d = summaries.get((cfg, mode))
            if d is None:
                continue
            p = d["per_K"]
            row = []
            for k in (1024, 128, 32):
                a = p[str(k)]["alignment"]
                cs = p[str(k)]["class_structure"]
                m = p[str(k)]["mechanism"]
                row.append(
                    f"K={k:>4d}: MMD={a['mmd_median']:.3f}  "
                    f"C2ST={a['c2st_auc']:.3f}  "
                    f"LR={cs['lr_m2p']['macro_auc']:.3f}  "
                    f"phi={m['phi']['circular_r2']:.3f}"
                )
            print(f"  {cfg:<14} {mode:<10}  " + "   |   ".join(row))

    print("\nALL PASS" if all_ok else "\nFAILED -- see above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
