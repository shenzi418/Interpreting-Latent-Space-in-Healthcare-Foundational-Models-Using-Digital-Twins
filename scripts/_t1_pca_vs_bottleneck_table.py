"""One-shot helper: print PCA-at-K and Bottleneck-at-K side by side for the Track 1 report."""

from __future__ import annotations

import json
from pathlib import Path

PCA_PATH = Path("outputs/dim_scan/exp7_baseline_summary_combined.json")
BOTTLE_DIR = Path("outputs/tier1_eval")
KS = ["1024", "256", "64", "16"]


def _safe(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> None:
    pca = json.load(PCA_PATH.open())

    print("=" * 100)
    print("PCA-at-K (post-hoc on exp7_baseline, combined fit)")
    print("=" * 100)
    print(f"{'K':>6} {'MMDm':>8} {'MMDmb':>8} {'C2ST':>8} {'LR_M2P':>8} {'LR_P2M':>8} {'phi_R2c':>9} {'z_R2':>7}")
    for k in KS:
        a = pca["per_K"][k]["alignment"]
        c = pca["per_K"][k]["class_structure"]
        m = pca["per_K"][k]["mechanism"]
        print(
            f"{k:>6} {a['mmd_median']:8.4f} {a['mmd_multibw']:8.4f} {a['c2st_auc']:8.4f} "
            f"{c['lr_m2p']['macro_auc']:8.4f} {c['lr_p2m']['macro_auc']:8.4f} "
            f"{m['phi']['circular_r2']:9.4f} {m['z']['r2']:7.4f}"
        )

    print()
    print("=" * 100)
    print("Bottleneck-at-K (head-only training, exp7_baseline ref for K=1024)")
    print("=" * 100)
    print(f"{'K':>6} {'F1':>7} {'MMDm':>8} {'MMDmb':>8} {'C2ST':>8} {'LR_M2P':>8} {'LR_P2M':>8} {'phi_R2c':>9} {'z_R2':>7} {'4c_anat':>8}")
    cfg_map = {
        "1024": BOTTLE_DIR / "exp7_baseline_ref_summary.json",
        "256": BOTTLE_DIR / "exp7_bottleneck_K256_summary.json",
        "64": BOTTLE_DIR / "exp7_bottleneck_K64_summary.json",
        "16": BOTTLE_DIR / "exp7_bottleneck_K16_summary.json",
    }
    for k, path in cfg_map.items():
        if not path.exists():
            print(f"{k:>6}  [missing {path.name}]")
            continue
        d = json.load(path.open())
        f1 = _safe(d, "classification", "macro_f1", default=float("nan"))
        a = d["alignment"]
        cs = d["class_structure"]
        m = d["mechanism"]
        an = _safe(d, "anatomy_pipeline_A", "p2m_macro_f1", default=float("nan"))
        print(
            f"{k:>6} {f1:7.4f} {a['mmd_median']:8.4f} {a['mmd_multibw']:8.4f} {a['c2st_auc']:8.4f} "
            f"{cs['lr_m2p']['macro_auc']:8.4f} {cs['lr_p2m']['macro_auc']:8.4f} "
            f"{m['phi']['circular_r2']:9.4f} {m['z']['r2']:7.4f} {an:8.4f}"
        )


if __name__ == "__main__":
    main()
