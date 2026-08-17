"""Re-extract every Phase-B2 paired latent-vs-control number straight from the
run JSONs, so report tables can be checked against artifacts rather than notes.

Prints one row per (cell, endpoint) with the two macro-F1s, the paired delta,
its bootstrap CI, and both paired p-values. Read-only.

    python scripts/_audit_paired_grid.py
"""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# (label, run directory).  The baseline/target/fold-10 paired stats live in
# phase_b2_smoke_paired/ -- that run produced the first paired statistic, the
# directory name notwithstanding.
CELLS = [
    ("target / baseline / fold10", "outputs/phase_b2_smoke_paired"),
    ("target / baseline / fold10 (5cfg)", "outputs/phase_b2_exp8_spatial54"),
    ("target / medalonly / fold10", "outputs/phase_b2_medalonly_fold10_target"),
    ("target / medalonly / allfolds", "outputs/phase_b2_medalonly_allfolds_target"),
    ("strict / baseline / fold10 (5cfg)", "outputs/phase_b2_exp8_spatial54_measscaler"),
    ("strict / baseline / fold10 (paired)", "outputs/phase_b2_baseline_fold10_measscaler_paired"),
    ("strict / medalonly / fold10", "outputs/phase_b2_medalonly_fold10_measscaler"),
    ("strict / medalonly / allfolds", "outputs/phase_b2_medalonly_allfolds_target_pool_measured"),
]
ENDPOINTS = ["cross_domain_4c", "cross_domain_2c", "in_domain_4c"]


def _f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "--"


def main() -> None:
    hdr = (f"{'cell':36s} {'cfg':24s} {'endpoint':16s} {'Z':>7s} {'ctrl':>7s} "
           f"{'delta':>8s} {'CI95':>20s} {'p_swap':>8s} {'p_boot':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for label, rel in CELLS:
        path = REPO_ROOT / rel / "cross_domain_4c_pipelineA.json"
        if not path.exists():
            print(f"{label:36s} [missing: {rel}]")
            continue
        blob = json.loads(path.read_text(encoding="utf-8"))
        n_ptbxl = blob["metadata"].get("n_ptbxl_primary_4c")
        for cfg, res in blob["results"].items():
            if not isinstance(res, dict):
                continue
            paired = res.get("paired_Z_vs_features") or {}
            for endpoint in ENDPOINTS:
                z = res.get("Z", {}).get(endpoint, {}).get("macro_f1")
                ctrl = res.get("ecg_features", {}).get(endpoint, {}).get("macro_f1")
                if z is None:
                    continue
                p = paired.get(endpoint) or {}
                ci = p.get("delta_ci95") or [None, None]
                ci_s = (f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
                        if isinstance(ci[0], (int, float)) else "--")
                print(f"{label:36s} {cfg:24s} {endpoint:16s} {_f(z):>7s} {_f(ctrl):>7s} "
                      f"{_f(p.get('delta_macro_f1_a_minus_b')):>8s} {ci_s:>20s} "
                      f"{_f(p.get('p_two_sided_paired_swap')):>8s} "
                      f"{_f(p.get('p_a_beats_b_bootstrap')):>8s}")
        print(f"{'':36s} (n_ptbxl_primary_4c = {n_ptbxl})")


if __name__ == "__main__":
    main()
