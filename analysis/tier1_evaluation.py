"""TRACK 1b' / Tier 1 evaluation: alignment + class structure + mechanism + 4c anatomy.

Runs the same 4-block evaluation suite that dim_scan.py uses (alignment,
class structure, mechanism) PLUS the Pipeline-A 4-class anatomy classifier
from phase_b2_infarct_decoding.py, on the K-d latents produced by Tier 1's
bottleneck training. Compares K ∈ {1024 (exp7_baseline reference),
256, 64, 16} on a single common table.

Blocks
------
1. Alignment (test pool, subsampled 2000/domain):
   - MMD median-bandwidth, MMD multi-bandwidth
   - C2ST AUROC, kNN-5 mixing

2. Class structure (test pool, 3-class shared remap):
   - KMeans-3 (combined / medalcare / ptbxl) Acc/NMI/ARI
   - LR M→P + LR P→M (macro-AUC, per-class AUC, accuracy)
   - kNN-5 M→P, kNN-5 P→M
   - Cosine intra/inter/cross gaps

3. Mechanism (MedalCare MI subset, in-domain):
   - phi circular R^2 (sin/cos Ridge)
   - z R^2, size R^2 (Ridge)
   - rho_eps_max AUC (LogReg)

4. Anatomy (Pipeline A, 4-class coronary territory):
   - In-domain on MedalCare-test (n≈1.2k)
   - Cross-domain on PTB-XL primary 4c subset (n≈438)
   - 4-class macro-F1 + balanced acc; cross-domain also 2-class collapsed

Outputs
-------
- outputs/tier1_eval/<config>_summary.json     (per-config full block)
- outputs/tier1_eval/cross_config_table.json   (compact comparison table)
- outputs/tier1_eval/cross_config_table.md     (markdown rendering)
- outputs/tier1_eval/frontier_tier1.png        (4-panel frontier figure)

Usage
-----
    python analysis/tier1_evaluation.py
        [--configs exp7_baseline_ref exp7_bottleneck_K256 ...]
        [--out outputs/tier1_eval]
        [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dim_scan import (  # noqa: E402
    SCAN_N_BOOT,
    SCAN_N_PERM,
    SCAN_N_PERM_BINARY,
    alignment_block,
    class_structure_block,
    mechanism_block,
    remap_test_split,
)
from analysis.phase_b2_infarct_decoding import (  # noqa: E402
    N_BOOT,
    N_PERM_BINARY,
    SEED,
    TERRITORIES_4C,
    TERRITORIES_2C,
    TERRITORY_4C_TO_2C,
    PTBXL_SUBCLASS_PATH,
    load_targets,
    pipeline_a_for_source,
)


# ---------------------------------------------------------------------------
# Per-config latent paths.
# exp7_baseline_ref uses the legacy 1024-d backbone latents (Linear(1024,3) head).
# exp7_bottleneck_K{K} uses our Tier-1 trained 2-layer bottleneck head.
# ---------------------------------------------------------------------------

LATENT_ROOT = REPO_ROOT / "outputs" / "latents"


def _legacy_paths(stem: str) -> Dict[str, Path]:
    """exp7_baseline-style paths: test has no '_test' suffix."""
    return {
        "medal_train": LATENT_ROOT / f"{stem}_medalcare_train" / "latents.npz",
        "medal_test":  LATENT_ROOT / f"{stem}_medalcare"        / "latents.npz",
        "ptb_train":   LATENT_ROOT / f"{stem}_ptbxl_train"      / "latents.npz",
        "ptb_test":    LATENT_ROOT / f"{stem}_ptbxl"            / "latents.npz",
    }


def _bottleneck_paths(stem: str) -> Dict[str, Path]:
    """exp7_bottleneck_K{K}-style paths: explicit '_test' suffix everywhere."""
    return {
        "medal_train": LATENT_ROOT / f"{stem}_medalcare_train" / "latents.npz",
        "medal_test":  LATENT_ROOT / f"{stem}_medalcare_test"  / "latents.npz",
        "ptb_train":   LATENT_ROOT / f"{stem}_ptbxl_train"     / "latents.npz",
        "ptb_test":    LATENT_ROOT / f"{stem}_ptbxl_test"      / "latents.npz",
    }


# Canonical (config_label -> path_dict) used by --configs and the main loop.
DEFAULT_CONFIGS: Tuple[str, ...] = (
    "exp7_baseline_ref",
    "exp7_bottleneck_K256",
    "exp7_bottleneck_K64",
    "exp7_bottleneck_K16",
)
CONFIG_LATENT_PATHS: Dict[str, Dict[str, Path]] = {
    "exp7_baseline_ref":    _legacy_paths("exp7"),
    "exp7_bottleneck_K256": _bottleneck_paths("exp7_bottleneck_K256"),
    "exp7_bottleneck_K64":  _bottleneck_paths("exp7_bottleneck_K64"),
    "exp7_bottleneck_K16":  _bottleneck_paths("exp7_bottleneck_K16"),
}
# Nominal K for table labelling (also used to detect the K=1024 reference row).
CONFIG_K: Dict[str, int] = {
    "exp7_baseline_ref":    1024,
    "exp7_bottleneck_K256": 256,
    "exp7_bottleneck_K64":  64,
    "exp7_bottleneck_K16":  16,
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.keys()}


def load_config_4splits(config: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {split -> {'Z', 'Y'}} for medal_train/test + ptb_train/test."""
    paths = CONFIG_LATENT_PATHS[config]
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"missing latents for config {config!r}: {path}")
        d = _load_npz(path)
        out[name] = {"Z": d["Z"].astype(np.float64), "Y": d["Y"].astype(np.float32)}
    return out


def load_ptbxl_4c_subset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PTB-XL primary 4c subset indices, 4c labels, and 2c labels.

    Indices are positions into the PTB-XL TEST latent array (row_idx column).
    """
    df = pd.read_csv(PTBXL_SUBCLASS_PATH)
    mask = df["territory_4c"].isin(TERRITORIES_4C)
    sub = df[mask].copy()
    idx = sub["row_idx"].to_numpy()
    y4 = sub["territory_4c"].to_numpy()
    y2 = sub["territory_2c"].to_numpy()
    return idx, y4, y2


# ---------------------------------------------------------------------------
# Per-config eval
# ---------------------------------------------------------------------------

def evaluate_one_config(
    config: str,
    splits: Dict[str, Dict[str, np.ndarray]],
    targets: Dict[str, Dict[str, np.ndarray]],
    ptb_4c_idx: np.ndarray,
    ptb_4c_y4: np.ndarray,
    ptb_4c_y2: np.ndarray,
    *,
    seed: int,
    anatomy_n_boot: int,
    anatomy_n_perm: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    Z_m_tr = splits["medal_train"]["Z"]
    Z_m_te = splits["medal_test"]["Z"]
    Z_p_tr = splits["ptb_train"]["Z"]
    Z_p_te = splits["ptb_test"]["Z"]
    Y_m_te = splits["medal_test"]["Y"]
    Y_p_te = splits["ptb_test"]["Y"]
    K = int(Z_m_tr.shape[1])
    print(f"\n[{config}] K={K}  shapes  Z_m_tr={Z_m_tr.shape}  Z_p_tr={Z_p_tr.shape}")

    # --- (1) Alignment + (2) Class structure on shared-3 valid test rows ---
    mask_m, Y_m_shared = remap_test_split(Y_m_te, "medalcare")
    mask_p, Y_p_shared = remap_test_split(Y_p_te, "ptbxl")
    Z_m_te_v = Z_m_te[mask_m]
    Z_p_te_v = Z_p_te[mask_p]
    Y_m_te_v = Y_m_shared[mask_m]
    Y_p_te_v = Y_p_shared[mask_p]
    print(
        f"[{config}] shared3 valid: medal {int(mask_m.sum())}/{len(mask_m)}, "
        f"ptb {int(mask_p.sum())}/{len(mask_p)}"
    )

    t0 = time.time()
    align = alignment_block(Z_m_te_v, Z_p_te_v, rng=rng, seed=seed)
    print(
        f"[{config}] alignment OK ({time.time()-t0:.1f}s)  "
        f"MMDm={align['mmd_median']:.4f}  MMDmb={align['mmd_multibw']:.4f}  "
        f"C2ST={align['c2st_auc']:.3f}  kNN-mix={align['knn_mixing']:.3f}"
    )

    t0 = time.time()
    classes = class_structure_block(Z_m_te_v, Y_m_te_v, Z_p_te_v, Y_p_te_v, seed=seed)
    lr_m2p = classes["lr_m2p"]["macro_auc"]
    lr_p2m = classes["lr_p2m"]["macro_auc"]
    km_comb = classes["kmeans"]["combined"]["accuracy"]
    print(
        f"[{config}] class struct OK ({time.time()-t0:.1f}s)  "
        f"LR M→P={lr_m2p:.3f}  LR P→M={lr_p2m:.3f}  KMeans-comb={km_comb:.3f}"
    )

    # --- (3) Mechanism on MedalCare MI subset (in-domain) ---
    idx_train = targets["train"]["idx_in_split"]
    idx_test = targets["test"]["idx_in_split"]
    Z_m_tr_mi = Z_m_tr[idx_train]
    Z_m_te_mi = Z_m_te[idx_test]
    print(f"[{config}] MI subset  train={idx_train.size}  test={idx_test.size}")
    t0 = time.time()
    mech = mechanism_block(Z_m_tr_mi, Z_m_te_mi, targets, rng)
    print(
        f"[{config}] mechanism OK ({time.time()-t0:.1f}s)  "
        f"phi_R2c={mech['phi']['circular_r2']:.3f}  z_R2={mech['z']['r2']:.3f}  "
        f"size_R2={mech['size']['r2']:.3f}  rho_AUC={mech['rho_eps_max']['auc']:.3f}"
    )

    # --- (4) Anatomy: 4c territory classifier (Pipeline A) ---
    terr4c_train = np.array(targets["train"]["territory_4c"].tolist(), dtype=object)
    terr4c_test = np.array(targets["test"]["territory_4c"].tolist(), dtype=object)
    # Standardize on MedalCare MI train subset.
    z_scaler = StandardScaler().fit(Z_m_tr_mi)
    X_m_tr_std = z_scaler.transform(Z_m_tr_mi)
    X_m_te_std = z_scaler.transform(Z_m_te_mi)
    # PTB-XL primary 4c subset (cross-domain).
    Z_p_te_4c = Z_p_te[ptb_4c_idx]
    X_p_te_std = z_scaler.transform(Z_p_te_4c)
    print(
        f"[{config}] anatomy  MedalCare-train MI={X_m_tr_std.shape[0]}, "
        f"MedalCare-test MI={X_m_te_std.shape[0]}, "
        f"PTB-XL primary 4c={X_p_te_std.shape[0]}"
    )
    t0 = time.time()
    pipeline_a = pipeline_a_for_source(
        src_name=f"Z[{K}d]",
        X_train_std=X_m_tr_std,
        X_test_std=X_m_te_std,
        X_ptbxl_std=X_p_te_std,
        y_train_4c=terr4c_train,
        y_test_4c=terr4c_test,
        y_ptbxl_4c=ptb_4c_y4,
        rng=rng, n_boot=anatomy_n_boot, n_perm=anatomy_n_perm,
    )
    print(
        f"[{config}] anatomy OK ({time.time()-t0:.1f}s)  "
        f"in-dom 4c macroF1={pipeline_a['in_domain_4c']['macro_f1']:.3f}  "
        f"cross 4c macroF1={pipeline_a['cross_domain_4c']['macro_f1']:.3f}  "
        f"cross 2c macroF1={pipeline_a['cross_domain_2c']['macro_f1']:.3f}"
    )

    return {
        "config": config,
        "K": K,
        "shapes": {
            "medal_train_full": list(Z_m_tr.shape),
            "medal_test_full":  list(Z_m_te.shape),
            "ptb_train_full":   list(Z_p_tr.shape),
            "ptb_test_full":    list(Z_p_te.shape),
            "medal_test_valid": int(mask_m.sum()),
            "ptb_test_valid":   int(mask_p.sum()),
            "medal_train_mi":   int(idx_train.size),
            "medal_test_mi":    int(idx_test.size),
            "ptb_test_4c":      int(ptb_4c_idx.size),
        },
        "alignment": align,
        "class_structure": classes,
        "mechanism": mech,
        "anatomy_pipeline_a": pipeline_a,
    }


# ---------------------------------------------------------------------------
# Cross-config table + figure
# ---------------------------------------------------------------------------

_KEY_LIST: List[Tuple[str, str]] = [
    ("MMD_median",   "alignment.mmd_median"),
    ("MMD_multibw",  "alignment.mmd_multibw"),
    ("C2ST_AUC",     "alignment.c2st_auc"),
    ("kNN_mix",      "alignment.knn_mixing"),
    ("KMeans_comb_acc", "class_structure.kmeans.combined.accuracy"),
    ("LR_M2P_AUC",   "class_structure.lr_m2p.macro_auc"),
    ("LR_P2M_AUC",   "class_structure.lr_p2m.macro_auc"),
    ("kNN5_M2P",     "class_structure.knn5_m2p"),
    ("kNN5_P2M",     "class_structure.knn5_p2m"),
    ("cos_intra",    "class_structure.cosine_summary.avg_intra_class"),
    ("cos_inter",    "class_structure.cosine_summary.avg_inter_class"),
    ("cos_cross",    "class_structure.cosine_summary.avg_cross_domain_same_class"),
    ("phi_R2_circ",  "mechanism.phi.circular_r2"),
    ("z_R2",         "mechanism.z.r2"),
    ("size_R2",      "mechanism.size.r2"),
    ("rho_AUC",      "mechanism.rho_eps_max.auc"),
    ("anat_ind_F1_4c",  "anatomy_pipeline_a.in_domain_4c.macro_f1"),
    ("anat_ind_bal",    "anatomy_pipeline_a.in_domain_4c.balanced_accuracy"),
    ("anat_cd_F1_4c",   "anatomy_pipeline_a.cross_domain_4c.macro_f1"),
    ("anat_cd_F1_2c",   "anatomy_pipeline_a.cross_domain_2c.macro_f1"),
]


def _json_default(o: object) -> object:
    """JSON encoder fallback for numpy types (ndarrays, scalars)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _get_deep(d: Dict[str, object], dotted: str) -> Optional[float]:
    node: object = d
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, (int, float)):
        return float(node)
    return None


def build_cross_table(results: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    ordered = [c for c in DEFAULT_CONFIGS if c in results]
    # Append any extra configs (passed via --configs) at the end.
    for c in results:
        if c not in ordered:
            ordered.append(c)
    for cfg in ordered:
        r = results[cfg]
        row: Dict[str, object] = {"config": cfg, "K": CONFIG_K.get(cfg, r.get("K"))}
        for short, dotted in _KEY_LIST:
            row[short] = _get_deep(r, dotted)
        rows.append(row)
    return {"rows": rows, "columns": ["config", "K"] + [k for k, _ in _KEY_LIST]}


def render_markdown_table(table: Dict[str, object]) -> str:
    cols: List[str] = table["columns"]  # type: ignore[assignment]
    rows: List[Dict[str, object]] = table["rows"]  # type: ignore[assignment]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    out_lines = [header, sep]
    for r in rows:
        cells: List[str] = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("—")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                cells.append(str(v))
        out_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(out_lines)


def render_frontier_figure(
    results: Dict[str, Dict[str, object]], out_path: Path,
) -> None:
    """4-panel frontier:  alignment / class / mechanism / anatomy  vs K."""
    panels = [
        ("alignment.c2st_auc",                "C2ST AUROC (synth vs real)", "linear"),
        ("class_structure.lr_m2p.macro_auc",  "LR M→P macro-AUC (class)",   "linear"),
        ("mechanism.phi.circular_r2",         "phi circular R² (mechanism)", "linear"),
        ("anatomy_pipeline_a.cross_domain_2c.macro_f1",
                                              "anatomy cross-domain 2c F1",  "linear"),
    ]
    ordered = [c for c in DEFAULT_CONFIGS if c in results]
    ks = [CONFIG_K[c] for c in ordered]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, (key, label, yscale) in zip(axes, panels):
        ys = [_get_deep(results[c], key) for c in ordered]
        valid = [(k, y) for k, y in zip(ks, ys) if y is not None]
        if not valid:
            ax.set_visible(False)
            continue
        kk, yy = zip(*sorted(valid))
        ax.plot(kk, yy, marker="o", color="#1f77b4")
        ax.set_xscale("log", base=2)
        ax.set_xticks([16, 64, 256, 1024])
        ax.set_xticklabels(["16", "64", "256", "1024"])
        ax.set_xlabel("K (latent dim, log scale)")
        ax.set_ylabel(label)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Tier 1 trained-bottleneck frontier  (4-panel)", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier 1 bottleneck evaluation suite.")
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS),
                   help=f"Configs to evaluate. Default: {DEFAULT_CONFIGS}")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "tier1_eval",
                   help="Output directory for JSON / PNG.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--anatomy-n-boot", type=int, default=N_BOOT,
                   help="Bootstrap resamples for anatomy CIs (default 1000).")
    p.add_argument("--anatomy-n-perm", type=int, default=N_PERM_BINARY,
                   help="Permutations for anatomy p-values (default 200).")
    p.add_argument("--quick", action="store_true",
                   help="Reduce bootstrap+permutation budgets for a fast sanity run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.quick:
        anatomy_n_boot = 100
        anatomy_n_perm = 30
    else:
        anatomy_n_boot = args.anatomy_n_boot
        anatomy_n_perm = args.anatomy_n_perm
    print(f"[setup] out={out.relative_to(REPO_ROOT)}  seed={args.seed}  "
          f"alignment N_BOOT={SCAN_N_BOOT} N_PERM={SCAN_N_PERM}  "
          f"anatomy N_BOOT={anatomy_n_boot} N_PERM={anatomy_n_perm}")
    print(f"[setup] configs: {args.configs}")

    # Load shared targets ONCE.
    targets = load_targets()
    ptb_idx, ptb_y4, ptb_y2 = load_ptbxl_4c_subset()
    print(f"[setup] PTB-XL primary 4c subset: n={ptb_idx.size}")

    results: Dict[str, Dict[str, object]] = {}
    for cfg in args.configs:
        if cfg not in CONFIG_LATENT_PATHS:
            raise ValueError(
                f"unknown config {cfg!r}; known: {sorted(CONFIG_LATENT_PATHS)}"
            )
        splits = load_config_4splits(cfg)
        cfg_out = evaluate_one_config(
            cfg, splits, targets, ptb_idx, ptb_y4, ptb_y2,
            seed=args.seed,
            anatomy_n_boot=anatomy_n_boot,
            anatomy_n_perm=anatomy_n_perm,
        )
        json_path = out / f"{cfg}_summary.json"
        json_path.write_text(
            json.dumps(cfg_out, indent=2, default=_json_default),
            encoding="utf-8",
        )
        print(f"[save] {json_path.relative_to(REPO_ROOT)}")
        results[cfg] = cfg_out

    # Cross-config table + figure.
    table = build_cross_table(results)
    (out / "cross_config_table.json").write_text(
        json.dumps(table, indent=2, default=_json_default), encoding="utf-8",
    )
    (out / "cross_config_table.md").write_text(
        render_markdown_table(table), encoding="utf-8",
    )
    render_frontier_figure(results, out / "frontier_tier1.png")
    print(f"\n[done] cross-config table -> {(out / 'cross_config_table.md').relative_to(REPO_ROOT)}")
    print(f"[done] frontier figure   -> {(out / 'frontier_tier1.png').relative_to(REPO_ROOT)}")

    # Print final compact summary to stdout.
    print("\n" + render_markdown_table(table))


if __name__ == "__main__":
    main()
