"""Comprehensive Exp 7 evaluation: K-means, cross-domain probes, cosine similarity.

Runs all metrics across multiple experiment configurations for a complete comparison table.
Configurations: Exp 5, Exp 6, Exp 7 baseline, Exp 7 ccmmd.

Usage:
  python analysis/exp7_full_evaluation.py --outdir outputs/exp7_full_evaluation
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LATENT_DIR = REPO_ROOT / "outputs" / "latents"

SHARED_CLASSES = ["NORM", "MI", "CD"]
MEDALCARE_REMAP = {0: 0, 1: 1, 2: 2, 3: 2, 5: 2, 7: 2}
MEDALCARE_KEEP = {0, 1, 2, 3, 5, 7}
PTBXL_REMAP = {0: 0, 1: 1, 4: 2}
PTBXL_KEEP = {0, 1, 4}

CONFIGS = {
    "exp5": {"medal_prefix": "exp5_medalcare", "ptb_prefix": "exp5_ptbxl",
             "label": "Exp 5: Joint+Adapters (dual-head, 8/5-class)"},
    "exp6": {"medal_prefix": "exp6_medalcare", "ptb_prefix": "exp6_ptbxl",
             "label": "Exp 6: Joint+Adapters+MMD (dual-head, 8/5-class)"},
    "exp5_3class": {"medal_prefix": "exp5_3class_medalcare", "ptb_prefix": "exp5_3class_ptbxl",
                    "label": "Exp 5 (3-class redo): Dual-head, no alignment"},
    "exp6_3class": {"medal_prefix": "exp6_3class_medalcare", "ptb_prefix": "exp6_3class_ptbxl",
                    "label": "Exp 6 (3-class redo): Dual-head + ccMMD"},
    "exp7_baseline": {"medal_prefix": "exp7_medalcare", "ptb_prefix": "exp7_ptbxl",
                      "label": "Exp 7: Shared Head (baseline)"},
    "exp7_ccmmd": {"medal_prefix": "exp7_ccmmd_medalcare", "ptb_prefix": "exp7_ccmmd_ptbxl",
                   "label": "Exp 7: Shared Head + ccMMD"},
    "exp7_norm": {"medal_prefix": "exp7_norm_medalcare", "ptb_prefix": "exp7_norm_ptbxl",
                  "label": "Exp 7: Shared Head (norm-fixed)"},
    # Post-INLP aligned variants (latents transformed by analysis/inlp_alignment.py)
    "exp7_baseline_inlp": {
        "medal_prefix": "exp7_medalcare_inlp", "ptb_prefix": "exp7_ptbxl_inlp",
        "label": "Exp 7 baseline + INLP",
    },
    "exp7_ccmmd_inlp": {
        "medal_prefix": "exp7_ccmmd_medalcare_inlp", "ptb_prefix": "exp7_ccmmd_ptbxl_inlp",
        "label": "Exp 7 ccMMD + INLP",
    },
    "exp5_3class_inlp": {
        "medal_prefix": "exp5_3class_medalcare_inlp", "ptb_prefix": "exp5_3class_ptbxl_inlp",
        "label": "Exp 5 3class + INLP (sensitivity)",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive Exp 7 evaluation.")
    p.add_argument("--outdir", type=Path, default=REPO_ROOT / "outputs" / "exp7_full_evaluation")
    p.add_argument("--latent-dir", type=Path, default=LATENT_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--configs", nargs="+", default=None,
        help="Optional subset of CONFIGS keys to evaluate. Default: all.",
    )
    return p.parse_args()


def load_and_remap(latent_dir: Path, subdir: str, domain: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load latents and remap to 3-class shared labels, filtering irrelevant samples."""
    npz = np.load(latent_dir / subdir / "latents.npz")
    Z = npz["Z"]
    Y_orig = npz["Y"]

    if domain == "medalcare":
        remap, keep = MEDALCARE_REMAP, MEDALCARE_KEEP
        n_orig = Y_orig.shape[1]
    else:
        remap, keep = PTBXL_REMAP, PTBXL_KEEP
        n_orig = Y_orig.shape[1]

    valid_cols = [c for c in keep if c < n_orig]
    mask = np.zeros(len(Y_orig), dtype=bool)
    for col in valid_cols:
        mask |= (Y_orig[:, col] > 0.5)

    Z_filt = Z[mask]
    Y_orig_filt = Y_orig[mask]

    Y_shared = np.zeros((len(Z_filt), 3), dtype=np.float32)
    for src, tgt in remap.items():
        if src < n_orig:
            Y_shared[:, tgt] = np.clip(Y_shared[:, tgt] + Y_orig_filt[:, src], 0, 1)

    return Z_filt, Y_shared


def argmax_label(Y: np.ndarray) -> np.ndarray:
    return np.argmax(Y, axis=1)


# ---------------------------------------------------------------------------
# A3: K-means Clustering
# ---------------------------------------------------------------------------

def hungarian_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Clustering accuracy via Hungarian matching."""
    n_classes = max(true_labels.max(), pred_labels.max()) + 1
    cost = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        cost[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-cost)
    return float(cost[row_ind, col_ind].sum()) / len(true_labels)


def kmeans_analysis(Z: np.ndarray, labels: np.ndarray, k: int = 3, seed: int = 42) -> Dict:
    scaler = StandardScaler()
    Z_sc = scaler.fit_transform(Z)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    pred = km.fit_predict(Z_sc)

    acc = hungarian_accuracy(labels, pred)
    nmi = float(normalized_mutual_info_score(labels, pred))
    ari = float(adjusted_rand_score(labels, pred))
    return {"accuracy": acc, "nmi": nmi, "ari": ari}


def run_kmeans_all(configs_data: Dict, seed: int) -> Dict:
    """Run k-means on combined, MedalCare-only, and PTB-XL-only for each config."""
    results = {}
    for name, data in configs_data.items():
        Z_m, Y_m, Z_p, Y_p = data["Z_m"], data["Y_m"], data["Z_p"], data["Y_p"]
        labels_m, labels_p = argmax_label(Y_m), argmax_label(Y_p)

        Z_all = np.vstack([Z_m, Z_p])
        labels_all = np.concatenate([labels_m, labels_p])

        r = {
            "combined": kmeans_analysis(Z_all, labels_all, seed=seed),
            "medalcare": kmeans_analysis(Z_m, labels_m, seed=seed),
            "ptbxl": kmeans_analysis(Z_p, labels_p, seed=seed),
        }
        results[name] = r
        print(f"  {name}:")
        for scope, metrics in r.items():
            print(f"    {scope}: Acc={metrics['accuracy']:.4f}, NMI={metrics['nmi']:.4f}, ARI={metrics['ari']:.4f}")
    return results


# ---------------------------------------------------------------------------
# A4: Cross-Domain Transfer Probes
# ---------------------------------------------------------------------------

def cross_domain_probe(Z_train, labels_train, Z_test, labels_test, seed: int) -> Dict:
    """Train logistic regression on one domain, evaluate on another."""
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Z_train)
    Z_te = scaler.transform(Z_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             multi_class="multinomial", random_state=seed)
    clf.fit(Z_tr, labels_train)
    acc = float(clf.score(Z_te, labels_test))

    proba = clf.predict_proba(Z_te)
    per_class_auc = {}
    for cls_idx, cls_name in enumerate(SHARED_CLASSES):
        binary = (labels_test == cls_idx).astype(int)
        if binary.sum() > 0 and (1 - binary).sum() > 0:
            per_class_auc[cls_name] = float(roc_auc_score(binary, proba[:, cls_idx]))

    macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else 0.0
    return {"accuracy": acc, "macro_auc": macro_auc, "per_class_auc": per_class_auc}


def cross_domain_knn(Z_train, labels_train, Z_test, labels_test, k: int = 5) -> float:
    """kNN classifier trained on one domain, evaluated on other."""
    scaler = StandardScaler()
    Z_tr = scaler.fit_transform(Z_train)
    Z_te = scaler.transform(Z_test)
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(Z_tr, labels_train)
    return float(knn.score(Z_te, labels_test))


def run_cross_domain_all(configs_data: Dict, seed: int) -> Dict:
    results = {}
    for name, data in configs_data.items():
        Z_m, Y_m, Z_p, Y_p = data["Z_m"], data["Y_m"], data["Z_p"], data["Y_p"]
        labels_m, labels_p = argmax_label(Y_m), argmax_label(Y_p)

        medal_to_ptb = cross_domain_probe(Z_m, labels_m, Z_p, labels_p, seed)
        ptb_to_medal = cross_domain_probe(Z_p, labels_p, Z_m, labels_m, seed)
        knn5_m2p = cross_domain_knn(Z_m, labels_m, Z_p, labels_p, k=5)
        knn5_p2m = cross_domain_knn(Z_p, labels_p, Z_m, labels_m, k=5)
        knn15_m2p = cross_domain_knn(Z_m, labels_m, Z_p, labels_p, k=15)
        knn15_p2m = cross_domain_knn(Z_p, labels_p, Z_m, labels_m, k=15)

        r = {
            "logistic_medal_to_ptb": medal_to_ptb,
            "logistic_ptb_to_medal": ptb_to_medal,
            "knn5_medal_to_ptb": knn5_m2p,
            "knn5_ptb_to_medal": knn5_p2m,
            "knn15_medal_to_ptb": knn15_m2p,
            "knn15_ptb_to_medal": knn15_p2m,
        }
        results[name] = r
        print(f"  {name}:")
        print(f"    LR medal→ptb: acc={medal_to_ptb['accuracy']:.4f}, AUC={medal_to_ptb['macro_auc']:.4f}")
        print(f"    LR ptb→medal: acc={ptb_to_medal['accuracy']:.4f}, AUC={ptb_to_medal['macro_auc']:.4f}")
        print(f"    kNN-5 m→p={knn5_m2p:.4f}, p→m={knn5_p2m:.4f}")
        print(f"    kNN-15 m→p={knn15_m2p:.4f}, p→m={knn15_p2m:.4f}")
    return results


# ---------------------------------------------------------------------------
# A5: Cosine Similarity
# ---------------------------------------------------------------------------

def pairwise_cosine_sim(X: np.ndarray, Y: np.ndarray, n_pairs: int = 5000,
                        seed: int = 42) -> float:
    """Average cosine similarity between random pairs from X and Y."""
    rng = np.random.RandomState(seed)
    n_pairs = min(n_pairs, len(X) * len(Y))
    idx_x = rng.randint(0, len(X), n_pairs)
    idx_y = rng.randint(0, len(Y), n_pairs)
    sims = []
    for i, j in zip(idx_x, idx_y):
        sim = 1.0 - cosine_dist(X[i], Y[j])
        sims.append(sim)
    return float(np.mean(sims))


def run_cosine_all(configs_data: Dict, seed: int) -> Dict:
    results = {}
    for name, data in configs_data.items():
        Z_m, Y_m, Z_p, Y_p = data["Z_m"], data["Y_m"], data["Z_p"], data["Y_p"]
        labels_m, labels_p = argmax_label(Y_m), argmax_label(Y_p)

        scaler = StandardScaler()
        Z_all = np.vstack([Z_m, Z_p])
        scaler.fit(Z_all)
        Zm = scaler.transform(Z_m)
        Zp = scaler.transform(Z_p)

        r = {"intra_class": {}, "inter_class": {}, "cross_domain": {}}

        for cls_idx, cls_name in enumerate(SHARED_CLASSES):
            m_cls = Zm[labels_m == cls_idx]
            p_cls = Zp[labels_p == cls_idx]

            if len(m_cls) >= 2:
                r["intra_class"][f"{cls_name}_medal"] = pairwise_cosine_sim(m_cls, m_cls, seed=seed)
            if len(p_cls) >= 2:
                r["intra_class"][f"{cls_name}_ptb"] = pairwise_cosine_sim(p_cls, p_cls, seed=seed)
            if len(m_cls) >= 1 and len(p_cls) >= 1:
                r["cross_domain"][cls_name] = pairwise_cosine_sim(m_cls, p_cls, seed=seed)

        for i, cls_a in enumerate(SHARED_CLASSES):
            for j, cls_b in enumerate(SHARED_CLASSES):
                if i >= j:
                    continue
                m_a = Zm[labels_m == i]
                m_b = Zm[labels_m == j]
                p_a = Zp[labels_p == i]
                p_b = Zp[labels_p == j]
                if len(m_a) >= 1 and len(m_b) >= 1:
                    r["inter_class"][f"{cls_a}_vs_{cls_b}_medal"] = pairwise_cosine_sim(m_a, m_b, seed=seed)
                if len(p_a) >= 1 and len(p_b) >= 1:
                    r["inter_class"][f"{cls_a}_vs_{cls_b}_ptb"] = pairwise_cosine_sim(p_a, p_b, seed=seed)

        avg_intra = np.mean(list(r["intra_class"].values())) if r["intra_class"] else 0
        avg_inter = np.mean(list(r["inter_class"].values())) if r["inter_class"] else 0
        avg_cross = np.mean(list(r["cross_domain"].values())) if r["cross_domain"] else 0
        r["summary"] = {
            "avg_intra_class": float(avg_intra),
            "avg_inter_class": float(avg_inter),
            "avg_cross_domain_same_class": float(avg_cross),
            "intra_inter_gap": float(avg_intra - avg_inter),
        }

        results[name] = r
        print(f"  {name}:")
        print(f"    Avg intra-class: {avg_intra:.4f}")
        print(f"    Avg inter-class: {avg_inter:.4f}")
        print(f"    Avg cross-domain (same class): {avg_cross:.4f}")
        print(f"    Intra-inter gap: {avg_intra - avg_inter:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.outdir}\n")

    configs_data = {}
    if args.configs is not None:
        unknown = [c for c in args.configs if c not in CONFIGS]
        if unknown:
            raise SystemExit(f"Unknown --configs entries: {unknown}. Known: {list(CONFIGS)}")
        config_items = [(k, CONFIGS[k]) for k in args.configs]
    else:
        config_items = list(CONFIGS.items())
    for config_name, cfg in config_items:
        medal_path = args.latent_dir / cfg["medal_prefix"] / "latents.npz"
        ptb_path = args.latent_dir / cfg["ptb_prefix"] / "latents.npz"
        if not medal_path.exists() or not ptb_path.exists():
            print(f"  [SKIP] {config_name}: missing latent files")
            continue

        Z_m, Y_m = load_and_remap(args.latent_dir, cfg["medal_prefix"], "medalcare")
        Z_p, Y_p = load_and_remap(args.latent_dir, cfg["ptb_prefix"], "ptbxl")
        configs_data[config_name] = {"Z_m": Z_m, "Y_m": Y_m, "Z_p": Z_p, "Y_p": Y_p,
                                     "label": cfg["label"]}
        lm, lp = argmax_label(Y_m), argmax_label(Y_p)
        print(f"  Loaded {config_name}: Medal={Z_m.shape[0]} ({dict(zip(SHARED_CLASSES, [(lm==i).sum() for i in range(3)]))}), "
              f"PTB={Z_p.shape[0]} ({dict(zip(SHARED_CLASSES, [(lp==i).sum() for i in range(3)]))})")

    if not configs_data:
        print("No configurations loaded. Exiting.")
        return

    print("\n" + "=" * 60)
    print("A3: K-MEANS CLUSTERING (k=3)")
    print("=" * 60)
    kmeans_results = run_kmeans_all(configs_data, args.seed)

    print("\n" + "=" * 60)
    print("A4: CROSS-DOMAIN TRANSFER PROBES")
    print("=" * 60)
    cross_domain_results = run_cross_domain_all(configs_data, args.seed)

    print("\n" + "=" * 60)
    print("A5: COSINE SIMILARITY ANALYSIS")
    print("=" * 60)
    cosine_results = run_cosine_all(configs_data, args.seed)

    report = {
        "configs": {k: v["label"] for k, v in configs_data.items()},
        "kmeans": kmeans_results,
        "cross_domain_transfer": cross_domain_results,
        "cosine_similarity": cosine_results,
    }

    report_path = args.outdir / "full_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full evaluation report to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
