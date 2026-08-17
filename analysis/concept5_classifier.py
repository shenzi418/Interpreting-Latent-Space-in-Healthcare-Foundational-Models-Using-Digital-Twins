"""Task 2a — Five-concept-vector classifier (synth->real interpretable bridge).

Hypothesis
----------
The ECGFounder K=1024 latent decodes four MedalCare biophysical isch-zone
parameters (phi, z, size, transmurality) with high in-domain R² but
near-zero direct latent-space alignment to PTB-XL (C2ST≈1). If those
predicted concepts are *clinically meaningful* — i.e. PTB-XL ECGs of the
same true territory produce similar (phi, z, size, transmurality) vectors
as MedalCare ECGs of that territory — then a tiny logistic-regression
classifier trained on the 5-dim concept vector from MedalCare should
generalise to PTB-XL.

This script:

1. Loads the K=1024 baseline latents (``exp7_*``) for MedalCare-train MI,
   MedalCare-test MI, PTB-XL primary-4c subset.
2. Trains four probes on the MedalCare-train MI rows:
     - phi probe   : RidgeCV on (sin, cos)
     - z probe     : RidgeCV (scalar)
     - size probe  : RidgeCV (scalar)
     - trans probe : LogisticRegression (binary)
3. Applies all four probes to each of {med_train, med_test, ptbxl_primary}
   and assembles the 5-vector (phi_sin, phi_cos, z_hat, size_hat,
   trans_logit). The transmurality channel is the LOG-ODDS (decision
   function) so the classifier can compare margins linearly.
4. Saves the predicted concept matrices to
   ``outputs/inlp_lowK/concept5_predictions.npz`` for downstream reuse.
5. Audits concept distribution per PTB-XL territory_4c — medians + IQR.
6. Trains a multinomial LogReg on the 5-vector from MedalCare-train,
   evaluates in-domain (MedalCare-test MI rows) and cross-domain
   (PTB-XL primary 4c, n=438).
7. Sensitivity: (a) drop-one-feature ablations, (b) MLP variant
   (sklearn MLPClassifier), (c) GT-theta upper bound (use the real
   MedalCare phi/z/size/trans values directly, not predicted).
8. Marta wants macro-OvR ROC-AUC alongside F1 — both reported with
   1000-bootstrap CI and 1000-permutation p.

Outputs
-------
outputs/inlp_lowK/concept5_classifier.json   — all metrics
outputs/inlp_lowK/concept5_predictions.npz   — predicted 5-vec matrices
outputs/inlp_lowK/concept5_ptbxl_per_territory.csv — audit table

Run::

    python analysis/concept5_classifier.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    r2_score,
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Pull L3 helpers from the sibling eval script
from analysis.eval_decoding_lowK import (  # noqa: E402
    score_block,
    macro_ovr_auc,
    load_Z,
    load_theta_targets,
    load_ptbxl_primary_4c,
    TERRITORIES_4C,
    SEED, N_BOOT, N_PERM,
    RIDGE_ALPHAS, LOGREG_CS_BIN, LOGREG_CS_TERR,
    derive_rng,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUT_DIR = REPO_ROOT / "outputs" / "inlp_lowK"
CONCEPT_NAMES = ["phi_sin", "phi_cos", "z_hat", "size_hat", "trans_logit"]


# ---------------------------------------------------------------------------
# Train probes
# ---------------------------------------------------------------------------

def fit_concept_probes(
    Z_train_mi_std: np.ndarray, theta_train: Dict[str, np.ndarray],
) -> Dict[str, object]:
    """Fit (phi, z, size, transmurality) probes on standardised K=1024 latents."""
    # phi (sin, cos) via RidgeCV multi-output
    Y_phi = np.column_stack([
        np.sin(theta_train["phi"]),
        np.cos(theta_train["phi"]),
    ])
    phi_model = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2").fit(Z_train_mi_std, Y_phi)

    z_model = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2").fit(
        Z_train_mi_std, theta_train["z"]
    )
    size_model = RidgeCV(alphas=RIDGE_ALPHAS, scoring="r2").fit(
        Z_train_mi_std, theta_train["size"]
    )

    # transmurality binary
    trans_bin = (theta_train["transmural"] > 0.5).astype(np.int64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_C, best_auc = None, -np.inf
    for C in LOGREG_CS_BIN:
        fold_aucs = []
        for tr, va in skf.split(Z_train_mi_std, trans_bin):
            est = LogisticRegression(
                C=C, penalty="l2", solver="lbfgs",
                max_iter=2000, class_weight="balanced",
            ).fit(Z_train_mi_std[tr], trans_bin[tr])
            fold_aucs.append(roc_auc_score(
                trans_bin[va], est.predict_proba(Z_train_mi_std[va])[:, 1],
            ))
        mu = float(np.mean(fold_aucs))
        if mu > best_auc:
            best_auc = mu
            best_C = float(C)
    trans_model = LogisticRegression(
        C=best_C, penalty="l2", solver="lbfgs",
        max_iter=2000, class_weight="balanced",
    ).fit(Z_train_mi_std, trans_bin)

    return {
        "phi": phi_model, "z": z_model, "size": size_model,
        "trans": trans_model,
        "trans_best_C": best_C, "trans_cv_auc": float(best_auc),
    }


def predict_concept_5vec(
    probes: Dict[str, object], Z_std: np.ndarray,
) -> np.ndarray:
    """(N, 5) matrix [phi_sin_pred, phi_cos_pred, z_hat, size_hat, trans_logit]."""
    Y_phi = probes["phi"].predict(Z_std)
    phi_sin = Y_phi[:, 0]
    phi_cos = Y_phi[:, 1]
    z_hat = probes["z"].predict(Z_std)
    sz_hat = probes["size"].predict(Z_std)
    # decision_function returns log-odds for binary LogReg
    trans_logit = probes["trans"].decision_function(Z_std)
    return np.column_stack([phi_sin, phi_cos, z_hat, sz_hat, trans_logit])


# ---------------------------------------------------------------------------
# Train + eval 5-vec classifier
# ---------------------------------------------------------------------------

def fit_classifier(
    X_train: np.ndarray, y_train: np.ndarray, *, Cs=LOGREG_CS_TERR,
) -> Tuple[LogisticRegression, float, Dict[str, float]]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores: Dict[str, float] = {}
    for C in Cs:
        fold_scores = []
        for tr, va in skf.split(X_train, y_train):
            est = LogisticRegression(
                C=C, penalty="l2", solver="lbfgs",
                class_weight="balanced", max_iter=4000,
                multi_class="multinomial",
            ).fit(X_train[tr], y_train[tr])
            yhat = est.predict(X_train[va])
            fold_scores.append(f1_score(
                y_train[va], yhat, labels=TERRITORIES_4C,
                average="macro", zero_division=0,
            ))
        cv_scores[str(float(C))] = float(np.mean(fold_scores))
    best_C = float(max(cv_scores, key=cv_scores.get))
    model = LogisticRegression(
        C=best_C, penalty="l2", solver="lbfgs",
        class_weight="balanced", max_iter=4000,
        multi_class="multinomial",
    ).fit(X_train, y_train)
    return model, best_C, cv_scores


def eval_classifier(
    model, X_train, y_train, X_test, y_test, X_ptbxl, y_ptbxl, rng,
) -> Dict[str, object]:
    in_dom = score_block(
        y_test, model.predict(X_test), model.predict_proba(X_test), rng=rng,
        proba_labels=list(model.classes_),
    )
    cd = score_block(
        y_ptbxl, model.predict(X_ptbxl), model.predict_proba(X_ptbxl), rng=rng,
        proba_labels=list(model.classes_),
    )
    return {"in_domain_4c": in_dom, "cross_domain_4c": cd}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ablations", action="store_true",
                    help="Run drop-one-feature ablations.")
    ap.add_argument("--mlp", action="store_true",
                    help="Add an MLP head variant.")
    ap.add_argument("--gt-upper-bound", action="store_true",
                    help="Add the GT-theta upper-bound baseline.")
    ap.add_argument("--all-sensitivity", action="store_true",
                    help="Shortcut for --ablations --mlp --gt-upper-bound.")
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "concept5_classifier.json")
    args = ap.parse_args()
    if args.all_sensitivity:
        args.ablations = args.mlp = args.gt_upper_bound = True
    args.out.parent.mkdir(parents=True, exist_ok=True)

    targets = load_theta_targets()
    ptbxl_idx, ptbxl_truth = load_ptbxl_primary_4c()
    print(f"[load] PTB-XL primary 4c subset n={ptbxl_idx.size}")
    print(f"       per-territory: {dict(pd.Series(ptbxl_truth).value_counts())}")

    # Use K=1024 baseline (exp7) — the same latents whose B2 in-domain
    # already passed; the question is whether the *concept space* generalises.
    Z_med_tr_full = load_Z(1024, "medalcare", "train")
    Z_med_te_full = load_Z(1024, "medalcare", "test")
    Z_ptb_te_full = load_Z(1024, "ptbxl", "test")

    idx_tr = targets["train"]["idx_in_split"]
    idx_te = targets["test"]["idx_in_split"]
    Z_med_tr = Z_med_tr_full[idx_tr]
    Z_med_te = Z_med_te_full[idx_te]
    Z_ptb = Z_ptb_te_full[ptbxl_idx]

    # Standardise on MedalCare train (the probe-fitting domain)
    scaler = StandardScaler().fit(Z_med_tr)
    Z_med_tr_std = scaler.transform(Z_med_tr)
    Z_med_te_std = scaler.transform(Z_med_te)
    Z_ptb_std = scaler.transform(Z_ptb)

    # Probes
    probes = fit_concept_probes(Z_med_tr_std, targets["train"])
    print(
        f"[probes] phi alpha={probes['phi'].alpha_}  "
        f"z alpha={probes['z'].alpha_:.3g}  "
        f"size alpha={probes['size'].alpha_:.3g}  "
        f"trans C={probes['trans_best_C']:.3g} cv_auc={probes['trans_cv_auc']:.3f}"
    )

    # Sanity: in-domain probe metrics on MedalCare test
    Y_phi_te = probes["phi"].predict(Z_med_te_std)
    phi_te_pred = np.arctan2(Y_phi_te[:, 0], Y_phi_te[:, 1])
    cos_diff = np.cos(phi_te_pred - targets["test"]["phi"])
    mean_phi = np.arctan2(
        np.sin(targets["test"]["phi"]).mean(),
        np.cos(targets["test"]["phi"]).mean(),
    )
    ss_res = float((1.0 - cos_diff).mean())
    ss_tot = float((1.0 - np.cos(targets["test"]["phi"] - mean_phi)).mean())
    phi_r2_circ = 1.0 - ss_res / max(ss_tot, 1e-12)
    z_te_pred = probes["z"].predict(Z_med_te_std)
    sz_te_pred = probes["size"].predict(Z_med_te_std)
    trans_te_bin = (targets["test"]["transmural"] > 0.5).astype(np.int64)
    trans_te_score = probes["trans"].predict_proba(Z_med_te_std)[:, 1]
    probe_metrics = {
        "phi_r2_circular": phi_r2_circ,
        "z_r2": float(r2_score(targets["test"]["z"], z_te_pred)),
        "size_r2": float(r2_score(targets["test"]["size"], sz_te_pred)),
        "transmurality_auc": float(roc_auc_score(trans_te_bin, trans_te_score))
            if len(np.unique(trans_te_bin)) == 2 else None,
    }
    print(f"[probes] in-domain test metrics: {probe_metrics}")

    # Predicted 5-vectors
    C_med_tr = predict_concept_5vec(probes, Z_med_tr_std)
    C_med_te = predict_concept_5vec(probes, Z_med_te_std)
    C_ptb = predict_concept_5vec(probes, Z_ptb_std)
    print(
        f"[concepts] shapes: med_train={C_med_tr.shape}  "
        f"med_test={C_med_te.shape}  ptbxl={C_ptb.shape}"
    )

    # Save predictions for downstream reuse
    np.savez(
        OUT_DIR / "concept5_predictions.npz",
        C_med_train=C_med_tr.astype(np.float32),
        C_med_test=C_med_te.astype(np.float32),
        C_ptbxl_primary=C_ptb.astype(np.float32),
        ptbxl_primary_row_idx=ptbxl_idx.astype(np.int64),
        ptbxl_primary_territory_4c=np.array(ptbxl_truth, dtype=object),
        concept_names=np.array(CONCEPT_NAMES),
    )
    print(f"[save] {(OUT_DIR / 'concept5_predictions.npz').relative_to(REPO_ROOT)}")

    # ----- PTB-XL concept-collapse audit -----
    audit_rows = []
    for terr in TERRITORIES_4C:
        mask = ptbxl_truth == terr
        if mask.sum() == 0:
            continue
        for j, name in enumerate(CONCEPT_NAMES):
            v = C_ptb[mask, j]
            audit_rows.append({
                "territory_4c": terr,
                "concept": name,
                "n": int(mask.sum()),
                "median": float(np.median(v)),
                "iqr_lo": float(np.percentile(v, 25)),
                "iqr_hi": float(np.percentile(v, 75)),
                "mean": float(v.mean()),
                "std": float(v.std()),
            })
    audit_df = pd.DataFrame(audit_rows)
    audit_path = OUT_DIR / "concept5_ptbxl_per_territory.csv"
    audit_df.to_csv(audit_path, index=False)
    print(f"[save] {audit_path.relative_to(REPO_ROOT)}")

    # Also compute the same audit on MedalCare-test for comparison.
    med_te_truth = np.asarray(targets["test"]["territory_4c"].tolist(), dtype=object)
    med_audit_rows = []
    for terr in TERRITORIES_4C:
        mask = med_te_truth == terr
        if mask.sum() == 0:
            continue
        for j, name in enumerate(CONCEPT_NAMES):
            v = C_med_te[mask, j]
            med_audit_rows.append({
                "territory_4c": terr, "concept": name,
                "n": int(mask.sum()),
                "median": float(np.median(v)),
                "iqr_lo": float(np.percentile(v, 25)),
                "iqr_hi": float(np.percentile(v, 75)),
                "mean": float(v.mean()),
                "std": float(v.std()),
            })
    med_audit_df = pd.DataFrame(med_audit_rows)
    med_audit_path = OUT_DIR / "concept5_medalcare_test_per_territory.csv"
    med_audit_df.to_csv(med_audit_path, index=False)
    print(f"[save] {med_audit_path.relative_to(REPO_ROOT)}")

    # m10 fix: each reported cell gets its own deterministic stream keyed on
    # its identity, so --ablations / --mlp / --gt-upper-bound no longer shift
    # the numbers of the cells that would have run anyway.

    # ----- Main classifier on predicted 5-vec -----
    y_train_4c = np.asarray(targets["train"]["territory_4c"].tolist(), dtype=object)
    y_test_4c = np.asarray(targets["test"]["territory_4c"].tolist(), dtype=object)
    print("\n[main] fitting LR on 5-concept predictions from MedalCare-train")
    cf_scaler = StandardScaler().fit(C_med_tr)
    Xc_tr = cf_scaler.transform(C_med_tr)
    Xc_te = cf_scaler.transform(C_med_te)
    Xc_pt = cf_scaler.transform(C_ptb)
    model, best_C, cv_scores = fit_classifier(Xc_tr, y_train_4c)
    print(f"[main] best_C={best_C:g}  cv_macro_f1={cv_scores[str(best_C)]:.3f}")
    main_block = eval_classifier(
        model, Xc_tr, y_train_4c, Xc_te, y_test_4c, Xc_pt, ptbxl_truth,
        derive_rng("concept5", "main", seed=SEED),
    )
    id_ = main_block["in_domain_4c"]; cd = main_block["cross_domain_4c"]
    print(f"[main] in_dom F1={id_['macro_f1']:.3f}  AUC={id_['macro_auc_ovr']}")
    print(
        f"[main] CD F1={cd['macro_f1']:.3f} CI={cd['macro_f1_ci95']} "
        f"p={cd['permutation_p_macro_f1']:.4f}  "
        f"AUC={cd['macro_auc_ovr']} CI={cd['macro_auc_ovr_ci95']} "
        f"p={cd['permutation_p_macro_auc_ovr']}"
    )

    results: Dict[str, object] = {
        "probe_in_domain_metrics": probe_metrics,
        "main": {
            "best_C": best_C,
            "cv_scores_per_C": cv_scores,
            "in_domain_4c": main_block["in_domain_4c"],
            "cross_domain_4c": main_block["cross_domain_4c"],
            "feature_names": CONCEPT_NAMES,
        },
    }

    # ----- Drop-one ablations -----
    if args.ablations:
        ablations: Dict[str, object] = {}
        for j, name in enumerate(CONCEPT_NAMES):
            mask = np.ones(5, dtype=bool); mask[j] = False
            Xa_tr = Xc_tr[:, mask]
            Xa_te = Xc_te[:, mask]
            Xa_pt = Xc_pt[:, mask]
            m_a, bc_a, _ = fit_classifier(Xa_tr, y_train_4c)
            blk = eval_classifier(
                m_a, Xa_tr, y_train_4c, Xa_te, y_test_4c, Xa_pt, ptbxl_truth,
                derive_rng("concept5", "ablate", name, seed=SEED),
            )
            ablations[f"drop_{name}"] = {
                "best_C": bc_a,
                "kept_features": [CONCEPT_NAMES[i] for i in range(5) if mask[i]],
                "in_domain_4c": blk["in_domain_4c"],
                "cross_domain_4c": blk["cross_domain_4c"],
            }
            cd_a = blk["cross_domain_4c"]
            print(
                f"[ablate drop {name}] CD F1={cd_a['macro_f1']:.3f}  "
                f"AUC={cd_a['macro_auc_ovr']}"
            )
        results["ablations"] = ablations

    # ----- MLP variant -----
    if args.mlp:
        print("\n[mlp] fitting 2-layer MLP on 5-vec predictions")
        # sklearn MLPClassifier early_stopping breaks on object-dtype labels (isnan
        # fails on strings); encode to ints for MLP fit, decode predictions back.
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(y_train_4c)
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu", alpha=1e-3,
            max_iter=2000, random_state=SEED, early_stopping=True,
        ).fit(Xc_tr, le.transform(y_train_4c))

        class _MLPWrapper:
            def __init__(self, base, le_):
                self.base = base; self.le = le_
                self.classes_ = le_.classes_
            def predict(self, X):
                return self.le.inverse_transform(self.base.predict(X))
            def predict_proba(self, X):
                return self.base.predict_proba(X)
        mlp_w = _MLPWrapper(mlp, le)
        mlp_block = eval_classifier(
            mlp_w, Xc_tr, y_train_4c, Xc_te, y_test_4c, Xc_pt, ptbxl_truth,
            derive_rng("concept5", "mlp", seed=SEED),
        )
        results["mlp"] = {
            "in_domain_4c": mlp_block["in_domain_4c"],
            "cross_domain_4c": mlp_block["cross_domain_4c"],
        }
        cd_m = mlp_block["cross_domain_4c"]
        print(f"[mlp] CD F1={cd_m['macro_f1']:.3f}  AUC={cd_m['macro_auc_ovr']}")

    # ----- GT upper bound -----
    if args.gt_upper_bound:
        # Build the GROUND-TRUTH 5-vec from MedalCare theta_mi targets.
        # (No PTB-XL counterpart — we can only evaluate in-domain.)
        gt_tr = np.column_stack([
            np.sin(targets["train"]["phi"]),
            np.cos(targets["train"]["phi"]),
            targets["train"]["z"],
            targets["train"]["size"],
            (targets["train"]["transmural"] > 0.5).astype(np.float64),
        ])
        gt_te = np.column_stack([
            np.sin(targets["test"]["phi"]),
            np.cos(targets["test"]["phi"]),
            targets["test"]["z"],
            targets["test"]["size"],
            (targets["test"]["transmural"] > 0.5).astype(np.float64),
        ])
        gt_scaler = StandardScaler().fit(gt_tr)
        Xg_tr = gt_scaler.transform(gt_tr)
        Xg_te = gt_scaler.transform(gt_te)
        m_g, bc_g, cv_g = fit_classifier(Xg_tr, y_train_4c)
        in_blk = score_block(
            y_test_4c, m_g.predict(Xg_te), m_g.predict_proba(Xg_te),
            rng=derive_rng("concept5", "gt_upper_bound", seed=SEED),
            proba_labels=list(m_g.classes_),
        )
        results["gt_upper_bound"] = {
            "best_C": bc_g,
            "cv_scores_per_C": cv_g,
            "in_domain_4c": in_blk,
            "note": "No cross-domain leg (PTB-XL has no biophysical GT).",
        }
        print(
            f"[gt-ub] in_dom F1={in_blk['macro_f1']:.3f}  "
            f"AUC={in_blk['macro_auc_ovr']}"
        )

    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {args.out.relative_to(REPO_ROOT)}")

    print()
    print("=" * 88)
    print("Concept5 summary")
    print("=" * 88)
    print(f"Probe in-domain: phi R²_circ={probe_metrics['phi_r2_circular']:.3f}  "
          f"z R²={probe_metrics['z_r2']:.3f}  size R²={probe_metrics['size_r2']:.3f}  "
          f"trans AUC={probe_metrics['transmurality_auc']}")
    print(f"Main 5-vec LR:")
    print(f"  in_dom: F1={id_['macro_f1']:.3f} AUC={id_['macro_auc_ovr']}")
    print(
        f"  CD:     F1={cd['macro_f1']:.3f} [{cd['macro_f1_ci95'][0]:.3f},"
        f"{cd['macro_f1_ci95'][1]:.3f}] p={cd['permutation_p_macro_f1']:.4f} | "
        f"AUC={cd['macro_auc_ovr']} CI={cd['macro_auc_ovr_ci95']} "
        f"p={cd['permutation_p_macro_auc_ovr']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
