"""Stage 4.2 — lead-permutation sensitivity sweep.

THE CLAIM UNDER TEST (a methods contribution, not a result about this thesis's
encoder): a lead-order corruption between domains is **invisible to the
distributional diagnostics the field uses** (C2ST, MMD) while it measurably
damages cross-domain transfer. If that holds, C2ST/MMD are insufficient as a
domain-gap audit, and transfer is the sensitive instrument. The 2026-08-10 audit
found exactly one instance of this (aVL/aVF: transfer p 0.76 -> 0.002, C2ST
pinned at 1.0000). One instance is an anecdote; a sweep over all 66 transpositions
is a diagnostic.

DESIGN, FIXED BEFORE ANY NUMBER EXISTS (the Part 16 discipline)
---------------------------------------------------------------
* **What is permuted.** PTB-XL test (the TARGET domain) at inference only. The
  territory probe is fit once on correctly-ordered MedalCare train latents and is
  **never refit**. Any movement in transfer is therefore attributable to the input
  permutation alone -- there is no second fitted object to absorb it.
* **Encoder.** `exp8_leadfix_baseline` -- the corrected encoder every other claim
  in this run is stated on.
* **Sweep set**, declared here in full:
    - identity (the reference cell)
    - all C(12,2) = 66 transpositions
    - 10 seeded random full permutations (seed 42) as a corruption upper bound
* **Metrics per permutation**: territory macro-F1 on the PTB-XL MI subset with a
  permutation p; linear C2ST (MedalCare test vs permuted PTB-XL test); unbiased
  multi-bandwidth MMD^2.
* **GBDT C2ST** on a pre-declared subset only, because it is the expensive one:
  identity, the historical (aVL,aVF) swap, all 15 within-limb-block
  transpositions, and the first 3 random permutations. Declared now so the subset
  cannot be chosen after seeing which cells are convenient. Linear C2ST alone is
  known to mislead here (report SS11, SS15.2b) -- GBDT is what the project's
  C2ST=1.0 claim actually rests on.

PRE-DECLARED READING
--------------------
The methods claim is SUPPORTED iff transfer macro-F1 varies materially across
permutations while C2ST stays pinned near 1.0. It FAILS if C2ST also moves
materially -- in which case C2ST is a usable lead-order detector after all, and
that is the honest finding.

HARD PRECONDITION
-----------------
The identity cell must reproduce the stored
`outputs/latents/exp8_leadfix_baseline_ptbxl_test/latents.npz` to within float32
tolerance. If it does not, this script is measuring its own re-implementation
rather than the permutation, and it aborts without writing anything.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# pylint: disable=wrong-import-position
from net1d import Net1D  # noqa: E402
from scripts.datasets import get_dataset  # noqa: E402
from scripts._diag_leadswap_ptbxl import (  # noqa: E402
    DEFAULT_PTBXL_ROOT, NET1D_ARCH,
)

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from eval_decoding_lowK import score_block, TERRITORIES_4C  # noqa: E402

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6"]
LIMB = list(range(6))          # I, II, III, aVR, aVL, aVF
HISTORICAL = (4, 5)            # aVL <-> aVF, the 2026-08-10 bug
ENCODER = "exp8_leadfix_baseline"
CKPT = REPO_ROOT / "outputs" / ENCODER / "checkpoints" / "linear_best.pt"
LATENT_DIR = REPO_ROOT / "outputs" / "latents"
OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "leadperm_sweep"
SEED = 42
N_RANDOM = 10
C2ST_N = 1000                  # per domain, matching the project's subsample rule
IDENTITY_TOL = 1e-3            # float32 GPU nondeterminism, on ~O(1) features


# ---------------------------------------------------------------------------
# model / latents
# ---------------------------------------------------------------------------

def build(device):
    """Rebuild the exp8 encoder.

    `use_adapter` is derived from the checkpoint's own keys rather than assumed:
    `exp8_leadfix_baseline/args.json` records `use_adapter: false`, and guessing
    wrong here would silently change the features the whole sweep is measured on.
    The identity precondition below is the real check on this.
    """
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    use_adapter = any("adapter" in k for k in sd)
    n_classes = sd["dense.weight"].shape[0]
    print(f"  checkpoint: use_adapter={use_adapter}  n_classes={n_classes}")
    model = Net1D(**NET1D_ARCH, n_classes=n_classes, use_adapter=use_adapter)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"missing checkpoint keys: {missing[:8]}")
    if unexpected:
        print(f"  [warn] unexpected keys ({len(unexpected)}): {unexpected[:4]}")
    model.return_features = True
    return model.to(device).eval()


@torch.no_grad()
def forward(model, X, perm, device, bs=64):
    idx = torch.tensor(perm, device=device)
    out = []
    for i in range(0, len(X), bs):
        x = X[i:i + bs].to(device, non_blocking=True).index_select(1, idx)
        _, f = model(x)
        out.append(f.cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def load_Z(name):
    return np.load(LATENT_DIR / name / "latents.npz",
                   allow_pickle=True)["Z"].astype(np.float64)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def mmd2_multibw(A, B, rng, n=1000):
    """Unbiased multi-bandwidth MMD^2, median-heuristic bandwidths x {0.5,1,2}.

    Pairwise squared distances via the Gram-matrix identity
    ``d2 = |x|^2 + |y|^2 - 2 x.y`` -- the naive broadcast
    ``((Z[:,None,:] - Z[None,:,:])**2).sum(-1)`` materialises an
    ``(2n, 2n, 1024)`` array (32 GB at n=1000) and is not runnable.
    """
    a = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    b = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    Z = np.vstack([a, b])
    sq = (Z * Z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    med = np.median(d2[d2 > 0])
    na, nb = len(a), len(b)
    tot = 0.0
    for mult in (0.5, 1.0, 2.0):
        K = np.exp(-d2 / (mult * med + 1e-12))
        Kaa, Kbb, Kab = K[:na, :na], K[na:, na:], K[:na, na:]
        tot += ((Kaa.sum() - np.trace(Kaa)) / (na * (na - 1))
                + (Kbb.sum() - np.trace(Kbb)) / (nb * (nb - 1))
                - 2.0 * Kab.mean())
    return float(tot / 3.0)


def c2st(A, B, rng, nonlinear=False, n=C2ST_N):
    """Domain-classifier AUROC, 5-fold CV, scaler fit inside each fold."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    m = min(n, len(A), len(B))
    a = A[rng.choice(len(A), m, replace=False)]
    b = B[rng.choice(len(B), m, replace=False)]
    X = np.vstack([a, b])
    y = np.r_[np.zeros(m), np.ones(m)]
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if nonlinear:
            clf = HistGradientBoostingClassifier(random_state=SEED)
        else:
            clf = LogisticRegression(max_iter=2000, random_state=SEED)
        clf.fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------

def perm_name(perm):
    if list(perm) == list(range(12)):
        return "identity"
    moved = [i for i in range(12) if perm[i] != i]
    if len(moved) == 2:
        return f"{LEADS[moved[0]]}<->{LEADS[moved[1]]}"
    return "perm:" + ",".join(LEADS[p] for p in perm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000,
                    help="permutation draws for the transfer p-value")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only run the first N cells")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device={dev}  encoder={ENCODER}")

    # ---- cache PTB-XL test inputs -----------------------------------------
    t0 = time.time()
    ds = get_dataset("ptbxl", root=DEFAULT_PTBXL_ROOT, split="test",
                     return_metadata=False)
    Xs = []
    for b in DataLoader(ds, batch_size=64, shuffle=False, num_workers=0):
        Xs.append(b[0])
    X = torch.cat(Xs, 0)
    print(f"cached PTB-XL test {tuple(X.shape)} in {time.time()-t0:.1f}s")

    model = build(dev)

    # ---- HARD PRECONDITION: identity reproduces the stored export ---------
    Z_id = forward(model, X, list(range(12)), dev)
    Z_ref = load_Z(f"{ENCODER}_ptbxl_test")
    if Z_id.shape != Z_ref.shape:
        print(f"ABORT: shape {Z_id.shape} != stored {Z_ref.shape}")
        return 2
    dmax = float(np.abs(Z_id - Z_ref).max())
    print(f"identity vs stored export: max|d| = {dmax:.3e} (tol {IDENTITY_TOL})")
    if dmax > IDENTITY_TOL:
        print("ABORT: identity cell does not reproduce the stored latents; "
              "this sweep would be measuring a re-implementation.")
        return 3
    print("PRECONDITION OK -- identity reproduces the stored export\n")

    # ---- fit the territory probe ONCE on MedalCare train -------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    Z_med_tr = load_Z(f"{ENCODER}_medalcare_train")
    Z_med_te = load_Z(f"{ENCODER}_medalcare_test")
    th_tr = dict(np.load(REPO_ROOT / "data" / "theta_mi_train.npz",
                         allow_pickle=True))
    idx = th_tr["idx_in_split"]
    terr = th_tr["territory_4c"].astype(str)
    keep = np.isin(terr, TERRITORIES_4C)
    Xtr, ytr = Z_med_tr[idx[keep]], terr[keep]

    df = pd.read_csv(REPO_ROOT / "data" / "ptbxl_mi_subclass.csv")
    sub = df[df["territory_4c"].isin(TERRITORIES_4C)]
    p_idx = sub["row_idx"].to_numpy()
    p_y = sub["territory_4c"].to_numpy()
    print(f"probe train n={len(ytr)}   PTB-XL MI subset n={len(p_y)}")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    best_C, best_cv = None, -np.inf
    for C in [0.001, 0.01, 0.1, 1.0]:
        cv = cross_val_score(
            LogisticRegression(C=C, max_iter=2000,
                               class_weight="balanced", random_state=SEED),
            Xtr_s, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            scoring="f1_macro").mean()
        if cv > best_cv:
            best_cv, best_C = cv, C
    probe = LogisticRegression(C=best_C, max_iter=2000,
                               class_weight="balanced",
                               random_state=SEED).fit(Xtr_s, ytr)
    print(f"probe: best_C={best_C}  cv_f1={best_cv:.4f}  (fit once, never refit)\n")

    # ---- build the pre-declared sweep set ---------------------------------
    cells = [("identity", list(range(12)))]
    for i, j in itertools.combinations(range(12), 2):
        p = list(range(12))
        p[i], p[j] = p[j], p[i]
        cells.append((perm_name(p), p))
    rr = np.random.default_rng(SEED)
    for k in range(N_RANDOM):
        p = list(rr.permutation(12))
        cells.append((f"random{k}", p))

    gbdt_cells = {"identity", perm_name(_hist_perm())}
    for i, j in itertools.combinations(LIMB, 2):
        p = list(range(12))
        p[i], p[j] = p[j], p[i]
        gbdt_cells.add(perm_name(p))
    gbdt_cells |= {f"random{k}" for k in range(3)}
    print(f"{len(cells)} cells; GBDT C2ST on {len(gbdt_cells)} pre-declared\n")

    if args.limit:
        cells = cells[:args.limit]

    rows = []
    t_start = time.time()
    for n_, (name, perm) in enumerate(cells, 1):
        Z_ptb = Z_id if name == "identity" else forward(model, X, perm, dev)
        Xp = scaler.transform(Z_ptb[p_idx])
        blk = score_block(p_y, probe.predict(Xp), probe.predict_proba(Xp),
                          rng=np.random.default_rng(SEED),
                          n_boot=args.n_boot, n_perm=args.n_perm,
                          proba_labels=list(probe.classes_))
        r = {
            "name": name, "perm": list(map(int, perm)),
            "macro_f1": blk["macro_f1"],
            # captured because the whole reading turns on whether the
            # cell-to-cell spread exceeds within-cell sampling noise at n=438
            "macro_f1_ci95": blk["macro_f1_ci95"],
            "p_macro_f1": blk["permutation_p_macro_f1"],
            "balanced_accuracy": blk["balanced_accuracy"],
            "c2st_linear": c2st(Z_med_te, Z_ptb, np.random.default_rng(SEED)),
            "mmd2": mmd2_multibw(Z_med_te, Z_ptb, np.random.default_rng(SEED)),
        }
        if name in gbdt_cells:
            r["c2st_gbdt"] = c2st(Z_med_te, Z_ptb, np.random.default_rng(SEED),
                                  nonlinear=True)
        rows.append(r)
        el = time.time() - t_start
        print(f"[{n_:3d}/{len(cells)}] {name:<22} f1={r['macro_f1']:.4f} "
              f"p={r['p_macro_f1']:.4f}  C2ST_lin={r['c2st_linear']:.4f}"
              + (f"  C2ST_gbdt={r['c2st_gbdt']:.4f}" if "c2st_gbdt" in r else "")
              + f"  MMD2={r['mmd2']:.4f}   [{el/60:.1f}m]")

    out = {
        "metadata": {
            "encoder": ENCODER, "permuted_domain": "ptbxl_test",
            "probe": {"best_C": best_C, "cv_f1": float(best_cv),
                      "n_train": int(len(ytr)), "refit": False},
            "n_eval_rows": int(len(p_y)), "seed": SEED,
            "n_perm": args.n_perm, "n_boot": args.n_boot,
            "c2st_subsample_per_domain": C2ST_N,
            "identity_max_abs_dev_vs_stored": dmax,
            "gbdt_cells_predeclared": sorted(gbdt_cells),
        },
        "rows": rows,
    }
    fp = OUT_DIR / "leadperm_sweep.json"
    fp.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {fp}  ({len(rows)} cells, {(time.time()-t_start)/60:.1f} min)")
    return 0


def _hist_perm():
    p = list(range(12))
    p[HISTORICAL[0]], p[HISTORICAL[1]] = p[HISTORICAL[1]], p[HISTORICAL[0]]
    return p


if __name__ == "__main__":
    raise SystemExit(main())
