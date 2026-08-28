"""DIAGNOSTIC (no retrain): export PTB-XL test latents with aVL/aVF swapped.

Motivation
----------
`scripts/datasets.py:93-95` declares the MedalCare WFDB lead order as
    ['I','II','III','aVR','aVF','aVL','V1'..'V6']
and permutes positions 4<->5 to reach the "target" order. But the WFDB files
written by `scripts/prepare_medalcare.py` are already in standard order
(`LEAD_ORDER` at :53, and the manifest's `lead_order` column agrees).

Verified empirically via the exact limb-lead identities
    aVR = -(I+II)/2      aVL = (I-III)/2      aVF = (II+III)/2
on 6 MedalCare records: channel 4 IS aVL and channel 5 IS aVF.

=> Every MedalCare batch ever fed to the model had aVL and aVF swapped.
   PTB-XL, which reindexes by `sig_name`, did not.

aVF is the inferior lead and aVL the high-lateral lead, so this is exactly the
kind of corruption that would (a) be invisible in-domain, (b) scramble the
frontal-plane angle phi between domains, and (c) survive MMD / INLP /
bottleneck / multi-task alignment (a channel permutation on one domain makes
the encoder a *different function of the input*, not a shifted distribution).

The decisive test needs no retraining: push PTB-XL through the SAME corrupted
convention and see whether cross-domain territory transfer recovers.

This script writes ONLY new directories. It does not touch any existing
artifact:
    outputs/latents/exp7_ptbxl_leadswap/latents.npz
    outputs/latents/exp7_ptbxl_train_leadswap/latents.npz   (--split train)
    outputs/analysis/leadswap_diag/pipeline_a_leadswap.json (--eval)

Usage
-----
    python scripts/_diag_leadswap_ptbxl.py --export --split test
    python scripts/_diag_leadswap_ptbxl.py --eval
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from net1d import Net1D  # pylint: disable=wrong-import-position
from scripts.datasets import get_dataset  # pylint: disable=wrong-import-position

DEFAULT_PTBXL_ROOT = (
    REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
CKPT = REPO_ROOT / "outputs" / "exp7_baseline" / "checkpoints" / "linear_best.pt"
LATENT_DIR = REPO_ROOT / "outputs" / "latents"
OUT_DIR = REPO_ROOT / "outputs" / "analysis" / "leadswap_diag"

NET1D_ARCH = dict(
    in_channels=12,
    base_filters=64,
    ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16,
    stride=2,
    groups_width=16,
    verbose=False,
    use_bn=False,
    use_do=False,
)

# Position 4 = aVL, position 5 = aVF in the standard TARGET_LEADS order used by
# PTBXLDataset. Swapping them reproduces what LVEF_12lead_cls_Dataset does to
# every MedalCare record.
SWAP = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def build_model(device: torch.device) -> Net1D:
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    n_classes = sd["dense.weight"].shape[0]
    model = Net1D(**NET1D_ARCH, n_classes=n_classes, use_adapter=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {missing[:8]}")
    if unexpected:
        print(f"  [WARN] unexpected keys ({len(unexpected)}): {unexpected[:4]}")
    model.return_features = True
    model.to(device).eval()
    return model


@torch.no_grad()
def export(split: str, swap: bool, outdir: Path, batch_size: int,
           device: torch.device, domain: str = "ptbxl",
           per_lead_norm: bool = False):
    """Export latents.

    `per_lead_norm` re-normalises each lead to zero mean / unit std. For the
    MedalCare loader (which applies a single GLOBAL scalar z-score) this is
    algebraically identical to having z-scored per-lead from the raw signal:
        global:   x' = (x - m)/s          (m, s scalars)
        per-lead: (x' - mean_l(x'))/std_l(x') = (x - mean_l(x))/std_l(x)
    so the global step cancels exactly. It therefore reproduces PTB-XL's
    `_z_score` convention without touching the Dataset class.
    """
    if domain == "ptbxl":
        ds = get_dataset("ptbxl", root=DEFAULT_PTBXL_ROOT, split=split,
                         return_metadata=False)
    else:
        man = REPO_ROOT / "data" / "medalcare_filtered_manifest_dataset_split.csv"
        df = pd.read_csv(man)
        if split != "all":
            df = df[df["split"].str.lower() == split.lower()].copy()
        ds = get_dataset("medalcare", ecg_path="", labels_df=df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    model = build_model(device)

    idx = torch.tensor(SWAP, device=device) if swap else None
    z_parts, p_parts, y_parts = [], [], []
    tag = f"{domain}/{split} swap={swap} perlead={per_lead_norm}"
    for batch in tqdm(loader, desc=tag, leave=False):
        x = batch[0].to(device, non_blocking=True)
        if idx is not None:
            x = x.index_select(1, idx)
        if per_lead_norm:
            x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-8)
        logits, feats = model(x)
        z_parts.append(feats.cpu().numpy())
        p_parts.append(torch.sigmoid(logits).cpu().numpy())
        y_parts.append(batch[1].numpy())

    Z = np.concatenate(z_parts, 0)
    P = np.concatenate(p_parts, 0)
    Y = np.concatenate(y_parts, 0)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "latents.npz", Z=Z, P=P, Y=Y)
    print(f"  wrote {outdir/'latents.npz'}  Z={Z.shape}")
    return Z


# ---------------------------------------------------------------------------
# Evaluation — Pipeline A, identical protocol to eval_decoding_lowK
# ---------------------------------------------------------------------------

def run_eval(seed: int = 42, med_train: str = "exp7_medalcare_train",
             med_test: str = "exp7_medalcare", tag_suffix: str = "",
             outfile: str = "pipeline_a_leadswap.json"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    sys.path.insert(0, str(REPO_ROOT / "analysis"))
    from eval_decoding_lowK import (  # pylint: disable=wrong-import-position
        score_block, TERRITORIES_4C,
    )

    rng = np.random.default_rng(seed)

    def L(name):
        return np.load(LATENT_DIR / name / "latents.npz", allow_pickle=True)["Z"].astype(np.float64)

    Z_med_tr = L(med_train)
    Z_med_te = L(med_test)
    Z_ptb_orig = L("exp7_ptbxl")
    Z_ptb_swap = L("exp7_ptbxl_leadswap")

    th_tr = dict(np.load(REPO_ROOT / "data" / "theta_mi_train.npz", allow_pickle=True))
    th_te = dict(np.load(REPO_ROOT / "data" / "theta_mi_test.npz", allow_pickle=True))

    def mi_subset(Z, th):
        idx = th["idx_in_split"]
        terr = th["territory_4c"].astype(str)
        keep = np.isin(terr, TERRITORIES_4C)
        return Z[idx[keep]], terr[keep]

    Xtr, ytr = mi_subset(Z_med_tr, th_tr)
    Xte, yte = mi_subset(Z_med_te, th_te)

    df = pd.read_csv(REPO_ROOT / "data" / "ptbxl_mi_subclass.csv")
    sub = df[df["territory_4c"].isin(TERRITORIES_4C)]
    p_idx = sub["row_idx"].to_numpy()
    p_y = sub["territory_4c"].to_numpy()

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    # Same C-selection protocol as fit_territory_classifier
    best_C, best_cv, cv_scores = None, -np.inf, {}
    for C in [0.001, 0.01, 0.1, 1.0]:
        clf = LogisticRegression(C=C, max_iter=2000, multi_class="multinomial",
                                 class_weight="balanced", random_state=seed)
        cv = cross_val_score(
            clf, Xtr_s, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            scoring="f1_macro",
        ).mean()
        cv_scores[str(C)] = float(cv)
        if cv > best_cv:
            best_cv, best_C = cv, C
    model = LogisticRegression(C=best_C, max_iter=2000, multi_class="multinomial",
                               class_weight="balanced", random_state=seed).fit(Xtr_s, ytr)
    print(f"[{tag_suffix or 'BASELINE'}] best_C={best_C}  cv_f1={best_cv:.4f}")

    res = {"med_train": med_train, "med_test": med_test,
           "best_C": best_C, "cv_scores": cv_scores,
           "in_domain": score_block(yte, model.predict(Xte_s),
                                    model.predict_proba(Xte_s), rng=rng,
                                    proba_labels=list(model.classes_))}
    print(f"  in-domain MedalCare macro-F1 {res['in_domain']['macro_f1']:.4f}")

    for tag, Zp in [("ptbxl_orig", Z_ptb_orig), ("ptbxl_leadswap", Z_ptb_swap)]:
        Xp = scaler.transform(Zp[p_idx])
        yhat = model.predict(Xp)
        res[tag] = score_block(p_y, yhat, model.predict_proba(Xp), rng=rng,
                               proba_labels=list(model.classes_))
        b = res[tag]
        cm = np.asarray(b["confusion_matrix"])
        auc = b["macro_auc_ovr"]
        print(f"\n=== {tag} ===")
        print(f"  macro-F1 {b['macro_f1']:.4f} "
              f"CI{[round(v, 3) for v in b['macro_f1_ci95']]} "
              f"p={b['permutation_p_macro_f1']:.4f}")
        print(f"  bal-acc  {b['balanced_accuracy']:.4f}   "
              f"macro-AUC {auc if auc is None else round(auc, 4)}")
        print(f"  rows=truth {list(TERRITORIES_4C)}, cols=pred")
        for lab, row in zip(TERRITORIES_4C, cm):
            print(f"    {lab:<16} {row}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / outfile, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {OUT_DIR/outfile}")

    d = res["ptbxl_leadswap"]["macro_f1"] - res["ptbxl_orig"]["macro_f1"]
    print(f"\nDELTA macro-F1 (leadswap - orig) = {d:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--export-medalcare-unswapped", action="store_true",
                    help="Re-export MedalCare with SWAP re-applied (involution) "
                         "so the model sees TRUE standard lead order.")
    ap.add_argument("--export-medalcare-perlead", action="store_true",
                    help="MedalCare, correct lead order AND per-lead z-score "
                         "(matches PTB-XL's convention).")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval-unswapped", action="store_true")
    ap.add_argument("--eval-perlead", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.export:
        print(f"device={device}")
        suffix = "" if args.split == "test" else f"_{args.split}"
        export(args.split, True, LATENT_DIR / f"exp7_ptbxl{suffix}_leadswap",
               args.batch_size, device, domain="ptbxl")
    if args.export_medalcare_unswapped:
        print(f"device={device}")
        suffix = "" if args.split == "test" else f"_{args.split}"
        export(args.split, True, LATENT_DIR / f"exp7_medalcare{suffix}_unswapped",
               args.batch_size, device, domain="medalcare")
    if args.export_medalcare_perlead:
        print(f"device={device}")
        suffix = "" if args.split == "test" else f"_{args.split}"
        export(args.split, True, LATENT_DIR / f"exp7_medalcare{suffix}_unswapped_perlead",
               args.batch_size, device, domain="medalcare", per_lead_norm=True)
    if args.eval:
        run_eval()
    if args.eval_unswapped:
        run_eval(med_train="exp7_medalcare_train_unswapped",
                 med_test="exp7_medalcare_unswapped",
                 tag_suffix="_UNSWAPPED-MEDALCARE",
                 outfile="pipeline_a_medalcare_unswapped.json")
    if args.eval_perlead:
        run_eval(med_train="exp7_medalcare_train_unswapped_perlead",
                 med_test="exp7_medalcare_unswapped_perlead",
                 tag_suffix="_UNSWAPPED+PERLEAD",
                 outfile="pipeline_a_medalcare_unswapped_perlead.json")


if __name__ == "__main__":
    main()
