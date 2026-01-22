import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.datasets import get_dataset  # pylint: disable=wrong-import-position
from finetune_model import ft_multihead_ECGFounder  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe domain separability on frozen encoder features."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Baseline joint checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-mmd",
        type=Path,
        default=None,
        help="Optional MMD joint checkpoint path for comparison.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "MedalRaw" / "medalcare_filtered_manifest.csv",
        help="MedalCare manifest CSV.",
    )
    parser.add_argument(
        "--ptbxl-root",
        type=Path,
        default=REPO_ROOT
        / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        help="PTB-XL root directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum samples per domain (balanced).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (defaults to outputs/domain_probe_*.json).",
    )
    return parser.parse_args()


def build_model(device: torch.device, checkpoint: Path) -> torch.nn.Module:
    base_fm_ckpt = REPO_ROOT / "checkpoint" / "12_lead_ECGFounder.pth"
    model = ft_multihead_ECGFounder(
        device=device,
        pth=str(base_fm_ckpt),
        n_medal_classes=8,
        n_ptb_classes=5,
        linear_prob=True,
    )
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_balanced_subsets(
    manifest: Path,
    ptbxl_root: Path,
    max_samples: int,
    seed: int,
) -> Tuple[Subset, Subset]:
    rng = np.random.default_rng(seed)
    medal_df = pd.read_csv(manifest)
    if "split" in medal_df.columns:
        medal_df = medal_df[medal_df["split"] == "test"].copy()
    dataset_medal = get_dataset("medalcare", ecg_path="", labels_df=medal_df)

    dataset_ptb = get_dataset("ptbxl", root=ptbxl_root, split="test")

    n_medal = len(dataset_medal)
    n_ptb = len(dataset_ptb)
    n_take = min(n_medal, n_ptb, max_samples)
    if n_take == 0:
        raise ValueError("No samples available for one of the domains.")

    medal_idx = rng.choice(n_medal, n_take, replace=False)
    ptb_idx = rng.choice(n_ptb, n_take, replace=False)
    return Subset(dataset_medal, medal_idx), Subset(dataset_ptb, ptb_idx)


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            _, z = model(x, task=task, return_features=True)
            feats.append(z.cpu().numpy())
    return np.concatenate(feats, axis=0)


def run_probe(
    model: torch.nn.Module,
    medal_subset: Subset,
    ptb_subset: Subset,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict:
    loader_medal = DataLoader(medal_subset, batch_size=batch_size, shuffle=False)
    loader_ptb = DataLoader(ptb_subset, batch_size=batch_size, shuffle=False)

    features_medal = extract_features(model, loader_medal, device, task="medalcare")
    features_ptb = extract_features(model, loader_ptb, device, task="ptbxl")

    X = np.concatenate([features_medal, features_ptb], axis=0)
    y = np.concatenate(
        [
            np.zeros(features_medal.shape[0], dtype=np.int64),
            np.ones(features_ptb.shape[0], dtype=np.int64),
        ],
        axis=0,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    return {
        "auc": float(auc),
        "num_samples_per_domain": int(features_medal.shape[0]),
        "num_features": int(features_medal.shape[1]),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    medal_subset, ptb_subset = get_balanced_subsets(
        args.manifest, args.ptbxl_root, args.max_samples, args.seed
    )

    results = {"seed": args.seed, "max_samples": args.max_samples}

    model = build_model(device, args.checkpoint)
    results["baseline"] = run_probe(
        model,
        medal_subset,
        ptb_subset,
        device,
        args.batch_size,
        args.seed,
    )

    if args.checkpoint_mmd:
        model_mmd = build_model(device, args.checkpoint_mmd)
        results["mmd"] = run_probe(
            model_mmd,
            medal_subset,
            ptb_subset,
            device,
            args.batch_size,
            args.seed,
        )

    out_path = args.output
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = REPO_ROOT / "outputs" / f"domain_probe_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved probe results to {out_path}")


if __name__ == "__main__":
    main()

