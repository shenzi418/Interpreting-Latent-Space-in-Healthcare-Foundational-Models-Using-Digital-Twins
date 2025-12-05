import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# Repo root & imports
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Adjust this import to your actual helper location
# (this assumes you have scripts/datasets.py with a get_dataset() function)
from scripts.datasets import get_dataset
from finetune_model import ft_multihead_ECGFounder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained multi-head model checkpoint (e.g. best.pt)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Latent Space: MedalCare vs PTB-XL",
        help="Plot title",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne_medal_ptb.png",
        help="Output filename (saved under outputs/)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading multi-head model from {args.checkpoint} ...")

    # ---------------------------------------------------------------------
    # 1. Build multi-head model and load trained weights
    #    - pth: base FM checkpoint (original ECGFounder)
    #    - args.checkpoint: fine-tuned multi-head checkpoint (joint Medal+PTB)
    # ---------------------------------------------------------------------
    base_fm_ckpt = REPO_ROOT / "checkpoint" / "12_lead_ECGFounder.pth"

    model = ft_multihead_ECGFounder(
        device=device,
        pth=str(base_fm_ckpt),
        n_medal_classes=8,
        n_ptb_classes=5,
        linear_prob=False,  # just inference here
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    # strict=False allows for extra keys like optim state or slightly different heads
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()

    # ---------------------------------------------------------------------
    # 2. Build MedalCare & PTB-XL test loaders (subsampled)
    # ---------------------------------------------------------------------
    print("Loading MedalCare test data ...")
    medal_manifest = REPO_ROOT / "MedalRaw" / "medalcare_filtered_manifest.csv"
    medal_df = pd.read_csv(medal_manifest)

    # Use test split if available; otherwise sample from full manifest
    if "split" in medal_df.columns:
        medal_df_test = medal_df[medal_df["split"] == "test"].copy()
    else:
        medal_df_test = medal_df.copy()

    # Subsample up to 500 synthetic samples for visualization
    if len(medal_df_test) > 500:
        medal_df_test = medal_df_test.sample(n=500, random_state=42)

    dataset_syn = get_dataset(
        "medalcare",
        ecg_path="",
        labels_df=medal_df_test,
    )
    loader_syn = DataLoader(dataset_syn, batch_size=32, shuffle=False)

    print("Loading PTB-XL test data ...")
    ptb_root = REPO_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    dataset_real = get_dataset(
        "ptbxl",
        root=ptb_root,
        split="test",
    )

    # Subsample up to 500 real samples as well
    n_real = len(dataset_real)
    n_take = min(n_real, 500)
    real_indices = np.random.RandomState(42).choice(n_real, n_take, replace=False)
    dataset_real_sub = torch.utils.data.Subset(dataset_real, real_indices)
    loader_real = DataLoader(dataset_real_sub, batch_size=32, shuffle=False)

    # ---------------------------------------------------------------------
    # 3. Extract latent features from the shared encoder
    #    We use model(..., task=..., return_features=True) to get deep_features
    # ---------------------------------------------------------------------
    features_list = []
    domain_labels = []  # 0 = synthetic (MedalCare), 1 = real (PTB-XL)

    print("Extracting latent features ...")
    with torch.no_grad():
        # Synthetic / MedalCare
        for batch in loader_syn:
            # assume (signals, labels, *extras); adjust indexing if your dataset returns a dict
            x = batch[0].to(device)
            # forward through MedalCare head; return_features=True gives (logits, deep_features)
            logits, z = model(x, task="medalcare", return_features=True)
            features_list.append(z.cpu().numpy())
            domain_labels.extend([0] * z.size(0))

        # Real / PTB-XL
        for batch in loader_real:
            x = batch[0].to(device)
            logits, z = model(x, task="ptbxl", return_features=True)
            features_list.append(z.cpu().numpy())
            domain_labels.extend([1] * z.size(0))

    features = np.concatenate(features_list, axis=0)
    domain_labels = np.array(domain_labels)
    print(f"Total features: {features.shape[0]} samples, dim={features.shape[1]}")

    # ---------------------------------------------------------------------
    # 4. Run t-SNE
    # ---------------------------------------------------------------------
    print("Running t-SNE (this may take a bit) ...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        init="pca",
        learning_rate="auto",
    )
    z_embedded = tsne.fit_transform(features)

    # ---------------------------------------------------------------------
    # 5. Plot and save
    # ---------------------------------------------------------------------
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output

    plt.figure(figsize=(10, 8))

    mask_syn = (domain_labels == 0)
    mask_real = (domain_labels == 1)

    plt.scatter(
        z_embedded[mask_syn, 0],
        z_embedded[mask_syn, 1],
        c="blue",
        label="Synthetic (MedalCare)",
        alpha=0.5,
        s=20,
    )
    plt.scatter(
        z_embedded[mask_real, 0],
        z_embedded[mask_real, 1],
        c="red",
        label="Real (PTB-XL)",
        alpha=0.5,
        s=20,
    )

    plt.title(args.title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ t-SNE plot saved to {out_path}")


if __name__ == "__main__":
    main()
