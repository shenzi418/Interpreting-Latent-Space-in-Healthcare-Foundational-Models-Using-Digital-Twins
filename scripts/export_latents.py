import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wfdb

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from net1d import Net1D


@dataclass
class HeadSpec:
    head_type: str
    output_dim: int
    hidden_dim: Optional[int] = None


class DenseInputHook:
    def __init__(self, module: nn.Module) -> None:
        self._buffer: Optional[torch.Tensor] = None
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:  # pragma: no cover - hook signature
        if not inputs:
            self._buffer = None
        else:
            self._buffer = inputs[0].detach()

    def pop(self) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError("Latent features were not captured for the latest batch.")
        features = self._buffer
        self._buffer = None
        return features

    def close(self) -> None:
        self._handle.remove()


class ECGWaveformDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        n_classes: int,
        label_columns: Sequence[str],
        base_dir: Path,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.n_classes = n_classes
        self.label_columns = list(label_columns)
        self.base_dir = base_dir
        self.input_leads = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.new_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_wfdb_path(self, raw_path: str) -> Path:
        wfdb_path = Path(raw_path)
        if wfdb_path.is_absolute():
            return wfdb_path
        return (self.base_dir / wfdb_path).resolve()

    def _normalise(self, signal: np.ndarray) -> np.ndarray:
        mean = signal.mean()
        std = signal.std()
        return (signal - mean) / (std + 1e-8)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        wfdb_path = self._resolve_wfdb_path(str(row["wfdb_path"]))
        try:
            signal_matrix, _ = wfdb.rdsamp(str(wfdb_path))
        except Exception as exc:  # pragma: no cover - relies on runtime data
            raise RuntimeError(f"Failed to load WFDB record at {wfdb_path}") from exc

        signal_matrix = np.nan_to_num(signal_matrix, nan=0.0)
        signal_matrix = signal_matrix.T  # (leads, time)
        signal_matrix = signal_matrix[self.lead_indices, :]
        signal_matrix = self._normalise(signal_matrix)
        signal_tensor = torch.from_numpy(signal_matrix.astype(np.float32))

        if self.label_columns:
            labels = row[self.label_columns].to_numpy(dtype=np.float32)
        else:
            labels = np.zeros(self.n_classes, dtype=np.float32)
        label_tensor = torch.from_numpy(labels)
        return signal_tensor, label_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export latent features and predictions for ECGFounder checkpoints."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt/.pth).")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest CSV describing records.")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=("train", "val", "test", "all"),
        help="Subset of the manifest to export. Use 'all' to keep every row.",
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Directory to store latents and indexes.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (0 on Windows).")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g. 'cuda', 'cuda:0', 'cpu'). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    if "wfdb_path" not in df.columns:
        raise ValueError("Manifest must contain a 'wfdb_path' column.")
    return df


def filter_manifest(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return df.copy()
    if "split" not in df.columns:
        raise ValueError("Manifest is missing 'split' column; cannot filter by split.")
    mask = df["split"].astype(str).str.lower() == split.lower()
    subset = df.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"No records found for split='{split}'.")
    return subset


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            clean_state[key[len("module."):]] = value
        else:
            clean_state[key] = value
    return clean_state


def infer_head_spec(state_dict: Dict[str, torch.Tensor]) -> HeadSpec:
    if "dense.weight" in state_dict and "dense.bias" in state_dict:
        output_dim = state_dict["dense.weight"].shape[0]
        return HeadSpec(head_type="linear", output_dim=output_dim)
    if "dense.2.weight" in state_dict and "dense.2.bias" in state_dict and "dense.0.weight" in state_dict:
        output_dim = state_dict["dense.2.weight"].shape[0]
        hidden_dim = state_dict["dense.0.weight"].shape[0]
        return HeadSpec(head_type="mlp", output_dim=output_dim, hidden_dim=hidden_dim)
    raise ValueError("Unable to infer head structure from checkpoint state_dict.")


def build_model(head_spec: HeadSpec, device: torch.device) -> Net1D:
    model = Net1D(
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
        n_classes=head_spec.output_dim,
    )
    in_features = model.dense.in_features
    if head_spec.head_type == "mlp":
        if head_spec.hidden_dim is None:
            raise ValueError("Missing hidden dimension for MLP head.")
        model.dense = nn.Sequential(
            nn.Linear(in_features, head_spec.hidden_dim),
            nn.ReLU(),
            nn.Linear(head_spec.hidden_dim, head_spec.output_dim),
        )
    elif head_spec.head_type == "linear":
        model.dense = nn.Linear(in_features, head_spec.output_dim)
    else:  # pragma: no cover - safeguarded by infer_head_spec
        raise ValueError(f"Unsupported head type: {head_spec.head_type}")
    model.to(device)
    return model


def collect_arrays(
    model: Net1D,
    loader: DataLoader,
    device: torch.device,
    expect_labels: bool,
    feature_hook: DenseInputHook,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    latent_batches: List[np.ndarray] = []
    prob_batches: List[np.ndarray] = []
    label_batches: List[np.ndarray] = []
    with torch.no_grad():
        for signals, labels in tqdm(loader, desc="Exporting", leave=False):
            signals = signals.to(device, non_blocking=True)
            logits = model(signals)
            features = feature_hook.pop()
            latent_batches.append(features.cpu().numpy())
            probabilities = torch.sigmoid(logits)
            prob_batches.append(probabilities.cpu().numpy())
            if expect_labels:
                label_batches.append(labels.cpu().numpy())
    z_array = np.concatenate(latent_batches, axis=0) if latent_batches else np.zeros((0, 0), dtype=np.float32)
    p_array = np.concatenate(prob_batches, axis=0) if prob_batches else np.zeros((0, 0), dtype=np.float32)
    y_array = None
    if expect_labels and label_batches:
        y_array = np.concatenate(label_batches, axis=0)
    return z_array, p_array, y_array


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    checkpoint_obj = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
    elif isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
    else:
        raise ValueError("Unsupported checkpoint format. Expected a dict with 'state_dict'.")
    state_dict = strip_module_prefix(state_dict)

    head_spec = infer_head_spec(state_dict)
    model = build_model(head_spec, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch. Missing: {missing}, Unexpected: {unexpected}")

    manifest_df = load_manifest(args.manifest)
    subset_df = filter_manifest(manifest_df, args.split)
    subset_df = subset_df.reset_index(drop=True)
    print(f"Loaded {len(subset_df)} records from split '{args.split}'.")

    label_columns = [col for col in subset_df.columns if col.startswith("label_")]
    has_labels = len(label_columns) > 0

    dataset = ECGWaveformDataset(
        dataframe=subset_df,
        n_classes=head_spec.output_dim,
        label_columns=label_columns,
        base_dir=args.manifest.parent.resolve(),
    )
    feature_hook = DenseInputHook(model.dense)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    try:
        z_array, p_array, y_array = collect_arrays(model, loader, device, has_labels, feature_hook)
    finally:
        feature_hook.close()
    if z_array.shape[0] != len(subset_df):
        raise RuntimeError(
            f"Mismatch between exported features ({z_array.shape[0]}) and manifest rows ({len(subset_df)})."
        )

    theta_raw_cols = [col for col in subset_df.columns if col.lower().startswith("theta_raw")]
    theta_cols = [col for col in subset_df.columns if col.lower().startswith("theta") and col not in theta_raw_cols]

    export_dir = args.outdir.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    latents_payload = {"Z": z_array, "P": p_array}
    if has_labels and y_array is not None:
        latents_payload["Y"] = y_array
    if theta_raw_cols:
        latents_payload["Theta_raw"] = subset_df[theta_raw_cols].to_numpy()
    if theta_cols:
        latents_payload["Theta"] = subset_df[theta_cols].to_numpy()

    npz_path = export_dir / "latents.npz"
    np.savez_compressed(npz_path, **latents_payload)
    print(f"Wrote latent arrays to {npz_path}")

    index_columns_to_drop = label_columns
    index_df = subset_df.drop(columns=index_columns_to_drop, errors="ignore")
    index_path = export_dir / "index.csv"
    index_df.to_csv(index_path, index=False)
    print(f"Wrote index with metadata to {index_path}")


if __name__ == "__main__":
    main()

