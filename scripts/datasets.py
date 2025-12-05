import ast
from pathlib import Path
from typing import Dict, Optional, Sequence, Type

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy import signal
from scipy.signal import resample
from torch.utils.data import Dataset

class LVEF_12lead_cls_Dataset_Marta(Dataset): 
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = 0
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -1]
        labels = labels.astype(np.float32)
        data = [wfdb.rdsamp(self.ecg_path+hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        result = self.check_nan_in_array(data)
        if result != 0:
            print(hash_file_name)
        data = data.squeeze(0) 
        data = np.transpose(data,  (1, 0))
        data = data[self.lead_indices, :]
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # Convert to torch tensors
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:  
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:  
            labels = labels.unsqueeze(1)
        return signal, labels   
    

class LVEF_12lead_cls_Dataset(Dataset):
    def __init__(
        self,
        ecg_path,
        labels_df,
        transform=None,
        include_metadata: bool = False,
        domain_column: Optional[str] = None,
        domain_map: Optional[dict] = None,
    ):
        """
        Args:
            ecg_path (str): Base path for ECG files (usually empty if wfdb_path is absolute).
            labels_df (DataFrame): DataFrame containing the annotations.
                Expected columns:
                - 'wfdb_path': Path to WFDB file (without .hea/.dat extension)
                - 'label_0' to 'label_7': One-hot encoded labels (8 binary columns)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]
        self.include_metadata = include_metadata
        self.domain_column = domain_column
        self.domain_map = domain_map
        if self.include_metadata and self.domain_column and self.domain_column not in labels_df.columns:
            raise ValueError(f"Domain column '{self.domain_column}' not found in labels dataframe.")
        
        # Extract label column names (handle both new manifest format and old format)
        self.label_cols = [f'label_{i}' for i in range(8)]
        if all(col in labels_df.columns for col in self.label_cols):
            # New manifest format with named columns
            self.use_named_columns = True
        else:
            # Fallback to old format (last column or last 8 columns)
            self.use_named_columns = False

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get WFDB file path
        if 'wfdb_path' in self.labels_df.columns:
            # New manifest format
            wfdb_path = str(self.labels_df.iloc[idx]['wfdb_path'])
        else:
            # Old format: assume column 1 contains the path
            wfdb_path = str(self.labels_df.iloc[idx, 1])
        
        # Construct full path
        # If wfdb_path is absolute (starts with / or C:\), use it directly
        # Otherwise, prepend ecg_path
        from pathlib import Path
        wfdb_path_obj = Path(wfdb_path)
        if wfdb_path_obj.is_absolute():
            full_path = str(wfdb_path_obj)
        elif self.ecg_path:
            full_path = str(Path(self.ecg_path) / wfdb_path)
        else:
            full_path = wfdb_path
        
        # Load WFDB data
        try:
            data = wfdb.rdsamp(full_path)
            signal_data, meta = data
            data = np.array([signal_data])
        except Exception as e:
            print(f"[ERROR] Failed to load WFDB file {full_path}: {e}")
            raise
        
        data = np.nan_to_num(data, nan=0)
        result = self.check_nan_in_array(data)
        if result != 0:
            print(f"[WARN] NaN values found in {full_path}")
        
        data = data.squeeze(0) 
        data = np.transpose(data, (1, 0))  # Transpose to (leads, time)
        data = data[self.lead_indices, :]  # Reorder leads
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # Get labels (one-hot encoded: 8 binary values)
        if self.use_named_columns:
            labels = self.labels_df.iloc[idx][self.label_cols].values.astype(np.float32)
        else:
            # Fallback: try to get last 8 columns, or last column if single value
            if len(self.labels_df.columns) >= 8:
                labels = self.labels_df.iloc[idx, -8:].values.astype(np.float32)
            else:
                # Single label value (old format)
                labels = np.array([self.labels_df.iloc[idx, -1]], dtype=np.float32)
        
        # Convert to torch tensor
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:  
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:  
            # Ensure it's a row vector for multi-label classification
            labels = labels.unsqueeze(0) if labels.shape[0] == 1 else labels
        
        if self.include_metadata and self.domain_column and self.domain_column in self.labels_df.columns:
            domain_value = self.labels_df.iloc[idx][self.domain_column]
            if self.domain_map is not None:
                domain_id = int(self.domain_map.get(domain_value, -1))
            else:
                domain_id = -1
            domain_tensor = torch.tensor(domain_id, dtype=torch.long)
            return signal, labels, domain_tensor
        return signal, labels     
    
class LVEF_12lead_reg_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -2]
        labels = torch.tensor([labels], dtype=torch.float32)  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        return signal, labels     

class LVEF_1lead_cls_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        result = 0
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -1]
        labels = labels.astype(np.float32)
        data = [wfdb.rdsamp(self.ecg_path+hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        result = self.check_nan_in_array(data)
        if result != 0:
            print(hash_file_name)
        data = data.squeeze(0) 
        data = np.transpose(data,  (1, 0))
        data = data[self.lead_indices, :]
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        # Convert to torch tensors
        labels = torch.tensor(labels, dtype=torch.float)
        if labels.dim() == 0:  
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1:  
            labels = labels.unsqueeze(1)
        return signal, labels  
    
class LVEF_1lead_reg_Dataset(Dataset):
    def __init__(self, ecg_path, labels_df, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the annotations.
            data_dir (str): Directory path containing the numpy data files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.transform = transform
        self.ecg_path = ecg_path
        self.input_leads = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.new_leads = ['I']
        self.lead_indices = [self.input_leads.index(lead) for lead in self.new_leads]

    def __len__(self):
        return len(self.labels_df)

    def z_score_normalization(self,signal):
        return (signal - np.mean(signal)) / (np.std(signal) +1e-8) 

    def check_nan_in_array(self, arr):
        contains_nan = np.isnan(arr).any()
        return contains_nan
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        hash_file_name = str(self.labels_df.iloc[idx, 1])
        labels = self.labels_df.iloc[idx, -2]
        labels = torch.tensor([labels], dtype=torch.float32)  # Wrap the label in a list to create an extra dimension
        data = [wfdb.rdsamp(self.ecg_path + hash_file_name)]
        data = np.array([signal for signal, meta in data])
        data = np.nan_to_num(data, nan=0)
        data = data.squeeze(0)
        data = np.transpose(data, (1, 0))
        data = data[self.lead_indices, :]
        signal = self.z_score_normalization(data)
        signal = torch.FloatTensor(signal)

        return signal, labels     


class PTBXLDataset(Dataset):
    """Dataset wrapper for the PTB-XL diagnostic ECG collection."""

    TARGET_LEADS: Sequence[str] = (
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    )
    SUPERCLASS_LABELS: Sequence[str] = ("NORM", "MI", "STTC", "HYP", "CD")
    OFFICIAL_SPLITS: Dict[str, Sequence[int]] = {
        "train": tuple(range(1, 9)),
        "val": (9,),
        "test": (10,),
        "trainval": tuple(range(1, 10)),
        "all": tuple(range(1, 11)),
    }

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        sampling_rate: int = 500,
        signal_duration: float = 10.0,
        use_high_res: bool = True,
        fold_indices: Optional[Sequence[int]] = None,
        transform=None,
        return_metadata: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split.lower()
        self.sampling_rate = sampling_rate
        self.signal_duration = signal_duration
        self.target_num_samples = (
            int(round(self.sampling_rate * self.signal_duration))
            if self.sampling_rate and self.signal_duration
            else None
        )
        self.use_high_res = use_high_res
        self.transform = transform
        self.return_metadata = return_metadata

        self.database_path = self.root / "ptbxl_database.csv"
        self.scp_statements_path = self.root / "scp_statements.csv"
        if not self.database_path.exists():
            raise FileNotFoundError(f"Missing ptbxl_database.csv under {self.root}")
        if not self.scp_statements_path.exists():
            raise FileNotFoundError(f"Missing scp_statements.csv under {self.root}")

        self.database = pd.read_csv(self.database_path)
        self.scp_statements = pd.read_csv(self.scp_statements_path)
        if "scp_code" not in self.scp_statements.columns:
            unnamed_cols = [
                col for col in self.scp_statements.columns if not col or col.startswith("Unnamed")
            ]
            if unnamed_cols:
                self.scp_statements = self.scp_statements.rename(columns={unnamed_cols[0]: "scp_code"})
        if "scp_code" not in self.scp_statements.columns:
            raise ValueError("Column 'scp_code' missing from scp_statements.csv")

        self.superclass_to_index = {
            label: idx for idx, label in enumerate(self.SUPERCLASS_LABELS)
        }
        self.code_to_superclass = self._build_code_superclass_map()

        folds = self._resolve_folds(fold_indices)
        self.records = (
            self.database[self.database["strat_fold"].isin(folds)]
            .reset_index(drop=True)
        )
        if self.records.empty:
            raise ValueError(
                f"No PTB-XL samples found for split '{self.split}' (folds={folds})."
            )

        self.file_column = "filename_hr" if self.use_high_res else "filename_lr"
        if self.file_column not in self.records.columns:
            raise ValueError(
                f"Column '{self.file_column}' missing from PTB-XL metadata."
            )

        self.targets = self._build_targets(self.records["scp_codes"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        row = self.records.iloc[idx]
        wfdb_path = self.root / row[self.file_column]
        signal = self._load_signal(wfdb_path)
        if self.transform is not None:
            signal = self.transform(signal)
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        metadata = {
            "ecg_id": int(row["ecg_id"]),
            "patient_id": int(row["patient_id"]),
            "strat_fold": int(row["strat_fold"]),
            "scp_codes": row["scp_codes"],
            "path": str(wfdb_path),
        }
        if self.return_metadata:
            return signal_tensor, label_tensor, metadata
        return signal_tensor, label_tensor

    def _resolve_folds(self, fold_indices: Optional[Sequence[int]]) -> Sequence[int]:
        if fold_indices is not None:
            folds = tuple(int(fold) for fold in fold_indices)
            if not folds:
                raise ValueError("fold_indices must contain at least one fold id.")
            return folds
        if self.split not in self.OFFICIAL_SPLITS:
            available = ", ".join(self.OFFICIAL_SPLITS)
            raise ValueError(
                f"Unsupported split '{self.split}'. Choose from: {available}."
            )
        return self.OFFICIAL_SPLITS[self.split]

    def _build_code_superclass_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for _, row in self.scp_statements.iterrows():
            code = str(row["scp_code"]).strip().upper()
            diag_class = str(row.get("diagnostic_class", "")).strip().upper()
            if not code or diag_class not in self.SUPERCLASS_LABELS:
                continue
            mapping[code] = diag_class
        if not mapping:
            raise ValueError("Failed to derive any SCP → diagnostic class mappings.")
        return mapping

    def _build_targets(self, scp_series: pd.Series) -> np.ndarray:
        targets = np.zeros(
            (len(scp_series), len(self.SUPERCLASS_LABELS)), dtype=np.float32
        )
        for idx, raw_codes in enumerate(scp_series):
            for diag_class in self._codes_to_superclasses(raw_codes):
                targets[idx, self.superclass_to_index[diag_class]] = 1.0
        return targets

    def _codes_to_superclasses(self, raw_codes) -> Sequence[str]:
        parsed = self._parse_scp_codes(raw_codes)
        classes = []
        for code in parsed.keys():
            diag_class = self.code_to_superclass.get(str(code).upper())
            if diag_class:
                classes.append(diag_class)
        return classes

    @staticmethod
    def _parse_scp_codes(raw_codes):
        if isinstance(raw_codes, dict):
            return raw_codes
        if raw_codes is None or (isinstance(raw_codes, float) and np.isnan(raw_codes)):
            return {}
        if isinstance(raw_codes, str):
            try:
                parsed = ast.literal_eval(raw_codes)
                if isinstance(parsed, dict):
                    return parsed
            except (SyntaxError, ValueError):
                return {}
        return {}

    def _load_signal(self, wfdb_path: Path) -> np.ndarray:
        base_path = Path(wfdb_path)
        if not base_path.exists():
            hea_path = base_path.with_suffix(".hea")
            dat_path = base_path.with_suffix(".dat")
            if not (hea_path.exists() and dat_path.exists()):
                raise FileNotFoundError(
                    f"WFDB record not found: {base_path} "
                    f"(expected {hea_path.name} / {dat_path.name})"
                )
        signal_data, meta = wfdb.rdsamp(str(base_path))
        signal = np.asarray(signal_data, dtype=np.float32).T  # (leads, time)
        signal = self._reorder_leads(signal, meta.get("sig_name", []))
        signal = np.nan_to_num(signal, nan=0.0)
        signal = self._match_sampling(signal, meta.get("fs"))
        signal = self._match_length(signal)
        return self._z_score(signal)

    def _reorder_leads(
        self, signal: np.ndarray, source_leads: Sequence[str]
    ) -> np.ndarray:
        if not source_leads:
            raise ValueError(
                "WFDB metadata did not include lead names (sig_name missing)."
            )
        name_to_idx = {name.upper(): idx for idx, name in enumerate(source_leads)}
        ordered = []
        for lead in self.TARGET_LEADS:
            if lead.upper() not in name_to_idx:
                raise ValueError(f"Lead '{lead}' missing from WFDB record.")
            ordered.append(signal[name_to_idx[lead.upper()]])
        return np.stack(ordered, axis=0)

    def _match_sampling(self, signal: np.ndarray, source_fs: Optional[float]) -> np.ndarray:
        if self.sampling_rate is None or source_fs is None:
            return signal
        if np.isclose(source_fs, self.sampling_rate):
            return signal
        target_len = int(
            round(
                signal.shape[1] * (self.sampling_rate / max(source_fs, 1e-8))
            )
        )
        return resample(signal, target_len, axis=1)

    def _match_length(self, signal: np.ndarray) -> np.ndarray:
        if self.target_num_samples is None:
            return signal
        current = signal.shape[1]
        if current == self.target_num_samples:
            return signal
        if current > self.target_num_samples:
            return signal[:, : self.target_num_samples]
        pad_width = self.target_num_samples - current
        return np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")

    @staticmethod
    def _z_score(signal: np.ndarray) -> np.ndarray:
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        return (signal - mean) / (std + 1e-8)


DATASET_REGISTRY: Dict[str, Type[Dataset]] = {
    "medalcare": LVEF_12lead_cls_Dataset,
    "ptbxl": PTBXLDataset,
}


def get_dataset(name: str, **kwargs) -> Dataset:
    """Instantiate a dataset by name."""
    key = name.lower()
    if key not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
    dataset_cls = DATASET_REGISTRY[key]
    return dataset_cls(**kwargs)