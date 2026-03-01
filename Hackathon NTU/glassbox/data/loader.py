import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from data.feature_groups import FEATURE_NAMES, CHUNK_GROUPS


class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_heart_disease(test_size=0.2, random_state=42, batch_size=32):
    """
    Fetch UCI Heart Disease (Cleveland) via ucimlrepo and return
    train/test DataLoaders plus the scaler for later inference.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        X_raw = heart.data.features
        y_raw = heart.data.targets
    except Exception as e:
        print(f"ucimlrepo fetch failed ({e}), falling back to local CSV if available")
        X_raw, y_raw = _fallback_load()

    # Rename columns to our canonical names if needed
    col_map = {
        'age': 'age', 'sex': 'sex', 'cp': 'cp',
        'trestbps': 'trestbps', 'chol': 'chol', 'thalach': 'thalach',
        'fbs': 'fbs', 'restecg': 'restecg', 'oldpeak': 'oldpeak',
        'slope': 'slope', 'ca': 'ca', 'thal': 'thal', 'exang': 'exang',
    }
    X_raw = X_raw.rename(columns=col_map)
    X_raw = X_raw[FEATURE_NAMES]

    # Binary target: 0 = no disease, 1 = disease (any severity)
    if hasattr(y_raw, 'values'):
        y_vals = y_raw.values.ravel()
    else:
        y_vals = np.array(y_raw).ravel()
    y_binary = (y_vals > 0).astype(int)

    # Drop rows with NaN (a few in `ca` and `thal`)
    X_df = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
    mask = ~X_df.isnull().any(axis=1)
    X_clean = X_df[mask].values.astype(float)
    y_clean = y_binary[mask]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=random_state, stratify=y_clean
    )

    # Standardise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = HeartDiseaseDataset(X_train, y_train)
    test_ds = HeartDiseaseDataset(X_test, y_test)

    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, X_test, y_test


def get_chunk_tensors(x_batch):
    """
    Split a full feature tensor into per-chunk tensors.
    x_batch: (N, 13) tensor
    Returns dict of chunk_name -> (N, chunk_dim) tensor
    """
    chunks = {}
    for chunk_name, info in CHUNK_GROUPS.items():
        idx = info['indices']
        chunks[chunk_name] = x_batch[:, idx]
    return chunks


def _fallback_load():
    """Minimal fallback — raise if data unavailable."""
    raise RuntimeError(
        "Cannot load UCI Heart Disease dataset. "
        "Install ucimlrepo: pip install ucimlrepo"
    )
