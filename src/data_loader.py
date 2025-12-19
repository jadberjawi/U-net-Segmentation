from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularDataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list


class CardioDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_kaggle_cardio_bundle(
    csv_path: str,
    sep: str,
    target_col: str,
    id_col: str | None,
    test_size: float,
    seed: int,
    stratify: bool,
) -> TabularDataBundle:

    df = pd.read_csv(csv_path, sep=sep)

    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])

    if target_col not in df.columns:
        raise ValueError(f"Target col '{target_col}' not found. Columns: {list(df.columns)}")

    y = df[target_col].astype(int).to_numpy()
    X_df = df.drop(columns=[target_col])

    # All columns are numeric/ordinal in this dataset
    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float32)

    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=strat,
    )

    # Standardize using train statistics only (no leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return TabularDataBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.astype(np.float32),
        y_test=y_test.astype(np.float32),
        feature_names=feature_names,
    )
