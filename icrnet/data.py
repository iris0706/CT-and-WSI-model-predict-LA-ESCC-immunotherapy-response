"""Manifest, datasets and collation.

The manifest is a CSV with one row per patient:
    patient_id, center, cohort, label, ct_npy, wsi_features
cohort is one of training, internal_test, external_test, prospective_test (Methods, Study design and participants):
    only `training` rows are used for training and for the Youden threshold; each test cohort is evaluated once
    with the fixed model and threshold (scripts/predict.py, scripts/evaluate.py);
label is 1 for iRECIST responders (iCR/iPR) and 0 for non-responders (iSD/iUPD/iCPD);
ct_npy is the cropped, windowed uint8 volume written by scripts/preprocess_ct.py;
wsi_features is the (n_patches, dim) float32 array written by scripts/extract_wsi_features.py
(1,536-d UNI2-h embeddings by default; 1,024-d ResNet-50 features for the earlier pipeline).
Relative paths are resolved against the manifest's directory.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

from . import config

COHORTS = ("training", "internal_test", "external_test", "prospective_test")


def read_manifest(path):
    df = pd.read_csv(path, dtype={"patient_id": str})
    unknown = set(df["cohort"]) - set(COHORTS)
    if unknown:
        raise ValueError(f"unknown cohort labels: {sorted(unknown)}")
    root = os.path.dirname(os.path.abspath(path))
    for col in ("ct_npy", "wsi_features"):
        df[col] = [p if os.path.isabs(p) else os.path.join(root, p) for p in df[col]]
    return df


def load_ct(path, slices):
    """uint8 (D, H, W) -> float (1, slices, 128, 128): each slice resized to 128 x 128, then trilinear
    interpolation along the slice axis. The single channel is replicated to three with to_three_channels."""
    vol = torch.from_numpy(np.load(path).astype(np.float32) / 255.0)[None, None]
    vol = F.interpolate(vol, size=(slices, config.CT_SIZE, config.CT_SIZE), mode="trilinear", align_corners=False)
    return vol[0]


def to_three_channels(ct):
    """(..., 1, D, H, W) -> (..., 3, D, H, W) for the Kinetics-pretrained 3D ResNet."""
    return ct.repeat_interleave(3, dim=-4)


def augment_once(ct):
    """Training augmentation, applied once when the training set is loaded: a rotation within +/-10 degrees
    and a horizontal flip (p = 0.5), identical for all slices (the slices are treated as image channels)."""
    tf = T.Compose([T.RandomAffine(degrees=config.CT_ROTATION), T.RandomHorizontalFlip(p=0.5)])
    return tf(ct[0])[None]


def load_wsi_features(path, dim=None):
    """(n, dim) float32 patch embeddings from .npy, .pt or .h5 (dataset 'features'); dim is checked when given."""
    if path.endswith(".pt"):
        feats = torch.load(path)
    elif path.endswith(".h5"):
        import h5py
        with h5py.File(path, "r") as f:
            feats = torch.from_numpy(np.asarray(f["features"]))
    else:
        feats = torch.from_numpy(np.load(path))
    feats = feats.float()
    if feats.ndim != 2 or (dim is not None and feats.shape[1] != dim):
        raise ValueError(f"{path}: expected (n, {dim or 'dim'}) features, got {tuple(feats.shape)}")
    return feats


def feature_dim(path):
    """Embedding dimension of one feature file (used to size the WSI branch)."""
    return int(load_wsi_features(path).shape[1])


def load_training_pools(df, wsi_dim=None):
    """Training cohort grouped by class for episodic sampling: {label: [(ct (1, 80, 128, 128), wsi (n, dim)), ...]}.
    Each CT volume is augmented once here, as in the original implementation."""
    wsi_dim = wsi_dim or feature_dim(df["wsi_features"].iloc[0])
    pools = {0: [], 1: []}
    for row in df.itertuples():
        ct = augment_once(load_ct(row.ct_npy, config.CT_SLICES_TRAIN))
        pools[int(row.label)].append((ct, load_wsi_features(row.wsi_features, wsi_dim)))
    return pools


class PatientDataset(Dataset):
    """Inference dataset: CT resampled to 5 slices, no augmentation."""

    def __init__(self, df, wsi_dim=None):
        self.df = df.reset_index(drop=True)
        self.wsi_dim = wsi_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ct = to_three_channels(load_ct(row["ct_npy"], config.CT_SLICES_INFER))
        return ct, load_wsi_features(row["wsi_features"], self.wsi_dim), int(row["label"]), row["patient_id"]


def collate(batch):
    cts, bags, labels, ids = zip(*batch)
    return torch.stack(cts), list(bags), torch.tensor(labels, dtype=torch.long), list(ids)
