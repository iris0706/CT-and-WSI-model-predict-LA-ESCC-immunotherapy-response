"""ICR-Net and the unimodal CT-only and pathology-only models.

The WSI branch takes the frozen patch embeddings written by scripts/extract_wsi_features.py: 1,536-dimensional
UNI (UNI2-h) embeddings by default, or the 1,024-dimensional ImageNet ResNet-50 features of the earlier pipeline.
The embedding dimension is a constructor argument (`wsi_dim`) and is read back from a checkpoint by
`checkpoint_wsi_dim`, so weights trained with either encoder load (`model_from_checkpoint`).

Parameter names follow the original implementation (model_images.*, dimReduction.*, attention.*, fc),
so checkpoints trained with it load into these classes; see `load_checkpoint`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import R3D_18_Weights, r3d_18

from . import config


class MultiScaleSpatialAttention3D(nn.Module):
    """Global average and max pooling, parallel 1x1x1 / 3x3x3 / 5x5x5 3D convolutions with sigmoid,
    summed and multiplied element-wise with the input."""

    def __init__(self, in_channels=3):
        super().__init__()
        pooled = 2 * in_channels
        self.conv1 = nn.Conv3d(pooled, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv3d(pooled, 1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(pooled, 1, kernel_size=5, padding=2)

    def forward(self, x):
        pooled = torch.cat([F.adaptive_avg_pool3d(x, 1), F.adaptive_max_pool3d(x, 1)], dim=1)
        weights = torch.sigmoid(self.conv1(pooled)) + torch.sigmoid(self.conv2(pooled)) + torch.sigmoid(self.conv3(pooled))
        return x * weights


class ChannelAttention3D(nn.Module):
    """Adaptive pooling to 1x1x1 followed by two 1x1x1 convolutions with ReLU and sigmoid activations.
    The average- and max-pooled descriptors share the two convolutions and their outputs are summed."""

    def __init__(self, channels=3, hidden=3):
        super().__init__()
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)

    def _branch(self, pooled):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))

    def forward(self, x):
        return x * (self._branch(F.adaptive_avg_pool3d(x, 1)) + self._branch(F.adaptive_max_pool3d(x, 1)))


class ResNet3DEncoder(nn.Module):
    """3D ResNet-18 pretrained on Kinetics-400, without its pooling and classification head."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)

    def forward(self, x):
        r = self.resnet
        return r.layer4(r.layer3(r.layer2(r.layer1(r.stem(x)))))


class CTBranch(nn.Module):
    """CT volume (B, 3, 5, 128, 128) -> 512-dimensional CT vector."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.attention = MultiScaleSpatialAttention3D(in_channels=3)
        self.ca = ChannelAttention3D(channels=3, hidden=3)
        self.encoder = ResNet3DEncoder(pretrained)
        self.bn = nn.BatchNorm3d(config.FEATURE_DIM)

    def forward(self, x):
        x = self.ca(self.attention(x))
        x = self.bn(self.encoder(x))
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


class DimReduction(nn.Module):
    """Fully connected layer with ReLU reducing each patch embedding (wsi_dim) to 512 dimensions."""

    def __init__(self, in_dim=config.WSI_FEATURE_DIM, out_dim=config.FEATURE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return F.relu(self.fc1(x))


class WSIBranch(nn.Module):
    """List of patch-embedding matrices (n_i, wsi_dim) -> (B, 512) patient-level vectors by mean pooling."""

    def __init__(self, in_dim=config.WSI_FEATURE_DIM):
        super().__init__()
        self.dimReduction = DimReduction(in_dim)

    def forward(self, bags):
        return torch.stack([self.dimReduction(bag).mean(dim=0) for bag in bags])


class GatedAttention(nn.Module):
    """Gated attention over the stacked modality vectors: tanh pathway 512->128, sigmoid gating
    pathway 512->128, element-wise product, linear 128->1, softmax over modalities."""

    def __init__(self, dim=config.FEATURE_DIM, hidden=128):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden, 1)

    def forward(self, feats):
        # feats: (B, M, 512) -> modality weights (B, M)
        scores = self.attention_weights(self.attention_V(feats) * self.attention_U(feats)).squeeze(-1)
        return torch.softmax(scores, dim=1)


class ICRNet(nn.Module):
    """CT branch + WSI branch + gated-attention fusion; outputs the two ReLU-activated outputs of the final layer
    (B, 2), to which the softmax is applied, and the modality weights (B, 2).

    As in the implementation that generated the published predictions, a ReLU is applied to the outputs of the
    final fully connected layer before the softmax, and no dropout is used.
    """

    def __init__(self, pretrained=True, dropout=config.DROPOUT["fusion"], wsi_dim=config.WSI_FEATURE_DIM):
        super().__init__()
        self.model_images = CTBranch(pretrained)
        self.wsi = WSIBranch(wsi_dim)
        self.attention = GatedAttention()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(config.FEATURE_DIM, config.NUM_CLASSES)

    def forward(self, ct, bags):
        feats = torch.stack([self.model_images(ct), self.wsi(bags)], dim=1)   # (B, 2, 512)
        weights = self.attention(feats)
        fused = (weights.unsqueeze(-1) * feats).sum(dim=1)                   # (B, 512)
        return F.relu(self.fc(self.dropout(fused))), weights


class CTOnlyNet(nn.Module):
    """CT branch with the same classification head; trained under the same protocol as ICR-Net."""

    def __init__(self, pretrained=True, dropout=config.DROPOUT["ct"], wsi_dim=None):
        super().__init__()
        self.model_images = CTBranch(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(config.FEATURE_DIM, config.NUM_CLASSES)

    def forward(self, ct, bags=None):
        return self.fc(self.dropout(self.model_images(ct))), None


class PathologyOnlyNet(nn.Module):
    """WSI branch with the same classification head; trained under the same protocol as ICR-Net."""

    def __init__(self, pretrained=True, dropout=config.DROPOUT["wsi"], wsi_dim=config.WSI_FEATURE_DIM):
        super().__init__()
        self.wsi = WSIBranch(wsi_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(config.FEATURE_DIM, config.NUM_CLASSES)

    def forward(self, ct, bags):
        return self.fc(self.dropout(self.wsi(bags))), None


MODELS = {"fusion": ICRNet, "ct": CTOnlyNet, "wsi": PathologyOnlyNet}


def build_model(modality, pretrained=True, wsi_dim=config.WSI_FEATURE_DIM):
    return MODELS[modality](pretrained=pretrained, wsi_dim=wsi_dim)


# Keys of the original implementation that have a different name here or no counterpart.
_LEGACY_RENAMES = {"dimReduction.": "wsi.dimReduction."}
_LEGACY_UNUSED = ("model_images.fc.", "model_images.dropout.")
_WSI_WEIGHT_KEYS = ("wsi.dimReduction.fc1.weight", "dimReduction.fc1.weight")


def _state_dict(path_or_state, map_location="cpu"):
    state = torch.load(path_or_state, map_location=map_location) if isinstance(path_or_state, str) else path_or_state
    return state.get("model_state_dict", state)


def checkpoint_wsi_dim(path_or_state, map_location="cpu"):
    """Patch-embedding dimension a checkpoint was trained with (None for CT-only checkpoints)."""
    state = _state_dict(path_or_state, map_location)
    for key in _WSI_WEIGHT_KEYS:
        if key in state:
            return int(state[key].shape[1])
    return None


def load_checkpoint(model, path_or_state, map_location="cpu"):
    """Load a checkpoint (path or loaded dict) saved by this code or by the original implementation."""
    state = _state_dict(path_or_state, map_location)
    renamed = {}
    for key, value in state.items():
        if key.startswith(_LEGACY_UNUSED):
            continue
        for old, new in _LEGACY_RENAMES.items():
            if key.startswith(old):
                key = new + key[len(old):]
        renamed[key] = value
    model.load_state_dict(renamed, strict=True)
    return model


def model_from_checkpoint(modality, path_or_state, map_location="cpu", pretrained=False):
    """Build the model with the patch-embedding dimension stored in the checkpoint and load its weights."""
    state = torch.load(path_or_state, map_location=map_location) if isinstance(path_or_state, str) else path_or_state
    wsi_dim = checkpoint_wsi_dim(state) or config.WSI_FEATURE_DIM
    return load_checkpoint(build_model(modality, pretrained=pretrained, wsi_dim=wsi_dim), state, map_location)
