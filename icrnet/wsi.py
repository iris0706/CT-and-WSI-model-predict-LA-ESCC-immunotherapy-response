"""Frozen patch encoders and whole-slide tiling for the WSI branch.

Encoders
    uni2h        the pathology foundation model UNI, UNI2-h release (Hugging Face Hub: MahmoodLab/UNI2-h): a ViT-H/14
                 pretrained by self-supervision on histology tiles, 224-pixel input, 1,536-dimensional embedding.
                 The repository is gated: accept the terms on the model page, then run `huggingface-cli login` or set
                 HF_TOKEN. The weights are used frozen and are not redistributed with this code (CC BY-NC-ND 4.0).
    resnet50_l3  ImageNet ResNet-50 truncated after layer3 with global average pooling (1,024-dimensional), the
                 encoder of the earlier pipeline (256-pixel patches at 40x).

Tiling
    Patches are taken at a target resolution in micrometres per pixel. The slide resolution is read from the scanner
    metadata (MPP; the objective power is the fallback), so slides of different native resolution yield patches
    covering the same physical field. A patch counts as tissue when at least MIN_TISSUE_FRACTION of its
    pixels have an HSV saturation above TISSUE_SATURATION; predominantly blank patches are excluded. A low-resolution
    thumbnail skips grid cells that contain almost no tissue before any region is read at full resolution.
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from . import config

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TISSUE_SATURATION = 20          # HSV saturation (0-255) above which a pixel counts as tissue
MIN_TISSUE_FRACTION = 0.5       # patches with less tissue are treated as predominantly blank
PREFILTER_FRACTION = 0.05       # thumbnail tissue fraction below which a grid cell is skipped without reading it
THUMB_PX_PER_PATCH = 4          # thumbnail pixels per patch along each axis
OBJECTIVE_MPP = {20.0: 0.5, 40.0: 0.25, 80.0: 0.125}

ENCODERS = {
    "uni2h": dict(dim=1536, native_mpp=0.5, patch=224, hub="hf-hub:MahmoodLab/UNI2-h", arch="vit_giant_patch14_224"),
    "resnet50_l3": dict(dim=1024, native_mpp=0.25, patch=256),
}

# keyword arguments of the UNI2-h model card (timm >= 1.0.3)
UNI2H_KWARGS = dict(img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5, embed_dim=1536,
                    mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True, reg_tokens=8, dynamic_img_size=True)


def _build_uni2h(pretrained):
    import timm
    from timm.layers import SwiGLUPacked
    kwargs = dict(UNI2H_KWARGS, mlp_layer=SwiGLUPacked, act_layer=nn.SiLU)
    if pretrained:
        return timm.create_model(ENCODERS["uni2h"]["hub"], pretrained=True, **kwargs)
    return timm.create_model(ENCODERS["uni2h"]["arch"], pretrained=False, **kwargs)     # architecture only (tests)


def _build_resnet50_l3(pretrained):
    from torchvision.models import ResNet50_Weights, resnet50
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    return nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3,
                         nn.AdaptiveAvgPool2d(1), nn.Flatten())


class PatchEncoder(nn.Module):
    """Frozen encoder with its preprocessing: uint8 RGB patches (B, H, W, 3) -> float32 embeddings (B, dim)."""

    def __init__(self, name=config.WSI_ENCODER, pretrained=True, fp16=False):
        super().__init__()
        if name not in ENCODERS:
            raise KeyError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
        self.name, self.dim = name, ENCODERS[name]["dim"]
        self.net = _build_uni2h(pretrained) if name == "uni2h" else _build_resnet50_l3(pretrained)
        self.net.eval().requires_grad_(False)
        self.fp16 = fp16
        if fp16:
            self.net.half()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, patches):
        x = torch.as_tensor(np.asarray(patches)).to(self.mean.device).permute(0, 3, 1, 2).float() / 255
        x = (x - self.mean) / self.std
        out = self.net(x.half() if self.fp16 else x)
        return out.float()

    def embed(self, images):
        """List of PIL patches -> numpy (n, dim) float32."""
        return self(np.stack([np.asarray(im.convert("RGB")) for im in images])).cpu().numpy().astype(np.float32)


def slide_mpp(slide):
    """(um per pixel at level 0, source): from the MPP metadata, or from the objective power when the MPP is missing."""
    import openslide
    props = slide.properties
    mpp = props.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp:
        return float(mpp), "mpp"
    power = props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if power and float(power) in OBJECTIVE_MPP:
        return OBJECTIVE_MPP[float(power)], "objective power"
    raise ValueError("slide metadata lack both MPP and a known objective power; pass --slide-mpp")


def tissue_fraction(img):
    sat = np.asarray(img.convert("HSV"))[..., 1]
    return float((sat > TISSUE_SATURATION).mean())


def _candidate_cells(slide, step, cols, rows):
    """Boolean (rows, cols) mask of grid cells whose thumbnail shows at least PREFILTER_FRACTION tissue."""
    k = THUMB_PX_PER_PATCH
    thumb = slide.get_thumbnail((cols * k, rows * k)).convert("RGB")
    if thumb.size != (cols * k, rows * k):
        thumb = thumb.resize((cols * k, rows * k), Image.BILINEAR)
    sat = np.asarray(thumb.convert("HSV"))[..., 1] > TISSUE_SATURATION
    sat = sat[:rows * k, :cols * k].reshape(rows, k, cols, k).mean(axis=(1, 3))
    return sat >= PREFILTER_FRACTION


def iter_patches(slide, target_mpp=config.WSI_TARGET_MPP, patch=config.WSI_PATCH_SIZE, slide_mpp0=None):
    """Yield (x, y, RGB PIL patch) for non-overlapping tissue patches; x, y are level-0 coordinates."""
    mpp0 = slide_mpp0 or slide_mpp(slide)[0]
    step = int(round(patch * target_mpp / mpp0))            # level-0 footprint of one patch
    level = slide.get_best_level_for_downsample(step / patch)
    size = int(round(step / slide.level_downsamples[level]))  # footprint at the level that is read
    width, height = slide.dimensions
    cols, rows = (width - step) // step + 1, (height - step) // step + 1
    if cols < 1 or rows < 1:
        return
    candidates = _candidate_cells(slide, step, cols, rows)
    for j in range(rows):
        for i in range(cols):
            if not candidates[j, i]:
                continue
            x, y = i * step, j * step
            img = slide.read_region((x, y), level, (size, size)).convert("RGB")
            if size != patch:
                img = img.resize((patch, patch), Image.BILINEAR)
            if tissue_fraction(img) >= MIN_TISSUE_FRACTION:
                yield x, y, img
