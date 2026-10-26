# ICR-Net (Immunochemotherapy Response Network)

Multimodal deep-learning model that fuses pretreatment contrast-enhanced CT and endoscopic-biopsy H&E
whole-slide images (WSIs) to predict iRECIST-defined radiographic response to neoadjuvant
immunochemotherapy (ICT) in locally advanced esophageal squamous cell carcinoma.

## Model

- **CT branch**: tumor region of interest (mask bounding box + 10 voxels, soft-tissue window W400/L40,
  slices resized to 128 × 128, resampled to 80 slices for training and 5 for inference, replicated to 3
  channels) → multiscale spatial attention → channel attention → 3D ResNet-18 (Kinetics-400) → batch
  normalization and global average pooling → 512-d vector.
- **WSI branch**: non-overlapping 224 × 224-pixel tissue patches at 0.5 µm per pixel (20× equivalent, the working
  resolution of UNI2-h; each slide is downsampled from its native resolution, read from the scanner metadata, so all
  slides give patches of the same physical field),
  patches with less than half of their pixels tissue (HSV saturation > 20) excluded, ImageNet mean/SD normalization
  → frozen pathology foundation model UNI (UNI2-h release, ViT-H/14, 1,536-d
  embedding) → fully connected layer with ReLU (512-d; the dimension-reduction module of DTFD-MIL) → mean pooling
  over patches → 512-d vector. Only the fully connected layer is trained; the encoder is never fine-tuned.
- **Fusion**: the two vectors are stacked (2 × 512) and weighted by gated attention (tanh and sigmoid
  pathways 512→128, element-wise product, linear 128→1, softmax over modalities); the weighted sum passes
  through a fully connected layer (512→2), whose outputs pass through a ReLU and a softmax. As in the
  implementation that generated the published predictions, the fused model uses no dropout.
- **Unimodal models**: the CT branch or the WSI branch followed by dropout (0.4) and a fully connected layer
  (512→2) with softmax, trained under the same protocol.

## Patch encoder

`icrnet/wsi.py` holds the encoders and the tiling; `scripts/extract_wsi_features.py --encoder` selects one:

| `--encoder` | encoder | patches | embedding |
|---|---|---|---|
| `uni2h` (default) | UNI2-h (Hugging Face Hub `MahmoodLab/UNI2-h`, loaded with timm), frozen | 224 px at 0.5 µm/px | 1,536-d |
| `resnet50_l3` | ImageNet ResNet-50 truncated after layer3 (the earlier pipeline) | 256 px at 0.25 µm/px (`--mpp 0.25 --patch 256`) | 1,024-d |

UNI2-h is a gated repository released under CC BY-NC-ND 4.0: create a Hugging Face account, accept the terms on the
model page, then run `huggingface-cli login` (or set `HF_TOKEN`). timm downloads the weights at first use; they are
not part of this repository and may not be redistributed. The models take the embedding dimension from the feature
files (training) or from the checkpoint (inference), so weights trained with either encoder can be applied, provided
the features come from the same encoder.

## Cohorts

The manifest labels every patient as `training`, `internal_test`, `external_test` or `prospective_test`
(Methods, Study design and participants). The Center 1 development cohort is split 80/20 by stratified random
sampling into `training` and `internal_test` (`scripts/make_split.py`). Only `training` rows are used by
`scripts/train.py` and for the Youden threshold; the three test cohorts are each evaluated once with the fixed
weights and threshold (`scripts/predict.py`, `scripts/evaluate.py`), and no recalibration or updating is performed.

## Training protocol

Episodic training on class-balanced tasks from the training cohort only. Each task draws 5 responders and
5 non-responders as a support set and another 5 of each as a query set; the model takes 5 SGD steps on the
support set (learning rate 1e-3, momentum 0.9, weight decay 1e-3, unweighted cross-entropy) and the query set
only monitors performance. An epoch is 50 tasks; a single model is trained with seed 42. Training CT volumes
are augmented once at loading (rotation within ±10° and a horizontal flip). The probability threshold
(≥ 0.603019) is the training-cohort probability that maximized Youden's index and was then frozen. All
settings are in `icrnet/config.py`; `scripts/train.py` explains how this procedure relates to the original
implementation.

## Usage

```bash
pip install -r requirements.txt
huggingface-cli login            # once, after accepting the UNI2-h terms on https://huggingface.co/MahmoodLab/UNI2-h

# 1. Inputs
python scripts/preprocess_ct.py --input ct_list.csv --out-dir data/ct_npy
python scripts/extract_wsi_features.py --input wsi_list.csv --out-dir data/wsi_features      # UNI2-h, 0.5 µm/px, 224 px

# 2. Manifest (see icrnet/data.py for the columns); split the Center 1 development cohort 80/20 into training / internal_test
python scripts/make_split.py --manifest manifest_unsplit.csv --out manifest.csv

# 3. Training (the number of epochs for the published weights was not recorded)
for m in fusion ct wsi; do
  python scripts/train.py --manifest manifest.csv --modality $m --epochs 1 --out-dir runs/$m
done

# 4. Frozen-model predictions, threshold and Table 2
for m in fusion ct wsi; do
  python scripts/predict.py --manifest manifest.csv --modality $m --checkpoint runs/$m/$m.pth --out predictions_$m.csv
done
python scripts/derive_threshold.py --predictions predictions_fusion.csv
python scripts/evaluate.py --fusion predictions_fusion.csv --ct predictions_ct.csv --wsi predictions_wsi.csv --out table2.csv --check
```

`--check` compares the results with the values reported in the manuscript (`reference/table2_reported.json`).
Feature extraction writes `<patient_id>.npy` and `<patient_id>_coords.csv` per slide and skips slides that already
have an output, so an interrupted run resumes where it stopped.

## Data

Clinical, CT and WSI data are not included and are available from the corresponding authors on reasonable
request. Trained model weights are not publicly released. No patient identifiers are stored in this repository.
